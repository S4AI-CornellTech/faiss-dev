/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/impl/RemapIndices.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/invlists/InvertedLists.h>
#include <thrust/host_vector.h>
#include <faiss/gpu/impl/FlatIndex.cuh>
#include <faiss/gpu/impl/IVFAppend.cuh>
#include <faiss/gpu/impl/IVFBase.cuh>
#include <faiss/gpu/utils/CopyUtils.cuh>
#include <faiss/gpu/utils/DeviceDefs.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/ThrustUtils.cuh>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <string>
#include <unordered_map>
#include <unistd.h>

namespace faiss {
namespace gpu {

namespace {

constexpr uint64_t kPackedCacheMagic = 0x4653504b5f4d4554ULL; // "FSPK_MET"
constexpr uint64_t kPackedCacheVersion = 1;

struct PackedCacheHeader {
    uint64_t magic;
    uint64_t version;
    uint64_t nlist;
    uint64_t indicesOptions;
    uint64_t totalGpuBytes;
    uint64_t totalIndexBytes;
};

struct MMapFile {
    void* data = nullptr;
    size_t size = 0;
    int fd = -1;
};

MMapFile mapFileReadOnly(const std::string& path) {
    MMapFile mapped;
    int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        return mapped;
    }

    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        return mapped;
    }

    void* data = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (data == MAP_FAILED) {
        close(fd);
        return mapped;
    }

    mapped.data = data;
    mapped.size = static_cast<size_t>(st.st_size);
    mapped.fd = fd;
    return mapped;
}

void unmapFile(MMapFile& mapped) {
    if (mapped.data && mapped.data != MAP_FAILED) {
        munmap(mapped.data, mapped.size);
    }
    if (mapped.fd >= 0) {
        close(mapped.fd);
    }
    mapped.data = nullptr;
    mapped.size = 0;
    mapped.fd = -1;
}

bool readPackedMeta(
        const std::string& path,
        idx_t expectedNlist,
        IndicesOptions indicesOptions,
        std::vector<idx_t>& listSizes,
        std::vector<size_t>& codeOffsets,
        std::vector<size_t>& indexOffsets,
        size_t& totalGpuBytes,
        size_t& totalIndexBytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        return false;
    }

    PackedCacheHeader header{};
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in.good()) {
        return false;
    }

    if (header.magic != kPackedCacheMagic ||
        header.version != kPackedCacheVersion ||
        header.nlist != static_cast<uint64_t>(expectedNlist) ||
        header.indicesOptions != static_cast<uint64_t>(indicesOptions)) {
        return false;
    }

    listSizes.resize(header.nlist);
    codeOffsets.resize(header.nlist);
    indexOffsets.resize(header.nlist);

    for (uint64_t i = 0; i < header.nlist; ++i) {
        uint64_t value = 0;
        in.read(reinterpret_cast<char*>(&value), sizeof(value));
        listSizes[i] = static_cast<idx_t>(value);
    }
    for (uint64_t i = 0; i < header.nlist; ++i) {
        uint64_t value = 0;
        in.read(reinterpret_cast<char*>(&value), sizeof(value));
        codeOffsets[i] = static_cast<size_t>(value);
    }
    for (uint64_t i = 0; i < header.nlist; ++i) {
        uint64_t value = 0;
        in.read(reinterpret_cast<char*>(&value), sizeof(value));
        indexOffsets[i] = static_cast<size_t>(value);
    }

    if (!in.good()) {
        return false;
    }

    totalGpuBytes = static_cast<size_t>(header.totalGpuBytes);
    totalIndexBytes = static_cast<size_t>(header.totalIndexBytes);
    return true;
}

void writePackedMeta(
        const std::string& path,
        idx_t nlist,
        IndicesOptions indicesOptions,
        const std::vector<idx_t>& listSizes,
        const std::vector<size_t>& codeOffsets,
        const std::vector<size_t>& indexOffsets,
        size_t totalGpuBytes,
        size_t totalIndexBytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.good()) {
        return;
    }

    PackedCacheHeader header{};
    header.magic = kPackedCacheMagic;
    header.version = kPackedCacheVersion;
    header.nlist = static_cast<uint64_t>(nlist);
    header.indicesOptions = static_cast<uint64_t>(indicesOptions);
    header.totalGpuBytes = static_cast<uint64_t>(totalGpuBytes);
    header.totalIndexBytes = static_cast<uint64_t>(totalIndexBytes);

    out.write(reinterpret_cast<const char*>(&header), sizeof(header));

    for (idx_t i = 0; i < nlist; ++i) {
        uint64_t value = static_cast<uint64_t>(listSizes[i]);
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    for (idx_t i = 0; i < nlist; ++i) {
        uint64_t value = static_cast<uint64_t>(codeOffsets[i]);
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    for (idx_t i = 0; i < nlist; ++i) {
        uint64_t value = static_cast<uint64_t>(indexOffsets[i]);
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
}

} // namespace

IVFBase::DeviceIVFList::DeviceIVFList(GpuResources* res, const AllocInfo& info)
        : data(res, info), numVecs(0) {}

IVFBase::IVFBase(
        GpuResources* resources,
        int dim,
        idx_t nlist,
        faiss::MetricType metric,
        float metricArg,
        bool useResidual,
        bool interleavedLayout,
        IndicesOptions indicesOptions,
        MemorySpace space)
        : resources_(resources),
          metric_(metric),
          metricArg_(metricArg),
          dim_(dim),
          numLists_(nlist),
          useResidual_(useResidual),
          interleavedLayout_(interleavedLayout),
          indicesOptions_(indicesOptions),
          space_(space),
          deviceListDataPointers_(
                  resources,
                  AllocInfo(
                          AllocType::IVFLists,
                          getCurrentDevice(),
                          space,
                          resources->getDefaultStreamCurrentDevice())),
          deviceListIndexPointers_(
                  resources,
                  AllocInfo(
                          AllocType::IVFLists,
                          getCurrentDevice(),
                          space,
                          resources->getDefaultStreamCurrentDevice())),
          deviceListLengths_(
                  resources,
                  AllocInfo(
                          AllocType::IVFLists,
                          getCurrentDevice(),
                          space,
                          resources->getDefaultStreamCurrentDevice())),
          maxListLength_(0),
          packedLists_(false),
          packedListData_(
              resources,
              AllocInfo(
                  AllocType::IVFLists,
                  getCurrentDevice(),
                  space,
                  resources->getDefaultStreamCurrentDevice())),
          packedListIndices_(
              resources,
              AllocInfo(
                  AllocType::IVFLists,
                  getCurrentDevice(),
                  space,
                  resources->getDefaultStreamCurrentDevice())) {
    reset();
}

IVFBase::~IVFBase() {}

void IVFBase::reserveMemory(idx_t numVecs) {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    auto vecsPerList = numVecs / deviceListData_.size();
    if (vecsPerList < 1) {
        return;
    }

    auto bytesPerDataList = getGpuVectorsEncodingSize_(vecsPerList);

    for (auto& list : deviceListData_) {
        list->data.reserve(bytesPerDataList, stream);
    }

    if ((indicesOptions_ == INDICES_32_BIT) ||
        (indicesOptions_ == INDICES_64_BIT)) {
        // Reserve for index lists as well
        size_t bytesPerIndexList = vecsPerList *
                (indicesOptions_ == INDICES_32_BIT ? sizeof(int)
                                                   : sizeof(idx_t));

        for (auto& list : deviceListIndices_) {
            list->data.reserve(bytesPerIndexList, stream);
        }
    }

    // Update device info for all lists, since the base pointers may
    // have changed
    updateDeviceListInfo_(stream);
}

void IVFBase::reset() {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    deviceListData_.clear();
    deviceListIndices_.clear();
    deviceListDataPointers_.clear();
    deviceListIndexPointers_.clear();
    deviceListLengths_.clear();
    listOffsetToUserIndex_.clear();
    packedLists_ = false;
    packedListData_.clear();
    packedListIndices_.clear();
    packedListCodeOffsets_.clear();
    packedListIndexOffsets_.clear();

    auto info =
            AllocInfo(AllocType::IVFLists, getCurrentDevice(), space_, stream);

    for (idx_t i = 0; i < numLists_; ++i) {
        deviceListData_.emplace_back(
                std::unique_ptr<DeviceIVFList>(
                        new DeviceIVFList(resources_, info)));

        deviceListIndices_.emplace_back(
                std::unique_ptr<DeviceIVFList>(
                        new DeviceIVFList(resources_, info)));

        listOffsetToUserIndex_.emplace_back(std::vector<idx_t>());
    }

    deviceListDataPointers_.resize(numLists_, stream);
    deviceListDataPointers_.setAll(nullptr, stream);

    deviceListIndexPointers_.resize(numLists_, stream);
    deviceListIndexPointers_.setAll(nullptr, stream);

    deviceListLengths_.resize(numLists_, stream);
    deviceListLengths_.setAll(0, stream);

    maxListLength_ = 0;
}

idx_t IVFBase::getDim() const {
    return dim_;
}

size_t IVFBase::reclaimMemory() {
    // Reclaim all unused memory exactly
    return reclaimMemory_(true);
}

size_t IVFBase::reclaimMemory_(bool exact) {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    size_t totalReclaimed = 0;

    for (idx_t i = 0; i < deviceListData_.size(); ++i) {
        auto& data = deviceListData_[i]->data;
        totalReclaimed += data.reclaim(exact, stream);

        deviceListDataPointers_.setAt(i, (void*)data.data(), stream);
    }

    for (idx_t i = 0; i < deviceListIndices_.size(); ++i) {
        auto& indices = deviceListIndices_[i]->data;
        totalReclaimed += indices.reclaim(exact, stream);

        deviceListIndexPointers_.setAt(i, (void*)indices.data(), stream);
    }

    // Update device info for all lists, since the base pointers may
    // have changed
    updateDeviceListInfo_(stream);

    return totalReclaimed;
}

void IVFBase::updateDeviceListInfo_(cudaStream_t stream) {
    std::vector<idx_t> listIds(deviceListData_.size());
    for (idx_t i = 0; i < deviceListData_.size(); ++i) {
        listIds[i] = i;
    }

    updateDeviceListInfo_(listIds, stream);
}

void IVFBase::updateDeviceListInfo_(
        const std::vector<idx_t>& listIds,
        cudaStream_t stream) {
    idx_t listSize = listIds.size();
    HostTensor<idx_t, 1, true> hostListsToUpdate({listSize});
    HostTensor<idx_t, 1, true> hostNewListLength({listSize});
    HostTensor<void*, 1, true> hostNewDataPointers({listSize});
    HostTensor<void*, 1, true> hostNewIndexPointers({listSize});

    for (idx_t i = 0; i < listSize; ++i) {
        auto listId = listIds[i];
        auto& data = deviceListData_[listId];
        auto& indices = deviceListIndices_[listId];

        hostListsToUpdate[i] = listId;
        hostNewListLength[i] = data->numVecs;
        hostNewDataPointers[i] = data->data.data();
        hostNewIndexPointers[i] = indices->data.data();
    }

    // Copy the above update sets to the GPU
    DeviceTensor<idx_t, 1, true> listsToUpdate(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            hostListsToUpdate);
    DeviceTensor<idx_t, 1, true> newListLength(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            hostNewListLength);
    DeviceTensor<void*, 1, true> newDataPointers(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            hostNewDataPointers);
    DeviceTensor<void*, 1, true> newIndexPointers(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            hostNewIndexPointers);

    // Update all pointers to the lists on the device that may have
    // changed
    runUpdateListPointers(
            listsToUpdate,
            newListLength,
            newDataPointers,
            newIndexPointers,
            deviceListLengths_,
            deviceListDataPointers_,
            deviceListIndexPointers_,
            stream);
}

idx_t IVFBase::getNumLists() const {
    return numLists_;
}

idx_t IVFBase::getListLength(idx_t listId) const {
    FAISS_THROW_IF_NOT_FMT(
            listId < numLists_,
            "IVF list %ld is out of bounds (%ld lists total)",
            listId,
            numLists_);
    FAISS_ASSERT(listId < deviceListLengths_.size());
    FAISS_ASSERT(listId < deviceListData_.size());

    return deviceListData_[listId]->numVecs;
}

std::vector<idx_t> IVFBase::getListIndices(idx_t listId) const {
    FAISS_THROW_IF_NOT_FMT(
            listId < numLists_,
            "IVF list %ld is out of bounds (%ld lists total)",
            listId,
            numLists_);
    FAISS_ASSERT(listId < deviceListData_.size());
    FAISS_ASSERT(listId < deviceListLengths_.size());

    auto stream = resources_->getDefaultStreamCurrentDevice();

    if (packedLists_ &&
        (indicesOptions_ == INDICES_32_BIT ||
         indicesOptions_ == INDICES_64_BIT)) {
        size_t bytes = (indicesOptions_ == INDICES_32_BIT)
                ? deviceListData_[listId]->numVecs * sizeof(int)
                : deviceListData_[listId]->numVecs * sizeof(idx_t);

        std::vector<uint8_t> host(bytes);
        CUDA_VERIFY(cudaMemcpyAsync(
                host.data(),
                packedListIndices_.data() + packedListIndexOffsets_[listId],
                bytes,
                cudaMemcpyDeviceToHost,
                stream));

        if (indicesOptions_ == INDICES_32_BIT) {
            auto intInd = reinterpret_cast<const int*>(host.data());
            std::vector<idx_t> out(deviceListData_[listId]->numVecs);
            for (size_t i = 0; i < out.size(); ++i) {
                out[i] = (idx_t)intInd[i];
            }
            return out;
        }

        auto idxInd = reinterpret_cast<const idx_t*>(host.data());
        return std::vector<idx_t>(
                idxInd, idxInd + deviceListData_[listId]->numVecs);
    }

    if (indicesOptions_ == INDICES_32_BIT) {
        // The data is stored as int32 on the GPU
        FAISS_ASSERT(listId < deviceListIndices_.size());

        auto intInd = deviceListIndices_[listId]->data.copyToHost<int>(stream);

        std::vector<idx_t> out(intInd.size());
        for (size_t i = 0; i < intInd.size(); ++i) {
            out[i] = (idx_t)intInd[i];
        }

        return out;
    } else if (indicesOptions_ == INDICES_64_BIT) {
        // The data is stored as int64 on the GPU
        FAISS_ASSERT(listId < deviceListIndices_.size());

        return deviceListIndices_[listId]->data.copyToHost<idx_t>(stream);
    } else if (indicesOptions_ == INDICES_CPU) {
        // The data is not stored on the GPU
        FAISS_ASSERT(listId < listOffsetToUserIndex_.size());

        auto& userIds = listOffsetToUserIndex_[listId];

        // We should have the same number of indices on the CPU as we do vectors
        // encoded on the GPU
        FAISS_ASSERT(userIds.size() == deviceListData_[listId]->numVecs);

        // this will return a copy
        return userIds;
    } else {
        // unhandled indices type (includes INDICES_IVF)
        FAISS_ASSERT(false);
        return std::vector<idx_t>();
    }
}

std::vector<uint8_t> IVFBase::getListVectorData(idx_t listId, bool gpuFormat)
        const {
    FAISS_THROW_IF_NOT_FMT(
            listId < numLists_,
            "IVF list %ld is out of bounds (%ld lists total)",
            listId,
            numLists_);
    FAISS_ASSERT(listId < deviceListData_.size());
    FAISS_ASSERT(listId < deviceListLengths_.size());

    auto stream = resources_->getDefaultStreamCurrentDevice();

    auto& list = deviceListData_[listId];
    std::vector<uint8_t> gpuCodes;

    if (packedLists_) {
        size_t bytes = getGpuVectorsEncodingSize_(list->numVecs);
        gpuCodes.resize(bytes);
        CUDA_VERIFY(cudaMemcpyAsync(
                gpuCodes.data(),
                packedListData_.data() + packedListCodeOffsets_[listId],
                bytes,
                cudaMemcpyDeviceToHost,
                stream));
    } else {
        gpuCodes = list->data.copyToHost<uint8_t>(stream);
    }

    if (gpuFormat) {
        return gpuCodes;
    } else {
        // The GPU layout may be different than the CPU layout (e.g., vectors
        // rather than dimensions interleaved), translate back if necessary
        return translateCodesFromGpu_(std::move(gpuCodes), list->numVecs);
    }
}

void IVFBase::copyInvertedListsFrom(const InvertedLists* ivf) {
    idx_t nlist = ivf ? ivf->nlist : 0;
    if (nlist == 0) {
        return;
    }

    const std::string codesPath = "/dev/shm/gpu_codes_all.bin";
    const std::string indicesPath = "/dev/shm/gpu_indices_all.bin";
    const std::string metaPath = "/dev/shm/gpu_codes_all.meta";

    const char* packedEnv = std::getenv("FAISS_GPU_PACKED_LISTS");
    const char* packedDebugEnv = std::getenv("FAISS_GPU_PACKED_LISTS_DEBUG");
    const char* packedMmapEnv = std::getenv("FAISS_GPU_PACKED_LISTS_MMAP");
    bool usePacked = packedEnv && std::string(packedEnv) == "1";
    bool useMmap = packedMmapEnv && std::string(packedMmapEnv) == "1";

    // Packed lists require GPU-resident indices
    if (indicesOptions_ == INDICES_CPU) {
        usePacked = false;
    }

    std::vector<idx_t> listSizes(nlist);
    std::vector<size_t> listOffsets(nlist);
    std::vector<size_t> listGpuBytes(nlist);
    std::vector<size_t> listIndexOffsets(nlist, 0);
    size_t totalGpuBytes = 0;
    size_t totalIndexBytes = 0;

    bool metaLoaded = false;
    if (usePacked) {
        metaLoaded = readPackedMeta(
                metaPath,
                nlist,
                indicesOptions_,
                listSizes,
                listOffsets,
                listIndexOffsets,
                totalGpuBytes,
                totalIndexBytes);
    }
    if (packedDebugEnv) {
        std::cerr << "[faiss] packed_lists usePacked=" << usePacked
                  << " metaLoaded=" << metaLoaded
                  << " indicesOptions=" << static_cast<int>(indicesOptions_)
                  << " nlist=" << nlist << "\n";
    }

    if (!metaLoaded) {
        size_t running = 0;
        size_t indexRunning = 0;
        size_t indexSize = (indicesOptions_ == INDICES_32_BIT)
                ? sizeof(int)
                : sizeof(idx_t);

        for (idx_t i = 0; i < nlist; ++i) {
            listSizes[i] = ivf->list_size(i);
            listOffsets[i] = running;
            listGpuBytes[i] = getGpuVectorsEncodingSize_(listSizes[i]);
            running += listGpuBytes[i];

            if ((indicesOptions_ == INDICES_32_BIT) ||
                (indicesOptions_ == INDICES_64_BIT)) {
                listIndexOffsets[i] = indexRunning;
                indexRunning += listSizes[i] * indexSize;
            }
        }

        totalGpuBytes = running;
        totalIndexBytes = indexRunning;
    } else {
        for (idx_t i = 0; i < nlist; ++i) {
            listGpuBytes[i] = getGpuVectorsEncodingSize_(listSizes[i]);
        }
    }

    bool codesCacheOk = false;
    std::ifstream codesIn(codesPath, std::ios::binary | std::ios::ate);
    if (codesIn.good()) {
        auto fileSize = static_cast<size_t>(codesIn.tellg());
        if (fileSize == totalGpuBytes) {
            codesCacheOk = true;
            codesIn.seekg(0, std::ios::beg);
        }
    }

    if (packedDebugEnv) {
        std::cerr << "[faiss] packed_lists codesCacheOk=" << codesCacheOk
                  << " totalGpuBytes=" << totalGpuBytes << "\n";
    }

    if (codesCacheOk && usePacked && metaLoaded) {
        auto pinnedAlloc = resources_->getPinnedMemory();
        auto* pinnedBuf = static_cast<uint8_t*>(pinnedAlloc.first);
        size_t pinnedSize = pinnedAlloc.second;
        auto copyStream = resources_->getAsyncCopyStreamCurrentDevice();
        auto defaultStream = resources_->getDefaultStreamCurrentDevice();

        packedListData_.resize(totalGpuBytes, copyStream);

        auto copyFileToDevice = [&](
                std::ifstream& in,
                uint8_t* dst,
                size_t totalBytes) {
            size_t offset = 0;
            if (pinnedBuf && pinnedSize > 0) {
                while (offset < totalBytes) {
                    size_t chunk = std::min(pinnedSize, totalBytes - offset);
                    in.read(reinterpret_cast<char*>(pinnedBuf), chunk);
                    FAISS_THROW_IF_NOT_FMT(
                            in.good(),
                            "failed reading %zu bytes from cache",
                            chunk);
                    CUDA_VERIFY(cudaMemcpyAsync(
                            dst + offset,
                            pinnedBuf,
                            chunk,
                            cudaMemcpyHostToDevice,
                            copyStream));
                    offset += chunk;
                }
            } else {
                std::vector<uint8_t> temp(
                        std::min<size_t>(256 * 1024 * 1024, totalBytes));
                while (offset < totalBytes) {
                    size_t chunk = std::min(temp.size(), totalBytes - offset);
                    in.read(reinterpret_cast<char*>(temp.data()), chunk);
                    FAISS_THROW_IF_NOT_FMT(
                            in.good(),
                            "failed reading %zu bytes from cache",
                            chunk);
                    CUDA_VERIFY(cudaMemcpyAsync(
                            dst + offset,
                            temp.data(),
                            chunk,
                            cudaMemcpyHostToDevice,
                            copyStream));
                    offset += chunk;
                }
            }
        };

        double codesReadSec = 0.0;
        double codesCopySec = 0.0;
        double indicesReadSec = 0.0;
        double indicesCopySec = 0.0;

        auto copyFileToDeviceTimed = [&](
            std::ifstream& in,
            uint8_t* dst,
            size_t totalBytes,
            double& readSec,
            double& copySec) {
            size_t offset = 0;
            if (pinnedBuf && pinnedSize > 0) {
            while (offset < totalBytes) {
                size_t chunk = std::min(pinnedSize, totalBytes - offset);
                auto t0 = std::chrono::high_resolution_clock::now();
                in.read(reinterpret_cast<char*>(pinnedBuf), chunk);
                auto t1 = std::chrono::high_resolution_clock::now();
                FAISS_THROW_IF_NOT_FMT(
                    in.good(),
                    "failed reading %zu bytes from cache",
                    chunk);

                readSec += std::chrono::duration<double>(t1 - t0).count();

                auto t2 = std::chrono::high_resolution_clock::now();
                CUDA_VERIFY(cudaMemcpyAsync(
                    dst + offset,
                    pinnedBuf,
                    chunk,
                    cudaMemcpyHostToDevice,
                    copyStream));
                CUDA_VERIFY(cudaStreamSynchronize(copyStream));
                auto t3 = std::chrono::high_resolution_clock::now();

                copySec += std::chrono::duration<double>(t3 - t2).count();
                offset += chunk;
            }
            } else {
            std::vector<uint8_t> temp(
                std::min<size_t>(256 * 1024 * 1024, totalBytes));
            while (offset < totalBytes) {
                size_t chunk = std::min(temp.size(), totalBytes - offset);
                auto t0 = std::chrono::high_resolution_clock::now();
                in.read(reinterpret_cast<char*>(temp.data()), chunk);
                auto t1 = std::chrono::high_resolution_clock::now();
                FAISS_THROW_IF_NOT_FMT(
                    in.good(),
                    "failed reading %zu bytes from cache",
                    chunk);

                readSec += std::chrono::duration<double>(t1 - t0).count();

                auto t2 = std::chrono::high_resolution_clock::now();
                CUDA_VERIFY(cudaMemcpyAsync(
                    dst + offset,
                    temp.data(),
                    chunk,
                    cudaMemcpyHostToDevice,
                    copyStream));
                CUDA_VERIFY(cudaStreamSynchronize(copyStream));
                auto t3 = std::chrono::high_resolution_clock::now();

                copySec += std::chrono::duration<double>(t3 - t2).count();
                offset += chunk;
            }
            }
        };

        if (useMmap) {
            auto mapped = mapFileReadOnly(codesPath);
            if (mapped.data && mapped.size == totalGpuBytes) {
                madvise(mapped.data, mapped.size, MADV_WILLNEED);
                size_t offset = 0;
                size_t chunkSize = pinnedSize > 0 ? pinnedSize : (256 * 1024 * 1024);
                while (offset < totalGpuBytes) {
                    size_t chunk = std::min(chunkSize, totalGpuBytes - offset);
                    auto t2 = std::chrono::high_resolution_clock::now();
                    CUDA_VERIFY(cudaMemcpyAsync(
                            packedListData_.data() + offset,
                            static_cast<uint8_t*>(mapped.data) + offset,
                            chunk,
                            cudaMemcpyHostToDevice,
                            copyStream));
                    CUDA_VERIFY(cudaStreamSynchronize(copyStream));
                    auto t3 = std::chrono::high_resolution_clock::now();
                    codesCopySec += std::chrono::duration<double>(t3 - t2).count();
                    offset += chunk;
                }
                unmapFile(mapped);
            } else {
                if (mapped.data) {
                    unmapFile(mapped);
                }
                copyFileToDeviceTimed(
                        codesIn,
                        packedListData_.data(),
                        totalGpuBytes,
                        codesReadSec,
                        codesCopySec);
            }
        } else {
            copyFileToDeviceTimed(
                    codesIn,
                    packedListData_.data(),
                    totalGpuBytes,
                    codesReadSec,
                    codesCopySec);
        }

        bool indicesCacheOk = false;
        std::ifstream indicesIn;
        std::ofstream indicesOut;
        if ((indicesOptions_ == INDICES_32_BIT ||
             indicesOptions_ == INDICES_64_BIT) &&
            totalIndexBytes > 0) {
            indicesIn.open(indicesPath, std::ios::binary | std::ios::ate);
            if (indicesIn.good()) {
                auto size = static_cast<size_t>(indicesIn.tellg());
                if (size == totalIndexBytes) {
                    indicesCacheOk = true;
                    indicesIn.seekg(0, std::ios::beg);
                }
            }
        }

        if (packedDebugEnv) {
            std::cerr << "[faiss] packed_lists indicesCacheOk=" << indicesCacheOk
                      << " totalIndexBytes=" << totalIndexBytes << "\n";
        }

        if (indicesCacheOk) {
                packedListIndices_.resize(totalIndexBytes, copyStream);
            if (useMmap) {
                auto mapped = mapFileReadOnly(indicesPath);
                if (mapped.data && mapped.size == totalIndexBytes) {
                    madvise(mapped.data, mapped.size, MADV_WILLNEED);
                    size_t offset = 0;
                    size_t chunkSize = pinnedSize > 0 ? pinnedSize : (256 * 1024 * 1024);
                    while (offset < totalIndexBytes) {
                        size_t chunk = std::min(chunkSize, totalIndexBytes - offset);
                        auto t2 = std::chrono::high_resolution_clock::now();
                        CUDA_VERIFY(cudaMemcpyAsync(
                                packedListIndices_.data() + offset,
                                static_cast<uint8_t*>(mapped.data) + offset,
                                chunk,
                                cudaMemcpyHostToDevice,
                                copyStream));
                        CUDA_VERIFY(cudaStreamSynchronize(copyStream));
                        auto t3 = std::chrono::high_resolution_clock::now();
                        indicesCopySec +=
                                std::chrono::duration<double>(t3 - t2).count();
                        offset += chunk;
                    }
                    unmapFile(mapped);
                } else {
                    if (mapped.data) {
                        unmapFile(mapped);
                    }
                    copyFileToDeviceTimed(
                            indicesIn,
                            packedListIndices_.data(),
                            totalIndexBytes,
                            indicesReadSec,
                            indicesCopySec);
                }
            } else {
                copyFileToDeviceTimed(
                        indicesIn,
                        packedListIndices_.data(),
                        totalIndexBytes,
                        indicesReadSec,
                        indicesCopySec);
            }
        } else if (indicesOptions_ == INDICES_32_BIT ||
                   indicesOptions_ == INDICES_64_BIT) {
            if (totalIndexBytes > 0) {
                indicesOut.open(
                        indicesPath, std::ios::binary | std::ios::trunc);
            }
            packedListIndices_.resize(totalIndexBytes, copyStream);

            for (idx_t i = 0; i < nlist; ++i) {
                if (listSizes[i] == 0) {
                    continue;
                }

                if (indicesOptions_ == INDICES_32_BIT) {
                    std::vector<int> indices32(listSizes[i]);
                    auto ids = ivf->get_ids(i);
                    for (idx_t j = 0; j < listSizes[i]; ++j) {
                        auto ind = ids[j];
                        FAISS_ASSERT(
                                ind <=
                                (idx_t)std::numeric_limits<int>::max());
                        indices32[j] = (int)ind;
                    }

                        CUDA_VERIFY(cudaMemcpyAsync(
                            packedListIndices_.data() +
                                    listIndexOffsets[i],
                            indices32.data(),
                            listSizes[i] * sizeof(int),
                            cudaMemcpyHostToDevice,
                            copyStream));
                        if (indicesOut.good()) {
                        indicesOut.write(
                            reinterpret_cast<const char*>(
                                indices32.data()),
                            listSizes[i] * sizeof(int));
                        }
                } else {
                    CUDA_VERIFY(cudaMemcpyAsync(
                            packedListIndices_.data() +
                                    listIndexOffsets[i],
                            ivf->get_ids(i),
                            listSizes[i] * sizeof(idx_t),
                            cudaMemcpyHostToDevice,
                            copyStream));
                        if (indicesOut.good()) {
                        indicesOut.write(
                            reinterpret_cast<const char*>(
                                ivf->get_ids(i)),
                            listSizes[i] * sizeof(idx_t));
                        }
                }
            }
        }

        for (idx_t i = 0; i < nlist; ++i) {
            if (listSizes[i] == 0) {
                continue;
            }

            deviceListDataPointers_.setAt(
                    i,
                    (void*)(packedListData_.data() + listOffsets[i]),
                    copyStream);
            deviceListLengths_.setAt(i, listSizes[i], copyStream);
            deviceListData_[i]->numVecs = listSizes[i];
            maxListLength_ = std::max(maxListLength_, listSizes[i]);

            if ((indicesOptions_ == INDICES_32_BIT ||
                 indicesOptions_ == INDICES_64_BIT) &&
                totalIndexBytes > 0) {
                deviceListIndexPointers_.setAt(
                        i,
                        (void*)(packedListIndices_.data() +
                                listIndexOffsets[i]),
                        copyStream);
                deviceListIndices_[i]->numVecs = listSizes[i];
            }
        }

        packedLists_ = true;
        packedListCodeOffsets_ = listOffsets;
        packedListIndexOffsets_ = listIndexOffsets;

        if (packedDebugEnv) {
            auto toGb = [](size_t bytes) {
                return static_cast<double>(bytes) /
                        (1024.0 * 1024.0 * 1024.0);
            };

            if (totalGpuBytes > 0) {
                std::cerr << "[faiss] packed_lists codes read GB/s="
                          << (toGb(totalGpuBytes) /
                              std::max(1e-9, codesReadSec))
                          << " copy GB/s="
                          << (toGb(totalGpuBytes) /
                              std::max(1e-9, codesCopySec))
                          << "\n";
            }
            if (totalIndexBytes > 0) {
                std::cerr << "[faiss] packed_lists indices read GB/s="
                          << (toGb(totalIndexBytes) /
                              std::max(1e-9, indicesReadSec))
                          << " copy GB/s="
                          << (toGb(totalIndexBytes) /
                              std::max(1e-9, indicesCopySec))
                          << "\n";
            }
            std::cerr << "[faiss] packed_lists enabled; bulk upload complete\n";
        }
        streamWait({defaultStream}, {copyStream});
        return;
    }

    std::ofstream codesOut;
    std::ofstream indicesOut;

    if (totalGpuBytes > 0) {
        codesOut.open(codesPath, std::ios::binary | std::ios::trunc);
    }
    if ((indicesOptions_ == INDICES_32_BIT ||
         indicesOptions_ == INDICES_64_BIT) &&
        totalIndexBytes > 0) {
        indicesOut.open(indicesPath, std::ios::binary | std::ios::trunc);
    }

    if (!metaLoaded) {
        writePackedMeta(
                metaPath,
                nlist,
                indicesOptions_,
                listSizes,
                listOffsets,
                listIndexOffsets,
                totalGpuBytes,
                totalIndexBytes);
    }

    for (idx_t i = 0; i < nlist; ++i) {
        auto numVecs = listSizes[i];
        if (numVecs == 0) {
            continue;
        }

        auto gpuListSizeInBytes = getGpuVectorsEncodingSize_(numVecs);
        auto cpuListSizeInBytes = getCpuVectorsEncodingSize_(numVecs);

        std::vector<uint8_t> codesV(cpuListSizeInBytes);
        std::memcpy(codesV.data(), ivf->get_codes(i), cpuListSizeInBytes);
        auto translatedCodes = translateCodesToGpu_(std::move(codesV), numVecs);

        addEncodedGpuVectorsToList_(
                i,
                translatedCodes.data(),
                gpuListSizeInBytes,
                ivf->get_ids(i),
                numVecs);

        if (codesOut.good()) {
            codesOut.write(
                    reinterpret_cast<const char*>(translatedCodes.data()),
                    gpuListSizeInBytes);
        }

        if (indicesOut.good()) {
            if (indicesOptions_ == INDICES_32_BIT) {
                std::vector<int> indices32(numVecs);
                auto ids = ivf->get_ids(i);
                for (idx_t j = 0; j < numVecs; ++j) {
                    auto ind = ids[j];
                    FAISS_ASSERT(
                            ind <= (idx_t)std::numeric_limits<int>::max());
                    indices32[j] = (int)ind;
                }
                indicesOut.write(
                        reinterpret_cast<const char*>(indices32.data()),
                        numVecs * sizeof(int));
            } else {
                indicesOut.write(
                        reinterpret_cast<const char*>(ivf->get_ids(i)),
                        numVecs * sizeof(idx_t));
            }
        }
    }
}

void IVFBase::copyInvertedListsTo(InvertedLists* ivf) {
    for (idx_t i = 0; i < numLists_; ++i) {
        auto listIndices = getListIndices(i);
        auto listData = getListVectorData(i, false);

        ivf->add_entries(
                i, listIndices.size(), listIndices.data(), listData.data());
    }
}

void IVFBase::reconstruct_n(idx_t i0, idx_t n, float* out) {
    FAISS_THROW_MSG("not implemented");
}

void IVFBase::addEncodedVectorsToList_(
        idx_t listId,
        const void* codes,
        const idx_t* indices,
        idx_t numVecs) {
        FAISS_THROW_IF_NOT_MSG(
            !packedLists_,
            "cannot append to packed IVF lists");
    auto stream = resources_->getDefaultStreamCurrentDevice();

    // This list must already exist
    FAISS_ASSERT(listId < deviceListData_.size());

    // This list must currently be empty
    auto& listCodes = deviceListData_[listId];
    FAISS_ASSERT(listCodes->data.size() == 0);
    FAISS_ASSERT(listCodes->numVecs == 0);

    // If there's nothing to add, then there's nothing we have to do
    if (numVecs == 0) {
        return;
    }

    // The GPU might have a different layout of the memory
    auto gpuListSizeInBytes = getGpuVectorsEncodingSize_(numVecs);
    auto cpuListSizeInBytes = getCpuVectorsEncodingSize_(numVecs);

    // Translate the codes as needed to our preferred form
    std::vector<uint8_t> codesV(cpuListSizeInBytes);
    std::memcpy(codesV.data(), codes, cpuListSizeInBytes);
    auto translatedCodes = translateCodesToGpu_(std::move(codesV), numVecs);

    addEncodedGpuVectorsToList_(
            listId,
            translatedCodes.data(),
            gpuListSizeInBytes,
            indices,
            numVecs);
}

void IVFBase::addEncodedGpuVectorsToList_(
        idx_t listId,
        const uint8_t* gpuCodes,
        size_t gpuListSizeInBytes,
        const idx_t* indices,
        idx_t numVecs) {
    FAISS_THROW_IF_NOT_MSG(
            !packedLists_,
            "cannot append to packed IVF lists");

    auto stream = resources_->getDefaultStreamCurrentDevice();

        // This list must already exist
        FAISS_ASSERT(listId < deviceListData_.size());

        // This list must currently be empty
        auto& listCodes = deviceListData_[listId];
        FAISS_ASSERT(listCodes->data.size() == 0);
        FAISS_ASSERT(listCodes->numVecs == 0);

        // If there's nothing to add, then there's nothing we have to do
    if (numVecs == 0) {
        return;
    }

    listCodes->data.append(
            gpuCodes,
            gpuListSizeInBytes,
            stream,
            true /* exact reserved size */);
    listCodes->numVecs = numVecs;

        // Handle the indices as well
    addIndicesFromCpu_(listId, indices, numVecs);

    deviceListDataPointers_.setAt(
            listId, (void*)listCodes->data.data(), stream);
    deviceListLengths_.setAt(listId, numVecs, stream);

        // We update this as well, since the multi-pass algorithm uses it
    maxListLength_ = std::max(maxListLength_, numVecs);
}

void IVFBase::addIndicesFromCpu_(
        idx_t listId,
        const idx_t* indices,
        idx_t numVecs) {
        FAISS_THROW_IF_NOT_MSG(
            !packedLists_,
            "cannot append indices to packed IVF lists");
    auto stream = resources_->getDefaultStreamCurrentDevice();

    // This list must currently be empty
    auto& listIndices = deviceListIndices_[listId];
    FAISS_ASSERT(listIndices->data.size() == 0);
    FAISS_ASSERT(listIndices->numVecs == 0);

    if (indicesOptions_ == INDICES_32_BIT) {
        // Make sure that all indices are in bounds
        std::vector<int> indices32(numVecs);
        for (idx_t i = 0; i < numVecs; ++i) {
            auto ind = indices[i];
            FAISS_ASSERT(ind <= (idx_t)std::numeric_limits<int>::max());
            indices32[i] = (int)ind;
        }

        static_assert(sizeof(int) == 4, "");

        listIndices->data.append(
                (uint8_t*)indices32.data(),
                numVecs * sizeof(int),
                stream,
                true /* exact reserved size */);

        // We have added the given indices to the raw data vector; update the
        // count as well
        listIndices->numVecs = numVecs;
    } else if (indicesOptions_ == INDICES_64_BIT) {
        listIndices->data.append(
                (uint8_t*)indices,
                numVecs * sizeof(idx_t),
                stream,
                true /* exact reserved size */);

        // We have added the given indices to the raw data vector; update the
        // count as well
        listIndices->numVecs = numVecs;
    } else if (indicesOptions_ == INDICES_CPU) {
        // indices are stored on the CPU
        FAISS_ASSERT(listId < listOffsetToUserIndex_.size());

        auto& userIndices = listOffsetToUserIndex_[listId];
        userIndices.insert(userIndices.begin(), indices, indices + numVecs);
    } else {
        // indices are not stored
        FAISS_ASSERT(indicesOptions_ == INDICES_IVF);
    }

    deviceListIndexPointers_.setAt(
            listId, (void*)listIndices->data.data(), stream);
}

void IVFBase::updateQuantizer(Index* quantizer) {
    FAISS_THROW_IF_NOT(quantizer->is_trained);

    // Must match our basic IVF parameters
    FAISS_THROW_IF_NOT(quantizer->d == getDim());
    FAISS_THROW_IF_NOT(quantizer->ntotal == getNumLists());

    auto stream = resources_->getDefaultStreamCurrentDevice();

    // If the index instance is a GpuIndexFlat, then we can use direct access to
    // the centroids within.
    auto gpuQ = dynamic_cast<GpuIndexFlat*>(quantizer);
    if (gpuQ) {
        auto gpuData = gpuQ->getGpuData();

        if (gpuData->getUseFloat16()) {
            // The FlatIndex keeps its data in float16; we need to reconstruct
            // as float32 and store locally
            DeviceTensor<float, 2, true> centroids(
                    resources_,
                    makeSpaceAlloc(AllocType::FlatData, space_, stream),
                    {getNumLists(), getDim()});

            gpuData->reconstruct(0, gpuData->getSize(), centroids);

            ivfCentroids_ = std::move(centroids);
        } else {
            // The FlatIndex keeps its data in float32, so we can merely
            // reference it
            auto ref32 = gpuData->getVectorsFloat32Ref();

            // Create a DeviceTensor that merely references, doesn't own the
            // data
            auto refOnly = DeviceTensor<float, 2, true>(
                    ref32.data(), {ref32.getSize(0), ref32.getSize(1)});

            ivfCentroids_ = std::move(refOnly);
        }
    } else {
        // Otherwise, we need to reconstruct all vectors from the index and copy
        // them to the GPU, in order to have access as needed for residual
        // computation
        auto vecs = std::vector<float>(getNumLists() * getDim());
        quantizer->reconstruct_n(0, quantizer->ntotal, vecs.data());

        // Copy to a new DeviceTensor; this will own the data
        DeviceTensor<float, 2, true> centroids(
                resources_,
                makeSpaceAlloc(AllocType::FlatData, space_, stream),
                {quantizer->ntotal, quantizer->d});
        centroids.copyFrom(vecs, stream);

        ivfCentroids_ = std::move(centroids);
    }
}

void IVFBase::searchCoarseQuantizer_(
        Index* coarseQuantizer,
        int nprobe,
        // Guaranteed to be on device
        Tensor<float, 2, true>& vecs,
        Tensor<float, 2, true>& distances,
        Tensor<idx_t, 2, true>& indices,
        Tensor<float, 3, true>* residuals,
        Tensor<float, 3, true>* centroids) {
    auto stream = resources_->getDefaultStreamCurrentDevice();

    // The provided IVF quantizer may be CPU or GPU resident.
    // If GPU resident, we can simply call it passing the above output device
    // pointers.
    auto gpuQuantizer = tryCastGpuIndex(coarseQuantizer);
    if (gpuQuantizer) {
        // We can pass device pointers directly
        gpuQuantizer->search(
                vecs.getSize(0),
                vecs.data(),
                nprobe,
                distances.data(),
                indices.data());

        if (residuals) {
            gpuQuantizer->compute_residual_n(
                    vecs.getSize(0) * nprobe,
                    vecs.data(),
                    residuals->data(),
                    indices.data());
        }

        if (centroids) {
            gpuQuantizer->reconstruct_batch(
                    vecs.getSize(0) * nprobe,
                    indices.data(),
                    centroids->data());
        }
    } else {
        // temporary host storage for querying a CPU index
        auto cpuVecs = toHost<float, 2>(
                vecs.data(), stream, {vecs.getSize(0), vecs.getSize(1)});
        auto cpuDistances = std::vector<float>(vecs.getSize(0) * nprobe);
        auto cpuIndices = std::vector<idx_t>(vecs.getSize(0) * nprobe);

        coarseQuantizer->search(
                vecs.getSize(0),
                cpuVecs.data(),
                nprobe,
                cpuDistances.data(),
                cpuIndices.data());

        distances.copyFrom(cpuDistances, stream);

        // Did we also want to return IVF cell residuals for the query vectors?
        if (residuals) {
            // we need space for the residuals as well
            auto cpuResiduals =
                    std::vector<float>(vecs.getSize(0) * nprobe * dim_);

            coarseQuantizer->compute_residual_n(
                    vecs.getSize(0) * nprobe,
                    cpuVecs.data(),
                    cpuResiduals.data(),
                    cpuIndices.data());

            residuals->copyFrom(cpuResiduals, stream);
        }

        // Did we also want to return the IVF cell centroids themselves?
        if (centroids) {
            auto cpuCentroids =
                    std::vector<float>(vecs.getSize(0) * nprobe * dim_);

            coarseQuantizer->reconstruct_batch(
                    vecs.getSize(0) * nprobe,
                    cpuIndices.data(),
                    cpuCentroids.data());

            centroids->copyFrom(cpuCentroids, stream);
        }

        indices.copyFrom(cpuIndices, stream);
    }
}

idx_t IVFBase::addVectors(
        Index* coarseQuantizer,
        Tensor<float, 2, true>& vecs,
        Tensor<idx_t, 1, true>& indices) {
        FAISS_THROW_IF_NOT_MSG(
            !packedLists_,
            "cannot add vectors to packed IVF lists");
    FAISS_ASSERT(vecs.getSize(0) == indices.getSize(0));
    FAISS_ASSERT(vecs.getSize(1) == dim_);

    auto stream = resources_->getDefaultStreamCurrentDevice();

    // Determine which IVF lists we need to append to
    // We report distances from the shared query function, but we don't need
    // them
    DeviceTensor<float, 2, true> unusedIVFDistances(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {vecs.getSize(0), 1});

    // We do need the closest IVF cell IDs though
    DeviceTensor<idx_t, 2, true> ivfIndices(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {vecs.getSize(0), 1});

    // Calculate residuals for these vectors, if needed
    DeviceTensor<float, 3, true> residuals(
            resources_,
            makeTempAlloc(AllocType::Other, stream),
            {vecs.getSize(0), 1, dim_});

    searchCoarseQuantizer_(
            coarseQuantizer,
            1, // nprobe
            vecs,
            unusedIVFDistances,
            ivfIndices,
            useResidual_ ? &residuals : nullptr,
            nullptr);

    // Copy the lists that we wish to append to back to the CPU
    // FIXME: really this can be into pinned memory and a true async
    // copy on a different stream; we can start the copy early, but it's
    // tiny
    auto ivfIndicesHost = ivfIndices.copyToVector(stream);

    // Now we add the encoded vectors to the individual lists
    // First, make sure that there is space available for adding the new
    // encoded vectors and indices

    // list id -> vectors being added
    std::unordered_map<idx_t, std::vector<idx_t>> listToVectorIds;

    // vector id -> which list it is being appended to
    std::vector<idx_t> vectorIdToList(vecs.getSize(0));

    // vector id -> offset in list
    // (we already have vector id -> list id in listIds)
    std::vector<idx_t> listOffsetHost(ivfIndicesHost.size());

    // Number of valid vectors that we actually add; we return this
    idx_t numAdded = 0;

    for (idx_t i = 0; i < ivfIndicesHost.size(); ++i) {
        auto listId = ivfIndicesHost[i];

        // Add vector could be invalid (contains NaNs etc)
        if (listId < 0) {
            listOffsetHost[i] = -1;
            vectorIdToList[i] = -1;
            continue;
        }

        FAISS_ASSERT(listId < numLists_);
        ++numAdded;
        vectorIdToList[i] = listId;

        auto offset = deviceListData_[listId]->numVecs;

        auto it = listToVectorIds.find(listId);
        if (it != listToVectorIds.end()) {
            offset += it->second.size();
            it->second.push_back(i);
        } else {
            listToVectorIds[listId] = std::vector<idx_t>{i};
        }

        listOffsetHost[i] = offset;
    }

    // If we didn't add anything (all invalid vectors that didn't map to IVF
    // clusters), no need to continue
    if (numAdded == 0) {
        return 0;
    }

    // unique lists being added to
    std::vector<idx_t> uniqueLists;

    for (auto& vecs : listToVectorIds) {
        uniqueLists.push_back(vecs.first);
    }

    std::sort(uniqueLists.begin(), uniqueLists.end());

    // In the same order as uniqueLists, list the vectors being added to that
    // list contiguously (unique list 0 vectors ...)(unique list 1 vectors ...)
    // ...
    std::vector<idx_t> vectorsByUniqueList;

    // For each of the unique lists, the start offset in vectorsByUniqueList
    std::vector<idx_t> uniqueListVectorStart;

    // For each of the unique lists, where we start appending in that list by
    // the vector offset
    std::vector<idx_t> uniqueListStartOffset;

    // For each of the unique lists, find the vectors which should be appended
    // to that list
    for (auto ul : uniqueLists) {
        uniqueListVectorStart.push_back(vectorsByUniqueList.size());

        FAISS_ASSERT(listToVectorIds.count(ul) != 0);

        // The vectors we are adding to this list
        auto& vecs = listToVectorIds[ul];
        vectorsByUniqueList.insert(
                vectorsByUniqueList.end(), vecs.begin(), vecs.end());

        // How many vectors we previously had (which is where we start appending
        // on the device)
        uniqueListStartOffset.push_back(deviceListData_[ul]->numVecs);
    }

    // We terminate uniqueListVectorStart with the overall number of vectors
    // being added, which could be different than vecs.getSize(0) as some
    // vectors could be invalid
    uniqueListVectorStart.push_back(vectorsByUniqueList.size());

    // We need to resize the data structures for the inverted lists on
    // the GPUs, which means that they might need reallocation, which
    // means that their base address may change. Figure out the new base
    // addresses, and update those in a batch on the device
    {
        // Resize all of the lists that we are appending to
        for (auto& counts : listToVectorIds) {
            auto listId = counts.first;
            idx_t numVecsToAdd = counts.second.size();

            auto& codes = deviceListData_[listId];
            auto oldNumVecs = codes->numVecs;
            auto newNumVecs = codes->numVecs + numVecsToAdd;

            auto newSizeBytes = getGpuVectorsEncodingSize_(newNumVecs);
            codes->data.resize(newSizeBytes, stream);
            codes->numVecs = newNumVecs;

            auto& indices = deviceListIndices_[listId];
            if ((indicesOptions_ == INDICES_32_BIT) ||
                (indicesOptions_ == INDICES_64_BIT)) {
                size_t indexSize = (indicesOptions_ == INDICES_32_BIT)
                        ? sizeof(int)
                        : sizeof(idx_t);

                indices->data.resize(
                        indices->data.size() + numVecsToAdd * indexSize,
                        stream);
                FAISS_ASSERT(indices->numVecs == oldNumVecs);
                indices->numVecs = newNumVecs;

            } else if (indicesOptions_ == INDICES_CPU) {
                // indices are stored on the CPU side
                FAISS_ASSERT(listId < listOffsetToUserIndex_.size());

                auto& userIndices = listOffsetToUserIndex_[listId];
                userIndices.resize(newNumVecs);
            } else {
                // indices are not stored on the GPU or CPU side
                FAISS_ASSERT(indicesOptions_ == INDICES_IVF);
            }

            // This is used by the multi-pass query to decide how much scratch
            // space to allocate for intermediate results
            maxListLength_ = std::max(maxListLength_, newNumVecs);
        }

        // Update all pointers and sizes on the device for lists that we
        // appended to
        updateDeviceListInfo_(uniqueLists, stream);
    }

    // If we're maintaining the indices on the CPU side, update our
    // map. We already resized our map above.
    if (indicesOptions_ == INDICES_CPU) {
        // We need to maintain the indices on the CPU side
        HostTensor<idx_t, 1, true> hostIndices(indices, stream);

        for (idx_t i = 0; i < hostIndices.getSize(0); ++i) {
            idx_t listId = ivfIndicesHost[i];

            // Add vector could be invalid (contains NaNs etc)
            if (listId < 0) {
                continue;
            }

            auto offset = listOffsetHost[i];
            FAISS_ASSERT(offset >= 0);

            FAISS_ASSERT(listId < listOffsetToUserIndex_.size());
            auto& userIndices = listOffsetToUserIndex_[listId];

            FAISS_ASSERT(offset < userIndices.size());
            userIndices[offset] = hostIndices[i];
        }
    }

    // Copy the offsets to the GPU
    auto ivfIndices1dDevice = ivfIndices.downcastOuter<1>();
    auto residuals2dDevice = residuals.downcastOuter<2>();
    auto listOffsetDevice =
            toDeviceTemporary(resources_, listOffsetHost, stream);
    auto uniqueListsDevice = toDeviceTemporary(resources_, uniqueLists, stream);
    auto vectorsByUniqueListDevice =
            toDeviceTemporary(resources_, vectorsByUniqueList, stream);
    auto uniqueListVectorStartDevice =
            toDeviceTemporary(resources_, uniqueListVectorStart, stream);
    auto uniqueListStartOffsetDevice =
            toDeviceTemporary(resources_, uniqueListStartOffset, stream);

    // Actually encode and append the vectors
    appendVectors_(
            vecs,
            residuals2dDevice,
            indices,
            uniqueListsDevice,
            vectorsByUniqueListDevice,
            uniqueListVectorStartDevice,
            uniqueListStartOffsetDevice,
            ivfIndices1dDevice,
            listOffsetDevice,
            stream);

    // We added this number
    return numAdded;
}

} // namespace gpu
} // namespace faiss
