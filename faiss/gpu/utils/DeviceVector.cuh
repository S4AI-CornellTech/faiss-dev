/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/gpu/utils/StaticUtils.h>
#include <faiss/impl/FaissAssert.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace faiss {
namespace gpu {

// For growing GPU allocations:
// Below this size, we always round the allocation size up to the next highest
// power of 2
constexpr size_t kDeviceVector_2x_Limit = 4 * 1024 * 1024;

// Otherwise, below this size, we always round the allocation size up by a
// factor of 1.25. Otherwise, all reallocations are exact to the newly requested
// size.
constexpr size_t kDeviceVector_1_25x_Limit = 128 * 1024 * 1024;

/// A simple version of thrust::device_vector<T>, but has more control
/// over streams, whether resize() initializes new space with T() (which we
/// don't want), and control on how much the reserved space grows by
/// upon resize/reserve. It is also meant for POD types only.
///
/// Any new memory allocated is automatically zeroed before being presented to
/// the user.
template <typename T>
class DeviceVector {
   public:
    DeviceVector(GpuResources* res, AllocInfo allocInfo)
            : num_(0), capacity_(0), res_(res), allocInfo_(allocInfo) {
        FAISS_ASSERT(res_);
    }

    ~DeviceVector() {
        clear();
    }

    // Clear all allocated memory; reset to zero size
    void clear() {
        cacheOrRelease_();
        num_ = 0;
        capacity_ = 0;
    }

    size_t size() const {
        return num_;
    }
    size_t capacity() const {
        return capacity_;
    }
    T* data() {
        return (T*)alloc_.data;
    }
    const T* data() const {
        return (const T*)alloc_.data;
    }

    template <typename OutT>
    std::vector<OutT> copyToHost(cudaStream_t stream) const {
        FAISS_ASSERT(num_ * sizeof(T) % sizeof(OutT) == 0);

        std::vector<OutT> out((num_ * sizeof(T)) / sizeof(OutT));

        if (num_ > 0) {
            FAISS_ASSERT(data());
            CUDA_VERIFY(cudaMemcpyAsync(
                    out.data(),
                    data(),
                    num_ * sizeof(T),
                    cudaMemcpyDeviceToHost,
                    stream));
        }

        return out;
    }

    // Returns true if we actually reallocated memory
    // If `reserveExact` is true, then we reserve only the memory that
    // we need for what we're appending
    bool append(
            const T* d,
            size_t n,
            cudaStream_t stream,
            bool reserveExact = false) {
        bool mem = false;

        if (n > 0) {
            size_t reserveSize = num_ + n;
            if (!reserveExact) {
                reserveSize = getNewCapacity_(reserveSize);
            }

            mem = reserve(reserveSize, stream);

            int dev = getDeviceForAddress(d);
            if (dev == -1) {
                CUDA_VERIFY(cudaMemcpyAsync(
                        data() + num_,
                        d,
                        n * sizeof(T),
                        cudaMemcpyHostToDevice,
                        stream));
            } else {
                CUDA_VERIFY(cudaMemcpyAsync(
                        data() + num_,
                        d,
                        n * sizeof(T),
                        cudaMemcpyDeviceToDevice,
                        stream));
            }
            num_ += n;
        }

        return mem;
    }

    // Returns true if we actually reallocated memory
    bool resize(size_t newSize, cudaStream_t stream) {
        bool mem = false;

        if (newSize > capacity_) {
            mem = reserve(getNewCapacity_(newSize), stream);
        }

        // Don't bother zero initializing the newly accessible memory
        // (unlike thrust::device_vector)
        num_ = newSize;

        return mem;
    }

    // Returns true if we actually reallocated memory; skips zero-fill
    bool resizeNoInitExact(size_t newSize, cudaStream_t stream) {
        bool mem = false;

        if (newSize > capacity_) {
            realloc_(newSize, stream, false);
            mem = true;
        }

        num_ = newSize;
        return mem;
    }

    // Set all entries in the vector to `value`
    void setAll(const T& value, cudaStream_t stream) {
        if (num_ > 0) {
            thrust::fill(
                    thrust::cuda::par.on(stream), data(), data() + num_, value);
        }
    }

    // Set the specific value at a given index to `value`
    void setAt(size_t idx, const T& value, cudaStream_t stream) {
        FAISS_ASSERT(idx < num_);
        CUDA_VERIFY(cudaMemcpyAsync(
                data() + idx,
                &value,
                sizeof(T),
                cudaMemcpyHostToDevice,
                stream));
    }

    // Copy a specific value at a given index to the host
    T getAt(size_t idx, cudaStream_t stream) {
        FAISS_ASSERT(idx < num_);

        T out;
        CUDA_VERIFY(cudaMemcpyAsync(
                &out, data() + idx, sizeof(T), cudaMemcpyDeviceToHost, stream));

        return out;
    }

    // Clean up after oversized allocations, while leaving some space to
    // remain for subsequent allocations (if `exact` false) or to
    // exactly the space we need (if `exact` true); returns space
    // reclaimed in bytes
    size_t reclaim(bool exact, cudaStream_t stream) {
        size_t free = capacity_ - num_;

        if (exact) {
            realloc_(num_, stream, true);
            return free * sizeof(T);
        }

        // If more than 1/4th of the space is free, then we want to
        // truncate to only having 1/8th of the space free; this still
        // preserves some space for new elements, but won't force us to
        // double our size right away
        if (free > (capacity_ / 4)) {
            size_t newFree = capacity_ / 8;
            size_t newCapacity = num_ + newFree;

            size_t oldCapacity = capacity_;
            FAISS_ASSERT(newCapacity < oldCapacity);

            realloc_(newCapacity, stream, true);

            return (oldCapacity - newCapacity) * sizeof(T);
        }

        return 0;
    }

    // Returns true if we actually reallocated memory
    bool reserve(size_t newCapacity, cudaStream_t stream) {
        if (newCapacity <= capacity_) {
            return false;
        }

        // Otherwise, we need new space.
        realloc_(newCapacity, stream, true);
        return true;
    }

   private:
    void realloc_(size_t newCapacity, cudaStream_t stream, bool zeroFillNewSpace) {
        FAISS_ASSERT(num_ <= newCapacity);

        size_t newSizeInBytes = newCapacity * sizeof(T);
        size_t oldSizeInBytes = num_ * sizeof(T);

        // The new allocation will occur on this stream
        allocInfo_.stream = stream;

        GpuMemoryReservation newAlloc;
        size_t allocBytes = 0;
        if (tryGetCachedAlloc_(newSizeInBytes, newAlloc, allocBytes)) {
            newSizeInBytes = allocBytes;
            newCapacity = newSizeInBytes / sizeof(T);
        } else {
            newAlloc = res_->allocMemoryHandle(
                    AllocRequest(allocInfo_, newSizeInBytes));
            allocBytes = newSizeInBytes;
        }

        // Copy over any old data
        if (oldSizeInBytes > 0) {
            CUDA_VERIFY(cudaMemcpyAsync(
                newAlloc.data,
                data(),
                oldSizeInBytes,
                cudaMemcpyDeviceToDevice,
                stream));
        }

        // Zero out the new space past the data we just copied
        if (zeroFillNewSpace && newSizeInBytes > oldSizeInBytes) {
            CUDA_VERIFY(cudaMemsetAsync(
                (uint8_t*)newAlloc.data + oldSizeInBytes,
                0,
                newSizeInBytes - oldSizeInBytes,
                stream));
        }

        auto oldAlloc = std::move(alloc_);
        alloc_ = std::move(newAlloc);
        capacity_ = newCapacity;

        if (oldAlloc.res) {
            cacheAlloc_(std::move(oldAlloc));
        }
    }

    void cacheOrRelease_() {
        if (!alloc_.res) {
            return;
        }

        if (!cacheAlloc_(std::move(alloc_))) {
            alloc_.release();
        }
    }

    bool cacheAlloc_(GpuMemoryReservation&& alloc) {
        if (!alloc.res) {
            return false;
        }

        if (!cacheEnabled_() || alloc.size < cacheMinBytes_()) {
            return false;
        }

        auto key = cacheKey_();
        auto& map = cacheMap_();
        std::lock_guard<std::mutex> lock(cacheMutex_());
        auto& entry = map[key];
        if (entry.alloc.res && entry.size >= alloc.size) {
            return false;
        }

        entry.alloc = std::move(alloc);
        entry.size = entry.alloc.size;
        return true;
    }

    bool tryGetCachedAlloc_(
            size_t sizeBytes,
            GpuMemoryReservation& out,
            size_t& outBytes) {
        if (!cacheEnabled_() || sizeBytes < cacheMinBytes_()) {
            return false;
        }

        auto key = cacheKey_();
        auto& map = cacheMap_();
        std::lock_guard<std::mutex> lock(cacheMutex_());
        auto it = map.find(key);
        if (it == map.end()) {
            return false;
        }

        if (!it->second.alloc.res || it->second.size < sizeBytes) {
            return false;
        }

        out = std::move(it->second.alloc);
        outBytes = it->second.size;
        map.erase(it);
        return true;
    }

    struct CacheEntry {
        GpuMemoryReservation alloc;
        size_t size = 0;
    };

    uint64_t cacheKey_() const {
        return (static_cast<uint64_t>(allocInfo_.device) & 0xffff) |
                (static_cast<uint64_t>(allocInfo_.space) << 16) |
                (static_cast<uint64_t>(allocInfo_.type) << 24);
    }

    static std::unordered_map<uint64_t, CacheEntry>& cacheMap_() {
        static std::unordered_map<uint64_t, CacheEntry> map;
        return map;
    }

    static void clearCacheForResource_(GpuResources* res) {
        if (!res) {
            return;
        }
        auto& map = cacheMap_();
        std::lock_guard<std::mutex> lock(cacheMutex_());
        for (auto it = map.begin(); it != map.end();) {
            if (it->second.alloc.res == res) {
                it->second.alloc.release();
                it = map.erase(it);
            } else {
                ++it;
            }
        }
    }

    static std::mutex& cacheMutex_() {
        static std::mutex mutex;
        return mutex;
    }

    static bool cacheEnabled_() {
        static int enabled = -1;
        if (enabled < 0) {
            const char* env = std::getenv("FAISS_GPU_DEVICEVECTOR_CACHE");
            enabled = (env && std::strcmp(env, "1") == 0) ? 1 : 0;
        }
        return enabled == 1;
    }

    static size_t cacheMinBytes_() {
        static size_t minBytes = 0;
        if (minBytes == 0) {
            const char* env =
                    std::getenv("FAISS_GPU_DEVICEVECTOR_CACHE_MIN_BYTES");
            if (env && *env) {
                minBytes = std::strtoull(env, nullptr, 10);
            }
            if (minBytes == 0) {
                minBytes = 1024ULL * 1024ULL * 1024ULL;
            }
        }
        return minBytes;
    }

   public:
    // Clear cached allocations associated with the given resources instance.
    static void clearCacheForResource(GpuResources* res) {
        clearCacheForResource_(res);
    }

    size_t getNewCapacity_(size_t preferredSize) {
        if (preferredSize <= kDeviceVector_2x_Limit) {
            return utils::nextHighestPowerOf2(preferredSize);
        } else if (preferredSize <= kDeviceVector_1_25x_Limit) {
            return preferredSize + (preferredSize >> 2);
        } else {
            return preferredSize;
        }
    }

    /// Our current memory allocation, if any
    GpuMemoryReservation alloc_;

    /// current valid number of T present
    size_t num_;

    /// current space of T present (bytes is sizeof(T) * capacity_)
    size_t capacity_;

    /// Where we should allocate memory
    GpuResources* res_;

    /// How we should allocate memory
    AllocInfo allocInfo_;
};

} // namespace gpu
} // namespace faiss
