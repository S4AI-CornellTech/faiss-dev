// memcpy_test.cu
#include <cuda_runtime.h>
#include <cstring>
#include <iostream>

namespace {

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

double runCopy(
        void* dst,
        void* src,
        size_t bytes,
        cudaMemcpyKind kind,
        cudaStream_t stream,
        int warmupIters,
        int iters) {
    cudaEvent_t start;
    cudaEvent_t stop;

    checkCuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    checkCuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    for (int i = 0; i < warmupIters; ++i) {
        checkCuda(cudaMemcpyAsync(dst, src, bytes, kind, stream), "cudaMemcpyAsync warmup");
    }
    checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize warmup");

    checkCuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
    for (int i = 0; i < iters; ++i) {
        checkCuda(cudaMemcpyAsync(dst, src, bytes, kind, stream), "cudaMemcpyAsync");
    }
    checkCuda(cudaEventRecord(stop, stream), "cudaEventRecord(stop)");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");

    checkCuda(cudaEventDestroy(start), "cudaEventDestroy(start)");
    checkCuda(cudaEventDestroy(stop), "cudaEventDestroy(stop)");

    return (ms * 1e-3) / static_cast<double>(iters);
}

void printResult(const char* label, size_t bytes, double secPerIter) {
    double gb = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    double gbps = gb / secPerIter;
    std::cout << label << " avg GB/s: " << gbps << "\n";
}

} // namespace

int main(int argc, char** argv) {
    size_t bytes = 8ULL * 1024ULL * 1024ULL * 1024ULL; // 8 GB
    int iters = 10;
    int warmupIters = 2;
    bool doH2D = true;
    bool doD2H = true;

    if (argc > 1) {
        bytes = static_cast<size_t>(std::stoull(argv[1])) * 1024ULL * 1024ULL * 1024ULL;
    }
    if (argc > 2) {
        iters = std::stoi(argv[2]);
    }
    if (argc > 3) {
        std::string mode = argv[3];
        doH2D = (mode == "h2d" || mode == "both");
        doD2H = (mode == "d2h" || mode == "both");
    }

    void* h = nullptr;
    void* d = nullptr;
    cudaStream_t s = nullptr;

    checkCuda(cudaHostAlloc(&h, bytes, cudaHostAllocPortable), "cudaHostAlloc");
    checkCuda(cudaMalloc(&d, bytes), "cudaMalloc");
    checkCuda(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");

    std::memset(h, 0xA5, bytes);
    checkCuda(cudaMemsetAsync(d, 0x5A, bytes, s), "cudaMemsetAsync");
    checkCuda(cudaStreamSynchronize(s), "cudaStreamSynchronize init");

    std::cout << "Buffer size GB: "
              << (static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0))
              << "\n";
    std::cout << "Iterations: " << iters << " (warmup " << warmupIters << ")\n";

    if (doH2D) {
        double sec = runCopy(d, h, bytes, cudaMemcpyHostToDevice, s, warmupIters, iters);
        printResult("H2D", bytes, sec);
    }

    if (doD2H) {
        double sec = runCopy(h, d, bytes, cudaMemcpyDeviceToHost, s, warmupIters, iters);
        printResult("D2H", bytes, sec);
    }

    checkCuda(cudaFree(d), "cudaFree");
    checkCuda(cudaFreeHost(h), "cudaFreeHost");
    checkCuda(cudaStreamDestroy(s), "cudaStreamDestroy");
    return 0;
}
