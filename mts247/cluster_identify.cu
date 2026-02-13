// cosine_profile.cu
#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>

namespace {

inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

constexpr int NUM_VECS = 16384;   // 16K
constexpr int DIM      = 768;

// ----------------------------------------
// Kernel: compute cosine similarity
// ----------------------------------------
__global__ void cosineKernel(
        const float* __restrict__ vectors,
        const float* __restrict__ query,
        float* __restrict__ output,
        int dim) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NUM_VECS) return;

    const float* vec = vectors + idx * dim;

    float dot = 0.0f;
    float norm_vec = 0.0f;
    float norm_query = 0.0f;

    for (int i = 0; i < dim; ++i) {
        float v = vec[i];
        float q = query[i];
        dot += v * q;
        norm_vec += v * v;
        norm_query += q * q;
    }

    output[idx] = dot / (sqrtf(norm_vec) * sqrtf(norm_query) + 1e-8f);
}

// ----------------------------------------
// Timing helper
// ----------------------------------------
double profileKernel(
        const float* d_vectors,
        const float* d_query,
        float* d_output,
        int iters,
        int warmup) {

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "event create start");
    checkCuda(cudaEventCreate(&stop), "event create stop");

    dim3 block(256);
    dim3 grid((NUM_VECS + block.x - 1) / block.x);

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        cosineKernel<<<grid, block>>>(d_vectors, d_query, d_output, DIM);
    }
    checkCuda(cudaDeviceSynchronize(), "warmup sync");

    checkCuda(cudaEventRecord(start), "record start");

    for (int i = 0; i < iters; ++i) {
        cosineKernel<<<grid, block>>>(d_vectors, d_query, d_output, DIM);
    }

    checkCuda(cudaEventRecord(stop), "record stop");
    checkCuda(cudaEventSynchronize(stop), "event sync");

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "elapsed time");

    checkCuda(cudaEventDestroy(start), "destroy start");
    checkCuda(cudaEventDestroy(stop), "destroy stop");

    return (ms * 1e-3) / static_cast<double>(iters);
}

} // namespace

int main() {

    const size_t vecBytes = NUM_VECS * DIM * sizeof(float);
    const size_t queryBytes = DIM * sizeof(float);
    const size_t outputBytes = NUM_VECS * sizeof(float);

    // Host buffers
    std::vector<float> h_vectors(NUM_VECS * DIM);
    std::vector<float> h_query(DIM);

    // Random initialization
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    for (auto& v : h_vectors) v = dist(rng);
    for (auto& v : h_query)   v = dist(rng);

    // Device buffers
    float* d_vectors = nullptr;
    float* d_query   = nullptr;
    float* d_output  = nullptr;

    checkCuda(cudaMalloc(&d_vectors, vecBytes), "cudaMalloc vectors");
    checkCuda(cudaMalloc(&d_query, queryBytes), "cudaMalloc query");
    checkCuda(cudaMalloc(&d_output, outputBytes), "cudaMalloc output");

    checkCuda(cudaMemcpy(d_vectors, h_vectors.data(), vecBytes, cudaMemcpyHostToDevice),
              "memcpy vectors");
    checkCuda(cudaMemcpy(d_query, h_query.data(), queryBytes, cudaMemcpyHostToDevice),
              "memcpy query");

    int warmup = 5;
    int iters  = 50;

    double secPerIter = profileKernel(d_vectors, d_query, d_output, iters, warmup);

    std::cout << "Vectors: " << NUM_VECS << "\n";
    std::cout << "Dimension: " << DIM << "\n";
    std::cout << "Average time per full 16K cosine pass: "
              << secPerIter * 1e6 << " us\n";

    double totalFlops = static_cast<double>(NUM_VECS) * DIM * 4; 
    // 1 mul + 1 mul + 1 mul + 2 adds approx per dim
    double gflops = (totalFlops / secPerIter) / 1e9;

    std::cout << "Approx GFLOPs: " << gflops << "\n";

    cudaFree(d_vectors);
    cudaFree(d_query);
    cudaFree(d_output);

    return 0;
}
