#!/usr/bin/env python3
import os
import time
import csv
import gc

import faiss
import numpy as np
from tqdm import tqdm

# ==============================================================
# Config
# ==============================================================
INDEX_PATH = "/home/nvidia/Desktop/ivf_100m_sq8.faiss"
QUERY_PATH = "/home/nvidia/Desktop/triviaqa_encodings.npy"
TRIALS = 100
K = 5
NPROBE = 256
NUM_QUERIES = 1000  # limit queries for repeatable timing

USE_UNIFIED_MEMORY = False  # Set True only if you need UM capacity
PINNED_MEM_BYTES = 2 * 1024 * 1024 * 1024  # Large pinned staging improves H2D
TEMP_MEM_BYTES = 0 * 1024 * 1024 * 1024  # 0 to avoid pre-reserving device memory
REUSE_RESOURCES = True  # Avoid per-trial resource init overhead

def get_gpu_resources():
    res = faiss.StandardGpuResources()
    
    # Avoid reserving large temporary buffers that can cause OOM
    res.setTempMemory(TEMP_MEM_BYTES)
    
    # Pinned staging buffer for CPU -> GPU transfers
    res.setPinnedMemory(PINNED_MEM_BYTES)

    print(f"GPU Resources configured:")
    print(f"  Temp memory: {TEMP_MEM_BYTES / (1024**3):.1f} GB")
    print(f"  Pinned memory: {PINNED_MEM_BYTES / (1024**3):.1f} GB")
    
    return res

def load_faiss_gpu_index(cpu_index, nprobe, res):
    co = faiss.GpuClonerOptions()
    co.useUnifiedMemory = USE_UNIFIED_MEMORY
    co.useFloat16 = True
    co.usePrecomputed = False
    co.indicesOptions = faiss.INDICES_32_BIT
    
    print(f"\nStarting GPU transfer...")
    print(f"  Index type: {type(cpu_index).__name__}")
    print(f"  Total vectors: {cpu_index.ntotal:,}")
    if hasattr(cpu_index, 'nlist'):
        print(f"  IVF lists: {cpu_index.nlist:,}")
    
    t2 = time.perf_counter()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
    t_transfer = time.perf_counter()
    print(f"  Transfer completed: {t_transfer - t2:.3f}s")
    
    cuda_sync(res)
    t3 = time.perf_counter()
    print(f"  After sync: {t3 - t2:.3f}s")
    
    gpu_load_time = t3 - t2
    gpu_index.nprobe = nprobe
    
    return gpu_index, 0.0, gpu_load_time, gpu_load_time

def load_queries():
    queries = np.load(QUERY_PATH)
    if queries.dtype != np.float32:
        queries = queries.astype(np.float32, copy=False)
    if queries.ndim != 2:
        raise ValueError("queries must be a 2D array")
    if NUM_QUERIES > 0:
        queries = queries[:NUM_QUERIES]
    return queries

def cuda_sync(res):
    """
    Portable CUDA sync for FAISS 1.13.x
    """
    res.syncDefaultStreamCurrentDevice()

def main():
    # Fastest packed-transfer path with mmap + pin + persist
    os.environ.setdefault("FAISS_GPU_PACKED_LISTS", "1")
    os.environ.setdefault("FAISS_GPU_PACKED_LISTS_MMAP", "1")
    os.environ.setdefault("FAISS_GPU_PACKED_LISTS_MMAP_PIN", "1")
    os.environ.setdefault("FAISS_GPU_PACKED_LISTS_MMAP_PERSIST", "1")
    os.environ.setdefault("FAISS_GPU_PACKED_LISTS_MMAP_POPULATE", "1")
    os.environ.setdefault("FAISS_GPU_PACKED_LISTS_MMAP_PREFETCH", "1")
    os.environ.setdefault("FAISS_GPU_PACKED_LISTS_MMAP_MLOCK", "0")
    os.environ.setdefault("FAISS_GPU_PACKED_LISTS_DEBUG", "0")
    os.environ.setdefault("FAISS_GPU_DEVICEVECTOR_CACHE", "1")
    os.environ.setdefault("FAISS_GPU_DEVICEVECTOR_CACHE_MIN_BYTES", str(1 << 30))

    t0 = time.perf_counter()
    cpu_index = faiss.read_index(INDEX_PATH)
    t1 = time.perf_counter()
    cpu_load_time = t1 - t0

    queries = load_queries()
    gpu_times = []
    search_times = []

    res = get_gpu_resources() if REUSE_RESOURCES else None

    for trial in tqdm(range(TRIALS), desc="GPU transfers"):
        if not REUSE_RESOURCES:
            res = get_gpu_resources()
        gpu_index, _, gpu_load, _ = load_faiss_gpu_index(
            cpu_index, NPROBE, res)
        gpu_times.append(gpu_load)

        t_search_start = time.perf_counter()
        gpu_index.search(queries, K)
        cuda_sync(res)
        t_search_end = time.perf_counter()
        search_times.append(t_search_end - t_search_start)

        # Release GPU index to avoid accumulating allocations
        del gpu_index
        cuda_sync(res)
        gc.collect()

        # Explicit cleanup between trials
        if not REUSE_RESOURCES:
            del res
            gc.collect()

    if REUSE_RESOURCES:
        del res
        gc.collect()

    with open("/home/nvidia/Desktop/FINAL_ENHANCED_INDEX_TRANSFER_TIMES_100M_SQ8.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["trial", "disk_to_cpu_s", "cpu_to_gpu_s", "search_s"])
        for i, (gpu_time, search_time) in enumerate(
                zip(gpu_times, search_times), start=1):
            writer.writerow([i, cpu_load_time, gpu_time, search_time])

    print("\n===== RESULTS =====")
    print(f"Disk -> CPU Load Time (single): {cpu_load_time:.6f} s")
    print(
        f"Average CPU -> GPU Load Time ({TRIALS} trials): {sum(gpu_times) / TRIALS:.6f} s")
    print(
        f"Average GPU Search Time ({TRIALS} trials): {sum(search_times) / TRIALS:.6f} s")
    print("Saved per-trial times to transfer_times.csv\n")

if __name__ == "__main__":
    os.environ["FAISS_VERBOSE"] = "1"
    main()
