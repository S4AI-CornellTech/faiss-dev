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
QUERY_FILE = "/home/nvidia/Desktop/triviaqa_encodings.npy"
OUTPUT_PATH = "/home/nvidia/Desktop/ENHANCED_[tmp]_INDEX_TRANSFER_TIMES_100M_SQ8.csv"

NPROBE = 256
BATCH_SIZE = 32
RETRIEVED_DOCS = 5
MAX_BATCHES = 100
TRIALS = 2

USE_UNIFIED_MEMORY = False  # Set True only if you need UM capacity
PINNED_MEM_BYTES = 16 * 1024 * 1024 * 1024  # 4 GiB pinned staging
TRANSFER_ONLY = True  # Skip query timing when True

def get_gpu_resources():
    res = faiss.StandardGpuResources()
    
    # Increase temp memory even more for large indices
    res.setTempMemory(24 * 1024 * 1024 * 1024)  # 24 GB
    
    # Larger pinned memory pool
    res.setPinnedMemory(PINNED_MEM_BYTES)  # 8 GiB

    print(f"GPU Resources configured:")
    print(f"  Temp memory: 24 GB")
    print(f"  Pinned memory: {PINNED_MEM_BYTES / (1024**3):.1f} GB")
    
    return res

def load_faiss_gpu_index(cpu_index, nprobe, res):
    co = faiss.GpuClonerOptions()
    co.useUnifiedMemory = False
    co.useFloat16 = False
    co.usePrecomputed = False
    co.indicesOptions = faiss.INDICES_64_BIT
    
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

def cuda_sync(res):
    """
    Portable CUDA sync for FAISS 1.13.x
    """
    res.syncDefaultStreamCurrentDevice()

def perform_queries(index, k, embeddings, batch_size, res):
    query_times = []

    total_queries = min(len(embeddings), MAX_BATCHES * batch_size)

    for i in range(0, total_queries, batch_size):
        batch = embeddings[i:i + batch_size]

        t0 = time.perf_counter()
        index.search(batch, k)
        cuda_sync(res)
        t1 = time.perf_counter()

        query_times.append(t1 - t0)

    if not query_times:
        return 0.0

    return sum(query_times) / len(query_times)

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    queries = np.load(QUERY_FILE).astype(np.float32)

    cpu_index = None
    cpu_load_time = 0.0
    t0 = time.perf_counter()
    cpu_index = faiss.read_index(INDEX_PATH)
    t1 = time.perf_counter()
    cpu_load_time = t1 - t0

    sum_gpu = 0.0
    sum_query = 0.0

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cpu_load_time_s_once",
            "gpu_load_time_s",
            "total_load_time_s",
            "avg_query_time_s",
            "use_unified_memory",
            "pinned_mem_bytes",
            "transfer_only"
        ])

        for trial in tqdm(range(TRIALS), desc="Trials"):
            res = get_gpu_resources()

            index, _, gpu_load, _ = load_faiss_gpu_index(
                cpu_index, NPROBE, res
            )
            total_load = cpu_load_time + gpu_load

            avg_query_time = 0.0
            if not TRANSFER_ONLY:
                avg_query_time = perform_queries(
                    index, RETRIEVED_DOCS, queries, BATCH_SIZE, res
                )

            writer.writerow([
                cpu_load_time,
                gpu_load,
                total_load,
                avg_query_time,
                USE_UNIFIED_MEMORY,
                PINNED_MEM_BYTES,
                TRANSFER_ONLY
            ])
            sum_gpu += gpu_load
            sum_query += avg_query_time

            # Explicit cleanup between trials
            del index
            del res
            gc.collect()

    print("\n===== AVERAGES OVER ALL TRIALS =====")
    print(f"Average CPU Load Time:   {cpu_load_time:.6f} s")
    print(f"Average GPU Load Time:   {sum_gpu / TRIALS:.6f} s")
    print(f"Average Total Load Time: {cpu_load_time + (sum_gpu / TRIALS):.6f} s")
    if TRANSFER_ONLY:
        print("Average Query Latency:   0.000000 s (transfer only)\n")
    else:
        print(f"Average Query Latency:   {sum_query / TRIALS:.6f} s\n")

if __name__ == "__main__":
    os.environ["FAISS_VERBOSE"] = "1"
    main()
