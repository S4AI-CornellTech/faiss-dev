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
INDEX_PATH = "/home/nvidia/Desktop/ivf_100m_pq64.faiss"
QUERY_FILE = "/home/nvidia/Desktop/triviaqa_encodings.npy"
OUTPUT_PATH = "/home/nvidia/Desktop/ENHANCED_[Save_and_load_unified_memory]_INDEX_TRANSFER_TIMES_100M_PQ64.csv"

NPROBE = 256
BATCH_SIZE = 32
RETRIEVED_DOCS = 5
MAX_BATCHES = 32
TRIALS = 100

# ==============================================================
# Performance toggles
# ==============================================================
USE_UNIFIED_MEMORY = True  # Set True only if you need UM capacity
PINNED_MEM_BYTES = 4 * 1024 * 1024 * 1024  # 4 GiB pinned staging
TRANSFER_ONLY = True  # Skip query timing when True

# ==============================================================
# GPU setup (GH200-safe)
# ==============================================================
def get_gpu_resources():
    res = faiss.StandardGpuResources()

    # GH200 needs LARGE temp memory
    res.setTempMemory(16 * 1024 * 1024 * 1024)  # 16 GB

    # Use a pinned pool for faster H2D transfers
    res.setPinnedMemory(PINNED_MEM_BYTES)

    return res


def cuda_sync(res):
    """
    Portable CUDA sync for FAISS 1.13.x
    """
    res.syncDefaultStreamCurrentDevice()


# ==============================================================
# Index loading
# ==============================================================
def load_faiss_gpu_index(cpu_index, nprobe, res):
    cpu_load_time = 0.0

    # GPU clone options (GH200-safe)
    co = faiss.GpuClonerOptions()
    co.useUnifiedMemory = USE_UNIFIED_MEMORY
    co.useFloat16 = True

    # GPU transfer
    t2 = time.perf_counter()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
    cuda_sync(res)
    t3 = time.perf_counter()
    gpu_load_time = t3 - t2

    gpu_index.nprobe = nprobe

    return gpu_index, cpu_load_time, gpu_load_time, cpu_load_time + gpu_load_time


# ==============================================================
# Query benchmark
# ==============================================================
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


# ==============================================================
# Main
# ==============================================================
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
