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
INDEX_PATH = "/home/nvidia/Desktop/ivf_100m.faiss"
QUERY_FILE = "/home/nvidia/Desktop/triviaqa_encodings.npy"
OUTPUT_PATH = "/home/nvidia/Desktop/ENHANCED_[Save_and_load]_INDEX_TRANSFER_TIMES_100M.csv"

NPROBE = 256
BATCH_SIZE = 32
RETRIEVED_DOCS = 5
MAX_BATCHES = 100
TRIALS = 100

# ==============================================================
# GPU setup (GH200-safe)
# ==============================================================
def get_gpu_resources():
    res = faiss.StandardGpuResources()

    # GH200 needs LARGE temp memory
    res.setTempMemory(16 * 1024 * 1024 * 1024)  # 16 GB

    # Use default stream consistently
    res.setDefaultNullStreamAllDevices()

    return res


def cuda_sync(res):
    """
    Portable CUDA sync for FAISS 1.13.x
    """
    res.syncDefaultStreamCurrentDevice()


# ==============================================================
# Index loading
# ==============================================================
def load_faiss_gpu_index(index_path, nprobe, res):
    # CPU load
    t0 = time.time()
    cpu_index = faiss.read_index(index_path)
    t1 = time.time()
    cpu_load_time = t1 - t0

    # GPU clone options (GH200-safe)
    co = faiss.GpuClonerOptions()
    co.useUnifiedMemory = True     # CRITICAL on GH200
    co.useFloat16 = False          # Unsafe during IVF cloning

    # GPU transfer
    t2 = time.time()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
    cuda_sync(res)
    t3 = time.time()
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

        t0 = time.time()
        index.search(batch, k)
        cuda_sync(res)
        t1 = time.time()

        query_times.append(t1 - t0)

    return sum(query_times) / len(query_times)


# ==============================================================
# Main
# ==============================================================
def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    queries = np.load(QUERY_FILE).astype(np.float32)

    sum_cpu = 0.0
    sum_gpu = 0.0
    sum_total = 0.0
    sum_query = 0.0

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cpu_load_time_s",
            "gpu_load_time_s",
            "total_load_time_s",
            "avg_query_time_s"
        ])

        for trial in tqdm(range(TRIALS), desc="Trials"):
            res = get_gpu_resources()

            index, cpu_load, gpu_load, total_load = load_faiss_gpu_index(
                INDEX_PATH, NPROBE, res
            )

            avg_query_time = perform_queries(
                index, RETRIEVED_DOCS, queries, BATCH_SIZE, res
            )

            writer.writerow([
                cpu_load,
                gpu_load,
                total_load,
                avg_query_time
            ])

            sum_cpu += cpu_load
            sum_gpu += gpu_load
            sum_total += total_load
            sum_query += avg_query_time

            # Explicit cleanup between trials
            del index
            del res
            gc.collect()

    print("\n===== AVERAGES OVER ALL TRIALS =====")
    print(f"Average CPU Load Time:   {sum_cpu / TRIALS:.6f} s")
    print(f"Average GPU Load Time:   {sum_gpu / TRIALS:.6f} s")
    print(f"Average Total Load Time: {sum_total / TRIALS:.6f} s")
    print(f"Average Query Latency:   {sum_query / TRIALS:.6f} s\n")


if __name__ == "__main__":
    os.environ["FAISS_VERBOSE"] = "1"
    main()
