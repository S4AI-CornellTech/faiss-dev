#!/usr/bin/env python3
import os
import time
import csv
import gc
import shutil

import faiss
import numpy as np
from tqdm import tqdm

# ==============================================================
# Config
# ==============================================================
INDEX_SIZE = "100m"
INDEX_QUANTIZATION = "sq8"

INDEX_PATH = f"/data/indices/{INDEX_QUANTIZATION}/ivf_{INDEX_SIZE}_{INDEX_QUANTIZATION}.faiss"
# INDEX_PATH = f"/data/indices/shards/hydra_head_0.faiss"
QUERY_PATH = "triviaqa_encodings.npy"
TRIALS = 100
K = 5
NPROBE = 256
BATCH_SIZE = 32
NUM_QUERIES = 100  # limit queries for repeatable timing

USE_UNIFIED_MEMORY = False
PINNED_MEM_BYTES = 2 * 1024 * 1024 * 1024
TEMP_MEM_BYTES = 0
REUSE_RESOURCES = True


def clear_cache():
    cache_path = "/data/indices/hydra_cache_shards"
    if os.path.exists(cache_path):
        for filename in os.listdir(cache_path):
            file_path = os.path.join(cache_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")


def get_gpu_resources():
    res = faiss.StandardGpuResources()
    res.setTempMemory(TEMP_MEM_BYTES)
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

    t_start = time.perf_counter()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
    cuda_sync(res)
    t_end = time.perf_counter()

    gpu_load_time = t_end - t_start
    print(f"  Transfer completed: {gpu_load_time:.3f}s")

    gpu_index.nprobe = nprobe

    return gpu_index, gpu_load_time


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
    res.syncDefaultStreamCurrentDevice()


def main():
    # Enable optimized FAISS GPU paths
    os.environ.setdefault("FAISS_GPU_PACKED_LISTS", "1")
    os.environ.setdefault("FAISS_GPU_PACKED_LISTS_MMAP", "1")
    os.environ.setdefault("FAISS_GPU_DEVICEVECTOR_CACHE", "1")
    os.environ.setdefault("FAISS_GPU_DEVICEVECTOR_CACHE_MIN_BYTES", str(1 << 30))
    os.environ.setdefault("FAISS_GPU_PACKED_LISTS_PROFILE", "1")
    os.environ.setdefault("FAISS_GPU_PACKED_LISTS_DEBUG", "0")
    os.environ.setdefault("FAISS_GPU_PACKED_CACHE_PATH", "/data/indices/hydra_cache_shards/test")

    clear_cache()

    # ----------------------------------------------------------
    # Load CPU index
    # ----------------------------------------------------------
    t0 = time.perf_counter()
    cpu_index = faiss.read_index(INDEX_PATH)
    t1 = time.perf_counter()
    cpu_load_time = t1 - t0

    queries = load_queries()

    gpu_times = []
    batch_latency_times = []

    res = get_gpu_resources() if REUSE_RESOURCES else None

    # ----------------------------------------------------------
    # Trials
    # ----------------------------------------------------------
    for trial in tqdm(range(TRIALS), desc="Trials"):
        if not REUSE_RESOURCES:
            res = get_gpu_resources()

        gpu_index, gpu_load_time = load_faiss_gpu_index(
            cpu_index, NPROBE, res
        )
        gpu_times.append(gpu_load_time)

        # ----------------------------
        # Per-batch latency timing
        # ----------------------------
        batch_latencies = []
        all_distances = []
        all_indices = []

        num_queries = queries.shape[0]

        for start in range(0, num_queries, BATCH_SIZE):
            end = min(start + BATCH_SIZE, num_queries)
            batch = queries[start:end]

            t_batch_start = time.perf_counter()
            distances, indices = gpu_index.search(batch, K)
            cuda_sync(res)
            t_batch_end = time.perf_counter()

            batch_latencies.append(t_batch_end - t_batch_start)

            all_distances.append(distances)
            all_indices.append(indices)

        # Optional result concat (keeps behavior same)
        distances = np.vstack(all_distances)
        indices = np.vstack(all_indices)

        avg_batch_latency = sum(batch_latencies) / len(batch_latencies)
        batch_latency_times.append(avg_batch_latency)

        # Cleanup GPU index
        del gpu_index
        cuda_sync(res)
        gc.collect()

        if not REUSE_RESOURCES:
            del res
            gc.collect()

    if REUSE_RESOURCES:
        del res
        gc.collect()

    # ----------------------------------------------------------
    # Save Results
    # ----------------------------------------------------------
    os.makedirs("data", exist_ok=True)

    output_path = f"data/transfers/ENHANCED_TRANSFER_TIMES_{INDEX_SIZE}_{INDEX_QUANTIZATION}.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["trial", "disk_to_cpu_s", "cpu_to_gpu_s", "avg_batch_latency_s"]
        )
        for i, (gpu_time, batch_lat) in enumerate(
            zip(gpu_times, batch_latency_times), start=1
        ):
            writer.writerow([i, cpu_load_time, gpu_time, batch_lat])

    print("\n===== RESULTS =====")
    print(f"Disk -> CPU Load Time (single): {cpu_load_time:.6f} s")
    print(f"Average CPU -> GPU Load Time ({TRIALS} trials): {sum(gpu_times) / TRIALS:.6f} s")
    print(f"Average GPU Per-Batch Latency ({TRIALS} trials): {sum(batch_latency_times) / TRIALS:.6f} s")
    print(f"Saved per-trial times to {output_path}\n")


if __name__ == "__main__":
    os.environ["FAISS_VERBOSE"] = "1"
    main()
