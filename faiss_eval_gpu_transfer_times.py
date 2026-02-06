#!/usr/bin/env python3
import time
import faiss
import numpy as np
from tqdm import tqdm

# ==============================================================
# Config
# ==============================================================
INDEX_PATH = "/home/nvidia/Desktop/ivf_100m.faiss"
ITERATIONS = 100
OUTLIER_PERCENTILE = 10  # remove top/bottom 1%

# ==============================================================
# Setup GPU resources
# ==============================================================
def get_gpu_resources():
    res = faiss.StandardGpuResources()
    res.setTempMemory(1 << 26)  # 64 MB
    res.setDefaultNullStreamAllDevices()
    return res

# ==============================================================
# Main profiling
# ==============================================================
def main():
    print(f"Loading CPU index from {INDEX_PATH}...")
    cpu_index = faiss.read_index(INDEX_PATH)

    gpu_res = get_gpu_resources()

    # GpuClonerOptions
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    co.usePrecomputed = False
    co.reserveVecs = 0
    co.verbose = False

    transfer_times = []

    print(f"Profiling CPU→GPU transfer for {ITERATIONS} iterations...")
    for _ in tqdm(range(ITERATIONS), desc="CPU→GPU transfers"):
        t0 = time.time()
        _ = faiss.index_cpu_to_gpu(gpu_res, 0, cpu_index, co)
        t1 = time.time()
        transfer_times.append(t1 - t0)

    # Convert to NumPy array for percentile filtering
    transfer_times = np.array(transfer_times)
    lower = np.percentile(transfer_times, OUTLIER_PERCENTILE)
    upper = np.percentile(transfer_times, 100 - OUTLIER_PERCENTILE)

    filtered_times = transfer_times[(transfer_times >= lower) & (transfer_times <= upper)]

    print(f"CPU→GPU transfer time stats (excluding top/bottom {OUTLIER_PERCENTILE}% outliers):")
    print(f"  Min: {filtered_times.min():.6f} s")
    print(f"  Max: {filtered_times.max():.6f} s")
    print(f"  Mean: {filtered_times.mean():.6f} s")
    print(f"  Median: {np.median(filtered_times):.6f} s")

if __name__ == "__main__":
    main()
