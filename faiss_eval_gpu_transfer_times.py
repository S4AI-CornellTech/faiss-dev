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
TRIALS = 5

USE_UNIFIED_MEMORY = False  # Set True only if you need UM capacity
PINNED_MEM_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB pinned staging
TEMP_MEM_BYTES = 0  # 0 to avoid pre-reserving device memory

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

def main():
    t0 = time.perf_counter()
    cpu_index = faiss.read_index(INDEX_PATH)
    t1 = time.perf_counter()
    cpu_load_time = t1 - t0

    sum_gpu = 0.0

    for trial in tqdm(range(TRIALS), desc="GPU transfers"):
        res = get_gpu_resources()
        _, _, gpu_load, _ = load_faiss_gpu_index(cpu_index, 1, res)
        sum_gpu += gpu_load

        # Explicit cleanup between trials
        del res
        gc.collect()

    print("\n===== RESULTS =====")
    print(f"Disk -> CPU Load Time (single): {cpu_load_time:.6f} s")
    print(f"Average CPU -> GPU Load Time ({TRIALS} trials): {sum_gpu / TRIALS:.6f} s\n")

if __name__ == "__main__":
    os.environ["FAISS_VERBOSE"] = "1"
    main()
