#!/usr/bin/env python3
import os
import time
import csv
import gc
import shutil

import faiss
from tqdm import tqdm

try:
    import pycuda.driver as cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

# ==============================================================
# Config
# ==============================================================
HYDRA_SHARDS = [
    "/data/indices/shards/hydra_head_0.faiss", 
    "/data/indices/shards/hydra_head_1.faiss", 
    "/data/indices/shards/hydra_head_2.faiss", 
    "/data/indices/shards/hydra_head_3.faiss", 
    "/data/indices/shards/hydra_head_4.faiss", 
    "/data/indices/shards/hydra_head_5.faiss", 
    "/data/indices/shards/hydra_head_6.faiss", 
    "/data/indices/shards/hydra_head_7.faiss"
]

TRIALS = 5
USE_UNIFIED_MEMORY = False
PINNED_MEM_BYTES = 2 * 1024 * 1024 * 1024
TEMP_MEM_BYTES = 0
REUSE_RESOURCES = True

def get_gpu_resources():
    res = faiss.StandardGpuResources()
    res.setTempMemory(TEMP_MEM_BYTES)
    res.setPinnedMemory(PINNED_MEM_BYTES)
    return res


def cuda_sync(res):
    res.syncDefaultStreamCurrentDevice()


def clear_gpu_memory():
    """Clear GPU memory between transfers."""
    if CUDA_AVAILABLE:
        try:
            cuda.Context.synchronize()
            cuda.Device(0).synchronize()
        except Exception as e:
            print(f"Warning: Could not fully clear GPU memory: {e}")
    gc.collect()


def main():
    # Enable optimized FAISS GPU paths
    os.environ["FAISS_GPU_PACKED_LISTS"] = "1"
    os.environ["FAISS_GPU_PACKED_LISTS_MMAP"] = "1"
    os.environ["FAISS_GPU_DEVICEVECTOR_CACHE"] = "1"
    os.environ["FAISS_GPU_DEVICEVECTOR_CACHE_MIN_BYTES"] = str(1 << 30)
    os.environ["FAISS_GPU_PACKED_LISTS_PROFILE"] = "0"
    os.environ["FAISS_GPU_PACKED_LISTS_DEBUG"] = "0"

    # ----------------------------------------------------------
    # Results storage
    # ----------------------------------------------------------
    results = []

    # ----------------------------------------------------------
    # Profile each shard
    # ----------------------------------------------------------
    for shard_idx, shard_path in enumerate(HYDRA_SHARDS):
        print(f"\n{'='*60}")
        print(f"Profiling Shard {shard_idx}: {shard_path}")
        print(f"{'='*60}")

        # Create fresh GPU resources for this shard
        res = get_gpu_resources()

        # Measure disk to CPU transfer (once for reference)
        t0 = time.perf_counter()
        cpu_index = faiss.read_index(shard_path)
        t1 = time.perf_counter()
        disk_to_cpu_time = t1 - t0

        print(f"Disk -> CPU Load Time: {disk_to_cpu_time:.6f} s")
        print(f"  Index type: {type(cpu_index).__name__}")
        print(f"  Total vectors: {cpu_index.ntotal:,}")
        ntotal = cpu_index.ntotal
        if hasattr(cpu_index, 'nlist'):
            print(f"  IVF lists: {cpu_index.nlist:,}")

        # Measure CPU to GPU transfer (multiple trials, keep CPU index)
        gpu_to_cpu_times = []

        for trial in tqdm(range(TRIALS), desc=f"CPU->GPU Trials"):
            os.environ["FAISS_GPU_PACKED_CACHE_PATH"] = f"/data/indices/hydra_cache_shards/hydra_shard_{shard_idx}"
            
            co = faiss.GpuClonerOptions()
            co.useUnifiedMemory = USE_UNIFIED_MEMORY
            co.useFloat16 = True
            co.usePrecomputed = False
            co.indicesOptions = faiss.INDICES_32_BIT

            t_start = time.perf_counter()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
            cuda_sync(res)
            t_end = time.perf_counter()

            gpu_to_cpu_times.append(t_end - t_start)

            # Cleanup only GPU index, keep CPU index in memory
            del gpu_index
            cuda_sync(res)
            gc.collect()

        avg_gpu_time = sum(gpu_to_cpu_times) / len(gpu_to_cpu_times)
        min_gpu_time = min(gpu_to_cpu_times)
        max_gpu_time = max(gpu_to_cpu_times)

        print(f"CPU -> GPU Transfer Times ({TRIALS} trials):")
        print(f"  Average: {avg_gpu_time:.6f} s")
        print(f"  Min: {min_gpu_time:.6f} s")
        print(f"  Max: {max_gpu_time:.6f} s")

        # Store results
        for trial, gpu_time in enumerate(gpu_to_cpu_times, start=1):
            results.append({
                'shard_idx': shard_idx,
                'shard_name': os.path.basename(shard_path),
                'trial': trial,
                'disk_to_cpu_s': disk_to_cpu_time,
                'cpu_to_gpu_s': gpu_time,
                'ntotal': ntotal
            })

        # Cleanup CPU index and GPU resources after all trials for this shard
        del cpu_index
        del res
        clear_gpu_memory()

    if REUSE_RESOURCES:
        gc.collect()

    # ----------------------------------------------------------
    # Save Results
    # ----------------------------------------------------------
    os.makedirs("data", exist_ok=True)
    output_path = "data/HYDRA_SHARD_TRANSFER_TIMES.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=['shard_idx', 'shard_name', 'trial', 'disk_to_cpu_s', 'cpu_to_gpu_s', 'ntotal'])
        writer.writeheader()
        writer.writerows(results)

    # ----------------------------------------------------------
    # Summary Statistics
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"TRANSFER SUMMARY")
    print(f"{'='*60}")

    for shard_idx in range(len(HYDRA_SHARDS)):
        shard_results = [r for r in results if r['shard_idx'] == shard_idx]
        disk_to_cpu = shard_results[0]['disk_to_cpu_s']
        gpu_times = [r['cpu_to_gpu_s'] for r in shard_results]
        avg_gpu = sum(gpu_times) / len(gpu_times)
        
        print(f"\nShard {shard_idx}:")
        print(f"  Disk->CPU: {disk_to_cpu:.6f} s")
        print(f"  CPU->GPU (avg): {avg_gpu:.6f} s")
        print(f"  Vectors: {shard_results[0]['ntotal']:,}")

    print(f"\nFull results saved to {output_path}\n")


if __name__ == "__main__":
    os.environ["FAISS_VERBOSE"] = "0"
    main()
