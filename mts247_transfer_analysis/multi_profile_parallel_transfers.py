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
    # "/data/indices/shards/hydra_head_5.faiss", 
    # "/data/indices/shards/hydra_head_6.faiss", 
    # "/data/indices/shards/hydra_head_7.faiss"
]

TRIALS = 100
USE_UNIFIED_MEMORY = False
PINNED_MEM_BYTES = 2 * 1024 * 1024 * 1024
TEMP_MEM_BYTES = 0
REUSE_RESOURCES = True

def get_gpu_resources():
    res = faiss.StandardGpuResources()
    res.setTempMemory(TEMP_MEM_BYTES)
    res.setPinnedMemory(PINNED_MEM_BYTES)
    return res

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
    os.environ["FAISS_GPU_PACKED_LISTS_PROFILE"] = "1"
    os.environ["FAISS_GPU_PACKED_LISTS_DEBUG"] = "0"
    os.environ["FAISS_GPU_PREALLOCATE_MB"] = "61440"

    clear_cache()

    # ----------------------------------------------------------
    # Results storage
    # ----------------------------------------------------------
    results = []
    cpu_indices = []

    for shard_idx, shard_path in enumerate(HYDRA_SHARDS):
        print(f"\n{'='*60}")
        print(f"Disk to RAM for Shard {shard_idx}: {shard_path}")
        print(f"{'='*60}")

        # Measure disk to CPU transfer (once for reference)
        t0 = time.perf_counter()
        cpu_index = faiss.read_index(shard_path)
        t1 = time.perf_counter()
        disk_to_cpu_time = t1 - t0
        cpu_indices.append(cpu_index)

        print(f"Disk -> CPU Load Time: {disk_to_cpu_time:.6f} s")
        print(f"  Index type: {type(cpu_index).__name__}")
        print(f"  Total vectors: {cpu_index.ntotal:,}")
        if hasattr(cpu_index, 'nlist'):
            print(f"  IVF lists: {cpu_index.nlist:,}")

    # Measure CPU to GPU transfer (multiple trials, keep CPU index)
    gpu_to_cpu_times = []
    
    # Create reusable GPU resources that will persist across all trials
    # This allows GPU buffers to be pre-allocated on first load and reused
    persistent_res = get_gpu_resources()
    
    # Pre-allocate GPU buffer to maximum shard size to avoid fragmentation
    # Load the largest shard first (outside the trial loop) to pre-size the buffer
    # This prevents resize-induced fragmentation during trials
    print(f"\n{'='*60}")
    print(f"Pre-allocating GPU buffer to max shard size...")
    print(f"{'='*60}")
    
    max_shard_idx = 0
    max_shard_size = 0
    for shard_idx, shard_path in enumerate(HYDRA_SHARDS):
        shard_size = cpu_indices[shard_idx].ntotal
        if shard_size > max_shard_size:
            max_shard_size = shard_size
            max_shard_idx = shard_idx
    
    # Load largest shard to pre-allocate max buffer, then delete it
    # This primes the GPU memory and prevents fragmentation during trials
    print(f"Loading largest shard (Index {max_shard_idx}) to pre-allocate buffer...")

    # Set cache path for warmup phase
    os.environ["FAISS_GPU_PACKED_CACHE_PATH"] = f"/data/indices/hydra_cache_shards/hydra_shard_{max_shard_idx}"

    co = faiss.GpuClonerOptions()
    co.useUnifiedMemory = USE_UNIFIED_MEMORY
    co.useFloat16 = True
    co.usePrecomputed = False
    co.indicesOptions = faiss.INDICES_32_BIT
    
    buffer_warmup = faiss.index_cpu_to_gpu(persistent_res, 0, cpu_indices[max_shard_idx], co)
    persistent_res.syncDefaultStreamCurrentDevice()
    print(f"GPU buffer warmed up with Shard {max_shard_idx} (~{max_shard_size * 4 / 1e9:.1f}GB estimate)")
    
    # Delete the warmup buffer to free GPU memory before trials
    # The persistent_res remains, with pre-configured GPU memory management
    del buffer_warmup
    clear_gpu_memory()
    print(f"Warmup buffer deleted; persistent GPU resources ready for trials")
    
    # ----------------------------------------------------------
    # Profile each trial (outer) and shard (inner)
    # ----------------------------------------------------------
    for trial in tqdm(range(1, TRIALS + 1), desc="Trials"):
        print(f"\n{'='*60}")
        print(f"Starting Trial RAM -> GPU {trial}")
        print(f"{'='*60}")

        for shard_idx, shard_path in enumerate(HYDRA_SHARDS):
            # Reuse the same GPU resources (pre-warmed)
            # This prevents fragmentation during subsequent loads
            print(f"\nProfiling Shard {shard_idx}: {shard_path}")

            os.environ["FAISS_GPU_PACKED_CACHE_PATH"] = f"/data/indices/hydra_cache_shards/hydra_shard_{shard_idx}"

            co = faiss.GpuClonerOptions()
            co.useUnifiedMemory = USE_UNIFIED_MEMORY
            co.useFloat16 = True
            co.usePrecomputed = False
            co.indicesOptions = faiss.INDICES_32_BIT

            t_start = time.perf_counter()
            gpu_index = faiss.index_cpu_to_gpu(persistent_res, 0, cpu_indices[shard_idx], co)
            persistent_res.syncDefaultStreamCurrentDevice()
            t_end = time.perf_counter()

            cpu_to_gpu_time = t_end - t_start

            print(f"CPU -> GPU Transfer Time: {cpu_to_gpu_time:.6f} s")

            # Get ntotal for this specific shard
            ntotal = cpu_indices[shard_idx].ntotal

            results.append({
                'shard_idx': shard_idx,
                'shard_name': os.path.basename(shard_path),
                'trial': trial,
                'disk_to_cpu_s': disk_to_cpu_time,
                'cpu_to_gpu_s': cpu_to_gpu_time,
                'ntotal': ntotal
            })

            # Delete gpu_index to free GPU memory for next shard
            del gpu_index
            clear_gpu_memory()

    # Cleanup persistent GPU resources after all trials
    del persistent_res
    clear_gpu_memory()

    # ----------------------------------------------------------
    # Save Results
    # ----------------------------------------------------------
    os.makedirs("data", exist_ok=True)
    output_path = "data/parallel_shard_transfers.csv"

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
