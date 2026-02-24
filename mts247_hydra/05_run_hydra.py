#!/usr/bin/env python3
import os
import time
import csv
import gc
import torch
import shutil
import numpy as np
import faiss
import pandas as pd
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
    "/data/indices/shards/hydra_heda_4.faiss"
]
QUERY_PATH = "triviaqa_encodings.npy"
CENTROID_LIST = "/data/indices/shards/hydra_centroids.npy"
CENTROID_LOOKUP = "/data/indices/shards/centroid_to_shard_map.csv"

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

    # clear_cache()

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
    
    # Warm up GPU Searches
    for shard_idx, shard_path in enumerate(HYDRA_SHARDS):
        # Reuse the same GPU resources (pre-warmed)
        # This prevents fragmentation during subsequent loads
        print(f"\nProfiling Shard {shard_idx}: {shard_path}")

        os.environ["FAISS_GPU_PACKED_CACHE_PATH"] = (
            f"/data/indices/hydra_cache_shards/hydra_shard_{shard_idx}"
        )

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

        # Delete gpu_index to free GPU memory for next shard
        del gpu_index
        # Aggressively clear GPU memory
        persistent_res.syncDefaultStreamCurrentDevice()
        clear_gpu_memory()

    # Start HYDRA

    # --- 1. Load Queries ---
    queries = np.load(QUERY_PATH, mmap_mode='r')
    query_vectors = queries[:10].astype('float32')

    # --- 5. Load and Normalise Centroids ---
    print("Loading global centroids...")
    centroids = np.load(CENTROID_LIST).astype('float32')

    faiss.normalize_L2(query_vectors)
    faiss.normalize_L2(centroids)

    # --- 6. Build Centroid Index on GPU ---
    print("Moving centroid index to GPU...")
    d = centroids.shape[1]
    centroid_index_cpu = faiss.IndexFlatIP(d)
    centroid_index_gpu = faiss.index_cpu_to_gpu(persistent_res, 0, centroid_index_cpu)
    centroid_index_gpu.add(centroids)

    # --- 7. Search Top-100 Centroids per Query ---
    print("Computing cosine similarity for top 100 centroids on GPU...")
    k_centroids = 100
    similarities, centroid_ids = centroid_index_gpu.search(query_vectors, k_centroids)

    # --- 8. Map Centroid IDs → Shards (GPU) ---
    print("Loading centroid-to-shard map onto GPU...")
    df = pd.read_csv(CENTROID_LOOKUP, dtype={"centroid_id": int, "shard_id": int})

    num_centroids = df["centroid_id"].max() + 1
    centroid_to_shard = torch.full((num_centroids,), -1, device="cuda", dtype=torch.long)
    centroid_to_shard[
        torch.tensor(df["centroid_id"].values, device="cuda", dtype=torch.long)
    ] = torch.tensor(df["shard_id"].values, device="cuda", dtype=torch.long)

    retrieved_ids_gpu = torch.tensor(centroid_ids, device="cuda", dtype=torch.long)
    retrieved_shards  = centroid_to_shard[retrieved_ids_gpu]

    num_shards = int(torch.tensor(df["shard_id"].values).max().item()) + 1

    # --- 9. Print Per-Query Shard Hit Counts ---
    print(f"\n{'='*50}")
    print(f"Shard hit counts per query (top-{k_centroids} centroids):")

    for q in range(len(query_vectors)):
        shard_counts = torch.zeros(num_shards, device="cuda", dtype=torch.long)
        shard_counts.scatter_add_(
            0,
            retrieved_shards[q],
            torch.ones(k_centroids, device="cuda", dtype=torch.long)
        )
        shard_counts_cpu = shard_counts.cpu().numpy()

        print(f"\n  Query {q} | Top 5 Centroid IDs: {centroid_ids[q][:5]}")
        print(f"  {'Shard ID':<12} {'Centroid Hits':<16} {'Shard File'}")
        print(f"  {'-'*45}")
        for shard_id, count in enumerate(shard_counts_cpu):
            shard_name = os.path.basename(HYDRA_SHARDS[shard_id])
            print(f"  {shard_id:<12} {count:<16} {shard_name}")

    print(f"\n{'='*50}")

    # Cleanup persistent GPU resources after all trials
    del persistent_res
    clear_gpu_memory()


if __name__ == "__main__":
    os.environ["FAISS_VERBOSE"] = "0"
    main()
