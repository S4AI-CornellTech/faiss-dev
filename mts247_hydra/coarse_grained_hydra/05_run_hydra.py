#!/usr/bin/env python3
import os
import time
import csv
import gc
import glob
import re
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
SHARDS_DIR = "/data/indices/hydra/shards"
SHARD_GLOB = "hydra_head_*.faiss"
QUERY_PATH = "triviaqa_encodings.npy"
CENTROID_LIST = "/data/indices/hydra/hydra_centroids.npy"
CENTROID_LOOKUP = "/data/indices/hydra/centroid_to_shard_map.csv"

NUM_QUERIES=5

NUM_DOCS = 5
WARMUP_RUNS = 3
TRIALS = 100
USE_UNIFIED_MEMORY = False
PINNED_MEM_BYTES = 2 * 1024 * 1024 * 1024
TEMP_MEM_BYTES = 0
REUSE_RESOURCES = True


def discover_shards(shards_dir, shard_glob):
    shard_paths = glob.glob(os.path.join(shards_dir, shard_glob))

    def shard_sort_key(path):
        filename = os.path.basename(path)
        match = re.search(r"(\d+)", filename)
        shard_id = int(match.group(1)) if match else float("inf")
        return (shard_id, filename)

    shard_paths.sort(key=shard_sort_key)
    return shard_paths

def get_gpu_resources():
    res = faiss.StandardGpuResources()
    res.noTempMemory()  # Allocate permanent memory, not temporary
    res.setPinnedMemory(PINNED_MEM_BYTES)
    return res

def clear_cache():
    cache_path = "/data/indices/hydra/hydra_cache_shards"
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

def get_gpu_cloner_options():
    """Get GPU cloner options with configured parameters."""
    co = faiss.GpuClonerOptions()
    co.useUnifiedMemory = USE_UNIFIED_MEMORY
    co.useFloat16 = True
    co.usePrecomputed = False
    co.indicesOptions = faiss.INDICES_32_BIT
    return co

def load_cpu_indices(hydra_shards):
    """Load all shard indices from disk to CPU."""
    print("\n" + "="*60)
    print("Loading Shard Indices from Disk")
    print("="*60)
    
    cpu_indices = []
    for shard_idx, shard_path in enumerate(hydra_shards):
        print(f"\nShard {shard_idx}: {shard_path}")
        t0 = time.perf_counter()
        cpu_index = faiss.read_index(shard_path)
        t1 = time.perf_counter()
        
        cpu_indices.append(cpu_index)
        print(f"  Disk → CPU: {t1-t0:.6f}s | Vectors: {cpu_index.ntotal:,}")
        if hasattr(cpu_index, 'nlist'):
            print(f"  IVF Lists: {cpu_index.nlist:,}")
    
    return cpu_indices

def warmup_shards(persistent_res, cpu_indices, num_warmup_runs, hydra_shards):
    """Warmup GPU with repeated shard loads and searches."""
    print("\n" + "="*60)
    print(f"Warmup Phase ({num_warmup_runs} runs)")
    print("="*60)
    
    # Find the largest shard to pre-allocate max GPU memory
    largest_shard_idx = max(range(len(cpu_indices)), key=lambda i: cpu_indices[i].ntotal)
    print(f"\nLargest shard: {largest_shard_idx} with {cpu_indices[largest_shard_idx].ntotal:,} vectors")
    print("Loading largest shard FIRST to pre-allocate GPU memory pool...\n")
    
    warmup_times = {i: [] for i in range(len(hydra_shards))}
    
    for run in range(num_warmup_runs):
        print(f"\nWarmup Run {run + 1}/{num_warmup_runs}")
        
        # Create order: largest first, then all others in sequence
        shard_order = [largest_shard_idx] + [i for i in range(len(hydra_shards)) if i != largest_shard_idx]
        
        for shard_idx in shard_order:
            os.environ["FAISS_GPU_PACKED_CACHE_PATH"] = (
                f"/data/indices/hydra/hydra_cache_shards/hydra_shard_{shard_idx}"
            )
            
            t_start = time.perf_counter()
            gpu_index = faiss.index_cpu_to_gpu(persistent_res, 0, cpu_indices[shard_idx], get_gpu_cloner_options())
            persistent_res.syncDefaultStreamCurrentDevice()
            transfer_time = time.perf_counter() - t_start
            
            warmup_times[shard_idx].append(transfer_time)
            print(f"  Shard {shard_idx}: {transfer_time:.6f}s")
            
            del gpu_index
            persistent_res.syncDefaultStreamCurrentDevice()
            clear_gpu_memory()
    
    return warmup_times

def get_centroid_to_shard_mapping():
    """Load and build centroid to shard mapping on GPU."""
    print("Loading centroid-to-shard mapping...")
    df = pd.read_csv(CENTROID_LOOKUP, dtype={"centroid_id": int, "shard_id": int})
    
    num_centroids = df["centroid_id"].max() + 1
    centroid_to_shard = torch.full((num_centroids,), -1, device="cuda", dtype=torch.long)
    centroid_to_shard[
        torch.tensor(df["centroid_id"].values, device="cuda", dtype=torch.long)
    ] = torch.tensor(df["shard_id"].values, device="cuda", dtype=torch.long)
    
    num_shards = int(torch.tensor(df["shard_id"].values).max().item()) + 1
    return centroid_to_shard, num_shards

def analyze_shard_hits_per_query(query_idx, k_centroids, centroid_ids, retrieved_shards, num_shards):
    """Compute shard hit counts for a single query."""
    shard_counts = torch.zeros(num_shards, device="cuda", dtype=torch.long)
    shard_counts.scatter_add_(
        0,
        retrieved_shards[query_idx],
        torch.ones(k_centroids, device="cuda", dtype=torch.long)
    )
    shard_counts_cpu = shard_counts.cpu().numpy()
    
    max_shard_id = int(np.argmax(shard_counts_cpu))
    max_hits = shard_counts_cpu[max_shard_id]
    
    return max_shard_id, shard_counts_cpu

def print_shard_analysis(query_idx, k_centroids, centroid_ids, max_shard_id, max_hits, shard_counts_cpu, hydra_shards):
    """Print shard analysis results for a query."""
    print(f"\n  Query {query_idx} | Top 5 Centroid IDs: {centroid_ids[query_idx][:5]}")
    print(f"  {'Shard ID':<12} {'Centroid Hits':<16} {'Shard File'}")
    print(f"  {'-'*45}")
    for shard_id, count in enumerate(shard_counts_cpu):
        shard_name = os.path.basename(hydra_shards[shard_id])
        print(f"  {shard_id:<12} {count:<16} {shard_name}")
    print(f"  → Selected Shard: {max_shard_id} with {max_hits} hits")

def main():
    """Main HYDRA analysis pipeline."""
    # Enable optimized FAISS GPU paths
    os.environ["FAISS_GPU_PACKED_LISTS"] = "1"
    os.environ["FAISS_GPU_PACKED_LISTS_MMAP"] = "1"
    os.environ["FAISS_GPU_DEVICEVECTOR_CACHE"] = "1"
    os.environ["FAISS_GPU_DEVICEVECTOR_CACHE_MIN_BYTES"] = str(1 << 30)
    os.environ["FAISS_GPU_PACKED_LISTS_PROFILE"] = "1"
    os.environ["FAISS_GPU_PACKED_LISTS_DEBUG"] = "0"

    hydra_shards = discover_shards(SHARDS_DIR, SHARD_GLOB)
    if not hydra_shards:
        raise FileNotFoundError(
            f"No shard files found in {SHARDS_DIR} matching pattern {SHARD_GLOB}"
        )

    print(f"Found {len(hydra_shards)} shard files in {SHARDS_DIR} matching {SHARD_GLOB}")

    # ==============================================================
    # Phase 1: Initialize & Load Indices
    # ==============================================================
    cpu_indices = load_cpu_indices(hydra_shards)
    persistent_res = get_gpu_resources()
    
    warmup_times = warmup_shards(persistent_res, cpu_indices, WARMUP_RUNS, hydra_shards)
    
    # ==============================================================
    # Phase 2: Centroid Analysis
    # ==============================================================
    print("\n" + "="*60)
    print("Loading Queries & Centroids")
    print("="*60)
    
    queries = np.load(QUERY_PATH, mmap_mode='r')
    query_vectors = queries[:NUM_QUERIES].astype('float32')
    
    centroids = np.load(CENTROID_LIST).astype('float32')
    # DO NOT normalize - indices were built with unnormalized data and L2 metric
    # faiss.normalize_L2(query_vectors)
    # faiss.normalize_L2(centroids)
    
    # Build centroid index on GPU - use L2 to match shard metric
    d = centroids.shape[1]
    centroid_index_cpu = faiss.IndexFlatL2(d)
    centroid_index_gpu = faiss.index_cpu_to_gpu(persistent_res, 0, centroid_index_cpu)
    centroid_index_gpu.add(centroids)
    
    # Search top centroids
    k_centroids = 25
    similarities, centroid_ids = centroid_index_gpu.search(query_vectors, k_centroids)
    
    centroid_to_shard, num_shards = get_centroid_to_shard_mapping()
    retrieved_ids_gpu = torch.tensor(centroid_ids, device="cuda", dtype=torch.long)
    retrieved_shards = centroid_to_shard[retrieved_ids_gpu]
    
    # ==============================================================
    # Phase 3: Per-Query Shard Analysis & Retrieval
    # ==============================================================
    print("\n" + "="*60)
    print(f"Per-Query Analysis (top-{k_centroids} centroids)")
    print("="*60)
    
    analysis_results = []
    
    for q in range(len(query_vectors)):
        # Analyze shard hits
        max_shard_id, shard_counts_cpu = analyze_shard_hits_per_query(
            q, k_centroids, centroid_ids, retrieved_shards, num_shards
        )
        max_hits = shard_counts_cpu[max_shard_id]
        # print_shard_analysis(q, k_centroids, centroid_ids, max_shard_id, max_hits, shard_counts_cpu, hydra_shards)
        
        # Load selected shard and retrieve documents
        os.environ["FAISS_GPU_PACKED_CACHE_PATH"] = (
            f"/data/indices/hydra/hydra_cache_shards/hydra_shard_{max_shard_id}"
        )
        
        t_transfer_start = time.perf_counter()
        gpu_index = faiss.index_cpu_to_gpu(persistent_res, 0, cpu_indices[max_shard_id], get_gpu_cloner_options())
        persistent_res.syncDefaultStreamCurrentDevice()
        gpu_transfer_time = time.perf_counter() - t_transfer_start
        
        t_search_start = time.perf_counter()
        distances, indices = gpu_index.search(query_vectors[q:q+1], NUM_DOCS)
        cuda_sync(persistent_res)
        gpu_search_time = time.perf_counter() - t_search_start
        
        # Compute average warmup time for this shard
        avg_warmup_time = np.mean(warmup_times[max_shard_id]) if warmup_times[max_shard_id] else 0.0
        
        print(f"\n  GPU Transfer: {gpu_transfer_time:.6f}s | GPU Search: {gpu_search_time:.6f}s")
        print(f"  Avg Warmup Time: {avg_warmup_time:.6f}s")
        print(f"  Top-{NUM_DOCS} Docs: {indices[0]}")
        
        # Record results
        analysis_results.append({
            'query': q,
            'shard_id': max_shard_id,
            'gpu_transfer_time': gpu_transfer_time,
            'gpu_search_time': gpu_search_time,
            'warmup_time': avg_warmup_time,
            'best_retrieved_ids': str(indices[0].tolist())
        })
        
        del gpu_index
        persistent_res.syncDefaultStreamCurrentDevice()
        clear_gpu_memory()
    
    # ==============================================================
    # Phase 4: Save Results to CSV
    # ==============================================================
    output_file = "../data/hydra_analysis.csv"
    results_df = pd.DataFrame(analysis_results)
    results_df.to_csv(output_file, index=False)
    
    print("\n" + "="*60)
    print(f"Results saved to {output_file}")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # Cleanup
    del persistent_res
    clear_gpu_memory()


if __name__ == "__main__":
    os.environ["FAISS_VERBOSE"] = "0"
    main()
