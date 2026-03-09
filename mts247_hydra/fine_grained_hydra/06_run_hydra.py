#!/usr/bin/env python3
"""
hydra_benchmark.py  —  RAM-lean version
----------------------------------------
Replaces load_cpu_indices() + index_cpu_to_gpu(cpu_indices[shard_id], ...)
with a stub-based loader that reads GPU codes directly from the packed
disk cache, keeping zero inverted-list data in CPU RAM.

RAM budget (per shard, new vs old):
  OLD:  full CPU index  ~nlist * avg_list_size * code_size  (can be tens of GB total)
  NEW:  coarse quantizer only  ~nlist * d * 4 bytes  (a few hundred MB total)

Prerequisites
-------------
1. Run save_coarse_quantizers.py ONCE to extract centroid files.
2. The packed cache (gpu_codes_all.bin / .meta) must already exist for
   each shard under hydra_cache_shards/hydra_shard_<N>/.
   It is written on the first (warmup) run via the normal slow path,
   so run warmup once with the old script before switching to this one.
"""

import os
import time
import gc
import glob
import re
import torch
import numpy as np
import faiss
import pandas as pd
from tqdm import tqdm

try:
    import pycuda.driver as cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False

from gpu_index_loader import load_ivf_gpu_index_from_cache, _read_meta

# ==============================================================
# Config  (unchanged from original)
# ==============================================================
SHARDS_DIR      = "/data/indices/hydra/fine/shards"
SHARD_GLOB      = "hydra_head_*.faiss"
QUERY_PATH      = "../triviaqa_encodings.npy"
CENTROID_LIST   = "/data/indices/hydra/hydra_centroids.npy"
CENTROID_LOOKUP = "/data/indices/hydra/fine/centroid_to_shard_map.csv"

# Directory produced by save_coarse_quantizers.py
QUANTIZER_DIR   = "/data/indices/hydra/fine/coarse_quantizers"

# Packed cache root — one sub-dir per shard
CACHE_ROOT      = "/data/indices/hydra/fine/hydra_cache_shards"

NUM_QUERIES     = 500
NUM_CENTROIDS   = 40
NUM_DOCS        = 10
WARMUP_RUNS     = 2
USE_UNIFIED_MEMORY = False
PINNED_MEM_BYTES   = 2 * 1024 * 1024 * 1024

# ==============================================================
# Shard metadata  (replaces cpu_indices[i].nlist / .d / .metric_type)
# ==============================================================
class ShardMeta:
    """Lightweight stand-in for the parts of a CPU index we still need."""
    __slots__ = ("nlist", "d", "metric_type", "ntotal", "code_size",
                 "quantizer_path", "cache_dir")

    def __init__(self, shard_idx: int):
        meta_path = os.path.join(
            QUANTIZER_DIR, f"coarse_quantizer_shard_{shard_idx}.meta.txt"
        )
        qpath = os.path.join(
            QUANTIZER_DIR, f"coarse_quantizer_shard_{shard_idx}.faiss"
        )
        if not os.path.exists(meta_path) or not os.path.exists(qpath):
            raise FileNotFoundError(
                f"Missing coarse quantizer for shard {shard_idx}. "
                f"Run save_coarse_quantizers.py first.\n"
                f"  expected: {meta_path}\n  and:      {qpath}"
            )

        kv = {}
        with open(meta_path) as f:
            for line in f:
                k, v = line.strip().split("=", 1)
                kv[k] = v

        self.nlist         = int(kv["nlist"])
        self.d             = int(kv["d"])
        self.metric_type   = int(kv["metric"])
        self.ntotal        = int(kv["ntotal"])
        self.code_size     = int(kv["code_size"])
        self.quantizer_path = qpath
        self.cache_dir     = os.path.join(CACHE_ROOT, f"hydra_shard_{shard_idx}")


# ==============================================================
# Helpers  (identical to original)
# ==============================================================
def discover_shards(shards_dir, shard_glob):
    shard_paths = glob.glob(os.path.join(shards_dir, shard_glob))
    def key(p):
        m = re.search(r"(\d+)", os.path.basename(p))
        return (int(m.group(1)) if m else float("inf"), p)
    shard_paths.sort(key=key)
    return shard_paths


def get_gpu_resources():
    res = faiss.StandardGpuResources()
    res.noTempMemory()
    res.setPinnedMemory(PINNED_MEM_BYTES)
    return res


def cuda_sync(res):
    res.syncDefaultStreamCurrentDevice()


def clear_gpu_memory():
    if CUDA_AVAILABLE:
        try:
            cuda.Context.synchronize()
            cuda.Device(0).synchronize()
        except Exception as e:
            print(f"Warning: Could not fully clear GPU memory: {e}")
    gc.collect()


def get_gpu_cloner_options():
    co = faiss.GpuClonerOptions()
    co.useUnifiedMemory = USE_UNIFIED_MEMORY
    co.useFloat16       = True
    co.usePrecomputed   = False
    co.indicesOptions   = faiss.INDICES_32_BIT
    return co


# ==============================================================
# Lightweight shard loader  (replaces index_cpu_to_gpu(cpu_indices[i]))
# ==============================================================
def load_shard_to_gpu(persistent_res: faiss.GpuResources,
                      meta: ShardMeta,
                      shard_path: str) -> faiss.GpuIndex:
    """
    Load shard from packed disk cache onto GPU via IO_FLAG_MMAP.
    The shard .faiss file is mmap'd — inverted-list pages never enter RAM.
    Only the index header and quantizer centroids are actually read.
    """
    os.environ["FAISS_GPU_PACKED_CACHE_PATH"] = meta.cache_dir

    return load_ivf_gpu_index_from_cache(
        res             = persistent_res,
        gpu_id          = 0,
        shard_path      = shard_path,
        nlist           = meta.nlist,
        cache_dir       = meta.cache_dir,
        cloner_options  = get_gpu_cloner_options(),
    )


# ==============================================================
# Warmup  (structure identical to original, but no cpu_indices arg)
# ==============================================================
def warmup_shards(persistent_res, shard_metas, num_warmup_runs, hydra_shards):
    print("\n" + "="*60)
    print(f"Warmup Phase ({num_warmup_runs} runs)")
    print("="*60)

    largest_shard_idx = max(range(len(shard_metas)),
                            key=lambda i: shard_metas[i].ntotal)
    print(f"\nLargest shard: {largest_shard_idx} "
          f"with {shard_metas[largest_shard_idx].ntotal:,} vectors")
    print("Loading largest shard FIRST to pre-allocate GPU memory pool...\n")

    warmup_times = {i: [] for i in range(len(shard_metas))}

    for run in range(num_warmup_runs):
        print(f"\nWarmup Run {run + 1}/{num_warmup_runs}")
        shard_order = ([largest_shard_idx] +
                       [i for i in range(len(shard_metas))
                        if i != largest_shard_idx])

        for shard_idx in shard_order:
            t_start = time.perf_counter()
            gpu_index = load_shard_to_gpu(persistent_res, shard_metas[shard_idx], hydra_shards[shard_idx])
            persistent_res.syncDefaultStreamCurrentDevice()
            transfer_time = time.perf_counter() - t_start

            warmup_times[shard_idx].append(transfer_time)
            print(f"  Shard {shard_idx}: {transfer_time:.6f}s")

            del gpu_index
            persistent_res.syncDefaultStreamCurrentDevice()
            clear_gpu_memory()

    return warmup_times


# ==============================================================
# Centroid / shard mapping  (unchanged)
# ==============================================================
def get_centroid_to_shard_mapping():
    print("Loading centroid-to-shard mapping...")
    df = pd.read_csv(CENTROID_LOOKUP,
                     dtype={"centroid_id": int, "shard_id": int})
    num_centroids = df["centroid_id"].max() + 1
    centroid_to_shard = torch.full((num_centroids,), -1,
                                   device="cuda", dtype=torch.long)
    centroid_to_shard[
        torch.tensor(df["centroid_id"].values, device="cuda", dtype=torch.long)
    ] = torch.tensor(df["shard_id"].values, device="cuda", dtype=torch.long)
    num_shards = int(torch.tensor(df["shard_id"].values).max().item()) + 1
    return centroid_to_shard, num_shards


def analyze_shard_hits_per_query(query_idx, centroid_ids,
                                 retrieved_shards, num_shards):
    valid = retrieved_shards[query_idx]
    valid = valid[valid >= 0]
    counts = torch.zeros(num_shards, device="cuda", dtype=torch.long)
    counts.scatter_add_(0, valid,
                        torch.ones(len(valid), device="cuda", dtype=torch.long))
    counts_cpu = counts.cpu().numpy()
    hit_shard_ids = np.flatnonzero(counts_cpu > 0).tolist()
    return hit_shard_ids, counts_cpu


# ==============================================================
# Main
# ==============================================================
def main():
    os.environ["FAISS_GPU_PACKED_LISTS"]          = "1"
    os.environ["FAISS_GPU_PACKED_LISTS_MMAP"]     = "1"
    os.environ["FAISS_GPU_DEVICEVECTOR_CACHE"]    = "1"
    os.environ["FAISS_GPU_DEVICEVECTOR_CACHE_MIN_BYTES"] = str(1 << 30)
    os.environ["FAISS_GPU_PACKED_LISTS_PROFILE"]  = "0"
    os.environ["FAISS_GPU_PACKED_LISTS_DEBUG"]    = "0"

    hydra_shards = discover_shards(SHARDS_DIR, SHARD_GLOB)
    if not hydra_shards:
        raise FileNotFoundError(
            f"No shard files found in {SHARDS_DIR} matching {SHARD_GLOB}"
        )
    print(f"Found {len(hydra_shards)} shard files.")

    # ------------------------------------------------------------------
    # Phase 1: Load lightweight shard metadata (replaces load_cpu_indices)
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Loading Shard Metadata  (coarse quantizers only — no inverted lists)")
    print("="*60)

    shard_metas = []
    for shard_idx in range(len(hydra_shards)):
        meta = ShardMeta(shard_idx)
        shard_metas.append(meta)
        print(f"  Shard {shard_idx}: nlist={meta.nlist} d={meta.d} "
              f"ntotal={meta.ntotal:,} code_size={meta.code_size}")

    persistent_res = get_gpu_resources()

    # ------------------------------------------------------------------
    # Phase 2: Warmup (writes packed cache on first run if missing)
    # ------------------------------------------------------------------
    warmup_times = warmup_shards(persistent_res, shard_metas, WARMUP_RUNS, hydra_shards)

    # ------------------------------------------------------------------
    # Phase 3: Centroid analysis
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("Loading Queries & Centroids")
    print("="*60)

    queries       = np.load(QUERY_PATH, mmap_mode='r')
    query_vectors = queries[:NUM_QUERIES].astype('float32')

    centroids = np.load(CENTROID_LIST).astype('float32')
    d = centroids.shape[1]

    centroid_index_cpu = faiss.IndexFlatL2(d)
    centroid_index_gpu = faiss.index_cpu_to_gpu(persistent_res, 0, centroid_index_cpu)
    centroid_index_gpu.add(centroids)

    similarities, centroid_ids = centroid_index_gpu.search(query_vectors, NUM_CENTROIDS)

    centroid_to_shard, num_shards = get_centroid_to_shard_mapping()
    retrieved_ids_gpu = torch.tensor(centroid_ids, device="cuda", dtype=torch.long)
    retrieved_shards  = centroid_to_shard[retrieved_ids_gpu]

    # ------------------------------------------------------------------
    # Phase 4: Per-query retrieval
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print(f"Per-Query Analysis (top-{NUM_CENTROIDS} centroids)")
    print("="*60)

    analysis_results = []

    for q in range(len(query_vectors)):
        hit_shard_ids, shard_counts_cpu = analyze_shard_hits_per_query(
            q, centroid_ids, retrieved_shards, num_shards
        )

        if not hit_shard_ids:
            print("\n  No valid shard hits found; skipping query.")
            continue

        merged_distances = []
        merged_indices   = []
        merged_shards    = []
        total_gpu_transfer_time = 0.0
        total_gpu_search_time   = 0.0

        for shard_id in hit_shard_ids:
            meta = shard_metas[shard_id]

            t_transfer_start = time.perf_counter()
            gpu_index = load_shard_to_gpu(persistent_res, meta, hydra_shards[shard_id])
            persistent_res.syncDefaultStreamCurrentDevice()
            gpu_transfer_time = time.perf_counter() - t_transfer_start
            total_gpu_transfer_time += gpu_transfer_time

            if hasattr(gpu_index, 'nprobe'):
                gpu_index.nprobe = min(2048, meta.nlist)

            t_search_start = time.perf_counter()
            distances, indices = gpu_index.search(query_vectors[q:q+1], NUM_DOCS)
            cuda_sync(persistent_res)
            gpu_search_time = time.perf_counter() - t_search_start
            total_gpu_search_time += gpu_search_time

            merged_distances.append(distances[0])
            merged_indices.append(indices[0])
            merged_shards.append(
                np.full(indices.shape[1], shard_id, dtype=np.int32)
            )

            del gpu_index
            persistent_res.syncDefaultStreamCurrentDevice()
            clear_gpu_memory()

        merged_distances = np.concatenate(merged_distances)
        merged_indices   = np.concatenate(merged_indices)
        merged_shards    = np.concatenate(merged_shards)

        # Sort by metric: inner-product → descending, L2 → ascending
        metric_type = shard_metas[hit_shard_ids[0]].metric_type
        if metric_type in (faiss.METRIC_INNER_PRODUCT, faiss.METRIC_Jaccard):
            top_order = np.argsort(-merged_distances)[:NUM_DOCS]
        else:
            top_order = np.argsort(merged_distances)[:NUM_DOCS]

        final_docs       = merged_indices[top_order]
        final_scores     = merged_distances[top_order]
        final_doc_shards = merged_shards[top_order]

        avg_warmup_time = np.mean([
            np.mean(warmup_times[sid])
            for sid in hit_shard_ids if warmup_times[sid]
        ]) if hit_shard_ids else 0.0

        print(f"\n  Searched Shards: {hit_shard_ids}")
        print(f"  GPU Transfer (total): {total_gpu_transfer_time:.6f}s | "
              f"GPU Search (total): {total_gpu_search_time:.6f}s")
        print(f"  Avg Warmup Time (hit shards): {avg_warmup_time:.6f}s")
        print(f"  Top-{NUM_DOCS} Docs    (merged): {final_docs}")
        print(f"  Top-{NUM_DOCS} Scores  (merged): {final_scores}")
        print(f"  Top-{NUM_DOCS} Shards  (merged): {final_doc_shards}")

        analysis_results.append({
            'query':               q,
            'searched_shard_ids':  str(hit_shard_ids),
            'num_searched_shards': len(hit_shard_ids),
            'gpu_transfer_time':   total_gpu_transfer_time,
            'gpu_search_time':     total_gpu_search_time,
            'warmup_time':         avg_warmup_time,
            'best_retrieved_ids':  str(final_docs.tolist()),
            'top_doc_shards':      str(final_doc_shards.tolist()),
        })

    # ------------------------------------------------------------------
    # Phase 5: Save results
    # ------------------------------------------------------------------
    output_file = "../data/fine_grained_hydra_analysis.csv"
    results_df  = pd.DataFrame(analysis_results)
    results_df.to_csv(output_file, index=False)

    print("\n" + "="*60)
    print(f"Results saved to {output_file}")
    print("="*60)
    print(results_df.to_string(index=False))

    del persistent_res
    clear_gpu_memory()


if __name__ == "__main__":
    os.environ["FAISS_VERBOSE"] = "0"
    main()
