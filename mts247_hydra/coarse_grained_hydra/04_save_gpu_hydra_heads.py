#!/usr/bin/env python3
import os
import gc
import faiss
from tqdm import tqdm
import shutil
import glob
import re

# ==============================================================
# Config
# ==============================================================
SHARDS_DIR = "/data/indices/hydra/coarse/shards"
SHARD_GLOB = "hydra_head_*.faiss"

OUTPUT_DIR = "/data/indices/hydra/coarse/hydra_cache_shards"
PINNED_MEM_BYTES = 2 * 1024 * 1024 * 1024


def discover_shards(shards_dir, shard_glob):
    shard_paths = glob.glob(os.path.join(shards_dir, shard_glob))

    def shard_sort_key(path):
        filename = os.path.basename(path)
        match = re.search(r"(\d+)", filename)
        shard_id = int(match.group(1)) if match else float("inf")
        return (shard_id, filename)

    shard_paths.sort(key=shard_sort_key)
    return shard_paths

def clear_cache():
    cache_path = "/data/indices/hydra/coarse/hydra_cache_shards"
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
    res.setPinnedMemory(PINNED_MEM_BYTES)
    return res


def main():
    # Enable optimized FAISS GPU paths for shard caching
    os.environ["FAISS_GPU_PACKED_LISTS"] = "1"
    os.environ["FAISS_GPU_PACKED_LISTS_MMAP"] = "1"
    os.environ["FAISS_GPU_DEVICEVECTOR_CACHE"] = "1"

    hydra_shards = discover_shards(SHARDS_DIR, SHARD_GLOB)
    if not hydra_shards:
        raise FileNotFoundError(
            f"No shard files found in {SHARDS_DIR} matching pattern {SHARD_GLOB}"
        )

    print(f"Found {len(hydra_shards)} shard files in {SHARDS_DIR} matching {SHARD_GLOB}")

    res = get_gpu_resources()

    clear_cache()

    for i, shard_path in enumerate(tqdm(hydra_shards, desc="Saving shard contents via CPU->GPU transfer")):
        # Set cache path for this shard
        os.environ["FAISS_GPU_PACKED_CACHE_PATH"] = f"{OUTPUT_DIR}/hydra_shard_{i}"
        
        # Load shard from disk
        cpu_index = faiss.read_index(shard_path)
        
        # CPU to GPU transfer automatically saves shard contents
        co = faiss.GpuClonerOptions()
        co.useFloat16 = True
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
        res.syncDefaultStreamCurrentDevice()
        
        print(f"Saved Hydra Shard {i} contents via CPU->GPU transfer")
        print(f"  Index type: {type(cpu_index).__name__}")
        print(f"  Total vectors: {cpu_index.ntotal:,}")
        
        # Cleanup
        del gpu_index
        del cpu_index
        gc.collect()

    del res
    gc.collect()


if __name__ == "__main__":
    main()
