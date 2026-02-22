#!/usr/bin/env python3
import os
import gc
import faiss
from tqdm import tqdm
import shutil

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

OUTPUT_DIR = "/data/indices/hydra_cache_shards"
PINNED_MEM_BYTES = 2 * 1024 * 1024 * 1024

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
    res.setPinnedMemory(PINNED_MEM_BYTES)
    return res


def main():
    # Enable optimized FAISS GPU paths for shard caching
    os.environ["FAISS_GPU_PACKED_LISTS"] = "1"
    os.environ["FAISS_GPU_PACKED_LISTS_MMAP"] = "1"
    os.environ["FAISS_GPU_DEVICEVECTOR_CACHE"] = "1"

    res = get_gpu_resources()

    clear_cache()

    for i, shard_path in enumerate(tqdm(HYDRA_SHARDS, desc="Saving shard contents via CPU->GPU transfer")):
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
