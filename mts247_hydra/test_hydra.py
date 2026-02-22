import faiss
import numpy as np
import os
import torch
import pandas as pd
import time
import shutil
import gc

QUERY_PATH = "triviaqa_encodings.npy"
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
CENTROID_LIST = "/data/indices/shards/hydra_centroids.npy"
CENTROID_LOOKUP = "/data/indices/shards/centroid_to_shard_map.csv"

PINNED_MEM_BYTES = 2 * 1024 * 1024 * 1024  # 2GB
TEMP_MEM_BYTES   = 0
NPROBE = 256
USE_UNIFIED_MEMORY = False


def clear_dev_shm():
    shm_path = "/dev/shm"
    if os.path.exists(shm_path):
        for filename in os.listdir(shm_path):
            file_path = os.path.join(shm_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")


def cuda_sync(res):
    res.syncDefaultStreamCurrentDevice()


def get_gpu_resources():
    res = faiss.StandardGpuResources()
    res.setTempMemory(TEMP_MEM_BYTES)
    res.setPinnedMemory(PINNED_MEM_BYTES)
    print(f"  Temp memory:   {TEMP_MEM_BYTES / (1024**3):.1f} GB")
    print(f"  Pinned memory: {PINNED_MEM_BYTES / (1024**3):.1f} GB")
    return res


def load_shard_to_gpu(cpu_index, res):
    co = faiss.GpuClonerOptions()
    co.useUnifiedMemory = USE_UNIFIED_MEMORY
    co.useFloat16 = True
    co.usePrecomputed = False
    co.indicesOptions = faiss.INDICES_32_BIT

    t0 = time.perf_counter()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
    cuda_sync(res)  # sync immediately after transfer like the working code
    t1 = time.perf_counter()

    gpu_index.nprobe = NPROBE
    return gpu_index, t1 - t0


def main():
    os.environ.setdefault("FAISS_GPU_PACKED_LISTS", "1")
    os.environ.setdefault("FAISS_GPU_PACKED_LISTS_MMAP", "1")
    os.environ.setdefault("FAISS_GPU_DEVICEVECTOR_CACHE", "1")
    os.environ.setdefault("FAISS_GPU_DEVICEVECTOR_CACHE_MIN_BYTES", str(1 << 30))
    os.environ.setdefault("FAISS_GPU_PACKED_LISTS_PROFILE", "1")
    os.environ.setdefault("FAISS_GPU_PACKED_LISTS_DEBUG", "0")
    os.environ.setdefault("FAISS_GPU_PACKED_CACHE_PATH", "/dev/shm/test")
    clear_dev_shm()

    # --- 1. Load Queries ---
    queries = np.load(QUERY_PATH, mmap_mode='r')
    query_vectors = queries[:10].astype('float32')

    # --- 2. Load Shards into CPU Memory ---
    shard_quantizers = []

    print("Loading indices into CPU memory...")
    for i, path in enumerate(HYDRA_SHARDS):
        file_name = os.path.basename(path)
        print(f"[{i+1}/{len(HYDRA_SHARDS)}] Loading index: {file_name}...", end="\r")

        index = faiss.read_index(path)
        ivf_index = faiss.downcast_index(index)

        shard_quantizers.append({
            "name": file_name,
            "quantizer": ivf_index.quantizer,
            "nlist": ivf_index.nlist,
            "index": ivf_index
        })

        print(f"[{i+1}/{len(HYDRA_SHARDS)}] Loaded: {file_name} | "
              f"Type: {type(ivf_index).__name__} | "
              f"Clusters: {ivf_index.nlist} | "
              f"Vectors: {ivf_index.ntotal:,}")

    print(f"{'='*50}\nAll indices loaded into CPU memory.\n")

    # --- 3. Initialise GPU Resources ---
    print("Initialising GPU resources...")
    res = get_gpu_resources()

    # --- 4. Transfer shards to GPU one at a time ---
    print("\nMoving shard indices to GPU...")
    for i, shard in enumerate(shard_quantizers):
        os.environ["FAISS_GPU_PACKED_CACHE_PATH"] = f"/dev/shm/faiss_hydra_shard_{i}"
        os.makedirs(f"/dev/shm/faiss_hydra_shard_{i}", exist_ok=True)

        print(f"  [{i+1}/{len(shard_quantizers)}] Transferring {shard['name']}...")

        gpu_index, transfer_time = load_shard_to_gpu(shard["index"], res)

        print(f"    ✓ {shard['name']} | Transfer: {transfer_time:.4f}s | Vectors: {gpu_index.ntotal:,}")

        del gpu_index
        cuda_sync(res)
        gc.collect()

    print(f"{'='*50}\nAll shards transferred.\n")

    # --- 5. Load and Normalise Centroids ---
    print("Loading global centroids...")
    centroids = np.load(CENTROID_LIST).astype('float32')

    faiss.normalize_L2(query_vectors)
    faiss.normalize_L2(centroids)

    # --- 6. Build Centroid Index on GPU ---
    print("Moving centroid index to GPU...")
    d = centroids.shape[1]
    centroid_index_cpu = faiss.IndexFlatIP(d)
    centroid_index_gpu = faiss.index_cpu_to_gpu(res, 0, centroid_index_cpu)
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


if __name__ == "__main__":
    os.environ["FAISS_VERBOSE"] = "1"
    main()