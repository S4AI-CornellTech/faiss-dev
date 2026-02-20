import faiss
import numpy as np
import os
import torch
import pandas as pd

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

def main():
    # --- 1. Load First 10 Queries ---
    queries = np.load(QUERY_PATH, mmap_mode='r')
    first_10_queries = queries[:10].astype('float32')

    # --- 2. Load Shards and Extract Quantizers ---
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
            "index": ivf_index  # keep the full index for later search
        })

        print(f"[{i+1}/{len(HYDRA_SHARDS)}] Loaded: {file_name} | Clusters (nlist): {ivf_index.nlist}")

    print(f"{'='*50}\nAll indices loaded into memory.\n")

    # --- 3. Load Queries and Centroids ---
    queries = np.load(QUERY_PATH, mmap_mode='r')
    query_vectors = queries[:10].astype('float32')

    print("Loading global centroids...")
    centroids = np.load(CENTROID_LIST).astype('float32')

    # L2 Normalize for Cosine Similarity
    faiss.normalize_L2(query_vectors)
    faiss.normalize_L2(centroids)

    # --- 4. Move Centroids + Queries to GPU for Search ---
    print("Moving centroids to GPU...")
    res = faiss.StandardGpuResources()
    d = centroids.shape[1]

    # Build flat IP index on GPU directly
    centroid_index_cpu = faiss.IndexFlatIP(d)
    centroid_index_gpu = faiss.index_cpu_to_gpu(res, 0, centroid_index_cpu)
    centroid_index_gpu.add(centroids)

    # --- 5. Identify Best 100 Centroids per Query (on GPU) ---
    print("Computing Cosine Similarity for top 100 centroids on GPU...")
    k_centroids = 256
    similarities, centroid_ids = centroid_index_gpu.search(query_vectors, k_centroids)
    # centroid_ids shape: [num_queries, k_centroids]

    # --- 6. Map Retrieved Centroid IDs → Shards and Count ---
    print("Loading centroid-to-shard map onto GPU...")
    df = pd.read_csv(CENTROID_LOOKUP, dtype={"centroid_id": int, "shard_id": int})

    # Build a lookup tensor: index = centroid_id, value = shard_id
    num_centroids = df["centroid_id"].max() + 1
    centroid_to_shard = torch.full((num_centroids,), -1, device="cuda", dtype=torch.long)
    centroid_to_shard[
        torch.tensor(df["centroid_id"].values, device="cuda", dtype=torch.long)
    ] = torch.tensor(df["shard_id"].values, device="cuda", dtype=torch.long)

    # Retrieve shard for every centroid hit, per query
    retrieved_ids_gpu = torch.tensor(centroid_ids, device="cuda", dtype=torch.long)  # [num_queries, k]
    retrieved_shards  = centroid_to_shard[retrieved_ids_gpu]                          # [num_queries, k]

    num_shards = int(torch.tensor(df["shard_id"].values).max().item()) + 1

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
    main()