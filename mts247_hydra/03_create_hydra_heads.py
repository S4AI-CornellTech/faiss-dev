#!/usr/bin/env python3
import faiss
import numpy as np
import os
import csv

# ==============================================================
# CONFIG
# ==============================================================
INDEX_PATH = "/data/indices/hydra_ivf_500m_sq8.faiss"
OUTPUT_DIR = "/data/indices/shards/"
MAPPING_FILE = "centroid_to_shard_map.csv"
INDEX_SIZE = 3096  # Number of original clusters per shard
# ==============================================================

def extract_ivf_index(index):
    """Unwrap FAISS wrappers to reach the core IndexIVF."""
    while hasattr(index, 'index'):
        index = index.index
    if not isinstance(index, faiss.IndexIVF):
        raise ValueError(f"Expected IndexIVF, got {type(index)}")
    return index

def main():
    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Index file not found: {INDEX_PATH}")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Loading large index from {INDEX_PATH}...")
    index = faiss.read_index(INDEX_PATH, faiss.IO_FLAG_MMAP)
    ivf_index = extract_ivf_index(index)
    
    nlist = ivf_index.nlist
    d = ivf_index.d
    num_shards = int(np.ceil(nlist / INDEX_SIZE))
    
    print(f"Total nlist: {nlist} | Target Shards: {num_shards}")

    # 1. Extract and Cluster Centroids
    print("Extracting and grouping centroids...")
    centroids = ivf_index.quantizer.reconstruct_n(0, nlist)
    
    kmeans = faiss.Kmeans(d, num_shards, niter=20, verbose=True)
    kmeans.train(centroids)
    _, shard_assignments = kmeans.index.search(centroids, 1)
    shard_assignments = shard_assignments.flatten()

    # --- NEW: SAVE MAPPING TO FILE ---
    mapping_path = os.path.join(OUTPUT_DIR, MAPPING_FILE)
    print(f"Saving centroid-to-shard mapping to {mapping_path}...")
    with open(mapping_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["centroid_id", "shard_id"])
        # shard_assignments index matches the centroid_id
        for c_id, s_id in enumerate(shard_assignments):
            writer.writerow([c_id, s_id])
    # ---------------------------------

    # 2. Process Shards
    for s_id in range(num_shards):
        print(f"Creating shard {s_id}...")
        lists_in_shard = np.where(shard_assignments == s_id)[0].astype('int64')
        shard_nlist = len(lists_in_shard)
        
        quantizer = faiss.IndexFlatL2(d)
        shard_centroids = centroids[lists_in_shard]
        quantizer.add(shard_centroids) 

        shard_index = faiss.IndexIVFScalarQuantizer(
            quantizer, d, shard_nlist, faiss.ScalarQuantizer.QT_8bit
        )
        
        shard_index.is_trained = True
        shard_index.ntotal = 0
        invlists = ivf_index.invlists
        
        for new_list_idx, old_list_idx in enumerate(lists_in_shard):
            list_size = invlists.list_size(int(old_list_idx))
            if list_size == 0:
                continue
            
            ids_ptr = invlists.get_ids(int(old_list_idx))
            codes_ptr = invlists.get_codes(int(old_list_idx))
            
            shard_index.invlists.add_entries(
                int(new_list_idx), 
                list_size, 
                ids_ptr, 
                codes_ptr
            )
            shard_index.ntotal += list_size

        shard_name = f"hydra_head_{s_id}.faiss"
        output_file = os.path.join(OUTPUT_DIR, shard_name)
        faiss.write_index(shard_index, output_file)
        print(f"Successfully saved {shard_name} | {shard_nlist} clusters | {shard_index.ntotal} vectors")

    print(f"\nAll shards and mapping file created successfully in {OUTPUT_DIR}.")

if __name__ == "__main__":
    main()