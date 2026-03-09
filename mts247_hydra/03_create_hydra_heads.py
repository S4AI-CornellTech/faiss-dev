#!/usr/bin/env python3
import faiss
import numpy as np
import os
import csv
import argparse

# ==============================================================
# CONFIG
# ==============================================================
INDEX_PATH = "/data/indices/hydra/hydra_sphere_ivf_600m_sq8.faiss"
OUTPUT_DIR = "/data/indices/hydra/shards/"
MAPPING_FILE = "/data/indices/hydra/centroid_to_shard_map.csv"
# ==============================================================

def extract_ivf_index(index):
    """Unwrap FAISS wrappers to reach the core IndexIVF."""
    while hasattr(index, 'index'):
        index = index.index
    if not isinstance(index, faiss.IndexIVF):
        raise ValueError(f"Expected IndexIVF, got {type(index)}")
    return index

def main():
    parser = argparse.ArgumentParser(description="Shard a FAISS IVF index into smaller sub-indices.")
    parser.add_argument(
        "--index-size",
        type=int,
        default=100,
        help="Number of original clusters per shard (default: 1000)"
    )
    args = parser.parse_args()

    index_size = args.index_size

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"Index file not found: {INDEX_PATH}")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Loading index from {INDEX_PATH}...")
    index = faiss.read_index(INDEX_PATH, faiss.IO_FLAG_MMAP)
    ivf_index = extract_ivf_index(index)

    nlist = ivf_index.nlist
    d = ivf_index.d
    
    metric_type = ivf_index.metric_type
    metric_str = "IP" if metric_type == faiss.METRIC_INNER_PRODUCT else "L2"
    
    num_shards = int(np.ceil(nlist / index_size))

    print(f"Total nlist: {nlist} | d: {d} | Metric: {metric_str} | Index size: {index_size} | Target Shards: {num_shards}")

    print("Extracting and clustering centroids...")
    centroids = ivf_index.quantizer.reconstruct_n(0, nlist)

    kmeans = faiss.Kmeans(d, num_shards, niter=20, verbose=True)
    kmeans.train(centroids)
    _, shard_assignments = kmeans.index.search(centroids, 1)
    shard_assignments = shard_assignments.flatten()

    mapping_path = os.path.join(OUTPUT_DIR, MAPPING_FILE)
    print(f"Saving centroid-to-shard mapping to {mapping_path}...")
    with open(mapping_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["centroid_id", "shard_id"])
        for c_id, s_id in enumerate(shard_assignments):
            writer.writerow([c_id, s_id])

    invlists = ivf_index.invlists
    
    for s_id in range(num_shards):
        print(f"\n{'='*60}")
        print(f"Processing shard {s_id}/{num_shards}...")
        print(f"{'='*60}")
        
        lists_in_shard = np.where(shard_assignments == s_id)[0]
        shard_nlist = len(lists_in_shard)
        print(f"Shard contains {shard_nlist} nlists: {lists_in_shard[:5]}..." if len(lists_in_shard) > 5 else f"Shard contains {shard_nlist} nlists: {lists_in_shard}")

        all_vectors = []
        all_ids = []
        total_vectors = 0
        
        for idx, list_idx in enumerate(lists_in_shard):
            list_size = invlists.list_size(int(list_idx))
            if list_size == 0:
                continue
            
            ids_ptr = invlists.get_ids(int(list_idx))
            codes_ptr = invlists.get_codes(int(list_idx))
            
            try:
                ids = faiss.rev_swig_ptr(ids_ptr, list_size).copy()
                all_ids.append(ids)
                
                vectors = np.zeros((list_size, d), dtype=np.float32)
                for i in range(list_size):
                    ivf_index.reconstruct_from_offset(int(list_idx), i, faiss.swig_ptr(vectors[i]))
                
                all_vectors.append(vectors)
                total_vectors += list_size
                
            finally:
                invlists.release_ids(int(list_idx), ids_ptr)
                invlists.release_codes(int(list_idx), codes_ptr)
            
            if (idx + 1) % 100 == 0:
                print(f"  Processed {idx + 1}/{shard_nlist} nlists ({total_vectors} vectors so far)...")
        
        if len(all_vectors) == 0:
            print(f"  Warning: No vectors found in shard {s_id}, skipping...")
            continue
        
        print(f"  Concatenating {len(all_vectors)} batches...")
        shard_vectors = np.vstack(all_vectors)
        shard_ids = np.concatenate(all_ids)
        
        print(f"  Extracted {len(shard_vectors)} vectors with original IDs preserved")
        
        nlist_new = int(np.sqrt(len(shard_vectors)))
        print(f"  Creating IndexIVFScalarQuantizer with {nlist_new} clusters (sqrt of {len(shard_vectors)}) using {metric_str} metric...")
        
        if metric_type == faiss.METRIC_INNER_PRODUCT:
            quantizer = faiss.IndexFlatIP(d)
        else:
            quantizer = faiss.IndexFlatL2(d)
            
        shard_index = faiss.IndexIVFScalarQuantizer(
            quantizer, d, nlist_new, faiss.ScalarQuantizer.QT_8bit, metric_type
        )
        
        print(f"  Training SQ8 index...")
        shard_index.train(shard_vectors)
        
        shard_index.nprobe = min(nlist_new, 10)
        
        print(f"  Adding {len(shard_vectors)} vectors with original IDs...")
        shard_index.add_with_ids(shard_vectors, shard_ids)
        
        shard_name = f"hydra_head_{s_id}.faiss"
        output_file = os.path.join(OUTPUT_DIR, shard_name)
        faiss.write_index(shard_index, output_file)
        print(f"  ✓ Saved {shard_name} | {shard_index.ntotal} vectors | {nlist_new} clusters | nprobe={shard_index.nprobe}")

    print(f"\n{'='*60}")
    print(f"All {num_shards} shards created successfully in {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()