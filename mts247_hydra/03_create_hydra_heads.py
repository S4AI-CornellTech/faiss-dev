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

def copy_sq_params(src_ivf, dst_ivf):
    """Copy trained ScalarQuantizer parameters from source to destination."""
    src_trained = faiss.vector_to_array(src_ivf.sq.trained)
    faiss.copy_array_to_vector(src_trained, dst_ivf.sq.trained)
    dst_ivf.sq.code_size = src_ivf.sq.code_size

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
    src_code_size = ivf_index.invlists.code_size
    src_nprobe = ivf_index.nprobe
    num_shards = int(np.ceil(nlist / INDEX_SIZE))

    print(f"Total nlist: {nlist} | d: {d} | code_size: {src_code_size} | Target Shards: {num_shards}")

    # 1. Extract and cluster centroids
    print("Extracting and grouping centroids...")
    centroids = ivf_index.quantizer.reconstruct_n(0, nlist)

    kmeans = faiss.Kmeans(d, num_shards, niter=20, verbose=True)
    kmeans.train(centroids)
    _, shard_assignments = kmeans.index.search(centroids, 1)
    shard_assignments = shard_assignments.flatten()

    # Save mapping to file
    mapping_path = os.path.join(OUTPUT_DIR, MAPPING_FILE)
    print(f"Saving centroid-to-shard mapping to {mapping_path}...")
    with open(mapping_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["centroid_id", "shard_id"])
        for c_id, s_id in enumerate(shard_assignments):
            writer.writerow([c_id, s_id])

    # 2. Process shards
    for s_id in range(num_shards):
        print(f"\nCreating shard {s_id}...")
        lists_in_shard = np.where(shard_assignments == s_id)[0].astype('int64')
        shard_nlist = len(lists_in_shard)

        # Build quantizer seeded with the shard's centroids
        quantizer = faiss.IndexFlatL2(d)
        shard_centroids = centroids[lists_in_shard]
        quantizer.add(shard_centroids)

        shard_index = faiss.IndexIVFScalarQuantizer(
            quantizer, d, shard_nlist, faiss.ScalarQuantizer.QT_8bit
        )

        # --- FIX 1: Copy trained SQ parameters from the source index ---
        copy_sq_params(ivf_index, shard_index)
        shard_index.is_trained = True

        # --- FIX 2: Validate code size matches before copying raw bytes ---
        assert shard_index.invlists.code_size == src_code_size, (
            f"Code size mismatch: src={src_code_size}, shard={shard_index.invlists.code_size}"
        )

        # --- FIX 3: Set nprobe so the shard is actually searchable ---
        shard_index.nprobe = min(shard_nlist, src_nprobe)

        invlists = ivf_index.invlists

        for new_list_idx, old_list_idx in enumerate(lists_in_shard):
            list_size = invlists.list_size(int(old_list_idx))
            if list_size == 0:
                continue

            ids_ptr = invlists.get_ids(int(old_list_idx))
            codes_ptr = invlists.get_codes(int(old_list_idx))

            # --- FIX 4: Always release pointers after use ---
            try:
                shard_index.invlists.add_entries(
                    int(new_list_idx),
                    list_size,
                    ids_ptr,
                    codes_ptr
                )
                shard_index.ntotal += list_size
            finally:
                invlists.release_ids(int(old_list_idx), ids_ptr)
                invlists.release_codes(int(old_list_idx), codes_ptr)

        shard_name = f"hydra_head_{s_id}.faiss"
        output_file = os.path.join(OUTPUT_DIR, shard_name)
        faiss.write_index(shard_index, output_file)
        print(f"Saved {shard_name} | {shard_nlist} clusters | {shard_index.ntotal} vectors | nprobe={shard_index.nprobe}")

    print(f"\nAll shards and mapping file created successfully in {OUTPUT_DIR}.")

if __name__ == "__main__":
    main()