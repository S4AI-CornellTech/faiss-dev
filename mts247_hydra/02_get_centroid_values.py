#!/usr/bin/env python3
import faiss
import numpy as np
import os

# ==============================================================
# CONFIG
# ==============================================================
INDEX_PATH = "/data/indices/hydra_ivf_500m_sq8.faiss"
OUTPUT_PATH = "/data/indices/hydra_centroids.npy"
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

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    print(f"Mapping index from {INDEX_PATH} (Memory-Lean)...")
    
    # The magic happens here: IO_FLAG_MMAP tells FAISS 
    # "Don't load the vectors, just give me the structure."
    index = faiss.read_index(INDEX_PATH, faiss.IO_FLAG_MMAP)

    ivf_index = extract_ivf_index(index)
    nlist = ivf_index.nlist
    d = ivf_index.d

    # Access the quantizer and extract all centroid vectors
    # Even with MMAP, reconstruct_n works because centroids 
    # are stored in the quantizer, not the inverted lists.
    print(f"Extracting {nlist} centroids (dimension {d})...")
    centroids = ivf_index.quantizer.reconstruct_n(0, nlist)

    print(f"Centroids shape: {centroids.shape}") 

    print(f"Saving to {OUTPUT_PATH}...")
    np.save(OUTPUT_PATH, centroids)

    print("Done. Centroids saved without loading full vector data into RAM.")

if __name__ == "__main__":
    main()