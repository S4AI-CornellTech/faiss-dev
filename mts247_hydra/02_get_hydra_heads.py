#!/usr/bin/env python3
import faiss
import numpy as np
import os

# ==============================================================
# CONFIG
# ==============================================================
INDEX_PATH = "/data/indices/hydra_ivf_500m_sq8.faiss"
OUTPUT_PATH = "data/hydra_centroids.npy"
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

    print(f"Loading index from {INDEX_PATH}...")
    index = faiss.read_index(INDEX_PATH)

    ivf_index = extract_ivf_index(index)
    nlist = ivf_index.nlist

    # Access the quantizer and extract all centroid vectors
    print(f"Extracting {nlist} centroids...")
    centroids = ivf_index.quantizer.reconstruct_n(0, nlist)

    print(f"Centroids shape: {centroids.shape}") # (nlist, d)

    print(f"Saving to {OUTPUT_PATH}...")
    # Saving as a standard .npy file for a direct list/array
    np.save(OUTPUT_PATH, centroids)

    print("Done. Centroids saved as a flat array.")

if __name__ == "__main__":
    main()