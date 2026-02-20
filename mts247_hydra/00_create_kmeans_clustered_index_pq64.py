#!/usr/bin/env python3
import math
import numpy as np
import faiss
from tqdm import tqdm
import multiprocessing
import os

# --- Configuration ---
DIM = 768
TOTAL_VECTORS = 5_000_000_000  # 5 Billion
NUM_VECTORS_PER_BATCH = 100_000
THREADS = 70
OUTPUT_FILE = "/data/indices/hydra_ivf_5b_pq64.faiss"

# Index Specs
NLISTS = 262144  # 2^18 clusters for 5B scale
PQ_M = 64        # Number of sub-quantizers
PQ_NBITS = 8     # Bits per sub-quantizer

def generate_vectors(num_vectors, dim):
    """Generates and normalizes synthetic vectors."""
    vectors = np.random.uniform(low=-1.0, high=1.0, size=(num_vectors, dim)).astype("float32")
    faiss.normalize_L2(vectors)
    return vectors

def main():
    faiss.omp_set_num_threads(THREADS)
    print(f"Starting build for {TOTAL_VECTORS} vectors...")
    print(f"Configuration: DIM={DIM}, NLISTS={NLISTS}, PQ_M={PQ_M}")

    # -----------------------------------
    # 1️⃣ Train KMeans properly
    # -----------------------------------
    print("\n[1/4] Training K-Means centroids...")
    
    # Recommendation: 30-256 points per centroid. 
    # For 262k clusters, 5-10M vectors is a solid training set.
    train_size = 10_000_000 
    train_vectors = generate_vectors(train_size, DIM)

    # Use Flat Inner Product index as the quantizer
    quantizer = faiss.IndexFlatIP(DIM)
    
    # Explicit Clustering object
    clustering = faiss.Clustering(DIM, NLISTS)
    clustering.niter = 20
    clustering.max_points_per_centroid = 256
    clustering.train(train_vectors, quantizer)

    # -----------------------------------
    # 2️⃣ Initialize and Train IVF-PQ Index
    # -----------------------------------
    print("\n[2/4] Initializing and training IVF-PQ64 index...")
    
    index = faiss.IndexIVFPQ(
        quantizer,
        DIM,
        NLISTS,
        PQ_M,
        PQ_NBITS,
        faiss.METRIC_INNER_PRODUCT
    )

    # Train the Product Quantizer using the same training data
    index.train(train_vectors)
    del train_vectors # Immediate memory cleanup

    # -----------------------------------
    # 3️⃣ Add all 5B vectors in streaming batches
    # -----------------------------------
    print(f"\n[3/4] Adding {TOTAL_VECTORS} vectors...")

    num_batches = math.ceil(TOTAL_VECTORS / NUM_VECTORS_PER_BATCH)

    with tqdm(total=TOTAL_VECTORS, unit="vec") as pbar:
        for _ in range(num_batches):
            # Calculate remaining vectors for the final batch
            remaining = TOTAL_VECTORS - pbar.n
            current_batch_size = min(NUM_VECTORS_PER_BATCH, remaining)
            
            vectors = generate_vectors(current_batch_size, DIM)
            index.add(vectors)
            pbar.update(vectors.shape[0])

    # -----------------------------------
    # 4️⃣ Save
    # -----------------------------------
    print(f"\n[4/4] Saving index to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    faiss.write_index(index, OUTPUT_FILE)

    print("Done. Index build complete.")

if __name__ == "__main__":
    main()