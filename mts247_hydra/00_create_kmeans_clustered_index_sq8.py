#!/usr/bin/env python3
import math
import numpy as np
import faiss
from tqdm import tqdm
import multiprocessing
import os

DIM = 768
TOTAL_VECTORS = 500_000_000
NUM_VECTORS_PER_BATCH = 100_000
OUTPUT_FILE = "/data/indices/hydra_ivf_500m_sq8.faiss"

def generate_vectors(num_vectors, dim):
    return np.random.uniform(
        low=-1.0, high=1.0,
        size=(num_vectors, dim)
    ).astype("float32")

def main():

    print("Computing nlists...")
    nlists = int(math.sqrt(TOTAL_VECTORS))
    print(f"Total vectors: {TOTAL_VECTORS}")
    print(f"nlists (sqrt(N)): {nlists}")

    # -----------------------------------
    # 1️⃣ Train KMeans properly
    # -----------------------------------

    print("\nTraining k-means...")

    # FAISS Clustering object
    clustering = faiss.Clustering(DIM, nlists)

    # Training size recommendation
    train_size = min(40 * nlists, 5_000_000)
    print(f"Training on {train_size} vectors")

    train_vectors = generate_vectors(train_size, DIM)

    # Use flat index as quantizer during training
    quantizer = faiss.IndexFlatIP(DIM)

    clustering.train(train_vectors, quantizer)

    # -----------------------------------
    # 2️⃣ Build IVF index using trained centroids
    # -----------------------------------

    print("\nBuilding IVF-SQ8 index...")

    index = faiss.IndexIVFScalarQuantizer(
        quantizer,
        DIM,
        nlists,
        faiss.ScalarQuantizer.QT_8bit,
        faiss.METRIC_INNER_PRODUCT
    )

    index.train(train_vectors)

    # -----------------------------------
    # 3️⃣ Add all 500M vectors in streaming batches
    # -----------------------------------

    print("\nAdding 500M vectors...")

    num_batches = math.ceil(TOTAL_VECTORS / NUM_VECTORS_PER_BATCH)

    with tqdm(total=TOTAL_VECTORS, unit="vec") as pbar:
        for _ in range(num_batches):
            vectors = generate_vectors(NUM_VECTORS_PER_BATCH, DIM)

            if pbar.n + vectors.shape[0] > TOTAL_VECTORS:
                vectors = vectors[:TOTAL_VECTORS - pbar.n]

            index.add(vectors)
            pbar.update(vectors.shape[0])

    # -----------------------------------
    # 4️⃣ Save
    # -----------------------------------

    print("\nSaving index...")
    faiss.write_index(index, OUTPUT_FILE)

    print("Done.")

if __name__ == "__main__":
    main()