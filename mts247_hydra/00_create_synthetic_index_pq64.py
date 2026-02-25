#!/usr/bin/env python3
import argparse
import math
import numpy as np
import faiss
from tqdm import tqdm
import multiprocessing
import os

# Fixed number of vectors generated per batch.
NUM_VECTORS_PER_BATCH = 100_000

def generate_vectors(num_vectors, dim, queue):
    vectors = np.random.uniform(low=-1.0, high=1.0, size=(num_vectors, dim)).astype('float32')
    queue.put(vectors)

def build_single_index(target_count, dim, num_workers, output_dir):
    """Builds an IVF-PQ64 index optimized for the target_count."""
    
    # 1. Config & Validation
    nlists = int(math.sqrt(target_count))
    pq_m = 64
    pq_nbits = 8
    
    if dim % pq_m != 0:
        raise ValueError(f"Dimension {dim} must be divisible by pq_m ({pq_m})")

    # PQ requires more training data than SQ8 to build a good codebook
    # FAISS recommends at least 10x (2^nbits) per sub-quantizer
    train_size = min(max(2560, 40 * nlists), target_count, 1_000_000)
    
    label = f"{target_count // 1_000_000}m"
    filename = os.path.join(output_dir, f"hydra_ivf_{label}_pq64.faiss")
    
    print(f"\n--- Building {label} PQ64 Index ---")
    print(f"Target: {target_count} | nlists: {nlists} | Training on: {train_size}")

    # 2. Initialize and Train
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(
        quantizer, dim, nlists, pq_m, pq_nbits, faiss.METRIC_INNER_PRODUCT
    )
    
    print(f"Generating training data for {label}...")
    train_vecs = np.random.uniform(low=-1.0, high=1.0, size=(train_size, dim)).astype('float32')
    index.train(train_vecs)
    
    # 3. Fill the index
    num_batches = math.ceil(target_count / NUM_VECTORS_PER_BATCH)
    queue = multiprocessing.Queue(maxsize=num_workers)
    processes = []
    
    with tqdm(total=target_count, desc=f"Adding to {label}", unit="vec") as pbar:
        for _ in range(num_batches):
            if len(processes) < num_workers:
                p = multiprocessing.Process(target=generate_vectors, args=(NUM_VECTORS_PER_BATCH, dim, queue))
                p.start()
                processes.append(p)
            
            vectors = queue.get()
            if pbar.n + vectors.shape[0] > target_count:
                vectors = vectors[:target_count - pbar.n]
                
            index.add(vectors)
            pbar.update(vectors.shape[0])
            processes = [p for p in processes if p.is_alive()]

    # 4. Save
    print(f"Saving to {filename}...")
    faiss.write_index(index, filename)
    
    for p in processes:
        p.join()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--threads", type=int, default=multiprocessing.cpu_count())
    parser.add_argument("--output-dir", type=str, default="/data/indices")
    args = parser.parse_args()

    # Define all independent targets: 10m, 20m... 100m
    targets = [5_000_000_000]
    
    # Set FAISS internal threading
    faiss.omp_set_num_threads(args.threads)
    os.makedirs(args.output_dir, exist_ok=True)
    
    for count in targets:
        build_single_index(count, args.dim, args.threads, args.output_dir)

if __name__ == "__main__":
    main()