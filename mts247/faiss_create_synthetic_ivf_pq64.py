#!/usr/bin/env python3
import argparse
import math
import numpy as np
import faiss
from tqdm import tqdm
import multiprocessing
import os

NUM_VECTORS_PER_BATCH = 100_000

def parse_total_index_size(size_str):
    size_str = size_str.lower().strip()
    if size_str.endswith("k"):
        multiplier = 10**3
        number_part = size_str[:-1]
    elif size_str.endswith("m"):
        multiplier = 10**6
        number_part = size_str[:-1]
    elif size_str.endswith("b"):
        multiplier = 10**9
        number_part = size_str[:-1]
    else:
        raise ValueError("Index size must end with k, m, or b")

    return int(float(number_part) * multiplier)

def generate_vectors(num_vectors, dim, queue):
    vectors = np.random.uniform(-1.0, 1.0, size=(num_vectors, dim)).astype("float32")
    queue.put(vectors)

def create_faiss_index(total_vectors, dim, num_workers, num_vectors_per_batch):
    # -----------------------------
    # IVF-PQ configuration
    # -----------------------------
    nlists = 16_000
    pq_m = 64          # PQ64
    pq_nbits = 8       # 8 bits per subquantizer

    if dim % pq_m != 0:
        raise ValueError(f"dim ({dim}) must be divisible by pq_m ({pq_m})")

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(
        quantizer,
        dim,
        nlists,
        pq_m,
        pq_nbits,
        faiss.METRIC_INNER_PRODUCT,
    )

    # -----------------------------
    # Training
    # -----------------------------
    train_size = max(100_000, 50 * nlists)
    print(f"Training IVF-PQ64 with {train_size} vectors...")
    train_vectors = np.random.uniform(
        -1.0, 1.0, size=(train_size, dim)
    ).astype("float32")
    index.train(train_vectors)

    # -----------------------------
    # Parallel add
    # -----------------------------
    num_batches = math.ceil(total_vectors / num_vectors_per_batch)
    queue = multiprocessing.Queue(maxsize=num_workers)
    processes = []

    print("Adding vectors...")
    with tqdm(total=num_batches, unit="batch") as pbar:
        for _ in range(num_batches):
            if len(processes) < num_workers:
                p = multiprocessing.Process(
                    target=generate_vectors,
                    args=(num_vectors_per_batch, dim, queue),
                )
                p.start()
                processes.append(p)

            vectors = queue.get()
            index.add(vectors)
            pbar.update(1)

            processes = [p for p in processes if p.is_alive()]

        for p in processes:
            p.join()

    return index

def main():
    parser = argparse.ArgumentParser("FAISS IVF-PQ64 generator")
    parser.add_argument("--index-size", required=True, type=str)
    parser.add_argument("--dim", required=True, type=int)
    parser.add_argument("--threads", type=int, default=70)
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    total_vectors = parse_total_index_size(args.index_size)
    faiss.omp_set_num_threads(args.threads)

    print("Config:")
    print(f"  vectors : {total_vectors}")
    print(f"  dim     : {args.dim}")
    print(f"  threads : {args.threads}")

    index = create_faiss_index(
        total_vectors,
        args.dim,
        args.threads,
        NUM_VECTORS_PER_BATCH,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = f"{args.output_dir}/ivf_{args.index_size}_pq64.faiss"
    faiss.write_index(index, out_path)
    print(f"Saved index to {out_path}")

if __name__ == "__main__":
    main()
