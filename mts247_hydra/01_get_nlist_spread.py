#!/usr/bin/env python3
import argparse
import faiss
import numpy as np
import csv
import os


def extract_ivf_index(index):
    """
    Handles wrapped indexes (e.g., IndexPreTransform, IDMap)
    and returns the underlying IVF index.
    """
    if isinstance(index, faiss.IndexPreTransform):
        index = index.index

    if isinstance(index, faiss.IndexIDMap) or isinstance(index, faiss.IndexIDMap2):
        index = index.index

    return index


def main():
    parser = argparse.ArgumentParser(
        description="Inspect FAISS IVF index and dump nlist sizes to CSV"
    )
    parser.add_argument(
        "--index",
        type=str,
        default="/data/indices/hydra_ivf_500m_sq8.faiss",
        help="Path to FAISS index file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/nlist_sizes.csv",
        help="Output CSV file (default: nlist_sizes.csv)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.index):
        raise FileNotFoundError(f"Index file not found: {args.index}")

    print(f"Loading index from: {args.index}")
    index = faiss.read_index(args.index)

    # If GPU index, move to CPU
    if isinstance(index, faiss.IndexShards):
        index = faiss.index_gpu_to_cpu(index)

    index = extract_ivf_index(index)

    if not hasattr(index, "invlists"):
        raise ValueError("This index is not an IVF index (no inverted lists found).")

    nlists = index.nlist
    invlists = index.invlists

    print(f"Number of nlists: {nlists}")
    print("Counting vectors per list...")

    list_sizes = []

    for i in range(nlists):
        size = invlists.list_size(i)
        list_sizes.append(size)

    list_sizes = np.array(list_sizes)

    # Write to CSV
    print(f"Writing results to: {args.output}")
    with open(args.output, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["list_id", "vector_count"])
        for i, size in enumerate(list_sizes):
            writer.writerow([i, size])

    # Summary stats
    print("\n===== Summary =====")
    print(f"Total indexed vectors: {list_sizes.sum()}")
    print(f"Mean list size: {list_sizes.mean():.2f}")
    print(f"Std deviation: {list_sizes.std():.2f}")
    print(f"Min list size: {list_sizes.min()}")
    print(f"Max list size: {list_sizes.max()}")
    print(f"Imbalance ratio (max / mean): {list_sizes.max() / list_sizes.mean():.2f}")


if __name__ == "__main__":
    main()