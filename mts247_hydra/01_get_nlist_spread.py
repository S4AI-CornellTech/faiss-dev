#!/usr/bin/env python3
import argparse
import faiss
import numpy as np
import csv
import os


TARGET_GB = 90
BYTES_IN_GB = 1024 ** 3


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
        description="Inspect FAISS IVF index and analyze nlist memory usage"
    )
    parser.add_argument(
        "--index",
        type=str,
        default="/data/indices/hydra_ivf_300m_sq8.faiss",
        help="Path to FAISS index file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/nlist_sizes.csv",
        help="Output CSV file",
    )

    args = parser.parse_args()

    if not os.path.exists(args.index):
        raise FileNotFoundError(f"Index file not found: {args.index}")

    print(f"Loading index from: {args.index}")
    index = faiss.read_index(args.index)

    if isinstance(index, faiss.IndexShards):
        index = faiss.index_gpu_to_cpu(index)

    index = extract_ivf_index(index)

    if not hasattr(index, "invlists"):
        raise ValueError("This index is not an IVF index.")

    nlists = index.nlist
    invlists = index.invlists

    print(f"Number of nlists: {nlists}")
    print("Computing memory per list...")

    code_size = index.code_size  # bytes per vector for codes
    id_size = 8  # int64 ids

    list_info = []

    for i in range(nlists):
        vec_count = invlists.list_size(i)
        list_bytes = vec_count * (code_size + id_size)
        list_info.append((i, vec_count, list_bytes))

    # Convert to numpy structured array
    dtype = [("list_id", int), ("vector_count", int), ("bytes", int)]
    list_info = np.array(list_info, dtype=dtype)

    # Sort by memory descending
    list_info = np.sort(list_info, order="bytes")[::-1]

    # Accumulate until reaching 90 GB
    target_bytes = TARGET_GB * BYTES_IN_GB
    cumulative_bytes = 0
    cumulative_vectors = 0
    num_lists = 0

    for entry in list_info:
        cumulative_bytes += entry["bytes"]
        cumulative_vectors += entry["vector_count"]
        num_lists += 1

        if cumulative_bytes >= target_bytes:
            break

    # Write CSV (optional — unchanged functionality)
    print(f"Writing nlist sizes to: {args.output}")
    with open(args.output, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["list_id", "vector_count"])
        for entry in list_info:
            writer.writerow([entry["list_id"], entry["vector_count"]])

    # Summary
    print("\n===== 90GB Largest nlists Summary =====")
    print(f"Target memory: {TARGET_GB} GB")
    print(f"Number of largest nlists needed: {num_lists}")
    print(f"Total vectors in those lists: {cumulative_vectors}")
    print(f"Total memory used: {cumulative_bytes / BYTES_IN_GB:.2f} GB")

    print("\n===== Overall Stats =====")
    print(f"Total indexed vectors: {sum(list_info['vector_count'])}")
    print(f"Mean list size: {np.mean(list_info['vector_count']):.2f}")
    print(f"Max list size: {np.max(list_info['vector_count'])}")
    print(f"Imbalance ratio (max / mean): "
          f"{np.max(list_info['vector_count']) / np.mean(list_info['vector_count']):.2f}")


if __name__ == "__main__":
    main()