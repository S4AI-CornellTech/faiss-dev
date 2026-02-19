#!/usr/bin/env python3
import os
import csv
import time
import faiss
import numpy as np
from tqdm import tqdm

# === Config ===
index_quantization = "sq8"
QUERY_FILE = "triviaqa_encodings.npy"
OUTPUT_FILE = f"data/gpu_retrieval_test_ivf_{index_quantization}.csv"

# Parameter sweeps
NPROBE_LIST         = [256]
BATCH_SIZE_LIST     = [32]
RETRIEVED_DOCS_LIST = [5]
INDEX_SIZE_LIST     = ["10m", "20m", "30m", "40m", "50m",
                       "60m", "70m", "80m", "90m", "100m"]

MAX_BATCHES    = 1000
WARMUP_BATCHES = 10


# ==============================================================
# GPU Index Loader
# ==============================================================

def load_faiss_gpu_index(index_path, nprobe):
    cpu_index = faiss.read_index(index_path)

    res = faiss.StandardGpuResources()

    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    co.usePrecomputed = False
    co.reserveVecs = 0

    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
    gpu_index.nprobe = nprobe

    return gpu_index, res


# ==============================================================
# Batch helpers
# ==============================================================

def _iterate_batches(data, batch_size, limit_batches):
    limit = min(len(data), limit_batches * batch_size)
    for i in range(0, limit, batch_size):
        yield data[i:i + batch_size]


def warmup(index, k, embeddings, batch_size, warmup_batches):
    if warmup_batches <= 0:
        return
    for batch in _iterate_batches(embeddings, batch_size, warmup_batches):
        _ = index.search(batch, k)


def measure(index, k, embeddings, batch_size, measure_batches):
    times = []
    for batch in tqdm(
        _iterate_batches(embeddings, batch_size, measure_batches),
        total=measure_batches,
        desc=f"Measuring ({measure_batches} batches)",
        leave=False,
    ):
        start = time.time()
        _ = index.search(batch, k)
        end = time.time()
        times.append(end - start)

    if not times:
        return 0.0

    return sum(times) / len(times)


# ==============================================================
# Main
# ==============================================================

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    queries = np.load(QUERY_FILE).astype(np.float32)
    queries = np.ascontiguousarray(queries)

    print("[INFO] Using GPU for FAISS search")

    fieldnames = [
        "index_size",
        "nprobe",
        "batch_size",
        "retrieved_docs",
        "avg_query_time"
    ]

    with open(OUTPUT_FILE, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        total_iterations = (
            len(INDEX_SIZE_LIST)
            * len(NPROBE_LIST)
            * len(BATCH_SIZE_LIST)
            * len(RETRIEVED_DOCS_LIST)
        )

        current_iteration = 0

        for index_size in INDEX_SIZE_LIST:

            index_name = f"indices/ivf_{index_size}_{index_quantization}.faiss"

            for nprobe in NPROBE_LIST:

                print(f"\n[INFO] Loading GPU index: {index_name} | nprobe={nprobe}")
                index, res = load_faiss_gpu_index(index_name, nprobe)

                for batch_size in BATCH_SIZE_LIST:
                    for retrieved_docs in RETRIEVED_DOCS_LIST:

                        current_iteration += 1

                        print(
                            f"\n[RUN {current_iteration}/{total_iterations}] "
                            f"index_size={index_size} | "
                            f"nprobe={nprobe} | "
                            f"batch_size={batch_size} | "
                            f"k={retrieved_docs} | "
                            f"warmup={WARMUP_BATCHES} | "
                            f"measure={MAX_BATCHES}"
                        )

                        warmup(index, retrieved_docs,
                               queries, batch_size, WARMUP_BATCHES)

                        avg_batch_time_s = measure(
                            index,
                            retrieved_docs,
                            queries,
                            batch_size,
                            MAX_BATCHES,
                        )

                        writer.writerow({
                            "index_size": index_size,
                            "nprobe": nprobe,
                            "batch_size": batch_size,
                            "retrieved_docs": retrieved_docs,
                            "avg_query_time": avg_batch_time_s
                        })
                        f.flush()

                        print(f"[RESULT] {avg_batch_time_s:.6f}s avg per batch")

                del index
                del res

    print(f"\n[DONE] Wrote results to {OUTPUT_FILE}")


if __name__ == "__main__":
    os.environ.setdefault("FAISS_VERBOSE", "0")
    main()
