#!/usr/bin/env python3
import os
import csv
import time
import faiss
import numpy as np
from tqdm import tqdm

# === Config ===
index_size         = "100m"
index_quantization = "pq64"

INDEX_NAME = f"ivf_{index_size}_{index_quantization}.faiss"
QUERY_FILE = "triviaqa_encodings.npy"
OUTPUT_FILE = f"data/gpu_retrieval_test_ivf_{index_size}_{index_quantization}.csv"

# Sweep parameters (same as CPU script)
NPROBE_LIST         = [64, 128, 256, 512]
BATCH_SIZE_LIST     = [16, 32, 64, 128]
RETRIEVED_DOCS_LIST = [1, 5, 10, 25]

MAX_BATCHES    = 1000
WARMUP_BATCHES = 10


# ==============================================================
# GPU Index Loader (uses your working pattern)
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
# Batch helpers (identical structure to CPU script)
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
        _ = index.search(batch, k)   # EXACTLY like your working GPU file
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

    fieldnames = ["nprobe", "batch_size", "retrieved_docs", "avg_query_time"]

    with open(OUTPUT_FILE, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        total_iterations = (
            len(NPROBE_LIST)
            * len(BATCH_SIZE_LIST)
            * len(RETRIEVED_DOCS_LIST)
        )
        current_iteration = 0

        for nprobe in NPROBE_LIST:

            print(f"\n[INFO] Loading GPU index with nprobe={nprobe}")
            index, res = load_faiss_gpu_index(INDEX_NAME, nprobe)

            for batch_size in BATCH_SIZE_LIST:
                for retrieved_docs in RETRIEVED_DOCS_LIST:

                    current_iteration += 1

                    print(
                        f"\n[RUN {current_iteration}/{total_iterations}] "
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
