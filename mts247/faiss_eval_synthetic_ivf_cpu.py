#!/usr/bin/env python3
import os
import csv
import time
import faiss
import numpy as np
from tqdm import tqdm

# === Config ===
index_size      = "100m"
INDEX_NAME = f"/home/nvidia/Desktop/ivf_{index_size}.faiss"
QUERY_FILE = "/home/nvidia/Desktop/triviaqa_encodings.npy"
OUTPUT_FILE = f"/home/nvidia/Desktop/cpu_retrieval_test.csv"

NPROBE          = 256         # scalar
BATCH_SIZE      = 32         # scalar
RETRIEVED_DOCS  = 5           # scalar (top-k)
MAX_BATCHES     = 100        # measured batches (after warmup)
WARMUP_BATCHES  = 5           # unmeasured warmup batches
THREAD_RANGE    = (32, 33)     # inclusive

# === Helpers ===
def load_faiss_cpu_index(index_path, nprobe):
    index = faiss.read_index(index_path)
    # Set nprobe if supported (IndexIVF and descendants)
    if hasattr(index, "nprobe"):
        index.nprobe = int(nprobe)
    else:
        # Fallback via ParameterSpace if available
        try:
            ps = faiss.ParameterSpace()
            ps.set_index_parameter(index, "nprobe", int(nprobe))
        except Exception:
            pass
    return index

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
        leave=False
    ):
        start = time.time()
        _ = index.search(batch, k)
        end = time.time()
        times.append(end - start)
    if not times:
        return 0.0
    return sum(times) / len(times)  # average per-batch time in seconds

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Load queries once
    queries = np.load(QUERY_FILE).astype(np.float32)
    queries = np.ascontiguousarray(queries)

    print(f"[INFO] Loading CPU index: {INDEX_NAME} (nprobe={NPROBE})")
    index = load_faiss_cpu_index(INDEX_NAME, NPROBE)

    # Prepare CSV
    fieldnames = ["threads", "nprobe", "batch_size", "retrieved_docs", "avg_query_time"]
    with open(OUTPUT_FILE, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Sweep threads from 1 .. 72 (inclusive)
        t_min, t_max = THREAD_RANGE
        for t in range(t_min, t_max + 1):
            # Set FAISS OpenMP thread count
            faiss.omp_set_num_threads(int(t))

            print(f"[RUN] threads={t} | warmup={WARMUP_BATCHES} | measure={MAX_BATCHES} | "
                  f"bs={BATCH_SIZE} | k={RETRIEVED_DOCS}")

            # Warmup (unmeasured)
            warmup(index, RETRIEVED_DOCS, queries, BATCH_SIZE, WARMUP_BATCHES)

            # Measure
            avg_batch_time_s = measure(index, RETRIEVED_DOCS, queries, BATCH_SIZE, MAX_BATCHES)

            writer.writerow({
                "threads": t,
                "nprobe": NPROBE,
                "batch_size": BATCH_SIZE,
                "retrieved_docs": RETRIEVED_DOCS,
                "avg_query_time": avg_batch_time_s
            })
            f.flush()  # ensure progress hits disk each iteration

    print(f"[DONE] Wrote per-thread results to {OUTPUT_FILE}")

if __name__ == "__main__":
    # Verbosity can help confirm FAISS is using OMP on CPU
    os.environ.setdefault("FAISS_VERBOSE", "0")
    main()
