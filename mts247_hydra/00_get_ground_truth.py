#!/usr/bin/env python3
import os
import csv
import time
import faiss
import numpy as np
from tqdm import tqdm
import multiprocessing

# === Config ===
index_quantization = "sq8"
QUERY_FILE = "triviaqa_encodings.npy"
OUTPUT_FILE = f"data/hydra_monolithic_ground_truth.csv"

# Lists of parameters to iterate through
NPROBE_LIST          = [24494]         # list of nprobe values
BATCH_SIZE_LIST      = [1]          # list of batch sizes
RETRIEVED_DOCS_LIST  = [10]             # list of top-k values
INDEX_SIZE_LIST      = ["600m"]

MAX_BATCHES     = 500        # measured batches (after warmup)
WARMUP_BATCHES  = 0           # unmeasured warmup batches

# Automatically detect all available threads
NUM_THREADS = multiprocessing.cpu_count()

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

def get_resume_query_start(output_path):
    if not os.path.exists(output_path):
        return 0

    max_query = -1
    with open(output_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "query" not in reader.fieldnames:
            return 0

        for row in reader:
            query_value = row.get("query", "")
            try:
                query_idx = int(query_value)
            except (TypeError, ValueError):
                continue
            if query_idx > max_query:
                max_query = query_idx

    return max_query + 1

def measure(index, k, embeddings, batch_size, measure_batches, writer, output_file, run_meta):
    times = []
    for batch_idx, batch in enumerate(tqdm(
        _iterate_batches(embeddings, batch_size, measure_batches),
        total=measure_batches,
        desc=f"Measuring ({measure_batches} batches)",
        leave=False
    )):
        start = time.time()
        distances, indices = index.search(batch, k)
        end = time.time()
        search_time_s = end - start
        times.append(search_time_s)

        batch_start_query = run_meta["start_query_idx"] + (batch_idx * batch_size)
        for row_idx_in_batch in range(len(indices)):
            query_idx = batch_start_query + row_idx_in_batch
            top_k_ids = indices[row_idx_in_batch][:k].tolist() if len(indices[row_idx_in_batch]) > 0 else []

            writer.writerow({
                "query": query_idx,
                "index_size": run_meta["index_size"],
                "threads": run_meta["threads"],
                "nprobe": run_meta["nprobe"],
                "batch_size": run_meta["batch_size"],
                "num_retrieved_docs": run_meta["num_retrieved_docs"],
                "avg_query_time": search_time_s / max(1, len(batch)),
                "best_retrieved_ids": str(top_k_ids)
            })

        output_file.flush()  # persist to disk after each search() call

    if not times:
        return 0.0
    avg_time = sum(times) / len(times)
    return avg_time  # average per-batch time in seconds

def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    start_query_idx = get_resume_query_start(OUTPUT_FILE)

    # Load queries once
    queries = np.load(QUERY_FILE).astype(np.float32)
    queries = np.ascontiguousarray(queries)

    if start_query_idx >= len(queries):
        print(f"[INFO] Output already has all queries (next={start_query_idx}, total={len(queries)}). Nothing to do.")
        return

    queries = queries[start_query_idx:]
    print(f"[INFO] Resuming from query index {start_query_idx} ({len(queries)} queries remaining)")

    print(f"[INFO] Detected {NUM_THREADS} CPU threads")
    
    # Set FAISS to use all available threads
    faiss.omp_set_num_threads(NUM_THREADS)

    # Prepare CSV
    fieldnames = ["query", "index_size", "threads", "nprobe", "batch_size", "num_retrieved_docs", "avg_query_time", "best_retrieved_ids"]
    output_exists_and_has_data = os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0
    with open(OUTPUT_FILE, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not output_exists_and_has_data:
            writer.writeheader()

        # Calculate total iterations for progress tracking
        total_iterations = len(NPROBE_LIST) * len(BATCH_SIZE_LIST) * len(RETRIEVED_DOCS_LIST)
        current_iteration = 0

        # Iterate through all combinations of parameters
        for index_size in INDEX_SIZE_LIST:
            for nprobe in NPROBE_LIST:
                # Load index with current nprobe setting
                print(f"\n[INFO] Loading CPU index with nprobe={nprobe}")
                index_name = f"/data/indices/hydra/hydra_sphere_ivf_{index_size}_{index_quantization}.faiss"
                index = load_faiss_cpu_index(index_name, nprobe)
                
                for batch_size in BATCH_SIZE_LIST:
                    for retrieved_docs in RETRIEVED_DOCS_LIST:
                        current_iteration += 1
                        
                        print(f"\n[RUN {current_iteration}/{total_iterations}] "
                            f"index_size={index_size} | threads={NUM_THREADS} | nprobe={nprobe} | "
                            f"batch_size={batch_size} | k={retrieved_docs} | "
                            f"warmup={WARMUP_BATCHES} | measure={MAX_BATCHES}")

                        # Warmup (unmeasured)
                        warmup(index, retrieved_docs, queries, batch_size, WARMUP_BATCHES)

                        # Measure and stream per-search results to CSV
                        avg_batch_time_s = measure(
                            index,
                            retrieved_docs,
                            queries,
                            batch_size,
                            MAX_BATCHES,
                            writer,
                            f,
                            {
                                "index_size": index_size,
                                "threads": NUM_THREADS,
                                "nprobe": nprobe,
                                "batch_size": batch_size,
                                "num_retrieved_docs": retrieved_docs,
                                "start_query_idx": start_query_idx,
                            },
                        )
                        
                        print(f"[RESULT] {avg_batch_time_s:.6f}s avg per batch")

    print(f"\n[DONE] Wrote {total_iterations} results to {OUTPUT_FILE}")

if __name__ == "__main__":
    # Verbosity can help confirm FAISS is using OMP on CPU
    os.environ.setdefault("FAISS_VERBOSE", "0")
    main()