#!/usr/bin/env python3
import os
import csv
import time
import faiss
import numpy as np
import logging
from tqdm import tqdm

# ==============================================================
# Config
# ==============================================================
# index_size = "100m"
# INDEX_NAME = f"/share/suh-scrap/mts247/sustainable_rag/dataset/indices/ivf_pq64_{index_size}.faiss"
# QUERY_FILE = "/share/suh-scrap/mts247/triviaqa_encodings.npy"
# OUTPUT_FILE = f"temp_output/OUTPUT_GPU.csv"
LOG_FILE = f"/home/nvidia/Desktop/error.log"

index_size      = "100m"
INDEX_NAME = f"/home/nvidia/Desktop/ivf_{index_size}.faiss"
QUERY_FILE = "/home/nvidia/Desktop/triviaqa_encodings.npy"
OUTPUT_FILE = f"/home/nvidia/Desktop/gpu_retrieval_test.csv"
# OUTPUT_FILE = f"slurm_data/gpu_{index_size}_ivf_sq8_index_latency_results.csv"
# LOG_FILE = f"slurm_data/gpu_{index_size}_ivf_sq8_index_latency.log"

NPROBE = 256
BATCH_SIZE = 32
RETRIEVED_DOCS = 5
MAX_BATCHES = 100

# ==============================================================
# Setup logging
# ==============================================================
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    filename=LOG_FILE,
    filemode="a",
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ==============================================================
# Shared GPU Resources
# ==============================================================
def get_gpu_resources():
    res = faiss.StandardGpuResources()
    res.setTempMemory(1 << 26)  # 64MB
    res.setDefaultNullStreamAllDevices()
    return res


def load_faiss_gpu_index(index_path, nprobe, res):
    logger.info(f"Loading FAISS index from {index_path}")

    # --- Measure CPU index load ---
    t_start_cpu_load = time.time()
    cpu_index = faiss.read_index(index_path)
    t_end_cpu_load = time.time()

    # --- Measure CPU → GPU transfer ---
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    co.usePrecomputed = False
    co.reserveVecs = 0
    co.verbose = True

    t_start_gpu_load = time.time()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
    t_end_gpu_load = time.time()

    gpu_index.nprobe = nprobe

    cpu_load_time = t_end_cpu_load - t_start_cpu_load
    gpu_load_time = t_end_gpu_load - t_start_gpu_load
    total_load_time = cpu_load_time + gpu_load_time

    logger.info(f"[TIMER] CPU index load time: {cpu_load_time:.3f} s")
    logger.info(f"[TIMER] CPU→GPU transfer time: {gpu_load_time:.3f} s")
    logger.info(f"[TIMER] Total index load time: {total_load_time:.3f} s")

    return gpu_index, cpu_load_time, gpu_load_time, total_load_time


def perform_queries(index, k, embeddings, batch_size):
    query_times = []
    total_queries = min(len(embeddings), MAX_BATCHES * batch_size)
    num_batches = total_queries // batch_size

    logger.info(f"Running {num_batches} batches of size {batch_size}, topk={k}")

    for i in tqdm(
        range(0, total_queries, batch_size),
        desc=f"Batch size={batch_size}, topk={k}",
        leave=False,
    ):
        batch = embeddings[i:i + batch_size]
        t0 = time.time()
        _ = index.search(batch, k)
        t1 = time.time()
        query_times.append(t1 - t0)

    avg_time = sum(query_times) / len(query_times) if query_times else 0
    logger.info(f"[TIMER] Average query latency: {avg_time:.6f} s")
    return avg_time


# ==============================================================
# Main
# ==============================================================
def main():
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    logger.info(f"Loading queries from {QUERY_FILE}")
    queries = np.load(QUERY_FILE).astype(np.float32)

    res = get_gpu_resources()
    logger.info(f"GPU resources initialized.")

    try:
        index, cpu_load, gpu_load, total_load = load_faiss_gpu_index(
            INDEX_NAME, NPROBE, res
        )
        avg_query_time = perform_queries(index, RETRIEVED_DOCS, queries, BATCH_SIZE)

        with open(OUTPUT_FILE, mode="w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "nprobe",
                    "batch_size",
                    "retrieved_docs",
                    "cpu_load_time_s",
                    "gpu_load_time_s",
                    "total_load_time_s",
                    "avg_query_time_s",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "nprobe": NPROBE,
                    "batch_size": BATCH_SIZE,
                    "retrieved_docs": RETRIEVED_DOCS,
                    "cpu_load_time_s": cpu_load,
                    "gpu_load_time_s": gpu_load,
                    "total_load_time_s": total_load,
                    "avg_query_time_s": avg_query_time,
                }
            )

        logger.info(f"[DONE] Results saved to {OUTPUT_FILE}")
    except Exception as e:
        logger.exception(f"[ERROR] Query failed: {e}")


# ==============================================================
# Entry Point
# ==============================================================
if __name__ == "__main__":
    os.environ["FAISS_VERBOSE"] = "1"
    logger.info("=== Starting GPU FAISS profiling ===")
    main()
    logger.info("=== Profiling complete ===")
