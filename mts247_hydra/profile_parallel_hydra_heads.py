#!/usr/bin/env python3
import os
import time
import csv
import gc
import logging
from datetime import datetime

import faiss
from tqdm import tqdm

# ==============================================================
# Logging Setup
# ==============================================================
def setup_logging():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"hydra_bench_{timestamp}.log")

    # Create logger
    logger = logging.getLogger("HydraBench")
    logger.setLevel(logging.INFO)

    # File Handler (Detailed)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Console Handler (Clean)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger, log_file

log, LOG_PATH = setup_logging()

# ==============================================================
# Config
# ==============================================================
HYDRA_SHARDS = [
    "/data/indices/shards/hydra_head_0.faiss", 
    "/data/indices/shards/hydra_head_1.faiss", 
    "/data/indices/shards/hydra_head_2.faiss", 
    "/data/indices/shards/hydra_head_3.faiss", 
    # "/data/indices/shards/hydra_head_4.faiss", 
    # "/data/indices/shards/hydra_head_5.faiss", 
    # "/data/indices/shards/hydra_head_6.faiss", 
    # "/data/indices/shards/hydra_head_7.faiss", 
]

TRIALS = 5
USE_UNIFIED_MEMORY = False
PINNED_MEM_BYTES = 2 * 1024 * 1024 * 1024
TEMP_MEM_BYTES = 0

try:
    import pycuda.driver as cuda
    cuda.init() # Ensure CUDA is initialized
    CUDA_AVAILABLE = True
except Exception:
    CUDA_AVAILABLE = False

def get_gpu_resources():
    res = faiss.StandardGpuResources()
    res.setTempMemory(TEMP_MEM_BYTES)
    res.setPinnedMemory(PINNED_MEM_BYTES)
    return res

def clear_gpu_memory():
    if CUDA_AVAILABLE:
        try:
            # Synchronize before clearing
            device = cuda.Device(0)
            context = device.make_context()
            context.synchronize()
            context.pop()
        except Exception as e:
            log.debug(f"GPU Clear Warning: {e}")
    gc.collect()

def main():
    # FAISS Optimization Envs
    os.environ.update({
        "FAISS_GPU_PACKED_LISTS": "1",
        "FAISS_GPU_PACKED_LISTS_MMAP": "1",
        "FAISS_GPU_DEVICEVECTOR_CACHE": "1",
        "FAISS_GPU_DEVICEVECTOR_CACHE_MIN_BYTES": str(1 << 30),
    })

    results = []
    shard_metadata = {} # To store ntotal and disk times per shard

    log.info(f"{'='*60}\nSTEP 1: Loading Shards to RAM\n{'='*60}")
    
    loaded_indices = []
    for idx, path in enumerate(HYDRA_SHARDS):
        t0 = time.perf_counter()
        cpu_index = faiss.read_index(path)
        dt = time.perf_counter() - t0
        
        loaded_indices.append(cpu_index)
        shard_metadata[idx] = {
            'disk_to_cpu_s': dt,
            'ntotal': cpu_index.ntotal,
            'name': os.path.basename(path)
        }
        log.info(f"Loaded Shard {idx} | Time: {dt:.4f}s | Vectors: {cpu_index.ntotal:,}")

    log.info(f"\n{'='*60}\nSTEP 2: Starting GPU Transfer Trials ({TRIALS})\n{'='*60}")

    for trial in range(1, TRIALS + 1):
        # Progress bar for internal shard tracking
        pbar = tqdm(enumerate(loaded_indices), total=len(loaded_indices), desc=f"Trial {trial}/{TRIALS}")
        
        for idx, cpu_index in pbar:
            res = get_gpu_resources()
            
            co = faiss.GpuClonerOptions()
            co.useUnifiedMemory = USE_UNIFIED_MEMORY
            co.useFloat16 = True
            co.indicesOptions = faiss.INDICES_32_BIT

            # Transfer & Sync
            t_start = time.perf_counter()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index, co)
            res.syncDefaultStreamCurrentDevice()
            t_end = time.perf_counter()

            elapsed = t_end - t_start
            
            results.append({
                'shard_idx': idx,
                'shard_name': shard_metadata[idx]['name'],
                'trial': trial,
                'disk_to_cpu_s': shard_metadata[idx]['disk_to_cpu_s'],
                'cpu_to_gpu_s': elapsed,
                'ntotal': shard_metadata[idx]['ntotal']
            })

            # Explicit Cleanup
            del gpu_index
            del res
            clear_gpu_memory()

    # ----------------------------------------------------------
    # Save & Summarize
    # ----------------------------------------------------------
    os.makedirs("data", exist_ok=True)
    out_path = "data/HYDRA_SHARD_TRANSFER_TIMES.csv"
    
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    log.info(f"\n{'='*60}\nFINAL SUMMARY\n{'='*60}")
    for idx in shard_metadata:
        shard_runs = [r['cpu_to_gpu_s'] for r in results if r['shard_idx'] == idx]
        avg_gpu = sum(shard_runs) / len(shard_runs)
        log.info(f"Shard {idx} ({shard_metadata[idx]['name']}):")
        log.info(f"  Disk -> RAM: {shard_metadata[idx]['disk_to_cpu_s']:.4f}s")
        log.info(f"  RAM -> GPU (Avg of {TRIALS}): {avg_gpu:.4f}s")

    log.info(f"\nCSV results: {out_path}")
    log.info(f"Full log: {LOG_PATH}")

if __name__ == "__main__":
    main()