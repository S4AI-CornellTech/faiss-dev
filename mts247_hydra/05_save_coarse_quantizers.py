#!/usr/bin/env python3
"""
save_coarse_quantizers.py
-------------------------
ONE-TIME MIGRATION: Run this once with your existing CPU indices loaded.
It extracts and saves just the coarse quantizer (centroid weights) from
each shard — typically ~nlist * d * 4 bytes, so a few hundred MB total
vs tens of GB for the full index.

After running this, you can delete the CPU index from RAM and never load
it again. The benchmark script will use these tiny files instead.

Usage:
    python save_coarse_quantizers.py
"""

import os
import glob
import re
import time
import faiss

SHARDS_DIR   = "/data/indices/hydra/shards"
SHARD_GLOB   = "hydra_head_*.faiss"
QUANTIZER_DIR = "/data/indices/hydra/coarse_quantizers"


def discover_shards(shards_dir, shard_glob):
    shard_paths = glob.glob(os.path.join(shards_dir, shard_glob))
    def key(p):
        m = re.search(r"(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else float("inf")
    shard_paths.sort(key=key)
    return shard_paths


def main():
    os.makedirs(QUANTIZER_DIR, exist_ok=True)
    shard_paths = discover_shards(SHARDS_DIR, SHARD_GLOB)
    print(f"Found {len(shard_paths)} shards. Extracting coarse quantizers...\n")

    for shard_idx, shard_path in enumerate(shard_paths):
        out_path = os.path.join(QUANTIZER_DIR, f"coarse_quantizer_shard_{shard_idx}.faiss")

        if os.path.exists(out_path):
            print(f"  Shard {shard_idx}: already exists, skipping → {out_path}")
            continue

        print(f"  Shard {shard_idx}: loading {shard_path} ...")
        t0 = time.perf_counter()
        cpu_index = faiss.read_index(shard_path)
        t1 = time.perf_counter()

        # Extract the coarse quantizer and shard metadata.
        quantizer = faiss.downcast_index(cpu_index.quantizer)
        nlist     = cpu_index.nlist
        d         = cpu_index.d
        metric    = cpu_index.metric_type
        ntotal    = cpu_index.ntotal
        code_size = cpu_index.code_size if hasattr(cpu_index, "code_size") else d * 4

        faiss.write_index(quantizer, out_path)
        t2 = time.perf_counter()

        file_mb = os.path.getsize(out_path) / (1024 ** 2)
        print(f"    Load: {t1-t0:.2f}s | Save: {t2-t1:.2f}s | "
              f"nlist={nlist} d={d} ntotal={ntotal:,} code_size={code_size} | "
              f"Quantizer file: {file_mb:.1f} MB → {out_path}")

        # Save shard metadata (nlist, d, metric, ntotal, code_size) alongside.
        meta_out = out_path.replace(".faiss", ".meta.txt")
        with open(meta_out, "w") as f:
            f.write(f"nlist={nlist}\n")
            f.write(f"d={d}\n")
            f.write(f"metric={metric}\n")
            f.write(f"ntotal={ntotal}\n")
            f.write(f"code_size={code_size}\n")

        del cpu_index

    print(f"\nDone. Coarse quantizers saved to {QUANTIZER_DIR}")
    print("You can now run the benchmark without loading full CPU indices.")


if __name__ == "__main__":
    main()
