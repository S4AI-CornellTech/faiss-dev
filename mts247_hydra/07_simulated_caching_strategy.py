import os
import math
import csv
import ast
import re
import sys

def get_shard_data(directory):
    shards = []
    if not os.path.exists(directory):
        return []
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        if os.path.isfile(path):
            match = re.search(r'head_(\d+)', f)
            s_id = int(match.group(1)) if match else f
            shards.append({'name': f, 'id': s_id, 'bytes': os.path.getsize(path)})
    return sorted(shards, key=lambda x: x['bytes'], reverse=True)

def run_simulation(shard_dir, csv_path):
    shards = get_shard_data(shard_dir)
    if not shards:
        print(f"Error: No shards found in {shard_dir}")
        return

    # --- 1. THE TABLE (Balancing Logic) ---
    num_cols, max_gb = 4, 95
    n = len(shards)
    num_rows = math.ceil(n / num_cols)
    while num_rows <= n:
        valid_config = True
        for r in range(num_rows):
            row_sum_gb = sum(shards[c * num_rows + r]['bytes'] / (1024**3)
                             for c in range(num_cols) if (c * num_rows + r) < n)
            if row_sum_gb >= max_gb:
                valid_config = False; break
        if valid_config: break
        num_rows += 1

    print(f"\n{'Col 1':<30} | {'Col 2':<30} | {'Col 3':<30} | {'Col 4':<30}")
    print("-" * 130)

    shard_to_col = {}
    for r in range(num_rows):
        row_cells = []
        for c in range(num_cols):
            idx = (c * num_rows) + r
            if idx < n:
                s = shards[idx]
                shard_to_col[s['id']] = c
                label = f"{s['name']} ({s['bytes']/(1024**3):.1f}G)"
                row_cells.append(f"{label:<30}")
            else:
                row_cells.append(f"{' ':<30}")
        print(" | ".join(row_cells))

    # --- 2. SHARD-LEVEL CACHE SIMULATION ---
    cache_slots = [None] * num_cols
    hits, compulsory, conflict, total_shard_lookups = 0, 0, 0, 0
    
    total_original_latency = 0.0
    total_actual_latency = 0.0
    total_saved_transfer_time = 0.0
    query_count = 0

    try:
        with open(csv_path, mode='r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                query_count += 1
                raw_ids = row['searched_shard_ids']
                try:
                    searched_ids = ast.literal_eval(raw_ids)
                    if isinstance(searched_ids, int): searched_ids = [searched_ids]
                except:
                    searched_ids = [int(x) for x in re.findall(r'\d+', raw_ids)]

                num_shards_in_query = len(searched_ids)
                if num_shards_in_query == 0: continue

                t_transfer = float(row.get('gpu_transfer_time', 0))
                t_search = float(row.get('gpu_search_time', 0))
                
                # Metric 1: Pure CSV Baseline
                total_original_latency += (t_transfer + t_search)

                # Proportional costs per shard for granular evaluation
                t_transfer_per_shard = t_transfer / num_shards_in_query
                t_search_per_shard = t_search / num_shards_in_query

                for s_id in searched_ids:
                    total_shard_lookups += 1
                    if s_id not in shard_to_col:
                        total_actual_latency += (t_transfer_per_shard + t_search_per_shard)
                        continue

                    col_idx = shard_to_col[s_id]
                    resident = cache_slots[col_idx]

                    if resident == s_id:
                        hits += 1
                        # HIT: Transfer time is saved
                        total_actual_latency += t_search_per_shard
                        total_saved_transfer_time += t_transfer_per_shard
                    else:
                        # MISS: Add both search and transfer
                        if resident is None: compulsory += 1
                        else: conflict += 1
                        
                        total_actual_latency += (t_transfer_per_shard + t_search_per_shard)
                        cache_slots[col_idx] = s_id

        print(f"\n--- Performance Analysis (1 Row per Column) ---")
        print(f"Trace File:          {csv_path}")
        print(f"Total Shard Lookups: {total_shard_lookups}")
        print(f"Total Hits:          {hits}")
        print(f"Compulsory Misses:   {compulsory}")
        print(f"Conflict Misses:     {conflict}")
        
        if query_count > 0:
            hit_rate = (hits / total_shard_lookups) * 100 if total_shard_lookups > 0 else 0
            avg_orig = total_original_latency / query_count
            avg_actual = total_actual_latency / query_count
            avg_saved = total_saved_transfer_time / query_count
            
            print(f"Shard Hit Rate:      {hit_rate:.2f}%")
            print("-" * 50)
            print(f"Avg Original Latency: {avg_orig:.6f}s (Baseline)")
            print(f"Avg Actual Latency:   {avg_actual:.6f}s (With Caching)")
            print(f"Avg Time Saved:       {avg_saved:.6f}s per query")
            print("-" * 50)
            print(f"Total Trace Savings:  {total_saved_transfer_time:.4f}s")

    except FileNotFoundError:
        print(f"\nError: CSV trace '{csv_path}' not found.")

if __name__ == "__main__":
    SHARD_DIR = "/data/indices/hydra/shards/"
    trace_file = sys.argv[1] if len(sys.argv) > 1 else "data/hydra_analysis/1000_nlist_indices/hydra_analysis_centroids_10.csv"
    run_simulation(SHARD_DIR, trace_file)