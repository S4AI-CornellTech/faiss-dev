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

def natural_sort_key(s):
    """Key for natural sorting: 'centroids_2' comes before 'centroids_10'."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def run_batch_simulation(shard_dir, input_folder, output_csv):
    shards = get_shard_data(shard_dir)
    if not shards:
        print(f"Error: No shards found in {shard_dir}")
        return

    # --- 1. TABLE MAPPING ---
    num_cols, max_gb = 4, 95
    n = len(shards)
    num_rows = math.ceil(n / num_cols)
    while num_rows <= n:
        valid_config = True
        for r in range(num_rows):
            # Calculate row sum in GB
            row_sum_gb = 0
            for c in range(num_cols):
                idx = (c * num_rows) + r
                if idx < n:
                    row_sum_gb += shards[idx]['bytes'] / (1024**3)
            
            if row_sum_gb >= max_gb:
                valid_config = False
                break
        if valid_config: 
            break
        num_rows += 1

    shard_to_col = {}
    for r in range(num_rows):
        for c in range(num_cols):
            idx = (c * num_rows) + r
            if idx < n:
                shard_to_col[shards[idx]['id']] = c

    # --- 2. BATCH PROCESSING ---
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    summary_results = []

    if not os.path.exists(input_folder):
        print(f"Error: Input folder {input_folder} not found.")
        return

    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    csv_files.sort(key=natural_sort_key)
    
    print(f"Processing {len(csv_files)} files...\n")

    for filename in csv_files:
        csv_path = os.path.join(input_folder, filename)
        # Initialize cache: one slot per GPU column
        cache_slots = [None] * num_cols
        
        hits, total_shard_lookups = 0, 0
        total_orig_lat = 0.0
        total_actual_lat = 0.0
        total_saved_time = 0.0
        query_count = 0

        try:
            with open(csv_path, mode='r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    query_count += 1
                    raw_ids = row['searched_shard_ids']
                    try:
                        s_ids = ast.literal_eval(raw_ids)
                        if isinstance(s_ids, int): s_ids = [s_ids]
                    except:
                        s_ids = [int(x) for x in re.findall(r'\d+', raw_ids)]

                    num_shards = len(s_ids)
                    if num_shards == 0: continue

                    t_trans = float(row.get('gpu_transfer_time', 0))
                    t_srch = float(row.get('gpu_search_time', 0))
                    
                    total_orig_lat += (t_trans + t_srch)
                    t_trans_shard = t_trans / num_shards
                    t_srch_shard = t_srch / num_shards

                    for s_id in s_ids:
                        total_shard_lookups += 1
                        col_idx = shard_to_col.get(s_id)
                        
                        # If shard isn't in our mapped columns, it's a mandatory miss/penalty
                        if col_idx is None:
                            total_actual_lat += (t_trans_shard + t_srch_shard)
                            continue

                        # Cache Logic
                        if cache_slots[col_idx] == s_id:
                            hits += 1
                            total_actual_lat += t_srch_shard # No transfer time
                            total_saved_time += t_trans_shard
                        else:
                            total_actual_lat += (t_trans_shard + t_srch_shard)
                            cache_slots[col_idx] = s_id

            if query_count > 0:
                # Calculate hit rate percentage
                hr = (hits / total_shard_lookups * 100) if total_shard_lookups > 0 else 0
                
                res = {
                    'filename': filename,
                    'queries': query_count,
                    'hit_rate': hr,
                    'avg_orig_lat': total_orig_lat / query_count,
                    'avg_actual_lat': total_actual_lat / query_count,
                    'avg_time_saved': total_saved_time / query_count,
                    'total_time_saved': total_saved_time
                }
                summary_results.append(res)
                print(f"Done: {filename:<35} | Hit Rate: {hr:>6.2f}%")

        except Exception as e:
            print(f"Error on {filename}: {e}")

    # --- 3. SAVE SUMMARY ---
    with open(output_csv, mode='w', newline='') as f:
        # Added 'hit_rate' to the fieldnames
        fields = ['filename', 'queries', 'hit_rate', 'avg_orig_lat', 'avg_actual_lat', 'avg_time_saved', 'total_time_saved']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_results)
    
    print(f"\nFinal summary written to: {output_csv}")

if __name__ == "__main__":
    SHARD_DIR = "/data/indices/hydra/shards/"
    INPUT_DIR = "data/hydra_analysis/1000_nlist_indices/"
    OUTPUT_FILE = "data/cache_simulation_summary.csv"
    
    run_batch_simulation(SHARD_DIR, INPUT_DIR, OUTPUT_FILE)