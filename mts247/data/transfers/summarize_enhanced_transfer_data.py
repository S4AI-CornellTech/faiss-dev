#!/usr/bin/env python3
import os
import pandas as pd

BASE_NAME = "ENHANCED_TRANSFER_TIMES_{}_sq8.csv"
SIZES = range(10, 101, 10)  # 10m to 100m

results = []

for size in SIZES:
    size_label = f"{size}m"
    filename = BASE_NAME.format(size_label)

    if not os.path.exists(filename):
        print(f"Skipping missing file: {filename}")
        continue

    df = pd.read_csv(filename)

    # ---- Remove 2 largest cpu_to_gpu values ----
    if len(df) > 2:
        df_filtered = df.sort_values("cpu_to_gpu_s").iloc[:-2]
    else:
        print(f"Not enough rows to remove outliers in {filename}")
        df_filtered = df

    avg_disk_to_cpu = df["disk_to_cpu_s"].mean()  # unchanged
    avg_cpu_to_gpu = df_filtered["cpu_to_gpu_s"].mean()

    results.append({
        "size": size_label,
        "avg_disk_to_cpu_s": avg_disk_to_cpu,
        "avg_cpu_to_gpu_s_filtered": avg_cpu_to_gpu
    })

# Print results
print("\nPer-size averages (2 largest cpu_to_gpu_s removed):\n")
for r in results:
    print(
        f"{r['size']}: "
        f"disk_to_cpu = {r['avg_disk_to_cpu_s']:.6f}s, "
        f"cpu_to_gpu (filtered) = {r['avg_cpu_to_gpu_s_filtered']:.6f}s"
    )

# Optional: save to CSV
pd.DataFrame(results).to_csv("per_size_transfer_averages_filtered.csv", index=False)