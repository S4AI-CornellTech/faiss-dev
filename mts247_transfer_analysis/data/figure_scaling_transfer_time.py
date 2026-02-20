#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================
# Styling (clean + publication friendly)
# ==============================================================

plt.style.use("seaborn-v0_8-whitegrid")

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# ==============================================================
# Load Data
# ==============================================================

FILE = "enhanced_transfer_master_file.csv"
df = pd.read_csv(FILE)

df["index_size_m"] = df["index_size"].str.replace("m", "").astype(int)
df["cpu_to_gpu_time"] = df["cpu_to_gpu_time"].astype(str).str.replace("s", "").astype(float)
df["disk_to_cpu_time"] = df["disk_to_cpu_time"].astype(str).str.replace("s", "").astype(float)

df = df.sort_values("index_size_m")

# ==============================================================
# Extract columns
# ==============================================================

sizes = df["index_size_m"].values
cpu_retrieval = df["cpu_retrieval_time"].values
cpu_to_gpu = df["cpu_to_gpu_time"].values
gpu_retrieval = df["gpu_retrieval_time"].values
gpu_total = cpu_to_gpu + gpu_retrieval

# ==============================================================
# Plot
# ==============================================================

fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 6

# Colors
cpu_gpu_color = "#4C72B0"
gpu_ret_color = "#55A868"
cpu_line_color = "#C44E52"
gpu_total_color = "#8172B2"

# Stacked bars
ax.bar(
    sizes,
    cpu_to_gpu,
    width=bar_width,
    alpha=0.8,
    label="CPU → GPU Transfer",
    color=cpu_gpu_color,
)

ax.bar(
    sizes,
    gpu_retrieval,
    width=bar_width,
    bottom=cpu_to_gpu,
    alpha=0.8,
    label="GPU Retrieval",
    color=gpu_ret_color,
)

# Lines
ax.plot(
    sizes,
    cpu_retrieval,
    marker="o",
    linewidth=2.5,
    label="CPU Retrieval Time",
    color=cpu_line_color,
)

ax.plot(
    sizes,
    gpu_total,
    marker="o",
    linewidth=2.5,
    linestyle="--",
    label="GPU Total Retrieval Time",
    color=gpu_total_color,
)

# Labels and title
ax.set_xlabel("Datastore Size (Millions of Vectors)")
ax.set_ylabel("Latency (seconds)")
ax.set_title("Latency Scaling with Datastore Size")

# Remove top/right borders
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Cleaner legend
ax.legend(frameon=False, loc="upper left")

plt.tight_layout()
plt.savefig("figure_scaling_transfer_time.png", dpi=300, bbox_inches="tight")
