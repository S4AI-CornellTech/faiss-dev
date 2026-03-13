import pandas as pd
import matplotlib.pyplot as plt
import re
import os

# --- 1. Load and Clean Data ---
# Update this path if your CSV is in a different subdirectory
csv_path = 'cache_simulation_summary.csv'

if not os.path.exists(csv_path):
    # Try searching in 'data/' if not in root
    csv_path = os.path.join('data', 'cache_simulation_summary.csv')

if not os.path.exists(csv_path):
    print(f"Error: Could not find {csv_path}")
    exit()

df = pd.read_csv(csv_path)

# Extract numeric 'centroids' from filename (e.g., 'centroids_10.csv' -> 10)
df['centroids'] = df['filename'].str.extract(r'(\d+)').astype(int)
df = df.sort_values('centroids')

# --- 2. Styling Config (Matching your template style) ---
plt.style.use("seaborn-v0_8-white")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": "#444444",
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "text.color": "#222222",
})

# --- 3. Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

# --- Left Plot: Cache Hit Rate ---
ax1.plot(
    df["centroids"], 
    df["hit_rate"], 
    color="#4E79A7", 
    linewidth=3, 
    marker='o', 
    markersize=8, 
    markeredgecolor="white", 
    label='Actual Hit Rate', 
    zorder=3
)

ax1.set_title('Cache Hit Rate Analysis', fontsize=14, fontweight="bold", pad=15)
ax1.set_xlabel("Number of Centroids Searched", fontsize=12, labelpad=10)
ax1.set_ylabel("Hit Rate (%)", fontsize=12, labelpad=10)
ax1.set_ylim(0, max(df["hit_rate"].max() + 5, 20))

# --- Right Plot: Latency Reduction ---
ax2.plot(
    df["centroids"], 
    df["avg_orig_lat"], 
    marker='s', 
    color='#95a5a6', 
    linestyle='--', 
    linewidth=2, 
    label='Baseline (No Cache)', 
    zorder=3
)
ax2.plot(
    df["centroids"], 
    df["avg_actual_lat"], 
    marker='o', 
    color='#59A14F', 
    linewidth=3, 
    markersize=8, 
    markeredgecolor="white", 
    label='Hydra Performance (With Cache)', 
    zorder=4
)

ax2.set_title('End-to-End Latency Reduction', fontsize=14, fontweight="bold", pad=15)
ax2.set_xlabel("Number of Centroids Searched", fontsize=12, labelpad=10)
ax2.set_ylabel("Avg Query Latency (Seconds)", fontsize=12, labelpad=10)

# --- Common Formatting (Spines and Grids) ---
for ax in [ax1, ax2]:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#dddddd")
    ax.spines["bottom"].set_color("#dddddd")
    
    # Grid lines behind the data
    ax.yaxis.grid(True, linestyle="-", color="#f0f0f0", zorder=1)
    ax.xaxis.grid(True, linestyle="-", color="#f0f0f0", zorder=1)
    
    # Legend formatting
    legend = ax.legend(frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color("#444444")

plt.tight_layout()
output_name = 'simulated_caching_data.png'
plt.savefig(output_name, bbox_inches="tight")
print(f"Successfully generated: {output_name}")