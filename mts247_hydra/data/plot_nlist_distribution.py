#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    data = "nlist_sizes.csv"
    bins = 200
    plot_name = "plot_nlist_distribution.png"

    # ---- Constants ----
    DIM = 768
    BYTES_PER_VECTOR = DIM + 8  # SQ8 (1 byte per dim) + 8 byte id
    BYTES_TO_MB = 1024 ** 2

    # ---- Load CSV ----
    df = pd.read_csv(data)

    if "vector_count" not in df.columns:
        raise ValueError("CSV must contain a 'vector_count' column")

    vector_counts = df["vector_count"].values

    # Convert to MB
    sizes_mb = (vector_counts * BYTES_PER_VECTOR) / BYTES_TO_MB

    # ---- Plot ----
    plt.figure(figsize=(12, 7))

    n, bins_edges, patches = plt.hist(
        vector_counts,
        bins=bins,
        alpha=0.85
    )

    # Grid styling
    plt.grid(True, linestyle="--", alpha=0.3)

    # Mean and P95 lines
    mean_val = np.mean(vector_counts)
    p95_val = np.percentile(vector_counts, 95)

    plt.axvline(mean_val, linestyle="--", linewidth=2, label=f"Mean: {mean_val:,.0f}")
    plt.axvline(p95_val, linestyle=":", linewidth=2, label=f"P95: {p95_val:,.0f}")

    # ---- Custom X-axis labels ----
    # Show vector count + size in MB
    tick_locs = plt.xticks()[0]
    new_labels = []
    for val in tick_locs:
        size_mb = (val * BYTES_PER_VECTOR) / BYTES_TO_MB
        if val > 0:
            new_labels.append(f"{int(val/1000)}k\n({size_mb:.1f} MB)")
        else:
            new_labels.append("")

    plt.xticks(tick_locs, new_labels)

    plt.xlabel("Vectors per nlist (Size in MB)")
    plt.ylabel("Number of nlists")
    plt.title("Distribution of FAISS IVF nlist Sizes\n(768-dim SQ8 Quantized)")

    plt.legend()
    plt.tight_layout()

    plt.savefig(plot_name, dpi=300)
    print(f"Histogram saved to {plot_name}")


if __name__ == "__main__":
    main()