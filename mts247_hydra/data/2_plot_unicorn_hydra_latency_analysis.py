#!/usr/bin/env python3
import argparse
import csv
import glob
import math
import os
import re
from typing import Dict, List, Tuple


DEFAULT_OUTPUT = "2_unicorn_hydra_latency_100_nlist.png"


def parse_float(raw: str) -> float:
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def detect_column(headers: List[str], candidates: List[str], source: str) -> str:
    lower_map = {h.lower(): h for h in headers}
    for candidate in candidates:
        if candidate in headers:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    raise ValueError(f"{source}: could not find required column from {candidates}. Found {headers}")


def read_avg_latency(csv_path: str) -> float:
    with open(csv_path, newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        rows = list(reader)

    if not rows:
        raise ValueError(f"{csv_path}: file has no rows")

    headers = list(rows[0].keys())
    transfer_col = detect_column(headers, ["gpu_transfer_time", "GPU Transfer Time", "gpu transfer time"], csv_path)
    search_col = detect_column(headers, ["gpu_search_time", "GPU Search Time", "gpu search time"], csv_path)

    values: List[float] = []
    for row in rows:
        transfer = parse_float(row.get(transfer_col, ""))
        search = parse_float(row.get(search_col, ""))
        if math.isnan(transfer) or math.isnan(search):
            continue
        values.append(transfer + search)

    if not values:
        return 0.0
    return sum(values) / len(values)


def discover_centroid_files(folder: str) -> List[Tuple[int, str]]:
    files = []
    for path in glob.glob(os.path.join(folder, "hydra_analysis_centroids_*.csv")):
        basename = os.path.basename(path)
        match = re.search(r"hydra_analysis_centroids_(\d+)\.csv$", basename)
        if match:
            files.append((int(match.group(1)), path))
    return sorted(files, key=lambda x: x[0])


def system_folder_map(base_dir: str) -> Dict[str, str]:
    return {
        "acadia": os.path.join(base_dir, "unicorn_hydra_analysis", "acadia_1000_nlist_indices"),
        "rocky": os.path.join(base_dir, "unicorn_hydra_analysis", "rocky_500_nlist_indices"),
        "gh200": os.path.join(base_dir, "gh200_hydra_analysis", "100_nlist_indices"),
    }


def extract_nlist_size(folder_path: str) -> str:
    folder_name = os.path.basename(folder_path.rstrip(os.sep))
    match = re.search(r"(\d+)_nlist_indices", folder_name)
    return match.group(1) if match else "unknown"


def plot_latency(
    system_series: Dict[str, List[Tuple[int, float]]],
    system_folders: Dict[str, str],
    output_path: str,
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#444444",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "text.color": "#222222",
    })

    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    color_map = {
        "acadia": "#4E79A7",
        "rocky": "#F28E2B",
        "gh200": "#59A14F",
    }
    label_map = {
        "acadia": "Acadia (A6000 Ada)",
        "rocky": "Rocky (L4)",
        "gh200": "GH200",
    }

    for system in ["acadia", "rocky", "gh200"]:
        points = system_series.get(system, [])
        if not points:
            continue
        x = [centroids for centroids, _ in points]
        y = [latency for _, latency in points]
        nlist_size = extract_nlist_size(system_folders.get(system, ""))
        legend_label = f"{label_map.get(system, system)} ({nlist_size} nlist indices)"
        ax.plot(x, y, linewidth=2.5, label=legend_label, color=color_map.get(system), alpha=0.9)
        ax.scatter(x, y, s=50, color=color_map.get(system), edgecolors="white", linewidth=0.9, zorder=3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#dddddd")
    ax.spines["bottom"].set_color("#dddddd")
    ax.yaxis.grid(True, linestyle="-", color="#f0f0f0")
    ax.xaxis.grid(True, linestyle="-", color="#f0f0f0")

    ax.set_xlabel("Number of Centroids Searched", fontsize=12)
    ax.set_ylabel("Average Latency (Seconds)", fontsize=12)

    legend = ax.legend(frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color("#444444")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved plot: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Hydra latency using acadia/rocky data from unicorn_hydra_analysis "
            "and gh200 data from gh200_hydra_analysis/100_nlist_indices."
        )
    )
    parser.add_argument(
        "--base-dir",
        default=os.path.dirname(os.path.abspath(__file__)),
        help="Base data directory that contains unicorn_hydra_analysis and gh200_hydra_analysis",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output PNG path",
    )
    args = parser.parse_args()

    folders = system_folder_map(args.base_dir)

    missing_systems = [name for name, folder in folders.items() if not os.path.isdir(folder)]
    if missing_systems:
        missing_info = ", ".join(f"{name} -> {folders[name]}" for name in missing_systems)
        raise FileNotFoundError(
            "Missing required 100_nlist_indices folder(s): " + missing_info
        )

    series: Dict[str, List[Tuple[int, float]]] = {}
    for system, folder in folders.items():
        centroid_files = discover_centroid_files(folder)
        if not centroid_files:
            raise FileNotFoundError(f"No hydra_analysis_centroids_*.csv files found in {folder}")

        points: List[Tuple[int, float]] = []
        for centroids, csv_path in centroid_files:
            avg_latency = read_avg_latency(csv_path)
            points.append((centroids, avg_latency))
        series[system] = points

    plot_latency(series, folders, args.output)


if __name__ == "__main__":
    main()
