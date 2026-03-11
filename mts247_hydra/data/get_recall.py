#!/usr/bin/env python3
import ast
import csv
import glob
import math
import os
import re
from typing import Any, Dict, List, Tuple

# --- Config ---
GROUND_TRUTH_PATH = "hydra_monolithic_ground_truth.csv"
MONOLITHIC_PATH = "hydra_baseline_bs_1.csv"
HYDRA_ROOT_DIR = "hydra_analysis"
OUTPUT_PER_QUERY_PREFIX = ""
PARETO_OUTPUT = "pareto_accuracy_latency.png"
STACKED_BAR_OUTPUT = "stacked_bar_latency_nlists_1000.png"
STACKED_BAR_FOLDER = "1000_nlist_indices"


# --- Parsing helpers ---

def parse_ids(raw: Any) -> List[int]:
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    if not (text.startswith("[") and text.endswith("]")):
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []

    out = []
    for v in parsed:
        if v is None or isinstance(v, bool):
            continue
        if isinstance(v, float):
            if math.isnan(v):
                continue
            out.append(int(v))
        elif isinstance(v, int):
            out.append(v)
        else:
            s = str(v).strip()
            try:
                out.append(int(float(s)))
            except ValueError:
                pass
    return out


def parse_float(raw: Any) -> float:
    if raw is None:
        return float("nan")
    text = str(raw).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


# --- I/O ---

def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_per_query(path: str, rows: List[Dict[str, Any]]) -> None:
    fields = ["query", "ground_truth_count", "predicted_count", "intersection_count", "recall", "hit"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# --- Column detection ---

def detect_column(rows: List[Dict[str, str]], candidates: List[str], role: str, source: str) -> str:
    if not rows:
        raise ValueError(f"{source}: CSV has no rows, cannot detect {role} column")
    cols = list(rows[0].keys())
    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c in cols:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    raise ValueError(f"{source}: could not detect {role} column. Tried {candidates}, found: {cols}")


# --- Latency ---

def average_latency_monolithic(rows: List[Dict[str, str]], source: str) -> float:
    col = detect_column(rows, ["avg_query_time", "AvgQueryTime"], "latency", source)
    values = [parse_float(r.get(col, "")) for r in rows]
    valid = [v for v in values if not math.isnan(v)]
    return sum(valid) / len(valid) if valid else 0.0


def average_latency_hydra(rows: List[Dict[str, str]], source: str) -> Tuple[float, float, float]:
    t_col = detect_column(rows, ["gpu_transfer_time", "GPU Transfer Time", "gpu transfer time"], "gpu transfer time", source)
    s_col = detect_column(rows, ["gpu_search_time", "GPU Search Time", "gpu search time"], "gpu search time", source)

    transfers, searches = [], []
    for row in rows:
        t = parse_float(row.get(t_col, ""))
        s = parse_float(row.get(s_col, ""))
        if not (math.isnan(t) or math.isnan(s)):
            transfers.append(t)
            searches.append(s)

    n = len(transfers)
    avg_t = sum(transfers) / n if n else 0.0
    avg_s = sum(searches) / n if n else 0.0
    return avg_t + avg_s, avg_t, avg_s


# --- Recall ---

def compute_recall(
    gt_rows: List[Dict[str, str]],
    pred_rows: List[Dict[str, str]],
    query_col: str,
    gt_col: str,
    pred_col: str,
    top_k: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    def get_ids(row, col):
        ids = parse_ids(row.get(col, ""))
        return ids[:top_k] if top_k > 0 else ids

    gt_map = {str(r.get(query_col, "")).strip(): r for r in gt_rows if str(r.get(query_col, "")).strip()}
    pred_map = {str(r.get(query_col, "")).strip(): r for r in pred_rows if str(r.get(query_col, "")).strip()}

    per_query = []
    if gt_map and pred_map:
        pairs = [(q, gt_map[q], pred_map[q]) for q in sorted(gt_map.keys() & pred_map.keys())]
    else:
        pairs = [(str(i), gt_rows[i], pred_rows[i]) for i in range(min(len(gt_rows), len(pred_rows)))]

    for label, gt_row, pred_row in pairs:
        gt_set = set(get_ids(gt_row, gt_col))
        pred_set = set(get_ids(pred_row, pred_col))
        overlap = len(gt_set & pred_set)
        per_query.append({
            "query": label,
            "ground_truth_count": len(gt_set),
            "predicted_count": len(pred_set),
            "intersection_count": overlap,
            "recall": overlap / len(gt_set) if gt_set else 0.0,
            "hit": int(overlap > 0),
        })

    n = len(per_query)
    return per_query, {
        "evaluated_queries": float(n),
        "avg_recall": sum(x["recall"] for x in per_query) / n if n else 0.0,
        "hit_rate": sum(x["hit"] for x in per_query) / n if n else 0.0,
    }


# --- File discovery ---

def discover_fine_grained_sweeps(base_dir: str) -> List[Tuple[str, List[Tuple[int, str]]]]:
    if not os.path.isdir(base_dir):
        return []
    sweeps = []
    for entry in sorted(os.listdir(base_dir)):
        folder = os.path.join(base_dir, entry)
        if not os.path.isdir(folder):
            continue
        files = sorted(
            (int(m.group(1)), p)
            for p in glob.glob(os.path.join(folder, "hydra_analysis_centroids_*.csv"))
            if (m := re.search(r"hydra_analysis_centroids_(\d+)\.csv$", os.path.basename(p)))
        )
        if files:
            sweeps.append((entry, files))
    return sweeps


# --- Evaluation ---

def evaluate_system(
    name: str,
    path: str,
    gt_rows: List[Dict[str, str]],
    gt_query_col: str,
    gt_ids_col: str,
    top_k: int,
    output_prefix: str,
    is_monolithic: bool = False,
) -> Dict[str, Any]:
    rows = read_csv(path)
    q_col = detect_column(rows, ["query"], "query", name)
    ids_col = detect_column(rows, ["best_retrieved_ids"], "retrieved ids", name)

    # Normalize query column name if needed
    if gt_query_col != q_col:
        gt_rows = [{**r, "query": r.get(gt_query_col, "")} for r in gt_rows]
        rows = [{**r, "query": r.get(q_col, "")} for r in rows]
        q_col = gt_query_col = "query"

    per_query, summary = compute_recall(gt_rows, rows, gt_query_col, gt_ids_col, ids_col, top_k)

    avg_transfer = avg_search = 0.0
    if is_monolithic:
        avg_latency = average_latency_monolithic(rows, name)
    else:
        avg_latency, avg_transfer, avg_search = average_latency_hydra(rows, name)

    if output_prefix:
        write_per_query(f"{output_prefix}_{name.lower()}.csv", per_query)

    return {"system": name, "avg_latency": avg_latency,
            "avg_gpu_transfer_time": avg_transfer, "avg_gpu_search_time": avg_search, **summary}


# --- Plotting ---

def plot_pareto_accuracy_latency(
    mono: Dict[str, Any],
    sweep_groups: List[Tuple[str, List[Dict[str, Any]]]],
    output_path: str,
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#444444", "xtick.color": "#444444",
        "ytick.color": "#444444", "text.color": "#222222",
    })

    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

    for i, (name, summaries) in enumerate(sweep_groups):
        ordered = sorted(summaries, key=lambda x: x["avg_latency"])
        x = [s["avg_latency"] for s in ordered]
        y = [s["avg_recall"] for s in ordered]
        c = colors[i % len(colors)]
        ax.plot(x, y, color=c, linewidth=3, alpha=0.7, label=name, zorder=3)
        ax.scatter(x, y, color=c, s=60, edgecolors="white", linewidth=1, zorder=4)

    ax.scatter([mono["avg_latency"]], [mono["avg_recall"]],
               marker="*", s=250, color="#333333", edgecolor="#ffffff",
               linewidth=1.5, label="Monolithic Baseline", zorder=6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#dddddd")
    ax.spines["bottom"].set_color("#dddddd")
    ax.yaxis.grid(True, linestyle="-", color="#f0f0f0", zorder=1)
    ax.xaxis.grid(True)
    ax.set_xlim([0, 5])
    ax.set_xlabel("Latency (Seconds)", fontsize=14, labelpad=15, fontweight="300")
    ax.set_ylabel("Recall Accuracy", fontsize=14, labelpad=15, fontweight="300")

    legend = ax.legend(loc="lower right", frameon=False, fontsize=11, handletextpad=0.5)
    for t in legend.get_texts():
        t.set_color("#444444")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved pareto plot: {output_path}")


def plot_stacked_bar_latency(
    sweep_groups: List[Tuple[str, List[Dict[str, Any]]]],
    output_path: str,
    target_folder: str = "nlists_1000",
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    summaries = next((s for name, s in sweep_groups if target_folder in name), None)
    if not summaries:
        print(f"Warning: no sweep group matching '{target_folder}' found, skipping stacked bar chart")
        return

    ordered = sorted(summaries, key=lambda x: x["num_centroids"])
    labels = [str(s["num_centroids"]) for s in ordered]
    transfer = np.array([s["avg_gpu_transfer_time"] for s in ordered])
    search = np.array([s["avg_gpu_search_time"] for s in ordered])

    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#444444", "xtick.color": "#444444",
        "ytick.color": "#444444", "text.color": "#222222",
    })

    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    x = np.arange(len(labels))

    ax.bar(x, transfer, 0.55, color="#4E79A7", alpha=0.85, label="GPU Transfer Time", zorder=3)
    ax.bar(x, search, 0.55, bottom=transfer, color="#F28E2B", alpha=0.85, label="GPU Search Time", zorder=3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#dddddd")
    ax.spines["bottom"].set_color("#dddddd")
    ax.yaxis.grid(True, linestyle="-", color="#f0f0f0", zorder=1)
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Number of Centroids Searched", fontsize=14, labelpad=15, fontweight="300")
    ax.set_ylabel("Latency (Seconds)", fontsize=14, labelpad=15, fontweight="300")

    legend = ax.legend(frameon=False, fontsize=11, handletextpad=0.5)
    for t in legend.get_texts():
        t.set_color("#444444")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved stacked bar chart ({target_folder}): {output_path}")


# --- Main ---

def main(compare_top_k: int = 0) -> None:
    gt_rows = read_csv(GROUND_TRUTH_PATH)
    gt_query_col = detect_column(gt_rows, ["query", "Query"], "query", "ground_truth")
    gt_ids_col = detect_column(gt_rows, ["best_retrieved_ids", "TopDocs", "top_docs"], "retrieved ids", "ground_truth")
    compare_scope = f"top-{compare_top_k}" if compare_top_k > 0 else "full-list"

    mono = evaluate_system(
        name="Monolithic", path=MONOLITHIC_PATH,
        gt_rows=gt_rows, gt_query_col=gt_query_col, gt_ids_col=gt_ids_col,
        top_k=compare_top_k, output_prefix=OUTPUT_PER_QUERY_PREFIX, is_monolithic=True,
    )

    print("Recall Comparison")
    print("=================")
    print(f"Ground truth: {GROUND_TRUTH_PATH} (query={gt_query_col}, ids={gt_ids_col})")
    print(f"Comparison scope: {compare_scope}")
    print(f"Hydra sweep root: {HYDRA_ROOT_DIR}\n")
    print("Monolithic Baseline")
    print("-" * 55)
    print(f"  {'system':<20}  {'avg_latency_s':>14}  {'avg_recall':>11}  {'hit_rate':>9}  {'queries':>8}")
    print(f"  {'-'*20}  {'-'*14}  {'-'*11}  {'-'*9}  {'-'*8}")
    print(f"  {mono['system']:<20}  {mono['avg_latency']:>14.6f}  "
          f"{mono['avg_recall']:>11.6f}  {mono['hit_rate']:>9.6f}  {int(mono['evaluated_queries']):>8d}")

    sweep_groups = discover_fine_grained_sweeps(HYDRA_ROOT_DIR)
    if not sweep_groups:
        print(f"\nWarning: no hydra_analysis_centroids_*.csv files found under '{HYDRA_ROOT_DIR}'")
        return

    all_group_summaries = []
    for folder_name, sweep_files in sweep_groups:
        summaries = []
        for num_centroids, path in sweep_files:
            result = evaluate_system(
                name=f"FineGrainedHydra_{folder_name}_centroids_{num_centroids}",
                path=path, gt_rows=gt_rows, gt_query_col=gt_query_col, gt_ids_col=gt_ids_col,
                top_k=compare_top_k, output_prefix=OUTPUT_PER_QUERY_PREFIX,
            )
            result["num_centroids"] = num_centroids
            summaries.append(result)

        print(f"\n{'='*70}")
        print(f"Fine-Grained Hydra Sweep Summary ({folder_name})")
        print("=" * 70)
        print(f"  {'num_centroids':>14}  {'avg_transfer_s':>14}  {'avg_search_s':>14}  "
              f"{'avg_latency_s':>14}  {'avg_recall':>11}  {'hit_rate':>9}  {'queries':>8}")
        print(f"  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*11}  {'-'*9}  {'-'*8}")
        for s in summaries:
            print(f"  {s['num_centroids']:>14d}  {s['avg_gpu_transfer_time']:>14.6f}  "
                  f"{s['avg_gpu_search_time']:>14.6f}  {s['avg_latency']:>14.6f}  "
                  f"{s['avg_recall']:>11.6f}  {s['hit_rate']:>9.6f}  {int(s['evaluated_queries']):>8d}")
        print("=" * 70)
        all_group_summaries.append((folder_name, summaries))

    if PARETO_OUTPUT:
        try:
            plot_pareto_accuracy_latency(mono, all_group_summaries, PARETO_OUTPUT)
        except ImportError:
            print("\nWarning: matplotlib not installed, skipping pareto plot")

    if STACKED_BAR_OUTPUT:
        try:
            plot_stacked_bar_latency(all_group_summaries, STACKED_BAR_OUTPUT, STACKED_BAR_FOLDER)
        except ImportError:
            print("\nWarning: matplotlib/numpy not installed, skipping stacked bar chart")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute recall for monolithic and hydra sweep vs ground truth.")
    parser.add_argument("--compare-top-k", type=int, default=0,
                        help="Only compare top-k docs from each retrieved-id list (0 = full list)")
    args = parser.parse_args()
    main(compare_top_k=args.compare_top_k)