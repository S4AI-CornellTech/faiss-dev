#!/usr/bin/env python3
import argparse
import ast
import csv
import glob
import math
import os
import re
from typing import Dict, List, Tuple, Any


def parse_ids(raw: Any) -> List[int]:
    if raw is None:
        return []
    text = str(raw).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return []

    def to_int_list(values: List[Any]) -> List[int]:
        out: List[int] = []
        for value in values:
            if value is None:
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                out.append(value)
                continue
            if isinstance(value, float):
                if math.isnan(value):
                    continue
                out.append(int(value))
                continue
            s = str(value).strip()
            if not s:
                continue
            try:
                out.append(int(float(s)))
            except ValueError:
                pass
        return out

    if not (text.startswith("[") and text.endswith("]")):
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return to_int_list(parsed)


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def detect_column(rows: List[Dict[str, str]], candidates: List[str], role: str, source_name: str) -> str:
    if not rows:
        raise ValueError(f"{source_name}: CSV has no rows, cannot detect {role} column")
    columns = list(rows[0].keys())
    lower_to_actual = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lower_to_actual:
            return lower_to_actual[candidate.lower()]
    raise ValueError(
        f"{source_name}: could not detect {role} column. "
        f"Tried {candidates}, found columns: {columns}"
    )


def parse_float(raw: Any) -> float:
    if raw is None:
        return float("nan")
    text = str(raw).strip()
    if text == "" or text.lower() in {"nan", "none", "null"}:
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def average_latency_monolithic(rows: List[Dict[str, str]], source_name: str) -> Tuple[float, str]:
    latency_col = detect_column(rows, ["avg_query_time", "AvgQueryTime"], "latency", source_name)
    values = [parse_float(r.get(latency_col, "")) for r in rows]
    valid = [v for v in values if not math.isnan(v)]
    avg = (sum(valid) / len(valid)) if valid else 0.0
    return avg, latency_col


def average_latency_hydra(rows: List[Dict[str, str]], source_name: str) -> Tuple[float, float, float, str, str]:
    transfer_col = detect_column(
        rows,
        ["gpu_transfer_time", "GPU Transfer Time", "gpu transfer time"],
        "gpu transfer time",
        source_name,
    )
    search_col = detect_column(
        rows,
        ["gpu_search_time", "GPU Search Time", "gpu search time"],
        "gpu search time",
        source_name,
    )
    transfer_values: List[float] = []
    search_values: List[float] = []
    latencies: List[float] = []
    for row in rows:
        t_transfer = parse_float(row.get(transfer_col, ""))
        t_search = parse_float(row.get(search_col, ""))
        if math.isnan(t_transfer) or math.isnan(t_search):
            continue
        transfer_values.append(t_transfer)
        search_values.append(t_search)
        latencies.append(t_transfer + t_search)
    avg = (sum(latencies) / len(latencies)) if latencies else 0.0
    avg_transfer = (sum(transfer_values) / len(transfer_values)) if transfer_values else 0.0
    avg_search = (sum(search_values) / len(search_values)) if search_values else 0.0
    return avg, avg_transfer, avg_search, transfer_col, search_col


def build_row_map(rows: List[Dict[str, str]], query_col: str) -> Dict[str, Dict[str, str]]:
    mapped: Dict[str, Dict[str, str]] = {}
    for row in rows:
        query = row.get(query_col, "")
        if query is None:
            query = ""
        key = str(query).strip()
        if key != "":
            mapped[key] = row
    return mapped


def compute_recall(
    ground_truth_rows: List[Dict[str, str]],
    hydra_rows: List[Dict[str, str]],
    query_col: str,
    gt_col: str,
    hydra_col: str,
    compare_top_k: int = 0,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    per_query: List[Dict[str, Any]] = []
    gt_map = build_row_map(ground_truth_rows, query_col)
    hydra_map = build_row_map(hydra_rows, query_col)
    use_query_join = len(gt_map) > 0 and len(hydra_map) > 0

    if use_query_join:
        shared_queries = sorted(set(gt_map.keys()).intersection(hydra_map.keys()))
        for q in shared_queries:
            gt_ids = parse_ids(gt_map[q].get(gt_col, ""))
            pred_ids = parse_ids(hydra_map[q].get(hydra_col, ""))
            if compare_top_k > 0:
                gt_ids = gt_ids[:compare_top_k]
                pred_ids = pred_ids[:compare_top_k]
            gt_set = set(gt_ids)
            pred_set = set(pred_ids)
            recall = (len(gt_set & pred_set) / len(gt_set)) if gt_set else 0.0
            hit = 1 if len(gt_set & pred_set) > 0 else 0
            per_query.append({
                "query": q,
                "ground_truth_count": len(gt_set),
                "predicted_count": len(pred_set),
                "intersection_count": len(gt_set & pred_set),
                "recall": recall,
                "hit": hit,
            })
    else:
        pair_count = min(len(ground_truth_rows), len(hydra_rows))
        for i in range(pair_count):
            gt_ids = parse_ids(ground_truth_rows[i].get(gt_col, ""))
            pred_ids = parse_ids(hydra_rows[i].get(hydra_col, ""))
            if compare_top_k > 0:
                gt_ids = gt_ids[:compare_top_k]
                pred_ids = pred_ids[:compare_top_k]
            gt_set = set(gt_ids)
            pred_set = set(pred_ids)
            recall = (len(gt_set & pred_set) / len(gt_set)) if gt_set else 0.0
            hit = 1 if len(gt_set & pred_set) > 0 else 0
            per_query.append({
                "query": str(i),
                "ground_truth_count": len(gt_set),
                "predicted_count": len(pred_set),
                "intersection_count": len(gt_set & pred_set),
                "recall": recall,
                "hit": hit,
            })

    evaluated = len(per_query)
    avg_recall = sum(x["recall"] for x in per_query) / evaluated if evaluated else 0.0
    hit_rate = sum(x["hit"] for x in per_query) / evaluated if evaluated else 0.0
    summary = {
        "evaluated_queries": float(evaluated),
        "avg_recall": avg_recall,
        "hit_rate": hit_rate,
    }
    return per_query, summary


def write_per_query(path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["query", "ground_truth_count", "predicted_count", "intersection_count", "recall", "hit"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def discover_fine_grained_files(directory: str) -> List[Tuple[int, str]]:
    pattern = os.path.join(directory, "hydra_analysis_centroids_*.csv")
    found: List[Tuple[int, str]] = []
    for path in glob.glob(pattern):
        m = re.search(r"hydra_analysis_centroids_(\d+)\.csv$", os.path.basename(path))
        if m:
            found.append((int(m.group(1)), path))
    found.sort(key=lambda x: x[0])
    return found


def discover_fine_grained_sweeps(base_dir: str) -> List[Tuple[str, List[Tuple[int, str]]]]:
    if not os.path.isdir(base_dir):
        return []

    sweeps: List[Tuple[str, List[Tuple[int, str]]]] = []
    for entry in sorted(os.listdir(base_dir)):
        folder_path = os.path.join(base_dir, entry)
        if not os.path.isdir(folder_path):
            continue
        files = discover_fine_grained_files(folder_path)
        if files:
            sweeps.append((entry, files))
    return sweeps


def evaluate_system(
    system_name: str,
    system_path: str,
    gt_rows: List[Dict[str, str]],
    gt_query_col: str,
    gt_ids_col: str,
    compare_top_k: int,
    output_prefix: str,
    is_monolithic: bool = False,
) -> Dict[str, Any]:
    system_rows = read_csv(system_path)
    system_query_col = detect_column(system_rows, ["query"], "query", system_name)
    system_ids_col = detect_column(system_rows, ["best_retrieved_ids"], "retrieved ids", system_name)

    if gt_query_col != system_query_col:
        gt_rows_norm = [{**r, "query": r.get(gt_query_col, "")} for r in gt_rows]
        system_rows_norm = [{**r, "query": r.get(system_query_col, "")} for r in system_rows]
        per_query, summary = compute_recall(
            ground_truth_rows=gt_rows_norm,
            hydra_rows=system_rows_norm,
            query_col="query",
            gt_col=gt_ids_col,
            hydra_col=system_ids_col,
            compare_top_k=compare_top_k,
        )
    else:
        per_query, summary = compute_recall(
            ground_truth_rows=gt_rows,
            hydra_rows=system_rows,
            query_col=gt_query_col,
            gt_col=gt_ids_col,
            hydra_col=system_ids_col,
            compare_top_k=compare_top_k,
        )

    avg_transfer = 0.0
    avg_search = 0.0
    if is_monolithic:
        avg_latency, _ = average_latency_monolithic(system_rows, system_name)
    else:
        avg_latency, avg_transfer, avg_search, _, _ = average_latency_hydra(system_rows, system_name)

    if output_prefix:
        output_path = f"{output_prefix}_{system_name.lower()}.csv"
        write_per_query(output_path, per_query)

    return {
        "system": system_name,
        "avg_latency": avg_latency,
        "avg_gpu_transfer_time": avg_transfer,
        "avg_gpu_search_time": avg_search,
        **summary,
    }


def plot_pareto_accuracy_latency(
    monolithic_result: Dict[str, Any],
    sweep_group_summaries: List[Tuple[str, List[Dict[str, Any]]]],
    output_path: str,
) -> None:
    import importlib
    plt = importlib.import_module("matplotlib.pyplot")

    # Set the overall aesthetic to a clean white background
    plt.style.use('seaborn-v0_8-white')
    
    # Use a more modern font if available, fallback to sans-serif
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor": "#444444",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "text.color": "#222222"
    })

    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

    # A more sophisticated, "muted" color palette
    colors = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F"]

    # 1. Plot Sweep Groups with Smooth Lines
    for i, (folder_name, summaries) in enumerate(sweep_group_summaries):
        ordered = sorted(summaries, key=lambda x: x["avg_latency"])
        x = [s["avg_latency"] for s in ordered]
        y = [s["avg_recall"] for s in ordered]
        
        color = colors[i % len(colors)]
        
        # Plot line and points separately for more control
        ax.plot(x, y, color=color, linewidth=3, alpha=0.7, label=folder_name, zorder=3)
        ax.scatter(x, y, color=color, s=60, edgecolors='white', linewidth=1, zorder=4)

    # 2. Highlight the Monolithic Baseline (The "Anchor")
    ax.scatter(
        [monolithic_result["avg_latency"]],
        [monolithic_result["avg_recall"]],
        marker="*",
        s=250,
        color="#333333",
        edgecolor="#ffffff",
        linewidth=1.5,
        label="Monolithic Baseline",
        zorder=6,
        # Add a subtle "shadow" effect via a second larger scatter
        alpha=1.0
    )

    # 3. Aesthetics & Minimalist Grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#dddddd')
    ax.spines['bottom'].set_color('#dddddd')
    
    # Use a very light gray grid for the "y" axis only
    ax.yaxis.grid(True, linestyle='-', color='#f0f0f0', zorder=1)
    ax.xaxis.grid(True)
    ax.set_xlim([0, 5])

    # 4. Refined Labels (No Title)
    ax.set_xlabel("Latency (Seconds)", fontsize=14, labelpad=15, fontweight='300')
    ax.set_ylabel("Recall Accuracy", fontsize=14, labelpad=15, fontweight='300')

    # 5. Clean Legend
    legend = ax.legend(
        loc="lower right", 
        frameon=False, 
        fontsize=11, 
        handletextpad=0.5
    )
    for text in legend.get_texts():
        text.set_color("#444444")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', transparent=False)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute recall for monolithic baseline and fine-grained hydra sweep vs ground truth."
    )
    parser.add_argument("--ground-truth", default="hydra_monolithic_ground_truth.csv")
    parser.add_argument("--monolithic", default="hydra_baseline_bs_1.csv")
    parser.add_argument("--output-per-query-prefix", default="")
    parser.add_argument(
        "--pareto-output",
        default="pareto_accuracy_latency.png",
        help="Output image path for latency-vs-accuracy pareto plot (empty string disables plotting)",
    )
    parser.add_argument(
        "--compare-top-k",
        type=int,
        default=0,
        help="Only compare top-k docs from each retrieved-id list (0 = full list)",
    )
    args = parser.parse_args()

    gt_rows = read_csv(args.ground_truth)
    gt_query_col = detect_column(gt_rows, ["query", "Query"], "query", "ground_truth")
    gt_ids_col = detect_column(gt_rows, ["best_retrieved_ids", "TopDocs", "top_docs"], "retrieved ids", "ground_truth")

    compare_scope = f"top-{args.compare_top_k}" if args.compare_top_k > 0 else "full-list"

    hydra_root_dir = "hydra_analysis"
    sweep_groups = discover_fine_grained_sweeps(hydra_root_dir)

    # --- Monolithic baseline ---
    mono_result = evaluate_system(
        system_name="Monolithic",
        system_path=args.monolithic,
        gt_rows=gt_rows,
        gt_query_col=gt_query_col,
        gt_ids_col=gt_ids_col,
        compare_top_k=args.compare_top_k,
        output_prefix=args.output_per_query_prefix,
        is_monolithic=True,
    )

    # --- Monolithic summary table ---
    print("Recall Comparison")
    print("=================")
    print(f"Ground truth: {args.ground_truth} (query={gt_query_col}, ids={gt_ids_col})")
    print(f"Comparison scope: {compare_scope}")
    print(f"Hydra sweep root: {hydra_root_dir}")
    print()
    print("Monolithic Baseline")
    print("-" * 55)
    print(f"  {'system':<20}  {'avg_latency_s':>14}  {'avg_recall':>11}  {'hit_rate':>9}  {'queries':>8}")
    print(f"  {'-'*20}  {'-'*14}  {'-'*11}  {'-'*9}  {'-'*8}")
    print(
        f"  {mono_result['system']:<20}  "
        f"{mono_result['avg_latency']:>14.6f}  "
        f"{mono_result['avg_recall']:>11.6f}  "
        f"{mono_result['hit_rate']:>9.6f}  "
        f"{int(mono_result['evaluated_queries']):>8d}"
    )

    # --- Sweep summary tables (one table per hydra_analysis subfolder) ---
    if sweep_groups:
        all_sweep_group_summaries: List[Tuple[str, List[Dict[str, Any]]]] = []
        for folder_name, sweep_files in sweep_groups:
            sweep_summaries = []
            for num_centroids, sweep_path in sweep_files:
                result = evaluate_system(
                    system_name=f"FineGrainedHydra_{folder_name}_centroids_{num_centroids}",
                    system_path=sweep_path,
                    gt_rows=gt_rows,
                    gt_query_col=gt_query_col,
                    gt_ids_col=gt_ids_col,
                    compare_top_k=args.compare_top_k,
                    output_prefix=args.output_per_query_prefix,
                    is_monolithic=False,
                )
                result["num_centroids"] = num_centroids
                sweep_summaries.append(result)

            print()
            print("=" * 70)
            print(f"Fine-Grained Hydra Sweep Summary ({folder_name})")
            print("=" * 70)
            print(
                f"  {'num_centroids':>14}  {'avg_transfer_s':>14}  {'avg_search_s':>14}  "
                f"{'avg_latency_s':>14}  {'avg_recall':>11}  {'hit_rate':>9}  {'queries':>8}"
            )
            print(f"  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*14}  {'-'*11}  {'-'*9}  {'-'*8}")
            for s in sweep_summaries:
                print(
                    f"  {s['num_centroids']:>14d}  "
                    f"{s['avg_gpu_transfer_time']:>14.6f}  "
                    f"{s['avg_gpu_search_time']:>14.6f}  "
                    f"{s['avg_latency']:>14.6f}  "
                    f"{s['avg_recall']:>11.6f}  "
                    f"{s['hit_rate']:>9.6f}  "
                    f"{int(s['evaluated_queries']):>8d}"
                )
            print("=" * 70)

            all_sweep_group_summaries.append((folder_name, sweep_summaries))

        if args.pareto_output:
            try:
                plot_pareto_accuracy_latency(
                    monolithic_result=mono_result,
                    sweep_group_summaries=all_sweep_group_summaries,
                    output_path=args.pareto_output,
                )
                print(f"\nSaved pareto plot: {args.pareto_output}")
            except ImportError:
                print("\nWarning: matplotlib is not installed, skipping pareto plot generation")
    else:
        print(f"\nWarning: no hydra_analysis_centroids_*.csv files found under '{hydra_root_dir}'")


if __name__ == "__main__":
    main()