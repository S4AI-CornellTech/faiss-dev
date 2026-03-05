#!/usr/bin/env python3
import argparse
import ast
import csv
import math
import re
from typing import Dict, List, Tuple, Any


def parse_ids(raw: Any) -> List[int]:
    """Parse an ID field into a list of ints.

    Supports values like:
    - "123"
    - "[123, 456]"
    - "123,456"
    - "123|456"
    - "123 456"
    """
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

    if text.startswith("[") or text.startswith("("):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return to_int_list(list(parsed))
            return to_int_list([parsed])
        except Exception:
            pass

    parts = re.split(r"[,|;\s]+", text)
    return to_int_list(parts)


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


def average_latency_hydra(rows: List[Dict[str, str]], source_name: str) -> Tuple[float, str, str]:
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

    latencies: List[float] = []
    for row in rows:
        t_transfer = parse_float(row.get(transfer_col, ""))
        t_search = parse_float(row.get(search_col, ""))
        if math.isnan(t_transfer) or math.isnan(t_search):
            continue
        latencies.append(t_transfer + t_search)

    avg = (sum(latencies) / len(latencies)) if latencies else 0.0
    return avg, transfer_col, search_col


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

            gt_set = set(gt_ids)
            pred_set = set(pred_ids)
            recall = (len(gt_set & pred_set) / len(gt_set)) if gt_set else 0.0
            hit = 1 if len(gt_set & pred_set) > 0 else 0

            per_query.append(
                {
                    "query": q,
                    "ground_truth_count": len(gt_set),
                    "predicted_count": len(pred_set),
                    "intersection_count": len(gt_set & pred_set),
                    "recall": recall,
                    "hit": hit,
                }
            )
    else:
        pair_count = min(len(ground_truth_rows), len(hydra_rows))
        for i in range(pair_count):
            gt_ids = parse_ids(ground_truth_rows[i].get(gt_col, ""))
            pred_ids = parse_ids(hydra_rows[i].get(hydra_col, ""))

            gt_set = set(gt_ids)
            pred_set = set(pred_ids)
            recall = (len(gt_set & pred_set) / len(gt_set)) if gt_set else 0.0
            hit = 1 if len(gt_set & pred_set) > 0 else 0

            per_query.append(
                {
                    "query": str(i),
                    "ground_truth_count": len(gt_set),
                    "predicted_count": len(pred_set),
                    "intersection_count": len(gt_set & pred_set),
                    "recall": recall,
                    "hit": hit,
                }
            )

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
    fieldnames = [
        "query",
        "ground_truth_count",
        "predicted_count",
        "intersection_count",
        "recall",
        "hit",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute recall for monolithic baseline, hydra, and fine-grained hydra vs ground truth."
    )
    parser.add_argument("--ground-truth", default="hydra_monolithic_ground_truth.csv", help="Path to ground-truth CSV")
    parser.add_argument("--monolithic", default="hydra_baseline_bs_1.csv", help="Path to monolithic baseline CSV")
    parser.add_argument("--hydra", default="hydra_analysis.csv", help="Path to hydra CSV")
    parser.add_argument("--fine-grained-hydra", default="fine_grained_hydra_analysis.csv", help="Path to fine-grained hydra CSV")
    parser.add_argument("--output-per-query-prefix", default="", help="Optional prefix to save per-query CSVs (one per system)")

    args = parser.parse_args()

    gt_rows = read_csv(args.ground_truth)

    gt_query_col = detect_column(gt_rows, ["query", "Query"], "query", "ground_truth")
    gt_ids_col = detect_column(gt_rows, ["best_retrieved_ids", "TopDocs", "top_docs"], "retrieved ids", "ground_truth")

    systems = [
        ("Monolithic", args.monolithic),
        ("Hydra", args.hydra),
        ("FineGrainedHydra", args.fine_grained_hydra),
    ]

    print("Recall Comparison")
    print("=================")
    print(f"Ground truth: {args.ground_truth} (query={gt_query_col}, ids={gt_ids_col})")

    for system_name, system_path in systems:
        system_rows = read_csv(system_path)
        system_query_col = detect_column(system_rows, ["query"], "query", system_name)
        system_ids_col = detect_column(
            system_rows,
            ["best_retrieved_ids"],
            "retrieved ids",
            system_name,
        )

        per_query, summary = compute_recall(
            ground_truth_rows=gt_rows,
            hydra_rows=system_rows,
            query_col=gt_query_col if gt_query_col == system_query_col else "query",
            gt_col=gt_ids_col,
            hydra_col=system_ids_col,
        )

        if gt_query_col != system_query_col:
            gt_rows_norm = [{**r, "query": r.get(gt_query_col, "")} for r in gt_rows]
            system_rows_norm = [{**r, "query": r.get(system_query_col, "")} for r in system_rows]
            per_query, summary = compute_recall(
                ground_truth_rows=gt_rows_norm,
                hydra_rows=system_rows_norm,
                query_col="query",
                gt_col=gt_ids_col,
                hydra_col=system_ids_col,
            )

        if system_name == "Monolithic":
            avg_latency, latency_col = average_latency_monolithic(system_rows, system_name)
            latency_desc = f"avg({latency_col})"
        else:
            avg_latency, transfer_col, search_col = average_latency_hydra(system_rows, system_name)
            latency_desc = f"avg({transfer_col} + {search_col})"

        print(f"\n{system_name}:")
        print(f"  File:              {system_path}")
        print(f"  Query column:      {system_query_col}")
        print(f"  Retrieved-id col:  {system_ids_col}")
        print(f"  Avg latency (s):   {avg_latency:.6f} [{latency_desc}]")
        print(f"  Evaluated queries: {int(summary['evaluated_queries'])}")
        print(f"  Average recall:    {summary['avg_recall']:.6f}")
        print(f"  Hit rate:          {summary['hit_rate']:.6f}")

        if args.output_per_query_prefix:
            output_path = f"{args.output_per_query_prefix}_{system_name.lower()}.csv"
            write_per_query(output_path, per_query)
            print(f"  Saved per-query:   {output_path}")


if __name__ == "__main__":
    main()
