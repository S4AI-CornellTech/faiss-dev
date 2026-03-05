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
        description="Compute recall by comparing ground-truth IDs vs Hydra TopDocs IDs per query."
    )
    parser.add_argument("--ground-truth", default="hydra_monolithic_ground_truth.csv", help="Path to *_ground_truth.csv")
    # parser.add_argument("--hydra-analysis", default="hydra_analysis.csv", help="Path to *hydra_analysis.csv")
    parser.add_argument("--hydra-analysis", default="fine_grained_hydra_analysis.csv", help="Path to *hydra_analysis.csv")
    parser.add_argument("--query-column", default="query", help="Query key column used to join rows")
    parser.add_argument("--gt-column", default="best_retrieved_ids", help="Ground truth ID-list column")
    parser.add_argument("--hydra-column", default="TopDocs", help="Hydra predicted ID-list column")
    parser.add_argument(
        "--output-per-query",
        default="",
        help="Optional CSV path to save per-query recall details",
    )

    args = parser.parse_args()

    gt_rows = read_csv(args.ground_truth)
    hydra_rows = read_csv(args.hydra_analysis)

    per_query, summary = compute_recall(
        ground_truth_rows=gt_rows,
        hydra_rows=hydra_rows,
        query_col=args.query_column,
        gt_col=args.gt_column,
        hydra_col=args.hydra_column,
    )

    print("Recall Summary")
    print("==============")
    print(f"Evaluated queries: {int(summary['evaluated_queries'])}")
    print(f"Average recall:    {summary['avg_recall']:.6f}")
    print(f"Hit rate:          {summary['hit_rate']:.6f}")

    if args.output_per_query:
        write_per_query(args.output_per_query, per_query)
        print(f"Saved per-query results to: {args.output_per_query}")


if __name__ == "__main__":
    main()
