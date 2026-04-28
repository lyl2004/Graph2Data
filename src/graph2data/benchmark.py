"""Batch benchmark runner for synthetic assets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from .image_io import write_json
from .lines import LinePathExtractor
from .models import to_serializable
from .quality import evaluate_curve_path


def run_path_benchmark(case_dir: str) -> Dict:
    root = Path(case_dir)
    truth_axes_path = root / "truth_axes.json"
    truth_data_path = root / "truth_data.csv"
    mask_dir = root / "masks"

    with open(truth_axes_path, "r", encoding="utf-8") as f:
        truth_axes = json.load(f)

    extractor = LinePathExtractor()
    curve_metrics: List[Dict] = []
    for mask_path in sorted(mask_dir.glob("*.png")):
        curve_id = mask_path.stem
        path = extractor.extract_from_mask_file(str(mask_path), curve_id=curve_id)
        path_json = to_serializable(path)
        metrics = evaluate_curve_path(path_json, truth_axes, str(truth_data_path), curve_id)
        curve_metrics.append(metrics)

    valid = [m for m in curve_metrics if m.get("path_success")]
    summary = {
        "case_dir": str(root),
        "curve_count": len(curve_metrics),
        "valid_curve_count": len(valid),
        "mean_chamfer_distance_px": _mean(valid, "chamfer_distance_px"),
        "mean_hausdorff_distance_px": _mean(valid, "hausdorff_distance_px"),
        "mean_truth_to_pred_px": _mean(valid, "mean_truth_to_pred_px"),
        "mean_completed_to_truth_px": _mean([m for m in valid if m.get("mean_completed_to_truth_px") is not None], "mean_completed_to_truth_px"),
    }
    return {"summary": summary, "curves": curve_metrics}


def _mean(rows: List[Dict], key: str):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Graph2Data synthetic benchmark metrics.")
    parser.add_argument("--case", required=True, help="Synthetic case directory")
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    result = run_path_benchmark(args.case)
    if args.out:
        write_json(args.out, result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
