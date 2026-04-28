"""Minimal benchmark quality metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
from typing import Dict, List, Sequence, Tuple

from .image_io import write_json

PixelPoint = Tuple[float, float]


def evaluate_axis_detection(prediction: Dict, truth: Dict) -> Dict:
    """Compare pipeline axis JSON against synthetic axis truth."""
    pred_axis = prediction.get("axis", {})
    truth_plot = truth["plot_area"]
    pred_plot = (pred_axis.get("plot_area") or {}).get("bbox")
    if not pred_axis.get("success") or pred_plot is None:
        return {"axis_success": False, "error": pred_axis.get("error", "axis detection failed")}

    bbox_errors = {
        key: abs(float(pred_plot[key]) - float(truth_plot[key]))
        for key in ("x_min", "y_min", "x_max", "y_max")
    }
    origin_error = _point_distance(pred_axis.get("origin"), truth.get("origin"))
    x_end_error = _point_distance((pred_axis.get("x_axis") or {}).get("end"), truth.get("x_axis", {}).get("end"))
    y_end_error = _point_distance((pred_axis.get("y_axis") or {}).get("end"), truth.get("y_axis", {}).get("end"))

    return {
        "axis_success": True,
        "bbox_abs_error_px": bbox_errors,
        "bbox_mean_abs_error_px": sum(bbox_errors.values()) / len(bbox_errors),
        "origin_error_px": origin_error,
        "x_end_error_px": x_end_error,
        "y_end_error_px": y_end_error,
        "prediction_confidence": pred_axis.get("confidence", 0.0),
    }


def evaluate_curve_path(path_result: Dict, truth_axes: Dict, truth_data_path: str, curve_id: str) -> Dict:
    """Compare an extracted CurvePath JSON against synthetic truth data."""
    pred_points = _points_from_path(path_result)
    truth_points = _truth_curve_pixels(truth_data_path, truth_axes, curve_id)
    if not pred_points:
        return {"path_success": False, "error": "empty predicted path", "curve_id": curve_id}
    if not truth_points:
        return {"path_success": False, "error": "empty truth curve", "curve_id": curve_id}

    pred_to_truth = _nearest_distances(pred_points, truth_points)
    truth_to_pred = _nearest_distances(truth_points, pred_points)
    observed_points, completed_points = _split_observed_completed_points(path_result)
    observed_to_truth = _nearest_distances(observed_points, truth_points) if observed_points else []
    completed_to_truth = _nearest_distances(completed_points, truth_points) if completed_points else []
    chamfer = (sum(pred_to_truth) / len(pred_to_truth) + sum(truth_to_pred) / len(truth_to_pred)) / 2
    hausdorff = max(max(pred_to_truth), max(truth_to_pred))

    return {
        "path_success": True,
        "curve_id": curve_id,
        "pred_point_count": len(pred_points),
        "observed_point_count": len(observed_points),
        "completed_point_count": len(completed_points),
        "truth_point_count": len(truth_points),
        "mean_pred_to_truth_px": sum(pred_to_truth) / len(pred_to_truth),
        "mean_truth_to_pred_px": sum(truth_to_pred) / len(truth_to_pred),
        "mean_observed_to_truth_px": (sum(observed_to_truth) / len(observed_to_truth)) if observed_to_truth else None,
        "mean_completed_to_truth_px": (sum(completed_to_truth) / len(completed_to_truth)) if completed_to_truth else None,
        "chamfer_distance_px": chamfer,
        "hausdorff_distance_px": hausdorff,
        "p95_pred_to_truth_px": _percentile(pred_to_truth, 95),
        "p95_truth_to_pred_px": _percentile(truth_to_pred, 95),
        "path_confidence": path_result.get("confidence", 0.0),
        "path_warnings": path_result.get("warnings", []),
    }


def _points_from_path(path_result: Dict) -> List[PixelPoint]:
    points = []
    for p in path_result.get("pixel_points_ordered", []):
        points.append((float(p["x"]), float(p["y"])))
    return points


def _split_observed_completed_points(path_result: Dict) -> Tuple[List[PixelPoint], List[PixelPoint]]:
    points = _points_from_path(path_result)
    completed = set()
    for start, end in path_result.get("completed_ranges", []):
        for idx in range(int(start), int(end) + 1):
            completed.add(idx)
    observed_points = [point for idx, point in enumerate(points) if idx not in completed]
    completed_points = [point for idx, point in enumerate(points) if idx in completed]
    return observed_points, completed_points


def _truth_curve_pixels(truth_data_path: str, truth_axes: Dict, curve_id: str) -> List[PixelPoint]:
    data_range = truth_axes["data_range"]
    plot = truth_axes["plot_area"]
    points = []
    with open(truth_data_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("curve_id") != curve_id:
                continue
            x = float(row["x"])
            y = float(row["y"])
            points.append(_data_to_pixel(x, y, data_range, plot))
    return points


def _data_to_pixel(x: float, y: float, data_range: Dict, plot: Dict) -> PixelPoint:
    x_norm = (x - float(data_range["x_min"])) / (float(data_range["x_max"]) - float(data_range["x_min"]))
    y_norm = (y - float(data_range["y_min"])) / (float(data_range["y_max"]) - float(data_range["y_min"]))
    px = float(plot["x_min"]) + x_norm * (float(plot["x_max"]) - float(plot["x_min"]))
    py = float(plot["y_max"]) - y_norm * (float(plot["y_max"]) - float(plot["y_min"]))
    return px, py


def _nearest_distances(source: Sequence[PixelPoint], target: Sequence[PixelPoint]) -> List[float]:
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(target)
        distances, _ = tree.query(source, k=1)
        return [float(d) for d in distances]
    except Exception:
        distances = []
        for sx, sy in source:
            best = min(math.hypot(sx - tx, sy - ty) for tx, ty in target)
            distances.append(float(best))
        return distances


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * percentile / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo))


def _point_distance(a, b) -> float:
    if not a or not b:
        return float("nan")
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate pipeline output against benchmark truth.")
    parser.add_argument("--prediction", default=None, help="Pipeline JSON output")
    parser.add_argument("--truth_axes", required=True, help="Synthetic truth_axes.json")
    parser.add_argument("--path", default=None, help="CurvePath JSON output")
    parser.add_argument("--truth_data", default=None, help="Synthetic truth_data.csv")
    parser.add_argument("--curve_id", default=None, help="Curve id for path evaluation")
    parser.add_argument("--out", default=None, help="Optional metrics JSON output")
    args = parser.parse_args()

    with open(args.truth_axes, "r", encoding="utf-8") as f:
        truth = json.load(f)

    metrics = {}
    if args.prediction:
        with open(args.prediction, "r", encoding="utf-8") as f:
            prediction = json.load(f)
        metrics["axis"] = evaluate_axis_detection(prediction, truth)

    if args.path:
        if not args.truth_data or not args.curve_id:
            raise ValueError("--path evaluation requires --truth_data and --curve_id")
        with open(args.path, "r", encoding="utf-8") as f:
            path_result = json.load(f)
        metrics["path"] = evaluate_curve_path(path_result, truth, args.truth_data, args.curve_id)

    if not metrics:
        raise ValueError("Provide --prediction and/or --path to evaluate")

    if args.out:
        write_json(args.out, metrics)
    else:
        print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
