"""Batch benchmark runner for synthetic assets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

from .colors import ColorExtractorConfig, CurveColorExtractor
from .image_io import ensure_dir, load_bgr, write_json
from .legend import LegendDetector
from .lines import LinePathExtractor
from .mapping import map_curve_path_to_data
from .masks import CurveMaskExtractor
from .models import BoundingBox, CurvePrototype, DataRange, PlotArea
from .models import to_serializable
from .quality import evaluate_curve_path, evaluate_data_series
from .synthetic import SyntheticConfig, generate_benchmark


def run_path_benchmark(case_dir: str) -> Dict:
    root = Path(case_dir)
    truth_axes_path = root / "truth_axes.json"
    truth_data_path = root / "truth_data.csv"
    mask_dir = root / "masks"

    with open(truth_axes_path, "r", encoding="utf-8") as f:
        truth_axes = json.load(f)
    plot = truth_axes["plot_area"]
    data_range = truth_axes["data_range"]
    plot_area = PlotArea(BoundingBox(plot["x_min"], plot["y_min"], plot["x_max"], plot["y_max"]))
    data_range_model = DataRange(
        data_range["x_min"],
        data_range["x_max"],
        data_range["y_min"],
        data_range["y_max"],
        data_range.get("x_scale", "linear"),
        data_range.get("y_scale", "linear"),
    )

    extractor = LinePathExtractor()
    curve_metrics: List[Dict] = []
    for mask_path in sorted(mask_dir.glob("*.png")):
        curve_id = mask_path.stem
        path = extractor.extract_from_mask_file(str(mask_path), curve_id=curve_id)
        path_json = to_serializable(path)
        metrics = evaluate_curve_path(path_json, truth_axes, str(truth_data_path), curve_id)
        series = map_curve_path_to_data(path, plot_area, data_range_model)
        metrics.update(evaluate_data_series(to_serializable(series), str(truth_data_path), curve_id))
        curve_metrics.append(metrics)

    valid = [m for m in curve_metrics if m.get("path_success")]
    summary = {
        "case_dir": str(root),
        "curve_count": len(curve_metrics),
        "valid_curve_count": len(valid),
        "mean_completed_point_ratio": _mean_completed_point_ratio(valid),
        "mean_chamfer_distance_px": _mean(valid, "chamfer_distance_px"),
        "mean_hausdorff_distance_px": _mean(valid, "hausdorff_distance_px"),
        "mean_truth_to_pred_px": _mean(valid, "mean_truth_to_pred_px"),
        "mean_completed_to_truth_px": _mean([m for m in valid if m.get("mean_completed_to_truth_px") is not None], "mean_completed_to_truth_px"),
        "mean_data_y_mae": _mean(valid, "data_y_mae"),
        "mean_data_y_rmse": _mean(valid, "data_y_rmse"),
        "mean_data_y_max_abs_error": _mean(valid, "data_y_max_abs_error"),
        "mean_data_r2_at_pred_x": _mean(valid, "data_r2_at_pred_x"),
        "mean_data_x_coverage_ratio": _mean(valid, "data_x_coverage_ratio"),
    }
    return {"summary": summary, "curves": curve_metrics}


def run_suite(output_root: str) -> Dict:
    """Generate standard synthetic cases and run truth/predicted mask metrics."""
    ensure_dir(output_root)
    synthetic_root = os.path.join(output_root, "synthetic")
    result_root = ensure_dir(os.path.join(output_root, "results"))

    cases = [
        ("basic_curves", SyntheticConfig(seed=42, n_curves=6, palette="basic")),
        ("achromatic_curves", SyntheticConfig(seed=7, n_curves=6, palette="achromatic")),
        ("local_occlusion_curves", SyntheticConfig(seed=42, n_curves=6, palette="basic", local_occlusion=True)),
        ("crossing_curves", SyntheticConfig(seed=13, n_curves=6, palette="basic", crossing_curves=True)),
        ("legend_inside_curves", SyntheticConfig(seed=42, n_curves=6, palette="basic", legend_inside=True)),
    ]

    suite_cases = []
    for name, config in cases:
        manifest = generate_benchmark(synthetic_root, name=name, config=config)
        case_dir = os.path.dirname(manifest["image_path"])

        truth = run_path_benchmark(case_dir)
        pred = run_predicted_mask_benchmark(case_dir, output_dir=os.path.join(result_root, f"{name}_predicted_masks"))
        pred_excluded = None
        pred_excluded_path = None
        pred_detected = None
        pred_detected_path = None
        if config.legend_inside:
            pred_excluded = run_predicted_mask_benchmark(
                case_dir,
                output_dir=os.path.join(result_root, f"{name}_predicted_masks_exclude_legend"),
                exclude_legend=True,
            )
            pred_excluded_path = os.path.join(result_root, f"{name}_predicted_mask_exclude_legend_metrics.json")
            pred_detected = run_predicted_mask_benchmark(
                case_dir,
                output_dir=os.path.join(result_root, f"{name}_predicted_masks_detected_legend"),
                use_detected_legend=True,
            )
            pred_detected_path = os.path.join(result_root, f"{name}_predicted_mask_detected_legend_metrics.json")

        truth_path = os.path.join(result_root, f"{name}_truth_mask_metrics.json")
        pred_path = os.path.join(result_root, f"{name}_predicted_mask_metrics.json")
        write_json(truth_path, truth)
        write_json(pred_path, pred)
        if pred_excluded is not None and pred_excluded_path is not None:
            write_json(pred_excluded_path, pred_excluded)
        if pred_detected is not None and pred_detected_path is not None:
            write_json(pred_detected_path, pred_detected)

        case_result = {
            "name": name,
            "case_dir": case_dir,
            "truth_metrics": truth["summary"],
            "predicted_metrics": pred["summary"],
            "truth_metrics_path": truth_path,
            "predicted_metrics_path": pred_path,
        }
        if pred_excluded is not None and pred_excluded_path is not None:
            case_result["predicted_exclude_legend_metrics"] = pred_excluded["summary"]
            case_result["predicted_exclude_legend_metrics_path"] = pred_excluded_path
            case_result["legend_exclusion_delta"] = _legend_exclusion_delta(pred["summary"], pred_excluded["summary"])
        if pred_detected is not None and pred_detected_path is not None:
            case_result["predicted_detected_legend_metrics"] = pred_detected["summary"]
            case_result["predicted_detected_legend_metrics_path"] = pred_detected_path
            case_result["detected_legend_delta"] = _legend_exclusion_delta(pred["summary"], pred_detected["summary"])
        suite_cases.append(case_result)

    suite = {"output_root": output_root, "cases": suite_cases}
    write_json(os.path.join(result_root, "suite_summary.json"), suite)
    return suite


def run_predicted_mask_benchmark(
    case_dir: str,
    output_dir: Optional[str] = None,
    exclude_legend: bool = False,
    use_detected_legend: bool = False,
) -> Dict:
    root = Path(case_dir)
    image_path = root / "image.png"
    truth_axes_path = root / "truth_axes.json"
    truth_curves_path = root / "truth_curves.json"
    truth_data_path = root / "truth_data.csv"

    with open(truth_axes_path, "r", encoding="utf-8") as f:
        truth_axes = json.load(f)
    with open(truth_curves_path, "r", encoding="utf-8") as f:
        truth_curves = json.load(f)["curves"]

    image = load_bgr(str(image_path))
    plot = truth_axes["plot_area"]
    data_range = truth_axes["data_range"]
    plot_bbox = BoundingBox(plot["x_min"], plot["y_min"], plot["x_max"], plot["y_max"])
    plot_area = PlotArea(plot_bbox)
    data_range_model = DataRange(
        data_range["x_min"],
        data_range["x_max"],
        data_range["y_min"],
        data_range["y_max"],
        data_range.get("x_scale", "linear"),
        data_range.get("y_scale", "linear"),
    )
    detected_legends = []
    if use_detected_legend:
        detected_legends = LegendDetector().detect_from_image(image, plot_area)
        exclude_regions = [legend.bbox for legend in detected_legends]
    else:
        exclude_regions = _synthetic_legend_exclusions(truth_axes, truth_curves) if exclude_legend else []
    prototypes = CurveColorExtractor(ColorExtractorConfig(min_ratio=0.0004)).extract(
        image, plot_bbox, exclude_regions=exclude_regions
    )

    mask_output = Path(output_dir) if output_dir else root / "predicted_masks"
    mask_output.mkdir(parents=True, exist_ok=True)

    mask_extractor = CurveMaskExtractor()
    path_extractor = LinePathExtractor()
    rows: List[Dict] = []
    for truth in truth_curves:
        matched = _match_prototype(truth["color"], prototypes)
        if matched is None:
            rows.append({"path_success": False, "curve_id": truth["curve_id"], "error": "no color prototype match"})
            continue
        matched.curve_id = truth["curve_id"]
        mask, mask_info = mask_extractor.extract_mask(image, matched, plot_bbox, exclude_regions=exclude_regions)
        mask_path = mask_output / f"{truth['curve_id']}.png"
        cv2.imwrite(str(mask_path), mask)
        path = path_extractor.extract_from_mask_image(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), curve_id=truth["curve_id"])
        metrics = evaluate_curve_path(to_serializable(path), truth_axes, str(truth_data_path), truth["curve_id"])
        series = map_curve_path_to_data(path, plot_area, data_range_model)
        metrics.update(evaluate_data_series(to_serializable(series), str(truth_data_path), truth["curve_id"]))
        metrics.update(_evaluate_mask_against_truth(mask, root / truth["mask_path"]))
        metrics["matched_rgb"] = matched.rgb
        metrics["truth_color"] = truth["color"]
        metrics["mask_pixel_count"] = mask_info.pixel_count
        metrics["mask_confidence"] = mask_info.confidence
        rows.append(metrics)

    valid = [m for m in rows if m.get("path_success")]
    summary = {
        "case_dir": str(root),
        "mode": "predicted_mask",
        "exclude_legend": exclude_legend,
        "legend_exclusion_source": "image_heuristic" if use_detected_legend else ("synthetic_truth" if exclude_legend else "none"),
        "detected_legend_count": len(detected_legends),
        "curve_count": len(rows),
        "valid_curve_count": len(valid),
        "prototype_count": len(prototypes),
        "mean_completed_point_ratio": _mean_completed_point_ratio(valid),
        "mean_chamfer_distance_px": _mean(valid, "chamfer_distance_px"),
        "mean_hausdorff_distance_px": _mean(valid, "hausdorff_distance_px"),
        "mean_truth_to_pred_px": _mean(valid, "mean_truth_to_pred_px"),
        "mean_mask_iou": _mean(rows, "mask_iou"),
        "mean_mask_precision": _mean(rows, "mask_precision"),
        "mean_mask_recall": _mean(rows, "mask_recall"),
        "mean_mask_f1": _mean(rows, "mask_f1"),
        "mean_mask_tolerant_precision": _mean(rows, "mask_tolerant_precision"),
        "mean_mask_tolerant_recall": _mean(rows, "mask_tolerant_recall"),
        "mean_mask_tolerant_f1": _mean(rows, "mask_tolerant_f1"),
        "mean_completed_to_truth_px": _mean(
            [m for m in valid if m.get("mean_completed_to_truth_px") is not None], "mean_completed_to_truth_px"
        ),
        "mean_data_y_mae": _mean(valid, "data_y_mae"),
        "mean_data_y_rmse": _mean(valid, "data_y_rmse"),
        "mean_data_y_max_abs_error": _mean(valid, "data_y_max_abs_error"),
        "mean_data_r2_at_pred_x": _mean(valid, "data_r2_at_pred_x"),
        "mean_data_x_coverage_ratio": _mean(valid, "data_x_coverage_ratio"),
    }
    return {"summary": summary, "curves": rows, "prototypes": [to_serializable(p) for p in prototypes]}


def _synthetic_legend_exclusions(truth_axes: Dict, truth_curves: List[Dict]) -> List[BoundingBox]:
    """Return the known upper-left legend area for synthetic legend_inside cases.

    This gives a controlled benchmark for the best-case impact of legend
    exclusion. The OCR-based detector remains in pipeline.py for real images;
    synthetic truth makes the before/after metric comparison deterministic.
    """
    plot = truth_axes["plot_area"]
    if len(truth_curves) == 0:
        return []
    width = 210.0
    height = 34.0 + 20.0 * len(truth_curves)
    return [
        BoundingBox(
            float(plot["x_min"]) + 8.0,
            float(plot["y_min"]) + 8.0,
            min(float(plot["x_max"]), float(plot["x_min"]) + width),
            min(float(plot["y_max"]), float(plot["y_min"]) + height),
        )
    ]


def _match_prototype(hex_color: str, prototypes: List[CurvePrototype]) -> Optional[CurvePrototype]:
    if not prototypes:
        return None
    target = _hex_to_rgb(hex_color)
    best = min(prototypes, key=lambda p: _rgb_distance(target, p.rgb))
    # Return a shallow copy so assigning curve_id does not mutate the list for future matching semantics.
    return CurvePrototype(
        curve_id=best.curve_id,
        rgb=best.rgb,
        lab=best.lab,
        area=best.area,
        ratio=best.ratio,
        label=best.label,
        line_style=best.line_style,
        marker_style=best.marker_style,
        confidence=best.confidence,
        source=best.source,
    )


def _hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.strip().lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _rgb_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) ** 0.5


def _evaluate_mask_against_truth(predicted_mask, truth_mask_path: Path) -> Dict:
    truth_mask = cv2.imread(str(truth_mask_path), cv2.IMREAD_GRAYSCALE)
    if truth_mask is None:
        return {
            "mask_iou": None,
            "mask_precision": None,
            "mask_recall": None,
            "mask_f1": None,
            "mask_tolerant_precision": None,
            "mask_tolerant_recall": None,
            "mask_tolerant_f1": None,
            "mask_true_positive": 0,
            "mask_false_positive": 0,
            "mask_false_negative": 0,
        }
    if truth_mask.shape != predicted_mask.shape:
        truth_mask = cv2.resize(truth_mask, (predicted_mask.shape[1], predicted_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

    pred = predicted_mask > 0
    truth = truth_mask > 127
    true_positive = int((pred & truth).sum())
    false_positive = int((pred & ~truth).sum())
    false_negative = int((~pred & truth).sum())
    union = true_positive + false_positive + false_negative

    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    tolerant_precision, tolerant_recall, tolerant_f1 = _evaluate_tolerant_mask_match(pred, truth, tolerance_px=2)
    return {
        "mask_iou": _safe_divide(true_positive, union),
        "mask_precision": precision,
        "mask_recall": recall,
        "mask_f1": f1,
        "mask_tolerant_precision": tolerant_precision,
        "mask_tolerant_recall": tolerant_recall,
        "mask_tolerant_f1": tolerant_f1,
        "mask_true_positive": true_positive,
        "mask_false_positive": false_positive,
        "mask_false_negative": false_negative,
    }


def _evaluate_tolerant_mask_match(pred, truth, tolerance_px: int = 2) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    kernel_size = 2 * tolerance_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    pred_dilated = cv2.dilate(pred.astype("uint8"), kernel) > 0
    truth_dilated = cv2.dilate(truth.astype("uint8"), kernel) > 0
    pred_count = int(pred.sum())
    truth_count = int(truth.sum())
    precision = _safe_divide(int((pred & truth_dilated).sum()), pred_count)
    recall = _safe_divide(int((truth & pred_dilated).sum()), truth_count)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return precision, recall, f1


def _legend_exclusion_delta(before: Dict, after: Dict) -> Dict:
    keys = [
        "mean_chamfer_distance_px",
        "mean_hausdorff_distance_px",
        "mean_truth_to_pred_px",
        "mean_completed_to_truth_px",
        "mean_mask_iou",
        "mean_mask_f1",
        "mean_mask_tolerant_f1",
        "mean_data_y_mae",
        "mean_data_y_rmse",
        "mean_data_y_max_abs_error",
    ]
    delta = {}
    for key in keys:
        before_value = before.get(key)
        after_value = after.get(key)
        if before_value is None or after_value is None:
            delta[key] = None
        else:
            delta[key] = float(before_value) - float(after_value)
    return delta


def _mean(rows: List[Dict], key: str):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _mean_completed_point_ratio(rows: List[Dict]):
    ratios = []
    for row in rows:
        pred_count = row.get("pred_point_count")
        completed_count = row.get("completed_point_count")
        if pred_count:
            ratios.append(float(completed_count or 0) / float(pred_count))
    if not ratios:
        return None
    return sum(ratios) / len(ratios)


def _safe_divide(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Graph2Data synthetic benchmark metrics.")
    parser.add_argument("--case", default=None, help="Synthetic case directory")
    parser.add_argument("--mode", choices=["truth-mask", "predicted-mask"], default="truth-mask")
    parser.add_argument("--mask_out", default=None, help="Optional output directory for predicted masks")
    parser.add_argument("--exclude_legend", action="store_true", help="Exclude synthetic legend region in predicted-mask mode")
    parser.add_argument("--detect_legend", action="store_true", help="Exclude image-detected legend regions in predicted-mask mode")
    parser.add_argument("--suite", action="store_true", help="Generate and run the standard benchmark suite")
    parser.add_argument("--suite_out", default="benchmarks/suite", help="Suite output root")
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    if args.suite:
        result = run_suite(args.suite_out)
    elif not args.case:
        raise ValueError("--case is required unless --suite is used")
    elif args.mode == "truth-mask":
        result = run_path_benchmark(args.case)
    else:
        result = run_predicted_mask_benchmark(
            args.case,
            args.mask_out,
            exclude_legend=args.exclude_legend,
            use_detected_legend=args.detect_legend,
        )
    if args.out:
        write_json(args.out, result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
