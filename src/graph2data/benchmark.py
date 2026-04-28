"""Batch benchmark runner for synthetic assets."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

from .colors import ColorExtractorConfig, CurveColorExtractor
from .image_io import ensure_dir, load_bgr, write_json
from .instances import (
    cluster_markers_by_x_rank,
    cluster_markers_by_y,
    filter_border_line_components,
    group_line_style_curve_instances,
    group_marker_curve_instances,
    should_group_line_style_curve_instances,
    should_group_marker_curve_instances,
)
from .legend import LegendDetector
from .lines import LinePathExtractor
from .mapping import map_curve_path_to_data
from .masks import CurveMaskExtractor
from .models import BoundingBox, CurvePrototype, DataRange, LegendDetection, PlotArea
from .models import to_serializable
from .pipeline import (
    GraphExtractionPipeline,
    PipelineConfig,
    _prototype_bound_curve_paths,
    _prototype_bound_line_style_paths,
    _prototype_bound_marker_paths,
    _score_prototype_bindings,
)
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


def run_prototype_binding_benchmark(case_dir: str) -> Dict:
    """Evaluate diagnostic legend-prototype binding against synthetic truth order.

    Matplotlib legends list curves in plotting order, which matches truth_curves.
    This benchmark therefore maps detected legend item row order to truth curve
    order and checks whether the highest-scoring binding points to the same
    truth curve id after color-prototype matching.
    """
    root = Path(case_dir)
    image_path = root / "image.png"
    truth_axes_path = root / "truth_axes.json"
    truth_curves_path = root / "truth_curves.json"
    truth_data_path = root / "truth_data.csv"

    with open(truth_axes_path, "r", encoding="utf-8") as f:
        truth_axes = json.load(f)
    with open(truth_curves_path, "r", encoding="utf-8") as f:
        truth_curves = json.load(f)["curves"]

    data_range = truth_axes["data_range"]
    image = load_bgr(str(image_path))
    plot = truth_axes["plot_area"]
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
    pipeline_result = GraphExtractionPipeline(PipelineConfig(run_paths=True, data_range=data_range_model)).run(str(image_path))

    legend_detector = LegendDetector()
    legends = pipeline_result.legends or _synthetic_legend_detections(truth_axes, truth_curves)
    legend_items = legend_detector.extract_items(image, legends)
    for idx, item in enumerate(legend_items):
        if idx < len(truth_curves):
            truth_curve = truth_curves[idx]
            if item.label is None:
                item.label = truth_curve.get("label")
            item.line_style = _truth_line_style_label(truth_curve.get("linestyle"))
            item.marker_style = "repeated_compact" if truth_curve.get("marker") else "unknown"
    visual_prototypes = legend_detector.visual_prototypes_from_items(legend_items)

    exclude_regions = [legend.bbox for legend in legends]
    curves = CurveColorExtractor(ColorExtractorConfig(min_ratio=0.0004)).extract(image, plot_bbox, exclude_regions=exclude_regions)
    mask_extractor = CurveMaskExtractor()
    path_extractor = LinePathExtractor()
    curve_paths = []
    line_components = []
    marker_candidates = []
    for curve in curves:
        mask, _ = mask_extractor.extract_mask(image, curve, plot_bbox, exclude_regions=exclude_regions)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        curve_paths.append(path_extractor.extract_from_mask_image(mask_bgr, curve_id=curve.curve_id))
        line_components.extend(path_extractor.classify_components_from_mask_image(mask_bgr, curve_id=curve.curve_id))
        marker_candidates.extend(path_extractor.detect_marker_candidates_from_mask_image(mask_bgr, curve_id=curve.curve_id))
    line_components = filter_border_line_components(line_components, plot_bbox)

    truth_marker_centers = _truth_marker_centers(truth_axes, truth_curves, truth_data_path)
    marker_metrics = _evaluate_marker_candidate_recall(marker_candidates, truth_marker_centers)
    marker_curve_instances = []
    if should_group_marker_curve_instances(marker_candidates, line_components):
        marker_curve_instances = group_marker_curve_instances(
            marker_candidates,
            group_count=len([curve for curve in truth_curves if curve.get("marker")]) or None,
        )
    line_style_curve_instances = []
    if not marker_curve_instances and should_group_line_style_curve_instances(line_components, len(visual_prototypes)):
        line_style_curve_instances = group_line_style_curve_instances(
            line_components,
            group_count=len(visual_prototypes),
        )
    prototype_bindings = _score_prototype_bindings(
        visual_prototypes,
        curves,
        line_components,
        marker_candidates,
        marker_curve_instances=marker_curve_instances,
        line_style_curve_instances=line_style_curve_instances,
    )
    prototype_bound_paths = _prototype_bound_marker_paths(
        visual_prototypes,
        prototype_bindings,
        marker_curve_instances,
        curve_paths,
    )
    prototype_bound_paths.extend(
        _prototype_bound_curve_paths(
            visual_prototypes,
            prototype_bindings,
            curve_paths,
            line_components,
        )
    )
    prototype_bound_paths.extend(
        _prototype_bound_line_style_paths(
            visual_prototypes,
            prototype_bindings,
            line_style_curve_instances,
        )
    )
    marker_group_metrics = _evaluate_marker_candidate_grouping(
        marker_candidates,
        truth_marker_centers,
        group_count=len([curve for curve in truth_curves if curve.get("marker")]),
    )
    predicted_to_truth = _predicted_curve_to_truth_map(curves, truth_curves)
    marker_instance_to_truth = _marker_instance_to_truth_map(marker_curve_instances, marker_group_metrics)
    line_style_instance_to_truth = _line_style_instance_to_truth_map(line_style_curve_instances, truth_curves)
    best_by_prototype = _best_binding_by_prototype(prototype_bindings)
    rows = []
    for idx, prototype in enumerate(visual_prototypes):
        expected = truth_curves[idx]["curve_id"] if idx < len(truth_curves) else None
        best = best_by_prototype.get(prototype.prototype_id)
        predicted_truth = _binding_target_truth_curve(best, predicted_to_truth, marker_instance_to_truth, line_style_instance_to_truth)
        success = expected is not None and predicted_truth == expected
        rows.append(
            {
                "prototype_id": prototype.prototype_id,
                "legend_item_id": prototype.legend_item_id,
                "expected_curve_id": expected,
                "best_target_curve_id": best.target_curve_id if best is not None else None,
                "best_target_type": best.target_type if best is not None else None,
                "best_target_truth_curve_id": predicted_truth,
                "best_score": best.score if best is not None else None,
                "success": success,
                "warnings": (best.warnings if best is not None else ["missing_binding"]),
            }
        )

    bound_path_rows = _evaluate_prototype_bound_paths(
        prototype_bound_paths,
        rows,
        truth_axes,
        truth_data_path,
        plot_area,
        data_range_model,
    )
    evaluated = [row for row in rows if row["expected_curve_id"] is not None]
    success_count = sum(1 for row in evaluated if row["success"])
    valid_bound_paths = [row for row in bound_path_rows if row.get("path_success")]
    valid_bound_data = [row for row in bound_path_rows if row.get("data_success")]
    summary = {
        "case_dir": str(root),
        "mode": "prototype_binding",
        "legend_count": len(legends),
        "legend_item_count": len(legend_items),
        "curve_visual_prototype_count": len(visual_prototypes),
        "binding_count": len(prototype_bindings),
        "curve_count": len(truth_curves),
        "evaluated_prototype_count": len(evaluated),
        "binding_success_count": success_count,
        "binding_accuracy": _safe_divide(success_count, len(evaluated)),
        "mean_best_score": _mean(rows, "best_score"),
        "predicted_curve_count": len(curves),
        "marker_candidate_count": len(marker_candidates),
        "truth_marker_count": marker_metrics["truth_marker_count"],
        "matched_truth_marker_count": marker_metrics["matched_truth_marker_count"],
        "marker_candidate_recall": marker_metrics["marker_candidate_recall"],
        "marker_candidate_precision": marker_metrics["marker_candidate_precision"],
        "marker_group_count": marker_group_metrics["marker_group_count"],
        "marker_group_assignment_accuracy": marker_group_metrics["marker_group_assignment_accuracy"],
        "marker_grouping_method": marker_group_metrics["grouping_method"],
        "marker_curve_instance_count": len(marker_curve_instances),
        "line_style_curve_instance_count": len(line_style_curve_instances),
        "prototype_bound_path_count": len(prototype_bound_paths),
        "evaluated_prototype_bound_path_count": len(bound_path_rows),
        "valid_prototype_bound_path_count": len(valid_bound_paths),
        "valid_prototype_bound_data_count": len(valid_bound_data),
        "labeled_legend_item_count": sum(1 for item in legend_items if item.label),
        "labeled_prototype_count": sum(1 for prototype in visual_prototypes if prototype.label),
        "labeled_prototype_bound_path_count": sum(1 for path in prototype_bound_paths if path.label),
        "labeled_prototype_bound_data_count": sum(1 for row in valid_bound_data if row.get("label")),
        "mean_prototype_bound_chamfer_distance_px": _mean(valid_bound_paths, "chamfer_distance_px"),
        "mean_prototype_bound_hausdorff_distance_px": _mean(valid_bound_paths, "hausdorff_distance_px"),
        "mean_prototype_bound_completed_point_ratio": _mean_completed_point_ratio(valid_bound_paths),
        "mean_prototype_bound_data_y_rmse": _mean(valid_bound_data, "data_y_rmse"),
        "mean_prototype_bound_data_x_coverage_ratio": _mean(valid_bound_data, "data_x_coverage_ratio"),
        "line_component_count": len(line_components),
    }
    return {
        "summary": summary,
        "prototypes": rows,
        "predicted_to_truth_curve": predicted_to_truth,
        "marker_instance_to_truth_curve": marker_instance_to_truth,
        "line_style_instance_to_truth_curve": line_style_instance_to_truth,
        "marker_metrics": marker_metrics,
        "marker_group_metrics": marker_group_metrics,
        "prototype_bound_path_metrics": bound_path_rows,
        "pipeline": {
            "warnings": pipeline_result.warnings,
            "legend_items": [to_serializable(item) for item in legend_items],
            "prototype_bindings": [to_serializable(binding) for binding in prototype_bindings],
            "curve_paths": [to_serializable(path) for path in curve_paths],
            "line_style_curve_instances": [to_serializable(instance) for instance in line_style_curve_instances],
            "prototype_bound_data_series": [
                to_serializable(map_curve_path_to_data(path, plot_area, data_range_model))
                for path in prototype_bound_paths
            ],
            "prototype_bound_paths": [to_serializable(path) for path in prototype_bound_paths],
        },
    }


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


def _synthetic_legend_detections(truth_axes: Dict, truth_curves: List[Dict]) -> List[LegendDetection]:
    return [
        LegendDetection(
            bbox=bbox,
            confidence=1.0,
            source="synthetic_truth",
        )
        for bbox in _synthetic_legend_exclusions(truth_axes, truth_curves)
    ]


def _truth_line_style_label(value) -> str:
    text = str(value)
    if text in {"-", "solid"}:
        return "solid"
    if text in {"--", "-.", "dashed"} or "5, 1" in text:
        return "dashed"
    if text in {":", "dotted"} or "1, 1" in text:
        return "dotted"
    return "unknown"


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


def _predicted_curve_to_truth_map(predicted_curves: List[CurvePrototype], truth_curves: List[Dict]) -> Dict[str, str]:
    mapping = {}
    remaining_truth = list(truth_curves)
    for curve in predicted_curves:
        if not remaining_truth:
            break
        best = min(remaining_truth, key=lambda truth: _rgb_distance(_hex_to_rgb(truth["color"]), curve.rgb))
        mapping[curve.curve_id] = best["curve_id"]
        remaining_truth.remove(best)
    return mapping


def _best_binding_by_prototype(bindings) -> Dict[str, object]:
    best = {}
    for binding in bindings:
        current = best.get(binding.prototype_id)
        if current is None or binding.score > current.score:
            best[binding.prototype_id] = binding
    return best


def _binding_target_truth_curve(
    binding,
    predicted_to_truth: Dict[str, str],
    marker_instance_to_truth: Dict[str, str],
    line_style_instance_to_truth: Dict[str, str],
) -> Optional[str]:
    if binding is None:
        return None
    if getattr(binding, "target_type", "curve") == "marker_instance":
        return marker_instance_to_truth.get(binding.target_curve_id)
    if getattr(binding, "target_type", "curve") == "line_style_instance":
        return line_style_instance_to_truth.get(binding.target_curve_id)
    return predicted_to_truth.get(binding.target_curve_id)


def _evaluate_prototype_bound_paths(
    prototype_bound_paths,
    prototype_rows: List[Dict],
    truth_axes: Dict,
    truth_data_path: Path,
    plot_area: PlotArea,
    data_range_model: DataRange,
) -> List[Dict]:
    expected_by_path_id = {
        f"{row['prototype_id']}_bound_path": row.get("expected_curve_id")
        for row in prototype_rows
        if row.get("expected_curve_id") is not None
    }
    rows = []
    for path in prototype_bound_paths:
        expected_curve_id = expected_by_path_id.get(path.curve_id)
        if expected_curve_id is None:
            rows.append(
                {
                    "curve_id": path.curve_id,
                    "expected_curve_id": None,
                    "path_success": False,
                    "data_success": False,
                    "error": "missing expected curve for prototype-bound path",
                }
            )
            continue
        metrics = evaluate_curve_path(to_serializable(path), truth_axes, str(truth_data_path), expected_curve_id)
        series = map_curve_path_to_data(path, plot_area, data_range_model)
        metrics.update(evaluate_data_series(to_serializable(series), str(truth_data_path), expected_curve_id))
        metrics["prototype_bound_path_id"] = path.curve_id
        metrics["expected_curve_id"] = expected_curve_id
        metrics["label"] = path.label
        rows.append(metrics)
    return rows


def _line_style_instance_to_truth_map(line_style_curve_instances, truth_curves: List[Dict]) -> Dict[str, str]:
    ordered_instances = sorted(line_style_curve_instances, key=lambda instance: float(instance.center_y), reverse=True)
    mapping = {}
    for idx, instance in enumerate(ordered_instances):
        if idx < len(truth_curves):
            mapping[instance.instance_id] = truth_curves[idx]["curve_id"]
    return mapping


def _marker_instance_to_truth_map(marker_curve_instances, marker_group_metrics: Dict) -> Dict[str, str]:
    groups = marker_group_metrics.get("groups", [])
    mapping = {}
    available_groups = [
        group
        for group in groups
        if group.get("dominant_truth_curve_id") is not None and group.get("center_y") is not None
    ]
    used_group_indices = set()
    for instance in marker_curve_instances:
        if instance.center_y is None:
            continue
        best_idx = None
        best_distance = None
        for group_idx, group in enumerate(available_groups):
            if group_idx in used_group_indices:
                continue
            distance = abs(float(instance.center_y) - float(group["center_y"]))
            if best_distance is None or distance < best_distance:
                best_idx = group_idx
                best_distance = distance
        if best_idx is None:
            continue
        used_group_indices.add(best_idx)
        mapping[instance.instance_id] = available_groups[best_idx]["dominant_truth_curve_id"]
    return mapping


def _truth_marker_centers(truth_axes: Dict, truth_curves: List[Dict], truth_data_path: Path) -> List[Dict]:
    curve_meta = {curve["curve_id"]: curve for curve in truth_curves if curve.get("marker") and curve.get("marker_every")}
    if not curve_meta:
        return []
    plot = truth_axes["plot_area"]
    data_range = truth_axes["data_range"]
    by_curve: Dict[str, List[Dict]] = {curve_id: [] for curve_id in curve_meta}
    with open(truth_data_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            curve_id = row["curve_id"]
            if curve_id in by_curve:
                by_curve[curve_id].append(row)

    centers = []
    for curve_id, rows in by_curve.items():
        every = int(curve_meta[curve_id].get("marker_every") or 0)
        if every <= 0:
            continue
        for idx, row in enumerate(rows):
            if idx % every != 0:
                continue
            x = float(row["x"])
            y = float(row["y"])
            if data_range["x_max"] == data_range["x_min"] or data_range["y_max"] == data_range["y_min"]:
                continue
            x_norm = (x - data_range["x_min"]) / (data_range["x_max"] - data_range["x_min"])
            y_norm = (y - data_range["y_min"]) / (data_range["y_max"] - data_range["y_min"])
            px = plot["x_min"] + x_norm * (plot["x_max"] - plot["x_min"])
            py = plot["y_max"] - y_norm * (plot["y_max"] - plot["y_min"])
            centers.append(
                {
                    "curve_id": curve_id,
                    "point_index": idx,
                    "x": float(px),
                    "y": float(py),
                    "marker": curve_meta[curve_id].get("marker"),
                }
            )
    return centers


def _evaluate_marker_candidate_recall(marker_candidates, truth_centers: List[Dict], tolerance_px: float = 10.0) -> Dict:
    matched_truth = set()
    matched_candidates = set()
    for truth_idx, truth in enumerate(truth_centers):
        best_idx = None
        best_dist = None
        for candidate_idx, candidate in enumerate(marker_candidates):
            dist = math.hypot(float(candidate.center.x) - truth["x"], float(candidate.center.y) - truth["y"])
            if dist <= tolerance_px and (best_dist is None or dist < best_dist):
                best_dist = dist
                best_idx = candidate_idx
        if best_idx is not None:
            matched_truth.add(truth_idx)
            matched_candidates.add(best_idx)
    truth_count = len(truth_centers)
    candidate_count = len(marker_candidates)
    return {
        "truth_marker_count": truth_count,
        "marker_candidate_count": candidate_count,
        "matched_truth_marker_count": len(matched_truth),
        "matched_marker_candidate_count": len(matched_candidates),
        "marker_candidate_recall": _safe_divide(len(matched_truth), truth_count),
        "marker_candidate_precision": _safe_divide(len(matched_candidates), candidate_count),
        "tolerance_px": float(tolerance_px),
    }


def _evaluate_marker_candidate_grouping(
    marker_candidates,
    truth_centers: List[Dict],
    group_count: int,
    tolerance_px: float = 10.0,
) -> Dict:
    if group_count <= 0 or not marker_candidates or not truth_centers:
        return {
            "marker_group_count": 0,
            "matched_marker_for_grouping_count": 0,
            "marker_group_assignment_accuracy": None,
            "grouping_method": None,
            "groups": [],
        }

    groups = cluster_markers_by_x_rank(marker_candidates, group_count)
    if not groups["centers"]:
        groups = cluster_markers_by_y(marker_candidates, group_count)
    truth_by_index = []
    for candidate_idx, candidate in enumerate(marker_candidates):
        best_idx = None
        best_dist = None
        for truth_idx, truth in enumerate(truth_centers):
            dist = math.hypot(float(candidate.center.x) - truth["x"], float(candidate.center.y) - truth["y"])
            if dist <= tolerance_px and (best_dist is None or dist < best_dist):
                best_idx = truth_idx
                best_dist = dist
        if best_idx is not None:
            truth_by_index.append((candidate_idx, truth_centers[best_idx]["curve_id"]))

    group_truth_counts: Dict[int, Dict[str, int]] = {idx: {} for idx in range(len(groups["centers"]))}
    for candidate_idx, truth_curve_id in truth_by_index:
        group_idx = groups["assignments"].get(candidate_idx)
        if group_idx is None:
            continue
        group_truth_counts[group_idx][truth_curve_id] = group_truth_counts[group_idx].get(truth_curve_id, 0) + 1

    correct = 0
    grouped_rows = []
    for group_idx, counts in group_truth_counts.items():
        if counts:
            dominant_curve_id, dominant_count = max(counts.items(), key=lambda item: item[1])
            correct += dominant_count
        else:
            dominant_curve_id = None
        grouped_rows.append(
            {
                "group_id": f"marker_group_{group_idx:02d}",
                "center_y": groups["centers"][group_idx],
                "candidate_count": len(groups["members"].get(group_idx, [])),
                "dominant_truth_curve_id": dominant_curve_id,
                "truth_counts": counts,
            }
        )

    matched_count = len(truth_by_index)
    return {
        "marker_group_count": len(groups["centers"]),
        "matched_marker_for_grouping_count": matched_count,
        "marker_group_assignment_accuracy": _safe_divide(correct, matched_count),
        "grouping_method": groups.get("method", "unknown"),
        "groups": sorted(grouped_rows, key=lambda row: row["center_y"]),
    }


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
    parser.add_argument("--mode", choices=["truth-mask", "predicted-mask", "prototype-binding"], default="truth-mask")
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
    elif args.mode == "prototype-binding":
        result = run_prototype_binding_benchmark(args.case)
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
