from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import numpy as np
import cv2

from graph2data.benchmark import run_suite
from graph2data.image_io import load_bgr
from graph2data.legend import LegendDetector
from graph2data.lines import (
    LinePathExtractor,
    PathTracingConfig,
    _interpolate_gap,
    _interpolate_gap_points,
    _order_segments,
    _prune_short_spurs,
    _segment_connection_cost,
)
from graph2data.mapping import map_curve_path_to_data
from graph2data.masks import _filter_components
from graph2data.models import BoundingBox, CurvePath, DataRange, PlotArea, Point
from graph2data.pipeline import GraphExtractionPipeline, PipelineConfig
from graph2data.quality import evaluate_data_series
from graph2data.synthetic import SyntheticConfig, generate_benchmark


ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_paths_outputs_masks_paths_and_artifacts(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    result = GraphExtractionPipeline(
        PipelineConfig(
            run_paths=True,
            run_mapping=True,
            data_range=DataRange(0.0, 100.0, -10.0, 10.0),
            artifact_dir=str(artifact_dir),
            write_debug_artifacts=True,
        )
    ).run(str(ROOT / "tests" / "test1.png"))

    assert result.axis.success
    assert not result.warnings
    assert [legend for legend in result.legends if legend.source == "image_heuristic"]
    assert len(result.curves) > 0
    assert len(result.curve_masks) == len(result.curves)
    assert len(result.curve_paths) == len(result.curves)
    assert len(result.data_series) == len(result.curves)

    artifact_masks = result.artifacts.get("masks", {})
    assert set(artifact_masks) == {curve.curve_id for curve in result.curves}
    assert Path(result.artifacts["data_csv"]).is_file()
    assert Path(result.artifacts["quality_report"]).is_file()
    debug = result.artifacts.get("debug", {})
    assert Path(debug["overview"]).is_file()
    assert Path(debug["paths_overlay"]).is_file()
    assert set(debug["paths"]) == {curve.curve_id for curve in result.curves}

    for curve_mask in result.curve_masks:
        assert curve_mask.pixel_count > 0
        assert curve_mask.bbox is not None
        assert curve_mask.mask_path is not None
        assert Path(curve_mask.mask_path).is_file()

    for curve_path in result.curve_paths:
        assert len(curve_path.pixel_points_ordered) > 0
        assert curve_path.path_length_px > 0
        assert len(curve_path.confidence_per_point) == len(curve_path.pixel_points_ordered)
        assert 0.0 <= curve_path.confidence <= 1.0
        assert Path(debug["paths"][curve_path.curve_id]).is_file()

    for series in result.data_series:
        assert series.point_count > 0
        assert series.point_count == len(series.points)
        assert series.x_min is not None
        assert series.x_max is not None
        assert series.y_min is not None
        assert series.y_max is not None

    with open(result.artifacts["quality_report"], "r", encoding="utf-8") as f:
        quality_report = json.load(f)
    assert quality_report["axis"]["success"]
    assert quality_report["counts"]["curve_path_count"] == len(result.curve_paths)
    assert quality_report["counts"]["data_series_count"] == len(result.data_series)
    assert quality_report["path_summary"]["total_point_count"] > 0
    assert 0.0 <= quality_report["path_summary"]["mean_path_confidence"] <= 1.0
    assert len(quality_report["curves"]) == len(result.curves)


def test_mapping_converts_pixel_path_to_data_coordinates():
    curve_path = CurvePath(
        curve_id="curve",
        pixel_points_ordered=[
            Point(10.0, 220.0),
            Point(60.0, 120.0),
            Point(110.0, 20.0),
        ],
        completed_ranges=[(1, 1)],
        confidence_per_point=[1.0, 0.35, 1.0],
    )
    plot_area = PlotArea(BoundingBox(10.0, 20.0, 110.0, 220.0))
    data_range = DataRange(0.0, 10.0, -1.0, 1.0)

    series = map_curve_path_to_data(curve_path, plot_area, data_range)

    assert series.point_count == 3
    assert series.completed_point_count == 1
    assert series.points[0].x == 0.0
    assert series.points[0].y == -1.0
    assert series.points[1].x == 5.0
    assert series.points[1].y == 0.0
    assert series.points[1].completed
    assert series.points[1].confidence == 0.35
    assert series.points[2].x == 10.0
    assert series.points[2].y == 1.0


def test_data_series_quality_metrics_compare_at_predicted_x(tmp_path):
    truth_data = tmp_path / "truth_data.csv"
    with open(truth_data, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["curve_id", "x", "y"])
        writer.writeheader()
        writer.writerows(
            [
                {"curve_id": "curve", "x": 0.0, "y": 0.0},
                {"curve_id": "curve", "x": 5.0, "y": 10.0},
                {"curve_id": "curve", "x": 10.0, "y": 20.0},
                {"curve_id": "other", "x": 0.0, "y": 100.0},
            ]
        )
    data_series = {
        "curve_id": "curve",
        "points": [
            {"x": -1.0, "y": -2.0},
            {"x": 0.0, "y": 0.2},
            {"x": 2.5, "y": 5.2},
            {"x": 10.0, "y": 19.8},
        ],
    }

    metrics = evaluate_data_series(data_series, str(truth_data), "curve")

    assert metrics["data_success"]
    assert metrics["data_pred_point_count"] == 4
    assert metrics["data_comparable_point_count"] == 3
    assert metrics["data_x_coverage_ratio"] == 1.0
    assert metrics["data_y_rmse"] == pytest.approx(0.2)
    assert metrics["data_y_mae"] == pytest.approx(0.2)
    assert metrics["data_r2_at_pred_x"] > 0.999


def test_gap_linking_respects_tangent_angle_limit():
    config = PathTracingConfig(
        max_gap_px=20.0,
        max_gap_angle_deg=80.0,
        max_gap_tangent_angle_deg=30.0,
    )

    aligned = _interpolate_gap(
        (0, 0),
        (10, 0),
        config,
        prev_tangent=(5.0, 0.0),
        next_tangent=(5.0, 0.0),
    )
    mismatched = _interpolate_gap(
        (0, 0),
        (10, 0),
        config,
        prev_tangent=(0.0, 5.0),
        next_tangent=(5.0, 0.0),
    )

    assert aligned
    assert mismatched == []


def test_tangent_gap_interpolation_uses_smooth_curve_when_tangents_available():
    linear = _interpolate_gap_points(
        (0, 0),
        (10, 0),
        10,
        prev_tangent=(5.0, 5.0),
        next_tangent=(5.0, -5.0),
        use_tangent=False,
    )
    smooth = _interpolate_gap_points(
        (0, 0),
        (10, 0),
        10,
        prev_tangent=(5.0, 5.0),
        next_tangent=(5.0, -5.0),
        use_tangent=True,
    )

    assert linear
    assert all(y == 0 for _, y in linear)
    assert smooth
    assert max(y for _, y in smooth) > 0


def test_short_spur_pruning_removes_endpoint_branch_at_junction():
    main = {(x, 0) for x in range(11)}
    spur = {(5, -1), (5, -2), (5, -3)}
    pruned = _prune_short_spurs(main | spur, max_spur_length=4)

    assert main <= pruned
    assert not (spur & pruned)


def test_short_spur_pruning_preserves_simple_dash_component():
    dash = {(x, 0) for x in range(5)}
    pruned = _prune_short_spurs(dash, max_spur_length=4)

    assert pruned == dash


def test_mask_filter_removes_axis_ticks_but_keeps_inner_curve_segments():
    mask = np.zeros((100, 160), dtype=np.uint8)
    mask[20:23, 0:3] = 255
    mask[97:100, 42:45] = 255
    mask[45:47, 50:80] = 255
    mask[47:49, 80:108] = 255
    mask[49:51, 108:132] = 255

    filtered = _filter_components(mask, min_area=4, min_x_span=3)

    assert int(filtered[20:23, 0:16].sum()) == 0
    assert int(filtered[88:100, 42:45].sum()) == 0
    assert int(filtered[45:51, 50:132].sum()) > 0


def test_path_extractor_skips_compact_marker_components_when_line_anchor_exists():
    mask = np.zeros((80, 160), dtype=np.uint8)
    cv2.line(mask, (10, 40), (145, 40), 255, 3)
    for center in ((35, 22), (70, 58), (105, 22)):
        cv2.circle(mask, center, 6, 255, -1)

    config = PathTracingConfig(filter_marker_like_components=True)
    path = LinePathExtractor(config).extract_from_mask_image(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), curve_id="curve")
    ys = [point.y for point in path.pixel_points_ordered]

    assert len(path.pixel_points_ordered) >= 120
    assert min(ys) >= 38
    assert max(ys) <= 42
    assert any("marker_like_components_skipped=3" == warning for warning in path.warnings)


def test_path_extractor_keeps_dotted_curve_when_no_line_anchor_exists():
    mask = np.zeros((80, 160), dtype=np.uint8)
    for x in range(12, 145, 16):
        cv2.circle(mask, (x, 40), 3, 255, -1)

    config = PathTracingConfig(filter_marker_like_components=True)
    path = LinePathExtractor(config).extract_from_mask_image(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), curve_id="curve")

    assert path.component_count >= 8
    assert not any(warning.startswith("marker_like_components_skipped=") for warning in path.warnings)


def test_segment_connection_cost_rejects_large_backward_jump():
    current = _segment((0, 0), (10, 0), (5.0, 0.0), (5.0, 0.0))
    candidate = _segment((6, 0), (12, 0), (5.0, 0.0), (5.0, 0.0))

    assert _segment_connection_cost(current, candidate, PathTracingConfig(backward_x_tolerance_px=2.0)) is None


def test_segment_ordering_prefers_direction_continuity_over_nearest_turn():
    config = PathTracingConfig()
    current = _segment((0, 0), (10, 0), (5.0, 0.0), (5.0, 0.0))
    nearest_wrong_turn = _segment((12, 0), (12, 8), (0.0, 5.0), (0.0, 5.0))
    farther_continuation = _segment((16, 0), (24, 0), (5.0, 0.0), (5.0, 0.0))

    ordered = _order_segments([farther_continuation, nearest_wrong_turn, current], config)

    assert ordered[0] == current
    assert ordered[1] == farther_continuation
    assert ordered[2] == nearest_wrong_turn


def test_image_legend_detection_finds_in_plot_legend_without_ocr(tmp_path):
    manifest = generate_benchmark(
        str(tmp_path / "synthetic"),
        "legend_inside",
        SyntheticConfig(seed=42, n_curves=6, palette="basic", legend_inside=True),
    )
    image = load_bgr(manifest["image_path"])
    plot_area = _truth_plot_area(manifest["truth_axes_path"])
    legends = LegendDetector().detect_from_image(image, plot_area)

    assert any(legend.source == "image_heuristic" for legend in legends)

    legend = next(legend for legend in legends if legend.source == "image_heuristic")
    plot = plot_area.bbox
    assert legend.bbox.x_min <= plot.x_min + 30
    assert legend.bbox.y_min <= plot.y_min + 30
    assert legend.bbox.width >= 90
    assert legend.bbox.height >= 90


def test_image_legend_detection_finds_framed_lower_right_legend(tmp_path):
    manifest = generate_benchmark(
        str(tmp_path / "synthetic"),
        "legend_lower_right",
        SyntheticConfig(seed=42, n_curves=6, palette="basic", legend_inside=True, legend_loc="lower right"),
    )
    image = load_bgr(manifest["image_path"])
    plot_area = _truth_plot_area(manifest["truth_axes_path"])
    legends = LegendDetector().detect_from_image(image, plot_area)

    assert any(legend.source == "image_heuristic" for legend in legends)

    legend = next(legend for legend in legends if legend.source == "image_heuristic")
    plot = plot_area.bbox
    assert legend.bbox.x_max >= plot.x_max - 30
    assert legend.bbox.y_max >= plot.y_max - 30
    assert legend.bbox.width >= 90
    assert legend.bbox.height >= 90


def test_image_legend_detection_does_not_flag_outside_legend(tmp_path):
    manifest = generate_benchmark(
        str(tmp_path / "synthetic"),
        "legend_outside",
        SyntheticConfig(seed=42, n_curves=6, palette="basic", legend_inside=False),
    )
    image = load_bgr(manifest["image_path"])
    plot_area = _truth_plot_area(manifest["truth_axes_path"])
    legends = LegendDetector().detect_from_image(image, plot_area)

    assert [legend for legend in legends if legend.source == "image_heuristic"] == []


def _truth_plot_area(path: str) -> PlotArea:
    with open(path, "r", encoding="utf-8") as f:
        truth = json.load(f)
    bbox = truth["plot_area"]
    return PlotArea(BoundingBox(bbox["x_min"], bbox["y_min"], bbox["x_max"], bbox["y_max"]))


def _segment(start, end, start_tangent, end_tangent):
    min_x = min(start[0], end[0])
    max_x = max(start[0], end[0])
    return {
        "path": [start, end],
        "length": 8,
        "min_x": min_x,
        "max_x": max_x,
        "center_x": (start[0] + end[0]) / 2,
        "center_y": (start[1] + end[1]) / 2,
        "start": start,
        "end": end,
        "start_tangent": start_tangent,
        "end_tangent": end_tangent,
    }


def test_synthetic_benchmark_suite_metrics_stay_within_thresholds(tmp_path):
    suite = run_suite(str(tmp_path / "suite"))
    cases = {case["name"]: case for case in suite["cases"]}

    assert set(cases) == {
        "basic_curves",
        "achromatic_curves",
        "local_occlusion_curves",
        "crossing_curves",
        "legend_inside_curves",
    }

    for name in ("basic_curves", "achromatic_curves"):
        metrics = cases[name]["predicted_metrics"]
        assert metrics["curve_count"] == 6
        assert metrics["valid_curve_count"] == 6
        assert metrics["prototype_count"] == 6
        assert metrics["mean_chamfer_distance_px"] < 1.25
        assert metrics["mean_hausdorff_distance_px"] < 8.0
        assert metrics["mean_mask_tolerant_f1"] > 0.85
        assert metrics["mean_data_y_rmse"] < 0.03
        assert metrics["mean_data_r2_at_pred_x"] > 0.99
        assert metrics["mean_data_x_coverage_ratio"] > 0.95

    occlusion_metrics = cases["local_occlusion_curves"]["predicted_metrics"]
    assert occlusion_metrics["curve_count"] == 6
    assert occlusion_metrics["valid_curve_count"] == 6
    assert occlusion_metrics["prototype_count"] >= 6
    assert occlusion_metrics["mean_chamfer_distance_px"] < 1.35
    assert occlusion_metrics["mean_hausdorff_distance_px"] < 16.0
    assert occlusion_metrics["mean_completed_point_ratio"] > 0.30
    assert occlusion_metrics["mean_mask_tolerant_f1"] > 0.85
    assert occlusion_metrics["mean_data_y_rmse"] < 0.03
    assert occlusion_metrics["mean_data_r2_at_pred_x"] > 0.99
    assert occlusion_metrics["mean_data_x_coverage_ratio"] > 0.95

    crossing_metrics = cases["crossing_curves"]["predicted_metrics"]
    assert crossing_metrics["curve_count"] == 6
    assert crossing_metrics["valid_curve_count"] == 6
    assert crossing_metrics["prototype_count"] >= 6
    assert crossing_metrics["mean_chamfer_distance_px"] < 1.35
    assert crossing_metrics["mean_hausdorff_distance_px"] < 12.0
    assert crossing_metrics["mean_mask_tolerant_f1"] > 0.85
    assert crossing_metrics["mean_data_y_rmse"] < 0.03
    assert crossing_metrics["mean_data_r2_at_pred_x"] > 0.99
    assert crossing_metrics["mean_data_x_coverage_ratio"] > 0.95

    legend_case = cases["legend_inside_curves"]
    polluted = legend_case["predicted_metrics"]
    excluded = legend_case["predicted_exclude_legend_metrics"]
    detected = legend_case["predicted_detected_legend_metrics"]
    delta = legend_case["legend_exclusion_delta"]
    detected_delta = legend_case["detected_legend_delta"]

    assert polluted["valid_curve_count"] == 6
    assert polluted["mean_hausdorff_distance_px"] > 100.0
    assert polluted["mean_data_y_rmse"] > 0.10
    assert excluded["valid_curve_count"] == 6
    assert excluded["mean_chamfer_distance_px"] < 1.25
    assert excluded["mean_hausdorff_distance_px"] < 8.0
    assert excluded["mean_mask_tolerant_f1"] > 0.85
    assert excluded["mean_data_y_rmse"] < 0.03
    assert excluded["mean_data_r2_at_pred_x"] > 0.99
    assert delta["mean_hausdorff_distance_px"] > 100.0
    assert delta["mean_data_y_rmse"] > 0.10
    assert detected["detected_legend_count"] == 1
    assert detected["legend_exclusion_source"] == "image_heuristic"
    assert detected["valid_curve_count"] == 6
    assert detected["mean_chamfer_distance_px"] < 1.25
    assert detected["mean_hausdorff_distance_px"] < 8.0
    assert detected["mean_mask_tolerant_f1"] > 0.85
    assert detected["mean_data_y_rmse"] < 0.03
    assert detected["mean_data_r2_at_pred_x"] > 0.99
    assert detected_delta["mean_hausdorff_distance_px"] > 100.0
    assert detected_delta["mean_data_y_rmse"] > 0.10
