from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import numpy as np
import cv2

from graph2data.benchmark import run_path_benchmark, run_prototype_binding_benchmark, run_suite
from graph2data.image_io import load_bgr
from graph2data.instances import group_marker_curve_instances
from graph2data.legend import LegendDetector
from graph2data.lines import (
    LinePathExtractor,
    PathTracingConfig,
    _interpolate_gap,
    _interpolate_gap_points,
    _order_segments,
    _prune_short_spurs,
    _segment_connection_cost,
    _smooth_projection_path,
)
from graph2data.mapping import map_curve_path_to_data, map_curve_paths_to_data, write_data_series_csv
from graph2data.masks import _filter_components
from graph2data.models import BoundingBox, CurvePath, CurveVisualPrototype, DataRange, LegendItem, LineComponentClassification, LineStyleCurveInstance, MarkerCurveInstance, OCRTextBox, PlotArea, Point, PrototypeBinding
from graph2data.pipeline import _prototype_bound_curve_paths, _prototype_bound_line_style_paths, _prototype_bound_marker_paths
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
        label="Curve A",
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
    assert series.label == "Curve A"
    assert series.completed_point_count == 1
    assert series.points[0].x == 0.0
    assert series.points[0].y == -1.0
    assert series.points[1].x == 5.0
    assert series.points[1].y == 0.0
    assert series.points[1].completed
    assert series.points[1].confidence == 0.35
    assert series.points[2].x == 10.0
    assert series.points[2].y == 1.0


def test_data_series_csv_includes_curve_label(tmp_path):
    curve_path = CurvePath(
        curve_id="curve",
        label="Curve A",
        pixel_points_ordered=[Point(10.0, 220.0), Point(110.0, 20.0)],
    )
    plot_area = PlotArea(BoundingBox(10.0, 20.0, 110.0, 220.0))
    data_range = DataRange(0.0, 10.0, -1.0, 1.0)
    series = map_curve_path_to_data(curve_path, plot_area, data_range)
    csv_path = tmp_path / "curves.csv"

    write_data_series_csv(str(csv_path), [series])

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["curve_id"] == "curve"
    assert rows[0]["label"] == "Curve A"
    assert rows[0]["point_index"] == "0"


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


def test_path_extractor_rebuilds_low_coverage_marker_junction_path():
    mask = np.zeros((120, 240), dtype=np.uint8)
    cv2.line(mask, (10, 60), (230, 60), 255, 2)
    for x in np.linspace(30, 210, 8).astype(int):
        cv2.line(mask, (int(x), 15), (int(x), 105), 255, 2)

    path = LinePathExtractor(PathTracingConfig()).extract_from_mask_image(
        cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR),
        curve_id="line_marker",
    )

    assert "path_rebuilt_from_skeleton_x_projection" in path.warnings
    xs = [point.x for point in path.pixel_points_ordered]
    assert min(xs) <= 12
    assert max(xs) >= 228


def test_projection_path_smoothing_reduces_local_jitter():
    jagged = [(idx, 50 + (8 if idx % 2 else -8)) for idx in range(20)]
    smoothed = _smooth_projection_path(jagged, window_px=3)

    raw_jump = max(abs(a[1] - b[1]) for a, b in zip(jagged, jagged[1:]))
    smooth_jump = max(abs(a[1] - b[1]) for a, b in zip(smoothed, smoothed[1:]))
    assert smooth_jump < raw_jump
    assert [point[0] for point in smoothed] == [point[0] for point in jagged]


def test_component_classifier_distinguishes_line_marker_and_noise():
    mask = np.zeros((90, 180), dtype=np.uint8)
    cv2.line(mask, (10, 45), (150, 45), 255, 3)
    cv2.line(mask, (35, 12), (35, 24), 255, 2)
    cv2.line(mask, (29, 18), (41, 18), 255, 2)
    mask[72, 165] = 255

    extractor = LinePathExtractor(PathTracingConfig())
    components = extractor.classify_components_from_mask_image(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), curve_id="curve")
    labels = {component.class_label for component in components}

    assert "line_like" in labels
    assert "marker_like" in labels
    assert "noise" in labels


def test_marker_candidate_detector_uses_original_mask_shape_features():
    mask = np.zeros((90, 180), dtype=np.uint8)
    cv2.circle(mask, (25, 25), 7, 255, -1)
    cv2.rectangle(mask, (60, 18), (74, 32), 255, -1)
    cv2.line(mask, (112, 18), (112, 32), 255, 2)
    cv2.line(mask, (105, 25), (119, 25), 255, 2)

    extractor = LinePathExtractor(PathTracingConfig())
    markers = extractor.detect_marker_candidates_from_mask_image(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), curve_id="curve")
    shapes = {marker.shape for marker in markers}

    assert len(markers) == 3
    assert "circle_like" in shapes
    assert "square_like" in shapes
    assert "cross_like" in shapes
    assert all(marker.center.x > 0 and marker.center.y > 0 for marker in markers)


def test_marker_candidate_detector_finds_blobs_connected_to_line():
    mask = np.zeros((90, 220), dtype=np.uint8)
    cv2.line(mask, (10, 45), (205, 45), 255, 2)
    for center in ((45, 45), (100, 45), (155, 45)):
        cv2.circle(mask, center, 7, 255, -1)

    extractor = LinePathExtractor(PathTracingConfig())
    markers = extractor.detect_marker_candidates_from_mask_image(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), curve_id="curve")

    assert len(markers) >= 3
    assert sum(1 for marker in markers if "distance_transform_candidate" in marker.warnings) >= 3


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


def test_legend_item_extraction_splits_detected_legend_into_rows(tmp_path):
    manifest = generate_benchmark(
        str(tmp_path / "synthetic"),
        "legend_inside_items",
        SyntheticConfig(seed=42, n_curves=6, palette="basic", legend_inside=True),
    )
    image = load_bgr(manifest["image_path"])
    plot_area = _truth_plot_area(manifest["truth_axes_path"])
    detector = LegendDetector()
    legends = detector.detect_from_image(image, plot_area)

    assert legends

    items = detector.extract_items(image, legends)
    prototypes = detector.visual_prototypes_from_items(items)

    assert len(items) >= 4
    assert len(prototypes) >= 4
    assert all(item.sample_bbox is not None for item in items)
    assert any(item.text_bbox is not None for item in items)
    assert any(item.line_style in {"solid", "dashed", "dotted"} for item in items)
    assert all(item.foreground_pixel_count > 0 for item in items)


def test_legend_item_extraction_records_marker_style_when_visible(tmp_path):
    manifest = generate_benchmark(
        str(tmp_path / "synthetic"),
        "same_color_marker_legend_items",
        SyntheticConfig(seed=22, n_curves=4, same_color_marker_curves=True, legend_inside=True),
    )
    image = load_bgr(manifest["image_path"])
    plot_area = _truth_plot_area(manifest["truth_axes_path"])
    detector = LegendDetector()
    legends = detector.detect_from_image(image, plot_area)
    items = detector.extract_items(image, legends)

    assert len(items) == 4
    assert any(item.marker_style != "unknown" for item in items)


def test_legend_item_extraction_matches_same_gray_linestyle_curve_count(tmp_path):
    manifest = generate_benchmark(
        str(tmp_path / "synthetic"),
        "same_gray_linestyle_legend_items",
        SyntheticConfig(seed=23, n_curves=6, same_gray_linestyle_curves=True, legend_inside=True),
    )
    image = load_bgr(manifest["image_path"])
    plot_area = _truth_plot_area(manifest["truth_axes_path"])
    detector = LegendDetector()
    legends = detector.detect_from_image(image, plot_area)
    items = detector.extract_items(image, legends)

    assert len(items) == 6


def test_legend_item_labels_are_assigned_from_ocr_text_boxes():
    detector = LegendDetector()
    item = LegendItem(
        item_id="legend_00_item_00",
        legend_index=0,
        bbox=BoundingBox(10, 10, 120, 30),
        sample_bbox=BoundingBox(10, 10, 50, 30),
        text_bbox=BoundingBox(55, 10, 120, 30),
    )
    texts = [
        OCRTextBox(
            text="Curve",
            confidence=0.92,
            polygon=[],
            bbox=BoundingBox(58, 12, 86, 26),
            center=Point(72, 19),
        ),
        OCRTextBox(
            text="A",
            confidence=0.90,
            polygon=[],
            bbox=BoundingBox(91, 12, 103, 26),
            center=Point(97, 19),
        ),
    ]

    detector.assign_item_labels_from_ocr([item], texts)
    prototypes = detector.visual_prototypes_from_items(
        [
            LegendItem(
                item_id=item.item_id,
                legend_index=item.legend_index,
                bbox=item.bbox,
                sample_bbox=item.sample_bbox,
                text_bbox=item.text_bbox,
                rgb=(10, 20, 30),
                lab=(1.0, 2.0, 3.0),
                label=item.label,
                confidence=0.8,
            )
        ]
    )

    assert item.label == "Curve A"
    assert prototypes[0].label == "Curve A"


def test_synthetic_next_stage_visual_cases_emit_truth_metadata(tmp_path):
    cases = {
        "marker_curves": SyntheticConfig(seed=21, n_curves=4, marker_curves=True),
        "line_marker_curves": SyntheticConfig(seed=25, n_curves=4, line_marker_curves=True),
        "crossing_line_marker_curves": SyntheticConfig(seed=26, n_curves=4, crossing_line_marker_curves=True),
        "same_color_line_marker_curves": SyntheticConfig(seed=27, n_curves=4, same_color_line_marker_curves=True),
        "same_gray_line_marker_curves": SyntheticConfig(seed=28, n_curves=4, same_gray_line_marker_curves=True),
        "crossing_same_gray_line_marker_curves": SyntheticConfig(
            seed=29,
            n_curves=4,
            crossing_same_gray_line_marker_curves=True,
        ),
        "same_color_marker_curves": SyntheticConfig(seed=22, n_curves=4, same_color_marker_curves=True),
        "same_gray_linestyle_curves": SyntheticConfig(seed=23, n_curves=6, same_gray_linestyle_curves=True),
        "dense_legend_curves": SyntheticConfig(seed=24, n_curves=10, dense_legend_curves=True),
    }

    for name, config in cases.items():
        manifest = generate_benchmark(str(tmp_path / "synthetic"), name, config)
        root = Path(manifest["image_path"]).parent
        with open(manifest["truth_curves_path"], "r", encoding="utf-8") as f:
            curves = json.load(f)["curves"]

        assert Path(manifest["image_path"]).is_file()
        assert len(curves) == manifest["curve_count"]
        assert all((root / curve["mask_path"]).is_file() for curve in curves)
        assert all(cv2.countNonZero(cv2.imread(str(root / curve["mask_path"]), cv2.IMREAD_GRAYSCALE)) > 0 for curve in curves)

        if name == "marker_curves":
            assert all(curve["linestyle"] == "None" for curve in curves)
            assert len({curve["marker"] for curve in curves}) == len(curves)
        if name == "line_marker_curves":
            assert all(curve["linestyle"] == "-" for curve in curves)
            assert all(curve["marker"] for curve in curves)
            assert len({curve["color"] for curve in curves}) == len(curves)
        if name == "crossing_line_marker_curves":
            assert all(curve["linestyle"] == "-" for curve in curves)
            assert all(curve["marker"] for curve in curves)
            assert len({curve["color"] for curve in curves}) == len(curves)
            assert manifest["synthetic_config"]["crossing_curves"]
            assert manifest["synthetic_config"]["line_marker_curves"]
        if name == "same_color_line_marker_curves":
            assert all(curve["linestyle"] == "-" for curve in curves)
            assert all(curve["marker"] for curve in curves)
            assert len({curve["color"] for curve in curves}) == 1
            assert manifest["synthetic_config"]["line_marker_curves"]
            assert manifest["synthetic_config"]["same_color_marker_curves"]
        if name == "same_gray_line_marker_curves":
            assert all(curve["linestyle"] == "-" for curve in curves)
            assert all(curve["marker"] for curve in curves)
            assert all(_is_gray_hex(curve["color"]) for curve in curves)
            assert len({curve["color"] for curve in curves}) == len(curves)
            assert manifest["synthetic_config"]["line_marker_curves"]
            assert manifest["synthetic_config"]["same_gray_line_marker_curves"]
        if name == "crossing_same_gray_line_marker_curves":
            assert all(curve["linestyle"] == "-" for curve in curves)
            assert all(curve["marker"] for curve in curves)
            assert all(_is_gray_hex(curve["color"]) for curve in curves)
            assert len({curve["color"] for curve in curves}) == len(curves)
            assert manifest["synthetic_config"]["crossing_curves"]
            assert manifest["synthetic_config"]["line_marker_curves"]
            assert manifest["synthetic_config"]["same_gray_line_marker_curves"]
        if name == "same_color_marker_curves":
            assert len({curve["color"] for curve in curves}) == 1
            assert len({curve["marker"] for curve in curves}) == len(curves)
        if name == "same_gray_linestyle_curves":
            assert all(_is_gray_hex(curve["color"]) for curve in curves)
            assert len({str(curve["linestyle"]) for curve in curves}) >= 4
        if name == "dense_legend_curves":
            assert manifest["curve_count"] >= 10
            assert manifest["synthetic_config"]["legend_inside"]
            assert all(curve["marker"] for curve in curves)


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


def _is_gray_hex(value: str) -> bool:
    value = value.strip().lstrip("#")
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return r == g == b


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


def test_pipeline_debug_artifacts_include_legend_items_overlay(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    result = GraphExtractionPipeline(
        PipelineConfig(
            run_paths=True,
            artifact_dir=str(artifact_dir),
            write_debug_artifacts=True,
        )
    ).run(str(ROOT / "tests" / "test1.png"))

    debug = result.artifacts.get("debug", {})
    assert Path(debug["overview"]).is_file()
    assert Path(debug["legend_items"]).is_file()
    assert isinstance(result.legend_items, list)


def test_pipeline_debug_artifacts_include_component_classification_overlay(tmp_path):
    artifact_dir = tmp_path / "artifacts_components"
    result = GraphExtractionPipeline(
        PipelineConfig(
            run_paths=True,
            artifact_dir=str(artifact_dir),
            write_debug_artifacts=True,
        )
    ).run(str(ROOT / "tests" / "test1.png"))

    debug = result.artifacts.get("debug", {})
    assert Path(debug["component_classification"]).is_file()
    assert len(result.line_components) > 0

    with open(result.artifacts["quality_report"], "r", encoding="utf-8") as f:
        quality_report = json.load(f)
    assert quality_report["counts"]["line_component_count"] == len(result.line_components)
    assert "component_summary" in quality_report


def test_pipeline_debug_artifacts_include_marker_candidates_overlay(tmp_path):
    artifact_dir = tmp_path / "artifacts_markers"
    result = GraphExtractionPipeline(
        PipelineConfig(
            run_paths=True,
            artifact_dir=str(artifact_dir),
            write_debug_artifacts=True,
        )
    ).run(str(ROOT / "tests" / "test1.png"))

    debug = result.artifacts.get("debug", {})
    assert Path(debug["marker_candidates"]).is_file()
    assert Path(debug["marker_curve_instances"]).is_file()
    assert isinstance(result.marker_candidates, list)
    assert isinstance(result.marker_curve_instances, list)

    with open(result.artifacts["quality_report"], "r", encoding="utf-8") as f:
        quality_report = json.load(f)
    assert quality_report["counts"]["marker_candidate_count"] == len(result.marker_candidates)
    assert quality_report["counts"]["marker_curve_instance_count"] == len(result.marker_curve_instances)
    assert "marker_summary" in quality_report
    assert "marker_instance_summary" in quality_report


def test_marker_curve_instances_group_candidates_by_local_rank():
    mask = np.zeros((120, 240), dtype=np.uint8)
    for x in (40, 90, 140, 190):
        for y in (30, 60, 90):
            cv2.circle(mask, (x, y), 7, 255, -1)

    extractor = LinePathExtractor(PathTracingConfig())
    markers = extractor.detect_marker_candidates_from_mask_image(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), curve_id="curve")
    instances = group_marker_curve_instances(markers, group_count=3)

    assert len(instances) == 3
    assert all(instance.marker_count == 4 for instance in instances)
    assert {instance.grouping_method for instance in instances} == {"x_rank"}


def test_pipeline_quality_report_includes_prototype_binding_scores(tmp_path):
    artifact_dir = tmp_path / "artifacts_bindings"
    result = GraphExtractionPipeline(
        PipelineConfig(
            run_paths=True,
            artifact_dir=str(artifact_dir),
            write_debug_artifacts=True,
        )
    ).run(str(ROOT / "tests" / "test1.png"))

    assert result.curve_visual_prototypes
    assert result.prototype_bindings
    assert all(0.0 <= binding.score <= 1.0 for binding in result.prototype_bindings)

    with open(result.artifacts["quality_report"], "r", encoding="utf-8") as f:
        quality_report = json.load(f)
    assert quality_report["counts"]["prototype_binding_count"] == len(result.prototype_bindings)
    assert quality_report["binding_summary"]["binding_count"] == len(result.prototype_bindings)
    assert quality_report["binding_summary"]["best_by_prototype"]


def test_pipeline_outputs_prototype_bound_marker_paths(tmp_path):
    prototype = CurveVisualPrototype(
        prototype_id="legend_proto_00",
        legend_item_id="legend_00_item_00",
        label="Marker Curve",
        confidence=0.9,
    )
    instance = MarkerCurveInstance(
        instance_id="marker_instance_00",
        source_curve_id="curve_color_0",
        marker_ids=["m0", "m1", "m2"],
        points=[Point(30, 50), Point(10, 70), Point(20, 60)],
        marker_count=3,
        confidence=0.8,
    )
    binding = PrototypeBinding(
        binding_id="binding_0000",
        prototype_id=prototype.prototype_id,
        legend_item_id=prototype.legend_item_id,
        target_curve_id=instance.instance_id,
        target_type="marker_instance",
        score=0.95,
        confidence=0.85,
    )

    source_curve_path = CurvePath(
        curve_id="curve_color_0",
        pixel_points_ordered=[Point(10, 70), Point(20, 60), Point(30, 50), Point(40, 40)],
        confidence_per_point=[0.9, 0.9, 0.9, 0.9],
        endpoints=[Point(10, 70), Point(40, 40)],
        component_count=1,
        observed_pixel_count=4,
        path_length_px=42.0,
        confidence=0.88,
    )

    paths = _prototype_bound_marker_paths([prototype], [binding], [instance], [source_curve_path])

    assert len(paths) == 1
    path = paths[0]
    assert path.curve_id == "legend_proto_00_bound_path"
    assert path.label == "Marker Curve"
    assert [(point.x, point.y) for point in path.pixel_points_ordered] == [(10, 70), (20, 60), (30, 50), (40, 40)]
    assert path.observed_pixel_count == 4
    assert path.confidence == pytest.approx(0.8)
    assert "prototype_bound_marker_path" in path.warnings
    assert "marker_path_uses_source_curve_path" in path.warnings


def test_pipeline_outputs_prototype_bound_curve_paths():
    prototype = CurveVisualPrototype(
        prototype_id="legend_proto_02",
        legend_item_id="legend_00_item_02",
        label="Curve Path",
        confidence=0.9,
    )
    source_curve_path = CurvePath(
        curve_id="curve_color_2",
        pixel_points_ordered=[Point(15, 80), Point(25, 70), Point(35, 60)],
        completed_ranges=[(1, 1)],
        confidence_per_point=[0.95, 0.4, 0.95],
        endpoints=[Point(15, 80), Point(35, 60)],
        component_count=1,
        observed_pixel_count=2,
        completed_pixel_count=1,
        path_length_px=28.0,
        confidence=0.86,
    )
    binding = PrototypeBinding(
        binding_id="binding_0002",
        prototype_id=prototype.prototype_id,
        legend_item_id=prototype.legend_item_id,
        target_curve_id="curve_color_2",
        target_type="curve",
        score=0.97,
        confidence=0.83,
    )

    paths = _prototype_bound_curve_paths([prototype], [binding], [source_curve_path])

    assert len(paths) == 1
    path = paths[0]
    assert path.curve_id == "legend_proto_02_bound_path"
    assert path.label == "Curve Path"
    assert [(point.x, point.y) for point in path.pixel_points_ordered] == [(15, 80), (25, 70), (35, 60)]
    assert path.completed_ranges == [(1, 1)]
    assert path.completed_pixel_count == 1
    assert path.confidence_per_point == [0.95, 0.4, 0.95]
    assert "prototype_bound_curve_path" in path.warnings


def test_prototype_bound_curve_path_rebuilds_from_line_components_when_source_path_is_short():
    prototype = CurveVisualPrototype(
        prototype_id="legend_proto_03",
        legend_item_id="legend_00_item_03",
        label="Rebuilt Curve",
        confidence=0.9,
    )
    source_curve_path = CurvePath(
        curve_id="curve_color_3",
        pixel_points_ordered=[Point(10, 80), Point(20, 78)],
        confidence_per_point=[0.7, 0.7],
        endpoints=[Point(10, 80), Point(20, 78)],
        component_count=1,
        observed_pixel_count=100,
        path_length_px=10.0,
        confidence=0.7,
    )
    line_component = LineComponentClassification(
        curve_id="curve_color_3",
        component_id="component_00",
        class_label="line_like",
        bbox=BoundingBox(10, 70, 110, 82),
        pixel_count=120,
        width=100,
        height=12,
        max_span=100,
        aspect_ratio=100 / 12,
        path_points=[Point(x, 80 - x * 0.1) for x in range(10, 111, 10)],
        path_length_px=100,
        confidence=0.8,
    )
    binding = PrototypeBinding(
        binding_id="binding_0003",
        prototype_id=prototype.prototype_id,
        legend_item_id=prototype.legend_item_id,
        target_curve_id="curve_color_3",
        target_type="curve",
        score=0.97,
        confidence=0.83,
    )

    paths = _prototype_bound_curve_paths([prototype], [binding], [source_curve_path], [line_component])

    assert len(paths) == 1
    path = paths[0]
    assert len(path.pixel_points_ordered) > len(source_curve_path.pixel_points_ordered)
    assert path.observed_pixel_count == len(line_component.path_points)
    assert path.path_length_px > source_curve_path.path_length_px
    assert "curve_path_rebuilt_from_line_components" in path.warnings


def test_pipeline_maps_prototype_bound_marker_paths_to_data(tmp_path):
    prototype = CurveVisualPrototype(
        prototype_id="legend_proto_00",
        legend_item_id="legend_00_item_00",
        label="Marker Curve",
        confidence=0.9,
    )
    instance = MarkerCurveInstance(
        instance_id="marker_instance_00",
        source_curve_id="curve_color_0",
        marker_ids=["m0", "m1", "m2"],
        points=[Point(10, 220), Point(60, 120), Point(110, 20)],
        marker_count=3,
        confidence=0.8,
    )
    binding = PrototypeBinding(
        binding_id="binding_0000",
        prototype_id=prototype.prototype_id,
        legend_item_id=prototype.legend_item_id,
        target_curve_id=instance.instance_id,
        target_type="marker_instance",
        score=0.95,
        confidence=0.85,
    )
    bound_paths = _prototype_bound_marker_paths([prototype], [binding], [instance])
    plot_area = PlotArea(BoundingBox(10.0, 20.0, 110.0, 220.0))
    data_range = DataRange(0.0, 10.0, -1.0, 1.0)

    series_list = map_curve_paths_to_data(bound_paths, plot_area, data_range)
    csv_path = tmp_path / "prototype_bound_curves.csv"
    write_data_series_csv(str(csv_path), series_list)

    assert len(series_list) == 1
    assert series_list[0].curve_id == "legend_proto_00_bound_path"
    assert series_list[0].label == "Marker Curve"
    assert series_list[0].point_count == 3
    assert series_list[0].points[0].x == 0.0
    assert series_list[0].points[0].y == -1.0
    assert csv_path.is_file()


def test_prototype_bound_line_style_paths_use_component_centers():
    prototype = CurveVisualPrototype(
        prototype_id="legend_proto_01",
        legend_item_id="legend_00_item_01",
        label="Dashed Curve",
        line_style="dashed",
        confidence=0.9,
    )
    instance = LineStyleCurveInstance(
        instance_id="line_style_instance_00",
        source_curve_id="curve_color_0",
        component_ids=["c0", "c1", "c2"],
        points=[Point(50, 120), Point(10, 140), Point(30, 130)],
        component_count=3,
        estimated_line_style="dashed",
        confidence=0.75,
    )
    binding = PrototypeBinding(
        binding_id="binding_0001",
        prototype_id=prototype.prototype_id,
        legend_item_id=prototype.legend_item_id,
        target_curve_id=instance.instance_id,
        target_type="line_style_instance",
        score=0.92,
        confidence=0.82,
    )

    paths = _prototype_bound_line_style_paths([prototype], [binding], [instance])

    assert len(paths) == 1
    path = paths[0]
    assert path.curve_id == "legend_proto_01_bound_path"
    assert path.label == "Dashed Curve"
    assert (path.pixel_points_ordered[0].x, path.pixel_points_ordered[0].y) == (10, 140)
    assert (path.pixel_points_ordered[-1].x, path.pixel_points_ordered[-1].y) == (50, 120)
    assert len(path.pixel_points_ordered) > 3
    assert path.completed_ranges
    assert path.completed_pixel_count > 0
    assert min(path.confidence_per_point) < path.confidence
    assert path.component_count == 3
    assert path.confidence == pytest.approx(0.75)
    assert "prototype_bound_line_style_path" in path.warnings


def test_prototype_binding_benchmark_reports_synthetic_metrics(tmp_path):
    line_marker_path_manifest = generate_benchmark(
        str(tmp_path / "synthetic"),
        "line_marker_path",
        SyntheticConfig(seed=25, n_curves=4, line_marker_curves=True, legend_inside=False),
    )
    line_marker_path = run_path_benchmark(str(Path(line_marker_path_manifest["image_path"]).parent))
    line_marker_path_summary = line_marker_path["summary"]
    assert line_marker_path_summary["valid_curve_count"] == line_marker_path_summary["curve_count"]
    assert line_marker_path_summary["mean_data_y_rmse"] is not None
    assert line_marker_path_summary["mean_data_y_rmse"] < 0.03
    assert line_marker_path_summary["mean_data_x_coverage_ratio"] is not None
    assert line_marker_path_summary["mean_data_x_coverage_ratio"] > 0.95
    assert line_marker_path_summary["mean_path_coverage_ratio"] is not None
    assert line_marker_path_summary["mean_path_coverage_ratio"] > 0.75
    assert line_marker_path_summary["rebuilt_path_count"] >= 0

    for name, config in {
        "crossing_line_marker_path": SyntheticConfig(seed=26, n_curves=4, crossing_line_marker_curves=True, legend_inside=False),
        "same_color_line_marker_path": SyntheticConfig(seed=27, n_curves=4, same_color_line_marker_curves=True, legend_inside=False),
        "same_gray_line_marker_path": SyntheticConfig(seed=28, n_curves=4, same_gray_line_marker_curves=True, legend_inside=False),
        "crossing_same_gray_line_marker_path": SyntheticConfig(
            seed=29,
            n_curves=4,
            crossing_same_gray_line_marker_curves=True,
            legend_inside=False,
        ),
    }.items():
        manifest = generate_benchmark(str(tmp_path / "synthetic"), name, config)
        path_result = run_path_benchmark(str(Path(manifest["image_path"]).parent))
        path_summary = path_result["summary"]
        assert path_summary["valid_curve_count"] == path_summary["curve_count"]
        assert path_summary["mean_data_y_rmse"] is not None
        assert path_summary["mean_data_y_rmse"] < 0.03
        assert path_summary["mean_data_x_coverage_ratio"] is not None
        assert path_summary["mean_data_x_coverage_ratio"] > 0.95
        assert path_summary["mean_path_coverage_ratio"] is not None
        assert path_summary["mean_path_coverage_ratio"] > 0.75
        assert path_summary["rebuilt_path_count"] >= 1

    line_marker_manifest = generate_benchmark(
        str(tmp_path / "synthetic"),
        "line_marker_binding",
        SyntheticConfig(seed=25, n_curves=4, line_marker_curves=True, legend_inside=True),
    )
    line_marker_result = run_prototype_binding_benchmark(str(Path(line_marker_manifest["image_path"]).parent))
    line_marker_summary = line_marker_result["summary"]
    assert line_marker_summary["legend_item_count"] == line_marker_summary["curve_count"]
    assert line_marker_summary["curve_visual_prototype_count"] == line_marker_summary["curve_count"]
    assert line_marker_summary["binding_accuracy"] is not None
    assert line_marker_summary["binding_accuracy"] >= 0.75
    assert line_marker_summary["prototype_bound_path_count"] >= line_marker_summary["curve_count"]
    assert line_marker_summary["valid_prototype_bound_path_count"] >= line_marker_summary["curve_count"]
    assert line_marker_summary["valid_prototype_bound_data_count"] >= line_marker_summary["curve_count"]
    assert line_marker_summary["labeled_prototype_bound_path_count"] >= line_marker_summary["curve_count"]
    assert line_marker_summary["labeled_prototype_bound_data_count"] >= line_marker_summary["curve_count"]
    assert line_marker_summary["mean_prototype_bound_data_y_rmse"] is not None
    assert line_marker_summary["mean_prototype_bound_data_y_rmse"] < 0.08
    assert line_marker_summary["mean_prototype_bound_data_x_coverage_ratio"] is not None
    assert line_marker_summary["mean_prototype_bound_data_x_coverage_ratio"] > 0.90
    assert all(path["label"] for path in line_marker_result["pipeline"]["prototype_bound_paths"])
    assert all(series["label"] for series in line_marker_result["pipeline"]["prototype_bound_data_series"])
    assert any(
        "prototype_bound_curve_path" in path["warnings"] or "marker_path_uses_source_curve_path" in path["warnings"]
        for path in line_marker_result["pipeline"]["prototype_bound_paths"]
    )

    for name, config in {
        "crossing_line_marker_binding": SyntheticConfig(seed=26, n_curves=4, crossing_line_marker_curves=True, legend_inside=True),
        "same_color_line_marker_binding": SyntheticConfig(seed=27, n_curves=4, same_color_line_marker_curves=True, legend_inside=True),
        "same_gray_line_marker_binding": SyntheticConfig(seed=28, n_curves=4, same_gray_line_marker_curves=True, legend_inside=True),
    }.items():
        manifest = generate_benchmark(str(tmp_path / "synthetic"), name, config)
        result = run_prototype_binding_benchmark(str(Path(manifest["image_path"]).parent))
        summary = result["summary"]
        assert summary["legend_item_count"] == summary["curve_count"]
        assert summary["curve_visual_prototype_count"] == summary["curve_count"]
        assert summary["binding_accuracy"] is not None
        assert summary["binding_accuracy"] >= 0.75
        assert summary["prototype_bound_path_count"] >= summary["curve_count"]
        assert summary["valid_prototype_bound_path_count"] >= summary["curve_count"]
        assert summary["valid_prototype_bound_data_count"] >= summary["curve_count"]
        assert summary["labeled_prototype_bound_path_count"] >= summary["curve_count"]
        assert summary["labeled_prototype_bound_data_count"] >= summary["curve_count"]
        assert summary["mean_prototype_bound_data_y_rmse"] is not None
        assert summary["mean_prototype_bound_data_y_rmse"] < 0.05
        assert summary["mean_prototype_bound_data_x_coverage_ratio"] is not None
        assert summary["mean_prototype_bound_data_x_coverage_ratio"] > 0.85

    crossing_gray_manifest = generate_benchmark(
        str(tmp_path / "synthetic"),
        "crossing_same_gray_line_marker_binding",
        SyntheticConfig(seed=29, n_curves=4, crossing_same_gray_line_marker_curves=True, legend_inside=True),
    )
    crossing_gray_result = run_prototype_binding_benchmark(str(Path(crossing_gray_manifest["image_path"]).parent))
    crossing_gray_summary = crossing_gray_result["summary"]
    assert crossing_gray_summary["legend_item_count"] == crossing_gray_summary["curve_count"]
    assert crossing_gray_summary["curve_visual_prototype_count"] == crossing_gray_summary["curve_count"]
    assert crossing_gray_summary["binding_accuracy"] == pytest.approx(1.0)
    assert crossing_gray_summary["prototype_bound_path_count"] == crossing_gray_summary["curve_count"]
    assert crossing_gray_summary["valid_prototype_bound_path_count"] == crossing_gray_summary["curve_count"]
    assert crossing_gray_summary["valid_prototype_bound_data_count"] == crossing_gray_summary["curve_count"]
    assert crossing_gray_summary["mean_prototype_bound_data_y_rmse"] is not None
    assert crossing_gray_summary["mean_prototype_bound_data_y_rmse"] < 0.25
    assert crossing_gray_summary["mean_prototype_bound_data_x_coverage_ratio"] is not None
    assert crossing_gray_summary["mean_prototype_bound_data_x_coverage_ratio"] > 0.85
    assert all(row["success"] for row in crossing_gray_result["prototypes"] if row["expected_curve_id"] is not None)
    assert all(row["best_target_type"] == "curve" for row in crossing_gray_result["prototypes"] if row["expected_curve_id"] is not None)

    manifest = generate_benchmark(
        str(tmp_path / "synthetic"),
        "same_color_marker_binding",
        SyntheticConfig(seed=22, n_curves=4, same_color_marker_curves=True, legend_inside=True),
    )

    result = run_prototype_binding_benchmark(str(Path(manifest["image_path"]).parent))
    summary = result["summary"]

    assert summary["mode"] == "prototype_binding"
    assert summary["legend_count"] >= 1
    assert summary["legend_item_count"] == summary["curve_count"]
    assert summary["curve_visual_prototype_count"] == summary["curve_count"]
    assert summary["binding_count"] >= summary["curve_visual_prototype_count"]
    assert summary["evaluated_prototype_count"] >= 1
    assert summary["binding_accuracy"] is not None
    assert 0.0 <= summary["binding_accuracy"] <= 1.0
    assert summary["truth_marker_count"] > 0
    assert summary["marker_candidate_recall"] is not None
    assert 0.0 <= summary["marker_candidate_recall"] <= 1.0
    assert summary["marker_group_count"] >= 1
    assert summary["marker_group_assignment_accuracy"] is not None
    assert 0.0 <= summary["marker_group_assignment_accuracy"] <= 1.0
    assert result["marker_metrics"]["truth_marker_count"] == summary["truth_marker_count"]
    assert result["marker_group_metrics"]["marker_group_count"] == summary["marker_group_count"]
    assert result["marker_group_metrics"]["grouping_method"] in {"x_rank", "y_kmeans"}
    assert result["prototypes"]
    assert summary["binding_accuracy"] >= 0.75
    assert any(row["best_target_type"] == "marker_instance" for row in result["prototypes"])
    assert summary["prototype_bound_path_count"] >= summary["curve_count"]
    assert summary["valid_prototype_bound_path_count"] >= summary["curve_count"]
    assert summary["valid_prototype_bound_data_count"] >= summary["curve_count"]
    assert summary["mean_prototype_bound_data_y_rmse"] is not None
    assert summary["mean_prototype_bound_data_y_rmse"] < 0.05
    assert summary["mean_prototype_bound_data_x_coverage_ratio"] is not None
    assert summary["mean_prototype_bound_data_x_coverage_ratio"] > 0.75
    assert result["pipeline"]["prototype_bound_paths"]

    gray_manifest = generate_benchmark(
        str(tmp_path / "synthetic"),
        "same_gray_linestyle_binding",
        SyntheticConfig(seed=23, n_curves=6, same_gray_linestyle_curves=True, legend_inside=True),
    )
    gray_result = run_prototype_binding_benchmark(str(Path(gray_manifest["image_path"]).parent))
    gray_summary = gray_result["summary"]
    assert gray_summary["legend_item_count"] == gray_summary["curve_count"]
    assert gray_summary["curve_visual_prototype_count"] == gray_summary["curve_count"]
    assert gray_summary["marker_curve_instance_count"] == 0
    assert gray_summary["line_style_curve_instance_count"] == gray_summary["curve_count"]
    assert gray_summary["prototype_bound_path_count"] == gray_summary["curve_count"]
    assert gray_summary["valid_prototype_bound_path_count"] == gray_summary["curve_count"]
    assert gray_summary["mean_prototype_bound_data_y_rmse"] is not None
    assert gray_summary["mean_prototype_bound_data_x_coverage_ratio"] is not None
    assert gray_summary["mean_prototype_bound_data_y_rmse"] < 0.45
    assert gray_summary["mean_prototype_bound_data_x_coverage_ratio"] > 0.75
    assert gray_summary["mean_prototype_bound_completed_point_ratio"] is not None
    assert gray_summary["mean_prototype_bound_completed_point_ratio"] > 0.0
    assert gray_result["pipeline"]["prototype_bound_paths"]
    assert {
        instance["grouping_method"]
        for instance in gray_result["pipeline"]["line_style_curve_instances"]
    } == {"component_x_rank"}
    assert gray_summary["binding_accuracy"] >= 0.75
    assert all(row["best_target_type"] == "line_style_instance" for row in gray_result["prototypes"] if row["expected_curve_id"] is not None)
