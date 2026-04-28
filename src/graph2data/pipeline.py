"""Structured extraction pipeline entry point."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2

from .axes import AxisDetector, AxisDetectorConfig
from .colors import CurveColorExtractor
from .image_io import ensure_dir, load_bgr, write_json
from .layout import assign_text_to_regions, build_nine_grid
from .legend import LegendDetector
from .lines import LinePathExtractor, PathTracingConfig
from .mapping import map_curve_paths_to_data, write_data_series_csv
from .masks import CurveMaskExtractor
from .models import DataRange, PipelineResult
from .ocr import OCRDetector


@dataclass
class PipelineConfig:
    run_ocr: bool = False
    run_colors: bool = False
    run_masks: bool = False
    run_paths: bool = False
    run_mapping: bool = False
    data_range: Optional[DataRange] = None
    artifact_dir: Optional[str] = None
    write_debug_artifacts: bool = False
    use_adaptive_axis_threshold: bool = False
    use_morph_axis_close: bool = False
    line_filter_marker_like_components: bool = False


class GraphExtractionPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def run(self, image_path: str) -> PipelineResult:
        image = load_bgr(image_path)
        h, w = image.shape[:2]
        run_paths = self.config.run_paths or self.config.run_mapping
        run_masks = self.config.run_masks or run_paths
        run_colors = self.config.run_colors or run_masks
        axis_config = AxisDetectorConfig(
            use_adaptive=self.config.use_adaptive_axis_threshold,
            use_morph=self.config.use_morph_axis_close,
        )
        axis = AxisDetector(axis_config).detect(image)
        warnings = []
        ocr_results = []
        layout = None
        curves = []
        curve_masks = []
        curve_paths = []
        data_series = []
        legends = []
        artifacts = {}

        if self.config.run_ocr:
            try:
                ocr_results = OCRDetector().detect(image)
            except Exception as exc:
                warnings.append(f"OCR failed: {exc}")

        if axis.success and axis.plot_area is not None:
            layout = build_nine_grid(axis.image_size, axis.plot_area)
            legend_detector = LegendDetector()
            if ocr_results:
                layout = assign_text_to_regions(layout, ocr_results)
                legends = legend_detector.detect(ocr_results, axis.plot_area)
            image_legends = legend_detector.detect_from_image(image, axis.plot_area)
            legends = _merge_legend_detections(legends + image_legends)

            if run_colors:
                try:
                    curves = CurveColorExtractor().extract(
                        image,
                        axis.plot_area.bbox,
                        exclude_regions=[legend.bbox for legend in legends],
                    )
                except Exception as exc:
                    warnings.append(f"Color extraction failed: {exc}")

            if curves and run_masks:
                try:
                    curve_masks, curve_paths, mask_artifacts = self._extract_masks_and_paths(
                        image=image,
                        curves=curves,
                        plot_bbox=axis.plot_area.bbox,
                        exclude_regions=[legend.bbox for legend in legends],
                    )
                    if mask_artifacts:
                        artifacts["masks"] = mask_artifacts
                except Exception as exc:
                    warnings.append(f"Mask/path extraction failed: {exc}")

            if self.config.run_mapping:
                if curve_paths and self.config.data_range is not None:
                    try:
                        data_series = map_curve_paths_to_data(curve_paths, axis.plot_area, self.config.data_range)
                        if self.config.artifact_dir:
                            data_dir = ensure_dir(os.path.join(self.config.artifact_dir, "data"))
                            data_csv_path = os.path.join(data_dir, "curves.csv")
                            write_data_series_csv(data_csv_path, data_series)
                            artifacts["data_csv"] = data_csv_path
                    except Exception as exc:
                        warnings.append(f"Data mapping failed: {exc}")
                elif self.config.data_range is None:
                    warnings.append("Skipping data mapping because data_range is not set")
                else:
                    warnings.append("Skipping data mapping because curve paths are not available")

            if self.config.write_debug_artifacts:
                if self.config.artifact_dir:
                    try:
                        artifacts["debug"] = self._write_debug_artifacts(
                            image=image,
                            axis=axis,
                            legends=legends,
                            curves=curves,
                            curve_paths=curve_paths,
                        )
                    except Exception as exc:
                        warnings.append(f"Debug artifact generation failed: {exc}")
                else:
                    warnings.append("Skipping debug artifacts because artifact_dir is not set")
        else:
            warnings.append("Skipping layout and curve prototype extraction because axis detection failed")

        if self.config.artifact_dir:
            try:
                quality_dir = ensure_dir(os.path.join(self.config.artifact_dir, "quality"))
                quality_report_path = os.path.join(quality_dir, "report.json")
                write_json(
                    quality_report_path,
                    self._build_quality_report(
                        image_path=image_path,
                        axis=axis,
                        legends=legends,
                        curves=curves,
                        curve_masks=curve_masks,
                        curve_paths=curve_paths,
                        data_series=data_series,
                        warnings=warnings,
                    ),
                )
                artifacts["quality_report"] = quality_report_path
            except Exception as exc:
                warnings.append(f"Quality report generation failed: {exc}")

        return PipelineResult(
            image_path=image_path,
            image_size=(w, h),
            axis=axis,
            ocr=ocr_results,
            layout=layout,
            legends=legends,
            curves=curves,
            curve_masks=curve_masks,
            curve_paths=curve_paths,
            data_series=data_series,
            artifacts=artifacts,
            warnings=warnings,
        )

    def _extract_masks_and_paths(self, image, curves, plot_bbox, exclude_regions):
        mask_extractor = CurveMaskExtractor()
        path_extractor = LinePathExtractor(
            PathTracingConfig(filter_marker_like_components=self.config.line_filter_marker_like_components)
        )
        curve_masks = []
        curve_paths = []
        mask_artifacts = {}

        mask_dir = None
        if self.config.artifact_dir:
            mask_dir = ensure_dir(os.path.join(self.config.artifact_dir, "masks"))

        for prototype in curves:
            mask, mask_info = mask_extractor.extract_mask(
                image,
                prototype,
                plot_bbox,
                exclude_regions=exclude_regions,
            )
            if mask_dir:
                mask_path = os.path.join(mask_dir, f"{prototype.curve_id}.png")
                cv2.imwrite(mask_path, mask)
                mask_info.mask_path = mask_path
                mask_artifacts[prototype.curve_id] = mask_path
            curve_masks.append(mask_info)

            if self.config.run_paths or self.config.run_mapping:
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                curve_paths.append(path_extractor.extract_from_mask_image(mask_bgr, curve_id=prototype.curve_id))

        return curve_masks, curve_paths, mask_artifacts

    def _write_debug_artifacts(self, image, axis, legends, curves, curve_paths):
        debug_dir = ensure_dir(os.path.join(self.config.artifact_dir, "debug"))
        artifacts = {}

        overview = image.copy()
        if axis.plot_area is not None:
            _draw_bbox(overview, axis.plot_area.bbox, (0, 180, 255), 2)
        if axis.x_axis is not None:
            _draw_line(overview, axis.x_axis.start, axis.x_axis.end, (255, 0, 0), 2)
        if axis.y_axis is not None:
            _draw_line(overview, axis.y_axis.start, axis.y_axis.end, (0, 0, 255), 2)
        for legend in legends:
            _draw_bbox(overview, legend.bbox, (0, 255, 255), 2)
        _draw_curve_swatches(overview, curves)
        overview_path = os.path.join(debug_dir, "overview.png")
        cv2.imwrite(overview_path, overview)
        artifacts["overview"] = overview_path

        if curve_paths:
            path_dir = ensure_dir(os.path.join(debug_dir, "paths"))
            all_paths = image.copy()
            path_artifacts = {}
            curve_by_id = {curve.curve_id: curve for curve in curves}
            for curve_path in curve_paths:
                curve = curve_by_id.get(curve_path.curve_id)
                color = _curve_bgr(curve)
                _draw_curve_path(all_paths, curve_path, color, 1)
                per_curve = image.copy()
                if axis.plot_area is not None:
                    _draw_bbox(per_curve, axis.plot_area.bbox, (0, 180, 255), 2)
                _draw_curve_path(per_curve, curve_path, color, 2)
                path = os.path.join(path_dir, f"{curve_path.curve_id}.png")
                cv2.imwrite(path, per_curve)
                path_artifacts[curve_path.curve_id] = path

            all_paths_path = os.path.join(debug_dir, "paths_overlay.png")
            cv2.imwrite(all_paths_path, all_paths)
            artifacts["paths_overlay"] = all_paths_path
            artifacts["paths"] = path_artifacts

        return artifacts

    def _build_quality_report(
        self,
        image_path,
        axis,
        legends,
        curves,
        curve_masks,
        curve_paths,
        data_series,
        warnings,
    ):
        path_points = sum(len(path.pixel_points_ordered) for path in curve_paths)
        completed_points = sum(path.completed_pixel_count for path in curve_paths)
        confidence_values = [
            confidence
            for path in curve_paths
            for confidence in (path.confidence_per_point or [])
        ]
        low_confidence_points = sum(1 for confidence in confidence_values if confidence < 0.5)
        mask_by_id = {mask.curve_id: mask for mask in curve_masks}
        path_by_id = {path.curve_id: path for path in curve_paths}
        series_by_id = {series.curve_id: series for series in data_series}

        return {
            "image_path": image_path,
            "axis": {
                "success": axis.success,
                "confidence": axis.confidence,
                "warnings": axis.warnings,
                "error": axis.error,
            },
            "counts": {
                "legend_count": len(legends),
                "curve_prototype_count": len(curves),
                "curve_mask_count": len(curve_masks),
                "curve_path_count": len(curve_paths),
                "data_series_count": len(data_series),
            },
            "path_summary": {
                "total_point_count": path_points,
                "total_completed_point_count": completed_points,
                "completed_point_ratio": _safe_ratio(completed_points, path_points),
                "mean_path_confidence": _mean_values([path.confidence for path in curve_paths]),
                "low_confidence_point_count": low_confidence_points,
                "low_confidence_point_ratio": _safe_ratio(low_confidence_points, len(confidence_values)),
            },
            "data_summary": {
                "total_point_count": sum(series.point_count for series in data_series),
                "total_completed_point_count": sum(series.completed_point_count for series in data_series),
                "completed_point_ratio": _safe_ratio(
                    sum(series.completed_point_count for series in data_series),
                    sum(series.point_count for series in data_series),
                ),
                "series_with_warnings": sum(1 for series in data_series if series.warnings),
            },
            "curves": [
                _curve_quality_row(curve.curve_id, mask_by_id.get(curve.curve_id), path_by_id.get(curve.curve_id), series_by_id.get(curve.curve_id))
                for curve in curves
            ],
            "warnings": list(warnings),
        }


def _draw_bbox(image, bbox, color, thickness):
    p0 = (int(round(bbox.x_min)), int(round(bbox.y_min)))
    p1 = (int(round(bbox.x_max)), int(round(bbox.y_max)))
    cv2.rectangle(image, p0, p1, color, thickness)


def _draw_line(image, start, end, color, thickness):
    p0 = (int(round(start.x)), int(round(start.y)))
    p1 = (int(round(end.x)), int(round(end.y)))
    cv2.line(image, p0, p1, color, thickness)


def _draw_curve_path(image, curve_path, color, thickness, max_segment_px=8.0):
    points = curve_path.pixel_points_ordered
    if len(points) < 2:
        return
    for a, b in zip(points, points[1:]):
        if ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5 > max_segment_px:
            continue
        p0 = (int(round(a.x)), int(round(a.y)))
        p1 = (int(round(b.x)), int(round(b.y)))
        cv2.line(image, p0, p1, color, thickness)


def _draw_curve_swatches(image, curves):
    x0, y0 = 12, 12
    for idx, curve in enumerate(curves[:12]):
        y = y0 + idx * 16
        color = _curve_bgr(curve)
        cv2.rectangle(image, (x0, y), (x0 + 28, y + 10), color, -1)
        cv2.rectangle(image, (x0, y), (x0 + 28, y + 10), (40, 40, 40), 1)


def _curve_bgr(curve):
    if curve is None:
        return (0, 255, 0)
    r, g, b = curve.rgb
    return (int(b), int(g), int(r))


def _merge_legend_detections(legends):
    merged = []
    for legend in sorted(legends, key=lambda item: item.confidence, reverse=True):
        if any(_bbox_iou(legend.bbox, existing.bbox) > 0.4 for existing in merged):
            continue
        merged.append(legend)
    return merged


def _bbox_iou(a, b):
    x0 = max(a.x_min, b.x_min)
    y0 = max(a.y_min, b.y_min)
    x1 = min(a.x_max, b.x_max)
    y1 = min(a.y_max, b.y_max)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    return inter / max(union, 1.0)


def _safe_ratio(numerator: float, denominator: float):
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _mean_values(values: Sequence[float]):
    values = [float(value) for value in values]
    if not values:
        return None
    return sum(values) / len(values)


def _curve_quality_row(curve_id, mask, path, series):
    point_count = len(path.pixel_points_ordered) if path else 0
    completed_count = path.completed_pixel_count if path else 0
    confidence_values = (path.confidence_per_point or []) if path else []
    return {
        "curve_id": curve_id,
        "mask_pixel_count": mask.pixel_count if mask else 0,
        "path_point_count": point_count,
        "path_length_px": path.path_length_px if path else 0.0,
        "path_confidence": path.confidence if path else None,
        "completed_point_count": completed_count,
        "completed_point_ratio": _safe_ratio(completed_count, point_count),
        "low_confidence_point_ratio": _safe_ratio(
            sum(1 for confidence in confidence_values if confidence < 0.5),
            len(confidence_values),
        ),
        "data_point_count": series.point_count if series else 0,
        "data_completed_point_count": series.completed_point_count if series else 0,
        "warnings": (mask.warnings if mask else []) + (path.warnings if path else []) + (series.warnings if series else []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the structured Graph2Data extraction pipeline.")
    parser.add_argument("--img", required=True, help="Input image path")
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    parser.add_argument("--ocr", action="store_true", help="Run OCR")
    parser.add_argument("--colors", action="store_true", help="Run curve color prototype extraction")
    parser.add_argument("--masks", action="store_true", help="Generate curve masks from color prototypes")
    parser.add_argument("--paths", action="store_true", help="Generate ordered curve paths from extracted masks")
    parser.add_argument("--map_data", action="store_true", help="Map extracted curve paths into data coordinates")
    parser.add_argument("--x_min", type=float, default=None, help="Data-space x minimum for --map_data")
    parser.add_argument("--x_max", type=float, default=None, help="Data-space x maximum for --map_data")
    parser.add_argument("--y_min", type=float, default=None, help="Data-space y minimum for --map_data")
    parser.add_argument("--y_max", type=float, default=None, help="Data-space y maximum for --map_data")
    parser.add_argument("--x_scale", choices=["linear", "log10", "inverse"], default="linear")
    parser.add_argument("--y_scale", choices=["linear", "log10", "inverse"], default="linear")
    parser.add_argument("--artifact_dir", default=None, help="Optional directory for mask/debug artifacts")
    parser.add_argument("--debug_artifacts", action="store_true", help="Write visual debug overlays to artifact_dir")
    parser.add_argument("--axis_adaptive", action="store_true", help="Use adaptive thresholding for axis detection")
    parser.add_argument("--axis_morph", action="store_true", help="Use morphology close for axis detection")
    parser.add_argument(
        "--line_filter_marker_like",
        action="store_true",
        help="Experimental: skip compact marker-like components during path tracing when longer line components exist",
    )
    args = parser.parse_args()
    run_colors = args.colors or args.masks or args.paths or args.map_data
    run_masks = args.masks or args.paths or args.map_data
    run_paths = args.paths or args.map_data
    data_range = None
    if args.map_data:
        missing = [
            name
            for name, value in (
                ("--x_min", args.x_min),
                ("--x_max", args.x_max),
                ("--y_min", args.y_min),
                ("--y_max", args.y_max),
            )
            if value is None
        ]
        if missing:
            parser.error("--map_data requires " + ", ".join(missing))
        data_range = DataRange(args.x_min, args.x_max, args.y_min, args.y_max, args.x_scale, args.y_scale)

    result = GraphExtractionPipeline(
        PipelineConfig(
            run_ocr=args.ocr,
            run_colors=run_colors,
            run_masks=run_masks,
            run_paths=run_paths,
            run_mapping=args.map_data,
            data_range=data_range,
            artifact_dir=args.artifact_dir,
            write_debug_artifacts=args.debug_artifacts,
            use_adaptive_axis_threshold=args.axis_adaptive,
            use_morph_axis_close=args.axis_morph,
            line_filter_marker_like_components=args.line_filter_marker_like,
        )
    ).run(args.img)

    if args.out:
        write_json(args.out, result)
    else:
        import json

        from .models import to_serializable

        print(json.dumps(to_serializable(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
