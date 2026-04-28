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
from .instances import (
    filter_border_line_components,
    group_line_style_curve_instances,
    group_marker_curve_instances,
    should_group_line_style_curve_instances,
    should_group_marker_curve_instances,
)
from .layout import assign_text_to_regions, build_nine_grid
from .legend import LegendDetector
from .lines import LinePathExtractor, PathTracingConfig
from .mapping import map_curve_paths_to_data, write_data_series_csv
from .masks import CurveMaskExtractor
from .models import BoundingBox, CurvePath, DataRange, Point, PrototypeBinding, PipelineResult
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
        line_components = []
        marker_candidates = []
        marker_curve_instances = []
        line_style_curve_instances = []
        prototype_bindings = []
        prototype_bound_paths = []
        data_series = []
        prototype_bound_data_series = []
        legends = []
        legend_items = []
        curve_visual_prototypes = []
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
            if legends:
                try:
                    legend_items = legend_detector.extract_items(image, legends)
                    if ocr_results:
                        legend_detector.assign_item_labels_from_ocr(legend_items, ocr_results)
                    curve_visual_prototypes = legend_detector.visual_prototypes_from_items(legend_items)
                except Exception as exc:
                    warnings.append(f"Legend item extraction failed: {exc}")

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
                    curve_masks, curve_paths, line_components, marker_candidates, mask_artifacts = self._extract_masks_and_paths(
                        image=image,
                        curves=curves,
                        plot_bbox=axis.plot_area.bbox,
                        exclude_regions=[legend.bbox for legend in legends],
                    )
                    if mask_artifacts:
                        artifacts["masks"] = mask_artifacts
                except Exception as exc:
                    warnings.append(f"Mask/path extraction failed: {exc}")
            if axis.plot_area is not None and line_components:
                line_components = filter_border_line_components(line_components, axis.plot_area.bbox)

            if should_group_marker_curve_instances(marker_candidates, line_components):
                marker_curve_instances = group_marker_curve_instances(
                    marker_candidates,
                    group_count=len(curve_visual_prototypes) if curve_visual_prototypes else None,
                )
            if not marker_curve_instances and should_group_line_style_curve_instances(line_components, len(curve_visual_prototypes)):
                line_style_curve_instances = group_line_style_curve_instances(
                    line_components,
                    group_count=len(curve_visual_prototypes),
                )

            if curve_visual_prototypes and curves:
                prototype_bindings = _score_prototype_bindings(
                    curve_visual_prototypes,
                    curves,
                    line_components,
                    marker_candidates,
                    marker_curve_instances=marker_curve_instances,
                    line_style_curve_instances=line_style_curve_instances,
                )
                prototype_bound_paths = _prototype_bound_marker_paths(
                    curve_visual_prototypes,
                    prototype_bindings,
                    marker_curve_instances,
                    curve_paths,
                )
                prototype_bound_paths.extend(
                    _prototype_bound_curve_paths(
                        curve_visual_prototypes,
                        prototype_bindings,
                        curve_paths,
                        line_components,
                    )
                )
                prototype_bound_paths.extend(
                    _prototype_bound_line_style_paths(
                        curve_visual_prototypes,
                        prototype_bindings,
                        line_style_curve_instances,
                    )
                )

            if self.config.run_mapping:
                if curve_paths and self.config.data_range is not None:
                    try:
                        data_series = map_curve_paths_to_data(curve_paths, axis.plot_area, self.config.data_range)
                        if prototype_bound_paths:
                            prototype_bound_data_series = map_curve_paths_to_data(
                                prototype_bound_paths,
                                axis.plot_area,
                                self.config.data_range,
                            )
                        if self.config.artifact_dir:
                            data_dir = ensure_dir(os.path.join(self.config.artifact_dir, "data"))
                            data_csv_path = os.path.join(data_dir, "curves.csv")
                            write_data_series_csv(data_csv_path, data_series)
                            artifacts["data_csv"] = data_csv_path
                            if prototype_bound_data_series:
                                prototype_bound_csv_path = os.path.join(data_dir, "prototype_bound_curves.csv")
                                write_data_series_csv(prototype_bound_csv_path, prototype_bound_data_series)
                                artifacts["prototype_bound_data_csv"] = prototype_bound_csv_path
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
                            legend_items=legend_items,
                            curves=curves,
                            curve_paths=curve_paths,
                            line_components=line_components,
                            marker_candidates=marker_candidates,
                            marker_curve_instances=marker_curve_instances,
                            line_style_curve_instances=line_style_curve_instances,
                            prototype_bound_paths=prototype_bound_paths,
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
                        legend_items=legend_items,
                        curve_visual_prototypes=curve_visual_prototypes,
                        curves=curves,
                        curve_masks=curve_masks,
                        curve_paths=curve_paths,
                        line_components=line_components,
                        marker_candidates=marker_candidates,
                        marker_curve_instances=marker_curve_instances,
                        line_style_curve_instances=line_style_curve_instances,
                        prototype_bindings=prototype_bindings,
                        prototype_bound_paths=prototype_bound_paths,
                        data_series=data_series,
                        prototype_bound_data_series=prototype_bound_data_series,
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
            legend_items=legend_items,
            curve_visual_prototypes=curve_visual_prototypes,
            curves=curves,
            curve_masks=curve_masks,
            curve_paths=curve_paths,
            line_components=line_components,
            marker_candidates=marker_candidates,
            marker_curve_instances=marker_curve_instances,
            line_style_curve_instances=line_style_curve_instances,
            prototype_bindings=prototype_bindings,
            prototype_bound_paths=prototype_bound_paths,
            data_series=data_series,
            prototype_bound_data_series=prototype_bound_data_series,
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
        line_components = []
        marker_candidates = []
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

            mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            line_components.extend(path_extractor.classify_components_from_mask_image(mask_bgr, curve_id=prototype.curve_id))
            marker_candidates.extend(path_extractor.detect_marker_candidates_from_mask_image(mask_bgr, curve_id=prototype.curve_id))
            if self.config.run_paths or self.config.run_mapping:
                curve_paths.append(path_extractor.extract_from_mask_image(mask_bgr, curve_id=prototype.curve_id))

        return curve_masks, curve_paths, line_components, marker_candidates, mask_artifacts

    def _write_debug_artifacts(
        self,
        image,
        axis,
        legends,
        legend_items,
        curves,
        curve_paths,
        line_components,
        marker_candidates,
        marker_curve_instances,
        line_style_curve_instances,
        prototype_bound_paths,
    ):
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

        if legend_items:
            legend_overlay = image.copy()
            for legend in legends:
                _draw_bbox(legend_overlay, legend.bbox, (0, 255, 255), 2)
            for item in legend_items:
                _draw_bbox(legend_overlay, item.bbox, (0, 180, 0), 2)
                if item.sample_bbox is not None:
                    _draw_bbox(legend_overlay, item.sample_bbox, (255, 128, 0), 1)
                if item.text_bbox is not None:
                    _draw_bbox(legend_overlay, item.text_bbox, (255, 0, 255), 1)
                _draw_legend_item_label(legend_overlay, item)
            legend_items_path = os.path.join(debug_dir, "legend_items.png")
            cv2.imwrite(legend_items_path, legend_overlay)
            artifacts["legend_items"] = legend_items_path

        if line_components:
            component_overlay = image.copy()
            if axis.plot_area is not None:
                _draw_bbox(component_overlay, axis.plot_area.bbox, (0, 180, 255), 2)
            for component in line_components:
                color = _component_class_color(component.class_label)
                _draw_bbox(component_overlay, component.bbox, color, 1)
            component_path = os.path.join(debug_dir, "component_classification.png")
            cv2.imwrite(component_path, component_overlay)
            artifacts["component_classification"] = component_path

        if marker_candidates:
            marker_overlay = image.copy()
            if axis.plot_area is not None:
                _draw_bbox(marker_overlay, axis.plot_area.bbox, (0, 180, 255), 2)
            for marker in marker_candidates:
                _draw_bbox(marker_overlay, marker.bbox, (255, 128, 0), 1)
                _draw_marker_center(marker_overlay, marker)
            marker_path = os.path.join(debug_dir, "marker_candidates.png")
            cv2.imwrite(marker_path, marker_overlay)
            artifacts["marker_candidates"] = marker_path

        if marker_candidates or marker_curve_instances:
            instance_overlay = image.copy()
            if axis.plot_area is not None:
                _draw_bbox(instance_overlay, axis.plot_area.bbox, (0, 180, 255), 2)
            for idx, instance in enumerate(marker_curve_instances):
                color = _instance_color(idx)
                _draw_marker_instance(instance_overlay, instance, color)
            instance_path = os.path.join(debug_dir, "marker_curve_instances.png")
            cv2.imwrite(instance_path, instance_overlay)
            artifacts["marker_curve_instances"] = instance_path

        if line_style_curve_instances:
            line_instance_overlay = image.copy()
            if axis.plot_area is not None:
                _draw_bbox(line_instance_overlay, axis.plot_area.bbox, (0, 180, 255), 2)
            for idx, instance in enumerate(line_style_curve_instances):
                _draw_line_style_instance(line_instance_overlay, instance, _instance_color(idx))
            line_instance_path = os.path.join(debug_dir, "line_style_curve_instances.png")
            cv2.imwrite(line_instance_path, line_instance_overlay)
            artifacts["line_style_curve_instances"] = line_instance_path

        if prototype_bound_paths:
            bound_path_overlay = image.copy()
            if axis.plot_area is not None:
                _draw_bbox(bound_path_overlay, axis.plot_area.bbox, (0, 180, 255), 2)
            for idx, path in enumerate(prototype_bound_paths):
                _draw_confidence_curve_path(bound_path_overlay, path, _instance_color(idx), 2, max_segment_px=80.0)
            bound_path_path = os.path.join(debug_dir, "prototype_bound_paths.png")
            cv2.imwrite(bound_path_path, bound_path_overlay)
            artifacts["prototype_bound_paths"] = bound_path_path

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
        legend_items,
        curve_visual_prototypes,
        curves,
        curve_masks,
        curve_paths,
        line_components,
        marker_candidates,
        marker_curve_instances,
        line_style_curve_instances,
        prototype_bindings,
        prototype_bound_paths,
        data_series,
        prototype_bound_data_series,
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
                "legend_item_count": len(legend_items),
                "curve_visual_prototype_count": len(curve_visual_prototypes),
                "curve_prototype_count": len(curves),
                "curve_mask_count": len(curve_masks),
                "curve_path_count": len(curve_paths),
                "line_component_count": len(line_components),
                "marker_candidate_count": len(marker_candidates),
                "marker_curve_instance_count": len(marker_curve_instances),
                "line_style_curve_instance_count": len(line_style_curve_instances),
                "prototype_binding_count": len(prototype_bindings),
                "prototype_bound_path_count": len(prototype_bound_paths),
                "data_series_count": len(data_series),
                "prototype_bound_data_series_count": len(prototype_bound_data_series),
            },
            "legends": [
                {
                    "bbox": legend.bbox,
                    "confidence": legend.confidence,
                    "source": legend.source,
                    "warnings": legend.warnings,
                }
                for legend in legends
            ],
            "legend_items": legend_items,
            "line_components": line_components,
            "component_summary": _component_summary(line_components),
            "marker_candidates": marker_candidates,
            "marker_summary": _marker_summary(marker_candidates),
            "marker_curve_instances": marker_curve_instances,
            "marker_instance_summary": _marker_instance_summary(marker_curve_instances),
            "line_style_curve_instances": line_style_curve_instances,
            "line_style_instance_summary": _line_style_instance_summary(line_style_curve_instances),
            "prototype_bindings": prototype_bindings,
            "binding_summary": _binding_summary(prototype_bindings),
            "prototype_bound_paths": prototype_bound_paths,
            "prototype_bound_path_summary": _prototype_bound_path_summary(prototype_bound_paths),
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
            "prototype_bound_data_series": prototype_bound_data_series,
            "prototype_bound_data_summary": {
                "total_point_count": sum(series.point_count for series in prototype_bound_data_series),
                "total_completed_point_count": sum(series.completed_point_count for series in prototype_bound_data_series),
                "completed_point_ratio": _safe_ratio(
                    sum(series.completed_point_count for series in prototype_bound_data_series),
                    sum(series.point_count for series in prototype_bound_data_series),
                ),
                "series_with_warnings": sum(1 for series in prototype_bound_data_series if series.warnings),
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


def _draw_confidence_curve_path(image, curve_path, color, thickness, max_segment_px=8.0):
    points = curve_path.pixel_points_ordered
    if len(points) < 2:
        return
    confidences = curve_path.confidence_per_point or [curve_path.confidence for _ in points]
    completed_color = (0, 200, 255)
    for idx, (a, b) in enumerate(zip(points, points[1:])):
        if ((a.x - b.x) ** 2 + (a.y - b.y) ** 2) ** 0.5 > max_segment_px:
            continue
        conf_a = confidences[idx] if idx < len(confidences) else curve_path.confidence
        conf_b = confidences[idx + 1] if idx + 1 < len(confidences) else curve_path.confidence
        segment_color = completed_color if min(conf_a, conf_b) < 0.5 else color
        p0 = (int(round(a.x)), int(round(a.y)))
        p1 = (int(round(b.x)), int(round(b.y)))
        cv2.line(image, p0, p1, segment_color, thickness)


def _draw_curve_swatches(image, curves):
    x0, y0 = 12, 12
    for idx, curve in enumerate(curves[:12]):
        y = y0 + idx * 16
        color = _curve_bgr(curve)
        cv2.rectangle(image, (x0, y), (x0 + 28, y + 10), color, -1)
        cv2.rectangle(image, (x0, y), (x0 + 28, y + 10), (40, 40, 40), 1)


def _draw_legend_item_label(image, item):
    x = int(round(item.bbox.x_min))
    y = max(12, int(round(item.bbox.y_min)) - 4)
    text = item.item_id
    if item.line_style != "unknown":
        text += f" {item.line_style}"
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)


def _component_class_color(class_label):
    if class_label == "line_like":
        return (0, 180, 0)
    if class_label == "marker_like":
        return (255, 128, 0)
    return (0, 0, 255)


def _draw_marker_center(image, marker):
    center = (int(round(marker.center.x)), int(round(marker.center.y)))
    cv2.drawMarker(image, center, (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=8, thickness=1)


def _draw_marker_instance(image, instance, color):
    points = sorted(instance.points, key=lambda point: (point.x, point.y))
    for point in points:
        center = (int(round(point.x)), int(round(point.y)))
        cv2.circle(image, center, 4, color, 1)
    for a, b in zip(points, points[1:]):
        p0 = (int(round(a.x)), int(round(a.y)))
        p1 = (int(round(b.x)), int(round(b.y)))
        cv2.line(image, p0, p1, color, 1)
    if points:
        cv2.putText(
            image,
            instance.instance_id,
            (int(round(points[0].x)), int(round(points[0].y)) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )


def _draw_line_style_instance(image, instance, color):
    if instance.x_min is None or instance.x_max is None or instance.y_min is None or instance.y_max is None:
        return
    bbox = BoundingBox(instance.x_min, instance.y_min, instance.x_max, instance.y_max)
    _draw_bbox(image, bbox, color, 2)
    label = f"{instance.instance_id} {instance.estimated_line_style}"
    cv2.putText(
        image,
        label,
        (int(round(instance.x_min)), max(12, int(round(instance.y_min)) - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        color,
        1,
        cv2.LINE_AA,
    )


def _instance_color(idx):
    colors = [
        (0, 0, 255),
        (0, 180, 0),
        (255, 0, 0),
        (0, 180, 255),
        (180, 0, 255),
        (255, 128, 0),
        (128, 0, 128),
        (0, 128, 128),
    ]
    return colors[idx % len(colors)]


def _component_summary(line_components):
    by_class = {}
    for component in line_components:
        by_class[component.class_label] = by_class.get(component.class_label, 0) + 1
    return {
        "by_class": by_class,
        "line_like_count": by_class.get("line_like", 0),
        "marker_like_count": by_class.get("marker_like", 0),
        "noise_count": by_class.get("noise", 0),
    }


def _marker_summary(marker_candidates):
    by_shape = {}
    for marker in marker_candidates:
        by_shape[marker.shape] = by_shape.get(marker.shape, 0) + 1
    return {
        "by_shape": by_shape,
        "candidate_count": len(marker_candidates),
        "mean_confidence": _mean_values([marker.confidence for marker in marker_candidates]),
    }


def _marker_instance_summary(marker_curve_instances):
    return {
        "instance_count": len(marker_curve_instances),
        "total_marker_count": sum(instance.marker_count for instance in marker_curve_instances),
        "mean_marker_count": _mean_values([instance.marker_count for instance in marker_curve_instances]),
        "mean_confidence": _mean_values([instance.confidence for instance in marker_curve_instances]),
        "grouping_methods": sorted({instance.grouping_method for instance in marker_curve_instances}),
    }


def _line_style_instance_summary(line_style_curve_instances):
    by_style = {}
    for instance in line_style_curve_instances:
        by_style[instance.estimated_line_style] = by_style.get(instance.estimated_line_style, 0) + 1
    return {
        "instance_count": len(line_style_curve_instances),
        "by_estimated_line_style": by_style,
        "mean_component_count": _mean_values([instance.component_count for instance in line_style_curve_instances]),
        "mean_confidence": _mean_values([instance.confidence for instance in line_style_curve_instances]),
        "grouping_methods": sorted({instance.grouping_method for instance in line_style_curve_instances}),
    }


def _prototype_bound_path_summary(prototype_bound_paths):
    total_points = sum(len(path.pixel_points_ordered) for path in prototype_bound_paths)
    total_completed_points = sum(path.completed_pixel_count for path in prototype_bound_paths)
    return {
        "path_count": len(prototype_bound_paths),
        "total_point_count": total_points,
        "total_completed_point_count": total_completed_points,
        "completed_point_ratio": _safe_ratio(total_completed_points, total_points),
        "mean_point_count": _mean_values([len(path.pixel_points_ordered) for path in prototype_bound_paths]),
        "mean_confidence": _mean_values([path.confidence for path in prototype_bound_paths]),
    }


def _prototype_bound_curve_paths(visual_prototypes, prototype_bindings, curve_paths, line_components=None):
    if not visual_prototypes or not prototype_bindings or not curve_paths:
        return []
    curve_path_by_id = {path.curve_id: path for path in curve_paths}
    line_components = list(line_components or [])
    best_by_prototype = {}
    for binding in prototype_bindings:
        current = best_by_prototype.get(binding.prototype_id)
        if current is None or binding.score > current.score:
            best_by_prototype[binding.prototype_id] = binding

    paths = []
    used_curve_ids = set()
    for prototype in visual_prototypes:
        binding = best_by_prototype.get(prototype.prototype_id)
        if binding is None or binding.target_type != "curve":
            continue
        if binding.target_curve_id in used_curve_ids:
            continue
        source = curve_path_by_id.get(binding.target_curve_id)
        if source is None or not source.pixel_points_ordered:
            continue
        used_curve_ids.add(binding.target_curve_id)
        confidence = min(float(binding.confidence), float(source.confidence), float(prototype.confidence))
        points = list(source.pixel_points_ordered)
        completed_ranges = list(source.completed_ranges)
        confidence_per_point = list(source.confidence_per_point) or [confidence for _ in source.pixel_points_ordered]
        endpoints = list(source.endpoints)
        junctions = list(source.junctions)
        component_count = source.component_count
        observed_pixel_count = source.observed_pixel_count
        completed_pixel_count = source.completed_pixel_count
        path_length_px = source.path_length_px
        warnings = [
            f"prototype_id={prototype.prototype_id}",
            f"legend_item_id={prototype.legend_item_id}",
            f"binding_id={binding.binding_id}",
            f"target_curve={source.curve_id}",
            "prototype_bound_curve_path",
        ]
        rebuilt = _rebuild_curve_path_from_line_components(source, line_components, confidence)
        if rebuilt is not None:
            (
                points,
                completed_ranges,
                confidence_per_point,
                endpoints,
                component_count,
                observed_pixel_count,
                completed_pixel_count,
                path_length_px,
            ) = rebuilt
            warnings.append("curve_path_rebuilt_from_line_components")
        paths.append(
            CurvePath(
                curve_id=f"{prototype.prototype_id}_bound_path",
                pixel_points_ordered=points,
                label=prototype.label,
                completed_ranges=completed_ranges,
                confidence_per_point=confidence_per_point,
                endpoints=endpoints,
                junctions=junctions,
                component_count=component_count,
                observed_pixel_count=observed_pixel_count,
                completed_pixel_count=completed_pixel_count,
                path_length_px=path_length_px,
                confidence=confidence,
                warnings=warnings + list(binding.warnings) + list(source.warnings),
            )
        )
    return paths


def _rebuild_curve_path_from_line_components(source_path, line_components, confidence):
    if not line_components or not source_path.pixel_points_ordered:
        return None
    source_points = list(source_path.pixel_points_ordered)
    observed_count = max(int(source_path.observed_pixel_count or 0), len(source_points))
    if observed_count <= 0 or len(source_points) / observed_count >= 0.55:
        return None
    components = [
        component
        for component in line_components
        if component.curve_id == source_path.curve_id and component.class_label == "line_like" and component.path_points
    ]
    if not components:
        return None
    observed_points = []
    for component in components:
        observed_points.extend(component.path_points)
    observed_points = _compress_line_style_points(sorted(observed_points, key=lambda point: (point.x, point.y)))
    if _point_x_span(observed_points) <= _point_x_span(source_points) * 1.15:
        return None
    points, completed_ranges, _ = _connect_line_style_points(observed_points)
    if len(points) <= len(source_points):
        return None
    completed_pixel_count = sum(max(0, end - start + 1) for start, end in completed_ranges)
    return (
        points,
        completed_ranges,
        _line_style_point_confidences(len(points), completed_ranges, confidence),
        [points[0], points[-1]] if len(points) >= 2 else list(points),
        len(components),
        len(observed_points),
        completed_pixel_count,
        _point_path_length(points),
    )


def _prototype_bound_marker_paths(visual_prototypes, prototype_bindings, marker_curve_instances, curve_paths=None):
    if not visual_prototypes or not prototype_bindings or not marker_curve_instances:
        return []
    marker_by_id = {instance.instance_id: instance for instance in marker_curve_instances}
    source_path_by_id = {path.curve_id: path for path in (curve_paths or [])}
    marker_instance_count_by_source = {}
    for instance in marker_curve_instances:
        marker_instance_count_by_source[instance.source_curve_id] = marker_instance_count_by_source.get(instance.source_curve_id, 0) + 1
    best_by_prototype = {}
    for binding in prototype_bindings:
        current = best_by_prototype.get(binding.prototype_id)
        if current is None or binding.score > current.score:
            best_by_prototype[binding.prototype_id] = binding

    paths = []
    used_marker_instance_ids = set()
    for prototype in visual_prototypes:
        binding = best_by_prototype.get(prototype.prototype_id)
        if binding is None or binding.target_type != "marker_instance":
            continue
        if binding.target_curve_id in used_marker_instance_ids:
            continue
        instance = marker_by_id.get(binding.target_curve_id)
        if instance is None or not instance.points:
            continue
        used_marker_instance_ids.add(binding.target_curve_id)
        confidence = min(float(binding.confidence), float(instance.confidence), float(prototype.confidence))
        source_path = None
        if marker_instance_count_by_source.get(instance.source_curve_id) == 1:
            source_path = source_path_by_id.get(instance.source_curve_id)
        warnings = [
            f"prototype_id={prototype.prototype_id}",
            f"legend_item_id={prototype.legend_item_id}",
            f"binding_id={binding.binding_id}",
            f"target_marker_instance={instance.instance_id}",
            "prototype_bound_marker_path",
        ]
        if source_path is not None and source_path.pixel_points_ordered:
            points = list(source_path.pixel_points_ordered)
            confidence = min(confidence, float(source_path.confidence))
            warnings.append("marker_path_uses_source_curve_path")
            completed_ranges = list(source_path.completed_ranges)
            confidence_per_point = list(source_path.confidence_per_point) or [confidence for _ in points]
            endpoints = list(source_path.endpoints)
            junctions = list(source_path.junctions)
            component_count = source_path.component_count
            observed_pixel_count = source_path.observed_pixel_count
            completed_pixel_count = source_path.completed_pixel_count
            path_length_px = source_path.path_length_px
            source_warnings = list(source_path.warnings)
        else:
            points = sorted(instance.points, key=lambda point: (point.x, point.y))
            completed_ranges = []
            confidence_per_point = [confidence for _ in points]
            endpoints = [points[0], points[-1]] if len(points) >= 2 else list(points)
            junctions = []
            component_count = 1
            observed_pixel_count = len(points)
            completed_pixel_count = 0
            path_length_px = _point_path_length(points)
            source_warnings = []
        paths.append(
            CurvePath(
                curve_id=f"{prototype.prototype_id}_bound_path",
                pixel_points_ordered=points,
                label=prototype.label,
                completed_ranges=completed_ranges,
                confidence_per_point=confidence_per_point,
                endpoints=endpoints,
                junctions=junctions,
                component_count=component_count,
                observed_pixel_count=observed_pixel_count,
                completed_pixel_count=completed_pixel_count,
                path_length_px=path_length_px,
                confidence=confidence,
                warnings=warnings + list(binding.warnings) + list(instance.warnings) + source_warnings,
            )
        )
    return paths


def _prototype_bound_line_style_paths(visual_prototypes, prototype_bindings, line_style_curve_instances):
    if not visual_prototypes or not prototype_bindings or not line_style_curve_instances:
        return []
    line_instance_by_id = {instance.instance_id: instance for instance in line_style_curve_instances}
    best_by_prototype = {}
    for binding in prototype_bindings:
        current = best_by_prototype.get(binding.prototype_id)
        if current is None or binding.score > current.score:
            best_by_prototype[binding.prototype_id] = binding

    paths = []
    used_line_instance_ids = set()
    for prototype in visual_prototypes:
        binding = best_by_prototype.get(prototype.prototype_id)
        if binding is None or binding.target_type != "line_style_instance":
            continue
        if binding.target_curve_id in used_line_instance_ids:
            continue
        instance = line_instance_by_id.get(binding.target_curve_id)
        if instance is None or not instance.points:
            continue
        used_line_instance_ids.add(binding.target_curve_id)
        observed_points = _compress_line_style_points(sorted(instance.points, key=lambda point: (point.x, point.y)))
        points, completed_ranges, gap_stats = _connect_line_style_points(observed_points)
        confidence = min(float(binding.confidence), float(instance.confidence), float(prototype.confidence))
        warnings = [
            f"prototype_id={prototype.prototype_id}",
            f"legend_item_id={prototype.legend_item_id}",
            f"binding_id={binding.binding_id}",
            f"target_line_style_instance={instance.instance_id}",
            "prototype_bound_line_style_path",
        ]
        if gap_stats["connected_gap_count"]:
            warnings.append(f"line_style_gaps_connected={gap_stats['connected_gap_count']}")
        if gap_stats["rejected_gap_count"]:
            warnings.append(f"line_style_gaps_rejected={gap_stats['rejected_gap_count']}")
        paths.append(
            CurvePath(
                curve_id=f"{prototype.prototype_id}_bound_path",
                pixel_points_ordered=points,
                label=prototype.label,
                completed_ranges=completed_ranges,
                confidence_per_point=_line_style_point_confidences(len(points), completed_ranges, confidence),
                endpoints=[points[0], points[-1]] if len(points) >= 2 else list(points),
                component_count=max(1, instance.component_count),
                observed_pixel_count=len(observed_points),
                completed_pixel_count=sum(max(0, end - start + 1) for start, end in completed_ranges),
                path_length_px=_point_path_length(points),
                confidence=confidence,
                warnings=warnings + list(binding.warnings) + list(instance.warnings),
            )
        )
    return paths


def _connect_line_style_points(points, max_gap_px=95.0, step_px=4.0):
    if len(points) < 2:
        return list(points), [], {"connected_gap_count": 0, "rejected_gap_count": 0}
    connected = [points[0]]
    completed_ranges = []
    connected_gap_count = 0
    rejected_gap_count = 0
    for idx, (prev, current) in enumerate(zip(points, points[1:])):
        distance = ((float(current.x) - float(prev.x)) ** 2 + (float(current.y) - float(prev.y)) ** 2) ** 0.5
        should_connect = _should_connect_line_style_gap(points, idx, distance, max_gap_px)
        if should_connect:
            steps = max(1, int(distance // step_px))
            start_idx = len(connected)
            for step in range(1, steps):
                t = step / float(steps)
                connected.append(
                    Point(
                        float(prev.x) + (float(current.x) - float(prev.x)) * t,
                        float(prev.y) + (float(current.y) - float(prev.y)) * t,
                    )
                )
            end_idx = len(connected) - 1
            if end_idx >= start_idx:
                completed_ranges.append((start_idx, end_idx))
                connected_gap_count += 1
        elif distance > 1.0:
            rejected_gap_count += 1
        connected.append(current)
    return connected, completed_ranges, {
        "connected_gap_count": connected_gap_count,
        "rejected_gap_count": rejected_gap_count,
    }


def _should_connect_line_style_gap(points, idx, distance, max_gap_px):
    if distance <= 1.0 or distance > max_gap_px:
        return False
    prev = points[idx]
    current = points[idx + 1]
    dx = float(current.x) - float(prev.x)
    dy = float(current.y) - float(prev.y)
    if dx <= 0:
        return False
    if abs(dy) > max(32.0, abs(dx) * 0.85):
        return False

    gap_angle = _angle_degrees(dx, dy)
    tangent_angles = []
    if idx > 0:
        before = points[idx - 1]
        before_dx = float(prev.x) - float(before.x)
        before_dy = float(prev.y) - float(before.y)
        if before_dx > 0:
            tangent_angles.append(_angle_degrees(before_dx, before_dy))
    if idx + 2 < len(points):
        after = points[idx + 2]
        after_dx = float(after.x) - float(current.x)
        after_dy = float(after.y) - float(current.y)
        if after_dx > 0:
            tangent_angles.append(_angle_degrees(after_dx, after_dy))
    if tangent_angles:
        max_diff = max(_angle_difference(gap_angle, angle) for angle in tangent_angles)
        if max_diff > 70.0:
            return False
    return True


def _angle_degrees(dx, dy):
    import math

    return math.degrees(math.atan2(float(dy), float(dx)))


def _angle_difference(a, b):
    diff = abs(float(a) - float(b)) % 360.0
    return min(diff, 360.0 - diff)


def _compress_line_style_points(points):
    if not points:
        return []
    grouped = {}
    for point in points:
        key = int(round(float(point.x)))
        grouped.setdefault(key, []).append(float(point.y))
    compressed = [
        Point(float(x_key), sum(y_values) / len(y_values))
        for x_key, y_values in sorted(grouped.items())
    ]
    return compressed


def _line_style_point_confidences(point_count, completed_ranges, observed_confidence, completed_confidence=0.35):
    confidences = [float(observed_confidence) for _ in range(point_count)]
    for start, end in completed_ranges:
        for idx in range(int(start), int(end) + 1):
            if 0 <= idx < len(confidences):
                confidences[idx] = float(min(confidences[idx], completed_confidence))
    return confidences


def _score_prototype_bindings(
    visual_prototypes,
    curves,
    line_components,
    marker_candidates,
    marker_curve_instances=None,
    line_style_curve_instances=None,
):
    bindings = []
    marker_curve_instances = marker_curve_instances or []
    line_style_curve_instances = line_style_curve_instances or []
    line_by_curve = {}
    marker_by_curve = {}
    for component in line_components:
        line_by_curve.setdefault(component.curve_id, []).append(component)
    for marker in marker_candidates:
        marker_by_curve.setdefault(marker.curve_id, []).append(marker)

    for proto_idx, proto in enumerate(visual_prototypes):
        for curve in curves:
            color_similarity = _rgb_similarity(proto.rgb, curve.rgb)
            line_similarity = _line_similarity(proto.line_style, line_by_curve.get(curve.curve_id, []))
            marker_similarity = _marker_similarity(proto.marker_style, marker_by_curve.get(curve.curve_id, []))
            score_parts = []
            weights = []
            warnings = []
            if color_similarity is not None:
                score_parts.append(0.55 * color_similarity)
                weights.append(0.55)
            else:
                warnings.append("missing_color_similarity")
            if line_similarity is not None:
                score_parts.append(0.20 * line_similarity)
                weights.append(0.20)
            else:
                warnings.append("missing_line_similarity")
            if marker_similarity is not None:
                score_parts.append(0.25 * marker_similarity)
                weights.append(0.25)
            else:
                warnings.append("missing_marker_similarity")

            score = sum(score_parts) / max(sum(weights), 1e-9)
            if proto.rgb is not None and _is_achromatic_rgb(proto.rgb) and proto.line_style != "unknown" and proto.marker_style == "unknown":
                score *= 0.75
                warnings.append("achromatic_line_style_prototype_penalizes_direct_curve")
            if getattr(curve, "confidence", 1.0) < 0.8:
                score *= max(0.25, float(curve.confidence))
                warnings.append("low_curve_prototype_confidence")
            if proto.marker_style == "unknown":
                warnings.append("prototype_marker_unknown")
            bindings.append(
                PrototypeBinding(
                    binding_id=f"binding_{len(bindings):04d}",
                    prototype_id=proto.prototype_id,
                    legend_item_id=proto.legend_item_id,
                    target_curve_id=curve.curve_id,
                    target_type="curve",
                    score=float(score),
                    color_similarity=color_similarity,
                    line_similarity=line_similarity,
                    marker_similarity=marker_similarity,
                    confidence=float(min(1.0, score * proto.confidence)),
                    warnings=warnings,
                )
            )
        ordered_marker_instances = sorted(marker_curve_instances, key=lambda item: float(item.center_y), reverse=True)
        for instance_idx, instance in enumerate(ordered_marker_instances):
            marker_similarity = _marker_instance_similarity(proto, instance)
            ordinal_similarity = _ordinal_similarity(proto_idx, instance_idx, len(visual_prototypes), len(marker_curve_instances))
            score = 0.65 * marker_similarity + 0.35 * ordinal_similarity
            warnings = []
            if proto.marker_style == "unknown":
                warnings.append("prototype_marker_unknown")
            bindings.append(
                PrototypeBinding(
                    binding_id=f"binding_{len(bindings):04d}",
                    prototype_id=proto.prototype_id,
                    legend_item_id=proto.legend_item_id,
                    target_curve_id=instance.instance_id,
                    target_type="marker_instance",
                    score=float(score),
                    color_similarity=None,
                    line_similarity=None,
                    marker_similarity=float(marker_similarity),
                    confidence=float(min(1.0, score * proto.confidence)),
                    source="marker_instance_score",
                    warnings=warnings,
                )
            )
        ordered_line_instances = sorted(line_style_curve_instances, key=lambda item: float(item.center_y), reverse=True)
        for instance_idx, instance in enumerate(ordered_line_instances):
            line_similarity = _line_style_instance_similarity(proto.line_style, instance.estimated_line_style)
            ordinal_similarity = _ordinal_similarity(proto_idx, instance_idx, len(visual_prototypes), len(line_style_curve_instances))
            score = 0.80 * ordinal_similarity + 0.10 * line_similarity + 0.10 * instance.confidence
            warnings = []
            if proto.line_style == "unknown":
                warnings.append("prototype_line_style_unknown")
            if instance.estimated_line_style == "unknown":
                warnings.append("line_instance_style_unknown")
            if proto.marker_style != "unknown":
                score *= 0.80
                warnings.append("marker_prototype_penalizes_line_style_instance")
            if proto.rgb is not None and not _is_achromatic_rgb(proto.rgb):
                score *= 0.70
                warnings.append("color_prototype_penalizes_line_style_instance")
            bindings.append(
                PrototypeBinding(
                    binding_id=f"binding_{len(bindings):04d}",
                    prototype_id=proto.prototype_id,
                    legend_item_id=proto.legend_item_id,
                    target_curve_id=instance.instance_id,
                    target_type="line_style_instance",
                    score=float(score),
                    color_similarity=None,
                    line_similarity=float(line_similarity),
                    marker_similarity=None,
                    confidence=float(min(1.0, score * proto.confidence)),
                    source="line_style_instance_score",
                    warnings=warnings,
                )
            )
    bindings.sort(key=lambda item: (item.prototype_id, -item.score, item.target_curve_id))
    return bindings


def _rgb_similarity(a, b):
    if a is None or b is None:
        return None
    dist = sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) ** 0.5
    return float(max(0.0, 1.0 - dist / 441.67295593))


def _line_similarity(line_style, components):
    if not components:
        return 0.0
    line_like_count = sum(1 for component in components if component.class_label == "line_like")
    marker_like_count = sum(1 for component in components if component.class_label == "marker_like")
    if line_style == "unknown":
        return 0.5 if line_like_count else 0.25
    if line_style == "solid":
        return 1.0 if line_like_count else 0.25
    if line_style in {"dashed", "dotted"}:
        if marker_like_count > line_like_count:
            return 0.85
        return 0.65 if len(components) > 3 else 0.35
    return 0.5


def _marker_similarity(marker_style, markers):
    if not markers:
        return 0.0
    if marker_style == "unknown":
        return 0.5
    known_shapes = {marker.shape for marker in markers if marker.shape != "unknown"}
    if not known_shapes:
        return 0.35
    if marker_style in known_shapes:
        return 1.0
    if marker_style in {"single_compact", "repeated_compact"}:
        return 0.75
    return 0.25


def _marker_instance_similarity(proto, instance):
    if instance.marker_count <= 0:
        return 0.0
    base = min(1.0, instance.marker_count / 8.0)
    if proto.marker_style == "unknown":
        return 0.80 * base
    return min(1.0, 0.80 * base + 0.20)


def _line_style_instance_similarity(prototype_style, instance_style):
    if prototype_style == "unknown" or instance_style == "unknown":
        return 0.45
    if prototype_style == instance_style:
        return 1.0
    if prototype_style in {"dashed", "dotted"} and instance_style in {"dashed", "dotted"}:
        return 0.65
    if prototype_style == "solid" and instance_style in {"dashed", "dotted"}:
        return 0.20
    if instance_style == "solid" and prototype_style in {"dashed", "dotted"}:
        return 0.20
    return 0.35


def _is_achromatic_rgb(rgb, tolerance=18.0):
    if rgb is None:
        return False
    r, g, b = [float(value) for value in rgb]
    return max(abs(r - g), abs(r - b), abs(g - b)) <= tolerance


def _ordinal_similarity(proto_idx, instance_idx, prototype_count, instance_count):
    if prototype_count <= 1 or instance_count <= 1:
        return 0.5
    scale = max(prototype_count - 1, instance_count - 1, 1)
    return max(0.0, 1.0 - abs(float(proto_idx) - float(instance_idx)) / float(scale))


def _binding_summary(bindings):
    if not bindings:
        return {"binding_count": 0, "best_by_prototype": []}
    best = {}
    for binding in bindings:
        current = best.get(binding.prototype_id)
        if current is None or binding.score > current.score:
            best[binding.prototype_id] = binding
    return {
        "binding_count": len(bindings),
        "prototype_count": len(best),
        "mean_best_score": _mean_values([binding.score for binding in best.values()]),
        "best_by_prototype": [
            {
                "prototype_id": binding.prototype_id,
                "legend_item_id": binding.legend_item_id,
                "target_curve_id": binding.target_curve_id,
                "target_type": binding.target_type,
                "score": binding.score,
                "warnings": binding.warnings,
            }
            for binding in sorted(best.values(), key=lambda item: item.prototype_id)
        ],
    }


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


def _point_path_length(points):
    if len(points) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(points, points[1:]):
        total += ((float(a.x) - float(b.x)) ** 2 + (float(a.y) - float(b.y)) ** 2) ** 0.5
    return float(total)


def _point_x_span(points):
    if not points:
        return 0.0
    xs = [float(point.x) for point in points]
    return max(xs) - min(xs)


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
