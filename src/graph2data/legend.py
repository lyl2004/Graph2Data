"""Heuristic legend region detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .models import BoundingBox, CurveVisualPrototype, LegendDetection, LegendItem, OCRTextBox, PlotArea


@dataclass
class LegendDetectorConfig:
    max_texts: int = 12
    padding_px: float = 10.0
    min_confidence: float = 0.15
    image_gray_threshold: int = 245
    image_min_component_area: int = 300
    image_min_width_px: int = 55
    image_min_height_px: int = 35
    image_max_area_ratio: float = 0.20
    image_min_area_ratio: float = 0.004
    image_min_density: float = 0.025
    image_max_density: float = 0.35
    image_max_top_offset_ratio: float = 0.25
    image_corner_margin_ratio: float = 0.06
    image_min_frame_score: float = 0.65
    image_unframed_search_width_ratio: float = 0.38
    image_unframed_search_height_ratio: float = 0.45
    image_unframed_min_dark_pixels: int = 260
    image_unframed_min_width_px: int = 80
    image_unframed_min_height_px: int = 45
    image_unframed_max_density: float = 0.30
    image_unframed_min_density: float = 0.015
    image_unframed_max_area_ratio: float = 0.14
    image_unframed_max_width_ratio: float = 0.36
    item_gray_threshold: int = 245
    item_min_foreground_pixels: int = 12
    item_min_row_foreground_px: int = 8
    item_min_height_px: int = 6
    item_max_height_px: int = 40
    item_row_gap_px: int = 7
    item_min_width_px: int = 26
    item_sample_max_width_px: int = 70
    item_sample_default_width_px: int = 46
    item_text_gap_px: int = 7


class LegendDetector:
    """Detect likely legend boxes inside the plot area from OCR text clusters.

    This is intentionally conservative. It is meant to mask obvious in-plot
    legends before color extraction, not to solve full legend parsing.
    """

    def __init__(self, config: Optional[LegendDetectorConfig] = None):
        self.config = config or LegendDetectorConfig()

    def detect(self, texts: List[OCRTextBox], plot_area: PlotArea) -> List[LegendDetection]:
        inside = [(idx, t) for idx, t in enumerate(texts) if plot_area.bbox.contains(t.center)]
        if not inside:
            return []

        clusters = self._cluster_by_rows(inside)
        detections: List[LegendDetection] = []
        for cluster in clusters:
            if len(cluster) > self.config.max_texts:
                continue
            bbox = self._cluster_bbox([t for _, t in cluster], plot_area.bbox)
            if bbox is None:
                continue
            area_ratio = (bbox.width * bbox.height) / max(plot_area.bbox.width * plot_area.bbox.height, 1.0)
            if area_ratio > 0.25:
                continue
            conf = min(1.0, 0.25 + 0.15 * len(cluster))
            if conf >= self.config.min_confidence:
                detections.append(
                    LegendDetection(
                        bbox=bbox,
                        confidence=conf,
                        text_indices=[idx for idx, _ in cluster],
                    )
                )
        return detections

    def detect_from_image(self, image_bgr: np.ndarray, plot_area: PlotArea) -> List[LegendDetection]:
        """Detect compact in-plot legend clusters without OCR.

        Framed Matplotlib-like legends often form a medium-sized connected
        component of text, sample lines, and frame pixels. Plot curves and grids
        are either full-plot components or much smaller fragments, so this
        method deliberately keeps only compact medium-density components.
        """
        crop, offset = self._plot_crop(image_bgr, plot_area.bbox)
        if crop.size == 0:
            return []

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mask = (gray < self.config.image_gray_threshold).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        h, w = mask.shape[:2]
        plot_area_px = max(1, h * w)
        candidates = []

        for idx in range(1, num):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            width = int(stats[idx, cv2.CC_STAT_WIDTH])
            height = int(stats[idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[idx, cv2.CC_STAT_AREA])
            box_area = max(1, width * height)
            area_ratio = box_area / plot_area_px
            density = area / box_area

            if area < self.config.image_min_component_area:
                continue
            if width < self.config.image_min_width_px or height < self.config.image_min_height_px:
                continue
            if area_ratio < self.config.image_min_area_ratio or area_ratio > self.config.image_max_area_ratio:
                continue
            if density < self.config.image_min_density or density > self.config.image_max_density:
                continue
            component = labels[y : y + height, x : x + width] == idx
            frame_score = _frame_score(component)
            near_top = (y / max(h, 1)) <= self.config.image_max_top_offset_ratio
            near_corner = self._is_near_plot_corner(x, y, width, height, w, h)
            if not near_top and not near_corner and frame_score < self.config.image_min_frame_score:
                continue

            bbox = self._component_bbox(x, y, width, height, offset, plot_area.bbox)
            if bbox is None:
                continue
            score = self._image_candidate_score(area_ratio, density, frame_score)
            candidates.append((score, bbox))

        if not candidates:
            candidates.extend(self._detect_unframed_upper_right_legend(gray, offset, plot_area.bbox))

        candidates.sort(key=lambda item: item[0], reverse=True)
        detections = []
        for score, bbox in candidates:
            if any(_bbox_iou(bbox, existing.bbox) > 0.4 or _bbox_containment(bbox, existing.bbox) > 0.85 for existing in detections):
                continue
            detections.append(
                LegendDetection(
                    bbox=bbox,
                    confidence=float(score),
                    source="image_heuristic",
                )
            )
        return detections

    def extract_items(self, image_bgr: np.ndarray, legends: List[LegendDetection]) -> List[LegendItem]:
        """Split detected legend boxes into visual sample rows.

        The output is intentionally diagnostic: it records where each legend
        item appears and which pixels look like its sample line/marker. Later
        stages can bind these prototypes to plot-area curves.
        """
        items: List[LegendItem] = []
        for legend_index, legend in enumerate(legends):
            crop, offset = self._legend_crop(image_bgr, legend.bbox)
            if crop.size == 0:
                continue
            rows = self._legend_item_rows(crop)
            for row_index, row in enumerate(rows):
                item = self._legend_item_from_row(
                    image_bgr=image_bgr,
                    crop=crop,
                    offset=offset,
                    legend_index=legend_index,
                    row_index=row_index,
                    row=row,
                )
                if item is not None:
                    items.append(item)
        return items

    def assign_item_labels_from_ocr(self, items: List[LegendItem], texts: List[OCRTextBox]) -> List[LegendItem]:
        """Attach OCR text to legend items by matching OCR boxes to item text regions."""
        if not items or not texts:
            return items

        for item in items:
            if item.text_bbox is None:
                item.warnings.append("legend_label_missing_text_bbox")
                continue

            matches = []
            for text in texts:
                cleaned = _normalize_label_text(text.text)
                if not cleaned:
                    continue
                center_inside = item.text_bbox.contains(text.center)
                overlap = _bbox_containment(text.bbox, item.text_bbox)
                if center_inside or overlap >= 0.20:
                    matches.append(text)

            if not matches:
                item.warnings.append("legend_label_missing")
                continue

            matches.sort(key=lambda text: (text.bbox.y_min, text.bbox.x_min))
            label = _normalize_label_text(" ".join(text.text for text in matches))
            if label:
                item.label = label
            else:
                item.warnings.append("legend_label_empty")
        return items

    def visual_prototypes_from_items(self, items: List[LegendItem]) -> List[CurveVisualPrototype]:
        prototypes: List[CurveVisualPrototype] = []
        for item in items:
            if item.rgb is None or item.lab is None or item.sample_bbox is None:
                continue
            prototypes.append(
                CurveVisualPrototype(
                    prototype_id=f"legend_proto_{len(prototypes):02d}",
                    legend_item_id=item.item_id,
                    rgb=item.rgb,
                    lab=item.lab,
                    sample_bbox=item.sample_bbox,
                    label=item.label,
                    line_style=item.line_style,
                    marker_style=item.marker_style,
                    confidence=item.confidence,
                    source="legend_item",
                )
            )
        return prototypes

    def _detect_unframed_upper_right_legend(self, gray: np.ndarray, offset, plot_bbox: BoundingBox):
        """Detect compact unframed legend clusters in the upper-right plot corner."""
        h, w = gray.shape[:2]
        sx0 = int(max(0, w * (1.0 - self.config.image_unframed_search_width_ratio)))
        sy0 = 0
        sx1 = w
        sy1 = int(max(1, h * self.config.image_unframed_search_height_ratio))
        roi = gray[sy0:sy1, sx0:sx1]
        if roi.size == 0:
            return []

        dark = (roi < self.config.image_gray_threshold).astype(np.uint8) * 255
        # Join nearby text glyphs, sample lines, and markers into a single legend cluster.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        joined = cv2.dilate(dark, kernel, iterations=1)
        num, labels, stats, _ = cv2.connectedComponentsWithStats((joined > 0).astype(np.uint8), connectivity=8)
        candidates = []
        for idx in range(1, num):
            x = int(stats[idx, cv2.CC_STAT_LEFT])
            y = int(stats[idx, cv2.CC_STAT_TOP])
            width = int(stats[idx, cv2.CC_STAT_WIDTH])
            height = int(stats[idx, cv2.CC_STAT_HEIGHT])
            height = _extend_component_down(dark, x, y, width, height)
            original_dark = dark[y : y + height, x : x + width] > 0
            dark_pixels = int(original_dark.sum())
            box_area = max(1, width * height)
            density = dark_pixels / box_area
            if dark_pixels < self.config.image_unframed_min_dark_pixels:
                continue
            if width < self.config.image_unframed_min_width_px or height < self.config.image_unframed_min_height_px:
                continue
            if width / max(w, 1) > self.config.image_unframed_max_width_ratio:
                continue
            if density < self.config.image_unframed_min_density or density > self.config.image_unframed_max_density:
                continue

            bbox = self._component_bbox(
                sx0 + x,
                sy0 + y,
                width,
                height,
                offset,
                plot_bbox,
            )
            if bbox is None:
                continue
            area_ratio = (bbox.width * bbox.height) / max(plot_bbox.width * plot_bbox.height, 1.0)
            if area_ratio > self.config.image_unframed_max_area_ratio:
                continue
            score = min(1.0, 0.45 + 0.30 * min(1.0, dark_pixels / 900.0) + 0.25 * min(1.0, width / 180.0))
            candidates.append((score, bbox))
        return candidates

    def _legend_crop(self, image_bgr: np.ndarray, bbox: BoundingBox):
        h, w = image_bgr.shape[:2]
        x0 = max(0, int(round(bbox.x_min)))
        y0 = max(0, int(round(bbox.y_min)))
        x1 = min(w, int(round(bbox.x_max)))
        y1 = min(h, int(round(bbox.y_max)))
        return image_bgr[y0:y1, x0:x1].copy(), (x0, y0)

    def _legend_item_rows(self, crop: np.ndarray) -> List[Tuple[int, int]]:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        foreground = (gray < self.config.item_gray_threshold).astype(np.uint8)
        foreground = _remove_frame_like_border(foreground)
        if not np.any(foreground):
            return []

        row_counts = np.count_nonzero(foreground, axis=1)
        active_rows = np.where(row_counts >= self.config.item_min_row_foreground_px)[0]
        if active_rows.size == 0:
            return []

        raw_bands: List[Tuple[int, int]] = []
        start = int(active_rows[0])
        prev = int(active_rows[0])
        for row in active_rows[1:]:
            row = int(row)
            if row - prev <= self.config.item_row_gap_px:
                prev = row
                continue
            raw_bands.append((start, prev + 1))
            start = row
            prev = row
        raw_bands.append((start, prev + 1))

        rows = []
        h, _ = foreground.shape[:2]
        for y0, y1 in raw_bands:
            y0 = max(0, y0 - 2)
            y1 = min(h, y1 + 3)
            band = foreground[y0:y1, :]
            xs = np.where(np.any(band > 0, axis=0))[0]
            if xs.size == 0:
                continue
            width = int(xs.max() - xs.min() + 1)
            height = int(y1 - y0)
            pixels = int(np.count_nonzero(band))
            if pixels < self.config.item_min_foreground_pixels:
                continue
            if width < self.config.item_min_width_px:
                continue
            if height < self.config.item_min_height_px or height > self.config.item_max_height_px:
                continue
            rows.append((y0, y1, pixels))
        return [(y0, y1) for y0, y1, _ in _filter_legend_item_row_artifacts(rows)]

    def _legend_item_from_row(
        self,
        image_bgr: np.ndarray,
        crop: np.ndarray,
        offset,
        legend_index: int,
        row_index: int,
        row: Tuple[int, int],
    ) -> Optional[LegendItem]:
        y0, y1 = row
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        foreground = (gray < self.config.item_gray_threshold).astype(np.uint8)
        foreground = _remove_frame_like_border(foreground)
        band = foreground[y0:y1, :]
        ys, xs = np.where(band > 0)
        if xs.size == 0:
            return None

        x_min = int(xs.min())
        x_max = int(xs.max()) + 1
        split_x = self._sample_text_split_x(band, x_min, x_max)
        sample_x0 = x_min
        sample_x1 = min(x_max, split_x)
        if sample_x1 - sample_x0 < 8:
            sample_x1 = min(x_max, sample_x0 + self.config.item_sample_default_width_px)
        sample_x1 = min(sample_x1, sample_x0 + self.config.item_sample_max_width_px)
        text_x0 = min(x_max, max(sample_x1 + self.config.item_text_gap_px, split_x + 1))

        item_bbox = _bbox_from_local(x_min, y0, x_max, y1, offset, pad=2.0)
        sample_bbox = _bbox_from_local(sample_x0, y0, sample_x1, y1, offset, pad=2.0)
        text_bbox = None
        if x_max > text_x0:
            text_bbox = _bbox_from_local(text_x0, y0, x_max, y1, offset, pad=2.0)

        rgb, lab, sample_pixels = _sample_color_and_lab(image_bgr, sample_bbox, threshold=self.config.item_gray_threshold)
        warnings = []
        if sample_pixels < self.config.item_min_foreground_pixels:
            warnings.append("low_sample_foreground")
        sample_mask = foreground[y0:y1, sample_x0:sample_x1]
        line_style = _classify_sample_line_style(sample_mask)
        marker_style = _classify_sample_marker_style(sample_mask)
        confidence = min(1.0, 0.35 + sample_pixels / 80.0)
        if text_bbox is None:
            warnings.append("text_region_not_found")
            confidence = min(confidence, 0.65)

        return LegendItem(
            item_id=f"legend_{legend_index:02d}_item_{row_index:02d}",
            legend_index=legend_index,
            bbox=item_bbox,
            sample_bbox=sample_bbox,
            text_bbox=text_bbox,
            rgb=rgb,
            lab=lab,
            foreground_pixel_count=int(sample_pixels),
            line_style=line_style,
            marker_style=marker_style,
            confidence=float(confidence),
            source="image_heuristic",
            warnings=warnings,
        )

    def _sample_text_split_x(self, row_mask: np.ndarray, x_min: int, x_max: int) -> int:
        col_active = np.any(row_mask > 0, axis=0)
        active = np.where(col_active)[0]
        if active.size <= 1:
            return min(x_max, x_min + self.config.item_sample_default_width_px)
        gaps = []
        for left, right in zip(active[:-1], active[1:]):
            gap = int(right - left - 1)
            if gap >= self.config.item_text_gap_px:
                gaps.append((gap, int(left), int(right)))
        min_sample_end = x_min + 12
        max_sample_end = min(x_max, x_min + self.config.item_sample_max_width_px)
        candidates = [
            (gap, left, right)
            for gap, left, right in gaps
            if min_sample_end <= left <= max_sample_end
        ]
        if candidates:
            _, left, _ = max(candidates, key=lambda item: item[0])
            return int(left + 1)
        return min(x_max, x_min + self.config.item_sample_default_width_px)

    def _cluster_by_rows(self, indexed_texts):
        rows = sorted(indexed_texts, key=lambda item: item[1].center.y)
        clusters = []
        current = []
        current_y = None
        for item in rows:
            text = item[1]
            if current_y is None or abs(text.center.y - current_y) <= max(14.0, text.bbox.height * 1.2):
                current.append(item)
                current_y = text.center.y if current_y is None else (current_y * 0.7 + text.center.y * 0.3)
            else:
                clusters.append(current)
                current = [item]
                current_y = text.center.y
        if current:
            clusters.append(current)

        merged = []
        i = 0
        while i < len(clusters):
            group = clusters[i]
            j = i + 1
            while j < len(clusters) and self._row_overlap(group, clusters[j]):
                group = group + clusters[j]
                j += 1
            merged.append(group)
            i = j
        return merged

    def _row_overlap(self, a, b) -> bool:
        ax0 = min(t.bbox.x_min for _, t in a)
        ax1 = max(t.bbox.x_max for _, t in a)
        bx0 = min(t.bbox.x_min for _, t in b)
        bx1 = max(t.bbox.x_max for _, t in b)
        return not (ax1 < bx0 or bx1 < ax0)

    def _cluster_bbox(self, texts: List[OCRTextBox], plot_bbox: BoundingBox):
        if not texts:
            return None
        pad = self.config.padding_px
        x_min = max(plot_bbox.x_min, min(t.bbox.x_min for t in texts) - pad - 35.0)
        y_min = max(plot_bbox.y_min, min(t.bbox.y_min for t in texts) - pad)
        x_max = min(plot_bbox.x_max, max(t.bbox.x_max for t in texts) + pad)
        y_max = min(plot_bbox.y_max, max(t.bbox.y_max for t in texts) + pad)
        if x_max <= x_min or y_max <= y_min:
            return None
        return BoundingBox(x_min, y_min, x_max, y_max)

    def _plot_crop(self, image_bgr: np.ndarray, plot_bbox: BoundingBox):
        h, w = image_bgr.shape[:2]
        x0 = max(0, int(plot_bbox.x_min))
        y0 = max(0, int(plot_bbox.y_min))
        x1 = min(w, int(plot_bbox.x_max))
        y1 = min(h, int(plot_bbox.y_max))
        return image_bgr[y0:y1, x0:x1], (x0, y0)

    def _component_bbox(self, x: int, y: int, width: int, height: int, offset, plot_bbox: BoundingBox):
        pad = self.config.padding_px
        x_off, y_off = offset
        x_min = max(plot_bbox.x_min, x_off + x - pad)
        y_min = max(plot_bbox.y_min, y_off + y - pad)
        x_max = min(plot_bbox.x_max, x_off + x + width + pad)
        y_max = min(plot_bbox.y_max, y_off + y + height + pad)
        if x_max <= x_min or y_max <= y_min:
            return None
        return BoundingBox(float(x_min), float(y_min), float(x_max), float(y_max))

    def _image_candidate_score(self, area_ratio: float, density: float, frame_score: float) -> float:
        area_score = min(1.0, area_ratio / 0.05)
        density_score = 1.0 - min(1.0, abs(density - 0.08) / 0.08)
        return max(
            self.config.min_confidence,
            min(1.0, 0.35 + 0.30 * area_score + 0.20 * density_score + 0.15 * frame_score),
        )

    def _is_near_plot_corner(self, x: int, y: int, width: int, height: int, plot_width: int, plot_height: int) -> bool:
        margin = self.config.image_corner_margin_ratio
        near_left = x / max(plot_width, 1) <= margin
        near_right = (x + width) / max(plot_width, 1) >= 1.0 - margin
        near_top = y / max(plot_height, 1) <= margin
        near_bottom = (y + height) / max(plot_height, 1) >= 1.0 - margin
        return (near_left or near_right) and (near_top or near_bottom)


def _bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    x0 = max(a.x_min, b.x_min)
    y0 = max(a.y_min, b.y_min)
    x1 = min(a.x_max, b.x_max)
    y1 = min(a.y_max, b.y_max)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    return inter / max(union, 1.0)


def _bbox_containment(a: BoundingBox, b: BoundingBox) -> float:
    x0 = max(a.x_min, b.x_min)
    y0 = max(a.y_min, b.y_min)
    x1 = min(a.x_max, b.x_max)
    y1 = min(a.y_max, b.y_max)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    smaller = max(1.0, min(a.width * a.height, b.width * b.height))
    return inter / smaller


def _normalize_label_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _frame_score(component_mask: np.ndarray) -> float:
    if component_mask.size == 0:
        return 0.0
    h, w = component_mask.shape[:2]
    if h < 6 or w < 6:
        return 0.0
    t = min(3, max(1, min(h, w) // 12))
    top = np.mean(np.any(component_mask[:t, :], axis=0))
    bottom = np.mean(np.any(component_mask[h - t :, :], axis=0))
    left = np.mean(np.any(component_mask[:, :t], axis=1))
    right = np.mean(np.any(component_mask[:, w - t :], axis=1))
    return float(min(top, bottom, left, right))


def _extend_component_down(dark: np.ndarray, x: int, y: int, width: int, height: int) -> int:
    """Extend an unframed legend box through vertically separated legend rows."""
    h, _ = dark.shape[:2]
    x0 = max(0, x - 8)
    x1 = min(dark.shape[1], x + width + 8)
    last = y + height
    gap = 0
    for row in range(y + height, h):
        row_dark = int(np.count_nonzero(dark[row, x0:x1]))
        if row_dark >= 4:
            last = row + 1
            gap = 0
        else:
            gap += 1
            if gap > 28:
                break
    return max(height, last - y)


def _remove_frame_like_border(mask: np.ndarray) -> np.ndarray:
    """Drop obvious legend frame pixels while keeping inner item strokes."""
    out = mask.copy()
    h, w = out.shape[:2]
    if h < 8 or w < 8:
        return out
    t = min(4, max(1, min(h, w) // 20))
    for row in range(t):
        if np.count_nonzero(out[row, :]) > 0.45 * w:
            out[row, :] = 0
    for row in range(max(0, h - t), h):
        if np.count_nonzero(out[row, :]) > 0.45 * w:
            out[row, :] = 0
    for col in range(t):
        if np.count_nonzero(out[:, col]) > 0.45 * h:
            out[:, col] = 0
    for col in range(max(0, w - t), w):
        if np.count_nonzero(out[:, col]) > 0.45 * h:
            out[:, col] = 0
    return out


def _filter_legend_item_row_artifacts(rows: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    filtered: List[Tuple[int, int, int]] = []
    for row in rows:
        y0, y1, pixels = row
        height = y1 - y0
        if filtered:
            prev_y0, prev_y1, prev_pixels = filtered[-1]
            near_previous = y0 <= prev_y1 + 4
            short_tail = height <= 12 and y0 >= prev_y0
            if near_previous and short_tail:
                continue
        filtered.append(row)
    return filtered


def _bbox_from_local(x0: int, y0: int, x1: int, y1: int, offset, pad: float = 0.0) -> BoundingBox:
    x_off, y_off = offset
    return BoundingBox(
        float(x_off + x0) - pad,
        float(y_off + y0) - pad,
        float(x_off + x1) + pad,
        float(y_off + y1) + pad,
    )


def _sample_color_and_lab(image_bgr: np.ndarray, bbox: BoundingBox, threshold: int):
    h, w = image_bgr.shape[:2]
    x0 = max(0, int(round(bbox.x_min)))
    y0 = max(0, int(round(bbox.y_min)))
    x1 = min(w, int(round(bbox.x_max)))
    y1 = min(h, int(round(bbox.y_max)))
    if x1 <= x0 or y1 <= y0:
        return None, None, 0
    crop = image_bgr[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    foreground = gray < threshold
    pixel_count = int(np.count_nonzero(foreground))
    if pixel_count <= 0:
        return None, None, 0
    mean_bgr = crop[foreground].mean(axis=0)
    lab_pix = cv2.cvtColor(np.uint8([[mean_bgr]]), cv2.COLOR_BGR2Lab)[0][0]
    rgb = (int(round(mean_bgr[2])), int(round(mean_bgr[1])), int(round(mean_bgr[0])))
    lab = (float(lab_pix[0]), float(lab_pix[1]), float(lab_pix[2]))
    return rgb, lab, pixel_count


def _classify_sample_line_style(sample_mask: np.ndarray) -> str:
    if sample_mask.size == 0 or not np.any(sample_mask):
        return "unknown"
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(sample_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    if not np.any(cleaned):
        cleaned = sample_mask.astype(np.uint8)
    num, _, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    spans = []
    compact = 0
    for idx in range(1, num):
        width = int(stats[idx, cv2.CC_STAT_WIDTH])
        height = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < 2:
            continue
        spans.append(width)
        if max(width, height) <= 8:
            compact += 1
    if not spans:
        return "unknown"
    if len(spans) <= 2 and max(spans) >= 18:
        return "solid"
    if compact >= 4:
        return "dotted"
    if len(spans) >= 3:
        return "dashed"
    return "unknown"


def _classify_sample_marker_style(sample_mask: np.ndarray) -> str:
    if sample_mask.size == 0 or not np.any(sample_mask):
        return "unknown"
    num, labels, stats, _ = cv2.connectedComponentsWithStats(sample_mask.astype(np.uint8), connectivity=8)
    compact_components = 0
    for idx in range(1, num):
        width = int(stats[idx, cv2.CC_STAT_WIDTH])
        height = int(stats[idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < 3:
            continue
        if max(width, height) <= 14 and abs(width - height) <= 5:
            compact_components += 1
    if compact_components >= 2:
        return "repeated_compact"
    if compact_components == 1:
        return "single_compact"
    return "unknown"
