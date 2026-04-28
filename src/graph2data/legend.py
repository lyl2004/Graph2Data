"""Heuristic legend region detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .models import BoundingBox, LegendDetection, OCRTextBox, PlotArea


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
