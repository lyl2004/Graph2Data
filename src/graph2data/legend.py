"""Heuristic legend region detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .models import BoundingBox, LegendDetection, OCRTextBox, PlotArea


@dataclass
class LegendDetectorConfig:
    max_texts: int = 12
    padding_px: float = 10.0
    min_confidence: float = 0.15


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
