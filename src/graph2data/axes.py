"""Axis and plot-area detection."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .models import AxisDetection, AxisLine, BoundingBox, PlotArea, Point


@dataclass
class AxisDetectorConfig:
    black_thresh: int = 100
    use_min_channel: bool = False
    use_adaptive: bool = False
    use_morph: bool = False
    hough_thresh_ratio: float = 0.06
    hough_min_len_ratio: float = 0.08
    hough_gap_ratio: float = 0.2
    min_axis_length_ratio: float = 0.35
    angle_tolerance: float = 5.0
    refine_range: int = 8


class AxisDetector:
    """Detect the primary x/y axes and infer a rectangular plot area."""

    def __init__(self, config: Optional[AxisDetectorConfig] = None):
        self.config = config or AxisDetectorConfig()

    def detect(self, image_bgr: np.ndarray) -> AxisDetection:
        start_time = time.time()
        h, w = image_bgr.shape[:2]
        binary = self._preprocess(image_bgr)
        lines = self._hough_lines(binary)
        if lines is None:
            return AxisDetection(
                success=False,
                image_size=(w, h),
                error="No lines detected",
                warnings=[f"processing_time={time.time() - start_time:.4f}s"],
            )

        horizontal, vertical = self._collect_candidates(lines, w, h)
        best_h = self._select_horizontal(horizontal, w)
        best_v = self._select_vertical(vertical, h)
        candidates = {"horizontal": horizontal, "vertical": vertical}

        if best_h is None or best_v is None:
            return AxisDetection(
                success=False,
                image_size=(w, h),
                candidates=candidates,
                error="No valid axis pair found",
                warnings=[f"processing_time={time.time() - start_time:.4f}s"],
            )

        x1, y1, x2, y2 = best_h["coords"]
        vx1, vy1, vx2, vy2 = best_v["coords"]
        x_min_axis, x_max_axis = min(x1, x2), max(x1, x2)
        y_min_axis, y_max_axis = min(vy1, vy2), max(vy1, vy2)

        final_y = self._refine_position(binary, best_h["pos"], axis=0, limit_range=(x_min_axis, x_max_axis))
        final_x = self._refine_position(binary, best_v["pos"], axis=1, limit_range=(y_min_axis, y_max_axis))

        origin = Point(float(final_x), float(final_y))
        x_end = Point(float(x_max_axis), float(final_y))
        y_end = Point(float(final_x), float(y_min_axis))

        x_axis = AxisLine(origin, x_end, "horizontal", confidence=best_h["score"])
        y_axis = AxisLine(origin, y_end, "vertical", confidence=best_v["score"])

        bbox = BoundingBox(
            x_min=float(final_x),
            y_min=float(y_min_axis),
            x_max=float(x_max_axis),
            y_max=float(final_y),
        )
        plot_conf = self._plot_confidence(bbox, w, h, best_h["score"], best_v["score"])
        plot_area = PlotArea(bbox=bbox, confidence=plot_conf)

        warnings = [f"processing_time={time.time() - start_time:.4f}s"]
        if bbox.width <= 0 or bbox.height <= 0:
            warnings.append("Detected plot area has non-positive dimensions")

        return AxisDetection(
            success=True,
            image_size=(w, h),
            origin=origin,
            x_axis=x_axis,
            y_axis=y_axis,
            plot_area=plot_area,
            confidence=plot_conf,
            candidates=candidates,
            warnings=warnings,
        )

    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        cfg = self.config
        if cfg.use_min_channel:
            gray = np.min(image_bgr, axis=2).astype(np.uint8)
        else:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        if cfg.use_adaptive:
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 10
            )
        elif cfg.use_min_channel:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        else:
            _, binary = cv2.threshold(gray, cfg.black_thresh, 255, cv2.THRESH_BINARY_INV)

        if cfg.use_morph:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    def _hough_lines(self, binary: np.ndarray):
        h, w = binary.shape
        long_side = max(h, w)
        min_len = max(20, int(long_side * self.config.hough_min_len_ratio))
        max_gap = max(5, int(min_len * self.config.hough_gap_ratio))
        threshold = max(30, int(long_side * self.config.hough_thresh_ratio))
        return cv2.HoughLinesP(binary, 1, np.pi / 180, threshold=threshold, minLineLength=min_len, maxLineGap=max_gap)

    def _collect_candidates(self, lines, width: int, height: int) -> Tuple[List[Dict], List[Dict]]:
        horizontal: List[Dict] = []
        vertical: List[Dict] = []
        tol = self.config.angle_tolerance
        min_h_len = width * self.config.min_axis_length_ratio
        min_v_len = height * self.config.min_axis_length_ratio

        for line in lines:
            x1, y1, x2, y2 = [float(v) for v in line[0]]
            dx, dy = x2 - x1, y2 - y1
            length = math.hypot(dx, dy)
            if length <= 0:
                continue
            angle = math.degrees(math.atan2(dy, dx))
            if angle < 0:
                angle += 180

            if (angle < tol or angle > 180 - tol) and length >= min_h_len:
                pos = (y1 + y2) / 2
                horizontal.append(
                    {
                        "coords": (x1, y1, x2, y2),
                        "length": length,
                        "pos": pos,
                        "center": ((x1 + x2) / 2, pos),
                        "score": min(1.0, length / max(width, 1)),
                    }
                )
            elif 90 - tol < angle < 90 + tol and length >= min_v_len:
                pos = (x1 + x2) / 2
                vertical.append(
                    {
                        "coords": (x1, y1, x2, y2),
                        "length": length,
                        "pos": pos,
                        "center": (pos, (y1 + y2) / 2),
                        "score": min(1.0, length / max(height, 1)),
                    }
                )

        return horizontal, vertical

    def _select_horizontal(self, candidates: List[Dict], image_width: int) -> Optional[Dict]:
        if not candidates:
            return None
        max_len = max(c["length"] for c in candidates)
        long_candidates = [c for c in candidates if c["length"] >= max_len * 0.65]
        long_candidates.sort(key=lambda c: (c["center"][1], c["length"]), reverse=True)
        return long_candidates[0]

    def _select_vertical(self, candidates: List[Dict], image_height: int) -> Optional[Dict]:
        if not candidates:
            return None
        max_len = max(c["length"] for c in candidates)
        long_candidates = [c for c in candidates if c["length"] >= max_len * 0.65]
        long_candidates.sort(key=lambda c: (c["center"][0], -c["length"]))
        return long_candidates[0]

    def _refine_position(self, binary: np.ndarray, rough_pos: float, axis: int, limit_range: Tuple[float, float]) -> int:
        h, w = binary.shape
        start = int(max(0, rough_pos - self.config.refine_range))
        end = int(min(binary.shape[1 - axis], rough_pos + self.config.refine_range + 1))
        limit_min, limit_max = [int(v) for v in limit_range]
        best_pos = int(rough_pos)
        best_density = -1

        for pos in range(start, end):
            if axis == 0:
                lo, hi = max(0, limit_min), min(w, limit_max)
                density = np.count_nonzero(binary[pos, lo:hi]) if hi > lo else 0
            else:
                lo, hi = max(0, limit_min), min(h, limit_max)
                density = np.count_nonzero(binary[lo:hi, pos]) if hi > lo else 0
            if density > best_density:
                best_density = density
                best_pos = pos
        return best_pos

    def _plot_confidence(self, bbox: BoundingBox, width: int, height: int, h_score: float, v_score: float) -> float:
        area_ratio = max(0.0, min(1.0, (bbox.width * bbox.height) / max(width * height, 1)))
        axis_score = (h_score + v_score) / 2
        return float(max(0.0, min(1.0, 0.75 * axis_score + 0.25 * area_ratio)))
