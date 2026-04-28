"""Curve mask generation from color prototypes."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .image_io import ensure_dir, load_bgr, write_json
from .models import BoundingBox, CurveMask, CurvePrototype, to_serializable


@dataclass
class MaskExtractorConfig:
    range_l: float = 55.0
    range_ab: float = 16.0
    morph_open: int = 1
    morph_close: int = 1
    min_component_area: int = 4
    min_component_x_span: int = 3
    inset_px: int = 3
    drop_border_components: bool = True
    drop_grid_like_components: bool = True
    drop_axis_tick_components: bool = True
    axis_tick_edge_margin_px: int = 4
    axis_tick_max_span_px: int = 3
    axis_tick_max_thickness_px: int = 4
    suppress_dark_neighbors_for_gray: bool = True
    dark_neighbor_l: int = 80
    dark_neighbor_dilate: int = 3


class CurveMaskExtractor:
    def __init__(self, config: Optional[MaskExtractorConfig] = None):
        self.config = config or MaskExtractorConfig()

    def extract_mask(
        self,
        image_bgr: np.ndarray,
        prototype: CurvePrototype,
        region: Optional[BoundingBox] = None,
        exclude_regions: Optional[Sequence[BoundingBox]] = None,
    ) -> Tuple[np.ndarray, CurveMask]:
        crop, offset = _crop(image_bgr, region, self.config.inset_px)
        if crop.size == 0:
            empty = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
            return empty, CurveMask(curve_id=prototype.curve_id, warnings=["empty region"])

        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab).astype(np.float32)
        target = np.array(prototype.lab, dtype=np.float32)
        diff_l = np.abs(lab[:, :, 0] - target[0])
        diff_ab = np.sqrt((lab[:, :, 1] - target[1]) ** 2 + (lab[:, :, 2] - target[2]) ** 2)
        range_l = min(self.config.range_l, 24.0) if prototype.source == "guided_gray" else self.config.range_l
        mask = ((diff_l <= range_l) & (diff_ab <= self.config.range_ab)).astype(np.uint8) * 255
        if exclude_regions:
            _apply_exclusions(mask, exclude_regions, offset)
        if self._is_gray_prototype(prototype):
            mask = self._suppress_dark_neighbors(mask, lab)
        mask = self._cleanup(mask)

        full = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
        x0, y0 = offset
        h, w = mask.shape[:2]
        full[y0 : y0 + h, x0 : x0 + w] = mask
        bbox = _mask_bbox(full)
        pixel_count = int(cv2.countNonZero(full))
        confidence = min(1.0, pixel_count / max(prototype.area, 1)) if prototype.area else 0.0
        result = CurveMask(
            curve_id=prototype.curve_id,
            pixel_count=pixel_count,
            bbox=bbox,
            confidence=float(confidence),
            source="color_threshold",
        )
        if pixel_count == 0:
            result.warnings.append("empty mask")
        return full, result

    def _is_gray_prototype(self, prototype: CurvePrototype) -> bool:
        lab = prototype.lab
        chroma = float(((lab[1] - 128.0) ** 2 + (lab[2] - 128.0) ** 2) ** 0.5)
        return self.config.suppress_dark_neighbors_for_gray and chroma <= 14.0 and lab[0] > self.config.dark_neighbor_l

    def _suppress_dark_neighbors(self, mask: np.ndarray, lab: np.ndarray) -> np.ndarray:
        dark = (lab[:, :, 0] <= self.config.dark_neighbor_l).astype(np.uint8) * 255
        k = max(1, int(self.config.dark_neighbor_dilate))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        dark_neighborhood = cv2.dilate(dark, kernel, iterations=1)
        out = mask.copy()
        out[dark_neighborhood > 0] = 0
        return out

    def _cleanup(self, mask: np.ndarray) -> np.ndarray:
        work = mask.copy()
        if self.config.morph_open > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.morph_open, self.config.morph_open))
            work = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel)
        if self.config.morph_close > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.config.morph_close, self.config.morph_close))
            work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)
        return _filter_components(
            work,
            self.config.min_component_area,
            self.config.min_component_x_span,
            drop_border_components=self.config.drop_border_components,
            drop_grid_like_components=self.config.drop_grid_like_components,
            drop_axis_tick_components=self.config.drop_axis_tick_components,
            axis_tick_edge_margin_px=self.config.axis_tick_edge_margin_px,
            axis_tick_max_span_px=self.config.axis_tick_max_span_px,
            axis_tick_max_thickness_px=self.config.axis_tick_max_thickness_px,
        )


def _filter_components(
    mask: np.ndarray,
    min_area: int,
    min_x_span: int,
    drop_border_components: bool = True,
    drop_grid_like_components: bool = True,
    drop_axis_tick_components: bool = True,
    axis_tick_edge_margin_px: int = 4,
    axis_tick_max_span_px: int = 3,
    axis_tick_max_thickness_px: int = 4,
) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask)
    h, w = mask.shape[:2]
    for idx in range(1, num):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        x = int(stats[idx, cv2.CC_STAT_LEFT])
        y = int(stats[idx, cv2.CC_STAT_TOP])
        width = int(stats[idx, cv2.CC_STAT_WIDTH])
        height = int(stats[idx, cv2.CC_STAT_HEIGHT])
        border_like = (
            (y <= 0 or y + height >= h) and width >= 0.75 * w and height <= 6
        ) or (
            (x <= 0 or x + width >= w) and height >= 0.75 * h and width <= 6
        )
        grid_like = (width >= 0.35 * w and height <= 4) or (height >= 0.35 * h and width <= 4)
        near_left_or_right = x <= axis_tick_edge_margin_px or x + width >= w - axis_tick_edge_margin_px
        near_top_or_bottom = y <= axis_tick_edge_margin_px or y + height >= h - axis_tick_edge_margin_px
        horizontal_tick_like = near_left_or_right and width <= axis_tick_max_span_px and height <= axis_tick_max_thickness_px
        vertical_tick_like = near_top_or_bottom and height <= axis_tick_max_span_px and width <= axis_tick_max_thickness_px
        if drop_border_components and border_like:
            continue
        if drop_grid_like_components and grid_like:
            continue
        if drop_axis_tick_components and (horizontal_tick_like or vertical_tick_like):
            continue
        if area >= min_area and width >= min_x_span:
            out[labels == idx] = 255
    return out


def _mask_bbox(mask: np.ndarray) -> Optional[BoundingBox]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return BoundingBox(float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))


def _apply_exclusions(mask: np.ndarray, exclude_regions: Sequence[BoundingBox], offset) -> None:
    x_off, y_off = offset
    h, w = mask.shape[:2]
    for bbox in exclude_regions:
        x0 = max(0, int(bbox.x_min - x_off))
        y0 = max(0, int(bbox.y_min - y_off))
        x1 = min(w, int(bbox.x_max - x_off))
        y1 = min(h, int(bbox.y_max - y_off))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 0


def _crop(image_bgr: np.ndarray, region: Optional[BoundingBox], inset_px: int = 0):
    if region is None:
        return image_bgr.copy(), (0, 0)
    h, w = image_bgr.shape[:2]
    x0 = max(0, int(region.x_min))
    y0 = max(0, int(region.y_min))
    x1 = min(w, int(region.x_max))
    y1 = min(h, int(region.y_max))
    if inset_px > 0 and x1 - x0 > 2 * inset_px and y1 - y0 > 2 * inset_px:
        x0 += inset_px
        y0 += inset_px
        x1 -= inset_px
        y1 -= inset_px
    return image_bgr[y0:y1, x0:x1].copy(), (x0, y0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a curve mask from a color prototype JSON entry.")
    parser.add_argument("--img", required=True, help="Input image")
    parser.add_argument("--prototype_json", required=True, help="JSON file containing one CurvePrototype object")
    parser.add_argument("--out_mask", required=True, help="Output mask PNG")
    parser.add_argument("--out_json", default=None, help="Optional CurveMask JSON output")
    args = parser.parse_args()

    image = load_bgr(args.img)
    import json

    with open(args.prototype_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    prototype = CurvePrototype(**data)
    mask, info = CurveMaskExtractor().extract_mask(image, prototype)
    ensure_dir(os.path.dirname(os.path.abspath(args.out_mask)))
    cv2.imwrite(args.out_mask, mask)
    info.mask_path = args.out_mask
    if args.out_json:
        write_json(args.out_json, info)
    else:
        print(json.dumps(to_serializable(info), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
