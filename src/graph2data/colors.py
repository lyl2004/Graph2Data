"""Curve color prototype extraction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np

from .models import BoundingBox, CurvePrototype


@dataclass
class ColorExtractorConfig:
    step: int = 1
    diff: int = 15
    min_area: int = 2
    min_l: int = 10
    max_l: int = 240
    min_chroma: int = 10
    merge_diff_l: int = 80
    merge_diff_ab: int = 20
    min_ratio: float = 0.001
    max_ratio: float = 0.90
    inset_px: int = 3
    include_achromatic: bool = True
    achromatic_min_area: int = 8
    dark_max_l: int = 95
    gray_min_l: int = 95
    gray_max_l: int = 195
    gray_max_chroma: int = 12
    fine_gray_merge_diff_l: int = 18


class CurveColorExtractor:
    """Extract curve color prototypes from a plot image or region."""

    def __init__(self, config: Optional[ColorExtractorConfig] = None):
        self.config = config or ColorExtractorConfig()

    def extract(
        self,
        image_bgr: np.ndarray,
        region: Optional[BoundingBox] = None,
        exclude_regions: Optional[Sequence[BoundingBox]] = None,
    ) -> List[CurvePrototype]:
        crop, offset = self._crop(image_bgr, region)
        if crop.size == 0:
            return []

        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
        lab_float = lab.astype(np.float32)
        h, w = crop.shape[:2]
        total_pixels = h * w
        cfg = self.config

        l_channel = lab[:, :, 0]
        a_centered = lab_float[:, :, 1] - 128
        b_centered = lab_float[:, :, 2] - 128
        chroma = np.sqrt(a_centered**2 + b_centered**2)
        ignore = (l_channel < cfg.min_l) | (l_channel > cfg.max_l) | (chroma < cfg.min_chroma)
        if exclude_regions:
            self._apply_exclusions(ignore, exclude_regions, offset)

        visited = np.zeros((h, w), dtype=bool)
        visited[ignore] = True
        mask_template = np.zeros((h + 2, w + 2), np.uint8)
        raw_regions = []

        for y in range(0, h, cfg.step):
            for x in range(0, w, cfg.step):
                if visited[y, x]:
                    continue
                mask = mask_template.copy()
                flags = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
                tolerance = (cfg.diff, cfg.diff, cfg.diff)
                area, _, _, _ = cv2.floodFill(lab, mask, (x, y), (0, 0, 0), tolerance, tolerance, flags)
                region_mask = mask[1 : h + 1, 1 : w + 1].astype(bool)
                visited[region_mask] = True
                if area < cfg.min_area:
                    continue
                mean_lab = cv2.mean(lab, mask=region_mask.astype(np.uint8))[:3]
                c_val = math.sqrt((mean_lab[1] - 128) ** 2 + (mean_lab[2] - 128) ** 2)
                if cfg.min_l <= mean_lab[0] <= cfg.max_l and c_val >= cfg.min_chroma:
                    raw_regions.append({"lab": np.array(mean_lab, dtype=np.float32), "area": int(area)})

        if cfg.include_achromatic:
            raw_regions.extend(self._extract_achromatic_regions(lab, chroma))

        merged = self._merge_regions(raw_regions)
        return self._to_prototypes(merged, total_pixels, offset)

    def _crop(self, image_bgr: np.ndarray, region: Optional[BoundingBox]):
        if region is None:
            return image_bgr.copy(), (0, 0)
        h, w = image_bgr.shape[:2]
        x0 = max(0, int(region.x_min))
        y0 = max(0, int(region.y_min))
        x1 = min(w, int(region.x_max))
        y1 = min(h, int(region.y_max))
        if self.config.inset_px > 0 and x1 - x0 > 2 * self.config.inset_px and y1 - y0 > 2 * self.config.inset_px:
            x0 += self.config.inset_px
            y0 += self.config.inset_px
            x1 -= self.config.inset_px
            y1 -= self.config.inset_px
        return image_bgr[y0:y1, x0:x1].copy(), (x0, y0)

    def _extract_achromatic_regions(self, lab: np.ndarray, chroma: np.ndarray):
        cfg = self.config
        l_channel = lab[:, :, 0]
        dark = l_channel <= cfg.dark_max_l
        gray = (chroma <= cfg.gray_max_chroma) & (l_channel >= cfg.gray_min_l) & (l_channel <= cfg.gray_max_l)
        regions = []
        for mask in (dark, gray):
            regions.extend(self._regions_from_mask(mask.astype(np.uint8), lab))
        return regions

    def _apply_exclusions(self, ignore: np.ndarray, exclude_regions: Sequence[BoundingBox], offset) -> None:
        x_off, y_off = offset
        h, w = ignore.shape[:2]
        for bbox in exclude_regions:
            x0 = max(0, int(bbox.x_min - x_off))
            y0 = max(0, int(bbox.y_min - y_off))
            x1 = min(w, int(bbox.x_max - x_off))
            y1 = min(h, int(bbox.y_max - y_off))
            if x1 > x0 and y1 > y0:
                ignore[y0:y1, x0:x1] = True

    def _regions_from_mask(self, mask: np.ndarray, lab: np.ndarray):
        cfg = self.config
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        regions = []
        for idx in range(1, num):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < cfg.achromatic_min_area:
                continue
            component_mask = (labels == idx).astype(np.uint8)
            mean_lab = cv2.mean(lab, mask=component_mask)[:3]
            regions.append({"lab": np.array(mean_lab, dtype=np.float32), "area": area})
        return regions

    def _merge_regions(self, raw_regions):
        cfg = self.config
        raw_regions.sort(key=lambda x: x["area"], reverse=True)
        palette = []
        for region in raw_regions:
            curr_lab = region["lab"]
            curr_area = region["area"]
            matched = False
            for p in palette:
                delta_l = abs(curr_lab[0] - p["lab"][0])
                delta_ab = math.hypot(curr_lab[1] - p["lab"][1], curr_lab[2] - p["lab"][2])
                if delta_l < cfg.merge_diff_l and delta_ab < cfg.merge_diff_ab:
                    new_area = p["area"] + curr_area
                    p["lab"] = (p["lab"] * p["area"] + curr_lab * curr_area) / new_area
                    p["area"] = new_area
                    p["count"] += 1
                    matched = True
                    break
            if not matched:
                palette.append({"lab": curr_lab, "area": curr_area, "count": 1})
        return palette

    def extract_with_gray_legend_guidance(
        self,
        image_bgr: np.ndarray,
        region: Optional[BoundingBox],
        legend_l_values: Sequence[float],
        target_count: int,
        exclude_regions: Optional[Sequence[BoundingBox]] = None,
    ) -> List[CurvePrototype]:
        prototypes = self.extract(image_bgr, region, exclude_regions=exclude_regions)
        if len(prototypes) >= target_count:
            return prototypes
        legend_levels = sorted(float(value) for value in legend_l_values if value is not None)
        if len(legend_levels) < target_count:
            return prototypes
        if max(legend_levels) - min(legend_levels) < 18.0:
            return prototypes
        fine_cfg = ColorExtractorConfig(
            **{
                **self.config.__dict__,
                "merge_diff_l": min(self.config.merge_diff_l, self.config.fine_gray_merge_diff_l),
                "gray_max_l": max(self.config.gray_max_l, 240),
                "max_l": max(self.config.max_l, 255),
            }
        )
        guided = CurveColorExtractor(fine_cfg).extract(image_bgr, region, exclude_regions=exclude_regions)
        if len(guided) > len(prototypes):
            for prototype in guided:
                prototype.source = "guided_gray"
            return guided
        return prototypes

    def _to_prototypes(self, palette, total_pixels: int, offset) -> List[CurvePrototype]:
        cfg = self.config
        prototypes: List[CurvePrototype] = []
        for p in palette:
            ratio = p["area"] / max(total_pixels, 1)
            if ratio < cfg.min_ratio or ratio > cfg.max_ratio:
                continue
            lab_pix = np.uint8([[p["lab"]]])
            bgr = cv2.cvtColor(lab_pix, cv2.COLOR_Lab2BGR)[0][0]
            rgb = (int(bgr[2]), int(bgr[1]), int(bgr[0]))
            lab_tuple = (float(p["lab"][0]), float(p["lab"][1]), float(p["lab"][2]))
            confidence = float(max(0.0, min(1.0, ratio / max(cfg.min_ratio * 10, 1e-9))))
            prototypes.append(
                CurvePrototype(
                    curve_id=f"curve_color_{len(prototypes)}",
                    rgb=rgb,
                    lab=lab_tuple,
                    area=int(p["area"]),
                    ratio=float(ratio),
                    confidence=confidence,
                    source="direct",
                )
            )
        prototypes.sort(key=lambda c: c.area, reverse=True)
        return prototypes
