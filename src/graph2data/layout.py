"""Plot-relative layout regions and OCR assignment."""

from __future__ import annotations

from typing import List

from .models import BoundingBox, LayoutRegion, LayoutResult, OCRTextBox, PlotArea, Point


REGION_LABELS = {
    "top_left": "top-left margin",
    "top": "title or top legend",
    "top_right": "top-right margin",
    "left": "y ticks or y label",
    "plot": "plot area",
    "right": "right legend or right axis",
    "bottom_left": "bottom-left margin",
    "bottom": "x ticks or x label",
    "bottom_right": "bottom-right margin",
}


def build_nine_grid(image_size, plot_area: PlotArea) -> LayoutResult:
    width, height = image_size
    b = plot_area.bbox
    xs = [0.0, b.x_min, b.x_max, float(width)]
    ys = [0.0, b.y_min, b.y_max, float(height)]
    ids = [
        ("top_left", 0, 0),
        ("top", 1, 0),
        ("top_right", 2, 0),
        ("left", 0, 1),
        ("plot", 1, 1),
        ("right", 2, 1),
        ("bottom_left", 0, 2),
        ("bottom", 1, 2),
        ("bottom_right", 2, 2),
    ]
    regions: List[LayoutRegion] = []
    for region_id, col, row in ids:
        bbox = BoundingBox(xs[col], ys[row], xs[col + 1], ys[row + 1])
        regions.append(LayoutRegion(region_id=region_id, label=REGION_LABELS[region_id], bbox=bbox))
    return LayoutResult(plot_area=plot_area, regions=regions)


def assign_text_to_regions(layout: LayoutResult, texts: List[OCRTextBox]) -> LayoutResult:
    for region in layout.regions:
        region.text_indices.clear()

    for idx, text in enumerate(texts):
        point = text.center
        matched = _find_region(layout.regions, point)
        if matched is not None:
            matched.text_indices.append(idx)
            text.region_id = matched.region_id
    return layout


def _find_region(regions: List[LayoutRegion], point: Point):
    for region in regions:
        if region.bbox.contains(point):
            return region
    return None
