"""Shared structured result models for the extraction pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Point:
    x: float
    y: float


@dataclass
class BoundingBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def contains(self, point: Point) -> bool:
        return self.x_min <= point.x <= self.x_max and self.y_min <= point.y <= self.y_max


@dataclass
class AxisLine:
    start: Point
    end: Point
    orientation: str
    confidence: float = 0.0


@dataclass
class PlotArea:
    bbox: BoundingBox
    confidence: float = 0.0
    source: str = "axis_detection"


@dataclass
class AxisDetection:
    success: bool
    image_size: Tuple[int, int]
    origin: Optional[Point] = None
    x_axis: Optional[AxisLine] = None
    y_axis: Optional[AxisLine] = None
    plot_area: Optional[PlotArea] = None
    confidence: float = 0.0
    candidates: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class OCRTextBox:
    text: str
    confidence: float
    polygon: List[Point]
    bbox: BoundingBox
    center: Point
    region_id: Optional[str] = None


@dataclass
class LayoutRegion:
    region_id: str
    label: str
    bbox: BoundingBox
    text_indices: List[int] = field(default_factory=list)


@dataclass
class LayoutResult:
    plot_area: PlotArea
    regions: List[LayoutRegion]


@dataclass
class LegendDetection:
    bbox: BoundingBox
    confidence: float = 0.0
    text_indices: List[int] = field(default_factory=list)
    source: str = "heuristic"
    warnings: List[str] = field(default_factory=list)


@dataclass
class CurvePrototype:
    curve_id: str
    rgb: Tuple[int, int, int]
    lab: Tuple[float, float, float]
    area: int
    ratio: float
    label: Optional[str] = None
    line_style: str = "unknown"
    marker_style: str = "unknown"
    confidence: float = 0.0
    source: str = "direct"


@dataclass
class CurveMask:
    curve_id: str
    mask_path: Optional[str] = None
    pixel_count: int = 0
    bbox: Optional[BoundingBox] = None
    confidence: float = 0.0
    source: str = "color_threshold"
    warnings: List[str] = field(default_factory=list)


@dataclass
class CurvePath:
    curve_id: str
    pixel_points_ordered: List[Point]
    completed_ranges: List[Tuple[int, int]] = field(default_factory=list)
    confidence_per_point: List[float] = field(default_factory=list)
    endpoints: List[Point] = field(default_factory=list)
    junctions: List[Point] = field(default_factory=list)
    component_count: int = 0
    observed_pixel_count: int = 0
    completed_pixel_count: int = 0
    path_length_px: float = 0.0
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    image_path: str
    image_size: Tuple[int, int]
    axis: AxisDetection
    ocr: List[OCRTextBox] = field(default_factory=list)
    layout: Optional[LayoutResult] = None
    legends: List[LegendDetection] = field(default_factory=list)
    curves: List[CurvePrototype] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def to_serializable(value: Any) -> Any:
    """Convert dataclasses and common numeric containers to JSON-safe values."""
    if is_dataclass(value):
        return {k: to_serializable(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(v) for v in value]
    if hasattr(value, "item"):
        return value.item()
    return value
