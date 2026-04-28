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
class LegendItem:
    item_id: str
    legend_index: int
    bbox: BoundingBox
    sample_bbox: Optional[BoundingBox] = None
    text_bbox: Optional[BoundingBox] = None
    rgb: Optional[Tuple[int, int, int]] = None
    lab: Optional[Tuple[float, float, float]] = None
    foreground_pixel_count: int = 0
    label: Optional[str] = None
    line_style: str = "unknown"
    marker_style: str = "unknown"
    confidence: float = 0.0
    source: str = "image_heuristic"
    warnings: List[str] = field(default_factory=list)


@dataclass
class CurveVisualPrototype:
    prototype_id: str
    legend_item_id: Optional[str] = None
    rgb: Optional[Tuple[int, int, int]] = None
    lab: Optional[Tuple[float, float, float]] = None
    sample_bbox: Optional[BoundingBox] = None
    label: Optional[str] = None
    line_style: str = "unknown"
    marker_style: str = "unknown"
    confidence: float = 0.0
    source: str = "legend_item"


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
    label: Optional[str] = None
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
class LineComponentClassification:
    curve_id: str
    component_id: str
    class_label: str
    bbox: BoundingBox
    pixel_count: int
    width: float
    height: float
    max_span: float
    aspect_ratio: float
    path_points: List[Point] = field(default_factory=list)
    endpoint_count: int = 0
    junction_count: int = 0
    path_length_px: float = 0.0
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class MarkerCandidate:
    curve_id: str
    marker_id: str
    center: Point
    bbox: BoundingBox
    pixel_count: int
    width: float
    height: float
    area_ratio: float
    fill_ratio: float
    aspect_ratio: float
    circularity: float = 0.0
    shape: str = "unknown"
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class MarkerCurveInstance:
    instance_id: str
    source_curve_id: str
    marker_ids: List[str]
    points: List[Point]
    marker_count: int = 0
    center_y: float = 0.0
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    grouping_method: str = "x_rank"
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class LineStyleCurveInstance:
    instance_id: str
    source_curve_id: str
    component_ids: List[str]
    points: List[Point] = field(default_factory=list)
    component_count: int = 0
    center_y: float = 0.0
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    estimated_line_style: str = "unknown"
    grouping_method: str = "component_y_rank"
    confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)


@dataclass
class PrototypeBinding:
    binding_id: str
    prototype_id: str
    target_curve_id: str
    target_type: str = "curve"
    legend_item_id: Optional[str] = None
    score: float = 0.0
    color_similarity: Optional[float] = None
    marker_similarity: Optional[float] = None
    line_similarity: Optional[float] = None
    confidence: float = 0.0
    source: str = "diagnostic_score"
    warnings: List[str] = field(default_factory=list)


@dataclass
class DataRange:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    x_scale: str = "linear"
    y_scale: str = "linear"


@dataclass
class DataPoint:
    x: float
    y: float
    pixel_x: float
    pixel_y: float
    confidence: float = 1.0
    completed: bool = False


@dataclass
class DataSeries:
    curve_id: str
    points: List[DataPoint]
    data_range: DataRange
    label: Optional[str] = None
    point_count: int = 0
    completed_point_count: int = 0
    x_min: Optional[float] = None
    x_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    image_path: str
    image_size: Tuple[int, int]
    axis: AxisDetection
    ocr: List[OCRTextBox] = field(default_factory=list)
    layout: Optional[LayoutResult] = None
    legends: List[LegendDetection] = field(default_factory=list)
    legend_items: List[LegendItem] = field(default_factory=list)
    curve_visual_prototypes: List[CurveVisualPrototype] = field(default_factory=list)
    curves: List[CurvePrototype] = field(default_factory=list)
    curve_masks: List[CurveMask] = field(default_factory=list)
    curve_paths: List[CurvePath] = field(default_factory=list)
    line_components: List[LineComponentClassification] = field(default_factory=list)
    marker_candidates: List[MarkerCandidate] = field(default_factory=list)
    marker_curve_instances: List[MarkerCurveInstance] = field(default_factory=list)
    line_style_curve_instances: List[LineStyleCurveInstance] = field(default_factory=list)
    prototype_bindings: List[PrototypeBinding] = field(default_factory=list)
    prototype_bound_paths: List[CurvePath] = field(default_factory=list)
    data_series: List[DataSeries] = field(default_factory=list)
    prototype_bound_data_series: List[DataSeries] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
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
