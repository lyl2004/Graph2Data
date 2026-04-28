"""Map ordered curve pixel paths into data coordinates."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, List, Sequence

from .image_io import write_json
from .models import BoundingBox, CurvePath, DataPoint, DataRange, DataSeries, PlotArea, Point, to_serializable


def map_curve_path_to_data(curve_path: CurvePath, plot_area: PlotArea, data_range: DataRange) -> DataSeries:
    bbox = plot_area.bbox
    _validate_mapping_inputs(bbox, data_range)
    completed_indices = _completed_indices(curve_path.completed_ranges)
    confidences = curve_path.confidence_per_point or [1.0] * len(curve_path.pixel_points_ordered)
    warnings: List[str] = []
    points: List[DataPoint] = []

    outside_count = 0
    for idx, pixel in enumerate(curve_path.pixel_points_ordered):
        if not bbox.contains(pixel):
            outside_count += 1
        x_norm = (pixel.x - bbox.x_min) / bbox.width
        y_norm = (bbox.y_max - pixel.y) / bbox.height
        points.append(
            DataPoint(
                x=_map_value(x_norm, data_range.x_min, data_range.x_max, data_range.x_scale),
                y=_map_value(y_norm, data_range.y_min, data_range.y_max, data_range.y_scale),
                pixel_x=float(pixel.x),
                pixel_y=float(pixel.y),
                confidence=float(confidences[idx]) if idx < len(confidences) else 1.0,
                completed=idx in completed_indices,
            )
        )

    if outside_count:
        warnings.append(f"points_outside_plot_area={outside_count}")
    return _build_series(curve_path.curve_id, points, data_range, warnings)


def map_curve_paths_to_data(
    curve_paths: Sequence[CurvePath],
    plot_area: PlotArea,
    data_range: DataRange,
) -> List[DataSeries]:
    return [map_curve_path_to_data(path, plot_area, data_range) for path in curve_paths]


def write_data_series_csv(path: str, series_list: Sequence[DataSeries]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["curve_id", "point_index", "x", "y", "pixel_x", "pixel_y", "confidence", "completed"])
        for series in series_list:
            for idx, point in enumerate(series.points):
                writer.writerow(
                    [
                        series.curve_id,
                        idx,
                        f"{point.x:.12g}",
                        f"{point.y:.12g}",
                        f"{point.pixel_x:.3f}",
                        f"{point.pixel_y:.3f}",
                        f"{point.confidence:.6g}",
                        int(point.completed),
                    ]
                )


def _build_series(curve_id: str, points: Sequence[DataPoint], data_range: DataRange, warnings: List[str]) -> DataSeries:
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    return DataSeries(
        curve_id=curve_id,
        points=list(points),
        data_range=data_range,
        point_count=len(points),
        completed_point_count=sum(1 for point in points if point.completed),
        x_min=min(xs) if xs else None,
        x_max=max(xs) if xs else None,
        y_min=min(ys) if ys else None,
        y_max=max(ys) if ys else None,
        warnings=warnings,
    )


def _completed_indices(ranges: Iterable[Sequence[int]]) -> set[int]:
    indices = set()
    for start, end in ranges:
        for idx in range(int(start), int(end) + 1):
            indices.add(idx)
    return indices


def _map_value(norm: float, vmin: float, vmax: float, scale: str) -> float:
    if scale == "linear":
        return float(vmin + norm * (vmax - vmin))
    if scale == "log10":
        if vmin <= 0 or vmax <= 0:
            raise ValueError("log10 scale requires positive bounds")
        log_min = math.log10(vmin)
        log_max = math.log10(vmax)
        return float(10 ** (log_min + norm * (log_max - log_min)))
    if scale == "inverse":
        if vmin == 0 or vmax == 0:
            raise ValueError("inverse scale bounds cannot be zero")
        inv_min = 1.0 / vmin
        inv_max = 1.0 / vmax
        return float(1.0 / (inv_min + norm * (inv_max - inv_min)))
    raise ValueError(f"unknown scale: {scale}")


def _validate_mapping_inputs(bbox: BoundingBox, data_range: DataRange) -> None:
    if bbox.width <= 0 or bbox.height <= 0:
        raise ValueError("plot area must have positive width and height")
    if data_range.x_min == data_range.x_max:
        raise ValueError("x data range cannot be zero")
    if data_range.y_min == data_range.y_max:
        raise ValueError("y data range cannot be zero")
    _map_value(0.5, data_range.x_min, data_range.x_max, data_range.x_scale)
    _map_value(0.5, data_range.y_min, data_range.y_max, data_range.y_scale)


def _curve_path_from_dict(data: dict) -> CurvePath:
    return CurvePath(
        curve_id=data["curve_id"],
        pixel_points_ordered=[Point(float(p["x"]), float(p["y"])) for p in data.get("pixel_points_ordered", [])],
        completed_ranges=[tuple(r) for r in data.get("completed_ranges", [])],
        confidence_per_point=[float(v) for v in data.get("confidence_per_point", [])],
        endpoints=[Point(float(p["x"]), float(p["y"])) for p in data.get("endpoints", [])],
        junctions=[Point(float(p["x"]), float(p["y"])) for p in data.get("junctions", [])],
        component_count=int(data.get("component_count", 0)),
        observed_pixel_count=int(data.get("observed_pixel_count", 0)),
        completed_pixel_count=int(data.get("completed_pixel_count", 0)),
        path_length_px=float(data.get("path_length_px", 0.0)),
        confidence=float(data.get("confidence", 0.0)),
        warnings=list(data.get("warnings", [])),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Map a CurvePath JSON file to data coordinates.")
    parser.add_argument("--path_json", required=True, help="CurvePath JSON input")
    parser.add_argument("--plot_x_min", type=float, required=True)
    parser.add_argument("--plot_y_min", type=float, required=True)
    parser.add_argument("--plot_x_max", type=float, required=True)
    parser.add_argument("--plot_y_max", type=float, required=True)
    parser.add_argument("--x_min", type=float, required=True)
    parser.add_argument("--x_max", type=float, required=True)
    parser.add_argument("--y_min", type=float, required=True)
    parser.add_argument("--y_max", type=float, required=True)
    parser.add_argument("--x_scale", choices=["linear", "log10", "inverse"], default="linear")
    parser.add_argument("--y_scale", choices=["linear", "log10", "inverse"], default="linear")
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    with open(args.path_json, "r", encoding="utf-8") as f:
        curve_path = _curve_path_from_dict(json.load(f))
    plot_area = PlotArea(BoundingBox(args.plot_x_min, args.plot_y_min, args.plot_x_max, args.plot_y_max))
    data_range = DataRange(args.x_min, args.x_max, args.y_min, args.y_max, args.x_scale, args.y_scale)
    series = map_curve_path_to_data(curve_path, plot_area, data_range)

    if args.out_csv:
        write_data_series_csv(args.out_csv, [series])
    if args.out_json:
        write_json(args.out_json, series)
    if not args.out_csv and not args.out_json:
        print(json.dumps(to_serializable(series), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
