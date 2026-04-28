"""Synthetic benchmark generation with image, masks, and ground truth."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from .image_io import ensure_dir, write_json


CURVE_STYLES: Sequence[Tuple[str, str]] = (
    ("#1f77b4", "-"),
    ("#ff7f0e", "--"),
    ("#2ca02c", "-."),
    ("#d62728", ":"),
    ("#9467bd", "-"),
    ("#8c564b", "--"),
)

ACHROMATIC_STYLES: Sequence[Tuple[str, str]] = (
    ("#000000", "-"),
    ("#777777", "--"),
    ("#1f77b4", "-."),
    ("#d62728", ":"),
    ("#2ca02c", "-"),
    ("#ff7f0e", "--"),
)

MARKER_STYLES: Sequence[str] = ("o", "s", "^", "D", "x", "+", "v", "P", "*", "h")


@dataclass
class SyntheticConfig:
    width_px: int = 1200
    height_px: int = 750
    dpi: int = 150
    axes_position: Tuple[float, float, float, float] = (0.12, 0.14, 0.66, 0.74)
    seed: int = 42
    n_curves: int = 6
    n_points: int = 800
    x_min: float = 0.0
    x_max: float = 30.0
    y_min: float = -1.0
    y_max: float = 5.0
    palette: str = "basic"
    legend_inside: bool = False
    legend_loc: str = "upper left"
    local_occlusion: bool = False
    occlusion_centers: Tuple[float, ...] = (10.0, 18.2, 24.4)
    occlusion_width: float = 0.55
    crossing_curves: bool = False
    marker_curves: bool = False
    same_color_marker_curves: bool = False
    same_gray_linestyle_curves: bool = False
    dense_legend_curves: bool = False


def generate_curve_family(config: SyntheticConfig) -> Tuple[np.ndarray, List[np.ndarray], List[Dict]]:
    """Generate overlapping but known smooth curves."""
    rng = np.random.default_rng(config.seed)
    x = np.linspace(config.x_min, config.x_max, config.n_points)
    y_list: List[np.ndarray] = []
    metadata: List[Dict] = []

    base = 0.4 * np.sin(0.55 * x) + 0.07 * x
    for idx in range(config.n_curves):
        color, linestyle, marker, marker_every = _visual_style_for_curve(idx, config)
        if config.crossing_curves:
            slopes = [0.58, -0.50, 0.34, -0.30, 0.18, -0.16]
            slope = slopes[idx % len(slopes)]
            phase = idx * 0.9
            center_offset = (idx - (config.n_curves - 1) / 2.0) * 0.045
            y = (
                2.05
                + center_offset
                + slope * (x - (config.x_min + config.x_max) / 2.0) / 6.0
                + 0.08 * np.sin(0.7 * x + phase)
            )
        else:
            phase = idx * 0.7
            amp = 0.18 + 0.04 * idx
            offset = idx * 0.38
            local_overlap = np.exp(-((x - 9.0) ** 2) / 18.0) * (0.25 - 0.04 * idx)
            wiggle = amp * np.sin(0.9 * x + phase) + 0.06 * np.cos(1.7 * x + phase / 2)
            y = base + wiggle + offset + local_overlap

            # Add one deterministic local occlusion-like bend region so path metrics are meaningful later.
            bend_center = 18.0 + rng.normal(0, 0.4)
            y += 0.12 * np.exp(-((x - bend_center) ** 2) / 2.5) * np.sin(2.5 + idx)
        y_list.append(y)
        metadata.append(
            {
                "curve_id": f"curve_{idx:02d}",
                "color": color,
                "linestyle": linestyle,
                "marker": marker,
                "marker_every": marker_every,
                "marker_size": 4.0 if marker else 0.0,
                "label": f"Curve {idx + 1}",
            }
        )

    return x, y_list, metadata


def generate_benchmark(output_dir: str, name: str = "basic_curves", config: SyntheticConfig | None = None) -> Dict:
    config = config or SyntheticConfig()
    config = _normalize_config(config)
    out_dir = ensure_dir(os.path.join(output_dir, name))
    mask_dir = ensure_dir(os.path.join(out_dir, "masks"))

    x, y_list, curve_meta = generate_curve_family(config)
    image_path = os.path.join(out_dir, "image.png")
    data_path = os.path.join(out_dir, "truth_data.csv")
    axes_path = os.path.join(out_dir, "truth_axes.json")
    curves_path = os.path.join(out_dir, "truth_curves.json")

    _render_image(image_path, x, y_list, curve_meta, config)
    _write_truth_data(data_path, x, y_list, curve_meta)
    mask_paths = _render_masks(mask_dir, x, y_list, curve_meta, config)

    image_size = (config.width_px, config.height_px)
    plot_bbox = _plot_bbox_pixels(config)
    truth_axes = {
        "image_size": {"width": config.width_px, "height": config.height_px},
        "plot_area": plot_bbox,
        "origin": {"x": plot_bbox["x_min"], "y": plot_bbox["y_max"]},
        "x_axis": {
            "start": {"x": plot_bbox["x_min"], "y": plot_bbox["y_max"]},
            "end": {"x": plot_bbox["x_max"], "y": plot_bbox["y_max"]},
        },
        "y_axis": {
            "start": {"x": plot_bbox["x_min"], "y": plot_bbox["y_max"]},
            "end": {"x": plot_bbox["x_min"], "y": plot_bbox["y_min"]},
        },
        "data_range": {
            "x_min": config.x_min,
            "x_max": config.x_max,
            "y_min": config.y_min,
            "y_max": config.y_max,
            "x_scale": "linear",
            "y_scale": "linear",
        },
    }
    write_json(axes_path, truth_axes)

    truth_curves = []
    for meta, mask_path in zip(curve_meta, mask_paths):
        truth_curves.append({**meta, "mask_path": os.path.relpath(mask_path, out_dir)})
    write_json(curves_path, {"curves": truth_curves})

    manifest = {
        "name": name,
        "image_path": image_path,
        "truth_axes_path": axes_path,
        "truth_curves_path": curves_path,
        "truth_data_path": data_path,
        "mask_dir": mask_dir,
        "image_size": image_size,
        "curve_count": len(curve_meta),
        "synthetic_config": asdict(config),
    }
    write_json(os.path.join(out_dir, "manifest.json"), manifest)
    return manifest


def _render_image(path: str, x: np.ndarray, y_list: List[np.ndarray], metadata: List[Dict], config: SyntheticConfig) -> None:
    fig, ax = _new_figure(config)
    for y, meta in zip(y_list, metadata):
        rendered_y = _apply_local_occlusion(x, y, config)
        ax.plot(x, rendered_y, **_plot_kwargs(meta, mask=False))
    ax.set_xlim(config.x_min, config.x_max)
    ax.set_ylim(config.y_min, config.y_max)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Response")
    ax.set_title(f"Synthetic Benchmark: {_case_title(config)}")
    ax.grid(True, linestyle="--", linewidth=0.6, color="#d0d0d0", alpha=0.8)
    if config.legend_inside:
        ax.legend(loc=config.legend_loc, fontsize=_legend_fontsize(config), frameon=True)
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=_legend_fontsize(config), frameon=True)
    fig.savefig(path, dpi=config.dpi)
    plt.close(fig)


def _render_masks(
    mask_dir: str, x: np.ndarray, y_list: List[np.ndarray], metadata: List[Dict], config: SyntheticConfig
) -> List[str]:
    paths: List[str] = []
    for y, meta in zip(y_list, metadata):
        fig, ax = _new_figure(config, facecolor="black")
        rendered_y = _apply_local_occlusion(x, y, config)
        mask_meta = {**meta, "color": "white"}
        ax.plot(x, rendered_y, **_plot_kwargs(mask_meta, mask=True))
        ax.set_xlim(config.x_min, config.x_max)
        ax.set_ylim(config.y_min, config.y_max)
        ax.set_axis_off()
        for spine in ax.spines.values():
            spine.set_visible(False)
        path = os.path.join(mask_dir, f"{meta['curve_id']}.png")
        fig.savefig(path, dpi=config.dpi, facecolor="black")
        plt.close(fig)
        paths.append(path)
    return paths


def _write_truth_data(path: str, x: np.ndarray, y_list: List[np.ndarray], metadata: List[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["curve_id", "x", "y", "color", "linestyle", "marker", "marker_every", "label"])
        for y, meta in zip(y_list, metadata):
            for x_val, y_val in zip(x, y):
                writer.writerow(
                    [
                        meta["curve_id"],
                        f"{float(x_val):.10g}",
                        f"{float(y_val):.10g}",
                        meta["color"],
                        meta["linestyle"],
                        meta.get("marker") or "",
                        meta.get("marker_every") or "",
                        meta["label"],
                    ]
                )


def _normalize_config(config: SyntheticConfig) -> SyntheticConfig:
    if config.dense_legend_curves:
        return replace(config, n_curves=max(config.n_curves, 10), legend_inside=True)
    return config


def _visual_style_for_curve(idx: int, config: SyntheticConfig) -> Tuple[str, str, str | None, int | None]:
    styles = ACHROMATIC_STYLES if config.palette == "achromatic" else CURVE_STYLES
    color, linestyle = styles[idx % len(styles)]
    marker = None
    marker_every = None

    if config.same_gray_linestyle_curves:
        gray_values = ("#222222", "#555555", "#777777", "#999999", "#bbbbbb", "#444444")
        linestyles = ("-", "--", "-.", ":", (0, (5, 1)), (0, (1, 1)))
        color = gray_values[idx % len(gray_values)]
        linestyle = linestyles[idx % len(linestyles)]
    if config.marker_curves:
        marker = MARKER_STYLES[idx % len(MARKER_STYLES)]
        linestyle = "None"
        marker_every = 35
    if config.same_color_marker_curves:
        color = "#555555"
        linestyle = "-"
        marker = MARKER_STYLES[idx % len(MARKER_STYLES)]
        marker_every = 45
    if config.dense_legend_curves:
        marker = marker or MARKER_STYLES[idx % len(MARKER_STYLES)]
        marker_every = marker_every or 70
    return color, linestyle, marker, marker_every


def _plot_kwargs(meta: Dict, mask: bool) -> Dict:
    kwargs = {
        "color": meta["color"],
        "linestyle": meta["linestyle"],
        "linewidth": 2.2 if mask else 1.7,
        "label": meta["label"],
    }
    marker = meta.get("marker")
    if marker:
        kwargs.update(
            {
                "marker": marker,
                "markersize": 6.0 if mask else float(meta.get("marker_size", 4.0)),
                "markevery": int(meta.get("marker_every") or 1),
                "markerfacecolor": meta["color"],
                "markeredgecolor": meta["color"],
                "markeredgewidth": 1.4 if mask else 1.0,
            }
        )
    return kwargs


def _case_title(config: SyntheticConfig) -> str:
    if config.marker_curves:
        return "Marker Curves"
    if config.same_color_marker_curves:
        return "Same Color Marker Curves"
    if config.same_gray_linestyle_curves:
        return "Same Gray Linestyle Curves"
    if config.dense_legend_curves:
        return "Dense Legend Curves"
    if config.crossing_curves:
        return "Crossing Curves"
    if config.local_occlusion:
        return "Local Occlusion Curves"
    if config.palette == "achromatic":
        return "Achromatic Curves"
    return "Basic Curves"


def _legend_fontsize(config: SyntheticConfig) -> int:
    return 7 if config.dense_legend_curves or config.n_curves > 8 else 8


def _apply_local_occlusion(x: np.ndarray, y: np.ndarray, config: SyntheticConfig) -> np.ndarray:
    if not config.local_occlusion:
        return y
    rendered = y.copy()
    for center in config.occlusion_centers:
        rendered[np.abs(x - center) <= config.occlusion_width / 2.0] = np.nan
    return rendered


def _new_figure(config: SyntheticConfig, facecolor: str = "white"):
    width_in = config.width_px / config.dpi
    height_in = config.height_px / config.dpi
    fig = plt.figure(figsize=(width_in, height_in), dpi=config.dpi, facecolor=facecolor)
    ax = fig.add_axes(config.axes_position, facecolor=facecolor)
    return fig, ax


def _plot_bbox_pixels(config: SyntheticConfig) -> Dict[str, float]:
    left, bottom, width, height = config.axes_position
    x_min = left * config.width_px
    x_max = (left + width) * config.width_px
    y_min = (1.0 - bottom - height) * config.height_px
    y_max = (1.0 - bottom) * config.height_px
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Graph2Data benchmark assets.")
    parser.add_argument("--out", default="benchmarks/synthetic", help="Output directory")
    parser.add_argument("--name", default="basic_curves", help="Benchmark case name")
    parser.add_argument("--curves", type=int, default=6, help="Number of curves")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--palette", choices=["basic", "achromatic"], default="basic", help="Synthetic curve palette")
    parser.add_argument("--legend_inside", action="store_true", help="Place legend inside the plot area")
    parser.add_argument("--legend_loc", default="upper left", help="Matplotlib legend location when --legend_inside is set")
    parser.add_argument("--local_occlusion", action="store_true", help="Remove short curve spans from rendered image/masks while keeping full truth data")
    parser.add_argument("--crossing_curves", action="store_true", help="Generate curves that lightly cross in the plot area")
    parser.add_argument("--marker_curves", action="store_true", help="Generate pure marker curves without connecting lines")
    parser.add_argument("--same_color_marker_curves", action="store_true", help="Generate curves with the same color but different markers")
    parser.add_argument("--same_gray_linestyle_curves", action="store_true", help="Generate gray curves distinguished mainly by line style")
    parser.add_argument("--dense_legend_curves", action="store_true", help="Generate a dense in-plot legend with at least ten curves")
    args = parser.parse_args()

    manifest = generate_benchmark(
        output_dir=args.out,
        name=args.name,
        config=SyntheticConfig(
            seed=args.seed,
            n_curves=args.curves,
            palette=args.palette,
            legend_inside=args.legend_inside,
            legend_loc=args.legend_loc,
            local_occlusion=args.local_occlusion,
            crossing_curves=args.crossing_curves,
            marker_curves=args.marker_curves,
            same_color_marker_curves=args.same_color_marker_curves,
            same_gray_linestyle_curves=args.same_gray_linestyle_curves,
            dense_legend_curves=args.dense_legend_curves,
        ),
    )
    print(f"Generated benchmark: {manifest['image_path']}")


if __name__ == "__main__":
    main()
