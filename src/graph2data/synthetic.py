"""Synthetic benchmark generation with image, masks, and ground truth."""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
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


@dataclass
class SyntheticConfig:
    width_px: int = 1200
    height_px: int = 750
    dpi: int = 150
    axes_position: Tuple[float, float, float, float] = (0.12, 0.14, 0.76, 0.74)
    seed: int = 42
    n_curves: int = 6
    n_points: int = 800
    x_min: float = 0.0
    x_max: float = 30.0
    y_min: float = -1.0
    y_max: float = 5.0


def generate_curve_family(config: SyntheticConfig) -> Tuple[np.ndarray, List[np.ndarray], List[Dict]]:
    """Generate overlapping but known smooth curves."""
    rng = np.random.default_rng(config.seed)
    x = np.linspace(config.x_min, config.x_max, config.n_points)
    y_list: List[np.ndarray] = []
    metadata: List[Dict] = []

    base = 0.4 * np.sin(0.55 * x) + 0.07 * x
    for idx in range(config.n_curves):
        color, linestyle = CURVE_STYLES[idx % len(CURVE_STYLES)]
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
                "label": f"Curve {idx + 1}",
            }
        )

    return x, y_list, metadata


def generate_benchmark(output_dir: str, name: str = "basic_curves", config: SyntheticConfig | None = None) -> Dict:
    config = config or SyntheticConfig()
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
    }
    write_json(os.path.join(out_dir, "manifest.json"), manifest)
    return manifest


def _render_image(path: str, x: np.ndarray, y_list: List[np.ndarray], metadata: List[Dict], config: SyntheticConfig) -> None:
    fig, ax = _new_figure(config)
    for y, meta in zip(y_list, metadata):
        ax.plot(x, y, color=meta["color"], linestyle=meta["linestyle"], linewidth=1.7, label=meta["label"])
    ax.set_xlim(config.x_min, config.x_max)
    ax.set_ylim(config.y_min, config.y_max)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Response")
    ax.set_title("Synthetic Benchmark: Basic Curves")
    ax.grid(True, linestyle="--", linewidth=0.6, color="#d0d0d0", alpha=0.8)
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    fig.savefig(path, dpi=config.dpi)
    plt.close(fig)


def _render_masks(
    mask_dir: str, x: np.ndarray, y_list: List[np.ndarray], metadata: List[Dict], config: SyntheticConfig
) -> List[str]:
    paths: List[str] = []
    for y, meta in zip(y_list, metadata):
        fig, ax = _new_figure(config, facecolor="black")
        ax.plot(x, y, color="white", linestyle=meta["linestyle"], linewidth=2.2)
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
        writer.writerow(["curve_id", "x", "y", "color", "linestyle", "label"])
        for y, meta in zip(y_list, metadata):
            for x_val, y_val in zip(x, y):
                writer.writerow(
                    [
                        meta["curve_id"],
                        f"{float(x_val):.10g}",
                        f"{float(y_val):.10g}",
                        meta["color"],
                        meta["linestyle"],
                        meta["label"],
                    ]
                )


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
    args = parser.parse_args()

    manifest = generate_benchmark(
        output_dir=args.out,
        name=args.name,
        config=SyntheticConfig(seed=args.seed, n_curves=args.curves),
    )
    print(f"Generated benchmark: {manifest['image_path']}")


if __name__ == "__main__":
    main()
