"""Batch benchmark runner for synthetic assets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

from .colors import ColorExtractorConfig, CurveColorExtractor
from .image_io import ensure_dir, load_bgr, write_json
from .lines import LinePathExtractor
from .masks import CurveMaskExtractor
from .models import BoundingBox, CurvePrototype
from .models import to_serializable
from .quality import evaluate_curve_path
from .synthetic import SyntheticConfig, generate_benchmark


def run_path_benchmark(case_dir: str) -> Dict:
    root = Path(case_dir)
    truth_axes_path = root / "truth_axes.json"
    truth_data_path = root / "truth_data.csv"
    mask_dir = root / "masks"

    with open(truth_axes_path, "r", encoding="utf-8") as f:
        truth_axes = json.load(f)

    extractor = LinePathExtractor()
    curve_metrics: List[Dict] = []
    for mask_path in sorted(mask_dir.glob("*.png")):
        curve_id = mask_path.stem
        path = extractor.extract_from_mask_file(str(mask_path), curve_id=curve_id)
        path_json = to_serializable(path)
        metrics = evaluate_curve_path(path_json, truth_axes, str(truth_data_path), curve_id)
        curve_metrics.append(metrics)

    valid = [m for m in curve_metrics if m.get("path_success")]
    summary = {
        "case_dir": str(root),
        "curve_count": len(curve_metrics),
        "valid_curve_count": len(valid),
        "mean_chamfer_distance_px": _mean(valid, "chamfer_distance_px"),
        "mean_hausdorff_distance_px": _mean(valid, "hausdorff_distance_px"),
        "mean_truth_to_pred_px": _mean(valid, "mean_truth_to_pred_px"),
        "mean_completed_to_truth_px": _mean([m for m in valid if m.get("mean_completed_to_truth_px") is not None], "mean_completed_to_truth_px"),
    }
    return {"summary": summary, "curves": curve_metrics}


def run_suite(output_root: str) -> Dict:
    """Generate standard synthetic cases and run truth/predicted mask metrics."""
    ensure_dir(output_root)
    synthetic_root = os.path.join(output_root, "synthetic")
    result_root = ensure_dir(os.path.join(output_root, "results"))

    cases = [
        ("basic_curves", SyntheticConfig(seed=42, n_curves=6, palette="basic")),
        ("achromatic_curves", SyntheticConfig(seed=7, n_curves=6, palette="achromatic")),
    ]

    suite_cases = []
    for name, config in cases:
        manifest = generate_benchmark(synthetic_root, name=name, config=config)
        case_dir = os.path.dirname(manifest["image_path"])

        truth = run_path_benchmark(case_dir)
        pred = run_predicted_mask_benchmark(case_dir, output_dir=os.path.join(result_root, f"{name}_predicted_masks"))

        truth_path = os.path.join(result_root, f"{name}_truth_mask_metrics.json")
        pred_path = os.path.join(result_root, f"{name}_predicted_mask_metrics.json")
        write_json(truth_path, truth)
        write_json(pred_path, pred)

        suite_cases.append(
            {
                "name": name,
                "case_dir": case_dir,
                "truth_metrics": truth["summary"],
                "predicted_metrics": pred["summary"],
                "truth_metrics_path": truth_path,
                "predicted_metrics_path": pred_path,
            }
        )

    suite = {"output_root": output_root, "cases": suite_cases}
    write_json(os.path.join(result_root, "suite_summary.json"), suite)
    return suite


def run_predicted_mask_benchmark(case_dir: str, output_dir: Optional[str] = None) -> Dict:
    root = Path(case_dir)
    image_path = root / "image.png"
    truth_axes_path = root / "truth_axes.json"
    truth_curves_path = root / "truth_curves.json"
    truth_data_path = root / "truth_data.csv"

    with open(truth_axes_path, "r", encoding="utf-8") as f:
        truth_axes = json.load(f)
    with open(truth_curves_path, "r", encoding="utf-8") as f:
        truth_curves = json.load(f)["curves"]

    image = load_bgr(str(image_path))
    plot = truth_axes["plot_area"]
    plot_bbox = BoundingBox(plot["x_min"], plot["y_min"], plot["x_max"], plot["y_max"])
    prototypes = CurveColorExtractor(ColorExtractorConfig(min_ratio=0.0004)).extract(image, plot_bbox)

    mask_output = Path(output_dir) if output_dir else root / "predicted_masks"
    mask_output.mkdir(parents=True, exist_ok=True)

    mask_extractor = CurveMaskExtractor()
    path_extractor = LinePathExtractor()
    rows: List[Dict] = []
    for truth in truth_curves:
        matched = _match_prototype(truth["color"], prototypes)
        if matched is None:
            rows.append({"path_success": False, "curve_id": truth["curve_id"], "error": "no color prototype match"})
            continue
        matched.curve_id = truth["curve_id"]
        mask, mask_info = mask_extractor.extract_mask(image, matched, plot_bbox)
        mask_path = mask_output / f"{truth['curve_id']}.png"
        cv2.imwrite(str(mask_path), mask)
        path = path_extractor.extract_from_mask_image(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), curve_id=truth["curve_id"])
        metrics = evaluate_curve_path(to_serializable(path), truth_axes, str(truth_data_path), truth["curve_id"])
        metrics["matched_rgb"] = matched.rgb
        metrics["truth_color"] = truth["color"]
        metrics["mask_pixel_count"] = mask_info.pixel_count
        metrics["mask_confidence"] = mask_info.confidence
        rows.append(metrics)

    valid = [m for m in rows if m.get("path_success")]
    summary = {
        "case_dir": str(root),
        "mode": "predicted_mask",
        "curve_count": len(rows),
        "valid_curve_count": len(valid),
        "prototype_count": len(prototypes),
        "mean_chamfer_distance_px": _mean(valid, "chamfer_distance_px"),
        "mean_hausdorff_distance_px": _mean(valid, "hausdorff_distance_px"),
        "mean_truth_to_pred_px": _mean(valid, "mean_truth_to_pred_px"),
        "mean_completed_to_truth_px": _mean(
            [m for m in valid if m.get("mean_completed_to_truth_px") is not None], "mean_completed_to_truth_px"
        ),
    }
    return {"summary": summary, "curves": rows, "prototypes": [to_serializable(p) for p in prototypes]}


def _match_prototype(hex_color: str, prototypes: List[CurvePrototype]) -> Optional[CurvePrototype]:
    if not prototypes:
        return None
    target = _hex_to_rgb(hex_color)
    best = min(prototypes, key=lambda p: _rgb_distance(target, p.rgb))
    # Return a shallow copy so assigning curve_id does not mutate the list for future matching semantics.
    return CurvePrototype(
        curve_id=best.curve_id,
        rgb=best.rgb,
        lab=best.lab,
        area=best.area,
        ratio=best.ratio,
        label=best.label,
        line_style=best.line_style,
        marker_style=best.marker_style,
        confidence=best.confidence,
        source=best.source,
    )


def _hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.strip().lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def _rgb_distance(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)) ** 0.5


def _mean(rows: List[Dict], key: str):
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Graph2Data synthetic benchmark metrics.")
    parser.add_argument("--case", default=None, help="Synthetic case directory")
    parser.add_argument("--mode", choices=["truth-mask", "predicted-mask"], default="truth-mask")
    parser.add_argument("--mask_out", default=None, help="Optional output directory for predicted masks")
    parser.add_argument("--suite", action="store_true", help="Generate and run the standard benchmark suite")
    parser.add_argument("--suite_out", default="benchmarks/suite", help="Suite output root")
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    if args.suite:
        result = run_suite(args.suite_out)
    elif not args.case:
        raise ValueError("--case is required unless --suite is used")
    elif args.mode == "truth-mask":
        result = run_path_benchmark(args.case)
    else:
        result = run_predicted_mask_benchmark(args.case, args.mask_out)
    if args.out:
        write_json(args.out, result)
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
