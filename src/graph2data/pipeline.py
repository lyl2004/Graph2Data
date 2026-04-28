"""Structured extraction pipeline entry point."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from .axes import AxisDetector, AxisDetectorConfig
from .colors import CurveColorExtractor
from .image_io import load_bgr, write_json
from .layout import assign_text_to_regions, build_nine_grid
from .legend import LegendDetector
from .models import PipelineResult
from .ocr import OCRDetector


@dataclass
class PipelineConfig:
    run_ocr: bool = False
    run_colors: bool = False
    use_adaptive_axis_threshold: bool = False
    use_morph_axis_close: bool = False


class GraphExtractionPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

    def run(self, image_path: str) -> PipelineResult:
        image = load_bgr(image_path)
        h, w = image.shape[:2]
        axis_config = AxisDetectorConfig(
            use_adaptive=self.config.use_adaptive_axis_threshold,
            use_morph=self.config.use_morph_axis_close,
        )
        axis = AxisDetector(axis_config).detect(image)
        warnings = []
        ocr_results = []
        layout = None
        curves = []
        legends = []

        if self.config.run_ocr:
            try:
                ocr_results = OCRDetector().detect(image)
            except Exception as exc:
                warnings.append(f"OCR failed: {exc}")

        if axis.success and axis.plot_area is not None:
            layout = build_nine_grid(axis.image_size, axis.plot_area)
            if ocr_results:
                layout = assign_text_to_regions(layout, ocr_results)
                legends = LegendDetector().detect(ocr_results, axis.plot_area)

            if self.config.run_colors:
                try:
                    curves = CurveColorExtractor().extract(
                        image,
                        axis.plot_area.bbox,
                        exclude_regions=[legend.bbox for legend in legends],
                    )
                except Exception as exc:
                    warnings.append(f"Color extraction failed: {exc}")
        else:
            warnings.append("Skipping layout and curve prototype extraction because axis detection failed")

        return PipelineResult(
            image_path=image_path,
            image_size=(w, h),
            axis=axis,
            ocr=ocr_results,
            layout=layout,
            legends=legends,
            curves=curves,
            warnings=warnings,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the structured Graph2Data extraction pipeline.")
    parser.add_argument("--img", required=True, help="Input image path")
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    parser.add_argument("--ocr", action="store_true", help="Run OCR")
    parser.add_argument("--colors", action="store_true", help="Run curve color prototype extraction")
    parser.add_argument("--axis_adaptive", action="store_true", help="Use adaptive thresholding for axis detection")
    parser.add_argument("--axis_morph", action="store_true", help="Use morphology close for axis detection")
    args = parser.parse_args()

    result = GraphExtractionPipeline(
        PipelineConfig(
            run_ocr=args.ocr,
            run_colors=args.colors,
            use_adaptive_axis_threshold=args.axis_adaptive,
            use_morph_axis_close=args.axis_morph,
        )
    ).run(args.img)

    if args.out:
        write_json(args.out, result)
    else:
        import json

        from .models import to_serializable

        print(json.dumps(to_serializable(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
