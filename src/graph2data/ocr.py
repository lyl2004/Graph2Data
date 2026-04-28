"""OCR extraction wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from .models import BoundingBox, OCRTextBox, Point


@dataclass
class OCRConfig:
    db_thresh: float = 0.3
    box_thresh: float = 0.5
    unclip_ratio: float = 1.2
    rec_thresh: float = 0.6
    padding_size: int = 50
    scale_factor: float = 2.0
    use_cuda: bool = False


class OCRDetector:
    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self._engine = None

    def detect(self, image_bgr: np.ndarray) -> List[OCRTextBox]:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:
            raise RuntimeError("rapidocr-onnxruntime is required for OCR") from exc

        img, pad, scale = self._preprocess(image_bgr)
        if self._engine is None:
            self._engine = RapidOCR(
                det_db_thresh=self.config.db_thresh,
                det_db_unclip_ratio=self.config.unclip_ratio,
                det_db_box_thresh=self.config.box_thresh,
                det_limit_side_len=max(img.shape[:2]),
                det_limit_type="max",
                det_use_cuda=self.config.use_cuda,
            )

        result_group, _ = self._engine(img, use_det=True, use_cls=False, use_rec=True)
        texts: List[OCRTextBox] = []
        if not result_group:
            return texts

        for box, text, confidence in result_group:
            if confidence < self.config.rec_thresh:
                continue
            restored = self._restore_box(box, pad, scale)
            if np.min(restored) < -20:
                continue
            points = [Point(float(x), float(y)) for x, y in restored]
            xs = [p.x for p in points]
            ys = [p.y for p in points]
            bbox = BoundingBox(min(xs), min(ys), max(xs), max(ys))
            center = Point(float(np.mean(xs)), float(np.mean(ys)))
            texts.append(
                OCRTextBox(
                    text=str(text),
                    confidence=float(confidence),
                    polygon=points,
                    bbox=bbox,
                    center=center,
                )
            )
        return texts

    def _preprocess(self, image_bgr: np.ndarray):
        pad = self.config.padding_size
        scale = self.config.scale_factor
        padded = cv2.copyMakeBorder(image_bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        h, w = padded.shape[:2]
        resized = cv2.resize(padded, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        return resized, pad, scale

    def _restore_box(self, box, pad: int, scale: float) -> np.ndarray:
        arr = np.array(box, dtype=np.float32)
        arr /= scale
        arr -= pad
        return arr
