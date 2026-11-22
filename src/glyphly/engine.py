"""The reader: image in, lines of words with boxes and confidences out."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from glyphly import models
from glyphly.detect import MAX_SIDE, TextBox, detect
from glyphly.layout import reading_order
from glyphly.recognize import Span, charset, recognize, word_spans
from glyphly.types import Box, Line, Page, Word

if TYPE_CHECKING:
    from PIL import Image

ImageLike = Any  # path, bytes, PIL image or H x W x 3 uint8 array


def to_rgb(image: ImageLike) -> np.ndarray:
    """Any supported input as an H x W x 3 uint8 RGB array (transparency on white)."""
    from PIL import Image as PILImage

    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=2)
        return np.ascontiguousarray(arr[:, :, :3].astype(np.uint8))
    if isinstance(image, (str, Path)):
        img: Image.Image = PILImage.open(image)
    elif isinstance(image, (bytes, bytearray)):
        import io

        img = PILImage.open(io.BytesIO(image))
    else:
        img = image
    if img.mode in ("RGBA", "LA", "P"):
        bg = PILImage.new("RGBA", img.size, (255, 255, 255, 255))
        img = PILImage.alpha_composite(bg, img.convert("RGBA"))
    return np.asarray(img.convert("RGB"))


def cut(rgb: np.ndarray, box: TextBox) -> np.ndarray:
    """The line's pixels, straightened: an upright crop, or a rotated resample."""
    if box.angle == 0.0:
        x0, y0, x1, y1 = box.bounds
        h, w = rgb.shape[:2]
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, math.ceil(x1)), min(h, math.ceil(y1))
        return rgb[y0 : max(y0 + 1, y1), x0 : max(x0 + 1, x1)]
    from PIL import Image as PILImage

    out_w, out_h = max(1, round(box.length)), max(1, round(box.thickness))
    c, s = math.cos(box.angle), math.sin(box.angle)
    # output (x, y) -> input: centre + (x - W/2) * (c, s) + (y - H/2) * (-s, c)
    a, b = c, -s
    d, e = s, c
    off_x = box.cx - out_w / 2 * c + out_h / 2 * s
    off_y = box.cy - out_w / 2 * s - out_h / 2 * c
    img = PILImage.fromarray(rgb).transform(
        (out_w, out_h),
        PILImage.Transform.AFFINE,
        (a, b, off_x, d, e, off_y),
        resample=PILImage.Resampling.BILINEAR,
        fillcolor=(255, 255, 255),
    )
    return np.asarray(img)


def _word_box(box: TextBox, span: Span) -> Box:
    """Page-aligned bounds of a word that runs from u0 to u1 along a (rotated) line."""
    c, s = math.cos(box.angle), math.sin(box.angle)
    ht = box.thickness / 2
    pts = []
    for u in (span.u0, span.u1):
        px, py = box.point(u)
        pts += [(px - ht * -s, py - ht * c), (px + ht * -s, py + ht * c)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _session(path: Path) -> Any:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


class Reader:
    """Loads the models once; ``read`` can then be called on many images."""

    def __init__(self, *, download: bool = True, max_side: int = MAX_SIDE) -> None:
        self._det = _session(models.ensure(models.DETECTOR, download=download))
        self._rec = _session(models.ensure(models.RECOGNIZER, download=download))
        self._chars = charset(self._rec)
        self.max_side = max_side

    def read(self, image: ImageLike) -> Page:
        rgb = to_rgb(image)
        h, w = rgb.shape[:2]
        boxes = detect(self._det, rgb, max_side=self.max_side)
        tilted = sum(b.angle != 0.0 for b in boxes)
        steps = [f"{len(boxes)} text lines found" + (f", {tilted} tilted" if tilted else "")]
        boxes.sort(key=lambda b: (b.cy, b.cx))
        crops = [cut(rgb, b) for b in boxes]
        lines: list[Line] = []
        for box, crop, (dec, share) in zip(
            boxes, crops, recognize(self._rec, self._chars, crops), strict=True
        ):
            length = float(crop.shape[1]) if box.angle == 0.0 else box.length
            spans = word_spans(dec, length / share, length)
            if not spans:
                continue
            words = [Word(sp.text, _word_box(box, sp), sp.confidence) for sp in spans]
            if box.angle == 0.0:  # upright crop: the crop's own left edge is u = 0
                x0 = max(0.0, box.bounds[0])
                words = [
                    Word(wd.text, (x0 + sp.u0, wd.box[1], x0 + sp.u1, wd.box[3]), wd.confidence)
                    for wd, sp in zip(words, spans, strict=True)
                ]
            chars = sum(len(wd.text) for wd in words)
            conf = sum(wd.confidence * len(wd.text) for wd in words) / chars
            lines.append(Line(words, box.bounds, conf))
        return Page(reading_order(lines), w, h, steps)


_default: Reader | None = None


def read(image: ImageLike) -> Page:
    """OCR one image with a shared default :class:`Reader`."""
    global _default
    if _default is None:
        _default = Reader()
    return _default.read(image)


__all__ = ["Reader", "cut", "read", "to_rgb"]
