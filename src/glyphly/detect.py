"""Find text lines: a segmentation network marks text pixels, numpy turns them into boxes.

The network (DB, "differentiable binarisation") outputs, for every pixel, the probability
that it lies inside a text line, with lines shrunk a little so neighbours stay apart.
Post-processing (written here in numpy, no OpenCV):

1. threshold the map and label connected regions (runs of pixels row by row, merged with
   a union-find);
2. drop regions whose mean probability is low (likely not text) or that are tiny;
3. measure each region's direction from its pixel moments, so tilted lines (skewed
   scans, phone photos) get a rotated box instead of an upright one that would also cut
   through the neighbouring lines;
4. grow each box back by the amount the network shrank it ("unclip").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import onnxruntime as ort

# ImageNet statistics, applied to BGR input as in the detector's training.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

THRESH = 0.3  # pixel probability above which a pixel is "text"
BOX_THRESH = 0.5  # minimum mean probability of a region to keep it
UNCLIP_RATIO = 1.6  # how far boxes are grown back (DB's shrink ratio, inverted)
MIN_SIDE = 3  # regions thinner than this (in map pixels) are noise
MAX_SIDE = 2048  # longest side fed to the network (larger pages are scaled down)
STRAIGHT = math.radians(0.5)  # below this tilt a line is cut out upright (no resampling)
ELONGATED = 2.0  # long axis at least this many times the short one to trust a direction


@dataclass
class Region:
    """A connected text region, accumulated run by run."""

    x0: int
    y0: int
    x1: int  # inclusive
    y1: int  # inclusive
    pixels: int = 0
    prob_sum: float = 0.0
    ends: list[tuple[int, int, int]] = field(default_factory=list)  # (y, start, end)
    # pixel moments for the direction: sums of x, y, x^2, y^2, xy
    sx: float = 0.0
    sy: float = 0.0
    sxx: float = 0.0
    syy: float = 0.0
    sxy: float = 0.0

    @property
    def score(self) -> float:
        return self.prob_sum / self.pixels if self.pixels else 0.0

    def add(self, y: int, start: int, end: int, prob: float) -> None:
        n = end - start + 1
        self.x0, self.x1 = min(self.x0, start), max(self.x1, end)
        self.y0, self.y1 = min(self.y0, y), max(self.y1, y)
        self.pixels += n
        self.prob_sum += prob
        self.ends.append((y, start, end))
        sum_x = n * (start + end) / 2
        # sum of x^2 over start..end
        sum_xx = (end * (end + 1) * (2 * end + 1) - (start - 1) * start * (2 * start - 1)) / 6
        self.sx += sum_x
        self.sy += n * y
        self.sxx += sum_xx
        self.syy += n * y * y
        self.sxy += y * sum_x

    def angle(self) -> float:
        """Direction of the region's long axis, in radians (0 = horizontal).

        0 when the region has no clear long axis (a single character, a dot): its
        direction would be noise.
        """
        n = self.pixels
        mx, my = self.sx / n, self.sy / n
        cxx = self.sxx / n - mx * mx
        cyy = self.syy / n - my * my
        cxy = self.sxy / n - mx * my
        half_diff = math.hypot((cxx - cyy) / 2, cxy)
        long_var, short_var = (cxx + cyy) / 2 + half_diff, (cxx + cyy) / 2 - half_diff
        if long_var < ELONGATED**2 * max(short_var, 1e-9):
            return 0.0
        return 0.5 * math.atan2(2 * cxy, cxx - cyy)


@dataclass
class TextBox:
    """A (possibly rotated) text-line rectangle in image pixels."""

    cx: float
    cy: float
    length: float  # along the line
    thickness: float  # across the line
    angle: float  # radians, 0 = horizontal

    def corners(self) -> list[tuple[float, float]]:
        c, s = math.cos(self.angle), math.sin(self.angle)
        hl, ht = self.length / 2, self.thickness / 2
        return [
            (self.cx + u * c - v * s, self.cy + u * s + v * c)
            for u, v in ((-hl, -ht), (hl, -ht), (hl, ht), (-hl, ht))
        ]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        pts = self.corners()
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    def point(self, u: float) -> tuple[float, float]:
        """Image position of the point ``u`` pixels from the line's start, on its axis."""
        c, s = math.cos(self.angle), math.sin(self.angle)
        d = u - self.length / 2
        return self.cx + d * c, self.cy + d * s


def _row_runs(row: np.ndarray) -> list[tuple[int, int]]:
    """(start, end inclusive) of each run of True values in a 1-D boolean array."""
    padded = np.concatenate(([False], row, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return list(zip(edges[0::2].tolist(), (edges[1::2] - 1).tolist(), strict=True))


def regions(mask: np.ndarray, prob: np.ndarray) -> list[Region]:
    """Connected regions (8-connectivity) of ``mask``."""
    parent: list[int] = []

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    runs: list[tuple[int, int, int]] = []  # (y, start, end)
    prev: list[int] = []  # run ids on the previous row
    for y in range(mask.shape[0]):
        current: list[int] = []
        j = 0
        for start, end in _row_runs(mask[y]):
            rid = len(runs)
            runs.append((y, start, end))
            parent.append(rid)
            # previous-row runs that touch this one (diagonals included)
            while j < len(prev) and runs[prev[j]][2] < start - 1:
                j += 1
            k = j
            while k < len(prev) and runs[prev[k]][1] <= end + 1:
                a, b = find(rid), find(prev[k])
                if a != b:
                    parent[max(a, b)] = min(a, b)
                k += 1
            current.append(rid)
        prev = current

    out: dict[int, Region] = {}
    for rid, (y, start, end) in enumerate(runs):
        root = find(rid)
        r = out.get(root)
        if r is None:
            r = out[root] = Region(start, y, end, y)
        r.add(y, start, end, float(prob[y, start : end + 1].sum()))
    return list(out.values())


def unclip(x0: float, y0: float, x1: float, y1: float, ratio: float = UNCLIP_RATIO) -> tuple:
    """Grow a box by DB's offset: area * ratio / perimeter on every side."""
    w, h = x1 - x0, y1 - y0
    d = w * h * ratio / (2 * (w + h)) if w + h > 0 else 0.0
    return x0 - d, y0 - d, x1 + d, y1 + d


def text_box(r: Region, sx: float, sy: float) -> TextBox:
    """The region's rectangle along its own direction, grown back, in image pixels."""
    a = r.angle()
    if abs(a) < STRAIGHT:
        a = 0.0
    c, s = math.cos(a), math.sin(a)
    # extent along (u) and across (v) the direction, from the run end points (map pixels,
    # pixel centres +- 0.5)
    us, vs = [], []
    for y, start, end in r.ends:
        for x in (start - 0.5, end + 0.5):
            for yy in (y - 0.5, y + 0.5):
                us.append(x * c + yy * s)
                vs.append(-x * s + yy * c)
    u0, u1, v0, v1 = min(us), max(us), min(vs), max(vs)
    u0, v0, u1, v1 = unclip(u0, v0, u1, v1)
    cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
    cx, cy = cu * c - cv * s, cu * s + cv * c
    # map -> image pixels (the two axes may be scaled slightly differently)
    length = math.hypot((u1 - u0) * c * sx, (u1 - u0) * s * sy)
    thickness = math.hypot((v1 - v0) * s * sx, (v1 - v0) * c * sy)
    angle = math.atan2(s * sy, c * sx)
    return TextBox(cx * sx, cy * sy, length, thickness, angle)


def _input(rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, float, float]:
    from PIL import Image

    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    nh = max(32, round(h * scale / 32) * 32)
    nw = max(32, round(w * scale / 32) * 32)
    img = rgb if (nh, nw) == (h, w) else np.asarray(Image.fromarray(rgb).resize((nw, nh)))
    x = (img[:, :, ::-1].astype(np.float32) / 255.0 - _MEAN) / _STD  # RGB -> BGR
    return x.transpose(2, 0, 1)[None], w / nw, h / nh


def detect(
    session: ort.InferenceSession, rgb: np.ndarray, *, max_side: int = MAX_SIDE
) -> list[TextBox]:
    """Text-line boxes in pixels of ``rgb`` (H x W x 3 uint8)."""
    x, sx, sy = _input(rgb, max_side)
    prob = np.asarray(session.run(None, {session.get_inputs()[0].name: x})[0])[0, 0]
    boxes = []
    for r in regions(prob > THRESH, prob):
        if min(r.x1 - r.x0, r.y1 - r.y0) + 1 < MIN_SIDE or r.score < BOX_THRESH:
            continue
        boxes.append(text_box(r, sx, sy))
    return boxes


__all__ = ["Region", "TextBox", "detect", "regions", "text_box", "unclip"]
