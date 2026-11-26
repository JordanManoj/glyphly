"""Word-level recognizers (PARSeq) for scripts the line models do not read.

IndicPhotoOCR (Bhashini, IIT Jodhpur) trained PARSeq recognizers for Indian scripts that
have no open line recognizer, such as Malayalam, Bengali, Kannada, Gujarati, Gurmukhi and
Odia. They read one word image (32 x 128 pixels) at a time, up to 25 characters, so
glyphly cuts each detected line into words at blank columns, cuts words wider than
``MAX_ASPECT`` times their height at their faintest columns (Malayalam words run long),
reads the pieces and joins them. The weights (Apache-2.0) are converted to ONNX with
glyphly's own converter; see NOTICE.

The result has the same shape as a line recognizer's: characters with positions and
probabilities, so word boxes, confidence and script choice work unchanged.
"""

from __future__ import annotations

import itertools
import json
from typing import TYPE_CHECKING, Any

import numpy as np

from glyphly.recognize import Decoded

if TYPE_CHECKING:
    import onnxruntime as ort

MAX_ASPECT = 4.0  # chosen on real Malayalam scans: median CER 33.9% (no cut) -> 26.0%
GAP = 0.3  # a blank run this many line heights wide separates two words
BATCH = 64


class WordModel:
    """A PARSeq word recognizer exported to ONNX with glyphly's metadata."""

    def __init__(self, session: ort.InferenceSession) -> None:
        meta = json.loads(session.get_modelmeta().custom_metadata_map["glyphly"])
        self.session = session
        self.itos: list[str] = meta["itos"]
        self.eos: int = meta["eos"]
        self.height, self.width = meta["img_size"]

    def read_images(self, images: list[np.ndarray]) -> list[tuple[list[str], list[float]]]:
        """Characters and their probabilities for each word image."""
        from PIL import Image

        prepared = []
        for img in images:
            small = Image.fromarray(img).convert("RGB")
            small = small.resize((self.width, self.height), Image.Resampling.BICUBIC)
            x = (np.asarray(small, dtype=np.float32) / 255.0 - 0.5) / 0.5
            prepared.append(x.transpose(2, 0, 1))
        out: list[tuple[list[str], list[float]]] = []
        for b in range(0, len(prepared), BATCH):
            probs = np.asarray(self.session.run(None, {"x": np.stack(prepared[b : b + BATCH])})[0])
            for row in probs:
                chars: list[str] = []
                conf: list[float] = []
                for step in row:
                    k = int(step.argmax())
                    if k == self.eos:
                        break
                    chars.append(self.itos[k])
                    conf.append(float(step[k]))
                out.append((chars, conf))
        return out


def is_word_model(session: Any) -> bool:
    meta = session.get_modelmeta().custom_metadata_map
    return "glyphly" in meta and json.loads(meta["glyphly"]).get("kind") == "parseq"


def _ink_columns(crop: np.ndarray) -> np.ndarray:
    gray = crop.mean(axis=2) if crop.ndim == 3 else crop.astype(np.float32)
    return (gray < gray.mean() - 0.5 * gray.std()).sum(axis=0)


def split_words(crop: np.ndarray) -> list[tuple[int, int]]:
    """Column ranges of the words on a line image (blank runs of GAP x height split)."""
    ink = _ink_columns(crop) > 0
    gap = max(3, int(GAP * crop.shape[0]))
    words: list[tuple[int, int]] = []
    start: int | None = None
    blank = 0
    for x, on in enumerate(ink.tolist()):
        if on:
            if start is None:
                start = x
            blank = 0
        elif start is not None:
            blank += 1
            if blank >= gap:
                words.append((start, x - blank + 1))
                start, blank = None, 0
    if start is not None:
        words.append((start, len(ink)))
    return [(a, b) for a, b in words if b - a > 2]


def split_long(crop: np.ndarray, a: int, b: int) -> list[tuple[int, int]]:
    """Cut columns a..b into pieces no wider than MAX_ASPECT x height, at faint columns."""
    h = crop.shape[0]
    w = b - a
    if w <= MAX_ASPECT * h:
        return [(a, b)]
    ink = _ink_columns(crop[:, a:b])
    n = int(np.ceil(w / (MAX_ASPECT * h)))
    cuts = [0]
    for k in range(1, n):
        target = k * w // n
        lo, hi = max(cuts[-1] + h // 2, target - h // 2), min(w - h // 2, target + h // 2)
        if hi > lo:
            cuts.append(lo + int(np.argmin(ink[lo:hi])))
    cuts.append(w)
    return [(a + c0, a + c1) for c0, c1 in itertools.pairwise(cuts) if c1 - c0 > 2]


def recognize_words(model: WordModel, crops: list[np.ndarray]) -> list[tuple[Decoded, float]]:
    """Read line images word by word; same output shape as ``recognize.recognize``.

    Steps are pixel columns of the line image: each character is placed evenly inside its
    word piece, and a space sits in the gap between words, so ``word_spans`` finds the
    words' boxes.
    """
    plans: list[list[list[tuple[int, int]]]] = []  # line -> word -> pieces
    images: list[np.ndarray] = []
    for crop in crops:
        pad = max(2, crop.shape[0] // 8)
        words = []
        for a, b in split_words(crop):
            pieces = split_long(crop, a, b)
            for p0, p1 in pieces:
                images.append(crop[:, max(0, p0 - pad) : min(crop.shape[1], p1 + pad)])
            words.append(pieces)
        plans.append(words)
    read = model.read_images(images) if images else []
    out: list[tuple[Decoded, float]] = []
    k = 0
    for crop, words in zip(crops, plans, strict=True):
        chars: list[str] = []
        steps: list[int] = []
        probs: list[float] = []
        for w, pieces in enumerate(words):
            if w and chars:
                chars.append(" ")
                steps.append((words[w - 1][-1][1] + pieces[0][0]) // 2)
                probs.append(1.0)
            for p0, p1 in pieces:
                got, conf = read[k]
                k += 1
                for i, (c, p) in enumerate(zip(got, conf, strict=True)):
                    chars.append(c)
                    steps.append(p0 + int((i + 0.5) * (p1 - p0) / max(1, len(got))))
                    probs.append(p)
        out.append((Decoded(chars, steps, probs, max(1, crop.shape[1])), 1.0))
    return out


__all__ = ["MAX_ASPECT", "WordModel", "is_word_model", "recognize_words", "split_words"]
