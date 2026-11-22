"""Read one text line: a recognition network plus CTC decoding, with word positions.

Each line image is scaled to 48 px high (any width), and the network outputs, for every
column step, a probability for each character (and a "blank"). CTC greedy decoding keeps
the most likely character per step, merges repeats and drops blanks. Because every kept
character comes from a known column step, its position in the line is known too, so each
word gets its own box (the network has no word-box output).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import onnxruntime as ort

HEIGHT = 48
BATCH = 6
MAX_WIDTH = 3200  # longer line images are scaled down to this width


@dataclass
class Decoded:
    chars: list[str]
    steps: list[int]  # column step of each character
    probs: list[float]  # probability of each character
    n_steps: int


def charset(session: ort.InferenceSession) -> list[str]:
    """Index -> character: blank, the model's own character list, then a space."""
    meta = session.get_modelmeta().custom_metadata_map
    chars = meta["character"].splitlines()
    return ["", *chars, " "]


def ctc_decode(probs: np.ndarray, chars: list[str]) -> Decoded:
    """Greedy CTC decoding of one line's (steps x classes) probability matrix."""
    best = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    out_c: list[str] = []
    out_s: list[int] = []
    out_p: list[float] = []
    prev = 0
    for t, k in enumerate(best.tolist()):
        if k != 0 and k != prev:
            out_c.append(chars[k])
            out_s.append(t)
            out_p.append(float(conf[t]))
        prev = k
    return Decoded(out_c, out_s, out_p, probs.shape[0])


@dataclass
class Span:
    """A word as positions along its line: ``u0``..``u1`` pixels from the line's start."""

    text: str
    u0: float
    u1: float
    confidence: float


def word_spans(dec: Decoded, padded_length: float, length: float) -> list[Span]:
    """Split decoded characters into words at spaces, located along the line.

    ``padded_length`` is the line length (pixels) the network's steps span, including
    batch padding; ``length`` is the line's real length.
    """
    if not dec.chars:
        return []
    step_w = padded_length / dec.n_steps
    words: list[Span] = []
    cur: list[int] = []

    def flush() -> None:
        if not cur:
            return
        text = "".join(dec.chars[i] for i in cur)
        left = max(0.0, (dec.steps[cur[0]] - 1) * step_w)
        right = min(length, (dec.steps[cur[-1]] + 2) * step_w)
        conf = sum(dec.probs[i] for i in cur) / len(cur)
        words.append(Span(text, left, right, conf))
        cur.clear()

    for i, c in enumerate(dec.chars):
        if c.isspace():
            flush()
        else:
            cur.append(i)
    flush()
    return words


def _prepare(crop: np.ndarray) -> np.ndarray:
    from PIL import Image

    h, w = crop.shape[:2]
    nw = max(8, min(MAX_WIDTH, int(np.ceil(HEIGHT * w / max(h, 1)))))
    img = np.asarray(Image.fromarray(crop).resize((nw, HEIGHT), Image.Resampling.BILINEAR))
    x = (img[:, :, ::-1].astype(np.float32) / 255.0 - 0.5) / 0.5  # RGB -> BGR, [-1, 1]
    return x.transpose(2, 0, 1)


def recognize(
    session: ort.InferenceSession, chars: list[str], crops: list[np.ndarray]
) -> list[tuple[Decoded, float]]:
    """Decode each crop; returns (decoded, share of the padded width that is the crop)."""
    prepared = [_prepare(c) for c in crops]
    order = sorted(range(len(crops)), key=lambda i: prepared[i].shape[2])
    results: list[tuple[Decoded, float] | None] = [None] * len(crops)
    name = session.get_inputs()[0].name
    for b in range(0, len(order), BATCH):
        idx = order[b : b + BATCH]
        width = max(prepared[i].shape[2] for i in idx)
        batch = np.zeros((len(idx), 3, HEIGHT, width), dtype=np.float32)
        for row, i in enumerate(idx):
            batch[row, :, :, : prepared[i].shape[2]] = prepared[i]
        out = np.asarray(session.run(None, {name: batch})[0])
        for row, i in enumerate(idx):
            results[i] = (ctc_decode(out[row], chars), prepared[i].shape[2] / width)
    return [r for r in results if r is not None]


__all__ = ["Decoded", "Span", "charset", "ctc_decode", "recognize", "word_spans"]
