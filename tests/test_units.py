"""Parts that need no model files."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from glyphly import models
from glyphly.detect import Region, regions, text_box, unclip
from glyphly.engine import cut, to_rgb
from glyphly.recognize import ctc_decode, word_spans


def test_regions_label_separate_blobs_and_join_diagonals() -> None:
    mask = np.zeros((6, 10), dtype=bool)
    mask[1, 1:4] = True
    mask[2, 4] = True  # touches the first blob diagonally
    mask[4, 7:9] = True  # separate
    found = sorted(regions(mask, mask.astype(np.float32)), key=lambda r: r.x0)
    assert [(r.x0, r.y0, r.x1, r.y1, r.pixels) for r in found] == [
        (1, 1, 4, 2, 4),
        (7, 4, 8, 4, 2),
    ]
    assert found[0].score == pytest.approx(1.0)


def test_u_shape_is_one_region() -> None:
    mask = np.zeros((4, 7), dtype=bool)
    mask[0:3, 1] = True
    mask[0:3, 5] = True
    mask[3, 1:6] = True  # the bottom joins the two arms only on the last row
    assert len(regions(mask, mask.astype(np.float32))) == 1


def test_unclip_grows_by_area_over_perimeter() -> None:
    x0, y0, x1, y1 = unclip(0, 0, 100, 10, ratio=1.6)
    d = 100 * 10 * 1.6 / (2 * 110)
    assert (x0, y0, x1, y1) == pytest.approx((-d, -d, 100 + d, 10 + d))


def _band(angle_deg: float) -> Region:
    """A thin line of pixels at the given angle, accumulated as a region."""
    mask = np.zeros((200, 400), dtype=bool)
    a = math.radians(angle_deg)
    for t in range(300):
        x = 50 + t * math.cos(a)
        y = 100 + t * math.sin(a)
        for dy in range(-3, 4):
            mask[round(y) + dy, round(x)] = True
    (region,) = regions(mask, mask.astype(np.float32))
    return region


def test_direction_of_a_tilted_line() -> None:
    box = text_box(_band(5), 1.0, 1.0)
    assert math.degrees(box.angle) == pytest.approx(5, abs=0.3)
    assert box.length > 300 > 4 * box.thickness


def test_short_blob_stays_upright() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:12, 5:12] = True  # a square: no direction to trust
    (region,) = regions(mask, mask.astype(np.float32))
    assert region.angle() == 0.0


def test_ctc_merges_repeats_and_drops_blanks() -> None:
    chars = ["", "a", "b", " "]
    best = [1, 1, 0, 1, 2, 0, 3, 2]  # a a _ a b _ space b  ->  "aab b"
    probs = np.full((len(best), 4), 0.01, dtype=np.float32)
    for t, k in enumerate(best):
        probs[t, k] = 0.9
    dec = ctc_decode(probs, chars)
    assert "".join(dec.chars) == "aab b"
    assert dec.steps == [0, 3, 4, 6, 7]
    spans = word_spans(dec, padded_length=80.0, length=80.0)
    assert [s.text for s in spans] == ["aab", "b"]
    assert spans[0].u0 < spans[0].u1 <= spans[1].u0 + 20
    assert spans[1].confidence == pytest.approx(0.9)


def test_upright_cut_and_rgb_conversion() -> None:
    from PIL import Image

    rgba = Image.new("RGBA", (30, 20), (0, 0, 0, 0))  # transparent -> white
    rgb = to_rgb(rgba)
    assert rgb.shape == (20, 30, 3)
    assert rgb[0, 0].tolist() == [255, 255, 255]
    from glyphly.detect import TextBox

    crop = cut(rgb, TextBox(15, 10, 10, 4, 0.0))
    assert crop.shape == (4, 10, 3)
    tilted = cut(rgb, TextBox(15, 10, 10, 4, math.radians(10)))
    assert tilted.shape == (4, 10, 3)


def test_model_fingerprint_is_checked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GLYPHLY_HOME", str(tmp_path))
    good = b"model bytes"
    fake = models.ModelFile("m.onnx", "https://example.invalid/m.onnx",
                            hashlib.sha256(good).hexdigest(), 0.0)  # fmt: skip
    (tmp_path / "m.onnx").write_bytes(good)
    assert models.ensure(fake, download=False) == tmp_path / "m.onnx"
    (tmp_path / "m.onnx").write_bytes(b"tampered")
    with pytest.raises(models.ModelError, match="glyphly download"):
        models.ensure(fake, download=False)
    assert not (tmp_path / "m.onnx").exists()  # a bad file is never kept
