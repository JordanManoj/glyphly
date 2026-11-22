"""End to end with the real models. Skipped when they are not downloaded (CI fetches them)."""

from __future__ import annotations

import json
import math

import pytest
from PIL import Image, ImageDraw, ImageFont

from glyphly import ModelError, Reader
from glyphly.cli import main

LINES = ["The harbour council met on 14 March 2026.", "Budget: 2,450 for the river sensor."]


@pytest.fixture(scope="module")
def reader() -> Reader:
    try:
        return Reader(download=False)
    except ModelError:
        pytest.skip("model files not downloaded (run `glyphly download`)")


def _page(angle: float = 0.0) -> Image.Image:
    font = ImageFont.load_default(size=36)
    img = Image.new("RGB", (1100, 260), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(LINES):
        draw.text((40, 50 + i * 80), line, fill="black", font=font)
    if angle:
        img = img.rotate(angle, expand=True, fillcolor="white", resample=Image.Resampling.BICUBIC)
    return img


def _norm(text: str) -> str:
    return " ".join(text.split())


def test_reads_lines_with_word_boxes(reader: Reader) -> None:
    page = reader.read(_page())
    assert [_norm(line.text) for line in page.lines] == LINES
    assert page.confidence is not None
    assert page.confidence > 0.9
    first = page.lines[0].words[0]
    assert first.text == "The"
    x0, top, x1, bottom = first.box
    assert 20 < x0 < x1 < 160
    assert 30 < top < bottom < 120


def test_reads_a_tilted_page(reader: Reader) -> None:
    page = reader.read(_page(angle=4))
    assert [_norm(line.text) for line in page.lines] == LINES
    assert any("tilted" in s for s in page.steps)


def test_cli_json(reader: Reader, tmp_path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "page.png"
    _page().save(path)
    assert main(["read", str(path), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["width"] == 1100
    assert _norm(" ".join(w["text"] for w in out["lines"][0]["words"])) == LINES[0]
    assert all(math.isfinite(v) for v in out["lines"][0]["box"])
