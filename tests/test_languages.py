"""Languages: script models, right-to-left order, a real Cyrillic page."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from glyphly import ModelError, Reader
from glyphly.engine import logical_order
from glyphly.models import SCRIPT_MODELS, home, script_for


def test_language_codes_map_to_scripts() -> None:
    assert script_for("ru") == "cyrillic"
    assert script_for("uk-UA") == "cyrillic"
    assert script_for("ar") == script_for("fa") == "arabic"
    assert script_for("hi") == "devanagari"
    assert script_for("en") is None  # read by the default model
    assert script_for("zh") is None
    assert set(SCRIPT_MODELS) >= {"cyrillic", "arabic", "devanagari", "korean", "thai"}


def test_right_to_left_text_is_reversed_but_numbers_are_not() -> None:
    # an Arabic line as a model reads it, left to right: letters reversed, numbers not
    visual = list("2026 بأ 14 دج")
    assert "".join(logical_order(visual)) == "جد 14 أب 2026"


def _cyrillic_font() -> ImageFont.FreeTypeFont | None:
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, 40)
    return None


TEXT = "Городской совет утвердил бюджет 2450"


@pytest.fixture(scope="module")
def reader() -> Reader:
    try:
        return Reader(download=False)
    except ModelError:
        pytest.skip("model files not downloaded (run `glyphly download --all`)")


def test_auto_reads_a_russian_page(reader: Reader) -> None:
    font = _cyrillic_font()
    if font is None:
        pytest.skip("no font with Cyrillic letters on this machine")
    if not (home() / SCRIPT_MODELS["cyrillic"].name).exists():
        pytest.skip("the Cyrillic model is not downloaded (glyphly download --all)")
    img = Image.new("RGB", (1000, 140), "white")
    ImageDraw.Draw(img).text((30, 40), TEXT, fill="black", font=font)
    page = reader.read(img)
    assert " ".join(page.text.split()) == TEXT
    assert page.lines[0].script == "cyrillic"
    assert any("cyrillic" in s for s in page.steps)


def test_latin_page_never_loads_script_models(reader: Reader) -> None:
    font = ImageFont.load_default(size=40)
    img = Image.new("RGB", (1000, 140), "white")
    ImageDraw.Draw(img).text(
        (30, 40), "The council approved the budget 2450", fill="black", font=font
    )
    page = reader.read(img)
    assert page.lines[0].script == "default"
    assert len(page.steps) == 1  # no other script was tried
