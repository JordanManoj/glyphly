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


def _font_or_skip(path: str, size: int = 40) -> ImageFont.FreeTypeFont:
    if not Path(path).is_file():
        pytest.skip(f"font {path} is not on this machine")
    return ImageFont.truetype(path, size)


def test_malayalam_is_read_with_its_word_model(reader: Reader) -> None:
    font = _font_or_skip(r"C:\Windows\Fonts\Nirmala.ttc")
    if not (home() / SCRIPT_MODELS["malayalam"].name).exists():
        pytest.skip("the Malayalam model is not downloaded (glyphly download --all)")
    img = Image.new("RGB", (1200, 300), "white")
    draw = ImageDraw.Draw(img)
    draw.text((30, 40), "റെക്കോർഡ് ബാറ്റിങ്ങുമായി വൈഭവ് സൂര്യവംശി", fill="black", font=font)
    draw.text((30, 160), "ലോക ക്രിക്കറ്റിൽ ഒരിക്കൽ മാത്രം സംഭവിക്കുന്ന", fill="black", font=font)
    page = Reader(lang=["auto", "ml"], download=False).read(img)
    assert page.rotation == 0
    assert all(line.script == "malayalam" for line in page.lines)
    malayalam = sum("\u0d00" <= c <= "\u0d7f" for c in page.text)
    assert malayalam > 0.8 * len(page.text.replace(" ", "").replace("\n", ""))
    # PIL draws without shaping, so exact words vary by machine; the script must not


def test_word_models_are_not_guessed_in_auto_mode(reader: Reader) -> None:
    """Word models read any script confidently, so auto mode never picks them by itself."""
    font = _font_or_skip(r"C:\Windows\Fonts\Nirmala.ttc")
    img = Image.new("RGB", (1200, 160), "white")
    ImageDraw.Draw(img).text((30, 40), "ലോക ക്രിക്കറ്റിൽ ഒരിക്കൽ", fill="black", font=font)
    page = reader.read(img)
    assert all(line.script != "malayalam" for line in page.lines)


def test_unknown_script_is_not_flipped_and_is_flagged(reader: Reader) -> None:
    """Hebrew has no model yet: the page must stay upright and say it is unsure."""
    font = _font_or_skip(r"C:\Windows\Fonts\arial.ttf")
    img = Image.new("RGB", (1200, 300), "white")
    draw = ImageDraw.Draw(img)
    draw.text((30, 40), "מועצת העיר אישרה את התקציב החדש", fill="black", font=font)
    draw.text((30, 160), "הישיבה נערכה ביום שלישי בבוקר", fill="black", font=font)
    page = reader.read(img)
    assert page.rotation == 0
    assert page.confidence is not None
    assert page.confidence < 0.8
    assert any("low confidence" in s for s in page.steps)
