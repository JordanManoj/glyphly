"""The hard OCR suite: cases where OCR engines really differ.

uv run python -m bench.ocr --suite hard --split test

Same rules as bench/ocr.py: our own text, exact ground truth, nothing third-party.
Each case has its own layout, so the reading order is part of what is measured:

- small-8pt-150dpi  small print from a low-resolution scan
- two-columns       a two-column page; columns must not be interleaved
- table             a ruled table of names, numbers and percentages, read row by row
- inverted          white text on a dark background
- colour            dark red text on light blue paper
- perspective       a page photographed at an angle (keystone distortion)
- cheap-scan        120 dpi, slight tilt, blur and heavy JPEG together
"""

from __future__ import annotations

import random
from collections.abc import Callable

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from bench.ocr import (
    DPI,
    FONTS,
    NAMES,
    PAGE_W,
    PAGES,
    WORDS,
    Sample,
    _font,
    _jpeg,
    _rotate,
    _scale,
    page_text,
    render,
    split_of,
)


def _wrap(text: str, font: object, width: int) -> list[str]:
    lines, cur = [], ""
    for w in text.split():
        trial = f"{cur} {w}".strip()
        if font.getlength(trial) > width and cur:  # type: ignore[attr-defined]
            lines.append(cur)
            cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def two_columns(paragraphs: list[str], font_name: str) -> tuple[Image.Image, str]:
    px = round(10 * DPI / 72)
    font = _font(font_name, px)
    margin, gutter = 200, 120
    col_w = (PAGE_W - 2 * margin - gutter) // 2
    lines = [ln for p in paragraphs for ln in _wrap(p, font, col_w)]
    half = (len(lines) + 1) // 2
    lead = round(px * 1.35)
    img = Image.new("L", (PAGE_W, 2 * margin + lead * half), 255)
    d = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        col, row = divmod(i, half)
        d.text((margin + col * (col_w + gutter), margin + row * lead), ln, font=font, fill=0)
    return _scale(img, 200), "\n".join(lines)


def table(seed: int, font_name: str) -> tuple[Image.Image, str]:
    rng = random.Random(f"table-{seed}")
    rows = [["Name", "Count", "Item", "Share"]]
    for _ in range(9):
        rows.append(
            [
                rng.choice(NAMES),
                str(rng.randint(10, 9999)),
                rng.choice(WORDS),
                f"{rng.randint(0, 99)}.{rng.randint(0, 9)}%",
            ]
        )
    px = round(10 * DPI / 72)
    font = _font(font_name, px)
    cell_w, cell_h, x0, y0 = 420, round(px * 2.0), 200, 200
    img = Image.new("L", (x0 * 2 + cell_w * 4, y0 * 2 + cell_h * len(rows)), 255)
    d = ImageDraw.Draw(img)
    for r, row in enumerate(rows):
        for c, cell in enumerate(row):
            d.text((x0 + c * cell_w + 24, y0 + r * cell_h + px // 3), cell, font=font, fill=0)
    for r in range(len(rows) + 1):
        d.line([(x0, y0 + r * cell_h), (x0 + 4 * cell_w, y0 + r * cell_h)], fill=0, width=3)
    for c in range(5):
        d.line([(x0 + c * cell_w, y0), (x0 + c * cell_w, y0 + len(rows) * cell_h)], fill=0, width=3)
    return _scale(img, 200), "\n".join(" ".join(row) for row in rows)


def _perspective(img: Image.Image) -> Image.Image:
    w, h = img.size
    pad = Image.new("L", (w + 200, h + 200), 255)
    pad.paste(img, (100, 100))
    w, h = pad.size
    # Source quad (corners read from the page): top edge narrower, as if photographed
    # from below; Pillow maps this quad onto the full output rectangle.
    quad = (-0.16 * w, 0, -0.02 * w, h, 1.02 * w, h, 1.16 * w, 0)
    return pad.transform(
        (w, h), Image.Transform.QUAD, quad, Image.Resampling.BICUBIC, fillcolor=255
    )


def _colour(gray: Image.Image) -> Image.Image:
    return ImageOps.colorize(gray, black=(140, 20, 20), white=(205, 225, 250))


PageCase = Callable[[list[str], str], Image.Image]
SIMPLE: dict[str, PageCase] = {
    "small-8pt-150dpi": lambda paras, font: _scale(render(paras, font, pt=8), 150),
    "inverted": lambda paras, font: ImageOps.invert(_scale(render(paras, font), 200)),
    "colour": lambda paras, font: _colour(_scale(render(paras, font), 200)),
    "perspective": lambda paras, font: _perspective(_scale(render(paras, font), 200)),
    "cheap-scan": lambda paras, font: _jpeg(
        _rotate(_scale(render(paras, font), 120), 1.0).filter(ImageFilter.GaussianBlur(0.9)), 25
    ),
}
CONDITIONS = ["small-8pt-150dpi", "two-columns", "table", *list(SIMPLE)[1:]]


def hard_samples(split: str) -> list[Sample]:
    out = []
    for page in range(PAGES):
        if split != "all" and split_of(page) != split:
            continue
        paras = page_text(page)
        truth = "\n".join(paras)
        for font in FONTS:
            for name, fn in SIMPLE.items():
                out.append(Sample(page, font, name, fn(paras, font), truth))
            img, cols_truth = two_columns(paras, font)
            out.append(Sample(page, font, "two-columns", img, cols_truth))
            img, table_truth = table(page, font)
            out.append(Sample(page, font, "table", img, table_truth))
    return out


__all__ = ["CONDITIONS", "hard_samples"]
