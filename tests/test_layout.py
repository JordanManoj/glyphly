"""Reading order: two columns, table rows, titles over columns."""

from __future__ import annotations

from glyphly.layout import reading_order
from glyphly.types import Line, Word


def _line(text: str, x0: float, top: float, x1: float, h: float = 20) -> Line:
    words = text.split()
    step = (x1 - x0) / len(words)
    return Line(
        [
            Word(w, (x0 + i * step, top, x0 + (i + 1) * step, top + h), 1.0)
            for i, w in enumerate(words)
        ],
        (x0, top, x1, top + h),
        1.0,
    )


def _texts(lines: list[Line]) -> list[str]:
    return [ln.text for ln in reading_order(lines)]


def test_two_columns_are_read_one_after_the_other() -> None:
    left = [_line(f"left line {i} of prose", 100, 100 + 30 * i, 500) for i in range(4)]
    right = [_line(f"right line {i} of prose", 600, 100 + 30 * i, 1000) for i in range(4)]
    mixed = [ln for pair in zip(right, left, strict=True) for ln in pair]  # scrambled input
    assert _texts(mixed) == [ln.text for ln in left + right]


def test_title_over_two_columns() -> None:
    title = _line("A title that spans both columns of the page", 100, 40, 1000)
    left = [_line(f"left line {i} of prose", 100, 100 + 30 * i, 500) for i in range(3)]
    right = [_line(f"right line {i} of prose", 600, 100 + 30 * i, 1000) for i in range(3)]
    assert _texts([*right, *left, title]) == [title.text] + [ln.text for ln in left + right]


def test_table_is_read_row_by_row_with_cells_joined() -> None:
    cells = []
    for r, row in enumerate([["Name", "Count", "Share"], ["Venn", "96", "5.9%"]]):
        for c, cell in enumerate(row):
            jitter = (c % 2) * 2  # cells of one row are not at exactly the same height
            cells.append(_line(cell, 100 + 300 * c, 100 + 40 * r + jitter, 180 + 300 * c))
    assert _texts(list(reversed(cells))) == ["Name Count Share", "Venn 96 5.9%"]


def test_single_column_keeps_top_to_bottom() -> None:
    lines = [_line(f"line {i} with some words", 100, 100 + 30 * i, 600) for i in range(5)]
    assert _texts(list(reversed(lines))) == [ln.text for ln in lines]


def test_tilted_lines_with_overlapping_outer_boxes_stay_apart() -> None:
    lines = []
    for i in range(3):
        top = 100 + 40 * i
        words = [
            Word(f"w{i}{k}", (100 + 150 * k, top + 6 * k, 240 + 150 * k, top + 6 * k + 25), 1.0)
            for k in range(6)
        ]
        # the outer box spans the 30 px tilt, so neighbouring outer boxes overlap
        lines.append(Line(words, (100, top, 990, top + 55), 1.0))
    assert _texts(list(reversed(lines))) == [ln.text for ln in lines]
