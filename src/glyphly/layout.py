"""Reading order: columns are read one after the other, table rows left to right.

Sorting lines top to bottom interleaves the two columns of a page, and tiny height
differences between the cells of one table row scramble their order. So lines are
ordered by a recursive "XY-cut":

1. find the widest empty gap across the region, vertical (a gutter between columns) or
   horizontal (space between blocks of lines), and split there, recursively;
2. a vertical split is only allowed when both sides read like prose (lines of several
   words): that is what separates two columns of text from the columns of a table,
   whose cells hold a word or two and must be read row by row;
3. what cannot be split further is grouped into rows by vertical overlap and each row
   is read left to right.
"""

from __future__ import annotations

import statistics

from glyphly.types import Line

PROSE_WORDS = 3  # median words per line for a side to count as running text
MIN_GUTTER = 1.5  # a column gutter is at least this many line heights wide


def _gaps(spans: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Empty intervals between the merged spans, as (start, end)."""
    spans = sorted(spans)
    out: list[tuple[float, float]] = []
    reach = spans[0][1]
    for a, b in spans[1:]:
        if a > reach:
            out.append((reach, a))
        reach = max(reach, b)
    return out


def _prose(lines: list[Line]) -> bool:
    return statistics.median(len(ln.words) for ln in lines) >= PROSE_WORDS


def _band(ln: Line) -> tuple[float, float]:
    """The line's own height band around its middle, from its word boxes.

    A tilted line's outer box is much taller than the line (it spans the tilt), and the
    outer boxes of neighbouring tilted lines overlap; the middle word's box does not.
    """
    mid = ln.words[len(ln.words) // 2].box if ln.words else ln.box
    return mid[1], mid[3]


def _rows(lines: list[Line]) -> list[Line]:
    """Group lines whose height bands overlap into rows; each row left to right."""
    rows: list[list[Line]] = []
    for ln in sorted(lines, key=lambda l_: sum(_band(l_)) / 2):
        top, bottom = _band(ln)
        if rows:
            r_top = min(_band(x)[0] for x in rows[-1])
            r_bottom = max(_band(x)[1] for x in rows[-1])
            overlap = min(bottom, r_bottom) - max(top, r_top)
            if overlap > 0.5 * min(bottom - top, r_bottom - r_top):
                rows[-1].append(ln)
                continue
        rows.append([ln])
    return [_merge(sorted(row, key=lambda l_: l_.box[0])) for row in rows]


def _merge(row: list[Line]) -> Line:
    """Side-by-side pieces of one row (table cells) become one line, left to right."""
    if len(row) == 1:
        return row[0]
    words = [w for ln in row for w in ln.words]
    box = (
        min(ln.box[0] for ln in row),
        min(ln.box[1] for ln in row),
        max(ln.box[2] for ln in row),
        max(ln.box[3] for ln in row),
    )
    chars = sum(len(w.text) for w in words)
    conf = sum(w.confidence * len(w.text) for w in words) / chars if chars else 0.0
    return Line(words, box, conf)


def reading_order(lines: list[Line]) -> list[Line]:
    if len(lines) <= 1:
        return list(lines)
    height = statistics.median(ln.box[3] - ln.box[1] for ln in lines)
    best: tuple[float, str, float] | None = None  # (gap width, axis, cut position)
    for a, b in _gaps([(ln.box[0], ln.box[2]) for ln in lines]):
        if b - a < MIN_GUTTER * height:
            continue
        cut = (a + b) / 2
        left = [ln for ln in lines if ln.box[2] <= cut]
        right = [ln for ln in lines if ln.box[0] >= cut]
        if _prose(left) and _prose(right) and (best is None or b - a > best[0]):
            best = (b - a, "x", cut)
    for a, b in _gaps([(ln.box[1], ln.box[3]) for ln in lines]):
        if best is None or b - a > best[0]:
            best = (b - a, "y", (a + b) / 2)
    if best is None:
        return _rows(lines)
    _, axis, cut = best
    lo = 0 if axis == "x" else 1
    hi = 2 if axis == "x" else 3
    first = [ln for ln in lines if ln.box[hi] <= cut]
    second = [ln for ln in lines if ln.box[lo] >= cut]
    if not first or not second or len(first) + len(second) != len(lines):
        return _rows(lines)
    return reading_order(first) + reading_order(second)


__all__ = ["reading_order"]
