"""Make the invisible font that glyphly's searchable PDFs embed (src/glyphly/export.py).

Needs fontTools (MIT), which glyphly itself does not depend on:

    uv run --with fonttools python tools/make_glyphless_font.py   -> base64 on stdout

Two empty glyphs, 500 units wide on a 1000-unit em: every character drawn with it is
invisible and takes the same width, which the PDF writer stretches to each word's box.
"""

from __future__ import annotations

import base64
import io

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen


def main() -> None:
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder([".notdef", "space"])
    fb.setupCharacterMap({0x20: "space"})
    empty = TTGlyphPen(None).glyph()
    fb.setupGlyf({".notdef": empty, "space": empty})
    fb.setupHorizontalMetrics({".notdef": (500, 0), "space": (500, 0)})
    fb.setupHorizontalHeader(ascent=1000, descent=0)
    fb.setupNameTable({"familyName": "GlyphLessFont", "styleName": "Regular"})
    fb.setupOS2(sTypoAscender=1000, sTypoDescender=0, usWinAscent=1000, usWinDescent=0)
    fb.setupPost()
    buf = io.BytesIO()
    fb.save(buf)
    print(base64.b64encode(buf.getvalue()).decode())


if __name__ == "__main__":
    main()
