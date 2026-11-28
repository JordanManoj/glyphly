"""Standard OCR output formats: hOCR, ALTO XML and searchable PDF.

* hOCR (https://kba.github.io/hocr-spec/1.2/): HTML with ``ocr_page`` / ``ocr_line`` /
  ``ocrx_word`` elements and their boxes and confidences, as Tesseract writes it.
* ALTO v4 (https://www.loc.gov/standards/alto/): the XML format of libraries and archives.
* Searchable PDF: each page image with its text laid invisibly over the words, so the text
  can be selected, searched and copied.

Boxes refer to the page as it was read: when glyphly turned a page to read it
(``page.rotation``), the image written to the PDF is turned the same way.
"""

from __future__ import annotations

import base64
import io
import zlib
from collections.abc import Sequence
from html import escape
from typing import TYPE_CHECKING

import numpy as np

from glyphly.__about__ import __version__
from glyphly.types import Box, Page

if TYPE_CHECKING:
    from PIL import Image


def _ibox(box: Box) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return round(x0), round(y0), round(x1), round(y1)


def _bbox(box: Box) -> str:
    return "bbox {} {} {} {}".format(*_ibox(box))


# --------------------------------------------------------------------------- hOCR
def to_hocr(pages: Page | Sequence[Page], *, images: Sequence[str] = ()) -> str:
    """hOCR document for one or more pages; ``images`` names each page's image file."""
    pages = [pages] if isinstance(pages, Page) else list(pages)
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" '
        '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">',
        '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">',
        "<head>",
        "<title></title>",
        '<meta http-equiv="Content-Type" content="text/html;charset=utf-8"/>',
        f'<meta name="ocr-system" content="glyphly {__version__}"/>',
        '<meta name="ocr-capabilities" content="ocr_page ocr_line ocrx_word ocrp_wconf"/>',
        "</head>",
        "<body>",
    ]
    for p, page in enumerate(pages, 1):
        title = [_bbox((0, 0, page.width, page.height)), f"ppageno {p - 1}"]
        if p <= len(images):
            title.insert(0, f'image "{escape(images[p - 1])}"')
        if page.rotation:
            title.append(f"textangle {page.rotation}")
        out.append(f"<div class='ocr_page' id='page_{p}' title='{'; '.join(title)}'>")
        for n, line in enumerate(page.lines, 1):
            out.append(
                f"<span class='ocr_line' id='line_{p}_{n}' "
                f"title='{_bbox(line.box)}; x_wconf {round(line.confidence * 100)}'>"
            )
            for k, word in enumerate(line.words, 1):
                out.append(
                    f"<span class='ocrx_word' id='word_{p}_{n}_{k}' "
                    f"title='{_bbox(word.box)}; x_wconf {round(word.confidence * 100)}'>"
                    f"{escape(word.text)}</span>"
                )
            out.append("</span>")
        out.append("</div>")
    out += ["</body>", "</html>", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- ALTO
def _pos(box: Box) -> str:
    x0, y0, x1, y1 = _ibox(box)
    return f'HPOS="{x0}" VPOS="{y0}" WIDTH="{max(0, x1 - x0)}" HEIGHT="{max(0, y1 - y0)}"'


def to_alto(pages: Page | Sequence[Page], *, images: Sequence[str] = ()) -> str:
    """ALTO v4 XML for one or more pages (one TextBlock per line, word confidence as WC)."""
    pages = [pages] if isinstance(pages, Page) else list(pages)
    src = escape(images[0]) if images else ""
    out = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xsi:schemaLocation="http://www.loc.gov/standards/alto/ns-v4# '
        'http://www.loc.gov/alto/v4/alto-4-2.xsd">',
        "<Description>",
        "<MeasurementUnit>pixel</MeasurementUnit>",
        f"<sourceImageInformation><fileName>{src}</fileName></sourceImageInformation>",
        '<OCRProcessing ID="OCR_0"><ocrProcessingStep><processingSoftware>'
        f"<softwareName>glyphly</softwareName><softwareVersion>{__version__}"
        "</softwareVersion></processingSoftware></ocrProcessingStep></OCRProcessing>",
        "</Description>",
        "<Layout>",
    ]
    for p, page in enumerate(pages, 1):
        out.append(
            f'<Page ID="page_{p}" PHYSICAL_IMG_NR="{p}" '
            f'WIDTH="{page.width}" HEIGHT="{page.height}">'
        )
        out.append(f"<PrintSpace {_pos((0, 0, page.width, page.height))}>")
        for n, line in enumerate(page.lines, 1):
            out.append(f'<TextBlock ID="block_{p}_{n}" {_pos(line.box)}>')
            out.append(f'<TextLine ID="line_{p}_{n}" {_pos(line.box)}>')
            for k, word in enumerate(line.words, 1):
                if k > 1:
                    out.append("<SP/>")
                out.append(
                    f'<String ID="string_{p}_{n}_{k}" {_pos(word.box)} '
                    f'WC="{word.confidence:.2f}" CONTENT="{escape(word.text)}"/>'
                )
            out += ["</TextLine>", "</TextBlock>"]
        out += ["</PrintSpace>", "</Page>"]
    out += ["</Layout>", "</alto>", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------- PDF
_EM = 500  # width of every glyph in the invisible font, in 1/1000 of the font size
# An invisible TrueType font (two empty glyphs, 500 units wide), made with fontTools by
# tools/make_glyphless_font.py; embedded so viewers need no font of their own.
_GLYPHLESS = (
    "AAEAAAAKAIAAAwAgT1MvMkT/RUAAAAEoAAAAYGNtYXAADABzAAABkAAAADRnbHlmAAAAAAAAAcwAAAABaGVhZC13"
    "3H4AAACsAAAANmhoZWED6QH2AAAA5AAAACRobXR4AfQAAAAAAYgAAAAGbG9jYQAAAAAAAAHEAAAABm1heHAAAwAC"
    "AAABCAAAACBuYW1lGZ8ZNAAAAdAAAABycG9zdAAHAAAAAAJEAAAAJgABAAAAAQAAgYYiQl8PPPUAAwPoAAAAAOcw"
    "zc8AAAAA5zDNzwAAAAAAAAAAAAAAAwACAAAAAAAAAAEAAAPoAAAAAAH0AAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAB"
    "AAEAAAACAAAAAAAAAAAAAgAAAAAAAAAAAAAAAAAAAAAAAwH0AZAABQAEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAPz8/PwAAACAAIAPoAAAAAAPoAAAAAAAAAAAAAAAAAAAAAAAgAAAB9AAA"
    "AAAAAAAAAAIAAAADAAAAFAADAAEAAAAUAAQAIAAAAAQABAABAAAAIP//AAAAIP///+EAAQAAAAAAAAAAAAAAAAAA"
    "AAAAAAAEADYAAQAAAAAAAQANAAAAAQAAAAAAAgAHAA0AAwABBAkAAQAaABQAAwABBAkAAgAOAC5HbHlwaExlc3NG"
    "b250UmVndWxhcgBHAGwAeQBwAGgATABlAHMAcwBGAG8AbgB0AFIAZQBnAHUAbABhAHIAAAACAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAIAAAADAAA="
)


def _hex16(text: str) -> str:
    """Text as 2-byte character codes (Identity-H, code = Unicode for the BMP)."""
    codes = []
    for ch in text:
        cp = ord(ch)
        codes.append(cp if cp <= 0xFFFF else 0xFFFD)
    return "".join(f"{c:04X}" for c in codes)


def _to_unicode() -> bytes:
    ranges = "\n".join(f"<{h:02X}00> <{h:02X}FF> <{h:02X}00>" for h in range(256))
    cmap = (
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        "/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n"
        "1 begincodespacerange\n<0000> <FFFF>\nendcodespacerange\n"
        f"100 beginbfrange\n{chr(10).join(ranges.splitlines()[:100])}\nendbfrange\n"
        f"100 beginbfrange\n{chr(10).join(ranges.splitlines()[100:200])}\nendbfrange\n"
        f"56 beginbfrange\n{chr(10).join(ranges.splitlines()[200:])}\nendbfrange\n"
        "endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n"
    )
    return cmap.encode("ascii")


def _text_layer(page: Page, scale: float, height_pt: float) -> str:
    """Content-stream operators drawing every word invisibly over its box."""
    ops = ["BT", "3 Tr"]  # text render mode 3: neither filled nor stroked (invisible)
    for word in page.words:
        text = word.text.strip()
        if not text:
            continue
        x0, y0, x1, y1 = word.box
        h = max(1.0, (y1 - y0) * scale)
        w = max(1.0, (x1 - x0) * scale)
        size = h * 0.85
        natural = len(text) * size * _EM / 1000
        stretch = 100.0 * w / natural if natural else 100.0
        baseline = height_pt - y1 * scale + h * 0.15
        ops.append(
            f"/F1 {size:.2f} Tf {stretch:.2f} Tz "
            f"1 0 0 1 {x0 * scale:.2f} {baseline:.2f} Tm <{_hex16(text)}> Tj"
        )
    ops.append("ET")
    return "\n".join(ops)


def to_pdf(
    pages: Sequence[tuple[Image.Image, Page]], *, dpi: float | None = None, quality: int = 85
) -> bytes:
    """A searchable PDF: each image with its page's text invisible over the words.

    ``dpi`` sets the printed size (default: the image's own DPI, else 300).
    """
    objs: list[bytes] = []

    def add(body: bytes) -> int:
        objs.append(body)
        return len(objs)

    def stream(head: str, data: bytes) -> bytes:
        return f"<< {head} /Length {len(data)} >>\nstream\n".encode() + data + b"\nendstream"

    catalog = add(b"")  # filled in at the end
    tree = add(b"")
    info = add(f"<< /Producer (glyphly {__version__}) >>".encode())
    to_unicode = add(stream("", _to_unicode()))
    ttf = base64.b64decode(_GLYPHLESS)
    font_file = add(stream(f"/Length1 {len(ttf)}", ttf))
    descriptor = add(
        f"<< /Type /FontDescriptor /FontName /GlyphLessFont /Flags 5 "
        f"/FontBBox [0 0 500 1000] /ItalicAngle 0 /Ascent 1000 /Descent 0 "
        f"/CapHeight 1000 /StemV 80 /FontFile2 {font_file} 0 R >>".encode()
    )
    cid = add(
        f"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /GlyphLessFont "
        f"/CIDSystemInfo << /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> "
        f"/FontDescriptor {descriptor} 0 R /DW {_EM} /CIDToGIDMap /Identity >>".encode()
    )
    font = add(
        f"<< /Type /Font /Subtype /Type0 /BaseFont /GlyphLessFont /Encoding /Identity-H "
        f"/DescendantFonts [{cid} 0 R] /ToUnicode {to_unicode} 0 R >>".encode()
    )
    kids = []
    for image, page in pages:
        img = image.convert("RGB")
        if page.rotation:  # the boxes refer to the page as it was read
            img = img.rotate(page.rotation, expand=True)
        if img.size != (page.width, page.height):
            img = img.resize((page.width, page.height))
        own = image.info.get("dpi")
        res = dpi or (float(own[0]) if own and own[0] and own[0] > 1 else 300.0)
        scale = 72.0 / res
        wpt, hpt = page.width * scale, page.height * scale
        small = np.asarray(img.resize((32, 32)))
        grey = bool(
            (small[..., 0] == small[..., 1]).all() and (small[..., 1] == small[..., 2]).all()
        )
        buf = io.BytesIO()
        (img.convert("L") if grey else img).save(buf, "JPEG", quality=quality)
        space = "/DeviceGray" if grey else "/DeviceRGB"
        xobj = add(
            stream(
                f"/Type /XObject /Subtype /Image /Width {img.width} /Height {img.height} "
                f"/ColorSpace {space} /BitsPerComponent 8 /Filter /DCTDecode",
                buf.getvalue(),
            )
        )
        content = (
            f"q {wpt:.2f} 0 0 {hpt:.2f} 0 0 cm /Im0 Do Q\n" + _text_layer(page, scale, hpt)
        ).encode("utf-8")
        body = add(stream("/Filter /FlateDecode", zlib.compress(content)))
        kids.append(
            add(
                f"<< /Type /Page /Parent {tree} 0 R /MediaBox [0 0 {wpt:.2f} {hpt:.2f}] "
                f"/Resources << /XObject << /Im0 {xobj} 0 R >> /Font << /F1 {font} 0 R >> >> "
                f"/Contents {body} 0 R >>".encode()
            )
        )
    objs[catalog - 1] = f"<< /Type /Catalog /Pages {tree} 0 R >>".encode()
    objs[tree - 1] = (
        f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>"
    ).encode()
    out = io.BytesIO()
    out.write(b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for n, body in enumerate(objs, 1):
        offsets.append(out.tell())
        out.write(f"{n} 0 obj\n".encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write(f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode())
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode())
    out.write(
        f"trailer\n<< /Size {len(objs) + 1} /Root {catalog} 0 R /Info {info} 0 R >>\n"
        f"startxref\n{xref}\n%%EOF\n".encode()
    )
    return out.getvalue()


__all__ = ["to_alto", "to_hocr", "to_pdf"]
