"""hOCR, ALTO and searchable-PDF output, from a page built by hand (no models needed)."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zlib

from PIL import Image

from glyphly.export import to_alto, to_hocr, to_pdf
from glyphly.types import Line, Page, Word


def _page() -> Page:
    words = [
        Word("Grüße", (10.0, 20.0, 110.0, 50.0), 0.98),
        Word("<&>", (120.0, 20.0, 170.0, 50.0), 0.5),
    ]
    mal = [Word("മലയാളം", (10.0, 70.0, 200.0, 110.0), 0.9)]
    return Page([Line(words, (10, 20, 170, 50), 0.9), Line(mal, (10, 70, 200, 110), 0.9)], 400, 200)


def test_hocr_has_words_with_boxes_and_confidence() -> None:
    doc = ET.fromstring(to_hocr(_page(), images=["scan.png"]).encode("utf-8"))
    ns = {"h": "http://www.w3.org/1999/xhtml"}
    page = doc.find(".//h:div[@class='ocr_page']", ns)
    assert page is not None
    assert 'image "scan.png"' in page.get("title", "")
    assert "bbox 0 0 400 200" in page.get("title", "")
    words = doc.findall(".//h:span[@class='ocrx_word']", ns)
    assert [w.text for w in words] == ["Grüße", "<&>", "മലയാളം"]
    assert words[0].get("title") == "bbox 10 20 110 50; x_wconf 98"


def test_alto_has_strings_positions_and_confidence() -> None:
    doc = ET.fromstring(to_alto([_page(), _page()]).encode("utf-8"))
    ns = {"a": "http://www.loc.gov/standards/alto/ns-v4#"}
    assert len(doc.findall(".//a:Page", ns)) == 2
    strings = doc.findall(".//a:String", ns)
    assert strings[1].get("CONTENT") == "<&>"
    first = strings[0]
    assert (first.get("HPOS"), first.get("VPOS"), first.get("WIDTH"), first.get("HEIGHT")) == (
        "10", "20", "100", "30",
    )  # fmt: skip
    assert first.get("WC") == "0.98"
    assert len(doc.findall(".//a:SP", ns)) == 2  # between the words of each page's first line


def _objects(pdf: bytes) -> dict[int, bytes]:
    """Objects found at the offsets the cross-reference table gives."""
    xref = int(pdf.rsplit(b"startxref", 1)[1].split()[0])
    table = pdf[xref:].split(b"trailer")[0].split(b"\n")[2:]
    out = {}
    for n, row in enumerate(t for t in table if t.strip()):
        if n == 0:
            continue  # the free-list head
        offset = int(row.split()[0])
        assert pdf[offset:].startswith(f"{n} 0 obj".encode()), f"object {n} not at its offset"
        out[n] = pdf[offset : pdf.index(b"endobj", offset)]
    return out


def test_pdf_has_image_invisible_text_and_valid_xref() -> None:
    image = Image.new("RGB", (400, 200), "white")
    pdf = to_pdf([(image, _page()), (image, _page())], dpi=200)
    assert pdf.startswith(b"%PDF-1.5")
    assert pdf.rstrip().endswith(b"%%EOF")
    objs = _objects(pdf)
    pages = [o for o in objs.values() if b"/Type /Page " in o]
    assert len(pages) == 2
    assert b"/MediaBox [0 0 144.00 72.00]" in pages[0]  # 400 x 200 px at 200 dpi
    assert any(b"/FontFile2" in o for o in objs.values())  # the invisible font is embedded
    contents = [
        zlib.decompress(o.split(b"stream\n", 1)[1].rsplit(b"\nendstream", 1)[0])
        for o in objs.values()
        if b"/FlateDecode" in o
    ]
    text = contents[0].decode("ascii")
    assert "3 Tr" in text  # invisible
    hexes = re.findall(r"<([0-9A-F]+)> Tj", text)
    decoded = [bytes.fromhex(h).decode("utf-16-be") for h in hexes]
    assert decoded == ["Grüße", "<&>", "മലയാളം"]


def test_pdf_turns_the_image_like_the_page() -> None:
    page = _page()
    page.rotation = 90
    pdf = to_pdf([(Image.new("RGB", (200, 400), "white"), page)], dpi=72)
    assert b"/Width 400 /Height 200" in pdf  # turned to match the boxes
