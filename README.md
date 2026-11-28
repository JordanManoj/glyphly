# glyphly

**OCR for Python with nothing to install but the package.** Reads scans, photos and
images into text, with every word's position and confidence. No Tesseract, no OpenCV, no
system programs: `pip install glyphly` brings NumPy, Pillow and ONNX Runtime, all as
wheels for Windows, macOS and Linux.

> **Alpha.** On PyPI (`pip install --pre glyphly`); the engine works and is measured, but
> the API may still change. Feedback is very welcome.

```python
import glyphly

page = glyphly.read("scan.png")  # path, bytes, PIL image or NumPy array
print(page.text)  # lines in reading order
print(page.confidence)  # 0..1, character-weighted
for word in page.lines[0].words:
    print(word.text, word.box, word.confidence)  # box = (x0, top, x1, bottom) in pixels
```

```bash
glyphly read scan.png                         # text
glyphly read scan.png --json                  # lines, words, boxes, confidences
glyphly read p1.jpg p2.jpg -f pdf -o book.pdf # searchable PDF
glyphly read scan.png -f alto -o scan.xml     # ALTO XML (libraries, archives)
glyphly read scan.png -f hocr -o scan.hocr    # hOCR
glyphly download                              # fetch the model files now, for offline use
```

**Output formats.** Besides text and JSON, glyphly writes the standard OCR formats:
searchable PDF (each image with an invisible text layer laid over its words, so the text
can be selected, searched and copied), ALTO v4 XML (valid against the Library of Congress
schema) and hOCR. From Python: `glyphly.export.to_pdf([(image, page), ...])`,
`to_alto(pages)`, `to_hocr(pages)`.

For many images, create one `glyphly.Reader()` and call `reader.read(...)` on each: the
models load once.

## How it works

1. **Find text lines.** A small segmentation network (PP-OCRv6 detection) marks text
   pixels. glyphly's own post-processing, written in NumPy, groups them into lines and
   measures each line's direction, so tilted pages (skewed scans, phone photos) are cut
   out straight instead of slicing through neighbouring lines.
2. **Read each line.** A recognition network (PP-OCRv6) reads each straightened line at
   its full width (no width cap, so long lines and wide fonts are not squeezed), and CTC
   decoding turns its output into characters.
3. **Locate words.** Each character comes from a known column of the line, so every word
   gets its own box on the page.

The model files (31 MB) are downloaded once into `~/.cache/glyphly` (or `GLYPHLY_HOME`) and
checked against a pinned SHA-256 fingerprint before use. They come from PaddleOCR via the
RapidOCR project, both Apache-2.0 (see [NOTICE](NOTICE)). Replacing the recognition model
with one trained by this project is on the roadmap.

## Accuracy

Measured with `bench/ocr.py` on pages rendered from known text in 5 fonts, so the ground
truth is exact. Held-out test split (pages never used while building), character error
rate, lower is better:

| | glyphly | gleanly + Tesseract | RapidOCR (same models) | plain Tesseract |
| --- | --- | --- | --- | --- |
| Standard: clean, fax, blur, noise, 3° skew, JPEG, faint + shadow, phone photo (120 pages) | **under 0.05%** | 0.6% | 5.1% | 16.1% |
| Hard: small print, white on dark, colour, perspective, cheap scan, two columns, table (105 pages) | **0.1%** | 0.0% | 15.1% | 0.5% |

Two columns are read one after the other and tables row by row (each row on one line).
Weak spot left on the hard suite: cheap scans (0.3%). These are generated pages: real
scans and a comparison with docTR, EasyOCR, PaddleOCR and a paid OCR service come next
(see [docs/review.md](docs/review.md)).

## Languages

`glyphly.read()` works out the script by itself: Latin-script languages, Chinese and
Japanese are read by the default model; lines that look misread are re-read with the
matching script model (downloaded the first time a page needs it, about 8 MB each, checked
by SHA-256), right-to-left lines are put in reading order, and numbers skipped by the
Arabic-script model are read back in. Or choose: `Reader(lang="ru")`,
`Reader(lang=["hi", "en"])`, `glyphly read scan.png --lang ar`; `glyphly download --all`
fetches every model for offline use.

Measured with `bench/languages.py` (the same short notice with a date and an amount,
written in each language and drawn by a browser so every script is shaped correctly),
character error rate:

| Script | Languages measured | CER |
| --- | --- | --- |
| Latin | English, German, French, Spanish, Portuguese, Polish, Turkish | 0% |
| Latin | Vietnamese | 5.4% (stacked accents dropped) |
| Chinese, Japanese, Korean | zh, ja, ko | 0% |
| Greek, Thai, Tamil | el, th, ta | 0% |
| Cyrillic | Russian, Ukrainian | 5.0%, 1.2% |
| Devanagari, Telugu | Hindi, Telugu | 1.4%, 4.2% |
| Arabic script | Arabic, Persian | 5.3%, 6.3% |
| **Not supported yet** | Hebrew, Armenian, Georgian (no open model yet) | 83% (Hebrew) |

Average over the 20 languages measured this way: 1.4%. One sentence per language is a
smoke test, not a full benchmark.

**Indian scripts read word by word (new in 0.2).** Malayalam, Bengali, Assamese, Kannada,
Gujarati, Gurmukhi (Punjabi) and Odia have no open line recognizer, so glyphly cuts each
line into words and reads them with IndicPhotoOCR's word recognizers (converted to ONNX,
about 35 MB each, downloaded on first use). Auto mode finds them by itself with
**glyphly's own script identifier** (below); naming the language (`Reader(lang="ml")`,
`--lang ml`) still works and skips the guess. Measured on real scanned book pages from
Wikisource (`bench/real.py`, human-proofread text, 12 pages each, language named), median
character error:

| Language | Before (0.1) | glyphly 0.2 |
| --- | --- | --- |
| Bengali | 87% | 12% |
| Assamese | not readable | 14% |
| Malayalam (many 18th-19th century books) | 92% | 26% |
| Punjabi | not readable | 28% |

Real scans are much harder than drawn text (old typefaces, worn paper, page furniture the
transcriptions leave out), so these numbers are not comparable with the table above.

**Script identifier (new in 0.2.0a3).** A word recognizer reads any script with high
confidence (Malayalam's reads Hebrew and English lines at 0.84-0.97), so confidence cannot
tell which script a page is in. glyphly's own small network (1.1 MB, trained with
`tools/train_script_id.py` on lines drawn from Wikipedia text in 25 scripts plus lines of
real scans) looks at the line images instead; a page's longest lines vote. On lines of
real scanned pages it never saw in training it names the script correctly 94.2% of the time
(Bengali, Kannada, Telugu, Gurmukhi 100%, Latin 98%, Cyrillic 98%, Arabic 94%, Malayalam
92%; weakest: Hebrew 81%, Chinese 83%). It also recognises scripts glyphly cannot read yet
(Hebrew, Sinhala, Georgian, Armenian, Khmer, Lao, Myanmar, Ethiopic) and says so in
`page.steps` instead of returning confident-looking text.

## Status and roadmap

- [x] Engine on ONNX Runtime with no system dependencies; rotated lines; word boxes
- [x] Held-out and hard-suite results; two-column reading order; table rows
- [x] 20 languages in 11 scripts, detected automatically
- [x] Seven more Indian scripts (Malayalam, Bengali, Assamese, Kannada, Gujarati,
      Gurmukhi, Odia), read word by word
- [x] Page orientation (sideways, upside down); real-scan benchmark in 34 languages
- [x] Released on PyPI; [gleanly](https://github.com/JordanManoj/gleanly)'s OCR engine
- [x] Script identifier (glyphly's own model): Indian scripts without naming the language
- [x] Searchable PDF, ALTO XML and hOCR output
- [ ] Head-to-head real-scan comparison with Tesseract, PaddleOCR and EasyOCR (running)
- [ ] Speed: 10-45 s per page on a CPU today
- [ ] Hebrew, Fraktur, Georgian, Armenian, Sinhala, polytonic Greek; better Arabic on old
      prints
- [ ] glyphly's own trained recognition model
- [ ] Column and table reading order on complex layouts (gazettes, newspapers)

## Development

```bash
uv sync                      # package + dev tools
uv run glyphly download      # model files
uv run pytest
uv sync --group bench        # + RapidOCR baseline; gleanly[ocr] and Tesseract optional
uv run python -m bench.ocr --split dev
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
