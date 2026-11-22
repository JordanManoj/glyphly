# glyphly

**OCR for Python with nothing to install but the package.** Reads scans, photos and
images into text, with every word's position and confidence. No Tesseract, no OpenCV, no
system programs: `pip install glyphly` brings NumPy, Pillow and ONNX Runtime, all as
wheels for Windows, macOS and Linux.

> **Pre-alpha.** The engine works and is measured, but the API may change and it is not on
> PyPI yet. Feedback is very welcome.

```python
import glyphly

page = glyphly.read("scan.png")  # path, bytes, PIL image or NumPy array
print(page.text)  # lines in reading order
print(page.confidence)  # 0..1, character-weighted
for word in page.lines[0].words:
    print(word.text, word.box, word.confidence)  # box = (x0, top, x1, bottom) in pixels
```

```bash
glyphly read scan.png            # text
glyphly read scan.png --json     # lines, words, boxes, confidences
glyphly download                 # fetch the model files now, for offline use later
```

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

Measured with `bench/ocr.py`: pages rendered from known text in 5 fonts, then degraded
like real scans (fax resolution, blur, noise, 3° skew, heavy JPEG, faint print with a
shadow, a phone photo), so the ground truth is exact. Development split, character error
rate (lower is better), first run:

| | glyphly | gleanly + Tesseract | RapidOCR (same models) | plain Tesseract |
| --- | --- | --- | --- | --- |
| Times, Arial, Courier, 8 conditions | ~0.05%, worst page 0.3% | 0.6–0.8% | 4.3% | 16% |

The held-out test split and the harder page types (tables, two columns, white on dark,
small print) are being measured next; results will be listed here. These are generated
pages: real scans and a comparison with paid OCR services are planned before any claim
of "best".

## Status and roadmap

- [x] Engine on ONNX Runtime with no system dependencies; rotated lines; word boxes
- [ ] Held-out and hard-suite results; multi-column reading order; tables
- [ ] Release on PyPI; used by [gleanly](https://github.com/JordanManoj/gleanly) for OCR
- [ ] glyphly's own trained recognition model
- [ ] Real-scan benchmark; comparison with Docling, Marker and a paid OCR service

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
