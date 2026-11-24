# Changelog

All notable changes are listed here. Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- First engine: PP-OCRv6 small detection and recognition models on ONNX Runtime, with
  glyphly's own post-processing in numpy (no OpenCV): text regions labelled run by run,
  rotated line boxes from pixel moments (skewed scans and phone photos), lines read at full
  width, word boxes from CTC character positions.
- Models downloaded once and verified by SHA-256 (`glyphly download`, `GLYPHLY_HOME`).
- `glyphly read image.png [--json]`, `python -m glyphly`.
- OCR benchmark (generated pages, exact ground truth) taken over from gleanly.
- Reading order: two columns read one after the other, table rows left to right with each
  row on one line (held-out: two columns 69.7% -> 0.0% CER, tables 46.9% -> 0.1%).
- Model downloads try `GLYPHLY_MODEL_URL` (your own mirror), then a GitHub release mirror,
  then ModelScope; every file is checked against the same SHA-256.
- Release workflow: tests, installs the wheel fresh, reads an image, then publishes with
  PyPI trusted publishing.
- Calibrated confidence (`bench/confidence.py`): a long line that decoded to far fewer
  characters than its width holds is scaled down. On 179 pages with real mistakes (heavy
  blur, low resolution, noise, wrong-script readings) every page with more than 20% wrong
  characters now scores below 0.8 (14 of 14, was 9), no well-read page below 0.96, AUROC
  0.997; Russian read with the wrong model 0.97 -> 0.48.
- Languages: script models for Cyrillic, Greek, Arabic, Devanagari, Korean, Thai, Tamil
  and Telugu, chosen automatically per line (or with `lang=`), right-to-left reading order,
  digits restored in Arabic-script lines, `Line.script`, `glyphly download --all`,
  `glyphly read --lang`. Language benchmark (`bench/languages.py`): 20 languages, mean
  CER 1.4%; Hebrew and Bengali not supported yet.
