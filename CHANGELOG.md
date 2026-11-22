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
