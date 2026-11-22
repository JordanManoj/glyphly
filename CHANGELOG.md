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
