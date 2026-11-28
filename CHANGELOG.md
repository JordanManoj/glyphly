# Changelog

All notable changes are listed here. Versions follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Script identifier: glyphly's own 1.1 MB network tells which of 25 writing systems a page's
  lines are in (94.2% of lines right on real scans it never saw). Auto mode now uses the
  Malayalam, Bengali, Assamese, Kannada, Gujarati, Gurmukhi and Odia word models without
  being told the language, and reports scripts it cannot read yet (Hebrew, Sinhala,
  Georgian, Armenian, Khmer, Lao, Myanmar, Ethiopic) in `page.steps`. Downloaded with the
  default models (`glyphly download`). Training data and training: `tools/script_id_data.py`,
  `tools/train_script_id.py`.
- Real-scan benchmark engine `glyphly-auto` (no language named).

## [0.2.0a2] - 2026-11-27

### Fixed

- Lines read by a word model report calibrated confidence (x 0.8): the models give ~0.95
  while 12-28% of characters are wrong on real scans, so a Malayalam newspaper photo
  scored 0.97 and looked reliable.

## [0.2.0a1] - 2026-11-27

### Added

- Seven more Indian scripts: Malayalam, Bengali, Assamese, Kannada, Gujarati, Gurmukhi
  (Punjabi) and Odia, read word by word with IndicPhotoOCR's PARSeq recognizers converted to
  ONNX int8 (`tools/convert_parseq.py`). On real Malayalam book scans (Wikisource) the median
  character error goes from about 92% to 26%.
- Word models are used for the languages you name (`lang="ml"`, `lang=["auto", "bn"]`,
  `--lang ml`); auto mode does not guess them, because a word model reads any script with
  high confidence (Malayalam's reads Hebrew, Arabic and English lines at 0.84-0.97). Among
  named word scripts the one with the highest mean confidence wins; at most two word models
  stay loaded.
- ONNX Runtime's graph optimisations are saved next to each model on first load (word
  models load in 0.7 s instead of 3 s).
- Real-scan benchmark (`bench/real.py`): validated Wikisource pages in 34 languages with
  human-proofread text; resumable runs; blank and mismatched pages are rejected.

### Fixed

- A page is turned upside down only when the turned reading is confident: a Malayalam
  newspaper photo (no Malayalam model yet) was flipped because noise read slightly longer
  the other way up.
- Pages read with low confidence (below 0.8) say so in `Page.steps`, naming the likely
  causes, including a script glyphly cannot read yet.

## [0.1.0a1] - 2026-11-25

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
- Page orientation: sideways (90/270 degrees) and upside-down pages are turned upright
  before reading (`Page.rotation`); a turned test page went from 73-85% wrong to 0%.
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
