# Honest review

An unsugared scorecard and every known weakness, each with the work that fixes it and
how we know it is fixed. Re-checked at every milestone.

## Scorecard (2026-11-22, first version)

| Viewpoint | Score | Main reason |
| --- | --- | --- |
| AI engineer | 55 / 100 | Clean, dependency-free engine with a measured edge over RapidOCR (same models: 5.1% -> under 0.05% held-out); but synthetic benchmark only, layout fails, confidence uninformative |
| User with real workloads | 25 / 100 | Private, not on PyPI; ~3 s per page; one model host |
| CEO deciding to adopt | 30 / 100 | One maintainer, one week old, no users; recognition model is not ours (PaddleOCR, Apache-2.0) |
| **Overall** | **~37 / 100** | Promising engine, not yet a product |

## Weaknesses and the work that fixes them

### Accuracy and layout

| # | Weakness | Work | Done when | Status |
| --- | --- | --- | --- | --- |
| G1 | Reading order: two columns 69.7% CER, tables 46.9% (held-out hard suite; gleanly + Tesseract 0%) | XY-cut reading order: columns only between prose, table rows by overlap | Hard suite held-out at gleanly's level | done (2026-11-23: two columns 69.7% -> 0.0%, tables 46.9% -> 0.1%, hard mean 16.7% -> 0.1% held-out) |
| G2 | Benchmark too easy: generated English pages, 5 fonts; glyphly scores ~0% so it cannot guide work | Real-world suite: license-clean scans, receipts, forms, phone photos, multi-column pages, a few non-English pages | Suite in bench/ with published results | todo |
| G3 | Not compared with the strongest free and paid engines | docTR, EasyOCR, PaddleOCR, Tesseract, one paid OCR on the shared sample | Comparison table in README | todo |
| G4 | No page orientation (90/180/270 degrees), vertical or curved text | Orientation check (try rotations on a few lines, keep the most confident) | Rotated pages read like upright ones | todo |
| G5 | No table structure or paragraphs in the output | Rows/cells from the ruling lines and cell alignment; paragraph blocks | Tables come out as rows of cells | todo |
| G6 | Languages beyond English untested (the model knows 18,708 characters) | Non-English pages in the benchmark; a language note in the README | Results per script published | done v0 (2026-11-24: 20 languages / 11 scripts, mean CER 1.4%; Hebrew, Bengali unsupported; one sentence per language, real documents still to do) |
| G7 | Word boxes are approximate (from CTC positions), never checked | Benchmark word boxes against the rendered truth | Box accuracy (IoU) reported | todo |

### Trust

| # | Weakness | Work | Done when | Status |
| --- | --- | --- | --- | --- |
| G8 | Confidence does not predict errors (Spearman 0.10 vs 0.50 for gleanly + Tesseract) | Per-character minimum, low-probability character count, detection score, agreement checks | Spearman >= 0.6 on the benchmark | todo |

### Speed and availability

| # | Weakness | Work | Done when | Status |
| --- | --- | --- | --- | --- |
| G9 | ~3 s per page on one CPU core (gleanly + Tesseract ~1.1 s) | Batch lines across pages, use all cores, smaller detector input, quantised model | Under 1 s per page, accuracy unchanged | todo |
| G10 | One model host (ModelScope); an outage blocks new installs | Mirror the files (GitHub Releases), try mirrors in order, same SHA-256 | Install works with ModelScope blocked | code done (own mirror via GLYPHLY_MODEL_URL, GitHub, ModelScope); GitHub copy goes live with the public release |
| G11 | Not installable: private repo, no PyPI release, no user guide | Release workflow (trusted publishing), guide, public repo | `pip install glyphly` works for a stranger | todo |
| G12 | gleanly still needs Tesseract for OCR | gleanly uses glyphly when installed, Tesseract stays optional | gleanly's OCR tests pass with glyphly alone | todo (after G1) |

### Ownership

| # | Weakness | Work | Done when | Status |
| --- | --- | --- | --- | --- |
| G13 | The recognition model is PaddleOCR's, not ours | Train our own recognizer, only where the real-world suite (G2) shows the current one fails | Our model beats it held-out | todo (after G2) |
| G14 | One maintainer, no users | Release, share the video, collect 5-10 outside users | Issues from people other than the maintainer | todo |

## Order of work

G1 -> G10 + G11 (release) -> G12 (gleanly switches) -> G2 + G3 (prove it) -> G8, G4, G9 ->
G5-G7 -> G13. After G12 the main project (gleanly) takes priority again; the rest runs
alongside.

## Review log

- **2026-11-22:** first version. Standard held-out suite: under 0.05% CER (gleanly +
  Tesseract 0.6%, RapidOCR 5.1%, plain Tesseract 16.1%). Hard suite: 0% on small print,
  white on dark, colour, perspective; 0.3% cheap scan; two columns 69.7% and tables 46.9%
  (reading order), which blocks gleanly from switching.
- **2026-11-23:** G1 done: XY-cut reading order (columns only between prose, table rows
  by word-height overlap, tilted lines kept apart). Held-out hard suite 0.1% (two columns
  0.0%, tables 0.1%); tilted standard pages unchanged (0.02%). G10 mirror support written.
- **2026-11-24:** languages (G6). Default model alone: 9 of 22 languages readable (mean CER
  43.5%). Added 8 script models chosen per line by score, page-script ties, right-to-left
  order and digit recovery: 20 of 22 (mean 1.4% on those). Found: confidence stays high on
  wrong-script text (Russian read as digits at 0.97), more evidence for G8.
