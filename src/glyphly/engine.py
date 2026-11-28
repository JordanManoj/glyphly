"""The reader: image in, lines of words with boxes and confidences out."""

from __future__ import annotations

import itertools
import math
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from glyphly import models, scriptid
from glyphly.detect import MAX_SIDE, TextBox, detect
from glyphly.layout import reading_order
from glyphly.recognize import Decoded, Span, charset, recognize, word_spans
from glyphly.types import Box, Line, Page, Word
from glyphly.words import WordModel, is_word_model, recognize_words

if TYPE_CHECKING:
    from PIL import Image

ImageLike = Any  # path, bytes, PIL image or H x W x 3 uint8 array


def to_rgb(image: ImageLike) -> np.ndarray:
    """Any supported input as an H x W x 3 uint8 RGB array (transparency on white)."""
    from PIL import Image as PILImage

    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=2)
        return np.ascontiguousarray(arr[:, :, :3].astype(np.uint8))
    if isinstance(image, (str, Path)):
        img: Image.Image = PILImage.open(image)
    elif isinstance(image, (bytes, bytearray)):
        import io

        img = PILImage.open(io.BytesIO(image))
    else:
        img = image
    if img.mode in ("RGBA", "LA", "P"):
        bg = PILImage.new("RGBA", img.size, (255, 255, 255, 255))
        img = PILImage.alpha_composite(bg, img.convert("RGBA"))
    return np.asarray(img.convert("RGB"))


def cut(rgb: np.ndarray, box: TextBox) -> np.ndarray:
    """The line's pixels, straightened: an upright crop, or a rotated resample."""
    if box.angle == 0.0:
        x0, y0, x1, y1 = box.bounds
        h, w = rgb.shape[:2]
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, math.ceil(x1)), min(h, math.ceil(y1))
        return rgb[y0 : max(y0 + 1, y1), x0 : max(x0 + 1, x1)]
    from PIL import Image as PILImage

    out_w, out_h = max(1, round(box.length)), max(1, round(box.thickness))
    c, s = math.cos(box.angle), math.sin(box.angle)
    # output (x, y) -> input: centre + (x - W/2) * (c, s) + (y - H/2) * (-s, c)
    a, b = c, -s
    d, e = s, c
    off_x = box.cx - out_w / 2 * c + out_h / 2 * s
    off_y = box.cy - out_w / 2 * s - out_h / 2 * c
    img = PILImage.fromarray(rgb).transform(
        (out_w, out_h),
        PILImage.Transform.AFFINE,
        (a, b, off_x, d, e, off_y),
        resample=PILImage.Resampling.BILINEAR,
        fillcolor=(255, 255, 255),
    )
    return np.asarray(img)


def _word_box(box: TextBox, span: Span) -> Box:
    """Page-aligned bounds of a word that runs from u0 to u1 along a (rotated) line."""
    c, s = math.cos(box.angle), math.sin(box.angle)
    ht = box.thickness / 2
    pts = []
    for u in (span.u0, span.u1):
        px, py = box.point(u)
        pts += [(px - ht * -s, py - ht * c), (px + ht * -s, py + ht * c)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


ReadFn = Callable[[list[np.ndarray]], list[tuple[Decoded, float]]]


def _reader(session: Any) -> ReadFn:
    """Line crops -> readings, for a line recognizer (CTC) or a word recognizer (PARSeq)."""
    if is_word_model(session):
        model = WordModel(session)
        return lambda crops: recognize_words(model, crops)
    chars = charset(session)
    return lambda crops: recognize(session, chars, crops)


def _session(path: Path) -> Any:
    """An ONNX Runtime session. The portable graph optimisations (constant folding, node
    fusion) are saved next to the model the first time (per ONNX Runtime version), so
    later loads skip them: a word model loads in 0.6 s instead of 3 s. CPU-specific layout
    changes are never saved (a saved NCHWc layout failed on other machines); they are
    redone at each load, which is cheap."""
    import onnxruntime as ort

    def options() -> Any:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        return opts

    cpu = ["CPUExecutionProvider"]
    optimized = path.with_name(f"{path.stem}.ext-ort{ort.__version__}.onnx")
    if optimized.exists() and optimized.stat().st_mtime >= path.stat().st_mtime:
        try:
            return ort.InferenceSession(str(optimized), options(), providers=cpu)
        except Exception:  # unreadable leftover: rebuild it below
            optimized.unlink(missing_ok=True)
    try:
        opts = options()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        opts.optimized_model_filepath = str(optimized)
        ort.InferenceSession(str(path), opts, providers=cpu)
        return ort.InferenceSession(str(optimized), options(), providers=cpu)
    except Exception:  # e.g. a read-only cache folder: load without saving
        optimized.unlink(missing_ok=True)
        return ort.InferenceSession(str(path), options(), providers=cpu)


# A line looks misread (wrong script) when it decodes to far fewer characters than its
# width holds, or with low confidence. Wide characters (Chinese, Japanese, Korean) count
# double. Measured on bench/languages.py: well-read lines hold about 2.3-2.9 characters
# per line height, wrong-script lines 0-1.6.
SPARSE = 1.8
UNSURE = 0.9
SWITCH = 1.15  # another script's reading must score this much higher to replace a line
SAMPLE_LINES = 6  # suspicious lines used to find which scripts a page needs
ORIENT_LINES = 4  # longest lines read to tell which way up a page is
# A page is turned 180 degrees only when the turned reading is confident. Upside-down
# Latin text turned upright reads at ~0.99; a script no model knows reads as noise
# (~0.5) both ways up and must stay as it is (a Malayalam newspaper photo was flipped).
ORIENT_SURE = 0.8
# Page confidence below this is reported in the steps (bench/confidence.py: pages with
# over 20% wrong characters all scored below 0.8, well-read pages 0.96 or more).
UNSURE = 0.8


def _wide(ch: str) -> bool:
    o = ord(ch)
    return (
        0x1100 <= o <= 0x11FF
        or 0x2E80 <= o <= 0x9FFF
        or 0xAC00 <= o <= 0xD7AF
        or 0xF900 <= o <= 0xFAFF
    )


def _suspicious(dec: Decoded, crop: np.ndarray) -> bool:
    if not dec.chars:
        return True
    units = sum(2 if _wide(c) else 1 for c in dec.chars)
    per_height = units / max(1.0, crop.shape[1] / max(1, crop.shape[0]))
    conf = sum(dec.probs) / len(dec.probs)
    return per_height < SPARSE or conf < UNSURE


WORD_TRY = 0.8  # a line read with lower mean confidence by every line model tries word models
WORD_SURE = 0.9  # the last page's word script reading this surely stops the search
WORD_PROBE_LINES = 2  # unsure lines read by every word model to tell the script
WORD_CACHE = 2  # word models kept loaded between pages
# Word models are overconfident: on real Malayalam, Bengali and Punjabi book scans their mean
# character confidence is ~0.95 while 12-28% of characters are wrong (bench/real.py). Their
# confidences are scaled so such pages land below 0.8, where readers are warned.
WORD_CALIBRATION = 0.8
# The script identifier's page vote (mean probability over the longest lines) needed to
# use the word models of that script, or to report a script glyphly cannot read.
PAGE_SCRIPT_SURE = 0.5
# When the identifier is sure, only scripts with at least this share of the page vote may
# re-read lines: similar scripts (Kannada and Telugu, Bengali and Devanagari) otherwise read
# each other's lines confidently. Low enough to keep a second script of a mixed page.
PAGE_SCRIPT_KEEP = 0.1


def _mean(dec: Decoded) -> float:
    return sum(dec.probs) / len(dec.probs) if dec.probs else 0.0


def _score(dec: Decoded) -> float:
    """How much confidently read text a decoding holds."""
    return sum(dec.probs)


GAP = 2.5  # an empty stretch wider than this many character spacings may hold a number
TIE = 0.95  # a page-script reading within 5% of the default one counts as a tie
# A lone short word ("мост.") is ambiguous on its own; the page's script decides more
# easily there.
SHORT_LINE = 8
TIE_SHORT = 0.6


def _x(dec: Decoded, share: float, k: int) -> float:
    """Position of character ``k`` along its line, as a share of the line's length."""
    return (dec.steps[k] + 0.5) / dec.n_steps / share


# Confidence calibration (bench/confidence.py, review item G8). The network's character
# probabilities rank pages well but stay high on text read with the wrong script model
# (Russian read as a few digits: 0.97). A long line that decoded to far fewer characters
# than its width holds is scaled down by how sparse it is. On the study set, pages with
# more than 20% wrong characters fell below 0.8 in 14 of 14 cases (9 before) while no
# well-read page went below 0.96. Short lines (table cells, single words) are exempt.
DENSE = 2.0  # characters per line height in a normally read line (wide characters x2)
LONG_LINE = 6.0  # only lines at least this many heights long are judged


def _density_factor(words: list[Word], crop: np.ndarray) -> float:
    aspect = crop.shape[1] / max(1, crop.shape[0])
    if aspect < LONG_LINE:
        return 1.0
    text = "".join(w.text for w in words)
    units = sum(2 if _wide(c) else 1 for c in text) + max(0, len(words) - 1)
    return min(1.0, units / aspect / DENSE)


def _ltr_run(ch: str) -> bool:
    return ch.isascii() and (ch.isalnum() or ch in ".,:%/-+")


def logical_order(chars: list[str]) -> list[str]:
    """Right-to-left text read left to right -> reading order, numbers kept left to right."""
    out = chars[::-1]
    i = 0
    while i < len(out):
        if _ltr_run(out[i]):
            j = i
            while j < len(out) and _ltr_run(out[j]):
                j += 1
            out[i:j] = out[i:j][::-1]
            i = j
        else:
            i += 1
    return out


class Reader:
    """Loads the models once; ``read`` can then be called on many images.

    ``lang``: ``"auto"`` (default) reads Latin-script languages, Chinese and Japanese with
    the default model, and re-reads lines that look misread with the other script models
    (Cyrillic, Greek, Arabic, Devanagari, Korean, Thai, Tamil, Telugu), downloading them
    the first time a page needs them. Give language codes (``"ru"``, ``["hi", "en"]``) to
    load just the models those languages need.
    """

    def __init__(
        self,
        *,
        lang: str | list[str] = "auto",
        download: bool = True,
        max_side: int = MAX_SIDE,
    ) -> None:
        self._download = download
        self._det = _session(models.ensure(models.DETECTOR, download=download))
        self._rec = _session(models.ensure(models.RECOGNIZER, download=download))
        self._chars = charset(self._rec)
        self._scripts: dict[str, ReadFn] = {}
        self._missing: set[str] = set()  # script models that could not be loaded
        self._last_word: str | None = None  # word script that won the previous page
        self._disabled: set[str] = set()  # scripts never used (benchmarks compare without)
        self._word_lru: list[str] = []  # loaded word models, least recently used first
        self._sid: scriptid.ScriptId | None = None
        self._sid_tried = False
        self.max_side = max_side
        langs = [lang] if isinstance(lang, str) else list(lang)
        self.auto = "auto" in langs
        self.wanted = sorted({sc for code in langs if (sc := models.script_for(code))})
        for script in self.wanted:
            self._script(script)

    def _word_order(self) -> list[str]:
        """Word scripts to try, the last page's winner first."""
        rest = sorted(models.WORD_SCRIPTS - {self._last_word or ""})
        return [self._last_word, *rest] if self._last_word else rest

    def _keep_word_model(self, script: str) -> None:
        """Word models are large (~125 MB each in memory): keep the WORD_CACHE most
        recently used ones loaded, unload the rest."""
        self._word_lru = [k for k in self._word_lru if k != script] + [script]
        while len(self._word_lru) > WORD_CACHE:
            self._scripts.pop(self._word_lru.pop(0), None)

    def _script(self, script: str) -> ReadFn | None:
        """The recognizer for ``script``, loaded on first use; None if unavailable."""
        if script in self._disabled:
            return None
        if script not in self._scripts:
            try:
                path = models.ensure(models.SCRIPT_MODELS[script], download=self._download)
            except models.ModelError:
                if script in self.wanted:
                    raise
                self._missing.add(script)
                return None
            self._scripts[script] = _reader(_session(path))
        return self._scripts[script]

    def _identify(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """The page's scripts by the script identifier, most likely first (empty if the
        identifier is unavailable or the page has no long lines)."""
        if not self._sid_tried:
            self._sid_tried = True
            try:
                path = models.ensure(models.SCRIPT_ID, download=self._download)
                self._sid = scriptid.ScriptId(_session(path))
            except models.ModelError:
                self._missing.add("script identifier")
        if self._sid is None:
            return []
        return self._sid.page(crops)

    def _reread(
        self,
        crops: list[np.ndarray],
        decs: list[Decoded],
        shares: list[float],
        steps: list[str],
        first: list[tuple[Decoded, float]],
    ) -> list[str]:
        """Script used for each line ("" = default), re-reading lines where another wins."""
        used = [""] * len(crops)
        todo = list(range(len(crops)))
        candidates = list(self.wanted)
        if self.auto:
            todo = [i for i in todo if _suspicious(decs[i], crops[i])]
            if not todo:
                return used
            # find the scripts this page needs: for a sample of its suspicious lines, which
            # script model reads each one best?
            sample = sorted(todo, key=lambda i: -crops[i].shape[1])[:SAMPLE_LINES]
            best: dict[int, tuple[float, str]] = {
                i: (_score(decs[i]) * SWITCH + 1, "") for i in sample
            }
            sure = {i: _mean(decs[i]) for i in sample}  # best mean confidence so far
            word: dict[int, tuple[float, float, str]] = dict.fromkeys(sample, (0.0, 0.0, ""))
            # Stage 1: the small line models. Stage 2: the larger word models, only for
            # sampled lines no line model read confidently (an Arabic or Cyrillic page
            # never loads them; a Malayalam page does).
            line_scripts = [k for k in models.SCRIPT_MODELS if k not in models.WORD_SCRIPTS]
            # Word models read any script confidently (Malayalam's reads Hebrew, Arabic and
            # English lines at 0.84-0.97), so their confidence cannot tell which script a
            # line is in. They are used for languages the caller names, and for the script
            # the script identifier sees on the page.
            votes = self._identify(crops)
            seen, prob = votes[0] if votes else ("", 0.0)
            page_words: set[str] = set()
            if prob >= PAGE_SCRIPT_SURE:
                allowed = {
                    k
                    for c, p in votes
                    if p >= PAGE_SCRIPT_KEEP
                    for k in scriptid.MODELS.get(c, [c])
                }
                line_scripts = [k for k in line_scripts if k in allowed or k in self.wanted]
                found = scriptid.MODELS.get(seen, [seen])
                page_words = {k for k in found if k in models.WORD_SCRIPTS}
                if seen not in scriptid.DEFAULT and not any(
                    k in models.SCRIPT_MODELS for k in found
                ):
                    steps.append(
                        f"the page looks like {scriptid.name(seen)} script ({prob:.0%} sure), "
                        "which glyphly cannot read yet; the text below is unreliable"
                    )
                elif page_words - set(self.wanted):
                    steps.append(f"script identified: {scriptid.name(seen)} ({prob:.0%} sure)")
            named = [k for k in self._word_order() if k in self.wanted or k in page_words]
            for stage, scripts in enumerate((line_scripts, named)):
                lines = sample if stage == 0 else [i for i in sample if sure[i] < WORD_TRY]
                if stage == 1:  # the longest unsure lines are enough to tell the script
                    lines = lines[:WORD_PROBE_LINES]
                if not lines:
                    continue
                for script in scripts:
                    model = self._script(script)
                    if model is None:
                        continue
                    got = model([crops[i] for i in lines])
                    for i, (dec, _) in zip(lines, got, strict=True):
                        if stage == 0:
                            sure[i] = max(sure[i], _mean(dec))
                            if _score(dec) > best[i][0]:
                                best[i] = (_score(dec), script)
                        elif _mean(dec) > word[i][0]:
                            # Word models share one design, so their mean confidences
                            # compare directly; a wrong-script one reads more characters
                            # less surely and would win on summed confidence.
                            word[i] = (_mean(dec), _score(dec), script)
                    if stage == 1:
                        self._keep_word_model(script)
                        # Pages of a document share a language: when the script that won
                        # the last page reads these lines surely, the others are skipped.
                        if script == self._last_word and all(
                            word[i][2] == script and word[i][0] >= WORD_SURE for i in lines
                        ):
                            break
            winners = {w[2] for w in word.values() if w[2]}
            if len(winners) == 1:
                self._last_word = winners.pop()
            for i, (_, score, script) in word.items():
                if script and score > best[i][0]:
                    best[i] = (score, script)
            if self._missing:
                steps.append(
                    f"{len(todo)} lines may be in another script, but these models are not "
                    f"available: {', '.join(sorted(self._missing))} "
                    "(run `glyphly download --all` once while online)"
                )
            wins: dict[str, int] = {}
            for _, script in best.values():
                if script:
                    wins[script] = wins.get(script, 0) + 1
            top = sorted(wins, key=lambda k: -wins[k])[:2]
            candidates = sorted(set(candidates) | set(top))
            # word models are large: keep only those this page uses
            for script in models.WORD_SCRIPTS - set(candidates):
                self._scripts.pop(script, None)
        # read the lines with every candidate script; each line keeps its best reading
        readings: dict[str, list[tuple[Decoded, float]]] = {}
        for script in candidates:
            model = self._script(script)
            if model is not None:
                readings[script] = model([crops[i] for i in todo])
        switched: dict[str, int] = {}
        for k, i in enumerate(todo):
            floor = SWITCH * _score(decs[i]) + 1
            for script, got in readings.items():
                dec, share = got[k]
                if _score(dec) > floor:
                    floor = _score(dec)
                    decs[i], shares[i], used[i] = dec, share, script
            if used[i]:
                switched[used[i]] = switched.get(used[i], 0) + 1
        # Lines that look the same in two scripts ("MOCT." / "мост.") follow the page: when
        # one script reads at least half of the page's lines, the other lines are read with
        # it too and keep that reading unless the default one is clearly better.
        page_script = max(switched, key=lambda k: switched[k]) if switched else ""
        if self.auto and page_script and switched[page_script] * 2 >= len(crops):
            model = self._script(page_script)
            rest = [i for i in range(len(crops)) if not used[i]]
            if model is not None and rest:
                got = model([crops[i] for i in rest])
                for i, (dec, share) in zip(rest, got, strict=True):
                    tie = TIE_SHORT if len(decs[i].chars) <= SHORT_LINE else TIE
                    if dec.chars and _score(dec) >= tie * _score(decs[i]):
                        decs[i], shares[i], used[i] = dec, share, page_script
                        switched[page_script] += 1
        # The Arabic-script model drops digits; the default model reads them well. Put the
        # default reading's numbers back at their place on the line.
        for i, script in enumerate(used):
            if script in models.RTL_SCRIPTS:
                decs[i] = self._fill_gaps(crops[i], decs[i], shares[i])
        for script, n in switched.items():
            steps.append(f"{n} lines read with the {script} model")
        return used

    def _fill_gaps(self, crop: np.ndarray, dec: Decoded, share: float) -> Decoded:
        """Read the empty stretches of a line (in visual order) with the default model.

        The Arabic-script model skips Western digits; the gaps it leaves are cut out and
        read by the default model, which reads numbers well, and only digits (with their
        separators) are put back, between spaces.
        """
        if len(dec.chars) < 2:
            return dec
        xs = [_x(dec, share, k) for k in range(len(dec.chars))]
        spacing = statistics.median(b - a for a, b in itertools.pairwise(xs))
        gaps = [
            (k, a + spacing / 2, b - spacing / 2)
            for k, (a, b) in enumerate(itertools.pairwise(xs))
            if b - a > GAP * spacing
        ]
        if not gaps:
            return dec
        w = crop.shape[1]
        pieces = [crop[:, max(0, int(a * w)) : min(w, int(np.ceil(b * w)))] for _, a, b in gaps]
        got = recognize(self._rec, self._chars, pieces)
        chars, steps, probs = list(dec.chars), list(dec.steps), list(dec.probs)
        for (k, _, _), (piece, _) in sorted(zip(gaps, got, strict=True), key=lambda g: -g[0][0]):
            digits = [
                (c, p) for c, p in zip(piece.chars, piece.probs, strict=True)
                if c.isdigit() or c in ".,%/-"
            ]  # fmt: skip
            if not any(c.isdigit() for c, _ in digits):
                continue
            at = k + 1  # insert after character k (visual order)
            step = steps[k]
            new = [(" ", 1.0), *digits, (" ", 1.0)]
            chars[at:at] = [c for c, _ in new]
            probs[at:at] = [p for _, p in new]
            steps[at:at] = [step] * len(new)
        keep = [i for i in range(len(chars)) if not (chars[i] == " " and i and chars[i - 1] == " ")]
        return Decoded(
            [chars[i] for i in keep],
            [steps[i] for i in keep],
            [probs[i] for i in keep],
            dec.n_steps,
        )

    def _sample(
        self, rgb: np.ndarray, boxes: list[TextBox], flip: bool = False
    ) -> tuple[float, float]:
        """Score and mean character confidence of the page's longest lines (optionally
        turned 180 degrees)."""
        longest = sorted(boxes, key=lambda b: -b.length)[:ORIENT_LINES]
        crops = [cut(rgb, b) for b in longest]
        if flip:
            crops = [np.ascontiguousarray(c[::-1, ::-1]) for c in crops]
        probs = [p for d, _ in recognize(self._rec, self._chars, crops) for p in d.probs]
        return sum(probs), (sum(probs) / len(probs) if probs else 0.0)

    def _orient(
        self, rgb: np.ndarray, boxes: list[TextBox]
    ) -> tuple[np.ndarray, list[TextBox], int]:
        """Turn a sideways or upside-down page upright: (image, boxes, degrees turned).

        Sideways pages show up as text lines running vertically; the page is turned 90
        degrees both ways and the reading that scores higher wins. Upside-down pages read
        better turned: the longest lines are read both ways up.
        """
        if not boxes:
            return rgb, boxes, 0
        vertical = sum(abs(abs(b.angle) - math.pi / 2) < math.radians(20) for b in boxes)
        if vertical * 2 > len(boxes):
            best = (-1.0, rgb, boxes, 0)
            for k, degrees in ((1, 90), (3, 270)):  # np.rot90 turns counter-clockwise
                turned = np.ascontiguousarray(np.rot90(rgb, k))
                found = detect(self._det, turned, max_side=self.max_side)
                score, _ = self._sample(turned, found)
                if score > best[0]:
                    best = (score, turned, found, degrees)
            _, rgb, boxes, degrees = best
        else:
            degrees = 0
        flipped, sure = self._sample(rgb, boxes, flip=True)
        if sure >= ORIENT_SURE and flipped > SWITCH * self._sample(rgb, boxes)[0] + 1:
            rgb = np.ascontiguousarray(rgb[::-1, ::-1])
            boxes = detect(self._det, rgb, max_side=self.max_side)
            degrees = (degrees + 180) % 360
        return rgb, boxes, degrees

    def read(self, image: ImageLike) -> Page:
        rgb = to_rgb(image)
        boxes = detect(self._det, rgb, max_side=self.max_side)
        rgb, boxes, rotation = self._orient(rgb, boxes)
        h, w = rgb.shape[:2]
        tilted = sum(b.angle != 0.0 for b in boxes)
        steps = [f"{len(boxes)} text lines found" + (f", {tilted} tilted" if tilted else "")]
        if rotation:
            steps.insert(
                0, f"page turned {rotation} degrees to read it; boxes refer to the turned page"
            )
        boxes.sort(key=lambda b: (b.cy, b.cx))
        crops = [cut(rgb, b) for b in boxes]
        first = recognize(self._rec, self._chars, crops)
        decs = [d for d, _ in first]
        shares = [s for _, s in first]
        used = self._reread(crops, decs, shares, steps, first)
        lines: list[Line] = []
        for box, crop, dec, share, script in zip(boxes, crops, decs, shares, used, strict=True):
            length = float(crop.shape[1]) if box.angle == 0.0 else box.length
            spans = word_spans(dec, length / share, length)
            if not spans:
                continue
            words = [Word(sp.text, _word_box(box, sp), sp.confidence) for sp in spans]
            if box.angle == 0.0:  # upright crop: the crop's own left edge is u = 0
                x0 = max(0.0, box.bounds[0])
                words = [
                    Word(wd.text, (x0 + sp.u0, wd.box[1], x0 + sp.u1, wd.box[3]), wd.confidence)
                    for wd, sp in zip(words, spans, strict=True)
                ]
            if script in models.RTL_SCRIPTS:  # reading order: right to left
                words = [
                    Word("".join(logical_order(list(wd.text))), wd.box, wd.confidence)
                    for wd in reversed(words)
                ]
            factor = _density_factor(words, crop)
            if script in models.WORD_SCRIPTS:
                factor *= WORD_CALIBRATION
            if factor < 1.0:
                words = [Word(wd.text, wd.box, wd.confidence * factor) for wd in words]
            chars = sum(len(wd.text) for wd in words)
            conf = sum(wd.confidence * len(wd.text) for wd in words) / chars
            lines.append(Line(words, box.bounds, conf, script or "default"))
        page = Page(reading_order(lines), w, h, steps, rotation)
        conf = page.confidence
        if conf is not None and conf < UNSURE:
            steps.append(
                f"low confidence ({conf:.2f}): the image may be blurred or too small, or the "
                "text may be in a script glyphly does not read well (naming the language, "
                "e.g. lang='ml', helps when the script was not identified)"
            )
        return page


_default: Reader | None = None


def read(image: ImageLike) -> Page:
    """OCR one image with a shared default :class:`Reader`."""
    global _default
    if _default is None:
        _default = Reader()
    return _default.read(image)


__all__ = ["Reader", "cut", "logical_order", "read", "to_rgb"]
