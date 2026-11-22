"""OCR benchmark: pages rendered from known text, then degraded like real scans.

uv run python -m bench.ocr --split dev               # glyphly and every baseline installed
uv run python -m bench.ocr --split dev -v glyphly    # one variant
uv run python -m bench.ocr --suite hard --split test # held-out, hard page types

Baselines are used when present: plain Tesseract (the ``tesseract`` program), gleanly's
Tesseract pipeline (``gleanly[ocr]``) and RapidOCR (``rapidocr``); install them with
``uv sync --group bench``. Taken over from gleanly's OCR benchmark (same author, Apache-2.0).

Everything is generated here (our own word list, system fonts), so there is no
third-party data and the ground truth is exact. Metric: character error rate (CER,
edit distance / reference length, whitespace collapsed) and word error rate (WER).
Also reports how well OCR confidence predicts errors (Spearman, as for extraction).
"""

from __future__ import annotations

import argparse
import random
import re
import statistics
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

# Our own vocabulary: common English words, so Tesseract's dictionary helps as it would
# on real text, mixed with numbers, names and punctuation that it must read exactly.
WORDS = """
the of and to in a is that for it as was with be by on not he this are or his from at
which but have an they you were her she there been one all we their has would when
if so no what up out who them some could more time into only other new also about
two may then first any like now my such make over our even most made after many
before must through back years where much way well down should because each just
those people how too little state good very world still own see men work long here
get both between life being under never day same another know while last might us
great old year off come since against go came right used take three himself few house
use during without again place around however home small found thought went say part
once general high upon school every does got united left number course war until always
away something fact though water less public put think almost hand enough far took head
yet government system better set told nothing night end why called did eyes find going
look asked later knew point next city business give group toward young let room
president side social given present several order national possible rather second face
per among form important often things looked early white case john become large big
need four within felt along children saw best church ever least power development light
thing seemed family interest want members others mind country area done turned although
open god service certain kind problem began different door thus help means sense whole
matter perhaps itself york times law human line above name example action company hands
local show whether five history gave today either act feet across taken past quite
anything having seen death experience body word half really week car field money words
already information tell college shall together
harbour council budget river sensor bridge winter report quarter policy station market
""".split()  # noqa: SIM905 (kept as prose: easier to read and edit)
NAMES = ["Hollins", "Ashwater", "Marsh", "Okafor", "Venn", "Northpine", "Kettle", "Dara"]
FONTS = {
    "times": "times.ttf",
    "arial": "arial.ttf",
    "georgia": "georgia.ttf",
    "verdana": "verdana.ttf",
    "courier": "cour.ttf",
}
FONT_DIRS = [Path("C:/Windows/Fonts"), Path("/usr/share/fonts/truetype/msttcorefonts"),
             Path("/Library/Fonts"), Path("/System/Library/Fonts/Supplemental")]  # fmt: skip
DPI = 300
PAGE_W = 2480  # A4 width at 300 dpi
PAGES = 9  # page texts: every third one is held out for testing


def _sentence(rng: random.Random) -> str:
    n = rng.randint(7, 18)
    words = []
    for i in range(n):
        r = rng.random()
        if r < 0.05:
            w = rng.choice(NAMES)
        elif r < 0.10:
            w = rng.choice([str(rng.randint(2, 999)), f"{rng.randint(1, 99)}.{rng.randint(0, 9)}",
                            f"{rng.randint(1900, 2030)}", f"{rng.randint(1, 100)}%"])  # fmt: skip
        else:
            w = rng.choice(WORDS)
        if i and i < n - 1 and rng.random() < 0.08:
            w += ","
        words.append(w)
    words[0] = words[0][:1].upper() + words[0][1:]
    return " ".join(words) + rng.choice([".", ".", ".", "?", ";" if False else "."])


def page_text(seed: int) -> list[str]:
    """Three paragraphs of 3-5 sentences."""
    rng = random.Random(seed)
    return [" ".join(_sentence(rng) for _ in range(rng.randint(3, 5))) for _ in range(3)]


def _font(name: str, px: int) -> ImageFont.FreeTypeFont:
    for d in FONT_DIRS:
        p = d / FONTS[name]
        if p.exists():
            return ImageFont.truetype(str(p), px)
    raise SystemExit(f"font {FONTS[name]} not found; the OCR benchmark needs common fonts")


def render(paragraphs: list[str], font_name: str, pt: float = 11) -> Image.Image:
    px = round(pt * DPI / 72)
    font = _font(font_name, px)
    margin, width = 236, PAGE_W - 2 * 236
    lines: list[str | None] = []
    for para in paragraphs:
        cur = ""
        for w in para.split():
            trial = f"{cur} {w}".strip()
            if font.getlength(trial) > width and cur:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        lines.append(cur)
        lines.append(None)  # paragraph gap
    lead = round(px * 1.35)
    img = Image.new("L", (PAGE_W, margin * 2 + lead * len(lines)), 255)
    draw = ImageDraw.Draw(img)
    y = margin
    for ln in lines:
        if ln:
            draw.text((margin, y), ln, font=font, fill=0)
        y += lead if ln else lead // 2
    return img


# --------------------------------------------------------------------- degradations
def _scale(img: Image.Image, dpi: int) -> Image.Image:
    f = dpi / DPI
    return img.resize((round(img.width * f), round(img.height * f)), Image.Resampling.LANCZOS)


def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    import io

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("L")


def _noise(img: Image.Image, sigma: float, amount: float) -> Image.Image:
    """Seeded noise, so every run sees the same images (Pillow's effect_noise is not)."""
    rng = random.Random(f"noise-{img.size}")
    raw = bytes(
        max(0, min(255, round(rng.gauss(128, sigma)))) for _ in range(img.width * img.height)
    )
    noise = Image.frombytes("L", img.size, raw)
    return Image.blend(img, noise, amount)


def _lighting(img: Image.Image) -> Image.Image:
    grad = Image.linear_gradient("L").rotate(35, expand=False).resize(img.size)
    grad = grad.point([150 + v * 105 // 255 for v in range(256)])  # 150..255: a shadow
    return ImageChops.multiply(img, grad)


def _faint(img: Image.Image) -> Image.Image:
    return img.point([150 + v * 80 // 255 for v in range(256)])  # grey text, grey paper


def _rotate(img: Image.Image, deg: float) -> Image.Image:
    return img.rotate(deg, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=255)


CONDITIONS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "clean-300dpi": lambda im: im,
    "fax-100dpi": lambda im: _scale(im, 100),
    "blur": lambda im: _scale(im, 150).filter(ImageFilter.GaussianBlur(1.6)),
    "noise": lambda im: _noise(_scale(im, 200), 60, 0.35),
    "skew-3deg": lambda im: _rotate(_scale(im, 200), 3),
    "jpeg-q10": lambda im: _jpeg(_scale(im, 150), 10),
    "faint+shadow": lambda im: _lighting(_faint(_scale(im, 200))),
    "phone-photo": lambda im: _jpeg(
        _lighting(_rotate(_scale(im, 150), -1.5).filter(ImageFilter.GaussianBlur(0.8))), 40
    ),
}


# --------------------------------------------------------------------------- metrics
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def edit_distance(a: list[str] | str, b: list[str] | str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ocr: str, truth: str) -> float:
    t = _norm(truth)
    return edit_distance(_norm(ocr), t) / max(len(t), 1)


def wer(ocr: str, truth: str) -> float:
    t = _norm(truth).split()
    return edit_distance(_norm(ocr).split(), t) / max(len(t), 1)


# --------------------------------------------------------------------------- running
@dataclass
class Sample:
    page: int
    font: str
    condition: str
    image: Image.Image
    truth: str


def split_of(page: int) -> str:
    return "test" if page % 3 == 2 else "dev"


def samples(split: str) -> list[Sample]:
    out = []
    for page in range(PAGES):
        if split != "all" and split_of(page) != split:
            continue
        paras = page_text(page)
        truth = "\n".join(paras)
        for font in FONTS:
            base = render(paras, font)
            for cond, fn in CONDITIONS.items():
                out.append(Sample(page, font, cond, fn(base), truth))
    return out


OcrFn = Callable[[Image.Image], tuple[str, float | None]]


def variants() -> dict[str, OcrFn]:
    import glyphly

    reader = glyphly.Reader()

    def glyph(img: Image.Image) -> tuple[str, float | None]:
        page = reader.read(img)
        return page.text, page.confidence

    out: dict[str, OcrFn] = {"glyphly": glyph}
    try:
        from gleanly.errors import UnsupportedSourceError
        from gleanly.ocr.engine import ocr_image
        from gleanly.ocr.tesseract import find_tesseract, run_tesseract

        try:
            has_tesseract = bool(find_tesseract())
        except UnsupportedSourceError:
            has_tesseract = False
        if has_tesseract:

            def plain(img: Image.Image) -> tuple[str, float | None]:
                r = run_tesseract(img)
                return _words_text(r.words), r.confidence

            def gleanly(img: Image.Image) -> tuple[str, float | None]:
                r = ocr_image(img)
                return _words_text(r.words), r.confidence

            out["tesseract"] = plain
            out["gleanly"] = gleanly
    except ImportError:
        pass
    rapid = _rapidocr()
    if rapid is not None:
        out["rapidocr"] = rapid
    return out


def _rapidocr() -> OcrFn | None:
    """RapidOCR as packaged (same PP-OCRv6 models, its own pre- and post-processing)."""
    try:
        import logging

        from rapidocr import RapidOCR
    except ImportError:
        return None
    logging.getLogger("RapidOCR").setLevel(logging.WARNING)
    engine = RapidOCR()

    def run(img: Image.Image) -> tuple[str, float | None]:
        r: Any = engine(img.convert("RGB"))
        if not r.txts:
            return "", None
        txts = [str(t) for t in r.txts]
        scores = [float(x) for x in r.scores]
        chars = sum(len(t) for t in txts)
        conf = sum(x * len(t) for t, x in zip(txts, scores, strict=True)) / chars if chars else None
        return "\n".join(txts), conf

    return run


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    var = sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    return cov / var**0.5 if var else 0.0


def _words_text(words: list) -> str:  # type: ignore[type-arg]
    lines, cur = [], []
    for w in words:
        cur.append(w.text)
        if w.line_end:
            lines.append(" ".join(cur))
            cur = []
    if cur:
        lines.append(" ".join(cur))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.ocr")
    parser.add_argument("--split", default="dev", choices=["all", "dev", "test"])
    parser.add_argument(
        "--suite",
        default="standard",
        choices=["standard", "hard"],
        help="standard: 8 scan conditions; hard: layouts and conditions where engines differ",
    )
    parser.add_argument(
        "-v", "--variants", help="Comma-separated subset (glyphly, tesseract, gleanly, rapidocr)."
    )
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Parallel runs.")
    args = parser.parse_args(argv)

    if args.suite == "hard":
        from bench.ocr_hard import hard_samples

        data = hard_samples(args.split)
    else:
        data = samples(args.split)
    conditions = list(dict.fromkeys(s.condition for s in data))
    fns = variants()
    if args.variants:
        fns = {k: v for k, v in fns.items() if k in args.variants.split(",")}
    print(f"{len(data)} images ({args.split}): {PAGES} texts x {len(FONTS)} fonts x "
          f"{len(conditions)} conditions ({args.suite}), split by text")  # fmt: skip
    results: dict[str, list[tuple[Sample, float, float, float | None]]] = {}
    for name, fn in fns.items():
        start = time.perf_counter()

        def one(s: Sample, fn: OcrFn = fn) -> tuple[Sample, float, float, float | None]:
            text, conf = fn(s.image)
            return s, cer(text, s.truth), wer(text, s.truth), conf

        with ThreadPoolExecutor(args.jobs) as pool:
            results[name] = list(pool.map(one, data))
        secs = time.perf_counter() - start
        per = 1000 * secs / len(data)
        print(f"  {name}: {secs:.0f}s ({per:.0f} ms/image with {args.jobs} jobs)")

    names = list(results)
    head = "".join(f"{n:>10}" for n in names)
    print(f"\nCER by condition (lower is better)\n{'condition':16}{head}")
    for cond in conditions:
        cells = "".join(
            f"{statistics.fmean(c for s, c, _, _ in results[n] if s.condition == cond):>10.3f}"
            for n in names
        )
        print(f"{cond:16}{cells}")
    for label, idx in (("mean CER", 1), ("mean WER", 2)):
        cells = "".join(f"{statistics.fmean(r[idx] for r in results[n]):>10.3f}" for n in names)
        print(f"{label:16}{cells}")
    print(f"\nCER by font\n{'font':16}{head}")
    for font in FONTS:
        cells = "".join(
            f"{statistics.fmean(c for s, c, _, _ in results[n] if s.font == font):>10.3f}"
            for n in names
        )
        print(f"{font:16}{cells}")
    print("\nDoes OCR confidence predict accuracy? Spearman(confidence, 1 - CER):")
    for n in names:
        pairs = [(conf, 1 - c) for _, c, _, conf in results[n] if conf is not None]
        if len(pairs) > 2:
            print(f"  {n}: {spearman([p[0] for p in pairs], [p[1] for p in pairs]):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
