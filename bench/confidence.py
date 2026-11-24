"""Does glyphly's confidence predict its errors? (review item G8)

uv run python -m bench.confidence            # collect pages, compare confidence formulas

A confidence is useful when low values mean more mistakes. The OCR benchmark alone has
too few mistakes to tell (glyphly reads most of it perfectly), so this set adds pages
degraded far beyond it (heavy blur, a third of the resolution, heavy noise, JPEG at
quality 5) and the language pages read with the default model only (wrong script: many
mistakes). For every page we record the character error rate and the signals a formula
can use; formulas are chosen on the dev split and reported on the test split.
Metric: Spearman correlation between confidence and accuracy (1 - CER); higher is better.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from PIL import Image, ImageFilter

from bench.ocr import FONTS, cer, page_text, render, spearman, split_of

DATA = Path(__file__).parent / "data" / "confidence.json"


def _severe(img: Image.Image, kind: str, seed: int) -> Image.Image:
    if kind == "heavy-blur":
        return img.filter(ImageFilter.GaussianBlur(3.2))
    if kind == "third-res":
        small = img.resize((img.width // 3, img.height // 3), Image.Resampling.BILINEAR)
        return small.resize(img.size, Image.Resampling.BILINEAR)
    if kind == "heavy-noise":
        import numpy as np

        arr = np.asarray(img.convert("L"), dtype=np.float32)
        noise = np.random.default_rng(seed).normal(0, 90, arr.shape)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))
    if kind == "jpeg-q5":
        import io

        buf = io.BytesIO()
        img.convert("L").save(buf, "JPEG", quality=5)
        return Image.open(io.BytesIO(buf.getvalue())).copy()
    return img


def collect() -> list[dict]:
    import glyphly
    from bench.languages import SAMPLES
    from bench.languages import render as render_lang

    auto = glyphly.Reader()
    plain = glyphly.Reader(lang=[])  # default model only: wrong-script pages fail
    plain.auto = False
    rows: list[dict] = []

    def record(page: glyphly.Page, truth: str, split: str, kind: str) -> None:
        lines = []
        for ln in page.lines:
            w = ln.box[2] - ln.box[0]
            h = max(1.0, ln.box[3] - ln.box[1])
            lines.append({
                "chars": len(ln.text),
                "aspect": w / h,
                "words": [[wd.text, wd.confidence] for wd in ln.words],
            })  # fmt: skip
        rows.append({"split": split, "kind": kind, "cer": min(1.0, cer(page.text, truth)),
                     "lines": lines})  # fmt: skip

    kinds = ["clean", "heavy-blur", "third-res", "heavy-noise", "jpeg-q5"]
    for page_no in range(9):
        paras = page_text(page_no)
        truth = "\n".join(paras)
        for font in list(FONTS)[:3]:
            base = render(paras, font)
            for kind in kinds:
                img = _severe(base, kind, page_no)
                record(auto.read(img), truth, split_of(page_no), kind)
        print(f"page {page_no} done", flush=True)
    for i, (lang, (_, _, _, text)) in enumerate(SAMPLES.items()):
        img = render_lang(lang)
        split = "test" if i % 3 == 2 else "dev"
        record(auto.read(img), text, split, f"lang-auto-{lang}")
        record(plain.read(img), text, split, f"lang-default-{lang}")
    return rows


# ------------------------------------------------------------------ candidate formulas
def _wide(ch: str) -> bool:
    o = ord(ch)
    return 0x1100 <= o <= 0x11FF or 0x2E80 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF


def _chars_conf(row: dict) -> list[tuple[int, float]]:
    return [(len(t), c) for ln in row["lines"] for t, c in ln["words"]]


def f_mean(row: dict) -> float:
    pairs = _chars_conf(row)
    n = sum(k for k, _ in pairs)
    return sum(k * c for k, c in pairs) / n if n else 0.0


def f_min_word(row: dict) -> float:
    return min((c for _, c in _chars_conf(row)), default=0.0)


def f_unsure_share(row: dict) -> float:
    pairs = _chars_conf(row)
    n = sum(k for k, _ in pairs)
    return 1 - (sum(k for k, c in pairs if c < 0.9) / n if n else 1.0)


def _density(row: dict) -> float:
    """Characters read per line height, wide characters double (median over lines)."""
    vals = []
    for ln in row["lines"]:
        text = "".join(t for t, _ in ln["words"])
        units = sum(2 if _wide(c) else 1 for c in text) + max(0, len(ln["words"]) - 1)
        vals.append(units / max(1.0, ln["aspect"]))
    return statistics.median(vals) if vals else 0.0


def f_density(row: dict) -> float:
    return min(1.0, _density(row) / 2.3)


def f_mean_x_density(row: dict) -> float:
    return f_mean(row) * f_density(row)


def f_unsure_x_density(row: dict) -> float:
    return f_unsure_share(row) * f_density(row)


def f_geo(row: dict) -> float:
    return (f_mean(row) ** 4) * f_unsure_share(row) * f_density(row)


FORMULAS = {
    "mean (current)": f_mean,
    "weakest word": f_min_word,
    "share of sure characters": f_unsure_share,
    "line density": f_density,
    "mean x density": f_mean_x_density,
    "sure share x density": f_unsure_x_density,
    "mean^4 x sure share x density": f_geo,
}


def main() -> int:
    if "--collect" in sys.argv or not DATA.exists():
        DATA.parent.mkdir(parents=True, exist_ok=True)
        DATA.write_text(json.dumps(collect()), encoding="utf-8")
    rows = json.loads(DATA.read_text(encoding="utf-8"))
    for split in ("dev", "test"):
        part = [r for r in rows if r["split"] == split]
        acc = [1 - r["cer"] for r in part]
        bad = sum(r["cer"] > 0.05 for r in part)
        print(f"\n{split}: {len(part)} pages, {bad} with more than 5% wrong characters")
        for name, fn in FORMULAS.items():
            print(f"  {name:32} Spearman {spearman([fn(r) for r in part], acc):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
