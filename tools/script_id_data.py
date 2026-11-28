"""Training data for glyphly's script identifier (not shipped; run by maintainers).

    python tools/script_id_data.py text     # Wikipedia sentences per language -> data/text/
    python tools/script_id_data.py render   # pages drawn by a Chromium browser -> data/lines/

Text: random Wikipedia articles' plain-text extracts (CC BY-SA), used only to draw training
images; no text ends up in the model. Images: each language's sentences are drawn as
pages in several font families and sizes by a headless Chromium browser (every script
shaped correctly), and the pages are cut into lines with glyphly's own detector.
"""

from __future__ import annotations

import html
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "bench" / "data" / "script_id"
UA = "glyphly-bench/0.1 (+https://github.com/JordanManoj/glyphly)"

# script class -> Wikipedia languages written in it
SCRIPTS: dict[str, list[str]] = {
    "latin": ["en", "fr", "de", "es", "pl", "vi", "tr", "cs", "ro", "is", "hu", "sw"],
    "cyrillic": ["ru", "uk", "bg", "sr", "kk"],
    "greek": ["el"],
    "arabic": ["ar", "fa", "ur", "ps"],
    "hebrew": ["he", "yi"],
    "devanagari": ["hi", "mr", "ne"],
    "bengali": ["bn", "as"],
    "gurmukhi": ["pa"],
    "gujarati": ["gu"],
    "odia": ["or"],
    "tamil": ["ta"],
    "telugu": ["te"],
    "kannada": ["kn"],
    "malayalam": ["ml"],
    "sinhala": ["si"],
    "thai": ["th"],
    "lao": ["lo"],
    "khmer": ["km"],
    "myanmar": ["my"],
    "georgian": ["ka"],
    "armenian": ["hy"],
    "han": ["zh"],
    "japanese": ["ja"],
    "hangul": ["ko"],
    "ethiopic": ["am"],
}
RTL = {"arabic", "hebrew"}
FAMILIES = ["serif", "sans-serif", "Times New Roman", "Arial", "Segoe UI", "Georgia"]


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def fetch_text(lang: str, want: int = 400) -> list[str]:
    """Sentences (or sentence-like chunks) from random articles of one Wikipedia."""
    out: list[str] = []
    for _ in range(12):
        q = urllib.parse.urlencode({
            "action": "query", "format": "json", "generator": "random", "grnnamespace": 0,
            "grnlimit": 20, "prop": "extracts", "explaintext": 1, "exlimit": 20,
            "exchars": 1200,
        })  # fmt: skip
        try:
            pages = json.loads(_get(f"https://{lang}.wikipedia.org/w/api.php?{q}"))
        except OSError as exc:
            print(f"{lang}: {exc}", file=sys.stderr)
            break
        for page in pages.get("query", {}).get("pages", {}).values():
            for chunk in re.split(
                r"(?<=[.!?\u0964\u3002\u061f\u104b])\s+|\n+", page.get("extract", "")
            ):
                chunk = " ".join(chunk.split())
                if chunk.startswith("=") or len(chunk) < 15:
                    continue
                # scripts written without spaces or full stops (Thai, Lao, Khmer) give long
                # chunks: cut them into line-sized pieces
                out += [chunk[i : i + 90] for i in range(0, len(chunk), 90) if len(chunk[i:]) >= 15]
        time.sleep(0.5)
        if len(out) >= want:
            break
    return out[:want]


def text() -> None:
    out = DATA / "text"
    out.mkdir(parents=True, exist_ok=True)
    for script, langs in SCRIPTS.items():
        for lang in langs:
            path = out / f"{script}.{lang}.json"
            if path.exists():
                continue
            got = fetch_text(lang, want=400 if len(langs) == 1 else 200)
            path.write_text(json.dumps(got, ensure_ascii=False), encoding="utf-8")
            print(f"{script}/{lang}: {len(got)} sentences", file=sys.stderr)


def _browser() -> str:
    for b in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe"):  # fmt: skip
        if Path(b).is_file():
            return b
    raise SystemExit("needs Edge or Chrome")


def draw_page(sentences: list[str], lang: str, rtl: bool, out: Path, rng: random.Random) -> None:
    family = rng.choice(FAMILIES)
    size = rng.choice([18, 22, 26, 30, 36])
    weight = rng.choice(["normal", "normal", "bold"])
    lines = "".join(f"<div>{html.escape(s)}</div>" for s in sentences)
    page = (
        f'<!doctype html><html lang="{lang}" dir="{"rtl" if rtl else "ltr"}"><head>'
        '<meta charset="utf-8"><style>html,body{margin:0;background:#fff}'
        f"body{{font-family:'{family}';font-size:{size}px;font-weight:{weight};"
        "line-height:1.7;color:#111;padding:30px 40px;width:1100px}"
        "div{white-space:nowrap;overflow:hidden}</style></head>"
        f"<body>{lines}</body></html>"
    )
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "page.html"
        src.write_text(page, encoding="utf-8")
        height = int(len(sentences) * size * 1.7 + 80)
        cmd = [
            _browser(), "--headless=new", "--disable-gpu", "--hide-scrollbars",
            f"--user-data-dir={Path(tmp) / 'profile'}", "--force-device-scale-factor=1",
            f"--window-size=1200,{height}", f"--screenshot={out}", src.as_uri(),
        ]  # fmt: skip
        subprocess.run(cmd, capture_output=True, timeout=90, check=False, env=os.environ.copy())


def render(pages_per_script: int = 16) -> None:
    """Pages per script, cut into line images (PNG) with glyphly's detector."""
    import numpy as np
    from PIL import Image

    from glyphly import Reader
    from glyphly.detect import detect
    from glyphly.engine import cut, to_rgb

    reader = Reader(download=False)
    rng = random.Random(7)
    out = DATA / "lines"
    for script in SCRIPTS:
        folder = out / script
        folder.mkdir(parents=True, exist_ok=True)
        done = len(list(folder.glob("*.png")))
        files = sorted((DATA / "text").glob(f"{script}.*.json"))
        pool = [(f.stem.split(".")[1], s) for f in files for s in json.loads(f.read_text("utf-8"))]
        if not pool or done >= pages_per_script * 15:
            continue
        for k in range(pages_per_script):
            lang = rng.choice(sorted({lang for lang, _ in pool}))
            sents = rng.sample([s for la, s in pool if la == lang], k=min(20, len(pool)))
            with tempfile.TemporaryDirectory() as tmp:
                shot = Path(tmp) / "p.png"
                draw_page(sents, lang, script in RTL, shot, rng)
                if not shot.exists():
                    continue
                rgb = to_rgb(Image.open(shot))
            for i, box in enumerate(detect(reader._det, rgb, max_side=reader.max_side)):
                crop = cut(rgb, box)
                if crop.shape[1] < 2 * crop.shape[0]:  # page furniture, single marks
                    continue
                Image.fromarray(np.ascontiguousarray(crop)).convert("L").save(
                    folder / f"r{k:03d}_{i:03d}.png"
                )
        print(f"{script}: {len(list(folder.glob('*.png')))} lines", file=sys.stderr)


if __name__ == "__main__":
    {"text": text, "render": render}[sys.argv[1]]()
