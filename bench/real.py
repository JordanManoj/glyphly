"""Real scans in many languages: Wikisource pages with human-proofread transcriptions.

Wikisource volunteers transcribe scanned books, newspapers, letters, pamphlets and forms
page by page; a page marked "validated" has been proofread by two people. Each sample is
the page scan (from Wikimedia Commons) and that validated text, so this measures OCR on
real print: old typefaces, yellowed paper, bleed-through, columns, mixed scripts.

    python -m bench.real fetch --per-lang 10      # downloads into bench/data/real (git-ignored)
    python -m bench.real run --split dev -v glyphly,rapidocr

Ground truth caveats (they hit every engine alike): transcriptions omit some running
headers and page numbers, join hyphenated words and may modernise a few characters, so
no engine reaches 0%. Word F1 ignores reading order; CER does not.
Scans and texts keep their own licenses (mostly public domain); nothing is committed.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import re
import statistics
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from bench.ocr import edit_distance

DATA = Path(__file__).parent / "data" / "real"
UA = "glyphly-bench/0.1 (+https://github.com/JordanManoj/glyphly)"
WIDTH = 1920  # scan width requested from Wikimedia (a standard thumbnail size)
# Wikisource editions with page-by-page proofreading. Several have no validated pages or
# no scans; `fetch` skips those and says so.
LANGS = [
    "en", "fr", "de", "es", "it", "pt", "nl", "pl", "sv", "vi", "la",
    "ru", "uk", "be", "sr", "el",
    "ar", "fa", "he",
    "hi", "mr", "sa", "bn", "as", "or", "pa", "gu", "ta", "te", "kn", "ml",
    "zh", "ko", "th",
]  # fmt: skip
OcrFn = Callable[[Image.Image, str], tuple[str, float | None]]


# --------------------------------------------------------------------------- fetching
def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read()
        except OSError:
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    raise AssertionError("unreachable")


def _api(lang: str, **params: str) -> Any:
    query = urllib.parse.urlencode({"format": "json", "formatversion": "2", **params})
    time.sleep(0.3)  # polite: one request at a time, a little apart
    return json.loads(_get(f"https://{lang}.wikisource.org/w/api.php?{query}"))


_BLOCK = re.compile(r"</?(?:p|div|br|tr|li|h[1-6]|table|dd|dt|center|poem)\b[^>]*>", re.I)


def page_text(rendered: str) -> str:
    """Plain text of a rendered Page: footnote markers, styles and edit links removed."""
    s = re.sub(r"<(style|script)\b.*?</\1>", " ", rendered, flags=re.S | re.I)
    s = re.sub(r'<div class="prp-page-qualityheader.*?</div>', "", s, flags=re.S)  # "validated"
    s = re.sub(r'<sup[^>]*class="[^"]*reference[^"]*".*?</sup>', "", s, flags=re.S)
    s = re.sub(r'<span[^>]*class="[^"]*(?:mw-editsection|noprint)[^"]*".*?</span>', "", s,
               flags=re.S)  # fmt: skip
    s = _BLOCK.sub("\n", s)
    s = html.unescape(re.sub(r"<[^>]+>", "", s))
    lines = [" ".join(line.split()) for line in s.splitlines()]
    return "\n".join(line for line in lines if line)


_TITLE = re.compile(r"^[^:]+:(.+\.(?:djvu|pdf|jpe?g|png|tiff?))(?:/(\d+))?$", re.I)


def split_of(file: str) -> str:
    """Whole books stay in one split: every third file (by hash) is held out."""
    return "test" if int(hashlib.sha1(file.encode()).hexdigest(), 16) % 3 == 0 else "dev"


def fetch(lang: str, per_lang: int) -> int:
    msg = _api(lang, action="query", meta="allmessages",
               ammessages="proofreadpage_quality4_category")  # fmt: skip
    cat = msg["query"]["allmessages"][0].get("content", "")
    if not cat or cat.startswith("&lt;") or "proofreadpage" in cat:
        print(f"{lang}: no proofreading on this Wikisource", file=sys.stderr)
        return 0
    got = _api(lang, action="query", list="categorymembers", cmtitle=f"Category:{cat}",
               cmlimit="500", cmsort="timestamp", cmdir="desc")  # fmt: skip
    members = got.get("query", {}).get("categorymembers", [])
    by_file: dict[str, list[str]] = {}
    for m in members:
        if mm := _TITLE.match(m["title"]):
            by_file.setdefault(mm.group(1), []).append(m["title"])
    out = DATA / lang
    out.mkdir(parents=True, exist_ok=True)
    index = out / "index.jsonl"
    lines = index.read_text("utf-8").splitlines() if index.exists() else []
    rows = [json.loads(x) for x in lines]
    seen = {r["file"] for r in rows}
    n = sum(1 for r in rows if not r.get("rejected"))
    order = sorted(by_file.items(), key=lambda kv: hashlib.sha1(kv[0].encode()).digest())
    for file, titles in order:
        if n >= per_lang:
            break
        if file in seen:
            continue
        title = titles[len(titles) // 2]  # one page per book: more variety
        page = _TITLE.match(title).group(2)  # type: ignore[union-attr]
        page = str(int(page)) if page else page  # "৯৫" -> "95": Wikimedia wants ASCII digits
        try:
            info = _api(lang, action="query", titles=f"File:{file}", prop="imageinfo",
                        iiprop="url|extmetadata", iiurlwidth=str(WIDTH),
                        **({"iiurlparam": f"page{page}-{WIDTH}px"} if page else {}))  # fmt: skip
            ii = info["query"]["pages"][0]["imageinfo"][0]
            text = page_text(_api(lang, action="parse", page=title, prop="text")["parse"]["text"])
            if len(text) < 200:  # blank pages, plates, title pages
                continue
            img = Image.open(io.BytesIO(_get(ii.get("thumburl") or ii["url"]))).convert("RGB")
        except (OSError, KeyError, IndexError, ValueError) as exc:
            print(f"{lang}: skipped {title!r}: {exc}", file=sys.stderr)
            continue
        if blank(img):  # a white render instead of the scan: not usable, never retried
            with index.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"file": file, "rejected": "blank image"}) + "\n")
            continue
        sid = next(f"{k:03d}" for k in range(1000) if not (out / f"{k:03d}.jpg").exists())
        img.save(out / f"{sid}.jpg", quality=92)
        (out / f"{sid}.txt").write_text(text, encoding="utf-8")
        meta = ii.get("extmetadata", {})
        row = {
            "id": sid, "lang": lang, "title": title, "file": file, "split": split_of(file),
            "width": img.width,
            "license": meta.get("LicenseShortName", {}).get("value", ""),
            "source": f"https://{lang}.wikisource.org/wiki/{urllib.parse.quote(title)}",
        }  # fmt: skip
        with index.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        n += 1
    print(f"{lang}: {n} pages", file=sys.stderr)
    return n


# --------------------------------------------------------------------------- metrics
_SAME = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                       "\u2013": "-", "\u2014": "-", "\u00ad": None, "\u200b": None})  # fmt: skip


def norm(text: str) -> str:
    """NFC, one kind of quote and dash, line-end hyphens joined, single spaces."""
    t = unicodedata.normalize("NFC", text).translate(_SAME)
    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)
    return " ".join(t.split())


def cer(ocr: str, truth: str) -> float:
    t = norm(truth)
    return edit_distance(norm(ocr), t) / max(len(t), 1)


def word_f1(ocr: str, truth: str) -> float:
    """Order-free: share of words found, as F1 over word counts (punctuation stripped)."""

    def bag(s: str) -> Counter[str]:
        return Counter(w for w in (re.sub(r"^\W+|\W+$", "", x) for x in norm(s).split()) if w)

    o, t = bag(ocr), bag(truth)
    hit = sum((o & t).values())
    if not hit:
        return 0.0
    p, r = hit / sum(o.values()), hit / sum(t.values())
    return 2 * p * r / (p + r)


# --------------------------------------------------------------------------- engines
def _glyphly(base: bool) -> OcrFn:
    """glyphly in auto mode; ``base`` leaves out the word-level (PARSeq) script models, i.e.
    glyphly 0.1.0a1, for before/after comparisons."""
    from glyphly import Reader
    from glyphly.models import WORD_SCRIPTS

    holder: dict[str, Reader] = {}

    def run(img: Image.Image, lang: str) -> tuple[str, float | None]:
        if "r" not in holder:
            holder["r"] = Reader()
            if base:
                holder["r"]._disabled = set(WORD_SCRIPTS)
        page = holder["r"].read(img)
        return page.text, page.confidence

    return run


def engines(names: list[str]) -> dict[str, OcrFn]:
    out: dict[str, OcrFn] = {}
    for name in names:
        if name in ("glyphly", "glyphly-base"):
            out[name] = _glyphly(base=name == "glyphly-base")
        elif name == "rapidocr":
            from bench.ocr import _rapidocr

            fn = _rapidocr()
            if fn is not None:
                out[name] = lambda img, lang, fn=fn: fn(img)
        else:
            print(f"unknown engine {name!r}", file=sys.stderr)
    return out


# Pages whose scan does not show the transcribed text (checked by eye after `audit`
# flagged them): usually a book re-uploaded with its pages in another order.
EXCLUDED = Path(__file__).with_name("real_excluded.json")


def blank(img: Image.Image) -> bool:
    from PIL import ImageStat

    return ImageStat.Stat(img.convert("L")).stddev[0] < 8


def samples(split: str, langs: list[str] | None) -> list[dict[str, Any]]:
    excluded = set(json.loads(EXCLUDED.read_text("utf-8"))) if EXCLUDED.exists() else set()
    rows = []
    for index in sorted(DATA.glob("*/index.jsonl")):
        for line in index.read_text("utf-8").splitlines():
            row = json.loads(line)
            if row.get("rejected") or row.get("title") in excluded:
                continue
            if (split == "all" or row["split"] == split) and (not langs or row["lang"] in langs):
                rows.append(row)
    return rows


def audit() -> None:
    """Reject blank images already downloaded, and list pages to check by eye: those
    where far fewer text lines are found than the transcription has."""
    from glyphly import Reader
    from glyphly.detect import detect
    from glyphly.engine import to_rgb

    reader = Reader(download=False)
    for index in sorted(DATA.glob("*/index.jsonl")):
        rows = [json.loads(x) for x in index.read_text("utf-8").splitlines()]
        changed = False
        for row in rows:
            if row.get("rejected"):
                continue
            img_path = index.parent / f"{row['id']}.jpg"
            img = Image.open(img_path)
            row.setdefault("width", img.width)
            if blank(img):
                row["rejected"] = "blank image"
                img_path.unlink()
                img_path.with_suffix(".txt").unlink()
                changed = True
                continue
            truth_lines = len(img_path.with_suffix(".txt").read_text("utf-8").splitlines())
            found = len(detect(reader._det, to_rgb(img), max_side=reader.max_side))
            if truth_lines >= 8 and found < 0.3 * truth_lines:
                print(f"CHECK {row['lang']}/{row['id']}: {found} lines found, "
                      f"{truth_lines} transcribed | {row['title']}")  # fmt: skip
            changed = True
        if changed:
            index.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                             encoding="utf-8")  # fmt: skip


def run(split: str, names: list[str], langs: list[str] | None, out: Path | None) -> None:
    """Results are appended to ``out`` (JSON lines) page by page; a rerun skips pages already
    there, so an interrupted run resumes where it stopped."""
    fns = engines(names)
    results: list[dict[str, Any]] = []
    if out and out.exists():
        results = [json.loads(x) for x in out.read_text("utf-8").splitlines() if x.strip()]
    done = {(r["lang"], r["id"], r["engine"]) for r in results}
    for row in samples(split, langs):
        todo = [n for n in fns if (row["lang"], row["id"], n) not in done]
        if not todo:
            continue
        folder = DATA / row["lang"]
        img = Image.open(folder / f"{row['id']}.jpg")
        truth = (folder / f"{row['id']}.txt").read_text("utf-8")
        for name in todo:
            t0 = time.perf_counter()
            text, conf = fns[name](img, row["lang"])
            secs = time.perf_counter() - t0
            result = {**row, "engine": name, "cer": cer(text, truth), "f1": word_f1(text, truth),
                      "conf": conf, "secs": secs}  # fmt: skip
            results.append(result)
            if out:
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(f"{row['lang']} {row['id']} {name:12s} CER {result['cer']:6.1%} "
                  f"F1 {result['f1']:.3f} {secs:5.1f}s", file=sys.stderr, flush=True)  # fmt: skip
    keep = set(fns)
    report([r for r in results if r["engine"] in keep and (not langs or r["lang"] in langs)])


def report(results: list[dict[str, Any]]) -> None:
    names = sorted({r["engine"] for r in results})
    langs = sorted({r["lang"] for r in results})
    print("| lang | pages | " + " | ".join(f"{n} CER / F1" for n in names) + " |")
    print("|---|---|" + "---|" * len(names))
    for lang in [*langs, "ALL"]:
        cells = []
        pages = 0
        for n in names:
            rs = [r for r in results if r["engine"] == n and (lang == "ALL" or r["lang"] == lang)]
            pages = len(rs)
            cells.append(
                f"{statistics.median(r['cer'] for r in rs):.1%} / "
                f"{statistics.mean(r['f1'] for r in rs):.3f}"
                if rs
                else "-"
            )
        print(f"| {lang} | {pages} | " + " | ".join(cells) + " |")
    print("\nCER: median per page (lower is better). F1: mean word F1 (higher is better).")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bench.real")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--per-lang", type=int, default=10)
    f.add_argument("--langs", default=",".join(LANGS))
    sub.add_parser("audit")
    r = sub.add_parser("run")
    r.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    r.add_argument("-v", "--engines", default="glyphly,rapidocr")
    r.add_argument("--langs", default="")
    r.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "audit":
        audit()
    elif args.cmd == "fetch":
        total = sum(fetch(lang, args.per_lang) for lang in args.langs.split(","))
        print(f"{total} pages in {DATA}", file=sys.stderr)
    else:
        langs = [x for x in args.langs.split(",") if x] or None
        run(args.split, args.engines.split(","), langs, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
