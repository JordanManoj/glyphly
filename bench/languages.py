"""Language benchmark: one short text per language, rendered by a browser, read back.

uv run python -m bench.languages              # all languages, default models
uv run python -m bench.languages -l ru,ar,hi  # a subset

Pillow cannot shape complex scripts (Arabic joining, Indic conjuncts) without extra
native libraries, so pages are drawn by a headless Chromium browser (Edge or Chrome),
which shapes every script correctly. Images are cached in bench/data/languages/
(git-ignored). The texts are written for this benchmark: the same short notice in each
language, with a date and an amount, so digits are tested too. Metric: character error
rate (edit distance / reference length, whitespace collapsed).
"""

from __future__ import annotations

import argparse
import html
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from bench.ocr import cer

DATA = Path(__file__).parent / "data" / "languages"

# language code: (script, font family, right-to-left, text)
SAMPLES: dict[str, tuple[str, str, bool, str]] = {
    "en": ("latin", "Arial", False, "The city council approved the new budget of 2,450 for the bridge on 14 March 2026."),
    "de": ("latin", "Arial", False, "Der Stadtrat hat am 14. März 2026 das neue Budget von 2.450 für die Brücke beschlossen."),
    "fr": ("latin", "Arial", False, "Le conseil municipal a approuvé le 14 mars 2026 le nouveau budget de 2 450 pour le pont."),
    "es": ("latin", "Arial", False, "El ayuntamiento aprobó el 14 de marzo de 2026 el nuevo presupuesto de 2.450 para el puente."),
    "pt": ("latin", "Arial", False, "A câmara municipal aprovou em 14 de março de 2026 o novo orçamento de 2.450 para a ponte."),
    "pl": ("latin", "Arial", False, "Rada miasta zatwierdziła 14 marca 2026 nowy budżet w wysokości 2450 na most."),
    "tr": ("latin", "Arial", False, "Belediye meclisi 14 Mart 2026 tarihinde köprü için 2.450 tutarındaki yeni bütçeyi onayladı."),
    "vi": ("latin", "Arial", False, "Hội đồng thành phố đã phê duyệt ngân sách mới 2.450 cho cây cầu vào ngày 14 tháng 3 năm 2026."),
    "ru": ("cyrillic", "Arial", False, "Городской совет 14 марта 2026 года утвердил новый бюджет в размере 2450 на мост."),
    "uk": ("cyrillic", "Arial", False, "Міська рада 14 березня 2026 року затвердила новий бюджет у розмірі 2450 на міст."),
    "el": ("greek", "Arial", False, "Το δημοτικό συμβούλιο ενέκρινε στις 14 Μαρτίου 2026 τον νέο προϋπολογισμό 2.450 για τη γέφυρα."),
    "ar": ("arabic", "Arial", True, "وافق مجلس المدينة في 14 مارس 2026 على الميزانية الجديدة البالغة 2450 للجسر."),
    "fa": ("arabic", "Arial", True, "شورای شهر در 14 مارس 2026 بودجه جدید 2450 را برای پل تصویب کرد."),
    "hi": ("devanagari", "Nirmala UI", False, "नगर परिषद ने 14 मार्च 2026 को पुल के लिए 2450 का नया बजट मंज़ूर किया।"),
    "zh": ("cjk", "Microsoft YaHei", False, "市议会于2026年3月14日批准了大桥2450元的新预算。"),
    "ja": ("cjk", "Yu Gothic", False, "市議会は2026年3月14日に橋の新しい予算2450円を承認しました。"),
    "ko": ("korean", "Malgun Gothic", False, "시의회는 2026년 3월 14일 다리를 위한 새 예산 2450을 승인했습니다."),
    "th": ("thai", "Leelawadee UI", False, "สภาเมืองอนุมัติงบประมาณใหม่ 2450 สำหรับสะพานเมื่อวันที่ 14 มีนาคม 2026"),
    "ta": ("tamil", "Nirmala UI", False, "நகர சபை 14 மார்ச் 2026 அன்று பாலத்திற்கான 2450 புதிய பட்ஜெட்டை அங்கீகரித்தது."),
    "te": ("telugu", "Nirmala UI", False, "నగర మండలి 14 మార్చి 2026న వంతెన కోసం 2450 కొత్త బడ్జెట్‌ను ఆమోదించింది."),
    "he": ("hebrew", "Arial", True, "מועצת העיר אישרה ב-14 במרץ 2026 את התקציב החדש של 2450 לגשר."),
    "bn": ("bengali", "Nirmala UI", False, "নগর পরিষদ 14 মার্চ 2026 তারিখে সেতুর জন্য 2450 নতুন বাজেট অনুমোদন করেছে।"),
}  # fmt: skip

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "chromium",
    "google-chrome",
]


def _browser() -> str:
    for b in BROWSERS:
        if Path(b).is_file() or shutil.which(b):
            return b
    raise SystemExit("needs a Chromium browser (Edge or Chrome) to draw the pages")


def render(lang: str) -> Path:
    """The page for ``lang`` as a PNG (cached)."""
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / f"{lang}.png"
    if out.exists():
        return out
    _, font, rtl, text = SAMPLES[lang]
    page = (
        f'<!doctype html><html lang="{lang}" dir="{"rtl" if rtl else "ltr"}"><head>'
        '<meta charset="utf-8"><style>'
        "html,body{margin:0;background:#fff}"
        f"body{{font-family:'{font}';font-size:34px;line-height:1.5;color:#111;"
        "padding:40px 50px;width:1300px}"
        f"</style></head><body><p>{html.escape(text)}</p></body></html>"
    )
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "page.html"
        src.write_text(page, encoding="utf-8")
        cmd = [
            _browser(), "--headless=new", "--disable-gpu", "--hide-scrollbars",
            f"--user-data-dir={Path(tmp) / 'profile'}", "--force-device-scale-factor=1",
            "--window-size=1400,260", f"--screenshot={out}", src.as_uri(),
        ]  # fmt: skip
        subprocess.run(cmd, capture_output=True, timeout=60, check=False, env=os.environ.copy())
    if not out.exists():
        raise RuntimeError(f"browser did not write {out}")
    return out


def main(argv: list[str] | None = None) -> int:
    import glyphly

    parser = argparse.ArgumentParser(prog="bench.languages")
    parser.add_argument("-l", "--langs", help="comma-separated language codes")
    parser.add_argument("--lang-model", action="store_true",
                        help="read each page with Reader(lang=<code>) instead of the default")  # fmt: skip
    args = parser.parse_args(argv)
    langs = args.langs.split(",") if args.langs else list(SAMPLES)
    default = glyphly.Reader()
    rows = []
    for lang in langs:
        script, _, _, text = SAMPLES[lang]
        reader = glyphly.Reader(lang=lang) if args.lang_model else default
        page = reader.read(render(lang))
        rows.append((lang, script, cer(page.text, text), page.confidence or 0.0, page.text))
    print(f"{'lang':5} {'script':11} {'CER':>7} {'conf':>5}  read as")
    for lang, script, c, conf, got in rows:
        print(f"{lang:5} {script:11} {c:7.1%} {conf:5.2f}  {got[:60]}")
    print(f"\nmean CER {statistics.fmean(r[2] for r in rows):.1%} over {len(rows)} languages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
