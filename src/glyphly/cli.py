"""Command line: ``glyphly read scan.png``, ``glyphly download``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict

from glyphly.__about__ import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glyphly", description="OCR with nothing to install.")
    parser.add_argument("--version", action="version", version=f"glyphly {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    read = sub.add_parser("read", help="read the text in one or more images")
    read.add_argument("images", nargs="+")
    read.add_argument("--json", action="store_true", help="lines, words, boxes and confidences")
    dl = sub.add_parser("download", help="fetch the model files now (for offline use later)")
    dl.add_argument("--all", action="store_true", help="also every script model (about 70 MB)")
    read.add_argument("--lang", default="auto",
                      help="auto (default) or language codes, e.g. ru or hi,en")  # fmt: skip
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    from glyphly import models

    try:
        if args.command == "download":
            for path in models.download_all(scripts=args.all):
                print(f"ok: {path}")
            return 0
        from glyphly.engine import Reader

        lang = args.lang if args.lang == "auto" else args.lang.split(",")
        reader = Reader(lang=lang)
        for name in args.images:
            page = reader.read(name)
            if args.json:
                print(json.dumps({"image": name, **asdict(page)}, ensure_ascii=False))
            else:
                if len(args.images) > 1:
                    print(f"==> {name} <==")
                print(page.text)
    except models.ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


__all__ = ["main"]
