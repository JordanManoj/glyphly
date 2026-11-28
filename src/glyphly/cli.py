"""Command line: ``glyphly read scan.png``, ``glyphly download``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from glyphly.__about__ import __version__


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glyphly", description="OCR with nothing to install.")
    parser.add_argument("--version", action="version", version=f"glyphly {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    read = sub.add_parser("read", help="read the text in one or more images")
    read.add_argument("images", nargs="+")
    read.add_argument("--json", action="store_true", help="same as --format json")
    read.add_argument(
        "-f", "--format", choices=["text", "json", "hocr", "alto", "pdf"], default="text",
        help="text (default); json: lines, words, boxes, confidences; hocr or alto: standard "
        "OCR XML with word boxes; pdf: searchable PDF (the images with an invisible text layer)",
    )  # fmt: skip
    read.add_argument(
        "-o", "--output", help="write to this file instead of the screen (needed for pdf)"
    )
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

        fmt = "json" if args.json else args.format
        if fmt == "pdf" and not args.output:
            print("error: --format pdf needs --output FILE.pdf", file=sys.stderr)
            return 2
        lang = args.lang if args.lang == "auto" else args.lang.split(",")
        reader = Reader(lang=lang)
        pages = []
        chunks: list[str] = []
        for name in args.images:
            if fmt == "pdf":
                from PIL import Image

                image = Image.open(name)
                image.load()
                pages.append((image, reader.read(image)))
                continue
            page = reader.read(name)
            if fmt == "json":
                chunks.append(json.dumps({"image": name, **asdict(page)}, ensure_ascii=False))
            elif fmt in ("hocr", "alto"):
                pages.append((None, page))
            else:
                if len(args.images) > 1:
                    chunks.append(f"==> {name} <==")
                chunks.append(page.text)
        from glyphly import export

        if fmt == "pdf":
            Path(args.output).write_bytes(export.to_pdf(pages))
            print(f"ok: {args.output}", file=sys.stderr)
            return 0
        if fmt == "hocr":
            chunks = [export.to_hocr([p for _, p in pages], images=args.images)]
        elif fmt == "alto":
            chunks = [export.to_alto([p for _, p in pages], images=args.images)]
        text = "\n".join(chunks)
        if args.output:
            Path(args.output).write_text(text if text.endswith("\n") else text + "\n",
                                         encoding="utf-8")  # fmt: skip
        else:
            print(text)
    except models.ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


__all__ = ["main"]
