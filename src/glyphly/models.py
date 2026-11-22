"""Model files: where they come from, where they are kept, and how they are checked.

The neural networks are downloaded once, on first use, into a cache folder
(``GLYPHLY_HOME``, default ``~/.cache/glyphly``) and verified against a SHA-256
fingerprint pinned here, so a changed or corrupted file is never used.

Current models: PaddleOCR PP-OCRv6 "small" text detection and recognition, converted to
ONNX by the RapidOCR project. Both are Apache-2.0 (see NOTICE). They are a starting point:
glyphly's own trained recognition model is planned to replace them.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_BASE = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv6"


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str
    sha256: str
    size_mb: float


DETECTOR = ModelFile(
    "PP-OCRv6_det_small.onnx",
    f"{_BASE}/det/PP-OCRv6_det_small.onnx",
    "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
    9.9,
)
RECOGNIZER = ModelFile(
    "PP-OCRv6_rec_small.onnx",
    f"{_BASE}/rec/PP-OCRv6_rec_small.onnx",
    "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
    21.2,
)


class ModelError(RuntimeError):
    """A model file is missing, cannot be downloaded, or fails its fingerprint check."""


def home() -> Path:
    return Path(os.environ.get("GLYPHLY_HOME") or Path.home() / ".cache" / "glyphly")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure(model: ModelFile, *, download: bool = True) -> Path:
    """Path to a verified copy of ``model``, downloading it first if needed."""
    path = home() / model.name
    if path.exists():
        if _sha256(path) == model.sha256:
            return path
        path.unlink()  # corrupted or replaced: fetch again
    if not download:
        raise ModelError(
            f"{model.name} is not in {home()}. Run `glyphly download` once while online, or "
            f"place the file there yourself (SHA-256 {model.sha256})."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(model.url, timeout=60) as resp:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        got = _sha256(tmp)
        if got != model.sha256:
            raise ModelError(
                f"Downloaded {model.name} has SHA-256 {got}, expected {model.sha256}; "
                "it was not used."
            )
        tmp.replace(path)
    except OSError as exc:
        raise ModelError(f"Could not download {model.name} from {model.url}: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)
    return path


def download_all() -> list[Path]:
    return [ensure(m) for m in (DETECTOR, RECOGNIZER)]


__all__ = ["DETECTOR", "RECOGNIZER", "ModelError", "ModelFile", "download_all", "ensure", "home"]
