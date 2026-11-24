"""Model files: where they come from, where they are kept, and how they are checked.

The neural networks are downloaded once, on first use, into a cache folder
(``GLYPHLY_HOME``, default ``~/.cache/glyphly``) and verified against a SHA-256
fingerprint pinned here, so a changed or corrupted file is never used.

Sources are tried in order until one gives a file with the right fingerprint:
``GLYPHLY_MODEL_URL`` (a folder URL of your own mirror, e.g. inside a company network),
then the glyphly release mirror on GitHub, then ModelScope, where the files are published.

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

_MODELSCOPE = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv6"
# Same files, same fingerprints, attached to a glyphly release (used once the repo is public).
_GITHUB = "https://github.com/JordanManoj/glyphly/releases/download/models-v1"


@dataclass(frozen=True)
class ModelFile:
    name: str
    url: str  # where the file is published
    sha256: str
    size_mb: float

    def sources(self) -> list[str]:
        """Download URLs in the order they are tried."""
        out = []
        own = os.environ.get("GLYPHLY_MODEL_URL")
        if own:
            out.append(f"{own.rstrip('/')}/{self.name}")
        out += [f"{_GITHUB}/{self.name}", self.url]
        return out


DETECTOR = ModelFile(
    "PP-OCRv6_det_small.onnx",
    f"{_MODELSCOPE}/det/PP-OCRv6_det_small.onnx",
    "090f04abcd9d9a7498bc4ebf677e4cb9bdce1fe4197ddb7e529f1ef44e1ff94f",
    9.9,
)
RECOGNIZER = ModelFile(
    "PP-OCRv6_rec_small.onnx",
    f"{_MODELSCOPE}/rec/PP-OCRv6_rec_small.onnx",
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
    problems: list[str] = []
    for url in model.sources():
        problem = _fetch(url, path, model.sha256)
        if problem is None:
            return path
        problems.append(f"{url}: {problem}")
    raise ModelError(f"Could not download {model.name}. Tried:\n  " + "\n  ".join(problems))


def _fetch(url: str, path: Path, sha256: str) -> str | None:
    """Download ``url`` to ``path`` if its fingerprint matches; else the reason it failed."""
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=60) as resp:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
        got = _sha256(tmp)
        if got != sha256:
            return f"wrong SHA-256 {got} (file not used)"
        tmp.replace(path)
        return None
    except OSError as exc:  # HTTP errors, timeouts, no network
        return str(exc)
    finally:
        tmp.unlink(missing_ok=True)


def download_all(*, scripts: bool = False) -> list[Path]:
    """Fetch the default models, and every script model too when ``scripts`` is true."""
    files = [DETECTOR, RECOGNIZER, *(SCRIPT_MODELS.values() if scripts else ())]
    return [ensure(m) for m in files]


# --------------------------------------------------------------------------- scripts
# The default recognizer reads Latin-script languages, Chinese and Japanese. Other
# scripts have their own open recognizers (PaddleOCR PP-OCRv5 mobile, Apache-2.0, ONNX by
# RapidOCR), downloaded only when a page needs them.
_V5 = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/PP-OCRv5/rec"


def _v5(name: str, sha256: str, size_mb: float) -> ModelFile:
    file = f"{name}_PP-OCRv5_rec_mobile.onnx"
    return ModelFile(file, f"{_V5}/{file}", sha256, size_mb)


SCRIPT_MODELS: dict[str, ModelFile] = {
    "cyrillic": _v5(
        "cyrillic", "90f761b4bfcce0c8c561c0cb5c887b0971d3ec01c32164bdf7374a35b0982711", 8.1
    ),
    "greek": _v5("el", "b4368bccd557123c702b7549fee6cd1e94b581337d1c9b65310f109131542b7f", 7.8),
    "arabic": _v5(
        "arabic", "c1192e632d0baa9146ae5b756a0e635e3dc63c1733737ebfd1629e87144e9295", 8.0
    ),
    "devanagari": _v5(
        "devanagari", "d6f0a906580e3fa6b324a318718f1f31f268b6ea8ef985f91c2012a37f52c91e", 7.9
    ),
    "korean": _v5(
        "korean", "cd6e2ea50f6943ca7271eb8c56a877a5a90720b7047fe9c41a2e541a25773c9b", 13.5
    ),
    "thai": _v5("th", "de541dd83161c241ff426f7ecfd602a0ba77d686cf3ab9a6c255ea82fd08006e", 7.9),
    "tamil": _v5("ta", "a42448808b7dea87597336f12438935f40353f1949e8360acd9e06b4da21bfe1", 7.9),
    "telugu": _v5("te", "a3690451b50028a09a3316a1274f7c05728151ea3f8fd392696397a7fefcbd92", 7.9),
}
RTL_SCRIPTS = {"arabic"}

# ISO 639-1 codes -> script model. Languages not listed (Latin-script languages, Chinese,
# Japanese) are read by the default recognizer.
_LANGS = {
    "cyrillic": "ru uk be bg sr mk kk ky mn tg tt ba cv ce os ab",
    "greek": "el",
    "arabic": "ar fa ur ug ps sd ku ckb",
    "devanagari": "hi mr ne sa mai bho new",
    "korean": "ko",
    "thai": "th",
    "tamil": "ta",
    "telugu": "te",
}
LANG_SCRIPT: dict[str, str] = {code: s for s, codes in _LANGS.items() for code in codes.split()}


def script_for(lang: str) -> str | None:
    """The script model a language needs, or None if the default recognizer reads it."""
    return LANG_SCRIPT.get(lang.lower().split("-")[0])


__all__ = [
    "DETECTOR",
    "LANG_SCRIPT",
    "RECOGNIZER",
    "RTL_SCRIPTS",
    "SCRIPT_MODELS",
    "ModelError",
    "ModelFile",
    "download_all",
    "ensure",
    "home",
    "script_for",
]
