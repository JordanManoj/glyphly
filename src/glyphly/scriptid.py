"""Which writing system a page is in, from the line images alone.

A small convolutional network (glyphly's own, trained by ``tools/train_script_id.py`` on
lines drawn from Wikipedia text in 25 scripts plus lines of real scanned pages) looks at
a line and gives a probability for each script. A page's script is the average over its
longest lines. Recognizer confidence cannot answer this question: a word recognizer
reads any script with high confidence, so the line image itself has to be looked at.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

HEIGHT = 32
MAX_WIDTH = 1024
PAGE_LINES = 12  # the longest lines of a page vote on its script

# script-identifier classes -> glyphly script models that read them
MODELS = {"hangul": ["korean"], "bengali": ["bengali", "assamese"]}
# classes read by the default recognizer
DEFAULT = {"latin", "han", "japanese"}
NAMES = {"han": "Chinese", "hangul": "Korean", "myanmar": "Myanmar (Burmese)"}


def name(script: str) -> str:
    return NAMES.get(script, script.capitalize())


class ScriptId:
    def __init__(self, session: Any) -> None:
        meta = json.loads(session.get_modelmeta().custom_metadata_map["glyphly"])
        self.session = session
        self.classes: list[str] = meta["classes"]
        self.height = int(meta.get("height", HEIGHT))
        self.input = session.get_inputs()[0].name

    def _prepare(self, crop: np.ndarray) -> np.ndarray:
        """RGB line crop -> (1, 1, H, W) grey in [0, 1], as in training."""
        from PIL import Image

        g = Image.fromarray(np.ascontiguousarray(crop)).convert("L")
        w = max(8, round(g.width * self.height / max(1, g.height)))
        g = g.resize((min(w, MAX_WIDTH), self.height), Image.Resampling.BILINEAR)
        return (np.asarray(g, dtype=np.float32) / 255.0)[None, None]

    def lines(self, crops: list[np.ndarray]) -> np.ndarray:
        """Script probabilities for each line, shape (lines, classes)."""
        out = []
        for crop in crops:
            logits = self.session.run(None, {self.input: self._prepare(crop)})[0][0]
            e = np.exp(logits - logits.max())
            out.append(e / e.sum())
        return np.array(out).reshape(len(crops), len(self.classes))

    def page(self, crops: list[np.ndarray]) -> list[tuple[str, float]]:
        """Scripts of a page with their average probability, most likely first.

        Only lines at least four times as wide as high vote: short marks and numbers do
        not show their script."""
        lines = [c for c in crops if c.shape[1] >= 4 * c.shape[0]]
        lines = sorted(lines, key=lambda c: -c.shape[1])[:PAGE_LINES]
        if not lines:
            return []
        mean = self.lines(lines).mean(axis=0)
        order = np.argsort(-mean)
        return [(self.classes[k], float(mean[k])) for k in order]


__all__ = ["DEFAULT", "MODELS", "ScriptId", "name"]
