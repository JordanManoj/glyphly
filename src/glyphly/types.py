"""Results: a page of lines, each line a list of words with boxes and confidences.

Boxes are ``(x0, top, x1, bottom)`` in pixels of the input image.
"""

from __future__ import annotations

from dataclasses import dataclass, field

Box = tuple[float, float, float, float]


@dataclass
class Word:
    text: str
    box: Box
    confidence: float  # mean probability of the word's characters, 0..1


@dataclass
class Line:
    words: list[Word]
    box: Box
    confidence: float

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)


@dataclass
class Page:
    lines: list[Line]
    width: int
    height: int
    steps: list[str] = field(default_factory=list)  # what the engine did, for the record

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)

    @property
    def words(self) -> list[Word]:
        return [w for line in self.lines for w in line.words]

    @property
    def confidence(self) -> float | None:
        """Character-weighted mean confidence, None for an empty page."""
        chars = sum(len(w.text) for w in self.words)
        if not chars:
            return None
        return sum(w.confidence * len(w.text) for w in self.words) / chars


__all__ = ["Box", "Line", "Page", "Word"]
