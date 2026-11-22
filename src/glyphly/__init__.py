"""glyphly: OCR for Python with nothing to install but the package."""

from glyphly.__about__ import __version__
from glyphly.engine import Reader, read
from glyphly.models import ModelError
from glyphly.types import Line, Page, Word

__all__ = ["Line", "ModelError", "Page", "Reader", "Word", "__version__", "read"]
