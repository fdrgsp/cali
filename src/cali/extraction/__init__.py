"""Trace extraction, with the image/runner stack loaded only when requested."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._extraction_runner import ExtractionRunner

__all__ = ["ExtractionRunner"]


def __getattr__(name: str) -> Any:
    """Keep standalone backend imports independent of image readers and GUI code."""
    if name == "ExtractionRunner":
        from ._extraction_runner import ExtractionRunner

        globals()[name] = ExtractionRunner
        return ExtractionRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
