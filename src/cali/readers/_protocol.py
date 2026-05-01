"""Protocol defining the interface for all cali data readers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path
    from typing import Any

    import numpy as np
    import useq


@runtime_checkable
class CaliDataReader(Protocol):
    """Protocol for all cali data readers.

    Any reader that satisfies this interface can be used with CaliGui
    and CaliRunner. Existing readers (TensorstoreZarrReader, OMEZarrReader,
    TiffCollectionReader) already conform to this protocol.
    """

    @property
    def path(self) -> Path:
        """Return the path to the data source."""
        ...

    @property
    def sequence(self) -> useq.MDASequence | None:
        """Return the MDASequence describing the acquisition."""
        ...

    @property
    def metadata(self) -> list[dict] | dict:
        """Return frame-level metadata."""
        ...

    def isel(
        self,
        indexers: Mapping[str, int] | None = None,
        metadata: bool = False,
        **kwargs: Any,
    ) -> np.ndarray | tuple[np.ndarray, list[dict]]:
        """Select data by axis indices (e.g. p=0, t=1, c=0, z=0)."""
        ...

    def close(self) -> None:
        """Release resources and file handles."""
        ...
