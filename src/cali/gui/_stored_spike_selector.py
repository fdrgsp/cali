"""Results-level backend selection, independent of future extraction settings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget
from superqt.utils import signals_blocked

from cali.plot import get_stored_spike_capabilities

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

    from cali.sqlmodel._spike_settings import SpikeMethod


class _StoredSpikeSelector(QWidget):
    methodChanged = Signal(str)

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.method: SpikeMethod = "oasis"
        self.capabilities: dict[str, set[str]] | None = None
        self._run_id: int | None = None
        self._engine: Engine | None = None
        self._combo = QComboBox(self)
        self._combo.setToolTip("Spike outputs stored on the selected run.")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Spike Backend:", self))
        layout.addWidget(self._combo)
        self._combo.currentIndexChanged.connect(self._on_changed)
        self.setVisible(False)

    def refresh(
        self, engine: Engine | None, run_id: int | None, *, fov_name: str | None = None
    ) -> None:
        self.capabilities = (
            get_stored_spike_capabilities(engine, run_id, fov_name=fov_name)
            if engine is not None and run_id is not None
            else None
        )
        methods = [
            method
            for method in ("cascade", "oasis")
            if method in (self.capabilities or {})
        ]
        preferred = (
            self.method if self._run_id == run_id and self._engine is engine else None
        )
        with signals_blocked(self._combo):
            self._combo.clear()
            for method in methods:
                self._combo.addItem(method.upper(), method)
            index = self._combo.findData(preferred)
            self._combo.setCurrentIndex(index if index >= 0 else 0)
        self.method = self._combo.currentData() or "oasis"
        self._run_id = run_id
        self._engine = engine
        self.setVisible(len(methods) > 1)

    def plot_options(self) -> dict:
        return {
            "stored_spike_methods": tuple(self.capabilities)
            if self.capabilities is not None
            else None,
            "spike_method": self.method,
            "available_metrics": (
                self.capabilities.get(self.method, set())
                if self.capabilities is not None
                else None
            ),
            "stored_spike_capabilities": self.capabilities,
        }

    def _on_changed(self) -> None:
        method = self._combo.currentData()
        if method is not None:
            self.method = method
            self.methodChanged.emit(method)
