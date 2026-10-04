"""Extraction-owned spike output controls; construction is offline and Torch-free."""

from __future__ import annotations

import shlex

from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QWidget,
)
from superqt.utils import signals_blocked

from cali.sqlmodel._spike_settings import SpikeMethod, canonical_spike_methods

CASCADE_GUI_GATE = (
    "CASCADE extraction is awaiting the GUI release checks. "
    "Stored CASCADE results can be viewed without running inference."
)


class _SpikeOutputWidget(QGroupBox):
    methodsChanged = Signal(object)

    def __init__(
        self, parent: QWidget | None = None, *, cascade_enabled: bool = False
    ) -> None:
        super().__init__("Spike Outputs", parent)
        self._cascade_enabled = cascade_enabled
        self._read_only = False
        self._frame_rate = 10.0
        self._oasis = QCheckBox("OASIS", self)
        self._cascade = QCheckBox("CASCADE", self)
        self._oasis.setToolTip(
            "Retain OASIS spike amplitudes for analysis/comparison. OASIS denoising "
            "still runs for denoised ΔF/F even when this output is unchecked."
        )
        self._cascade.setToolTip("Retain CASCADE expected spikes per frame.")
        outputs = QHBoxLayout()
        outputs.addWidget(self._cascade)
        outputs.addWidget(self._oasis)
        outputs.addStretch()
        self._model = QComboBox(self)
        self._model.setEditable(True)
        self._model.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._model.setToolTip(
            "Choose a model explicitly. The offline catalogue is filtered by the "
            "configured acquisition rate; cached model configuration is verified "
            "again before inference. No nearest-rate model is chosen automatically."
        )
        self._device = QComboBox(self)
        for label, device in (
            ("Automatic", "auto"),
            ("CPU", "cpu"),
            ("CUDA", "cuda"),
            ("Apple MPS", "mps"),
        ):
            self._device.addItem(label, device)
        self._refresh = QPushButton("Refresh Local Models", self)
        self._verify = QPushButton("Verify Model / Show Details", self)
        self._install = QPushButton("Install / Download Instructions", self)
        self._status = QLabel(self)
        self._status.setWordWrap(True)
        self._info = QLabel(self)
        self._info.setWordWrap(True)
        layout = QFormLayout(self)
        layout.addRow(outputs)
        layout.addRow("CASCADE Model:", self._model)
        layout.addRow("CASCADE Device:", self._device)
        actions = QHBoxLayout()
        actions.addWidget(self._refresh)
        actions.addWidget(self._verify)
        actions.addWidget(self._install)
        layout.addRow(actions)
        layout.addRow(self._info)
        layout.addRow(self._status)
        self._oasis.toggled.connect(self._on_methods_changed)
        self._cascade.toggled.connect(self._on_methods_changed)
        self._model.currentTextChanged.connect(self._update_info)
        self._refresh.clicked.connect(self.refresh_models)
        self._verify.clicked.connect(self._verify_model)
        self._install.clicked.connect(self._show_install_instructions)
        self.reset()

    def methods(self) -> tuple[SpikeMethod, ...]:
        return canonical_spike_methods(
            [
                method
                for method, checked in (
                    ("oasis", self._oasis.isChecked()),
                    ("cascade", self._cascade.isChecked()),
                )
                if checked
            ]
        )

    def value(self) -> tuple[tuple[SpikeMethod, ...], str | None, str]:
        methods = self.methods()
        return (
            methods,
            self._model.currentText().strip() or None if "cascade" in methods else None,
            self._device.currentData() if "cascade" in methods else "auto",
        )

    def setValue(
        self, methods: tuple[SpikeMethod, ...], model: str | None, device: str
    ) -> None:
        methods = canonical_spike_methods(methods)
        device_index = self._device.findData(device)
        if device_index < 0:
            raise ValueError(f"Unknown CASCADE device: {device}")
        with signals_blocked(self._oasis), signals_blocked(self._cascade):
            self._oasis.setChecked("oasis" in methods)
            self._cascade.setChecked("cascade" in methods)
        self._model.setCurrentText(model or "")
        self._device.setCurrentIndex(device_index)
        self._update_enabled()
        self.methodsChanged.emit(methods)

    def reset(self) -> None:
        self.setValue(
            ("cascade",) if self._cascade_enabled else ("oasis",), None, "auto"
        )
        if self._cascade_enabled:
            self.refresh_models()

    def setReadOnly(self, read_only: bool) -> None:
        self._read_only = read_only
        self._update_enabled()

    def set_frame_rate(self, frame_rate: float) -> None:
        self._frame_rate = frame_rate
        if self._cascade.isChecked():
            self.refresh_models()

    def _on_methods_changed(self) -> None:
        if not self._oasis.isChecked() and not self._cascade.isChecked():
            sender = self.sender()
            if isinstance(sender, QCheckBox):
                with signals_blocked(sender):
                    sender.setChecked(True)
        self._update_enabled()
        if self._cascade.isChecked():
            self.refresh_models()
        self.methodsChanged.emit(self.methods())

    def _update_enabled(self) -> None:
        self._oasis.setEnabled(not self._read_only)
        self._cascade.setEnabled(self._cascade_enabled and not self._read_only)
        enabled = (
            self._cascade.isChecked() and self._cascade_enabled and not self._read_only
        )
        for widget in (
            self._model,
            self._device,
            self._refresh,
            self._verify,
            self._install,
        ):
            widget.setEnabled(enabled)
        self._status.setText(
            "Stored outputs; changing them requires re-extraction."
            if self._read_only
            else ("" if self._cascade_enabled else CASCADE_GUI_GATE)
        )

    def refresh_models(self) -> None:
        from cali._cascade_models import (
            CascadeModelError,
            compatible_cascade_models,
            get_cascade_catalogue,
        )

        selected = self._model.currentText()
        try:
            entries = compatible_cascade_models(
                get_cascade_catalogue(), self._frame_rate
            )
        except (CascadeModelError, OSError) as error:
            self._info.setText(str(error))
            return
        with signals_blocked(self._model):
            self._model.clear()
            self._model.addItem("", "Choose a model explicitly.")
            for entry in entries:
                self._model.addItem(entry.name, entry.info)
            # Preserve an unavailable or rate-incompatible selection for preflight.
            self._model.setCurrentText(selected)
        self._update_info()

    def _update_info(self) -> None:
        index = self._model.findText(self._model.currentText())
        info = self._model.itemData(index) if index >= 0 else None
        self._info.setText(
            str(info)
            if info
            else "Model availability and rate are checked before extraction."
        )

    def _show_install_instructions(self) -> None:
        from cali._cascade_models import cascade_model_dir

        command = shlex.join(
            [
                "cali",
                "cascade-download",
                self._model.currentText().strip() or "<model-name>",
                "--model-dir",
                str(cascade_model_dir()),
            ]
        )
        QMessageBox.information(
            self,
            "CASCADE Installation",
            "Install the optional inference dependency:\n"
            "python -m pip install 'cali[cascade]'\n\n"
            f"Download and verify the chosen model:\n{command}\n\n"
            "Then refresh the local models. Selecting CASCADE never downloads "
            "weights automatically or falls back to OASIS.",
        )

    def _verify_model(self) -> None:
        from cali._cascade_models import CascadeModelError, load_cascade_model

        name = self._model.currentText().strip()
        if not name:
            self._info.setText("Choose a model explicitly before verifying its cache.")
            return
        try:
            model = load_cascade_model(name)
        except (CascadeModelError, OSError) as error:
            self._info.setText(str(error))
            return
        kernel = "causal" if model.causal_kernel else "acausal"
        self._info.setText(
            f"Verified {model.name}: {model.sampling_rate:g} Hz, "
            f"smoothing {model.smoothing * 1000:g} ms, {kernel} kernel; "
            f"noise levels {', '.join(map(str, model.noise_levels))}; "
            f"{model.ensemble_size} models per noise level. "
            f"Minimum retained length: {model.minimum_frames} frames. "
            "The measured acquisition rate must match this model before inference."
        )
