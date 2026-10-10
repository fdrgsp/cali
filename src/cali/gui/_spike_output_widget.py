"""Extraction-owned spike output controls; construction is offline and Torch-free."""

from __future__ import annotations

import shlex
import sys
from threading import Event
from typing import TYPE_CHECKING

from qtpy.QtCore import QEvent, Qt, Signal, Slot
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import create_worker, signals_blocked

from cali.sqlmodel._spike_settings import SpikeMethod, canonical_spike_methods

from ._settings_tabs import guidance, settings_section

if TYPE_CHECKING:
    from superqt.utils import FunctionWorker

    from cali._cascade_models import CascadeModel

CASCADE_GUI_GATE = (
    "CASCADE extraction is awaiting the GUI release checks. "
    "Stored CASCADE results can be viewed without running inference."
)


def _terminal_command(arguments: list[str]) -> str:
    """Quote commands for PowerShell on Windows and POSIX shells elsewhere."""
    if sys.platform == "win32":
        return "& " + " ".join("'" + arg.replace("'", "''") + "'" for arg in arguments)
    return shlex.join(arguments)


class _OasisOutputGroup(QGroupBox):
    """Keep mandatory denoising usable when optional spike output is unchecked."""

    denoising: QWidget | None = None

    def enable_denoising(self) -> None:
        if self.denoising is not None:
            for widget in (self.denoising, *self.denoising.findChildren(QWidget)):
                widget.setEnabled(self.isEnabled())

    def event(self, event: QEvent | None) -> bool:
        handled = bool(super().event(event))
        if event is not None and event.type() in (
            QEvent.Type.ChildPolished,
            QEvent.Type.EnabledChange,
        ):
            # Qt disables unchecked groups' children during initialization and
            # ancestor enable changes. Denoising is independent of this check.
            self.enable_denoising()
        return handled


class _SpikeOutputWidget(QGroupBox):
    methodsChanged = Signal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        cascade_enabled: bool = False,
        oasis_settings: QWidget | None = None,
    ) -> None:
        super().__init__("Spike outputs — check one or both methods", parent)
        self._cascade_enabled = cascade_enabled
        self._read_only = False
        self._frame_rate = 10.0
        self._verified_model: CascadeModel | None = None
        self._download_worker: FunctionWorker[CascadeModel] | None = None
        self._download_name = ""
        self._download_cancel = Event()
        # This callback holds only the cancellation event, never the Qt widget.
        cancel_event = self._download_cancel
        self.destroyed.connect(cancel_event.set)
        self._oasis = _OasisOutputGroup("OASIS", self)
        self._cascade = QGroupBox("CASCADE", self)
        self._oasis.setCheckable(True)
        self._cascade.setCheckable(True)
        self._oasis.setToolTip(
            "Retain OASIS spike amplitudes for analysis/comparison. OASIS denoising "
            "still runs for denoised ΔF/F even when this output is unchecked."
        )
        self._cascade.setToolTip("Retain CASCADE expected spikes per frame.")
        self._model = QComboBox(self)
        self._model.setEditable(True)
        self._model.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._model.setMinimumContentsLength(20)
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
        self._refresh = QPushButton("Refresh models", self)
        self._verify = QPushButton("Verify model", self)
        self._install = QPushButton("Setup instructions...", self)
        self._install.setToolTip(
            "Show package-installation and model-download commands for this "
            "Python environment. Copy commands into your terminal, then restart "
            "cali after installation."
        )
        self._download = QPushButton("Download model", self)
        self._download.setToolTip(
            "Download the explicitly chosen model to the local cache and verify it. "
            "Selecting a model alone never downloads weights."
        )
        self._status = QLabel(self)
        self._status.setWordWrap(True)
        self._info = QLabel(self)
        self._info.setWordWrap(True)
        self._cascade_parameters = QWidget(self)
        form = QFormLayout(self._cascade_parameters)
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Pretrained model:", self._model)
        form.addRow("Compute device:", self._device)
        actions = QGridLayout()
        actions.addWidget(self._refresh, 0, 0)
        actions.addWidget(self._verify, 0, 1)
        actions.addWidget(self._download, 1, 0)
        actions.addWidget(self._install, 1, 1)
        form.addRow(actions)
        cascade_layout = QVBoxLayout(self._cascade)
        cascade_layout.addWidget(
            guidance(
                "Predicts expected spikes per frame using a pretrained neural network. "
                "Choose a model that matches your acquisition rate and indicator; use "
                "Verify model to inspect its smoothing, noise range and "
                "trace requirements."
            )
        )
        cascade_layout.addWidget(self._cascade_parameters)
        cascade_layout.addWidget(self._info)
        cascade_layout.addWidget(self._status)

        oasis_layout = QVBoxLayout(self._oasis)
        oasis_layout.addWidget(
            guidance(
                "Estimates relative spike amplitudes from calcium decay. Check this "
                "group to keep its spike output, or check both methods to compare."
            )
        )
        oasis_widgets = [oasis_settings] if oasis_settings is not None else []
        self._oasis_denoising = settings_section(
            "Calcium denoising (always runs)",
            "Denoising runs even when OASIS spike output is unchecked. Auto estimates "
            "the decay time from each trace; enter a known indicator decay time "
            "to use it instead.",
            *oasis_widgets,
        )
        oasis_layout.addWidget(self._oasis_denoising)
        self._oasis.denoising = self._oasis_denoising
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 12, 10, 10)
        self._download_feedback = QWidget(self)
        feedback_layout = QVBoxLayout(self._download_feedback)
        feedback_layout.setContentsMargins(0, 0, 0, 0)
        self._download_status = guidance("", self._download_feedback)
        feedback_layout.addWidget(self._download_status)
        progress_layout = QHBoxLayout()
        self._download_progress = QProgressBar(self._download_feedback)
        self._download_progress.setRange(0, 0)
        self._download_progress.setTextVisible(False)
        progress_layout.addWidget(self._download_progress, 1)
        self._cancel_download_btn = QPushButton(
            "Cancel download", self._download_feedback
        )
        self._cancel_download_btn.setToolTip(
            "Cancel without keeping an incomplete model. Cancellation waits for "
            "the current network operation to finish."
        )
        progress_layout.addWidget(self._cancel_download_btn)
        feedback_layout.addLayout(progress_layout)
        self._download_feedback.hide()
        layout.addWidget(self._download_feedback)
        layout.addWidget(self._cascade)
        layout.addWidget(self._oasis)
        self._oasis.toggled.connect(self._on_methods_changed)
        self._cascade.toggled.connect(self._on_methods_changed)
        self._model.currentTextChanged.connect(self._update_info)
        self._refresh.clicked.connect(self.refresh_models)
        self._verify.clicked.connect(self._verify_model)
        self._install.clicked.connect(self._show_install_instructions)
        self._download.clicked.connect(self._download_model)
        self._cancel_download_btn.clicked.connect(self._cancel_model_download)
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

    def is_downloading(self) -> bool:
        return self._download_worker is not None

    def _on_methods_changed(self) -> None:
        if not self._oasis.isChecked() and not self._cascade.isChecked():
            sender = self.sender()
            if isinstance(sender, QGroupBox):
                with signals_blocked(sender):
                    sender.setChecked(True)
        self._update_enabled()
        if self._cascade.isChecked():
            self.refresh_models()
        self.methodsChanged.emit(self.methods())

    def _update_enabled(self) -> None:
        self._oasis.setEnabled(not self._read_only)
        self._cascade.setEnabled(self._cascade_enabled and not self._read_only)
        # The check controls retained spikes, while denoising remains mandatory.
        self._oasis.enable_denoising()
        enabled = (
            self._cascade.isChecked()
            and self._cascade_enabled
            and not self._read_only
            and not self.is_downloading()
        )
        for widget in (
            self._model,
            self._device,
            self._refresh,
            self._verify,
            self._install,
        ):
            widget.setEnabled(enabled)
        self._download.setEnabled(enabled and bool(self._model.currentText().strip()))
        self._download_progress.setVisible(self.is_downloading())
        self._cancel_download_btn.setVisible(self.is_downloading())
        self._cancel_download_btn.setEnabled(
            self.is_downloading() and not self._download_cancel.is_set()
        )
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
            if (
                self._verified_model is not None
                and selected.strip() == self._verified_model.name
            ):
                self._update_info()
                self._info.setText(
                    f"{self._info.text()} Model list could not be refreshed: {error}"
                )
            else:
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
        if (
            self._verified_model is not None
            and self._model.currentText().strip() == self._verified_model.name
        ):
            self._show_model_details(self._verified_model)
            self._update_enabled()
            return
        self._verified_model = None
        index = self._model.findText(self._model.currentText())
        info = self._model.itemData(index) if index >= 0 else None
        self._info.setText(
            str(info)
            if info
            else "Model availability and rate are checked before extraction."
        )
        self._update_enabled()

    def _download_model(self) -> None:
        from cali._cascade_models import cascade_model_dir, download_cascade_model

        if (
            self.is_downloading()
            or not self._cascade_enabled
            or self._read_only
            or not self._cascade.isChecked()
            or not self.isEnabled()
        ):
            return
        name = self._model.currentText().strip()
        if not name:
            return
        cache = cascade_model_dir()
        self._download_name = name
        self._download_cancel.clear()
        self._download_status.setText(f"Downloading and verifying {name} to {cache}...")
        self._download_feedback.show()
        self._download_worker = create_worker(
            download_cascade_model,
            name,
            cache,
            cancel_requested=self._download_cancel.is_set,
            _start_thread=False,
            _connect={
                "returned": self._on_model_downloaded,
                "errored": self._on_model_download_error,
                "finished": self._on_model_download_finished,
            },
        )
        self._update_enabled()
        self._download_worker.start()

    def _cancel_model_download(self) -> None:
        if self.is_downloading():
            self._download_cancel.set()
            self._download_status.setText(
                f"Cancelling {self._download_name}; waiting for the current "
                "network read to finish..."
            )
            self._update_enabled()

    @Slot(object)  # type: ignore[untyped-decorator]
    def _on_model_downloaded(self, model: CascadeModel) -> None:
        self._download_status.setText(
            f"Verified {model.name} is cached at {model.directory}."
        )
        self.refresh_models()
        # Loading another run while downloading must not replace its model choice.
        if self._model.currentText().strip() == model.name:
            self._show_model_details(model)

    @Slot(object)  # type: ignore[untyped-decorator]
    def _on_model_download_error(self, error: Exception) -> None:
        from cali._cascade_models import CascadeDownloadCancelled

        if isinstance(error, CascadeDownloadCancelled):
            message = f"Cancelled {self._download_name}. No incomplete model was saved."
        else:
            message = f"Download failed for {self._download_name}: {error}"
        self._download_status.setText(message)

    @Slot()  # type: ignore[untyped-decorator]
    def _on_model_download_finished(self) -> None:
        self._download_worker = None
        self._update_enabled()

    def _show_install_instructions(self) -> None:
        from cali._cascade_models import cascade_model_dir
        from cali._cascade_package import CASCADE_PACKAGE_URL

        install_command = _terminal_command(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                f"CascadeTorch @ {CASCADE_PACKAGE_URL}",
            ]
        )
        command = _terminal_command(
            [
                sys.executable,
                "-m",
                "cali",
                "cascade-download",
                self._model.currentText().strip() or "<model-name>",
                "--model-dir",
                str(cascade_model_dir()),
            ]
        )
        terminal = "Windows PowerShell" if sys.platform == "win32" else "your terminal"
        instructions = (
            f"Run the commands below in {terminal}.\n\n"
            "From a cali development checkout:\n"
            "uv sync --extra cascade\n"
            "Keep any Cellpose extra flags you use, then launch with uv run cali.\n\n"
            "For an installed cali wheel, install the pinned inference dependency "
            "in this GUI's Python environment:\n"
            f"{install_command}\n\n"
            "Restart cali after installing. Then choose a model and use Download "
            "model in the GUI, or download and verify it with:\n"
            f"{command}\n\n"
            "After a CLI download, use Refresh models and Verify model in the GUI."
        )
        dialog = QMessageBox(
            QMessageBox.Icon.Information,
            "CASCADE Setup",
            "CASCADE setup instructions",
            QMessageBox.StandardButton.Ok,
            self,
        )
        dialog.setTextFormat(Qt.TextFormat.PlainText)
        dialog.setInformativeText(instructions)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        dialog.exec()

    def _verify_model(self) -> None:
        from cali._cascade_models import CascadeModelError, load_cascade_model

        name = self._model.currentText().strip()
        self._verified_model = None
        if not name:
            self._info.setText("Choose a model explicitly before verifying its cache.")
            return
        try:
            model = load_cascade_model(name)
        except (CascadeModelError, OSError) as error:
            self._info.setText(str(error))
            return
        self._show_model_details(model)

    def _show_model_details(self, model: CascadeModel) -> None:
        from cali._cascade_models import cascade_model_rate_matches

        self._verified_model = model
        kernel = "causal" if model.causal_kernel else "acausal"
        rate_status = (
            f"Configured acquisition rate {self._frame_rate:g} Hz matches this "
            "model within the allowed 1%. Extraction also checks the recording's "
            "acquisition timing."
            if cascade_model_rate_matches(self._frame_rate, model.sampling_rate)
            else (
                f"Rate mismatch: configured acquisition is {self._frame_rate:g} Hz "
                f"but this model requires {model.sampling_rate:g} Hz (allowed 1%). "
                "Choose a model for your actual acquisition rate; change the "
                "configured rate only if it does not describe your recording."
            )
        )
        self._info.setText(
            f"Verified {model.name}: {model.sampling_rate:g} Hz, "
            f"smoothing {model.smoothing * 1000:g} ms, {kernel} kernel; "
            f"noise levels {', '.join(map(str, model.noise_levels))}; "
            f"{model.ensemble_size} models per noise level. "
            f"Minimum retained length: {model.minimum_frames} frames. "
            f"{rate_status}"
        )
