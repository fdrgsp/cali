from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from superqt.utils import signals_blocked

from cali.gui._util import (
    _BrowseWidget,
    create_divider_line,
    show_error_dialog,
)

if TYPE_CHECKING:
    from cali.sqlmodel._model import DetectionSettings

MODEL_TYPE = "cpsam"
try:
    from importlib.metadata import PackageNotFoundError, version

    try:
        _cp_ver_str = version("cellpose")
        _cp_major = int(_cp_ver_str.split(".")[0])
        MODEL_TYPE = "cpsam" if _cp_major >= 4 else "cyto3"
    except (PackageNotFoundError, ValueError):
        pass
except ImportError:
    pass

FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
CUSTOM_MODEL_PATH = (
    Path(__file__).parent.parent
    / "detection"
    / "cellpose_models"
    / "cp3_img8_epoch7000_py"
)


@dataclass(frozen=True)
class CellposeSettingsData:
    model_type: str = MODEL_TYPE
    model_path: str | None = None
    diameter: float | None = None
    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.4
    min_size: int = 10
    normalize: bool = True
    batch_size: int = 8
    use_gpu: bool = True


class _DetectionGUI(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # MAIN WIDGET -----------------------------------------------------------------
        group_wdg = QGroupBox(self)
        group_layout = QVBoxLayout(group_wdg)
        group_layout.setContentsMargins(10, 10, 10, 10)
        group_layout.setSpacing(5)

        # CELLPOSE WIDGET -------------------------------------------------------------
        self._cellpose_wdg = _CellposeDetectionWidget(self)
        self._cellpose_wdg.setCheckable(True)
        self._cellpose_wdg.setChecked(True)

        # IMPORTED LABELS WIDGET ------------------------------------------------------
        self._imported_labels_wdg = _ImportedLabelsWidget(self)

        # MUTUAL EXCLUSION ------------------------------------------------------------
        self._cellpose_wdg.toggled.connect(self._on_cellpose_toggled)
        self._imported_labels_wdg.toggled.connect(self._on_imported_toggled)

        # SCROLL AREA WIDGET ---------------------------------------------------------
        detection_scroll_area = QScrollArea()
        detection_scroll_area.setWidgetResizable(True)
        detection_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        detection_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        # add detection widgets to scroll area
        group_layout.addWidget(self._cellpose_wdg)
        group_layout.addWidget(self._imported_labels_wdg)
        group_layout.addStretch(1)
        detection_scroll_area.setWidget(group_wdg)

        # MAIN LAYOUT -----------------------------------------------------------------
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(15)
        main_layout.addWidget(detection_scroll_area)

        # STYLING ---------------------------------------------------------------------
        cp = self._cellpose_wdg
        fixed_lbl_width = cp._cellprob_label.sizeHint().width()
        cp._models_combo_label.setMinimumWidth(fixed_lbl_width)
        cp._browse_custom_model._label.setMinimumWidth(fixed_lbl_width)
        cp._diameter_label.setMinimumWidth(fixed_lbl_width)
        cp._cellprob_label.setMinimumWidth(fixed_lbl_width)
        cp._flow_label.setMinimumWidth(fixed_lbl_width)
        cp._min_size_label.setMinimumWidth(fixed_lbl_width)
        cp._batch_label.setMinimumWidth(fixed_lbl_width)
        cp._normalize_label.setMinimumWidth(fixed_lbl_width)
        cp._use_gpu_label.setMinimumWidth(fixed_lbl_width)

    # PUBLIC METHODS ------------------------------------------------------------------

    def active_method(self) -> Literal["cellpose", "imported_labels"]:
        """Return which detection method is currently active."""
        if self._cellpose_wdg.isChecked():
            return "cellpose"
        return "imported_labels"

    def value(self) -> CellposeSettingsData:
        """Return the detection parameters of the selected method."""
        return self._cellpose_wdg.value()

    def setValue(
        self,
        value: CellposeSettingsData | None = None,
        *,
        method: str = "cellpose",
    ) -> None:
        """Set the detection parameters of the selected method."""
        if method == "imported_labels":
            with signals_blocked(self._imported_labels_wdg):
                self._imported_labels_wdg.setChecked(True)
            with signals_blocked(self._cellpose_wdg):
                self._cellpose_wdg.setChecked(False)
        else:
            if value is not None:
                self._cellpose_wdg.setValue(value)
            with signals_blocked(self._cellpose_wdg):
                self._cellpose_wdg.setChecked(True)
            with signals_blocked(self._imported_labels_wdg):
                self._imported_labels_wdg.setChecked(False)

    def enable(self, enabled: bool) -> None:
        """Enable or disable the detection GUI."""
        self._cellpose_wdg.setEnabled(enabled)
        self._imported_labels_wdg.setEnabled(enabled)

    def reset(self) -> None:
        """Reset the detection GUI to default values."""
        self._cellpose_wdg.setValue(CellposeSettingsData())
        self._imported_labels_wdg.reset()
        with signals_blocked(self._cellpose_wdg):
            self._cellpose_wdg.setChecked(True)
        with signals_blocked(self._imported_labels_wdg):
            self._imported_labels_wdg.setChecked(False)

    def to_model_settings(self) -> DetectionSettings:
        """Convert current GUI settings to DetectionSettings model.

        Returns
        -------
        DetectionSettings
            The detection settings model instance.
        """
        from datetime import datetime

        from cali.sqlmodel import DetectionSettings

        if self.active_method() == "imported_labels":
            return DetectionSettings(
                created_at=datetime.now(),
                method="imported_labels",
            )

        settings = self.value()
        return DetectionSettings(
            created_at=datetime.now(),
            method="cellpose",
            model_type=settings.model_type,
            custom_model=(
                settings.model_path if settings.model_type == "custom" else None
            ),
            diameter=None if settings.diameter == 0 else settings.diameter,
            cellprob_threshold=settings.cellprob_threshold,
            flow_threshold=settings.flow_threshold,
            min_size=settings.min_size,
            normalize=settings.normalize,
            batch_size=settings.batch_size,
            use_gpu=settings.use_gpu,
        )

    # PRIVATE METHODS -----------------------------------------------------------------

    def _on_cellpose_toggled(self, checked: bool) -> None:
        """When cellpose is checked, uncheck imported labels."""
        if checked:
            with signals_blocked(self._imported_labels_wdg):
                self._imported_labels_wdg.setChecked(False)
        elif not self._imported_labels_wdg.isChecked():
            # Don't allow both unchecked - re-check cellpose
            with signals_blocked(self._cellpose_wdg):
                self._cellpose_wdg.setChecked(True)

    def _on_imported_toggled(self, checked: bool) -> None:
        """When imported labels is checked, uncheck cellpose."""
        if checked:
            with signals_blocked(self._cellpose_wdg):
                self._cellpose_wdg.setChecked(False)
        elif not self._cellpose_wdg.isChecked():
            # Don't allow both unchecked - re-check imported
            with signals_blocked(self._imported_labels_wdg):
                self._imported_labels_wdg.setChecked(True)


class _SelectModelPath(_BrowseWidget):
    def __init__(
        self,
        parent: QWidget | None = None,
        label: str = "Custom Model",
        tooltip: str = "Choose the path to the custom Cellpose model.",
    ) -> None:
        super().__init__(parent, label, "", tooltip, is_dir=False)

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select the {self._label_text}.",
            "",
            "",
        )
        if path:
            self._path.setText(path)


class _CellposeDetectionWidget(QGroupBox):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setTitle("Cellpose")

        # Check installation
        cp_ver_str = None
        cp_major = 0
        try:
            from importlib.metadata import PackageNotFoundError, version

            try:
                cp_ver_str = version("cellpose")
                cp_major = int(cp_ver_str.split(".")[0])
            except (PackageNotFoundError, ValueError):
                pass
        except ImportError:
            pass

        self._cp_is_installed = cp_ver_str is not None

        # MODEL SELECTION WIDGETS -----------------------------------------------------
        self._model_wdg = QWidget(self)
        model_wdg_layout = QHBoxLayout(self._model_wdg)
        model_wdg_layout.setContentsMargins(0, 0, 0, 0)
        model_wdg_layout.setSpacing(5)
        self._models_combo_label = QLabel("Model Type:", self._model_wdg)
        self._models_combo_label.setSizePolicy(*FIXED)
        self._models_combo = QComboBox(self._model_wdg)
        models = ["cpsam"] if cp_major >= 4 else ["cyto3"]
        models.append("custom")

        self._models_combo.addItems(models)
        self._models_combo.currentTextChanged.connect(self._on_model_combo_changed)
        model_wdg_layout.addWidget(self._models_combo_label)
        model_wdg_layout.addWidget(self._models_combo, 1)

        self._browse_custom_model = _SelectModelPath(self)
        self._browse_custom_model.setValue(CUSTOM_MODEL_PATH)
        self._browse_custom_model.hide()

        # DIAMETER WIDGETS ------------------------------------------------------------
        self._diameter_wdg = QWidget(self)
        self._diameter_wdg.setToolTip(
            "Set the diameter of the cells. Leave 0 for automatic detection."
        )
        diameter_layout = QHBoxLayout(self._diameter_wdg)
        diameter_layout.setContentsMargins(0, 0, 0, 0)
        diameter_layout.setSpacing(5)
        self._diameter_label = QLabel("Diameter:", self._diameter_wdg)
        self._diameter_label.setSizePolicy(*FIXED)
        self._diameter_spin = QDoubleSpinBox(self._diameter_wdg)
        self._diameter_spin.setSpecialValueText("Auto")
        self._diameter_spin.setRange(0, 1000)
        self._diameter_spin.setValue(0)
        diameter_layout.addWidget(self._diameter_label)
        diameter_layout.addWidget(self._diameter_spin)

        # CELLPOSE THRESHOLDS ---------------------------------------------------------
        self._cellprob_wdg = QWidget(self)
        self._cellprob_wdg.setToolTip(
            "Cell probability threshold (all pixels > threshold are used "
            "for dynamics). Lower values detect more masks. Default is 0.0 ("
            "cellpose default)."
        )
        prob_layout = QHBoxLayout(self._cellprob_wdg)
        prob_layout.setContentsMargins(0, 0, 0, 0)
        prob_layout.setSpacing(5)
        self._cellprob_label = QLabel("Cell Probability Threshold:", self._cellprob_wdg)
        self._cellprob_label.setSizePolicy(*FIXED)
        self._cellprob_threshold_spin = QDoubleSpinBox(self._cellprob_wdg)
        self._cellprob_threshold_spin.setRange(-6.0, 6.0)
        self._cellprob_threshold_spin.setValue(0.0)
        self._cellprob_threshold_spin.setSingleStep(0.1)
        prob_layout.addWidget(self._cellprob_label)
        prob_layout.addWidget(self._cellprob_threshold_spin)

        self._flow_wdg = QWidget(self)
        self._flow_wdg.setToolTip(
            "Flow error threshold (all cells with errors below threshold are kept). "
            "Higher values detect more masks."
        )
        flow_layout = QHBoxLayout(self._flow_wdg)
        flow_layout.setContentsMargins(0, 0, 0, 0)
        flow_layout.setSpacing(5)
        self._flow_label = QLabel("Flow Threshold:", self._flow_wdg)
        self._flow_label.setSizePolicy(*FIXED)
        self._flow_threshold_spin = QDoubleSpinBox(self._flow_wdg)
        self._flow_threshold_spin.setRange(0.0, 3.0)
        self._flow_threshold_spin.setValue(0.4)
        self._flow_threshold_spin.setSingleStep(0.1)
        self._flow_threshold_spin.setToolTip(
            "Flow error threshold (all cells with errors below threshold are kept). "
            "Higher values detect more masks. Default is 0.4 (cellpose default)."
        )
        flow_layout.addWidget(self._flow_label)
        flow_layout.addWidget(self._flow_threshold_spin)

        # MIN SIZE WIDGET -------------------------------------------------------------
        self._min_size_wdg = QWidget(self)
        self._min_size_wdg.setToolTip(
            "Minimum number of pixels for a mask to be kept. Masks smaller than "
            "this will be removed as they are likely artifacts or debris. "
            "Default is 15 pixels. Set to 1 to keep all masks."
        )
        min_size_layout = QHBoxLayout(self._min_size_wdg)
        min_size_layout.setContentsMargins(0, 0, 0, 0)
        min_size_layout.setSpacing(5)
        self._min_size_label = QLabel("Min Mask Size:", self._min_size_wdg)
        self._min_size_label.setSizePolicy(*FIXED)
        self._min_size_spin = QSpinBox(self._min_size_wdg)
        self._min_size_spin.setSuffix(" pixels")
        self._min_size_spin.setRange(1, 10000)
        self._min_size_spin.setValue(15)
        min_size_layout.addWidget(self._min_size_label)
        min_size_layout.addWidget(self._min_size_spin)

        # NORMALIZE CHECKBOX ----------------------------------------------------------
        self._normalize_wdg = QWidget(self)
        self._normalize_wdg.setToolTip(
            "Normalize images before segmentation. "
            "This rescales pixel values to 0-1 range using 1st and 99th percentiles.\n"
            "By default, this is enabled (cellpose default)."
        )
        normalize_layout = QHBoxLayout(self._normalize_wdg)
        normalize_layout.setContentsMargins(0, 0, 0, 0)
        normalize_layout.setSpacing(5)
        self._normalize_label = QLabel("Normalize Images:", self._normalize_wdg)
        self._normalize_label.setSizePolicy(*FIXED)
        self._normalize_checkbox = QCheckBox(self._normalize_wdg)
        self._normalize_checkbox.setChecked(True)
        normalize_layout.addWidget(self._normalize_label)
        normalize_layout.addWidget(self._normalize_checkbox)
        normalize_layout.addStretch(1)

        # USE GPU CHECKBOX ------------------------------------------------------------
        self._use_gpu_wdg = QWidget(self)
        self._use_gpu_wdg.setToolTip(
            "Use GPU acceleration for faster processing if available.\n"
            "If GPU is not available, CPU will be used automatically."
        )
        use_gpu_layout = QHBoxLayout(self._use_gpu_wdg)
        use_gpu_layout.setContentsMargins(0, 0, 0, 0)
        use_gpu_layout.setSpacing(5)
        self._use_gpu_label = QLabel("Use GPU:", self._use_gpu_wdg)
        self._use_gpu_label.setSizePolicy(*FIXED)
        self._use_gpu_checkbox = QCheckBox(self._use_gpu_wdg)
        self._use_gpu_checkbox.setChecked(True)
        use_gpu_layout.addWidget(self._use_gpu_label)
        use_gpu_layout.addWidget(self._use_gpu_checkbox)
        use_gpu_layout.addStretch(1)

        # BATCH SIZE WIDGET -----------------------------------------------------------
        self._batch_wdg = QWidget(self)
        self._batch_wdg.setToolTip(
            "Number of images to process per batch. Higher values are faster "
            "but use more memory."
        )
        batch_layout = QHBoxLayout(self._batch_wdg)
        batch_layout.setContentsMargins(0, 0, 0, 0)
        batch_layout.setSpacing(5)
        self._batch_label = QLabel("Batch Size:", self._batch_wdg)
        self._batch_label.setSizePolicy(*FIXED)
        self._batch_size_spin = QSpinBox(self._batch_wdg)
        self._batch_size_spin.setRange(1, 32)
        self._batch_size_spin.setValue(8)
        batch_layout.addWidget(self._batch_label)
        batch_layout.addWidget(self._batch_size_spin)

        # LAYOUT ----------------------------------------------------------------------
        cp_wdg_layout = QVBoxLayout(self)
        cp_wdg_layout.setContentsMargins(10, 10, 10, 10)
        cp_wdg_layout.setSpacing(5)

        if cp_ver_str is None:
            warning_lbl = QLabel(
                "Cellpose is not installed!\n\n"
                "To use Cellpose detection, please install it:\n"
                "• For Cellpose 4: `uv sync --extra cp4`\n"
                "• For Cellpose 3: `uv sync --extra cp3`\n"
                "• Or via pip: `pip install cellpose`"
            )
            warning_lbl.setWordWrap(True)
            warning_lbl.setStyleSheet("color: #FF5555; font-weight: bold;")
            cp_wdg_layout.addWidget(warning_lbl)

            # Hide widgets
            self._model_wdg.hide()
            self._browse_custom_model.hide()
            self._diameter_wdg.hide()
            self._cellprob_wdg.hide()
            self._flow_wdg.hide()
            self._min_size_wdg.hide()
            self._normalize_wdg.hide()
            self._use_gpu_wdg.hide()
            self._batch_wdg.hide()
        else:
            cp_wdg_layout.addWidget(create_divider_line("Select Cellpose Model"))
            cp_wdg_layout.addWidget(self._model_wdg)
            cp_wdg_layout.addWidget(self._browse_custom_model)
            cp_wdg_layout.addWidget(create_divider_line("Cellpose Parameters"))
            cp_wdg_layout.addWidget(self._diameter_wdg)
            cp_wdg_layout.addWidget(self._cellprob_wdg)
            cp_wdg_layout.addWidget(self._flow_wdg)
            cp_wdg_layout.addWidget(self._min_size_wdg)
            cp_wdg_layout.addWidget(self._normalize_wdg)
            cp_wdg_layout.addWidget(self._use_gpu_wdg)
            cp_wdg_layout.addWidget(create_divider_line("Batch Processing"))
            cp_wdg_layout.addWidget(self._batch_wdg)

    # PUBLIC METHODS ------------------------------------------------------------------

    def value(self) -> CellposeSettingsData:
        """Return the current Cellpose parameters as a CellposeData object."""
        model_type = self._models_combo.currentText()
        model_path = self._browse_custom_model.value() if model_type == "custom" else ""
        diameter = self._diameter_spin.value()
        cellprob_threshold = self._cellprob_threshold_spin.value()
        flow_threshold = self._flow_threshold_spin.value()
        min_size = self._min_size_spin.value()
        normalize = self._normalize_checkbox.isChecked()
        batch_size = self._batch_size_spin.value()
        use_gpu = self._use_gpu_checkbox.isChecked()

        return CellposeSettingsData(
            model_type=model_type,
            model_path=model_path,
            diameter=None if diameter == 0 else diameter,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            min_size=min_size,
            normalize=normalize,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )

    def setValue(self, value: CellposeSettingsData) -> None:
        """Set the Cellpose parameters from a CellposeData object."""
        if not self._cp_is_installed:
            return

        # Set model type using findText to ensure it exists
        model_idx = self._models_combo.findText(value.model_type)
        if model_idx >= 0:
            self._models_combo.setCurrentIndex(model_idx)

        # Set custom model path if applicable
        if value.model_type == "custom" and value.model_path is not None:
            self._browse_custom_model.setValue(value.model_path)

        # Set other parameters
        self._diameter_spin.setValue(0 if value.diameter is None else value.diameter)
        self._cellprob_threshold_spin.setValue(value.cellprob_threshold)
        self._flow_threshold_spin.setValue(value.flow_threshold)
        self._min_size_spin.setValue(value.min_size)
        self._normalize_checkbox.setChecked(value.normalize)
        self._batch_size_spin.setValue(value.batch_size)
        self._use_gpu_checkbox.setChecked(value.use_gpu)

    # PRIVATE METHODS -----------------------------------------------------------------

    def _on_model_combo_changed(self, text: str) -> None:
        """Show or hide the custom model path widget."""
        if not self._cp_is_installed:
            return

        if text == "custom":
            self._browse_custom_model.show()
        else:
            self._browse_custom_model.hide()


class _ImportedLabelsWidget(QGroupBox):
    """Widget for importing pre-existing label TIFFs."""

    labelsImported = Signal(int)  # emits detection_settings_id

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setTitle("Imported Labels")
        self.setCheckable(True)
        self.setChecked(False)

        self.setToolTip(
            "Import pre-existing label TIFFs files and associate them to each Well and "
            "FOV in the plate."
        )

        self._database_path: str | None = None
        self._detection_settings_id: int | None = None
        self._n_imported_fovs: int = 0

        self._status_label = QLabel("No labels imported yet.")
        self._status_label.setSizePolicy(*FIXED)
        self._import_btn = QPushButton("Import Labels...")
        self._import_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._import_btn.clicked.connect(self._on_import_clicked)

        btn_status_layout = QHBoxLayout()
        btn_status_layout.setContentsMargins(0, 0, 0, 0)
        btn_status_layout.setSpacing(5)
        btn_status_layout.addWidget(self._import_btn)
        btn_status_layout.addWidget(self._status_label)
        btn_status_layout.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        layout.addWidget(create_divider_line("Import Label TIFFs"))
        layout.addLayout(btn_status_layout)

    # PUBLIC METHODS ------------------------------------------------------------------

    def set_database_path(self, path: str | None) -> None:
        """Set the active database path for the import dialog."""
        self._database_path = path

    def detection_settings_id(self) -> int | None:
        """Return the detection_settings_id from the last import."""
        return self._detection_settings_id

    def set_detection_settings_id(self, det_id: int | None) -> None:
        """Set the detection_settings_id (e.g. when restoring from a run)."""
        self._detection_settings_id = det_id
        if det_id is not None:
            self._status_label.setText(f"Previously imported (Detection ID: {det_id})")
        else:
            self._status_label.setText("No labels imported yet.")

    def reset(self) -> None:
        """Reset the imported labels widget."""
        self._detection_settings_id = None
        self._n_imported_fovs = 0
        self._status_label.setText("No labels imported yet.")

    # PRIVATE METHODS -----------------------------------------------------------------

    def _on_import_clicked(self) -> None:
        """Open the import labels dialog."""
        if not self._database_path:
            show_error_dialog(self, "Please load a database first.")
            return

        from cali.gui._import_labels_dialog import (
            _ImportLabelsDialog,  # pragma: no cover
        )

        dialog = _ImportLabelsDialog(
            self._database_path, parent=self
        )  # pragma: no cover
        if dialog.exec():  # pragma: no cover
            self._detection_settings_id = dialog._imported_detection_settings_id
            self._n_imported_fovs = len(dialog.value())
            self._status_label.setText(
                f"{self._n_imported_fovs} FOV(s) imported "
                f"(Detection ID: {self._detection_settings_id})"
            )
            if self._detection_settings_id is not None:
                self.labelsImported.emit(self._detection_settings_id)
