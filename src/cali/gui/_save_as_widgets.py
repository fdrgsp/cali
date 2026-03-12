from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ._util import _BrowseWidget, parse_lineedit_text, show_error_dialog

if TYPE_CHECKING:
    from ._cali_gui import CaliGui


class _SaveAsTiff(QDialog):
    def __init__(self, parent: CaliGui | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save As Tiff")

        # position selection widget
        pos_label = QLabel("Positions:")
        pos_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._pos_line_edit = QLineEdit()
        self._pos_line_edit.setPlaceholderText("0-10 or 0, 1, 2")
        tooltip = (
            "Select the Positions to save as .tiff."
            "\nLeave blank to segment all Positions.\n"
            "You can input single Positions (e.g. 30, 33) a range "
            "(e.g. 1-10), or a mix of single Positions and ranges "
            "(e.g. 1-10, 30, 50-65).\n"
            "NOTE: The Positions are 0-indexed."
        )
        pos_label.setToolTip(tooltip)
        self._pos_line_edit.setToolTip(tooltip)
        pos_wdg = QWidget()
        pos_layout = QHBoxLayout(pos_wdg)
        pos_layout.addWidget(pos_label)
        pos_layout.addWidget(self._pos_line_edit)
        pos_layout.setContentsMargins(0, 0, 0, 0)
        pos_layout.setSpacing(5)

        # save folder selection widget
        self._browse_widget = _BrowseWidget(
            self, "Save Path", "", "Select the path to save the .tiff files."
        )

        # ok, cancel buttons
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)

        # main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(pos_wdg)
        main_layout.addWidget(self._browse_widget)
        main_layout.addWidget(self._button_box)

    def accept(self) -> Any:
        """Override QDialog accept method."""
        path, _ = self.value()
        if not path or not Path(path).is_dir() or not Path(path).exists():
            show_error_dialog(self, "Please select a path to save the .tiff files.")
            return
        return super().accept()

    def value(self) -> tuple[str, list[int]]:
        """Return the selected path and positions list."""
        positions = parse_lineedit_text(self._pos_line_edit.text())
        return self._browse_widget.value(), positions


class _SaveLabelsAsTiff(QDialog):
    """Dialog to save labeled images (ROI masks) as TIFF files."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save Labels as Tiff")

        FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed

        # position selection widget
        pos_label = QLabel("Positions:")
        pos_label.setSizePolicy(*FIXED)
        self._pos_line_edit = QLineEdit()
        self._pos_line_edit.setPlaceholderText(
            "e.g. 0-10, 30, 33. Leave empty for all."
        )
        tooltip = (
            "Select the Positions to save labeled images for.\n"
            "Leave blank to save all Positions.\n"
            "You can input single Positions (e.g. 30, 33), a range "
            "(e.g. 1-10), or a mix (e.g. 1-10, 30, 50-65).\n"
            "NOTE: The Positions are 0-indexed."
        )
        pos_label.setToolTip(tooltip)
        self._pos_line_edit.setToolTip(tooltip)
        pos_wdg = QWidget()
        pos_layout = QHBoxLayout(pos_wdg)
        pos_layout.addWidget(pos_label)
        pos_layout.addWidget(self._pos_line_edit)
        pos_layout.setContentsMargins(0, 0, 0, 0)
        pos_layout.setSpacing(5)

        # detection settings selector
        det_label = QLabel("Detection ID:")
        det_label.setSizePolicy(*FIXED)
        self._detection_combo = QComboBox()
        self._detection_combo.setToolTip(
            "Select which detection results to use for the labeled images.\n"
            "If 'All' is selected, ROIs from all detection settings are included."
        )
        self._detection_combo.addItem("All", None)
        det_wdg = QWidget()
        det_layout = QHBoxLayout(det_wdg)
        det_layout.addWidget(det_label)
        det_layout.addWidget(self._detection_combo)
        det_layout.setContentsMargins(0, 0, 0, 0)
        det_layout.setSpacing(5)

        # save folder selection widget
        self._browse_widget = _BrowseWidget(
            self,
            "Save Path",
            "",
            "Select the directory to save the labeled .tiff files.",
        )

        # overwrite checkbox
        self._overwrite_cb = QCheckBox("Overwrite existing files")
        self._overwrite_cb.setToolTip(
            "If checked, existing labeled TIFF files will be overwritten."
        )

        # ok, cancel buttons
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)

        # main layout
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(pos_wdg)
        main_layout.addWidget(det_wdg)
        main_layout.addWidget(self._browse_widget)
        main_layout.addWidget(self._overwrite_cb)
        main_layout.addWidget(self._button_box)

    def populate_detection_settings(self, settings_list: list[tuple[int, str]]) -> None:
        """Populate the detection settings combobox.

        Parameters
        ----------
        settings_list : list[tuple[int, str]]
            List of (id, method) tuples for available detection settings.
        """
        self._detection_combo.clear()
        self._detection_combo.addItem("All", None)
        for settings_id, method in settings_list:
            self._detection_combo.addItem(
                f"Detection ID {settings_id} ({method})", settings_id
            )
        # Auto-select if only one detection available
        if len(settings_list) == 1:
            self._detection_combo.setCurrentIndex(1)

    def accept(self) -> Any:
        """Override QDialog accept method."""
        path = self._browse_widget.value()
        if not path or not Path(path).is_dir() or not Path(path).exists():
            show_error_dialog(
                self, "Please select a directory to save the labeled .tiff files."
            )
            return
        return super().accept()

    def value(self) -> tuple[str, list[int], int | None, bool]:
        """Return the selected path, positions, detection settings ID, and overwrite.

        Returns
        -------
        tuple[str, list[int], int | None, bool]
            (output_path, positions, detection_settings_id, overwrite)
        """
        positions = parse_lineedit_text(self._pos_line_edit.text())
        detection_settings_id = self._detection_combo.currentData()
        overwrite = self._overwrite_cb.isChecked()
        return self._browse_widget.value(), positions, detection_settings_id, overwrite
