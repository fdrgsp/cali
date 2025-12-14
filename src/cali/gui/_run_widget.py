from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon, QStandardItemModel
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt import QIconifyIcon

from ._util import create_divider_line, parse_lineedit_text

FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed


class _ChoosePositionsWidget(QWidget):
    """Widget to select the positions to analyze."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setToolTip(
            "Select the Positions to analyze. Leave blank to analyze all Positions.\n\n"
            "You can input single Positions (e.g. 30, 33), a range (e.g. 1-10), or a "
            "mix of single Positions and ranges (e.g. 1-10, 30, 50-65).\nLeave empty "
            "to analyze all Positions.\n\n"
            "NOTE: The Positions are 0-indexed."
        )

        self._pos_lbl = QLabel("Analyze Positions:")
        self._pos_lbl.setSizePolicy(*FIXED)
        self._pos_le = QLineEdit(self)
        self._pos_le.setPlaceholderText("e.g. 0-10, 30, 33. Leave empty for all.")

        pos_layout = QHBoxLayout(self)
        pos_layout.setContentsMargins(0, 0, 0, 0)
        pos_layout.setSpacing(5)
        pos_layout.addWidget(self._pos_lbl)
        pos_layout.addWidget(self._pos_le)

    # PUBLIC METHODS ------------------------------------------------------------

    def set_labels_width(self, width: int) -> None:
        """Set the width of the label."""
        self._pos_lbl.setFixedWidth(width)

    def value(self) -> str:
        """Get the current value of the positions line edit."""
        return cast("str", self._pos_le.text())

    def setValue(self, value: str) -> None:
        """Set the value of the positions line edit."""
        self._pos_le.setText(value)


@dataclass(frozen=True)
class CaliRunSettings:
    positions: list[int]
    run_detection: bool
    run_extraction: bool
    run_analysis: bool
    detection_settings_id: int | None
    extraction_settings_id: int | None


class _RunCaliWidget(QWidget):
    """Widget to display progress and control the execution of detection/analysis."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # progress bar
        self._progress_bar = QProgressBar(self)
        self._progress_pos_label = QLabel()
        self._elapsed_time_label = QLabel("00:00:00")

        # buttons
        from cali._constants import GREEN, RED

        self._save_settings_btn = QPushButton()
        self._save_settings_btn.setSizePolicy(*FIXED)
        self._save_settings_btn.setIcon(QIcon(QIconifyIcon("mdi:content-save-cog")))
        self._load_settings_btn = QPushButton()
        self._load_settings_btn.setSizePolicy(*FIXED)
        self._load_settings_btn.setIcon(QIcon(QIconifyIcon("mdi:cog-clockwise")))

        self._run_btn = QPushButton("Run")
        self._run_btn.setSizePolicy(*FIXED)
        self._run_btn.setIcon(QIconifyIcon("mdi:play", color=GREEN))
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setSizePolicy(*FIXED)
        self._cancel_btn.setIcon(QIcon(QIconifyIcon("mdi:stop", color=RED)))

        # positions selector
        self._positions_wdg = _ChoosePositionsWidget(self)

        # run options selector
        run_options_wdg = QWidget()
        self._run_options_lbl = QLabel("Run Options:")
        self._run_options_lbl.setSizePolicy(*FIXED)
        self._run_options_combo = QComboBox()
        items = [
            "Detection, Extraction and Analysis",
            "Detection and Extraction",
            "Extraction and Analysis (require detection)",
            "Detection Only",
            "Extraction Only (require detection)",
            "Analysis Only (require detection and extraction)",
        ]
        self._run_options_combo.addItems(items)
        self._run_options_combo.setToolTip(
            "Choose what to run:\n\n"
            "• Detection, Extraction and Analysis: Full pipeline\n"
            "• Detection and Extraction: Run detection and extraction only\n"
            "• Extraction and Analysis (require detection): Run extraction and\n"
            "  analysis using existing detection results (requires selecting a\n"
            "  Detection ID)\n"
            "• Detection Only: Run detection only to identify ROIs\n"
            "• Extraction Only: Run extraction using existing detection results\n"
            "  (requires selecting a Detection ID)\n"
            "• Analysis Only: Run analysis using existing extraction results\n"
            "  (requires selecting a Detection ID and Extraction ID)\n\n"
            "Smart Skipping:\n"
            "The system automatically detects which positions have already been \n"
            "processed with the exact same settings. If you request any stage \n"
            "for positions that have already been completed with identical \n"
            "settings, those positions will be automatically skipped to \n"
            "avoid redundant processing."
        )
        self._run_options_combo.currentTextChanged.connect(self._on_run_option_changed)

        # Detection settings selector (for Extraction/Analysis-only modes)
        self._detection_settings_combo = QComboBox()
        self._detection_settings_combo.setToolTip(
            "Select which detection results to use.\n\n"
            "Detection ID corresponds to the specific detection settings \n"
            "(method, parameters) used to identify ROIs. Required for \n"
            "extraction-only and analysis-only modes."
        )
        self._detection_settings_combo.setVisible(False)

        # Extraction settings selector (for Analysis-only mode)
        self._extraction_settings_combo = QComboBox()
        self._extraction_settings_combo.setToolTip(
            "Select which extraction results to use for analysis.\n\n"
            "Extraction ID corresponds to the specific extraction settings \n"
            "(neuropil correction, ΔF/F0 parameters) used. Required for \n"
            "analysis-only mode."
        )
        self._extraction_settings_combo.setVisible(False)

        run_options_layout = QHBoxLayout(run_options_wdg)
        run_options_layout.setContentsMargins(0, 0, 0, 0)
        run_options_layout.setSpacing(5)
        run_options_layout.addWidget(self._run_options_lbl)
        run_options_layout.addWidget(self._run_options_combo)
        run_options_layout.addWidget(self._detection_settings_combo)
        run_options_layout.addWidget(self._extraction_settings_combo)

        # main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)
        main_layout.addWidget(create_divider_line("Positions to Extract"))
        main_layout.addWidget(self._positions_wdg)
        main_layout.addWidget(create_divider_line("Run Options"))
        main_layout.addWidget(run_options_wdg)

        # run control layout
        run_control_layout = QHBoxLayout()
        run_control_layout.setContentsMargins(0, 0, 0, 0)
        run_control_layout.setSpacing(5)
        run_control_layout.addWidget(self._run_btn)
        run_control_layout.addWidget(self._cancel_btn)
        run_control_layout.addWidget(self._save_settings_btn)
        run_control_layout.addWidget(self._load_settings_btn)
        run_control_layout.addWidget(self._progress_bar)
        run_control_layout.addWidget(self._progress_pos_label)
        run_control_layout.addWidget(self._elapsed_time_label)
        main_layout.addLayout(run_control_layout)

        # Initially disable "Extraction Only" and "Analysis Only" options
        self._update_options_availability(has_detections=False, has_extractions=False)

    # PUBLIC METHODS --------------------------------------------------------------

    def enable(self, state: bool) -> None:
        """Enable or disable the widget but the cancel button."""
        self._positions_wdg.setEnabled(state)
        self._run_options_lbl.setEnabled(state)
        self._run_options_combo.setEnabled(state)
        self._detection_settings_combo.setEnabled(state)
        self._extraction_settings_combo.setEnabled(state)
        self._run_btn.setEnabled(state)
        self._save_settings_btn.setEnabled(state)
        self._load_settings_btn.setEnabled(state)

    def progress_bar_maximum(self) -> int:
        """Return the maximum value of the progress bar."""
        return cast("int", self._progress_bar.maximum())

    def set_progress_bar_range(self, minimum: int, maximum: int) -> None:
        """Set the range of the progress bar."""
        self._progress_bar.setRange(minimum, maximum)

    def set_progress_bar_text(self, text: str) -> None:
        """Update the progress bar label with custom text.

        Parameters
        ----------
        text : str
            Progress text to display
        """
        self._progress_pos_label.setText(text)

    def reset_progress_bar(self) -> None:
        """Reset the progress bar and elapsed time label."""
        self._progress_bar.reset()
        self._progress_bar.setValue(0)
        self._progress_pos_label.setText("")
        self._elapsed_time_label.setText("00:00:00")

    def reset_progress_value(self) -> None:
        """Reset only the progress bar value."""
        self._progress_bar.reset()
        self._progress_bar.setValue(0)

    def set_time_label(self, elapsed_time: str) -> None:
        """Update the elapsed time label."""
        self._elapsed_time_label.setText(elapsed_time)

    def update_progress_bar_plus_one(self) -> None:
        """Automatically update the progress bar value and label.

        The value is incremented by 1 each time this method is called.
        """
        value = self._progress_bar.value() + 1
        self._progress_bar.setValue(value)

    def reset(self) -> None:
        """Reset the widget to default values."""
        self.reset_progress_bar()
        self._positions_wdg.setValue("")
        self._run_options_combo.setCurrentIndex(0)
        self._detection_settings_combo.clear()

    def value(self) -> CaliRunSettings:
        """Get the current run settings.

        Returns
        -------
        CaliRunSettings
            Dataclass containing positions, run_detection, run_extraction,
            run_analysis, and settings IDs
        """
        option = self._run_options_combo.currentText()

        # Determine which settings IDs to use
        detection_settings_id = None
        extraction_settings_id = None

        # For "Only" modes that require existing settings
        extraction_and_analysis = "Extraction and Analysis (require detection)"
        if (
            "Extraction Only" in option
            or "Analysis Only" in option
            or option == extraction_and_analysis
        ):
            # Get detection settings ID (required for all these modes)
            detection_settings_id = self._detection_settings_combo.currentData()

            # Get extraction settings ID (only for Analysis Only)
            if "Analysis Only" in option:
                extraction_settings_id = self._extraction_settings_combo.currentData()

        # Determine which stages to run
        extraction_only = "Extraction Only (require detection)"
        analysis_only = "Analysis Only (require detection and extraction)"

        run_detection = "Detection" in option and option not in [
            extraction_only,
            analysis_only,
            extraction_and_analysis,
        ]
        run_extraction = "Extraction" in option and option != analysis_only
        run_analysis = "Analysis" in option

        return CaliRunSettings(
            positions=parse_lineedit_text(self._positions_wdg.value()),
            run_detection=run_detection,
            run_extraction=run_extraction,
            run_analysis=run_analysis,
            detection_settings_id=detection_settings_id,
            extraction_settings_id=extraction_settings_id,
        )

    def get_detection_settings_id(self) -> int | None:
        """Get the selected detection settings ID.

        Returns
        -------
        int | None
            Selected detection settings ID or None if not selected/visible
        """
        if self._detection_settings_combo.isVisible():
            return self._detection_settings_combo.currentData()  # type: ignore[no-any-return]
        return None

    def populate_detection_settings(self, settings_list: list[tuple[int, str]]) -> None:
        """Populate the detection settings combobox.

        Parameters
        ----------
        settings_list : list[tuple[int, str]]
            List of (id, method) tuples for available detection settings
        """
        self._detection_settings_combo.clear()
        self._detection_settings_combo.addItem("Select Detection ID...", None)
        for settings_id, method in settings_list:
            self._detection_settings_combo.addItem(
                f"Detection ID {settings_id} ({method})", settings_id
            )

        # Enable/disable options based on detection availability
        self._update_options_availability(
            has_detections=len(settings_list) > 0,
            has_extractions=False,  # Will be updated separately
        )

    def populate_extraction_settings(self, settings_list: list[int]) -> None:
        """Populate the extraction settings combobox.

        Parameters
        ----------
        settings_list : list[int]
            List of extraction settings IDs
        """
        self._extraction_settings_combo.clear()
        self._extraction_settings_combo.addItem("Select Extraction ID...", None)
        for settings_id in settings_list:
            self._extraction_settings_combo.addItem(
                f"Extraction ID {settings_id}", settings_id
            )

        # Update options availability
        has_detections = self._detection_settings_combo.count() > 1
        self._update_options_availability(
            has_detections=has_detections, has_extractions=len(settings_list) > 0
        )

    def _update_options_availability(
        self, has_detections: bool, has_extractions: bool
    ) -> None:
        """Enable or disable run options based on available settings.

        Parameters
        ----------
        has_detections : bool
            Whether any detection settings exist in the database
        has_extractions : bool
            Whether any extraction settings exist in the database
        """
        model = cast("QStandardItemModel", self._run_options_combo.model())

        # Index mapping:
        # 0: "Detection, Extraction and Analysis"
        # 1: "Detection and Extraction"
        # 2: "Extraction and Analysis (require detection)"
        # 3: "Detection Only"
        # 4: "Extraction Only (require detection)"
        # 5: "Analysis Only (require detection and extraction)"

        current_index = self._run_options_combo.currentIndex()

        # "Extraction Only" and "Extraction and Analysis" require detections
        extraction_only_item = model.item(4)
        if extraction_only_item:
            if has_detections:
                extraction_only_item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                )
            else:
                extraction_only_item.setFlags(Qt.ItemFlag.NoItemFlags)
                if current_index == 4:
                    self._run_options_combo.setCurrentIndex(0)

        extraction_and_analysis_item = model.item(2)
        if extraction_and_analysis_item:
            if has_detections:
                extraction_and_analysis_item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                )
            else:
                extraction_and_analysis_item.setFlags(Qt.ItemFlag.NoItemFlags)
                if current_index == 2:
                    self._run_options_combo.setCurrentIndex(0)

        # "Analysis Only" requires both detections and extractions
        analysis_only_item = model.item(5)
        if analysis_only_item:
            if has_detections and has_extractions:
                analysis_only_item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                )
            else:
                analysis_only_item.setFlags(Qt.ItemFlag.NoItemFlags)
                if current_index == 5:
                    self._run_options_combo.setCurrentIndex(0)

    def _on_run_option_changed(self, text: str) -> None:
        """Handle run option change to show/hide settings selectors.

        Parameters
        ----------
        text : str
            The selected run option text
        """
        # Show detection settings for "Extraction Only", "Analysis Only",
        # and "Extraction and Analysis"
        is_extraction_only = text == "Extraction Only (require detection)"
        is_analysis_only = text == "Analysis Only (require detection and extraction)"
        is_extraction_and_analysis = (
            text == "Extraction and Analysis (require detection)"
        )

        show_detection = (
            is_extraction_only or is_analysis_only or is_extraction_and_analysis
        )
        self._detection_settings_combo.setVisible(show_detection)

        # Show extraction settings only for "Analysis Only"
        self._extraction_settings_combo.setVisible(is_analysis_only)

        # Auto-select if only one option available
        if show_detection and self._detection_settings_combo.count() == 2:
            self._detection_settings_combo.setCurrentIndex(1)
        if is_analysis_only and self._extraction_settings_combo.count() == 2:
            self._extraction_settings_combo.setCurrentIndex(1)
