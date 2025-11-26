from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from fonticon_mdi6 import MDI6
from qtpy.QtCore import QElapsedTimer, QObject, Qt, QTimer, Signal
from qtpy.QtGui import QIcon, QStandardItemModel
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt.fonticon import icon

if TYPE_CHECKING:
    from pathlib import Path

FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed


def show_error_dialog(parent: QWidget, message: str) -> None:
    """Show an error dialog with the given message."""
    dialog = QMessageBox(parent)
    dialog.setWindowTitle("Error")
    dialog.setText(message)
    dialog.setIcon(QMessageBox.Icon.Critical)
    dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
    dialog.exec()


class _BrowseWidget(QWidget):
    pathSet = Signal(str)
    filePathSet = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        label: str = "",
        path: str | None = None,
        tooltip: str = "",
        *,
        is_dir: bool = True,
    ) -> None:
        super().__init__(parent)

        self._is_dir = is_dir

        self._current_path = path or ""

        self._label_text = label

        self._label = QLabel(f"{self._label_text}:")
        self._label.setSizePolicy(*FIXED)
        self._label.setToolTip(tooltip)

        self._path = QLineEdit()
        self._path.setText(self._current_path)
        self._browse_btn = QPushButton("Browse")
        self._browse_btn.clicked.connect(self._on_browse)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._label)
        layout.addWidget(self._path)
        layout.addWidget(self._browse_btn)

    def value(self) -> str:
        import os

        path_text = self._path.text()
        return str(os.path.normpath(path_text)) if path_text else ""

    def setValue(self, path: str | Path) -> None:
        self._path.setText(str(path))

    def clear(self) -> None:
        self._path.clear()
        self._current_path = ""

    def _on_browse(self) -> None:
        if self._is_dir:
            if path := QFileDialog.getExistingDirectory(
                self, f"Select the {self._label_text}.", self._current_path
            ):
                self._path.setText(path)
                self.pathSet.emit(path)
        else:
            path, _ = QFileDialog.getOpenFileName(
                self,
                f"Select the {self._label_text}.",
            )
            if path:
                self._path.setText(path)
                self.filePathSet.emit(path)


class _ElapsedTimer(QObject):
    """A timer to keep track of the elapsed time."""

    elapsed_time_updated = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._elapsed_timer = QElapsedTimer()
        self._time_timer = QTimer()
        self._time_timer.timeout.connect(self._update_elapsed_time)

    def start(self) -> None:
        self._elapsed_timer.start()
        self._time_timer.start(1000)

    def stop(self) -> None:
        self._elapsed_timer.invalidate()
        self._time_timer.stop()

    def _update_elapsed_time(self) -> None:
        elapsed_ms = self._elapsed_timer.elapsed()
        elapsed_time_str = self._format_elapsed_time(elapsed_ms)
        self.elapsed_time_updated.emit(elapsed_time_str)

    @staticmethod
    def _format_elapsed_time(milliseconds: int) -> str:
        seconds = milliseconds // 1000
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"


class _ProgressBarWidget(QDialog):
    """A progress bar that oscillates between 0 and a given range."""

    def __init__(self, parent: QWidget | None = None, *, text: str = "") -> None:
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.Sheet)

        self._label = QLabel(text)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimumWidth(200)
        self._progress_bar.setValue(0)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self._label)
        layout.addWidget(self._progress_bar)

    def show_progress_bar(self, value: bool) -> None:
        """Show/hide the progress bar while maintaining dialog and text visibility."""
        self._progress_bar.hide() if not value else self._progress_bar.show()

    def setText(self, text: str) -> None:
        """Set the text of the progress bar."""
        self._label.setText(text)

    def setValue(self, value: int) -> None:
        """Set the progress bar value."""
        self._progress_bar.setValue(value)

    def setRange(self, min: int, max: int) -> None:
        """Set the progress bar range."""
        self._progress_bar.setRange(min, max)

    def showPercentage(self, visible: bool) -> None:
        """Show or hide the percentage display on the progress bar."""
        self._progress_bar.setTextVisible(visible)


class _WaitingProgressBarWidget(QDialog):
    """A progress bar that oscillates between 0 and a given range."""

    def __init__(
        self, parent: QWidget | None = None, *, range: int = 50, text: str = ""
    ) -> None:
        super().__init__(parent)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)

        self._range = range

        self._text = text
        label = QLabel(self._text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._progress_bar = QProgressBar()
        self._progress_bar.setMinimumWidth(200)
        self._progress_bar.setRange(0, self._range)
        self._progress_bar.setValue(0)

        self._direction = 1

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_progress)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(label)
        layout.addWidget(self._progress_bar)

    def start(self) -> None:
        """Start the progress bar."""
        self.show()
        self._timer.start(50)

    def stop(self) -> None:
        """Stop the progress bar."""
        self.hide()
        self._timer.stop()

    def _update_progress(self) -> None:
        """Update the progress bar value.

        The progress bar value will oscillate between 0 and the range and back.
        """
        value = self._progress_bar.value()
        value += self._direction
        if value >= self._range:
            value = self._range
            self._direction = -1
        elif value <= 0:
            value = 0
            self._direction = 1
        self._progress_bar.setValue(value)


def parse_lineedit_text(input_str: str) -> list[int]:
    """Parse the input string and return a list of numbers."""
    parts = input_str.split(",")
    numbers: list[int] = []
    for part in parts:
        part = part.strip()  # remove any leading/trailing whitespace
        if "-" in part:
            with contextlib.suppress(ValueError):
                start, end = map(int, part.split("-"))
                numbers.extend(range(start, end + 1))
        else:
            with contextlib.suppress(ValueError):
                numbers.append(int(part))
    return numbers


def create_divider_line(text: str | None = None) -> QWidget:
    """Create a horizontal divider line, optionally with text.

    Parameters
    ----------
    text : str | None
        Optional text to display in front of the divider line

    Returns
    -------
    QWidget
        Widget containing the divider line and optional text
    """
    if text is None:
        return _create_line()
    # Create container widget for text + line
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(10)

    # Add text label
    label = QLabel(text)
    # make bold and increase font size
    label.setStyleSheet("font-weight: bold; font-size: 14px; color: rgb(0, 183, 0);")
    layout.addWidget(label)

    line = _create_line()
    layout.addWidget(line, 1)  # Give line stretch factor of 1

    return container


def _create_line() -> QFrame:
    """Create a horizontal line frame for use as a divider."""
    result = QFrame()
    # set color
    # result.setStyleSheet("color: rgb(0, 183, 0);")
    result.setFrameShape(QFrame.Shape.HLine)
    result.setFrameShadow(QFrame.Shadow.Plain)
    return result


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
    run_analysis: bool
    detection_settings_id: int | None


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

        self._run_btn = QPushButton("Run")
        self._run_btn.setSizePolicy(*FIXED)
        self._run_btn.setIcon(icon(MDI6.play, color=GREEN))
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setSizePolicy(*FIXED)
        self._cancel_btn.setIcon(QIcon(icon(MDI6.stop, color=RED)))

        # positions selector
        self._positions_wdg = _ChoosePositionsWidget(self)

        # run options selector
        run_options_wdg = QWidget()
        self._run_options_lbl = QLabel("Run Options:")
        self._run_options_lbl.setSizePolicy(*FIXED)
        self._run_options_combo = QComboBox()
        items = [
            "Detection and Analysis",
            "Detection Only",
            "Analysis Only (require at least one detection run in database)",
        ]
        self._run_options_combo.addItems(items)
        self._run_options_combo.setToolTip(
            "Choose what to run:\n\n"
            "• Detection: Run detection only to identify ROIs\n"
            "• Detection and Analysis: Run both detection and analysis\n"
            "• Analysis: Run analysis only using existing detection results\n"
            "  (requires selecting a Detection ID)\n\n"
            "Smart Detection Skipping:\n"
            "The system automatically detects which positions have already been \n"
            "processed with the exact same settings. If you request detection, \n"
            "analysis, or both for positions that have already been completed with \n"
            "identical settings, those positions will be automatically skipped to \n"
            "avoid redundant processing."
        )
        self._run_options_combo.currentTextChanged.connect(self._on_run_option_changed)

        # Detection settings selector (for Analysis-only mode)
        self._detection_settings_lbl = QLabel("Detection ID:")
        self._detection_settings_lbl.setSizePolicy(*FIXED)
        self._detection_settings_combo = QComboBox()
        self._detection_settings_combo.setToolTip(
            "Select which detection results to use for analysis.\n\n"
            "Detection ID corresponds to the specific detection settings \n"
            "(method, parameters) used to identify ROIs. You must select \n"
            "an existing detection to run analysis-only mode."
        )
        self._detection_settings_lbl.setVisible(False)
        self._detection_settings_combo.setVisible(False)

        run_options_layout = QHBoxLayout(run_options_wdg)
        run_options_layout.setContentsMargins(0, 0, 0, 0)
        run_options_layout.setSpacing(5)
        run_options_layout.addWidget(self._run_options_lbl)
        run_options_layout.addWidget(self._run_options_combo)
        run_options_layout.addWidget(self._detection_settings_lbl)
        run_options_layout.addWidget(self._detection_settings_combo)

        # main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)
        main_layout.addWidget(create_divider_line("Positions to Analyze"))
        main_layout.addWidget(self._positions_wdg)
        main_layout.addWidget(create_divider_line("Run Options"))
        main_layout.addWidget(run_options_wdg)

        # run control layout
        run_control_layout = QHBoxLayout()
        run_control_layout.setContentsMargins(0, 0, 0, 0)
        run_control_layout.setSpacing(5)
        run_control_layout.addWidget(self._run_btn)
        run_control_layout.addWidget(self._cancel_btn)
        run_control_layout.addWidget(self._progress_bar)
        run_control_layout.addWidget(self._progress_pos_label)
        run_control_layout.addWidget(self._elapsed_time_label)
        main_layout.addLayout(run_control_layout)

        # Initially disable "Analysis Only" option (no detections at init)
        self._update_analysis_only_availability(has_detections=False)

    # PUBLIC METHODS --------------------------------------------------------------

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
            Dataclass containing positions, run_detection, run_analysis,
            and detection_settings_id
        """
        option = self._run_options_combo.currentText()
        detection_settings_id = (
            self._detection_settings_combo.currentData()
            if option
            == "Analysis Only (require at least one detection run in database)"
            else None
        )
        return CaliRunSettings(
            positions=parse_lineedit_text(self._positions_wdg.value()),
            run_detection="Detection" in option,
            run_analysis="Analysis" in option,
            detection_settings_id=detection_settings_id,
        )

    def get_detection_settings_id(self) -> int | None:
        """Get the selected detection settings ID.

        Returns
        -------
        int | None
            Selected detection settings ID or None if not selected/visible
        """
        if self._detection_settings_combo.isVisible():
            return self._detection_settings_combo.currentData()
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

        # Enable/disable the "Analysis Only" option based on detection availability
        self._update_analysis_only_availability(has_detections=len(settings_list) > 0)

    def _update_analysis_only_availability(self, has_detections: bool) -> None:
        """Enable or disable the Analysis Only option based on detection availability.

        Parameters
        ----------
        has_detections : bool
            Whether any detection settings exist in the database
        """
        from qtpy.QtCore import Qt

        # Find the "Analysis Only" option (index 2)
        analysis_only_index = 2
        model = cast("QStandardItemModel", self._run_options_combo.model())
        item = model.item(analysis_only_index)

        if item is None:
            return

        if has_detections:
            # Enable the item
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        else:
            # Disable the item
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            # If currently selected, switch to first option
            if self._run_options_combo.currentIndex() == analysis_only_index:
                self._run_options_combo.setCurrentIndex(0)

    def _on_run_option_changed(self, text: str) -> None:
        """Handle run option change to show/hide detection settings selector.

        Parameters
        ----------
        text : str
            The selected run option text
        """
        # Show detection settings combo only for "Analysis" option (index 2)
        is_analysis_only = (
            text == "Analysis Only (require at least one detection run in database)"
        )
        self._detection_settings_lbl.setVisible(is_analysis_only)
        self._detection_settings_combo.setVisible(is_analysis_only)
        # if there is only one detection id, select it by default
        if is_analysis_only and self._detection_settings_combo.count() == 2:
            self._detection_settings_combo.setCurrentIndex(1)
