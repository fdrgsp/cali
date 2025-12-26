from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Literal, cast

from qtpy.QtCore import QElapsedTimer, QObject, Qt, QTimer, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from pathlib import Path

    from cali._constants import CorrelationDataType, TraceDataType

FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed

DIALOG_TYPES = Literal["error", "warning", "info"]
DIALOG_SYMBOLS = {
    "error": "❌",
    "warning": "⚠️",
    "info": "\u24d8",
}


def show_error_dialog(
    parent: QWidget, message: str, type: DIALOG_TYPES = "error", choice: bool = False
) -> QDialog | None:
    """Show an error dialog with the given message.

    When choice=True, returns the dialog for the caller to handle exec().
    When choice=False, shows the dialog and returns None.
    """
    dialog = QDialog(parent)
    symbol = DIALOG_SYMBOLS.get(type, "")
    dialog.setWindowTitle(f"{symbol} {type.capitalize()}")
    dialog.setModal(True)

    layout = QVBoxLayout(dialog)

    # QTextEdit for scrollable text
    text_edit = QTextEdit()
    text_edit.setPlainText(message)
    text_edit.setReadOnly(True)
    text_edit.setMinimumSize(300, 200)
    layout.addWidget(text_edit)

    # if choice is True, show Yes/No buttons and return dialog
    if choice:
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No
        )
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        return dialog

    # otherwise, show only OK button and exec immediately
    button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
    button_box.accepted.connect(dialog.accept)
    layout.addWidget(button_box)
    dialog.exec()
    return None


class _BrowseWidget(QWidget):
    pathSet = Signal(str)
    filePathSet = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        label: str = "",
        path: str | None = None,
        tooltip: str = "",
        file_filter: str = "",
        *,
        is_dir: bool = True,
    ) -> None:
        super().__init__(parent)

        self._is_dir = is_dir

        self._current_path = path or ""

        self._label_text = label

        self._file_filter = file_filter

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
                "",
                self._file_filter,
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
        self.showPercentage(True)

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


class _ExportGroup(QGroupBox):
    """Widget with export options."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setCheckable(True)
        self.setChecked(True)
        self.setStyleSheet("QGroupBox::title { font-size: 14px; }")
        self.setTitle("Enable/Disable Export Options (as .csv)")

        self._checkboxes: dict[str, tuple[QCheckBox, int, int]] = {}

        self._layout = QGridLayout(self)
        self._layout.setContentsMargins(20, 5, 5, 5)
        self._layout.setSpacing(10)

    def add_option(
        self,
        text: str,
        row: int,
        col: int,
        *,
        checked: bool = True,
    ) -> None:
        """Add an option checkbox.

        Parameters
        ----------
        text : str
            The label text for the checkbox option.
        row : int
            Grid row position.
        col : int
            Grid column position.
        checked : bool, optional
            Initial checked state, by default False.
        """
        if text in self._checkboxes:
            return

        checkbox = QCheckBox(text)
        checkbox.setChecked(checked)
        self._checkboxes[text] = (checkbox, row, col)
        self._layout.addWidget(checkbox, row, col)

    def add_stretch(self, direction: Literal["vertical", "horizontal"]) -> None:
        """Add stretch to prevent widgets from spreading when resizing.

        Parameters
        ----------
        direction : Literal["vertical", "horizontal"]
            Direction to add stretch. "vertical" adds stretch to the last row,
            "horizontal" adds stretch to the last column.
        """
        if direction == "vertical":
            self._layout.setRowStretch(self._layout.rowCount(), 1)
        elif direction == "horizontal":
            self._layout.setColumnStretch(self._layout.columnCount(), 1)

    def value(self) -> dict[str, tuple[bool, int, int]]:
        """Return the current widget state.

        Returns
        -------
        dict[str, tuple[bool, int, int]]
            Dictionary mapping option text to (checked_state, row, col).
        """
        return {
            text: (checkbox.isChecked(), row, col)
            for text, (checkbox, row, col) in self._checkboxes.items()
        }

    def setValue(self, values: dict[str, tuple[bool, int, int]]) -> None:
        """Set the widget state.

        Parameters
        ----------
        values : dict[str, tuple[bool, int, int]]
            Dictionary mapping option text to (checked_state, row, col).
        """
        # Clear existing layout
        for _, (checkbox, _, _) in list(self._checkboxes.items()):
            self._layout.removeWidget(checkbox)
            checkbox.deleteLater()
        self._checkboxes.clear()

        # Add options with stored positions
        for text, (checked, row, col) in values.items():
            self.add_option(text, row, col, checked=checked)

    def get_export_options(self) -> dict[TraceDataType | CorrelationDataType, bool]:
        """Return a dictionary with the export options selected by the user.

        Returns.
        -------
        dict[TraceDataType | CorrelationDataType, bool]
            Dictionary mapping option to a boolean indicating whether the user
            selected to export that data type.
        """
        export_data = self.value()
        return cast(
            "dict[TraceDataType | CorrelationDataType, bool]",
            {
                correlation_type: checked
                for correlation_type, (checked, _, _) in export_data.items()
                if checked
            },
        )
