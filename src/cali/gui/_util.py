from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, TypeVar

from qtpy.QtCore import QElapsedTimer, QObject, Qt, QTimer, Signal
from qtpy.QtWidgets import (
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

from cali._constants import (
    TS,
    ZR,
)
from cali.readers._ome_zarr_reader import OMEZarrReader
from cali.readers._tensorstore_zarr_reader import TensorstoreZarrReader

if TYPE_CHECKING:
    from pathlib import Path

# Define a type variable for the BaseClass
T = TypeVar("T", bound="BaseClass")


@dataclass
class BaseClass:
    """Base class for all classes in the package."""

    def replace(self: T, **kwargs: Any) -> T:
        """Replace the values of the dataclass with the given keyword arguments."""
        return replace(self, **kwargs)


# fmt: off
@dataclass
class ROIData(BaseClass):
    """Data container for ROI (Region of Interest) analysis results.

    This dataclass stores comprehensive analysis data for a single ROI including
    raw fluorescence traces, neuropil correction, calcium dynamics (dff, deconvolved),
    peak detection, inferred spikes, and experimental metadata.

    Parameters
    ----------
    well_fov_position : str
        Position identifier (e.g., "B5_0000_p0" for well B5, fov0, position 0)
    raw_trace : list[float] | None
        Original raw fluorescence trace before any neuropil correction
    corrected_trace : list[float] | None
        Raw fluorescence trace after neuropil correction (if enabled),
        otherwise same as raw_trace. This is used for all
        downstream analysis.
    neuropil_trace : list[float] | None
        Fluorescence trace from the neuropil (donut-shaped region around ROI)
    neuropil_correction_factor : float | None
        Correction factor used for neuropil subtraction
    dff : list[float] | None
        ΔF/F (delta F over F) - normalized fluorescence change
    dec_dff : list[float] | None
        Deconvolved ΔF/F trace (using OASIS algorithm) for calcium event detection
    peaks_dec_dff : list[float] | None
        Indices of detected peaks in the deconvolved trace
    peaks_amplitudes_dec_dff : list[float] | None
        Amplitude values of detected peaks in deconvolved trace
    peaks_prominence_dec_dff : float | None
        Prominence threshold used for peak detection
    peaks_height_dec_dff : float | None
        Height threshold used for peak detection
    inferred_spikes : list[float] | None
        Inferred spike probabilities from deconvolution
    inferred_spikes_threshold : float | None
        Threshold for spike detection
    dec_dff_frequency : float | None
        Frequency of calcium events in Hz
    condition_1 : str | None
        First experimental condition (e.g., genotype)
    condition_2 : str | None
        Second experimental condition (e.g., treatment)
    cell_size : float | None
        ROI area in µm² or pixels
    cell_size_units : str | None
        Units for cell_size ("µm" or "pixel")
    elapsed_time_list_ms : list[float] | None
        Timestamp for each frame in milliseconds
    total_recording_time_sec : float | None
        Total recording duration in seconds
    active : bool | None
        Whether the ROI shows calcium activity (has detected peaks)
    iei : list[float] | None
        Inter-event intervals between calcium peaks (in seconds)
    evoked_experiment : bool
        Whether this is an optogenetic stimulation experiment
    stimulated : bool
        Whether this ROI overlaps with the stimulated area
    stimulations_frames_and_powers : dict[str, int] | None
        Frame numbers and LED powers for stimulation events
    led_pulse_duration : str | None
        Duration of LED pulse in stimulation experiments
    led_power_equation : str | None
        Equation to calculate LED power density (mW/cm²)
    calcium_sync_jitter_window : int | None
        Jitter window (frames) for calcium peak synchrony analysis
    spikes_sync_cross_corr_lag : int | None
        Maximum lag (frames) for spike cross-correlation synchrony
    calcium_network_threshold : float | None
        Percentile threshold (0-100) for network connectivity
    spikes_burst_threshold : float | None
        Threshold (%) for burst detection in spike trains
    spikes_burst_min_duration : int | None
        Minimum burst duration in seconds
    spikes_burst_gaussian_sigma : float | None
        Sigma for Gaussian smoothing in burst detection (seconds)
    mask_coord_and_shape : tuple[tuple[list[int], list[int]], tuple[int, int]] | None
        ROI mask stored as ((y_coords, x_coords), (height, width))
    neuropil_mask_coord_and_shape : tuple | None
        Neuropil mask: ((y_coords, x_coords), (height, width))
    """

    well_fov_position: str = ""
    raw_trace: list[float] | None = None
    corrected_trace: list[float] | None = None
    neuropil_trace: list[float] | None = None
    neuropil_correction_factor: float | None = None
    dff: list[float] | None = None
    dec_dff: list[float] | None = None  # deconvolved dff with oasis package
    peaks_dec_dff: list[float] | None = None
    peaks_amplitudes_dec_dff: list[float] | None = None
    peaks_prominence_dec_dff: float | None = None
    peaks_height_dec_dff: float | None = None
    inferred_spikes: list[float] | None = None
    inferred_spikes_threshold: float | None = None
    dec_dff_frequency: float | None = None  # Hz
    condition_1: str | None = None
    condition_2: str | None = None
    cell_size: float | None = None
    cell_size_units: str | None = None
    elapsed_time_list_ms: list[float] | None = None  # in ms
    total_recording_time_sec: float | None = None  # in seconds
    active: bool | None = None
    iei: list[float] | None = None  # interevent interval
    evoked_experiment: bool = False
    stimulated: bool = False
    stimulations_frames_and_powers: dict[str, int] | None = None
    led_pulse_duration: str | None = None
    led_power_equation: str | None = None  # equation for LED power
    calcium_sync_jitter_window: int | None = None  # in frames
    spikes_sync_cross_corr_lag: int | None = None  # in frames
    calcium_network_threshold: float | None = None  # percentile (0-100)
    spikes_burst_threshold: float | None = None  # in percent
    spikes_burst_min_duration: int | None = None  # in seconds
    spikes_burst_gaussian_sigma: float | None = None  # in seconds
    # store ROI mask as coordinates (y_coords, x_coords) and shape (height, width)
    mask_coord_and_shape: tuple[tuple[list[int], list[int]], tuple[int, int]] | None = None  # noqa: E501
    # store neuropil mask as coordinates (y_coords, x_coords) and shape (height, width)
    neuropil_mask_coord_and_shape: tuple[tuple[list[int], list[int]], tuple[int, int]] | None = None  # noqa: E501
# fmt: on


def show_error_dialog(parent: QWidget, message: str) -> None:
    """Show an error dialog with the given message."""
    dialog = QMessageBox(parent)
    dialog.setWindowTitle("Error")
    dialog.setText(message)
    dialog.setIcon(QMessageBox.Icon.Critical)
    dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
    dialog.exec()


def load_data(data_path: str | Path) -> TensorstoreZarrReader | OMEZarrReader | None:
    """Load data from the given path using the appropriate reader."""
    data_path = str(data_path)
    # select which reader to use for the datastore
    if data_path.endswith(TS):
        # read tensorstore
        return TensorstoreZarrReader(data_path)
    elif data_path.endswith(ZR):
        # read ome zarr
        return OMEZarrReader(data_path)
    else:
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
        *,
        is_dir: bool = True,
    ) -> None:
        super().__init__(parent)

        self._is_dir = is_dir

        self._current_path = path or ""

        self._label_text = label

        self._label = QLabel(f"{self._label_text}:")
        self._label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
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


















# SYNCHRONY FUNCTIONS -----------------------------------------------------------------







