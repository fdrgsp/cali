from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QDoubleSpinBox,
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
from superqt import QIconifyIcon

from cali._constants import (
    DEFAULT_DFF_WINDOW,
    DEFAULT_FRAME_RATE,
    DEFAULT_NEUROPIL_CORRECTION_FACTOR,
    DEFAULT_NEUROPIL_INNER_RADIUS,
    DEFAULT_NEUROPIL_MIN_PIXELS,
)
from cali.sqlmodel import ExtractionSettings

from ._util import (
    create_divider_line,
)

FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed


@dataclass(frozen=True)
class MetadataData:
    """Data structure to hold metadata settings."""

    pixel_size: float | None = None  # micrometers (µm), None if 0
    frame_rate: float = DEFAULT_FRAME_RATE  # frames per second


@dataclass(frozen=True)
class ExtractionSettingsData:
    """Data structure to hold the extraction settings."""

    trace_extraction_data: TraceExtractionData | None = None
    metadata_data: MetadataData | None = None


@dataclass(frozen=True)
class NeuropilData:
    """Data structure to hold the neuropil correction settings."""

    neuropil_inner_radius: int = DEFAULT_NEUROPIL_INNER_RADIUS
    neuropil_min_pixels: int = DEFAULT_NEUROPIL_MIN_PIXELS
    neuropil_correction_factor: float = DEFAULT_NEUROPIL_CORRECTION_FACTOR


@dataclass(frozen=True)
class TraceExtractionData:
    """Data structure to hold the trace extraction settings."""

    dff_window_size: float = DEFAULT_DFF_WINDOW  # milliseconds
    decay_constant: float = 0.0  # seconds
    neuropil_inner_radius: int = DEFAULT_NEUROPIL_INNER_RADIUS
    neuropil_min_pixels: int = DEFAULT_NEUROPIL_MIN_PIXELS
    neuropil_correction_factor: float = DEFAULT_NEUROPIL_CORRECTION_FACTOR


class _ExtractionGUI(QWidget):
    progress_bar_updated = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # MAIN WIDGET -----------------------------------------------------------------
        group_wdg = QGroupBox(self)
        group_layout = QVBoxLayout(group_wdg)
        group_layout.setContentsMargins(10, 10, 10, 10)
        group_layout.setSpacing(5)

        # THREADS WIDGET -------------------------------------------------------------
        cpu_to_use = max((os.cpu_count() or 1) - 2, 1)
        threads_wdg = QWidget()
        threads_wdg.setToolTip(
            "Specify number of threads to use in the Thread Pool for the analysis.\n\n"
            "By default, the value is set to the number of CPUs - 2 "
            f"(in your system: {cpu_to_use}).\n\n"
            "Using the number of CPUs as reference because:\n"
            "• This analysis is CPU-intensive (math calculations, image processing)\n"
            "• More threads beyond CPU count creates context switching overhead\n"
            "• Each thread processes memory-intensive data\n"
            "• Optimal performance occurs when threads match available CPU cores.\n"
            "By default using CPU count - 2 to reserve some of the CPUs for the "
            "operating system and GUI responsiveness.\n"
            "If your system becomes unresponsive, consider reducing this number."
        )
        self._threads_lbl = QLabel("Number of Threads:", threads_wdg)
        self._threads_lbl.setSizePolicy(*FIXED)
        self._threads = QSpinBox(threads_wdg)
        self._threads.setRange(1, 100)
        self._threads.setValue(cpu_to_use)
        threads_layout = QHBoxLayout(threads_wdg)
        threads_layout.setContentsMargins(0, 0, 0, 0)
        threads_layout.setSpacing(5)
        threads_layout.addWidget(self._threads_lbl)
        threads_layout.addWidget(self._threads)

        # EXTRACTION WIDGETS ---------------------------------------------------------
        self._metadata_wdg = _MetadataWidget(self)
        self._neuropil_wdg = _NeuropilCorrectionWidget(self)
        self._trace_extraction_wdg = _TraceExtractionWidget(self)

        # SCROLL AREA WIDGET ---------------------------------------------------------
        analysis_scroll_area = QScrollArea()
        analysis_scroll_area.setWidgetResizable(True)
        analysis_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        analysis_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        # add extraction widgets to scroll area
        group_layout.addWidget(create_divider_line("Neuropil Settings"))
        group_layout.addWidget(self._neuropil_wdg)
        group_layout.addWidget(create_divider_line("ΔF/F0 and Deconvolution"))
        group_layout.addWidget(self._trace_extraction_wdg)
        group_layout.addWidget(create_divider_line("Metadata"))
        group_layout.addWidget(self._metadata_wdg)
        group_layout.addWidget(create_divider_line("Threads"))
        group_layout.addWidget(threads_wdg)
        group_layout.addStretch(1)
        analysis_scroll_area.setWidget(group_wdg)

        # MAIN LAYOUT -----------------------------------------------------------------
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(15)
        main_layout.addWidget(analysis_scroll_area)

        # STYLING ---------------------------------------------------------------------
        fix_width = self._threads_lbl.sizeHint().width()
        self._metadata_wdg.set_labels_width(fix_width)
        self._neuropil_wdg.set_labels_width(fix_width)
        self._trace_extraction_wdg.set_labels_width(fix_width)
        self._threads_lbl.setFixedWidth(fix_width)

    # PUBLIC METHODS ------------------------------------------------------------------

    @property
    def from_metadata(self) -> None:
        """Signal emitted when the 'Load From Metadata' button is clicked."""
        return self._metadata_wdg.from_metadata

    def value(self) -> ExtractionSettingsData:
        """Get the current values of the widget."""
        return ExtractionSettingsData(
            trace_extraction_data=self._trace_extraction_wdg.value(
                self._neuropil_wdg.value()
            ),
            metadata_data=self._metadata_wdg.value(),
        )

    def setValue(self, value: ExtractionSettingsData) -> None:
        """Set the values of the widget."""
        if value.trace_extraction_data is not None:
            self._trace_extraction_wdg.setValue(value.trace_extraction_data)
            # Also set the neuropil widget from trace extraction data
            neuropil_data = NeuropilData(
                value.trace_extraction_data.neuropil_inner_radius,
                value.trace_extraction_data.neuropil_min_pixels,
                value.trace_extraction_data.neuropil_correction_factor,
            )
            self._neuropil_wdg.setValue(neuropil_data)
        if value.metadata_data is not None:
            self._metadata_wdg.setValue(value.metadata_data)

    def reset(self) -> None:
        """Reset the widget to default values."""
        self._metadata_wdg.reset()
        self._neuropil_wdg.reset()
        self._trace_extraction_wdg.reset()

    def to_model_settings(self) -> ExtractionSettings:
        """Convert current GUI settings to ExtractionSettings model.

        Returns
        -------
        ExtractionSettings
            The ExtractionSettings model populated with current GUI values.
        """
        settings = self.value()

        # Extract nested data with defaults
        trace_data = settings.trace_extraction_data
        metadata_data = settings.metadata_data

        settings = ExtractionSettings(
            created_at=datetime.now(),
            threads=self._threads.value(),
            neuropil_inner_radius=(
                trace_data.neuropil_inner_radius if trace_data else 0
            ),
            neuropil_min_pixels=trace_data.neuropil_min_pixels if trace_data else 0,
            neuropil_correction_factor=(
                trace_data.neuropil_correction_factor if trace_data else 0.0
            ),
            decay_constant=trace_data.decay_constant if trace_data else 0.0,
            dff_window=(
                trace_data.dff_window_size if trace_data else DEFAULT_DFF_WINDOW
            ),
            frame_rate=(
                metadata_data.frame_rate if metadata_data else DEFAULT_FRAME_RATE
            ),
            pixel_size=metadata_data.pixel_size if metadata_data else None,
        )

        return settings


class _NeuropilCorrectionWidget(QWidget):
    """Widget to select the neuropil correction settings."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setToolTip(
            "Neuropil Correction - Background Subtraction from Surrounding Area.\n\n"
            "Removes contaminating fluorescence from out-of-focus neuropil "
            "(the area surrounding cells) to improve signal purity.\n\n"
            "Disabled by default (Inner Radius=0 or Min Pixels=0).\n"
            "Disable neuropil correction by setting EITHER Inner Radius OR "
            "Min Pixels to 0.\n\n"
            "Algorithm Overview (Suite2p Implementation):\n"
            "Creates a 'donut-shaped' neuropil mask around each cell by:\n"
            "1. Defining an inner 'forbidden zone' extending outward from cell edge\n"
            "2. Iteratively expanding the ROI pixel-by-pixel (5 pixels at a time)\n"
            "3. Excluding pixels belonging to other cells\n"
            "4. Continuing expansion until minimum pixel count is reached\n"
            "5. Corrected Fluorescence = Cell Fluorescence - "
            "(Factor x Neuropil Fluorescence)\n\n"
            "Parameters:\n"
            "• Inner Radius: Distance (in pixels) extending OUTWARD from the cell "
            " boundary to define the 'forbidden zone'.\n  This region is too close to "
            " the cell and excluded from neuropil due to potential contamination "
            " from optical blur/diffraction.\n  The neuropil region starts BEYOND this "
            " forbidden zone.\n  Larger values = more conservative (neuropil further "
            " from cell). Set to 0 to disable neuropil correction. Default: 0 pixels "
            " (suite2p default 2 pixels).\n"
            "• Min Pixels: Minimum number of pixels required in the neuropil mask "
            " for a reliable background measurement.\n  The algorithm automatically "
            " expands outward (5 pixels per iteration, up to 100 iterations) "
            " until this threshold is reached.\n  Set to 0 to disable neuropil "
            "correction. Default: 0 pixels (suite2p default 350 pixels).\n"
            "• Correction Factor: Scaling applied to neuropil fluorescence before "
            " subtraction. Accounts for the fact that neuropil contamination may "
            " differ from\n. the actual neuropil fluorescence levels. Range: 0.0-1.0, "
            " Default: 0.0 (suite2p default 0.70).\n"
            "Example with Inner Radius=2, Min Pixels=350:\n"
            "1. Cell boundary at position 0\n"
            "2. Forbidden zone: 0 to 2 pixels outward from cell edge (excluded)\n"
            "3. Initial expansion: 5 pixels at a time from forbidden zone boundary\n"
            "4. Remove any pixels overlapping with other cells\n"
            "5. Continue expanding until ≥350 valid pixels (max 100 iterations)\n"
            "6. Corrected signal = Cell - 0.7 x Neuropil"
        )

        self._neuropil_inner_radius_lbl = QLabel("Inner Radius (pixels):")
        self._neuropil_inner_radius_spin = QSpinBox(self)
        self._neuropil_inner_radius_spin.setRange(0, 100)
        self._neuropil_inner_radius_spin.setValue(DEFAULT_NEUROPIL_INNER_RADIUS)
        np_radius_wdg = QWidget(self)
        np_radius_layout = QHBoxLayout(np_radius_wdg)
        np_radius_layout.setContentsMargins(0, 0, 0, 0)
        np_radius_layout.setSpacing(5)
        np_radius_layout.addWidget(self._neuropil_inner_radius_lbl)
        np_radius_layout.addWidget(self._neuropil_inner_radius_spin)

        self._neuropil_min_px_lbl = QLabel("Min Pixels:")
        self._neuropil_min_px_spin = QSpinBox(self)
        self._neuropil_min_px_spin.setRange(0, 2000)
        self._neuropil_min_px_spin.setValue(DEFAULT_NEUROPIL_MIN_PIXELS)
        np_min_pixels_wdg = QWidget(self)
        np_min_pixels_layout = QHBoxLayout(np_min_pixels_wdg)
        np_min_pixels_layout.setContentsMargins(0, 0, 0, 0)
        np_min_pixels_layout.setSpacing(5)
        np_min_pixels_layout.addWidget(self._neuropil_min_px_lbl)
        np_min_pixels_layout.addWidget(self._neuropil_min_px_spin)

        self._neuropil_factor_lbl = QLabel("Correction Factor:")
        self._neuropil_factor_spin = QDoubleSpinBox(self)
        self._neuropil_factor_spin.setRange(0.0, 1.0)
        self._neuropil_factor_spin.setSingleStep(0.1)
        self._neuropil_factor_spin.setValue(DEFAULT_NEUROPIL_CORRECTION_FACTOR)
        np_factor_wdg = QWidget(self)
        np_factor_layout = QHBoxLayout(np_factor_wdg)
        np_factor_layout.setContentsMargins(0, 0, 0, 0)
        np_factor_layout.setSpacing(5)
        np_factor_layout.addWidget(self._neuropil_factor_lbl)
        np_factor_layout.addWidget(self._neuropil_factor_spin)

        neuropil_layout = QVBoxLayout(self)
        neuropil_layout.setContentsMargins(0, 0, 0, 0)
        neuropil_layout.setSpacing(5)
        neuropil_layout.addWidget(np_radius_wdg)
        neuropil_layout.addWidget(np_min_pixels_wdg)
        neuropil_layout.addWidget(np_factor_wdg)

    def value(self) -> NeuropilData:
        """Get the current values of the widget."""
        return NeuropilData(
            self._neuropil_inner_radius_spin.value(),
            self._neuropil_min_px_spin.value(),
            self._neuropil_factor_spin.value(),
        )

    def setValue(self, value: NeuropilData) -> None:
        """Set the values of the widget."""
        self._neuropil_inner_radius_spin.setValue(value.neuropil_inner_radius)
        self._neuropil_min_px_spin.setValue(value.neuropil_min_pixels)
        self._neuropil_factor_spin.setValue(value.neuropil_correction_factor)

    def reset(self) -> None:
        """Reset the widget to default values."""
        self._neuropil_inner_radius_spin.setValue(DEFAULT_NEUROPIL_INNER_RADIUS)
        self._neuropil_min_px_spin.setValue(DEFAULT_NEUROPIL_MIN_PIXELS)
        self._neuropil_factor_spin.setValue(DEFAULT_NEUROPIL_CORRECTION_FACTOR)

    def set_labels_width(self, width: int) -> None:
        """Set the width of the labels."""
        self._neuropil_inner_radius_lbl.setFixedWidth(width)
        self._neuropil_min_px_lbl.setFixedWidth(width)
        self._neuropil_factor_lbl.setFixedWidth(width)


class _TraceExtractionWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # ΔF/F0 windows
        self._dff_wdg = QWidget(self)
        self._dff_wdg.setToolTip(
            "Sliding Window Size for ΔF/F₀ Baseline (milliseconds)\n\n"
            "Controls the duration of the sliding window used to estimate the baseline "
            "fluorescence F₀ for ΔF/F₀ computation in single-photon calcium imaging."
            "\n\nHow the baseline is computed:\n"
            "• A centered sliding window is taken around each timepoint\n"
            "• The 10th percentile of fluorescence values within that window is used as"
            " the baseline F₀\n"
            "• ΔF/F₀ is calculated as: (F - F₀) / F₀\n\n"
            "Choosing the window size:\n"
            "• Large windows (10000-60000 ms): Very stable baseline; best for "
            "recordings with slow drift or bleaching\n"
            "• Medium windows (5000-15000 ms): Good all-purpose choice; follows "
            "baseline variations without tracking individual transients too closely\n"
            "• Small windows (<2000 ms): Baseline begins to follow the activity itself,"
            " which can reduce ΔF/F₀ amplitude and distort transients; not recommended"
            " in most cases\n\n"
            "Recommended default: 5000-10000 ms (5-10 seconds), depending on frame rate"
            " and expected drift. Default: 10000 ms (15 seconds)"
        )
        self._dff_lbl = QLabel("ΔF/F0 Window (ms):")
        self._dff_lbl.setSizePolicy(*FIXED)
        self._dff_window_size_spin = QDoubleSpinBox(self)
        self._dff_window_size_spin.setRange(0.1, 1000000)
        self._dff_window_size_spin.setSingleStep(100)
        self._dff_window_size_spin.setValue(DEFAULT_DFF_WINDOW)
        dff_layout = QHBoxLayout(self._dff_wdg)
        dff_layout.setContentsMargins(0, 0, 0, 0)
        dff_layout.setSpacing(5)
        dff_layout.addWidget(self._dff_lbl)
        dff_layout.addWidget(self._dff_window_size_spin)

        # Deconvolution decay constant
        self._dec_wdg = QWidget(self)
        self._dec_wdg.setToolTip(
            "Decay constant (tau) for calcium indicator deconvolution.\n"
            "Set to 0 for automatic estimation by OASIS algorithm.\n\n"
            "The decay constant represents how quickly the calcium indicator\n"
            "returns to baseline after a calcium transient."
        )
        self._decay_const_lbl = QLabel("Decay Constant (s):")
        self._decay_const_lbl.setSizePolicy(*FIXED)
        self._decay_constant_spin = QDoubleSpinBox(self)
        self._decay_constant_spin.setDecimals(2)
        self._decay_constant_spin.setRange(0.0, 10.0)
        self._decay_constant_spin.setSingleStep(0.1)
        self._decay_constant_spin.setSpecialValueText("Auto")
        dec_wdg_layout = QHBoxLayout(self._dec_wdg)
        dec_wdg_layout.setContentsMargins(0, 0, 0, 0)
        dec_wdg_layout.setSpacing(5)
        dec_wdg_layout.addWidget(self._decay_const_lbl)
        dec_wdg_layout.addWidget(self._decay_constant_spin)

        # main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._dff_wdg)
        layout.addWidget(self._dec_wdg)

    # PUBLIC METHODS ------------------------------------------------------------------

    def set_labels_width(self, width: int) -> None:
        """Set the width of the labels."""
        self._dff_lbl.setFixedWidth(width)
        self._decay_const_lbl.setFixedWidth(width)

    def value(self, neuropil_data: NeuropilData) -> TraceExtractionData:
        """Get the current values of the widget."""
        return TraceExtractionData(
            self._dff_window_size_spin.value(),
            self._decay_constant_spin.value(),
            neuropil_data.neuropil_inner_radius,
            neuropil_data.neuropil_min_pixels,
            neuropil_data.neuropil_correction_factor,
        )

    def setValue(self, value: TraceExtractionData) -> None:
        """Set the values of the widget."""
        self._dff_window_size_spin.setValue(value.dff_window_size)
        self._decay_constant_spin.setValue(value.decay_constant)

    def reset(self) -> None:
        """Reset the widget to default values."""
        self._dff_window_size_spin.setValue(DEFAULT_DFF_WINDOW)
        self._decay_constant_spin.setValue(0.0)


class _MetadataWidget(QWidget):
    """Widget for metadata settings including pixel size and frame rate."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Pixel Size widget
        self._pixel_size_wdg = QWidget(self)
        self._pixel_size_wdg.setToolTip(
            "Physical size of each pixel in micrometers (µm).\n\n"
            "Used to convert pixel-based measurements to physical units.\n"
            "Set to 0 to use pixels as the unit (no conversion).\n\n"
            "Default: 0 (use pixels)"
        )
        self._pixel_size_lbl = QLabel("Pixel Size:")
        self._pixel_size_lbl.setSizePolicy(*FIXED)
        self._pixel_size_spin = QDoubleSpinBox(self)
        self._pixel_size_spin.setSuffix(" µm")
        self._pixel_size_spin.setDecimals(4)
        self._pixel_size_spin.setRange(0.0, 100.0)
        self._pixel_size_spin.setSingleStep(0.1)
        self._pixel_size_spin.setValue(0.0)
        self._pixel_size_spin.setSpecialValueText("Use Pixels")
        pixel_size_layout = QHBoxLayout(self._pixel_size_wdg)
        pixel_size_layout.setContentsMargins(0, 0, 0, 0)
        pixel_size_layout.setSpacing(5)
        pixel_size_layout.addWidget(self._pixel_size_lbl)
        pixel_size_layout.addWidget(self._pixel_size_spin)

        # Frame Rate widget
        self._frame_rate_wdg = QWidget(self)
        self._frame_rate_wdg.setToolTip(
            "Acquisition frame rate in frames per second (fps).\n\n"
            "This is used to convert time-based parameters (e.g., DFF window in "
            "milliseconds) to frames for processing.\n\n"
            "Tip: This is typically the inverse of exposure time:\n"
            "• Exposure = 50ms → Frame Rate = 20 fps (1000/50)\n"
            "• Exposure = 100ms → Frame Rate = 10 fps (1000/100)"
        )
        self._frame_rate_lbl = QLabel("Frame Rate:")
        self._frame_rate_lbl.setSizePolicy(*FIXED)
        self._frame_rate_spin = QDoubleSpinBox(self)
        self._frame_rate_spin.setSuffix(" fps")
        self._frame_rate_spin.setDecimals(2)
        self._frame_rate_spin.setRange(0.01, 1000.0)
        self._frame_rate_spin.setSingleStep(1.0)
        self._frame_rate_spin.setValue(DEFAULT_FRAME_RATE)
        frame_rate_layout = QHBoxLayout(self._frame_rate_wdg)
        frame_rate_layout.setContentsMargins(0, 0, 0, 0)
        frame_rate_layout.setSpacing(5)
        frame_rate_layout.addWidget(self._frame_rate_lbl)
        frame_rate_layout.addWidget(self._frame_rate_spin)

        # Left side: metadata fields vertically
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5)
        left_layout.addWidget(self._pixel_size_wdg)
        left_layout.addWidget(self._frame_rate_wdg)

        # Right side: FromMetaButton extending vertically
        self._from_meta_btn = FromMetaButton(self, "Load From Metadata")
        self._from_meta_btn.setToolTip(
            "Try to load pixel size and frame rate from the acquisition metadata."
        )
        self._from_meta_btn.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
        )

        # Main layout: left fields + right button
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addLayout(left_layout)
        layout.addWidget(self._from_meta_btn)

    # PUBLIC METHODS ------------------------------------------------------------------

    @property
    def from_metadata(self) -> None:
        """Signal emitted when the 'Load From Metadata' button is clicked."""
        return self._from_meta_btn.clicked  # type: ignore

    def set_labels_width(self, width: int) -> None:
        """Set the width of the labels."""
        self._pixel_size_lbl.setFixedWidth(width)
        self._frame_rate_lbl.setFixedWidth(width)

    def value(self) -> MetadataData:
        """Get the current values of the widget."""
        pixel_size = self._pixel_size_spin.value()
        return MetadataData(
            pixel_size=pixel_size if pixel_size > 0 else None,
            frame_rate=self._frame_rate_spin.value(),
        )

    def setValue(self, value: MetadataData) -> None:
        """Set the values of the widget."""
        if value.pixel_size is not None:
            self._pixel_size_spin.setValue(value.pixel_size)
        else:
            self._pixel_size_spin.setValue(0.0)
        self._frame_rate_spin.setValue(value.frame_rate)

    def reset(self) -> None:
        """Reset the widget to default values."""
        self._pixel_size_spin.setValue(0.0)
        self._frame_rate_spin.setValue(DEFAULT_FRAME_RATE)


class FromMetaButton(QPushButton):
    """Custom button for loading metadata from files."""

    def __init__(self, parent: QWidget | None = None, text: str = "") -> None:
        super().__init__(text, parent)
        self.setIcon(QIconifyIcon("mdi:file-document-box-search"))
        self.setFixedWidth(200)
