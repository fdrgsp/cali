from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from fonticon_mdi6 import MDI6
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QDialog,
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
from superqt.fonticon import icon

from cali._constants import (
    DEFAULT_DFF_WINDOW,
    DEFAULT_NEUROPIL_CORRECTION_FACTOR,
    DEFAULT_NEUROPIL_INNER_RADIUS,
    DEFAULT_NEUROPIL_MIN_PIXELS,
)
from cali.sqlmodel import ExtractionSettings

from ._plate_map import PlateMapData, PlateMapWidget
from ._util import (
    create_divider_line,
)

if TYPE_CHECKING:
    import useq


FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed


@dataclass(frozen=True)
class ExtractionSettingsData:
    """Data structure to hold the extraction settings."""

    plate_map_data: (
        tuple[useq.WellPlate | None, list[PlateMapData], list[PlateMapData]] | None
    ) = None
    trace_extraction_data: TraceExtractionData | None = None


@dataclass(frozen=True)
class NeuropilData:
    """Data structure to hold the neuropil correction settings."""

    neuropil_inner_radius: int
    neuropil_min_pixels: int
    neuropil_correction_factor: float


@dataclass(frozen=True)
class TraceExtractionData:
    """Data structure to hold the trace extraction settings."""

    dff_window_size: int
    decay_constant: float
    neuropil_inner_radius: int
    neuropil_min_pixels: int
    neuropil_correction_factor: float


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
        threads_lbl = QLabel("Number of Threads:")
        threads_lbl.setSizePolicy(*FIXED)
        self._threads = QSpinBox()
        self._threads.setRange(1, 100)
        self._threads.setValue(cpu_to_use)
        threads_layout = QHBoxLayout(threads_wdg)
        threads_layout.setContentsMargins(0, 0, 0, 0)
        threads_layout.setSpacing(5)
        threads_layout.addWidget(threads_lbl)
        threads_layout.addWidget(self._threads)

        # EXTRACTION WIDGETS ---------------------------------------------------------
        self._plate_map_wdg = _PlateMapWidget(self)
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
        group_layout.addWidget(create_divider_line("Plate Map"))
        group_layout.addWidget(self._plate_map_wdg)
        group_layout.addWidget(create_divider_line("Neuropil Settings"))
        group_layout.addWidget(self._neuropil_wdg)
        group_layout.addWidget(create_divider_line("ΔF/F0 and Deconvolution"))
        group_layout.addWidget(self._trace_extraction_wdg)
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
        fix_width = self._neuropil_wdg._neuropil_inner_radius_lbl.sizeHint().width()
        self._trace_extraction_wdg.set_labels_width(fix_width)
        self._plate_map_wdg.set_labels_width(fix_width)
        self._neuropil_wdg.set_labels_width(fix_width)
        self._trace_extraction_wdg.set_labels_width(fix_width)
        threads_lbl.setFixedWidth(fix_width)

    # PUBLIC METHODS ------------------------------------------------------------------

    def value(self) -> ExtractionSettingsData:
        """Get the current values of the widget."""
        return ExtractionSettingsData(
            self._plate_map_wdg.value(),
            self._trace_extraction_wdg.value(self._neuropil_wdg.value()),
        )

    def setValue(self, value: ExtractionSettingsData) -> None:
        """Set the values of the widget."""
        if value.plate_map_data is not None:
            plate, genotype_map, treatment_map = value.plate_map_data
            self._plate_map_wdg.setValue(plate, genotype_map, treatment_map)
        if value.trace_extraction_data is not None:
            self._trace_extraction_wdg.setValue(value.trace_extraction_data)
            # Also set the neuropil widget from trace extraction data
            neuropil_data = NeuropilData(
                value.trace_extraction_data.neuropil_inner_radius,
                value.trace_extraction_data.neuropil_min_pixels,
                value.trace_extraction_data.neuropil_correction_factor,
            )
            self._neuropil_wdg.setValue(neuropil_data)

    def reset(self) -> None:
        """Reset the widget to default values."""
        self._plate_map_wdg.clear()
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
            # frame_rate=self._frame_rate_wdg.value(),
        )

        return settings


class _PlateMapWidget(QWidget):
    """Widget to show and edit the plate maps."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._plate: useq.WellPlate | None = None

        # label
        self._plate_map_lbl = QLabel("Set/Edit Plate Map:")
        self._plate_map_lbl.setSizePolicy(*FIXED)

        # button to show the plate map dialog
        self._plate_map_btn = QPushButton("Show/Edit Plate Map")
        self._plate_map_btn.setIcon(icon(MDI6.view_comfy))
        self._plate_map_btn.clicked.connect(self._show_plate_map_dialog)

        # dialog to show the plate maps
        self._plate_map_dialog = QDialog(self)
        plate_map_layout = QHBoxLayout(self._plate_map_dialog)
        plate_map_layout.setContentsMargins(10, 10, 10, 10)
        plate_map_layout.setSpacing(5)
        self._plate_map_genotype = PlateMapWidget(self, title="Genotype Map")
        self._plate_map_treatment = PlateMapWidget(self, title="Treatment Map")
        plate_map_layout.addWidget(self._plate_map_genotype)
        plate_map_layout.addWidget(self._plate_map_treatment)

        # main layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._plate_map_lbl)
        layout.addWidget(self._plate_map_btn)
        layout.addStretch(1)

    # PUBLIC METHODS ------------------------------------------------------------------

    def value(
        self,
    ) -> tuple[useq.WellPlate | None, list[PlateMapData], list[PlateMapData]]:
        """Get the plate map data."""
        return (
            self._plate,
            self._plate_map_genotype.value(),
            self._plate_map_treatment.value(),
        )

    def setValue(
        self,
        plate: useq.WellPlate | None,
        genotype_map: list[PlateMapData],
        treatment_map: list[PlateMapData],
    ) -> None:
        """Set the plate map data."""
        self.setPlate(plate)
        self._plate_map_genotype.setValue(genotype_map)
        self._plate_map_treatment.setValue(treatment_map)

    def set_labels_width(self, width: int) -> None:
        """Set the width of the labels."""
        self._plate_map_lbl.setFixedWidth(width)

    def setPlate(self, plate: useq.WellPlate | None) -> None:
        """Set the plate for the plate maps."""
        self._plate = plate
        if plate is None:
            self.clear()
            return
        self._plate_map_genotype.setPlate(plate)
        self._plate_map_treatment.setPlate(plate)

    def clear(self) -> None:
        """Clear the plate map data."""
        self._plate_map_genotype.clear()
        self._plate_map_treatment.clear()

    # PRIVATE METHODS -----------------------------------------------------------------

    def _show_plate_map_dialog(self) -> None:
        """Show the plate map dialog."""
        # ensure the dialog is visible and properly positioned
        if self._plate_map_dialog.isHidden() or not self._plate_map_dialog.isVisible():
            self._plate_map_dialog.show()
        # always try to bring to front and activate
        self._plate_map_dialog.raise_()
        self._plate_map_dialog.activateWindow()
        # force focus on the dialog
        self._plate_map_dialog.setFocus()


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
            "Controls the sliding window size for calculating ΔF/F₀ baseline "
            "(expressed in frames).\n\n"
            "The algorithm uses a sliding window to estimate the background "
            "fluorescence:\n"
            "• For each timepoint, calculates the 10th percentile within the window\n"
            "• Window extends from current timepoint backwards by window_size/2 "
            "frames\n"
            "• ΔF/F₀ = (fluorescence - background) / background\n\n"
            "Window size considerations:\n"
            "• Larger values (200-500): More stable baseline, good for slow drifts\n"
            "• Smaller values (50-100): More adaptive, follows local fluorescence "
            "changes\n"
            "• Too small (<20): May track signal itself, reducing ΔF/F₀ sensitivity\n"
            "• Too large (>1000): May not adapt to legitimate baseline shifts."
        )
        self._dff_lbl = QLabel("ΔF/F0 Window Size:")
        self._dff_lbl.setSizePolicy(*FIXED)
        self._dff_window_size_spin = QSpinBox(self)
        self._dff_window_size_spin.setRange(0, 10000)
        self._dff_window_size_spin.setSingleStep(1)
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
