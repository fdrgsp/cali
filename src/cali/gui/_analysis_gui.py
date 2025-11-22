from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from fonticon_mdi6 import MDI6
from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from superqt.fonticon import icon

from cali._constants import (
    DEFAULT_BURST_GAUSS_SIGMA,
    DEFAULT_BURST_THRESHOLD,
    DEFAULT_CALCIUM_NETWORK_THRESHOLD,
    DEFAULT_CALCIUM_SYNC_JITTER_WINDOW,
    DEFAULT_DFF_WINDOW,
    DEFAULT_FRAME_RATE,
    DEFAULT_HEIGHT,
    DEFAULT_MIN_BURST_DURATION,
    DEFAULT_NEUROPIL_CORRECTION_FACTOR,
    DEFAULT_NEUROPIL_INNER_RADIUS,
    DEFAULT_NEUROPIL_MIN_PIXELS,
    DEFAULT_PEAKS_DISTANCE,
    DEFAULT_SPIKE_SYNCHRONY_MAX_LAG,
    DEFAULT_SPIKE_THRESHOLD,
    EVOKED,
    GLOBAL_HEIGHT,
    GLOBAL_SPIKE_THRESHOLD,
    GREEN,
    MULTIPLIER,
    RED,
    SPONTANEOUS,
)

# from ._plate_map import PlateMapWidget
from ._plate_map import PlateMapData, PlateMapWidget
from ._util import (
    _BrowseWidget,
    _ChoosePositionsWidget,
    create_divider_line,
    parse_lineedit_text,
)

if TYPE_CHECKING:
    import useq

    from cali.sqlmodel import AnalysisSettings

FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed


@dataclass(frozen=True)
class AnalysisSettingsData:
    """Data structure to hold the analysis settings."""

    plate_map_data: (
        tuple[useq.WellPlate | None, list[PlateMapData], list[PlateMapData]] | None
    ) = None
    experiment_type_data: ExperimentTypeData | None = None
    trace_extraction_data: TraceExtractionData | None = None
    calcium_peaks_data: CalciumPeaksData | None = None
    spikes_data: SpikeData | None = None


@dataclass(frozen=True)
class ExperimentTypeData:
    """Data structure to hold the experiment type settings."""

    experiment_type: str | None = None
    led_power_equation: str | None = None
    led_pulse_duration: float | None = None
    led_pulse_powers: list[float] | None = None
    led_pulse_on_frames: list[int] | None = None
    stimulation_area_path: str | None = None


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


@dataclass(frozen=True)
class CalciumPeaksData:
    """Data structure to hold the calcium peaks settings."""

    peaks_height: float
    peaks_height_mode: str
    peaks_distance: int
    peaks_prominence_multiplier: float
    calcium_synchrony_jitter: int
    calcium_network_threshold: float


@dataclass(frozen=True)
class SpikeData:
    """Data structure to hold the spikes settings."""

    spike_threshold: float
    spike_threshold_mode: str
    burst_threshold: float
    burst_min_duration: int
    burst_blur_sigma: float
    synchrony_lag: int


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


class FromMetaButton(QPushButton):
    """Button to load values from metadata."""

    def __init__(self, parent: QWidget | None = None, text: str = "") -> None:
        super().__init__(parent)
        self.setText(text)
        # self.setIcon(icon(MDI6.database_search))
        # self.setIconSize(QSize(24, 24))


class _ExperimentTypeWidget(QWidget):
    """Widget to select the type of experiment (spontaneous or evoked)...

    ...and related settings.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # experiment type combo
        self._experiment_type_lbl = QLabel("Experiment Type:")
        self._experiment_type_lbl.setSizePolicy(*FIXED)
        self._experiment_type_combo = QComboBox()
        self._experiment_type_combo.addItems([SPONTANEOUS, EVOKED])
        self._experiment_type_combo.currentTextChanged.connect(
            self._on_activity_changed
        )
        experiment_type_layout = QHBoxLayout()
        experiment_type_layout.setContentsMargins(0, 0, 0, 0)
        experiment_type_layout.setSpacing(5)
        experiment_type_layout.addWidget(self._experiment_type_lbl)
        experiment_type_layout.addWidget(self._experiment_type_combo)

        # path selector for stimulated area mask
        self._stimulation_area_path_dialog = _BrowseWidget(
            self,
            label="Stimulated Area File",
            tooltip=(
                "Select the path to the image of the stimulated area.\n"
                "The image should either be a binary mask or a grayscale image where "
                "the stimulated area is brighter than the rest.\n"
                "Accepted formats: .tif, .tiff."
            ),
            is_dir=False,
        )
        self._stimulation_area_path_dialog.hide()

        # LED power equation widget
        self._led_power_eq = QWidget(self)
        self._led_power_eq.setToolTip(
            "Insert an equation to convert the LED power to mW.\n"
            "Supported formats:\n"
            "• Linear: y = m*x + q (e.g. y = 2*x + 3)\n"
            "• Quadratic: y = a*x^2 + b*x + c (e.g. y = 0.5*x^2 + 2*x + 1)\n"
            "• Exponential: y = a*exp(b*x) + c (e.g. y = 2*exp(0.1*x) + 1)\n"
            "• Power: y = a*x^b + c (e.g. y = 2*x^0.5 + 1)\n"
            "• Logarithmic: y = a*log(x) + b (e.g. y = 2*log(x) + 1)\n"
            "Leave empty to use values from the acquisition metadata (%)."
        )
        self._led_eq_lbl = QLabel("LED Power Equation:")
        self._led_eq_lbl.setSizePolicy(*FIXED)
        self._led_power_equation_le = QLineEdit(self)
        self._led_power_equation_le.setPlaceholderText(
            "e.g. y = 2*x + 3 (Leave empty to use values from acquisition metadata)"
        )
        led_layout = QHBoxLayout(self._led_power_eq)
        led_layout.setContentsMargins(0, 0, 0, 0)
        led_layout.setSpacing(5)
        led_layout.addWidget(self._led_eq_lbl)
        led_layout.addWidget(self._led_power_equation_le)
        self._led_power_eq.hide()

        # LED pulse duration widget
        self._led_pulse_duration_wdg = QWidget(self)
        self._led_pulse_duration_wdg.setToolTip(
            "Duration of each LED pulse in milliseconds."
        )
        self._led_pulse_duration_lbl = QLabel("LED Pulse Duration (ms):")
        self._led_pulse_duration_lbl.setSizePolicy(*FIXED)
        self._led_pulse_duration_spin = QDoubleSpinBox(self)
        self._led_pulse_duration_spin.setRange(0.0, 10000.0)
        led_pulse_layout = QHBoxLayout(self._led_pulse_duration_wdg)
        led_pulse_layout.setContentsMargins(0, 0, 0, 0)
        led_pulse_layout.setSpacing(5)
        led_pulse_layout.addWidget(self._led_pulse_duration_lbl)
        led_pulse_layout.addWidget(self._led_pulse_duration_spin)
        self._led_power_eq.hide()

        # LED pulse powers widget
        self._led_powers_wdg = QWidget(self)
        self._led_powers_wdg.setToolTip(
            "List of LED pulse powers corresponding to each stimulation frame.\n"
            "Values should be in percentage (%), separated by commas "
            "(e.g. 20, 40, 60, 80).\n"
            "The length of this list should match the length of the 'Stimulation "
            "Frames' list."
        )
        self._led_powers_lbl = QLabel("LED Pulse Powers (%):")
        self._led_powers_lbl.setSizePolicy(*FIXED)
        self._led_powers_le = QLineEdit(self)
        self._led_powers_le.setPlaceholderText("e.g. 20, 40, 60, 80")
        led_powers_layout = QHBoxLayout(self._led_powers_wdg)
        led_powers_layout.setContentsMargins(0, 0, 0, 0)
        led_powers_layout.setSpacing(5)
        led_powers_layout.addWidget(self._led_powers_lbl)
        led_powers_layout.addWidget(self._led_powers_le)
        self._led_powers_wdg.hide()

        # LED pulse on frames widget
        self._led_pulse_on_frames_wdg = QWidget(self)
        self._led_pulse_on_frames_wdg.setToolTip(
            "List of frames where the LED was ON during the experiment.\n"
            "Values should be integers separated by commas (e.g. 1, 5, 10, 15).\n"
            "The length of this list should match the length of the 'LED Pulse Powers' "
            "list."
        )
        self.led_pulse_on_frames_lbl = QLabel("Stimulation Frames:")
        self.led_pulse_on_frames_lbl.setSizePolicy(*FIXED)
        self._led_pulse_on_frames_le = QLineEdit(self)
        self._led_pulse_on_frames_le.setPlaceholderText("e.g. 1, 5, 10, 15")
        led_pulse_on_frames_layout = QHBoxLayout(self._led_pulse_on_frames_wdg)
        led_pulse_on_frames_layout.setContentsMargins(0, 0, 0, 0)
        led_pulse_on_frames_layout.setSpacing(5)
        led_pulse_on_frames_layout.addWidget(self.led_pulse_on_frames_lbl)
        led_pulse_on_frames_layout.addWidget(self._led_pulse_on_frames_le)
        self._led_pulse_on_frames_wdg.hide()

        # led settings left
        left_setting_layout = QVBoxLayout()
        left_setting_layout.setContentsMargins(0, 0, 0, 0)
        left_setting_layout.setSpacing(7)
        left_setting_layout.addWidget(self._led_pulse_duration_wdg)
        left_setting_layout.addWidget(self._led_powers_wdg)
        left_setting_layout.addWidget(self._led_pulse_on_frames_wdg)
        # led from meta button right
        self._from_meta_btn = FromMetaButton(self, "Load From Metadata")
        self._from_meta_btn.setToolTip(
            "Try to load the LED settings from the acquisition metadata."
        )
        self._from_meta_btn.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding
        )
        self._from_meta_btn.hide()

        # left/right widget
        left_right_layout = QHBoxLayout()
        left_right_layout.setContentsMargins(0, 0, 0, 0)
        left_right_layout.setSpacing(5)
        left_right_layout.addLayout(left_setting_layout)
        left_right_layout.addWidget(self._from_meta_btn)

        # main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addLayout(experiment_type_layout)
        layout.addWidget(self._stimulation_area_path_dialog)
        layout.addWidget(self._led_power_eq)
        layout.addLayout(left_right_layout)

    # PUBLIC METHODS ------------------------------------------------------------------

    def set_labels_width(self, width: int) -> None:
        """Set the width of the labels."""
        self._experiment_type_lbl.setFixedWidth(width)
        self._stimulation_area_path_dialog._label.setFixedWidth(width)
        self._led_eq_lbl.setFixedWidth(width)
        self._led_pulse_duration_lbl.setFixedWidth(width)
        self._led_powers_lbl.setFixedWidth(width)
        self.led_pulse_on_frames_lbl.setFixedWidth(width)

    def value(self) -> ExperimentTypeData:
        """Get the current values of the widget."""
        return ExperimentTypeData(
            self._experiment_type_combo.currentText(),
            self._led_power_equation_le.text(),
            self._led_pulse_duration_spin.value(),
            self._parse_to_list(self._led_powers_le.text()),
            self._parse_to_list(self._led_pulse_on_frames_le.text()),  # type: ignore
            self._stimulation_area_path_dialog.value(),
        )

    def setValue(self, value: ExperimentTypeData) -> None:
        """Set the values of the widget."""
        if value.led_power_equation is not None:
            self._led_power_equation_le.setText(value.led_power_equation)
        if value.stimulation_area_path is not None:
            self._stimulation_area_path_dialog.setValue(value.stimulation_area_path)
        if value.led_pulse_duration is not None:
            self._led_pulse_duration_spin.setValue(value.led_pulse_duration)
        if value.led_pulse_powers is not None:
            self._led_powers_le.setText(
                ", ".join(str(power) for power in value.led_pulse_powers)
            )
        if value.led_pulse_on_frames is not None:
            self._led_pulse_on_frames_le.setText(
                ", ".join(str(frame) for frame in value.led_pulse_on_frames)
            )
        if value.experiment_type is not None:
            self._experiment_type_combo.setCurrentText(value.experiment_type)
            # update visibility based on experiment type
            self._on_activity_changed(value.experiment_type)

    def reset(self) -> None:
        """Clear the widget values."""
        self._experiment_type_combo.setCurrentText(SPONTANEOUS)
        self._led_power_equation_le.clear()
        self._stimulation_area_path_dialog.clear()
        self._led_pulse_duration_spin.setValue(0.0)
        self._led_powers_le.clear()
        self._led_pulse_on_frames_le.clear()

    # PRIVATE METHODS -----------------------------------------------------------------

    def _parse_to_list(self, text: str) -> list[int | float]:
        """Parse a comma-separated string into a list of floats."""
        parsed: list[float | int] = []
        for val in text.split(","):
            val = val.strip()
            try:
                power = float(val)
                parsed.append(power)
            except ValueError:
                continue
        return parsed

    def _on_activity_changed(self, text: str) -> None:
        """Show or hide the stimulation area path and LED power widgets."""
        if text == EVOKED:
            self._stimulation_area_path_dialog.show()
            self._led_power_eq.show()
            self._led_pulse_duration_wdg.show()
            self._led_powers_wdg.show()
            self._led_pulse_on_frames_wdg.show()
            self._from_meta_btn.show()
        else:
            self._stimulation_area_path_dialog.hide()
            self._led_power_eq.hide()
            self._led_pulse_duration_wdg.hide()
            self._led_powers_wdg.hide()
            self._led_pulse_on_frames_wdg.hide()
            self._from_meta_btn.hide()


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
        self._dff_lbl = QLabel("ΔF/F0 Window Size")
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


class _PeaksHeightWidget(QWidget):
    """Widget to select the peaks height multiplier."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setToolTip(
            "Peak height threshold for detecting calcium transients in deconvolved "
            "ΔF/F0 traces using scipy.signal.find_peaks.\n\n"
            "Two modes:\n"
            "• Global Minimum: Same absolute threshold applied to ALL ROIs across "
            "ALL FOVs. Peaks below this value are rejected everywhere.\n\n"
            "• Noise Multiplier: Adaptive threshold computed individually for EACH "
            "ROI in EACH FOV.\n"
            "  Threshold = noise_level * multiplier, where noise_level "
            "is calculated per ROI using Median Absolute Deviation (MAD).\n\n"
            "For example, a multiplier of 3.0 can be use to detect events 3 standard "
            "deviations above noise."
        )

        self._peaks_height_lbl = QLabel("Minimum Peaks Height:")
        self._peaks_height_lbl.setSizePolicy(*FIXED)

        self._peaks_height_spin = QDoubleSpinBox(self)
        self._peaks_height_spin.setDecimals(4)
        self._peaks_height_spin.setRange(0.0, 100000.0)
        self._peaks_height_spin.setSingleStep(0.01)
        self._peaks_height_spin.setValue(DEFAULT_HEIGHT)

        self._global_peaks_height = QRadioButton("Use as Global Minimum Peaks Height")

        self._height_multiplier = QRadioButton("Use as Noise Level Multiplier")
        self._height_multiplier.setChecked(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._peaks_height_lbl)
        layout.addWidget(self._peaks_height_spin, 1)
        layout.addWidget(self._height_multiplier, 0)
        layout.addWidget(self._global_peaks_height, 0)

    # PUBLIC METHODS ------------------------------------------------------------------

    def value(self) -> tuple[float, str]:
        """Return the value of the peaks height multiplier."""
        return (
            self._peaks_height_spin.value(),
            GLOBAL_HEIGHT if self._global_peaks_height.isChecked() else MULTIPLIER,
        )

    def setValue(self, value: tuple[float, str]) -> None:
        """Set the value of the peaks height widget."""
        height, mode = value
        self._peaks_height_spin.setValue(height)
        self._global_peaks_height.setChecked(mode == GLOBAL_HEIGHT)
        self._height_multiplier.setChecked(mode == MULTIPLIER)


class _CalciumPeaksWidget(QWidget):
    """Widget to select the calcium peaks settings."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # peaks height
        self._peaks_height = _PeaksHeightWidget(self)

        # peaks minimum distance
        self._peaks_distance_wdg = QWidget(self)
        self._peaks_distance_wdg.setToolTip(
            "Minimum distance between peaks in frames.\n"
            "This prevents detecting multiple peaks from the same calcium event.\n\n"
            "Example: If exposure time = 50ms and you want 100ms minimum separation,\n"
            "set distance = 2 frames (100ms ÷ 50ms = 2 frames).\n\n"
            "• Higher values: More conservative, fewer detected peaks\n"
            "• Lower values: More sensitive, may detect noise or incomplete decay\n"
            "• Minimum value: 1 (adjacent frames allowed)."
        )
        self._peaks_distance_lbl = QLabel("Minimum Peaks Distance:")
        self._peaks_distance_lbl.setSizePolicy(*FIXED)
        self._peaks_distance_spin = QSpinBox(self)
        self._peaks_distance_spin.setRange(1, 1000)
        self._peaks_distance_spin.setSingleStep(1)
        self._peaks_distance_spin.setValue(2)
        peaks_distance_layout = QHBoxLayout(self._peaks_distance_wdg)
        peaks_distance_layout.setContentsMargins(0, 0, 0, 0)
        peaks_distance_layout.setSpacing(5)
        peaks_distance_layout.addWidget(self._peaks_distance_lbl)
        peaks_distance_layout.addWidget(self._peaks_distance_spin)

        # peaks prominence
        self._peaks_prominence_wdg = QWidget(self)
        self._peaks_prominence_wdg.setToolTip(
            "Controls the prominence threshold multiplier for peak validation.\n"
            "Prominence measures how much a peak stands out from surrounding\n"
            "baseline, helping distinguish real calcium events from noise.\n\n"
            "Prominence threshold = noise_level * multiplier\n\n"
            "• Value of 1.0: Uses noise level as prominence threshold\n"
            "• Values >1.0: Requires peaks to be more prominent than noise level\n"
            "• Values <1.0: More lenient, allows peaks closer to noise level\n\n"
            "Increase if detecting too many noise artifacts as peaks."
        )
        self._peaks_prominence_lbl = QLabel("Peaks Prominence Multiplier:")
        self._peaks_prominence_lbl.setSizePolicy(*FIXED)
        self._peaks_prominence_multiplier_spin = QDoubleSpinBox(self)
        self._peaks_prominence_multiplier_spin.setDecimals(4)
        self._peaks_prominence_multiplier_spin.setRange(0, 100000.0)
        self._peaks_prominence_multiplier_spin.setSingleStep(0.01)
        self._peaks_prominence_multiplier_spin.setValue(1)
        peaks_prominence_layout = QHBoxLayout(self._peaks_prominence_wdg)
        peaks_prominence_layout.setContentsMargins(0, 0, 0, 0)
        peaks_prominence_layout.setSpacing(5)
        peaks_prominence_layout.addWidget(self._peaks_prominence_lbl)
        peaks_prominence_layout.addWidget(self._peaks_prominence_multiplier_spin)

        # synchrony jitter window
        self._calcium_synchrony_wdg = QWidget(self)
        self._calcium_synchrony_wdg.setToolTip(
            "Calcium Peak Synchrony Analysis Settings\n\n"
            "Jitter Window Parameter:\n"
            "Controls the temporal tolerance for detecting synchronous "
            "calcium peaks.\n\n"
            "What the value means:\n"
            "• Value = 2: Peaks within ±2 frames are considered synchronous\n"
            "• Larger values detect more synchrony but may include false positives\n"
            "• Smaller values are more strict but may miss genuine synchrony\n\n"
            "Example with Jitter = 2:\n"
            "ROI 1 peaks: [10, 25, 40]  ROI 2 peaks: [12, 24, 41]\n"
            "Result: All pairs are synchronous (differences ≤ 2 frames)."
        )
        self._calcium_jitter_window_lbl = QLabel("Synchrony Jitter (frames):")
        self._calcium_jitter_window_lbl.setSizePolicy(*FIXED)
        self._calcium_synchrony_jitter_spin = QSpinBox(self)
        self._calcium_synchrony_jitter_spin.setRange(0, 100)
        self._calcium_synchrony_jitter_spin.setSingleStep(1)
        self._calcium_synchrony_jitter_spin.setValue(DEFAULT_CALCIUM_SYNC_JITTER_WINDOW)
        calcium_synchrony_layout = QHBoxLayout(self._calcium_synchrony_wdg)
        calcium_synchrony_layout.setContentsMargins(0, 0, 0, 0)
        calcium_synchrony_layout.setSpacing(5)
        calcium_synchrony_layout.addWidget(self._calcium_jitter_window_lbl)
        calcium_synchrony_layout.addWidget(self._calcium_synchrony_jitter_spin)

        # network connectivity threshold
        self._calcium_network_wdg = QWidget(self)
        self._calcium_network_wdg.setToolTip(
            "Network Connectivity Threshold (Percentile)\n\n"
            "Controls which correlation values become network connections.\n"
            "Uses PERCENTILE-based thresholding, not absolute correlation values.\n\n"
            "How it works:\n"
            "• Calculates percentile of ALL pairwise correlations\n"
            "• Only correlations above this percentile become connections\n"
            "• 90th percentile = top 10% of correlations become edges\n"
            "• 95th percentile = top 5% (more conservative)\n"
            "• 80th percentile = top 20% (more liberal)\n\n"
            "Important: A 0.95 correlation may show as 'not connected'\n"
            "if most correlations in your data are higher (e.g., 0.96-0.99).\n"
            "This ensures only the STRONGEST connections are shown\n"
            "relative to your specific dataset."
        )
        self._calcium_network_lbl = QLabel("Network Threshold (%):")
        self._calcium_network_lbl.setSizePolicy(*FIXED)
        self._calcium_network_threshold_spin = QDoubleSpinBox(self)
        self._calcium_network_threshold_spin.setRange(1.0, 100.0)
        self._calcium_network_threshold_spin.setSingleStep(5.0)
        self._calcium_network_threshold_spin.setDecimals(1)
        self._calcium_network_threshold_spin.setValue(DEFAULT_CALCIUM_NETWORK_THRESHOLD)
        calcium_network_layout = QHBoxLayout(self._calcium_network_wdg)
        calcium_network_layout.setContentsMargins(0, 0, 0, 0)
        calcium_network_layout.setSpacing(5)
        calcium_network_layout.addWidget(self._calcium_network_lbl)
        calcium_network_layout.addWidget(self._calcium_network_threshold_spin)

        # main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._peaks_height)
        layout.addWidget(self._peaks_distance_wdg)
        layout.addWidget(self._peaks_prominence_wdg)
        layout.addWidget(self._calcium_synchrony_wdg)
        layout.addWidget(self._calcium_network_wdg)

    # PUBLIC METHODS ------------------------------------------------------------------

    def set_labels_width(self, width: int) -> None:
        """Set the width of the labels."""
        self._peaks_height._peaks_height_lbl.setFixedWidth(width)
        self._peaks_distance_lbl.setFixedWidth(width)
        self._peaks_prominence_lbl.setFixedWidth(width)
        self._calcium_jitter_window_lbl.setFixedWidth(width)
        self._calcium_network_lbl.setFixedWidth(width)

    def value(self) -> CalciumPeaksData:
        """Get the current values of the widget."""
        return CalciumPeaksData(
            *self._peaks_height.value(),
            self._peaks_distance_spin.value(),
            self._peaks_prominence_multiplier_spin.value(),
            self._calcium_synchrony_jitter_spin.value(),
            self._calcium_network_threshold_spin.value(),
        )

    def setValue(self, value: CalciumPeaksData) -> None:
        """Set the values of the widget."""
        self._peaks_height.setValue((value.peaks_height, value.peaks_height_mode))
        self._peaks_distance_spin.setValue(value.peaks_distance)
        self._peaks_prominence_multiplier_spin.setValue(
            value.peaks_prominence_multiplier
        )
        self._calcium_synchrony_jitter_spin.setValue(value.calcium_synchrony_jitter)
        self._calcium_network_threshold_spin.setValue(value.calcium_network_threshold)

    def reset(self) -> None:
        """Reset the widget to default values."""
        self._peaks_height.setValue((DEFAULT_HEIGHT, MULTIPLIER))
        self._peaks_distance_spin.setValue(2)
        self._peaks_prominence_multiplier_spin.setValue(1)
        self._calcium_synchrony_jitter_spin.setValue(DEFAULT_CALCIUM_SYNC_JITTER_WINDOW)
        self._calcium_network_threshold_spin.setValue(DEFAULT_CALCIUM_NETWORK_THRESHOLD)


class _SpikeThresholdWidget(QWidget):
    """Widget to select the spike threshold multiplier."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setToolTip(
            "Spike detection threshold for identifying spikes in OASIS-deconvolved "
            "inferred spike traces.\n\n"
            "Two modes:\n"
            "• Global Minimum: Same absolute threshold applied to ALL ROIs across "
            "ALL FOVs. Spike amplitudes below this value are rejected (set to 0) "
            "everywhere.\n\n"
            "• Noise Multiplier: Adaptive threshold computed individually for EACH "
            "ROI in EACH FOV.\n"
            "  For ROIs with ≥10 detected spikes: "
            "Threshold = 10th_percentile_of_spikes * multiplier\n"
            "  For ROIs with <10 spikes: Threshold = 0.01 * multiplier (fallback)"
        )

        self._spike_threshold_lbl = QLabel("Spike Detection Threshold:")
        self._spike_threshold_lbl.setSizePolicy(*FIXED)

        self._spike_threshold_spin = QDoubleSpinBox(self)
        self._spike_threshold_spin.setDecimals(4)
        self._spike_threshold_spin.setRange(0.0, 10000.0)
        self._spike_threshold_spin.setSingleStep(0.1)
        self._spike_threshold_spin.setValue(DEFAULT_SPIKE_THRESHOLD)

        self._global_spike_threshold = QRadioButton("Use as Global Minimum Threshold")

        self._threshold_multiplier = QRadioButton("Use as Noise Level Multiplier")
        self._threshold_multiplier.setChecked(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._spike_threshold_lbl)
        layout.addWidget(self._spike_threshold_spin, 1)
        layout.addWidget(self._threshold_multiplier, 0)
        layout.addWidget(self._global_spike_threshold, 0)

    # PUBLIC METHODS ------------------------------------------------------------------

    def value(self) -> tuple[float, str]:
        """Return the value of the spike threshold."""
        return (
            self._spike_threshold_spin.value(),
            (
                GLOBAL_SPIKE_THRESHOLD
                if self._global_spike_threshold.isChecked()
                else MULTIPLIER
            ),
        )

    def setValue(self, value: tuple[float, str]) -> None:
        """Set the value of the spike threshold widget."""
        threshold, mode = value
        self._spike_threshold_spin.setValue(threshold)
        self._global_spike_threshold.setChecked(mode == GLOBAL_SPIKE_THRESHOLD)
        self._threshold_multiplier.setChecked(mode == MULTIPLIER)


class _BurstWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setToolTip(
            "Settings to control the detection of network bursts in population "
            "activity.\n\n"
            "• Burst Threshold:\n"
            "   Minimum percentage of ROIs that must be active simultaneously to "
            "detect a network burst.\n"
            "   Population activity above this threshold is considered burst "
            "activity.\n"
            "   Higher values (50-80%) detect only strong network-wide events.\n"
            "   Lower values (10-30%) capture weaker coordinated activity.\n\n"
            "• Burst Min Duration (frames):\n"
            "   Minimum duration (in frames) for a detected burst to be "
            "considered valid.\n"
            "   Filters out brief spikes that don't represent sustained "
            "network activity.\n"
            "   Higher values ensure only sustained bursts are detected.\n\n"
            "• Burst Gaussian Blur Sigma:\n"
            "   Gaussian smoothing applied to population activity before "
            "burst detection.\n"
            "   Reduces noise and connects nearby activity peaks into "
            "coherent bursts.\n"
            "   Higher values (2-5) provide more smoothing, merging closer events.\n"
            "   Lower values (0.5-1) preserve temporal precision but may "
            "fragment bursts.\n"
            "   Set to 0 to disable smoothing."
        )

        self._burst_threshold_lbl = QLabel("Burst Threshold (%):")
        self._burst_threshold_lbl.setSizePolicy(*FIXED)
        self._burst_threshold = QDoubleSpinBox(self)
        self._burst_threshold.setDecimals(2)
        self._burst_threshold.setRange(0.0, 100.0)
        self._burst_threshold.setSingleStep(1)
        self._burst_threshold.setValue(DEFAULT_BURST_THRESHOLD)

        self._burst_min_threshold_label = QLabel("Burst Min Duration (frames):")
        self._burst_min_threshold_label.setSizePolicy(*FIXED)
        self._burst_min_duration_frames = QSpinBox(self)
        self._burst_min_duration_frames.setRange(0, 100)
        self._burst_min_duration_frames.setSingleStep(1)
        self._burst_min_duration_frames.setValue(DEFAULT_MIN_BURST_DURATION)

        self._burst_blur_label = QLabel("Burst Gaussian Blur Sigma:")
        self._burst_blur_label.setSizePolicy(*FIXED)
        self._burst_blur_sigma = QDoubleSpinBox(self)
        self._burst_blur_sigma.setDecimals(2)
        self._burst_blur_sigma.setRange(0.0, 100.0)
        self._burst_blur_sigma.setSingleStep(0.5)
        self._burst_blur_sigma.setValue(DEFAULT_BURST_GAUSS_SIGMA)

        burst_layout = QGridLayout(self)
        burst_layout.setContentsMargins(0, 0, 0, 0)
        burst_layout.setSpacing(5)
        burst_layout.addWidget(self._burst_threshold_lbl, 0, 0)
        burst_layout.addWidget(self._burst_threshold, 0, 1)
        burst_layout.addWidget(self._burst_min_threshold_label, 1, 0)
        burst_layout.addWidget(self._burst_min_duration_frames, 1, 1)
        burst_layout.addWidget(self._burst_blur_label, 2, 0)
        burst_layout.addWidget(self._burst_blur_sigma, 2, 1)

    # PUBLIC METHODS ------------------------------------------------------------------

    def value(self) -> tuple[float, int, float]:
        """Return the burst detection parameters."""
        return (
            self._burst_threshold.value(),
            self._burst_min_duration_frames.value(),
            self._burst_blur_sigma.value(),
        )

    def setValue(self, value: tuple[float, int, float]) -> None:
        """Set the value of the burst widget."""
        threshold, duration, sigma = value
        self._burst_threshold.setValue(threshold)
        self._burst_min_duration_frames.setValue(duration)
        self._burst_blur_sigma.setValue(sigma)


class _SpikeWidget(QWidget):
    """Widget to select the spike detection settings."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # spikes threshold
        self._spike_threshold_wdg = _SpikeThresholdWidget(self)

        # burst detection settings
        self._burst_wdg = _BurstWidget(self)

        # spike synchrony settings
        self._spike_synchrony_wdg = QWidget(self)
        self._spike_synchrony_wdg.setToolTip(
            "Inferred Spike Synchrony Analysis Settings\n\n"
            "Max Lag Parameter:\n"
            "Controls the maximum temporal offset for cross-correlation analysis.\n\n"
            "What the value means:\n"
            "• Value = 5: Checks correlations within ±5 frames window\n"
            "• Algorithm slides one spike train over another, looking for "
            "best match within this range\n"
            "• Takes the MAXIMUM correlation found within the lag window\n"
            "• Larger values are more permissive, smaller values more strict\n\n"
            "Example with Max Lag = 5:\n"
            "ROI 1 spikes: [10, 25, 40]  ROI 2 spikes: [12, 24, 41]\n"
            "Algorithm finds high correlation at lag +2 and -1 frames\n"
            "Result: High synchrony score based on best alignment."
        )
        self._spikes_sync_cross_corr_lag = QLabel("Synchrony Lag (frames):")
        self._spikes_sync_cross_corr_lag.setSizePolicy(*FIXED)
        self._spikes_sync_cross_corr_max_lag = QSpinBox(self)
        self._spikes_sync_cross_corr_max_lag.setRange(0, 100)
        self._spikes_sync_cross_corr_max_lag.setSingleStep(1)
        self._spikes_sync_cross_corr_max_lag.setValue(5)
        spikes_sync_cross_corr_layout = QHBoxLayout(self._spike_synchrony_wdg)
        spikes_sync_cross_corr_layout.setContentsMargins(0, 0, 0, 0)
        spikes_sync_cross_corr_layout.setSpacing(DEFAULT_SPIKE_SYNCHRONY_MAX_LAG)
        spikes_sync_cross_corr_layout.addWidget(self._spikes_sync_cross_corr_lag)
        spikes_sync_cross_corr_layout.addWidget(self._spikes_sync_cross_corr_max_lag)

        # main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._spike_threshold_wdg)
        layout.addWidget(self._burst_wdg)
        layout.addWidget(self._spike_synchrony_wdg)

    # PUBLIC METHODS ------------------------------------------------------------------

    def set_labels_width(self, width: int) -> None:
        """Set the width of the labels."""
        self._spike_threshold_wdg._spike_threshold_lbl.setFixedWidth(width)
        self._burst_wdg._burst_threshold_lbl.setFixedWidth(width)
        self._burst_wdg._burst_min_threshold_label.setFixedWidth(width)
        self._burst_wdg._burst_blur_label.setFixedWidth(width)
        self._spikes_sync_cross_corr_lag.setFixedWidth(width)

    def value(self) -> SpikeData:
        """Get the current values of the widget."""
        spike_threshold, spike_threshold_mode = self._spike_threshold_wdg.value()
        burst_threshold, burst_min_duration, burst_blur_sigma = self._burst_wdg.value()
        synchrony_lag = self._spikes_sync_cross_corr_max_lag.value()

        return SpikeData(
            spike_threshold=spike_threshold,
            spike_threshold_mode=spike_threshold_mode,
            burst_threshold=burst_threshold,
            burst_min_duration=burst_min_duration,
            burst_blur_sigma=burst_blur_sigma,
            synchrony_lag=synchrony_lag,
        )

    def setValue(self, value: SpikeData) -> None:
        """Set the values of the widget."""
        tr = (value.spike_threshold, value.spike_threshold_mode)
        self._spike_threshold_wdg.setValue(tr)
        bst = (value.burst_threshold, value.burst_min_duration, value.burst_blur_sigma)
        self._burst_wdg.setValue(bst)
        self._spikes_sync_cross_corr_max_lag.setValue(value.synchrony_lag)

    def reset(self) -> None:
        """Reset the widget to default values."""
        self._spike_threshold_wdg.setValue((DEFAULT_SPIKE_THRESHOLD, MULTIPLIER))
        self._burst_wdg.setValue(
            (
                DEFAULT_BURST_THRESHOLD,
                DEFAULT_MIN_BURST_DURATION,
                DEFAULT_BURST_GAUSS_SIGMA,
            )
        )
        self._spikes_sync_cross_corr_max_lag.setValue(DEFAULT_SPIKE_SYNCHRONY_MAX_LAG)


class _FrameRateWidget(QWidget):
    """Widget to select the frame rate of the experiment."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setToolTip(
            "Set the frame rate (in frames per second or Hz) of the imaging "
            "experiment.\n\n"
            "This value is used to convert frame-based measurements into time-based "
            "units during analysis ONLY IF there are no metadata available.\n"
            "If the data contains metadata, the frame rate will be extracted for each "
            "position automatically."
        )

        self._frame_rate_lbl = QLabel("Frame Rate (fps):")
        self._frame_rate_lbl.setSizePolicy(*FIXED)

        self._frame_rate_spin = QDoubleSpinBox(self)
        self._frame_rate_spin.setDecimals(2)
        self._frame_rate_spin.setRange(0.01, 1000.0)
        self._frame_rate_spin.setSingleStep(0.5)
        self._frame_rate_spin.setValue(DEFAULT_FRAME_RATE)

        self._from_meta_btn = FromMetaButton(self, "Load From Metadata")
        self._from_meta_btn.setToolTip(
            "Try to load the frame rate from the image metadata (from exposure time "
            " and number of frames)."
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._frame_rate_lbl)
        layout.addWidget(self._frame_rate_spin, 1)
        layout.addWidget(self._from_meta_btn, 0)

    # PUBLIC METHODS ------------------------------------------------------------------

    def value(self) -> float:
        """Return the frame rate value."""
        return self._frame_rate_spin.value()  # type: ignore

    def setValue(self, value: float) -> None:
        """Set the frame rate value."""
        self._frame_rate_spin.setValue(value)

    def set_labels_width(self, width: int) -> None:
        """Set the width of the label."""
        self._frame_rate_lbl.setFixedWidth(width)

    def reset(self) -> None:
        """Reset the widget to default values."""
        self._frame_rate_spin.setValue(DEFAULT_FRAME_RATE)


class _RunAnalysisWidget(QWidget):
    """Widget to display the progress of the analysis."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # progress bar
        self._progress_bar = QProgressBar(self)
        self._progress_pos_label = QLabel()
        self._elapsed_time_label = QLabel("00:00:00")

        # buttons
        self._run_btn = QPushButton("Run")
        self._run_btn.setSizePolicy(*FIXED)
        self._run_btn.setIcon(icon(MDI6.play, color=GREEN))
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setSizePolicy(*FIXED)
        self._cancel_btn.setIcon(QIcon(icon(MDI6.stop, color=RED)))

        # threads selector
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
            "By default using CPU count - 2 to reserves 2 CPUs for the operating "
            "system and GUI responsiveness.\n"
            "If your system becomes unresponsive, consider reducing this number."
        )
        threads_lbl = QLabel("Threads:")
        threads_lbl.setSizePolicy(*FIXED)
        self._threads = QSpinBox()
        self._threads.setFixedWidth(60)
        self._threads.setRange(1, 100)
        self._threads.setValue(cpu_to_use)
        threads_layout = QHBoxLayout(threads_wdg)
        threads_layout.setContentsMargins(0, 0, 0, 0)
        threads_layout.setSpacing(5)
        threads_layout.addWidget(threads_lbl)
        threads_layout.addWidget(self._threads)

        # main layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)
        main_layout.addWidget(self._run_btn)
        main_layout.addWidget(self._cancel_btn)
        main_layout.addWidget(threads_wdg)
        main_layout.addWidget(self._progress_bar)
        main_layout.addWidget(self._progress_pos_label)
        main_layout.addWidget(self._elapsed_time_label)

    def progress_bar_maximum(self) -> int:
        """Return the maximum value of the progress bar."""
        return cast("int", self._progress_bar.maximum())

    def set_progress_bar_label(self, text: str) -> None:
        """Update the progress label with elapsed time."""
        self._progress_pos_label.setText(text)

    def set_progress_bar_range(self, minimum: int, maximum: int) -> None:
        """Set the range of the progress bar."""
        self._progress_bar.setRange(minimum, maximum)

    def reset_progress_bar(self) -> None:
        """Reset the progress bar and elapsed time label."""
        self._progress_bar.reset()
        self._progress_bar.setValue(0)
        self._progress_pos_label.setText("[0/0]")
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
        self._progress_pos_label.setText(f"[{value}/{self._progress_bar.maximum()}]")

    def reset(self) -> None:
        """Reset the widget to default values."""
        self.reset_progress_bar()
        self._threads.setValue(max((os.cpu_count() or 1) - 2, 1))


class _AnalysisGUI(QWidget):
    progress_bar_updated = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # MAIN WIDGET -----------------------------------------------------------------
        group_wdg = QGroupBox(self)
        group_layout = QVBoxLayout(group_wdg)
        group_layout.setContentsMargins(10, 10, 10, 10)
        group_layout.setSpacing(5)

        # ANALYSIS WIDGETS -----------------------------------------------------------
        self._plate_map_wdg = _PlateMapWidget(self)
        self._experiment_type_wdg = _ExperimentTypeWidget(self)
        self._neuropil_wdg = _NeuropilCorrectionWidget(self)
        self._trace_extraction_wdg = _TraceExtractionWidget(self)
        self._calcium_peaks_wdg = _CalciumPeaksWidget(self)
        self._spike_wdg = _SpikeWidget(self)
        # self._frame_rate_wdg = _FrameRateWidget(self)

        # SCROLL AREA WIDGET ---------------------------------------------------------
        analysis_scroll_area = QScrollArea()
        analysis_scroll_area.setWidgetResizable(True)
        analysis_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        analysis_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        # add cellpose and caiman widgets to scroll area
        group_layout.addWidget(create_divider_line("Plate Map"))
        group_layout.addWidget(self._plate_map_wdg)
        group_layout.addWidget(create_divider_line("Type of Experiment"))
        group_layout.addWidget(self._experiment_type_wdg)
        # group_layout.addWidget(create_divider_line("Frame Rate"))
        # group_layout.addWidget(self._frame_rate_wdg)
        group_layout.addWidget(create_divider_line("Neuropil Settings"))
        group_layout.addWidget(self._neuropil_wdg)
        group_layout.addWidget(create_divider_line("ΔF/F0 and Deconvolution"))
        group_layout.addWidget(self._trace_extraction_wdg)
        group_layout.addWidget(create_divider_line("Calcium Peaks"))
        group_layout.addWidget(self._calcium_peaks_wdg)
        group_layout.addWidget(create_divider_line("Spikes and Bursts"))
        group_layout.addWidget(self._spike_wdg)
        # group_layout.addWidget(self._frame_rate_wdg)
        group_layout.addStretch(1)
        analysis_scroll_area.setWidget(group_wdg)

        # BOTTOM WIDGET ---------------------------------------------------------------

        self._positions_wdg = _ChoosePositionsWidget(self)
        self._run_analysis_wdg = _RunAnalysisWidget(self)
        run_wdg = QGroupBox(self)
        run_layout = QVBoxLayout(run_wdg)
        run_layout.setContentsMargins(0, 0, 0, 0)
        run_layout.setSpacing(5)
        run_layout.addWidget(create_divider_line("Positions to Analyze"))
        run_layout.addWidget(self._positions_wdg)
        run_layout.addWidget(self._run_analysis_wdg)

        # MAIN LAYOUT -----------------------------------------------------------------
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(15)
        main_layout.addWidget(analysis_scroll_area)
        main_layout.addWidget(run_wdg)

        # STYLING ---------------------------------------------------------------------
        fix_width = self._calcium_peaks_wdg._peaks_prominence_lbl.sizeHint().width()
        self._plate_map_wdg.set_labels_width(fix_width)
        self._experiment_type_wdg.set_labels_width(fix_width)
        self._neuropil_wdg.set_labels_width(fix_width)
        self._trace_extraction_wdg.set_labels_width(fix_width)
        self._calcium_peaks_wdg.set_labels_width(fix_width)
        self._spike_wdg.set_labels_width(fix_width)
        self._positions_wdg.set_labels_width(fix_width)
        # self._frame_rate_wdg.set_labels_width(fix_width)

    # PROPERTIES ----------------------------------------------------------------------

    @property
    def run(self):  # noqa: ANN202
        """Signal emitted when the run button is clicked."""
        return self._run_analysis_wdg._run_btn.clicked

    @property
    def cancel(self):  # noqa: ANN202
        """Signal emitted when the cancel button is clicked."""
        return self._run_analysis_wdg._cancel_btn.clicked

    @property
    def from_metadata(self):  # noqa: ANN202
        """Signal emitted when the 'Load From Metadata' button is clicked."""
        return self._experiment_type_wdg._from_meta_btn.clicked

    # PUBLIC METHODS ------------------------------------------------------------------

    def value(self) -> AnalysisSettingsData:
        """Get the current values of the widget."""
        return AnalysisSettingsData(
            self._plate_map_wdg.value(),
            self._experiment_type_wdg.value(),
            self._trace_extraction_wdg.value(self._neuropil_wdg.value()),
            self._calcium_peaks_wdg.value(),
            self._spike_wdg.value(),
        )

    def setValue(self, value: AnalysisSettingsData) -> None:
        """Set the values of the widget."""
        if value.plate_map_data is not None:
            plate, genotype_map, treatment_map = value.plate_map_data
            self._plate_map_wdg.setValue(plate, genotype_map, treatment_map)
        if value.experiment_type_data is not None:
            self._experiment_type_wdg.setValue(value.experiment_type_data)
        if value.trace_extraction_data is not None:
            self._trace_extraction_wdg.setValue(value.trace_extraction_data)
            # Also set the neuropil widget from trace extraction data
            neuropil_data = NeuropilData(
                value.trace_extraction_data.neuropil_inner_radius,
                value.trace_extraction_data.neuropil_min_pixels,
                value.trace_extraction_data.neuropil_correction_factor,
            )
            self._neuropil_wdg.setValue(neuropil_data)
        if value.calcium_peaks_data is not None:
            self._calcium_peaks_wdg.setValue(value.calcium_peaks_data)
        if value.spikes_data is not None:
            self._spike_wdg.setValue(value.spikes_data)

    def positions(self) -> list[int]:
        """Get the positions to analyze."""
        return parse_lineedit_text(self._positions_wdg.value())

    def set_positions(self, positions: list[int]) -> None:
        """Set the positions to analyze."""
        self._positions_wdg.setValue(",".join(map(str, positions)))

    def enable(self, enable: bool) -> None:
        """Enable or disable the widget."""
        self._plate_map_wdg.setEnabled(enable)
        self._experiment_type_wdg.setEnabled(enable)
        # self._frame_rate_wdg.setEnabled(enable)
        self._neuropil_wdg.setEnabled(enable)
        self._trace_extraction_wdg.setEnabled(enable)
        self._calcium_peaks_wdg.setEnabled(enable)
        self._spike_wdg.setEnabled(enable)
        self._positions_wdg.setEnabled(enable)

    def reset(self) -> None:
        """Reset the widget to default values."""
        self._plate_map_wdg.clear()
        self._experiment_type_wdg.reset()
        # self._frame_rate_wdg.reset()
        self._neuropil_wdg.reset()
        self._trace_extraction_wdg.reset()
        self._calcium_peaks_wdg.reset()
        self._spike_wdg.reset()
        self._positions_wdg.setValue("")
        self._run_analysis_wdg.reset()

    def to_model_settings(self) -> AnalysisSettings:
        """Convert current GUI settings to AnalysisSettings model.

        Returns
        -------
        AnalysisSettings
            The AnalysisSettings model populated with current GUI values.
        """
        from datetime import datetime

        from cali.sqlmodel import AnalysisSettings

        settings = self.value()

        # Extract nested data with defaults
        trace_data = settings.trace_extraction_data
        peaks_data = settings.calcium_peaks_data
        spikes_data = settings.spikes_data
        exp_type_data = settings.experiment_type_data

        settings = AnalysisSettings(
            created_at=datetime.now(),
            threads=self._run_analysis_wdg._threads.value(),
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
            peaks_height_value=(
                peaks_data.peaks_height if peaks_data else DEFAULT_HEIGHT
            ),
            peaks_height_mode=(
                peaks_data.peaks_height_mode if peaks_data else MULTIPLIER
            ),
            peaks_distance=(
                peaks_data.peaks_distance if peaks_data else DEFAULT_PEAKS_DISTANCE
            ),
            peaks_prominence_multiplier=(
                peaks_data.peaks_prominence_multiplier if peaks_data else 1.0
            ),
            calcium_sync_jitter_window=(
                peaks_data.calcium_synchrony_jitter
                if peaks_data
                else DEFAULT_CALCIUM_SYNC_JITTER_WINDOW
            ),
            calcium_network_threshold=(
                peaks_data.calcium_network_threshold
                if peaks_data
                else DEFAULT_CALCIUM_NETWORK_THRESHOLD
            ),
            spike_threshold_value=(
                spikes_data.spike_threshold if spikes_data else DEFAULT_SPIKE_THRESHOLD
            ),
            spike_threshold_mode=(
                spikes_data.spike_threshold_mode if spikes_data else MULTIPLIER
            ),
            burst_threshold=(
                spikes_data.burst_threshold if spikes_data else DEFAULT_BURST_THRESHOLD
            ),
            burst_min_duration=(
                spikes_data.burst_min_duration
                if spikes_data
                else DEFAULT_MIN_BURST_DURATION
            ),
            burst_gaussian_sigma=(
                spikes_data.burst_blur_sigma
                if spikes_data
                else DEFAULT_BURST_GAUSS_SIGMA
            ),
            spikes_sync_cross_corr_lag=(
                spikes_data.synchrony_lag
                if spikes_data
                else DEFAULT_SPIKE_SYNCHRONY_MAX_LAG
            ),
            led_power_equation=(
                exp_type_data.led_power_equation if exp_type_data else None
            ),
            stimulation_mask_path=(
                exp_type_data.stimulation_area_path if exp_type_data else None
            ),
            # frame_rate=self._frame_rate_wdg.value(),
        )

        return settings

    def update_progress_label(self, elapsed_time: str) -> None:
        """Update the progress label with elapsed time."""
        self._run_analysis_wdg.set_time_label(elapsed_time)
