"""Analysis GUI for configuring analysis settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cali._constants import (
    DEFAULT_BURST_GAUSS_SIGMA,
    DEFAULT_BURST_THRESHOLD,
    DEFAULT_CALCIUM_NETWORK_THRESHOLD,
    DEFAULT_CALCIUM_SYNC_JITTER_WINDOW,
    DEFAULT_HEIGHT,
    DEFAULT_MIN_BURST_DURATION,
    DEFAULT_PEAKS_DISTANCE,
    DEFAULT_SPIKE_SYNCHRONY_MAX_LAG,
    DEFAULT_SPIKE_THRESHOLD,
    MULTIPLIER,
)

from ._extraction_gui import (
    CalciumPeaksData,
    SpikeData,
    _CalciumPeaksWidget,
    _SpikeWidget,
)
from ._util import create_divider_line

if TYPE_CHECKING:
    from cali.sqlmodel import AnalysisSettings

FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed


@dataclass(frozen=True)
class AnalysisSettingsData:
    """Data structure to hold the analysis settings."""

    calcium_peaks_data: CalciumPeaksData | None = None
    spikes_data: SpikeData | None = None


class _AnalysisGUI(QWidget):
    """GUI widget for configuring analysis settings."""

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

        # ANALYSIS WIDGETS -----------------------------------------------------------
        self._calcium_peaks_wdg = _CalciumPeaksWidget(self)
        self._spike_wdg = _SpikeWidget(self)

        # SCROLL AREA WIDGET ---------------------------------------------------------
        analysis_scroll_area = QScrollArea()
        analysis_scroll_area.setWidgetResizable(True)
        analysis_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        analysis_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        # add analysis widgets to scroll area
        group_layout.addWidget(create_divider_line("Calcium Peaks"))
        group_layout.addWidget(self._calcium_peaks_wdg)
        group_layout.addWidget(create_divider_line("Spikes and Bursts"))
        group_layout.addWidget(self._spike_wdg)
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
        fix_width = self._calcium_peaks_wdg._peaks_prominence_lbl.sizeHint().width()
        self._calcium_peaks_wdg.set_labels_width(fix_width)
        self._spike_wdg.set_labels_width(fix_width)
        threads_lbl.setFixedWidth(fix_width)

    # PUBLIC METHODS ------------------------------------------------------------------

    def value(self) -> AnalysisSettingsData:
        """Get the current values of the widget."""
        return AnalysisSettingsData(
            self._calcium_peaks_wdg.value(),
            self._spike_wdg.value(),
        )

    def setValue(self, value: AnalysisSettingsData) -> None:
        """Set the values of the widget."""
        if value.calcium_peaks_data is not None:
            self._calcium_peaks_wdg.setValue(value.calcium_peaks_data)
        if value.spikes_data is not None:
            self._spike_wdg.setValue(value.spikes_data)

    def enable(self, enable: bool) -> None:
        """Enable or disable the widget."""
        self._calcium_peaks_wdg.setEnabled(enable)
        self._spike_wdg.setEnabled(enable)

    def reset(self) -> None:
        """Reset the widget to default values."""
        self._calcium_peaks_wdg.reset()
        self._spike_wdg.reset()

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
        peaks_data = settings.calcium_peaks_data
        spikes_data = settings.spikes_data

        settings = AnalysisSettings(
            created_at=datetime.now(),
            threads=self._threads.value(),
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
        )

        return settings
