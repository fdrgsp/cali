"""Analysis GUI for configuring analysis settings."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from datetime import datetime
from typing import TYPE_CHECKING, cast

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QRadioButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from superqt import QCollapsible
from superqt.utils import signals_blocked

from cali._constants import (
    CALCIUM_DEN_DFF_CORRELATION,
    CALCIUM_DFF_CORRELATION,
    CLUSTER_LABELS,
    DEFAULT_BURST_GAUSS_SIGMA,
    DEFAULT_BURST_THRESHOLD,
    DEFAULT_CALCIUM_BURST_THRESHOLD,
    DEFAULT_CCG_N_SHUFFLES,
    DEFAULT_CLUSTER_MAX_K,
    DEFAULT_CLUSTER_METHOD,
    DEFAULT_CLUSTER_N_CLUSTERS,
    DEFAULT_ENABLE_RISING_EDGE_ANALYSIS,
    DEFAULT_FRAME_RATE,
    DEFAULT_HEIGHT,
    DEFAULT_MIN_BURST_DURATION,
    DEFAULT_PEAKS_DISTANCE,
    DEFAULT_SPIKE_SYNC_JITTER_WINDOW,
    DEFAULT_SPIKE_SYNCHRONY_MAX_LAG,
    DEFAULT_SPIKE_THRESHOLD,
    EVOKED,
    GLOBAL_HEIGHT,
    GLOBAL_SPIKE_THRESHOLD,
    INFERRED_SPIKES_CCG_ZSCORE,
    INFERRED_SPIKES_CCG_ZSCORE_RISING_EDGES,
    INFERRED_SPIKES_CROSS_CORRELATION,
    INFERRED_SPIKES_CROSS_CORRELATION_LAGS,
    INFERRED_SPIKES_CROSS_CORRELATION_LAGS_RISING_EDGES,
    INFERRED_SPIKES_CROSS_CORRELATION_RISING_EDGES,
    INFERRED_SPIKES_SYNCHRONY,
    INFERRED_SPIKES_SYNCHRONY_RISING_EDGES,
    INFERRED_SPIKES_THRESHOLDED_BINARY,
    MULTI_WELL_AGGREGATED_DATA,
    MULTIPLIER,
    SPONTANEOUS,
    CorrelationDataType,
)
from cali.sqlmodel import SpikeAnalysisSettings
from cali.sqlmodel._spike_settings import (
    CASCADE_AP_THRESHOLD,
    SpikeMethod,
    canonical_spike_methods,
)

from ._extraction_gui import FromMetaButton
from ._settings_tabs import _SettingsTabs, guidance, settings_section
from ._util import _BrowseWidget, _ExportGroup

if TYPE_CHECKING:
    from cali.sqlmodel import AnalysisSettings


FIXED = QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed


@dataclass(frozen=True)
class AnalysisSettingsData:
    """Data structure to hold the analysis settings."""

    enable_calcium: bool = True
    enable_spikes: bool = True
    calcium_peaks_data: CalciumPeaksData | None = None
    spikes_data: SpikeData | None = None
    spike_settings: tuple[SpikeData, ...] | None = None
    experiment_type_data: ExperimentTypeData | None = None
    frame_rate: float = DEFAULT_FRAME_RATE
    threads: int = max((os.cpu_count() or 1) - 2, 1)
    n_processes: int = max((os.cpu_count() or 1) - 2, 1)
    export_options: dict[str, tuple[bool, int, int]] | None = None
    export_enabled: bool = False


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
class CalciumPeaksData:
    """Data structure to hold the calcium peaks settings."""

    peaks_height: float = DEFAULT_HEIGHT
    peaks_height_mode: str = MULTIPLIER
    peaks_distance: float = DEFAULT_PEAKS_DISTANCE  # milliseconds
    peaks_prominence_multiplier: float = 2.0
    burst_threshold: float = DEFAULT_CALCIUM_BURST_THRESHOLD
    burst_min_duration: float = DEFAULT_MIN_BURST_DURATION  # milliseconds
    burst_blur_sigma: float = DEFAULT_BURST_GAUSS_SIGMA  # milliseconds
    cluster_n_clusters: int = DEFAULT_CLUSTER_N_CLUSTERS
    cluster_max_k: int = DEFAULT_CLUSTER_MAX_K


@dataclass(frozen=True)
class SpikeData:
    """Data structure to hold the spikes settings."""

    spike_threshold: float | None = DEFAULT_SPIKE_THRESHOLD
    spike_threshold_mode: str = MULTIPLIER
    burst_threshold: float = DEFAULT_BURST_THRESHOLD
    burst_min_duration: float = DEFAULT_MIN_BURST_DURATION  # milliseconds
    burst_blur_sigma: float = DEFAULT_BURST_GAUSS_SIGMA  # milliseconds
    synchrony_lag: float = DEFAULT_SPIKE_SYNCHRONY_MAX_LAG  # milliseconds
    synchrony_jitter: float = DEFAULT_SPIKE_SYNC_JITTER_WINDOW  # milliseconds
    ccg_n_shuffles: int = DEFAULT_CCG_N_SHUFFLES
    enable_rising_edge_analysis: bool = DEFAULT_ENABLE_RISING_EDGE_ANALYSIS
    method: SpikeMethod = "oasis"
    cascade_ap_threshold_fraction: float | None = None

    def to_model(self) -> SpikeAnalysisSettings:
        return SpikeAnalysisSettings(
            method=self.method,
            threshold_value=self.spike_threshold,
            threshold_mode=self.spike_threshold_mode,
            cascade_ap_threshold_fraction=self.cascade_ap_threshold_fraction,
            burst_threshold=self.burst_threshold,
            burst_min_duration=self.burst_min_duration,
            burst_gaussian_sigma=self.burst_blur_sigma,
            spikes_sync_cross_corr_lag=self.synchrony_lag,
            spikes_sync_jitter_window=self.synchrony_jitter,
            ccg_n_shuffles=self.ccg_n_shuffles,
            enable_rising_edge_analysis=self.enable_rising_edge_analysis,
        )

    @classmethod
    def from_model(cls, settings: SpikeAnalysisSettings) -> SpikeData:
        return cls(
            method=canonical_spike_methods((settings.method,))[0],
            spike_threshold=settings.threshold_value,
            spike_threshold_mode=settings.threshold_mode,
            cascade_ap_threshold_fraction=settings.cascade_ap_threshold_fraction,
            burst_threshold=settings.burst_threshold,
            burst_min_duration=settings.burst_min_duration,
            burst_blur_sigma=settings.burst_gaussian_sigma,
            synchrony_lag=settings.spikes_sync_cross_corr_lag,
            synchrony_jitter=settings.spikes_sync_jitter_window,
            ccg_n_shuffles=settings.ccg_n_shuffles,
            enable_rising_edge_analysis=settings.enable_rising_edge_analysis,
        )


class _AnalysisGUI(QWidget):
    """GUI widget for configuring analysis settings."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

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
        threads_lbl = QLabel("Number of Threads:", threads_wdg)
        threads_lbl.setSizePolicy(*FIXED)
        self._threads = QSpinBox(threads_wdg)
        self._threads.setRange(1, 100)
        self._threads.setValue(cpu_to_use)
        threads_layout = QHBoxLayout(threads_wdg)
        threads_layout.setContentsMargins(0, 0, 0, 0)
        threads_layout.setSpacing(5)
        threads_layout.addWidget(threads_lbl)
        threads_layout.addWidget(self._threads)

        # N_PROCESSES WIDGET ---------------------------------------------------------
        self._n_processes_wdg = QWidget()
        self._n_processes_wdg.setToolTip(
            "Number of worker processes for parallel CCG computation.\n\n"
            "By default, the value is set to the number of CPUs - 2 "
            f"(in your system: {cpu_to_use}).\n\n"
            "CCG (Cross-Correlogram) computation is the slowest part of FOV analysis.\n"
            "It uses multiprocessing to parallelize across ROI pairs.\n\n"
            f"• Your system: {cpu_to_use} processes (auto)\n"
            "• Higher values: Faster but more memory usage\n"
            "• Lower values: Slower but less resource intensive\n\n"
            "Note: This is separate from 'threads' which controls ROI extraction."
        )
        n_processes_lbl = QLabel("CCG Worker Processes:", self._n_processes_wdg)
        n_processes_lbl.setSizePolicy(*FIXED)
        self._n_processes = QSpinBox(self._n_processes_wdg)
        self._n_processes.setRange(1, 100)
        self._n_processes.setValue(cpu_to_use)
        n_processes_layout = QHBoxLayout(self._n_processes_wdg)
        n_processes_layout.setContentsMargins(0, 0, 0, 10)
        n_processes_layout.setSpacing(5)
        n_processes_layout.addWidget(n_processes_lbl)
        n_processes_layout.addWidget(self._n_processes)

        # ANALYSIS WIDGETS -----------------------------------------------------------
        self._experiment_type_wdg = _ExperimentTypeWidget(self)
        self._calcium_peaks_wdg = _CalciumPeaksWidget(self)
        self._spike_wdg = _SpikeWidget(self)
        self._cascade_spike_wdg = _SpikeWidget(self, method="cascade")
        self._spike_tabs = QTabWidget(self)
        self._method_pages = {
            "oasis": settings_section(
                "OASIS activity",
                "Thresholds apply to relative OASIS spike amplitudes. "
                "A noise multiplier adapts to each ROI; a global threshold uses "
                "the same "
                "value for every ROI.",
                self._spike_wdg,
            ),
            "cascade": settings_section(
                "CASCADE activity",
                "Thresholds apply to expected spikes per frame. "
                "Model AP Threshold uses the selected model's response to one action "
                "potential; a global threshold requires an explicit "
                "spikes/frame value.",
                self._cascade_spike_wdg,
            ),
        }
        self._spike_methods: tuple[SpikeMethod, ...] = ("oasis",)
        self.set_spike_methods(("oasis",))
        self._metadata_wdg = _MetadataWidget(self)

        # ENABLE CHECKBOXES ----------------------------------------------------------
        self._enable_calcium_cb = QCheckBox()
        self._enable_calcium_cb.setChecked(True)
        self._enable_calcium_cb.clicked.connect(self._on_enable_changed)
        self._enable_calcium_cb.setToolTip(
            "Compute calcium events, bursts and ROI grouping."
        )

        self._enable_spikes_cb = QCheckBox()
        self._enable_spikes_cb.setChecked(True)
        self._enable_spikes_cb.clicked.connect(self._on_enable_changed)
        self._enable_spikes_cb.setToolTip(
            "Analyze all selected or stored spike outputs."
        )

        self._export_group = _ExportGroup()
        # Multi-Well Aggregated Data
        self._export_group.add_section_label("Multi-Well Aggregated Data", 0, 0)
        self._export_group.add_option(MULTI_WELL_AGGREGATED_DATA, 1, 0, checked=True)
        # Calcium correlations
        self._export_group.add_section_label("Calcium Correlations", 2, 0)
        self._export_group.add_option(CALCIUM_DFF_CORRELATION, 3, 0, checked=False)
        self._export_group.add_option(CALCIUM_DEN_DFF_CORRELATION, 4, 0, checked=False)
        # Cluster Analysis
        self._export_group.add_section_label("Cluster Analysis", 5, 0)
        self._export_group.add_option(CLUSTER_LABELS, 6, 0, checked=False)
        # Inferred Spikes - Thresholded Binary
        self._export_group.add_section_label("Inferred Spikes (Thresholded)", 7, 0)
        self._export_group.add_option(
            INFERRED_SPIKES_THRESHOLDED_BINARY, 8, 0, checked=False
        )
        self._export_group.add_option(INFERRED_SPIKES_SYNCHRONY, 9, 0, checked=False)
        self._export_group.add_option(
            INFERRED_SPIKES_CROSS_CORRELATION, 10, 0, checked=False
        )
        self._export_group.add_option(
            INFERRED_SPIKES_CROSS_CORRELATION_LAGS, 11, 0, checked=False
        )
        self._export_group.add_option(INFERRED_SPIKES_CCG_ZSCORE, 12, 0, checked=False)
        # Inferred Spikes - Thresholded Rising Edges
        self._export_group.add_section_label("Inferred Spikes (Rising Edges)", 13, 0)
        self._export_group.add_option(
            INFERRED_SPIKES_SYNCHRONY_RISING_EDGES, 14, 0, checked=False
        )
        self._export_group.add_option(
            INFERRED_SPIKES_CROSS_CORRELATION_RISING_EDGES, 15, 0, checked=False
        )
        self._export_group.add_option(
            INFERRED_SPIKES_CROSS_CORRELATION_LAGS_RISING_EDGES, 16, 0, checked=False
        )
        self._export_group.add_option(
            INFERRED_SPIKES_CCG_ZSCORE_RISING_EDGES, 17, 0, checked=False
        )
        self._export_group.add_stretch("horizontal")

        self._settings_tabs = _SettingsTabs(self)
        self._settings_tabs.add_page(
            "Experiment",
            "Choose spontaneous activity or a stimulated experiment. The acquisition "
            "rate converts time-based thresholds into frames.",
            self._experiment_type_wdg,
            self._metadata_wdg,
        )
        self._settings_tabs.add_page(
            "Calcium events",
            "Check this tab to detect calcium events, network bursts and groups of "
            "ROIs with similar activity. At least one analysis tab must stay checked.",
            self._calcium_peaks_wdg,
            checkbox=self._enable_calcium_cb,
        )
        self._settings_tabs.add_page(
            "Spike activity",
            "Check this tab to analyze every selected spike output. Each method keeps "
            "its own thresholds and synchrony settings. Choose outputs in Extraction; "
            "analysis-only uses the outputs already stored in the selected extraction.",
            self._spike_tabs,
            checkbox=self._enable_spikes_cb,
        )
        self._settings_tabs.add_page(
            "Processing and export",
            "Analysis results are saved in the database. Choose optional CSV exports "
            "here; exporting does not change the analysis.",
            settings_section(
                "Processing",
                "Threads process ROIs. CCG worker processes compare "
                "spike timing between ROI pairs; more workers use more memory.",
                threads_wdg,
                self._n_processes_wdg,
            ),
            self._export_group,
        )
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self._settings_tabs)

        # STYLING ---------------------------------------------------------------------
        fix_width = self._calcium_peaks_wdg._peaks_prominence_lbl.sizeHint().width()
        self._experiment_type_wdg.set_labels_width(fix_width)
        self._calcium_peaks_wdg.set_labels_width(fix_width)
        self._spike_wdg.set_labels_width(fix_width)
        self._cascade_spike_wdg.set_labels_width(fix_width)
        self._metadata_wdg.set_labels_width(fix_width)
        threads_lbl.setFixedWidth(fix_width)
        n_processes_lbl.setFixedWidth(fix_width)

    # PUBLIC METHODS ------------------------------------------------------------------

    @property
    def from_metadata(self) -> None:
        """Signal emitted when the 'Load From Metadata' button is clicked."""
        return self._experiment_type_wdg.from_metadata

    @property
    def from_metadata_frame_rate(self) -> None:
        """Signal emitted when the metadata 'Load From Metadata' button is clicked."""
        return self._metadata_wdg.from_metadata

    def value(self) -> AnalysisSettingsData:
        """Get the current values of the widget."""
        return AnalysisSettingsData(
            enable_calcium=self._enable_calcium_cb.isChecked(),
            enable_spikes=self._enable_spikes_cb.isChecked(),
            calcium_peaks_data=self._calcium_peaks_wdg.value(),
            spikes_data=self._spike_wdg.value(),
            spike_settings=tuple(
                self._method_widget(method).value() for method in self._spike_methods
            ),
            experiment_type_data=self._experiment_type_wdg.value(),
            frame_rate=self._metadata_wdg.value(),
            threads=self._threads.value(),
            n_processes=self._n_processes.value(),
            export_options=self._export_group.value(),
            export_enabled=self._export_group.isChecked(),
        )

    def setValue(self, value: AnalysisSettingsData) -> None:
        """Set the values of the widget."""
        self._enable_calcium_cb.setChecked(value.enable_calcium)
        self._enable_spikes_cb.setChecked(value.enable_spikes)
        self._calcium_peaks_wdg.setEnabled(value.enable_calcium)
        self._spike_wdg.setEnabled(value.enable_spikes)
        self._spike_tabs.setEnabled(value.enable_spikes)
        self._n_processes_wdg.setVisible(value.enable_spikes)
        if value.calcium_peaks_data is not None:
            self._calcium_peaks_wdg.setValue(value.calcium_peaks_data)
        if value.spikes_data is not None:
            self._spike_wdg.setValue(value.spikes_data)
        if value.spike_settings is not None:
            methods = canonical_spike_methods(
                [child.method for child in value.spike_settings]
            )
            if len(methods) != len(value.spike_settings):
                raise ValueError("Each spike analysis method must occur exactly once.")
            for child in value.spike_settings:
                child.to_model()  # Validate method/mode pairs before updating controls.
            for child in value.spike_settings:
                self._method_widget(child.method).setValue(child)
            self.set_spike_methods(methods)
        if value.experiment_type_data is not None:
            self._experiment_type_wdg.setValue(value.experiment_type_data)
        self._metadata_wdg.setValue(value.frame_rate)
        self._threads.setValue(value.threads)
        self._n_processes.setValue(value.n_processes)
        if value.export_options is not None:
            self._export_group.setValue(value.export_options)
        if value.export_enabled is not None:
            self._export_group.setChecked(value.export_enabled)

    def reset(self) -> None:
        """Reset the widget to default values."""
        self._enable_calcium_cb.setChecked(True)
        self._enable_spikes_cb.setChecked(True)
        self._calcium_peaks_wdg.setEnabled(True)
        self._spike_wdg.setEnabled(True)
        self._spike_tabs.setEnabled(True)
        self._n_processes_wdg.setVisible(True)
        self._experiment_type_wdg.reset()
        self._calcium_peaks_wdg.reset()
        self._spike_wdg.reset()
        self._cascade_spike_wdg.reset()
        self._metadata_wdg.reset()
        self._threads.setValue(max((os.cpu_count() or 1) - 2, 1))
        self._n_processes.setValue(max((os.cpu_count() or 1) - 2, 1))

    def get_export_options(self) -> dict[CorrelationDataType, bool] | None:
        """Return export options selected as dict[CorrelationDataType, bool]."""
        if not self._export_group.isChecked():
            return None
        return cast(
            "dict[CorrelationDataType, bool]", self._export_group.get_export_options()
        )

    # PRIVATE METHODS -----------------------------------------------------------------

    def _on_enable_changed(self) -> None:
        """Handle enable/disable checkbox changes with at-least-one validation."""
        calcium_on = self._enable_calcium_cb.isChecked()
        spikes_on = self._enable_spikes_cb.isChecked()

        # Enforce at least one must be checked.
        if not calcium_on and not spikes_on:
            sender = self.sender()
            if isinstance(sender, QCheckBox):
                sender.setChecked(True)
            calcium_on = self._enable_calcium_cb.isChecked()
            spikes_on = self._enable_spikes_cb.isChecked()

        self._calcium_peaks_wdg.setEnabled(calcium_on)
        self._spike_wdg.setEnabled(spikes_on)
        self._spike_tabs.setEnabled(spikes_on)
        self._n_processes_wdg.setVisible(spikes_on)

    def _method_widget(self, method: SpikeMethod) -> _SpikeWidget:
        return self._spike_wdg if method == "oasis" else self._cascade_spike_wdg

    def set_spike_methods(self, methods: tuple[SpikeMethod, ...]) -> None:
        """Select tabs without discarding either method's independent settings."""
        self._spike_methods = canonical_spike_methods(methods)
        self._spike_tabs.clear()
        for method in ("cascade", "oasis"):
            if method in self._spike_methods:
                self._spike_tabs.addTab(self._method_pages[method], method.upper())

    def to_model_settings(
        self, spike_methods: tuple[SpikeMethod, ...] | None = None
    ) -> AnalysisSettings:
        """Convert current GUI settings to AnalysisSettings model.

        Returns
        -------
        AnalysisSettings
            The AnalysisSettings model populated with current GUI values.
        """
        from cali.sqlmodel import AnalysisSettings

        methods = canonical_spike_methods(
            self._spike_methods if spike_methods is None else spike_methods
        )
        if spike_methods is not None:
            self.set_spike_methods(methods)
        settings = self.value()

        # Extract nested data with defaults
        peaks_data = settings.calcium_peaks_data
        experiment_type_data = settings.experiment_type_data

        return AnalysisSettings(
            created_at=datetime.now(),
            enable_calcium=settings.enable_calcium,
            enable_spikes=settings.enable_spikes,
            threads=self._threads.value(),
            n_processes=self._n_processes.value(),
            frame_rate=settings.frame_rate,
            spike_settings=[
                self._method_widget(method).value().to_model() for method in methods
            ],
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
            calcium_burst_threshold=(
                peaks_data.burst_threshold
                if peaks_data
                else DEFAULT_CALCIUM_BURST_THRESHOLD
            ),
            calcium_burst_min_duration=(
                peaks_data.burst_min_duration
                if peaks_data
                else DEFAULT_MIN_BURST_DURATION
            ),
            calcium_burst_gaussian_sigma=(
                peaks_data.burst_blur_sigma if peaks_data else DEFAULT_BURST_GAUSS_SIGMA
            ),
            cluster_method=DEFAULT_CLUSTER_METHOD,
            cluster_n_clusters=(
                peaks_data.cluster_n_clusters
                if peaks_data
                else DEFAULT_CLUSTER_N_CLUSTERS
            ),
            cluster_max_k=(
                peaks_data.cluster_max_k if peaks_data else DEFAULT_CLUSTER_MAX_K
            ),
            experiment_type=(
                experiment_type_data.experiment_type
                if experiment_type_data and experiment_type_data.experiment_type
                else SPONTANEOUS
            ),
            led_power_equation=(
                experiment_type_data.led_power_equation
                if experiment_type_data
                else None
            ),
            led_pulse_duration=(
                experiment_type_data.led_pulse_duration
                if experiment_type_data
                else None
            ),
            led_pulse_powers=(
                experiment_type_data.led_pulse_powers if experiment_type_data else None
            ),
            led_pulse_on_frames=(
                experiment_type_data.led_pulse_on_frames
                if experiment_type_data
                else None
            ),
            stimulation_mask_path=(
                experiment_type_data.stimulation_area_path
                if experiment_type_data
                else None
            ),
        )


class _ExperimentTypeWidget(QWidget):
    """Widget to select the type of experiment.

    Allows selection between spontaneous or evoked activity and related settings.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # experiment type combo
        self._experiment_type_lbl = QLabel("Experiment Type:", self)
        self._experiment_type_lbl.setSizePolicy(*FIXED)
        self._experiment_type_combo = QComboBox(self)
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
                "Select the path to the image of the stimulated area.\\n"
                "The image should be a binary mask.\\n"
                "Accepted formats: .tif, .tiff."
            ),
            file_filter="TIFF Files (*.tif *.tiff);;All Files (*)",
            is_dir=False,
        )
        self._stimulation_area_path_dialog.hide()

        # LED power equation widget
        self._led_power_eq = QWidget(self)
        self._led_power_eq.setToolTip(
            "Insert an equation to convert the LED power to mW.\\n"
            "Supported formats:\\n"
            "• Linear: y = m*x + q (e.g. y = 2*x + 3)\\n"
            "• Quadratic: y = a*x^2 + b*x + c (e.g. y = 0.5*x^2 + 2*x + 1)\\n"
            "• Exponential: y = a*exp(b*x) + c (e.g. y = 2*exp(0.1*x) + 1)\\n"
            "• Power: y = a*x^b + c (e.g. y = 2*x^0.5 + 1)\\n"
            "• Logarithmic: y = a*log(x) + b (e.g. y = 2*log(x) + 1)\\n"
            "Leave empty to use values from the acquisition metadata (%)."
        )
        self._led_eq_lbl = QLabel("LED Power Equation:", self._led_power_eq)
        self._led_eq_lbl.setSizePolicy(*FIXED)
        self._led_power_equation_le = QLineEdit(self._led_power_eq)
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
        self._led_pulse_duration_lbl = QLabel(
            "LED Pulse Duration:", self._led_pulse_duration_wdg
        )
        self._led_pulse_duration_lbl.setSizePolicy(*FIXED)
        self._led_pulse_duration_spin = QDoubleSpinBox(self._led_pulse_duration_wdg)
        self._led_pulse_duration_spin.setSuffix(" ms")
        self._led_pulse_duration_spin.setRange(0.0, 10000.0)
        led_pulse_layout = QHBoxLayout(self._led_pulse_duration_wdg)
        led_pulse_layout.setContentsMargins(0, 0, 0, 0)
        led_pulse_layout.setSpacing(5)
        led_pulse_layout.addWidget(self._led_pulse_duration_lbl)
        led_pulse_layout.addWidget(self._led_pulse_duration_spin)
        self._led_pulse_duration_wdg.hide()

        # LED pulse powers widget
        self._led_powers_wdg = QWidget(self)
        self._led_powers_wdg.setToolTip(
            "List of LED pulse powers corresponding to each stimulation frame.\\n"
            "Values should be in percentage (%), separated by commas "
            "(e.g. 20, 40, 60, 80).\\n"
            "The length of this list should match the length of the 'Stimulation "
            "Frames' list."
        )
        self._led_powers_lbl = QLabel("LED Pulse Powers (%):", self._led_powers_wdg)
        self._led_powers_lbl.setSizePolicy(*FIXED)
        self._led_powers_le = QLineEdit(self._led_powers_wdg)
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
            "List of frames where the LED was ON during the experiment.\\n"
            "Values should be integers separated by commas (e.g. 1, 5, 10, 15).\\n"
            "The length of this list should match the length of the 'LED Pulse Powers' "
            "list."
        )
        self.led_pulse_on_frames_lbl = QLabel(
            "Stimulation Frames:", self._led_pulse_on_frames_wdg
        )
        self.led_pulse_on_frames_lbl.setSizePolicy(*FIXED)
        self._led_pulse_on_frames_le = QLineEdit(self._led_pulse_on_frames_wdg)
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

    @property
    def from_metadata(self) -> None:
        """Signal emitted when the 'Load From Metadata' button is clicked."""
        return self._from_meta_btn.clicked  # type: ignore

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
        if (exp_type := self._experiment_type_combo.currentText()) == SPONTANEOUS:
            return ExperimentTypeData(exp_type)
        return ExperimentTypeData(
            exp_type,
            self._led_power_equation_le.text(),
            self._led_pulse_duration_spin.value(),
            self._parse_float_list(self._led_powers_le.text()),
            self._parse_int_list(self._led_pulse_on_frames_le.text()),
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
            with signals_blocked(self._experiment_type_combo):
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

    def _parse_int_list(self, text: str) -> list[int]:
        """Parse a comma-separated string into a list of integers."""
        parsed: list[int] = []
        for val in text.split(","):
            val = val.strip()
            try:
                parsed.append(int(float(val)))  # float() first handles "3.0" input
            except ValueError:
                continue
        return parsed

    def _parse_float_list(self, text: str) -> list[float]:
        """Parse a comma-separated string into a list of floats."""
        parsed: list[float] = []
        for val in text.split(","):
            val = val.strip()
            try:
                parsed.append(float(val))
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


class _PeaksHeightWidget(QWidget):
    """Widget to select the peaks height multiplier."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setToolTip(
            "Peak height threshold for detecting calcium transients in denoised "
            "ΔF/F0 traces using scipy.signal.find_peaks.\n\n"
            "Two modes:\n"
            "• Global Minimum: Same absolute threshold applied to ALL ROIs across "
            "ALL FOVs. Peaks below this value are rejected everywhere.\n\n"
            "• Noise Multiplier: Adaptive threshold computed individually for EACH "
            "ROI in EACH FOV.\n"
            "  Threshold = noise_level * multiplier, where noise_level is estimated "
            "  during OASIS deconvolution: the noise standard deviation (sn)\n"
            "  is estimated independently for each ROI based on the high-frequency "
            "components of the raw fluorescence trace using an autoregressive (AR) "
            "noise model.\n\n"
            "For example, a multiplier of 3.0 detects events that exceed three times "
            "the estimated noise level."
        )

        self._peaks_height_lbl = QLabel("Minimum Peaks Height:")
        self._peaks_height_lbl.setSizePolicy(*FIXED)

        self._peaks_height_spin = QDoubleSpinBox(self)
        self._peaks_height_spin.setDecimals(4)
        self._peaks_height_spin.setRange(0.0, 100000.0)
        self._peaks_height_spin.setSingleStep(0.01)
        self._peaks_height_spin.setValue(DEFAULT_HEIGHT)

        self._global_peaks_height = QRadioButton("Global threshold")

        self._height_multiplier = QRadioButton("Noise multiplier")
        self._height_multiplier.setChecked(True)

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._peaks_height_lbl, 0, 0)
        layout.addWidget(self._peaks_height_spin, 0, 1)
        layout.addWidget(self._height_multiplier, 1, 0)
        layout.addWidget(self._global_peaks_height, 1, 1)
        layout.setColumnStretch(1, 1)

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
            "Minimum distance between peaks in milliseconds.\n"
            "This prevents detecting multiple peaks from the same calcium event.\n\n"
            "Example: At 10 fps (100ms per frame), setting 200ms ensures peaks are\n"
            "at least 2 frames apart (200ms ÷ 100ms = 2 frames).\n\n"
            "• Higher values: More conservative, fewer detected peaks\n"
            "• Lower values: More sensitive, may detect noise or incomplete decay\n"
            "• Typical range: 100-1000ms depending on calcium indicator dynamics."
        )
        self._peaks_distance_lbl = QLabel(
            "Minimum Peaks Distance:", self._peaks_distance_wdg
        )
        self._peaks_distance_lbl.setSizePolicy(*FIXED)
        self._peaks_distance_spin = QDoubleSpinBox(self._peaks_distance_wdg)
        self._peaks_distance_spin.setSuffix(" ms")
        self._peaks_distance_spin.setDecimals(2)
        self._peaks_distance_spin.setRange(1.0, 10000.0)
        self._peaks_distance_spin.setSingleStep(10.0)
        self._peaks_distance_spin.setValue(200.0)  # 2 frames at 10fps
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
        self._peaks_prominence_lbl = QLabel(
            "Peaks Prominence Multiplier:", self._peaks_prominence_wdg
        )
        self._peaks_prominence_lbl.setSizePolicy(*FIXED)
        self._peaks_prominence_multiplier_spin = QDoubleSpinBox(
            self._peaks_prominence_wdg
        )
        self._peaks_prominence_multiplier_spin.setDecimals(4)
        self._peaks_prominence_multiplier_spin.setRange(0, 100000.0)
        self._peaks_prominence_multiplier_spin.setSingleStep(0.01)
        self._peaks_prominence_multiplier_spin.setValue(1)
        peaks_prominence_layout = QHBoxLayout(self._peaks_prominence_wdg)
        peaks_prominence_layout.setContentsMargins(0, 0, 0, 0)
        peaks_prominence_layout.setSpacing(5)
        peaks_prominence_layout.addWidget(self._peaks_prominence_lbl)
        peaks_prominence_layout.addWidget(self._peaks_prominence_multiplier_spin)

        # burst widget
        self._burst_wdg = _BurstWidget(self)

        # Cluster analysis - number of clusters
        self._n_clusters_wdg = QWidget(self)
        self._n_clusters_wdg.setToolTip(
            "Number of clusters for Hierarchical (average/UPGMA linkage) clustering on "
            "the pairwise denoised ΔF/F correlation matrix.\n\n"
            "• 0 (Auto): Automatically detect the optimal K by scanning K = 2 … Max K\n"
            "  and selecting the K with the highest average silhouette score.\n"
            "• Positive integer: Force exactly that many clusters, skipping the "
            "scan.\n\n"
            "When a fixed number is set, 'Auto-detect Max K' is disabled."
        )
        self._n_clusters_lbl = QLabel("Cluster N Clusters:", self._n_clusters_wdg)
        self._n_clusters_lbl.setSizePolicy(*FIXED)
        self._n_clusters_spin = QSpinBox(self._n_clusters_wdg)
        self._n_clusters_spin.setRange(0, 50)
        self._n_clusters_spin.setValue(DEFAULT_CLUSTER_N_CLUSTERS)
        self._n_clusters_spin.setSpecialValueText("Auto")
        n_clusters_layout = QHBoxLayout(self._n_clusters_wdg)
        n_clusters_layout.setContentsMargins(0, 0, 0, 0)
        n_clusters_layout.setSpacing(5)
        n_clusters_layout.addWidget(self._n_clusters_lbl)
        n_clusters_layout.addWidget(self._n_clusters_spin)

        # Cluster analysis - auto-detect max k
        self._max_k_wdg = QWidget(self)
        self._max_k_wdg.setToolTip(
            "Upper bound of the silhouette-score scan when 'Cluster N Clusters' is "
            "0 (Auto).\n\n"
            "The algorithm evaluates every K from 2 up to this value and picks the one "
            "with the highest score. Increasing this allows finding more fine-grained "
            "structure at the cost of longer computation.\n\n"
            "Has no effect when a fixed cluster count is specified."
        )
        self._max_k_lbl = QLabel("Auto-detect Max K:", self._max_k_wdg)
        self._max_k_lbl.setSizePolicy(*FIXED)
        self._max_k_spin = QSpinBox(self._max_k_wdg)
        self._max_k_spin.setRange(2, 50)
        self._max_k_spin.setValue(DEFAULT_CLUSTER_MAX_K)
        max_k_layout = QHBoxLayout(self._max_k_wdg)
        max_k_layout.setContentsMargins(0, 0, 0, 0)
        max_k_layout.setSpacing(5)
        max_k_layout.addWidget(self._max_k_lbl)
        max_k_layout.addWidget(self._max_k_spin)

        self._n_clusters_spin.valueChanged.connect(self._on_n_clusters_changed)

        # main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(
            settings_section(
                "Event detection",
                "Find peaks in calcium fluorescence. Height rejects "
                "small events; spacing separates nearby peaks; prominence rejects "
                "peaks that barely rise above their surroundings.",
                self._peaks_height,
                self._peaks_distance_wdg,
                self._peaks_prominence_wdg,
            )
        )
        layout.addWidget(
            settings_section(
                "Network bursts",
                "Identify intervals when a large fraction of ROIs is "
                "active together. Duration rejects brief fluctuations; smoothing joins "
                "activity over nearby time points.",
                self._burst_wdg,
            )
        )
        layout.addWidget(
            settings_section(
                "ROI grouping",
                "Group ROIs with similar calcium activity. Auto estimates "
                "the number of groups up to the selected maximum.",
                self._n_clusters_wdg,
                self._max_k_wdg,
            )
        )

    # PUBLIC METHODS ------------------------------------------------------------------

    def set_labels_width(self, width: int) -> None:
        """Set the width of the labels."""
        self._peaks_height._peaks_height_lbl.setFixedWidth(width)
        self._peaks_distance_lbl.setFixedWidth(width)
        self._peaks_prominence_lbl.setFixedWidth(width)
        self._burst_wdg._burst_threshold_lbl.setFixedWidth(width)
        self._burst_wdg._burst_min_threshold_label.setFixedWidth(width)
        self._burst_wdg._burst_blur_label.setFixedWidth(width)
        self._n_clusters_lbl.setFixedWidth(width)
        self._max_k_lbl.setFixedWidth(width)

    def value(self) -> CalciumPeaksData:
        """Get the current values of the widget."""
        burst_threshold, burst_min_duration, burst_blur_sigma = self._burst_wdg.value()
        return CalciumPeaksData(
            peaks_height=self._peaks_height.value()[0],
            peaks_height_mode=self._peaks_height.value()[1],
            peaks_distance=self._peaks_distance_spin.value(),
            peaks_prominence_multiplier=self._peaks_prominence_multiplier_spin.value(),
            burst_threshold=burst_threshold,
            burst_min_duration=burst_min_duration,
            burst_blur_sigma=burst_blur_sigma,
            cluster_n_clusters=self._n_clusters_spin.value(),
            cluster_max_k=self._max_k_spin.value(),
        )

    def setValue(self, value: CalciumPeaksData) -> None:
        """Set the values of the widget."""
        self._peaks_height.setValue((value.peaks_height, value.peaks_height_mode))
        self._peaks_distance_spin.setValue(value.peaks_distance)
        self._peaks_prominence_multiplier_spin.setValue(
            value.peaks_prominence_multiplier
        )
        bst = (value.burst_threshold, value.burst_min_duration, value.burst_blur_sigma)
        self._burst_wdg.setValue(bst)
        self._n_clusters_spin.setValue(value.cluster_n_clusters)
        self._max_k_spin.setValue(value.cluster_max_k)

    def reset(self) -> None:
        """Reset the widget to default values."""
        self._peaks_height.setValue((DEFAULT_HEIGHT, MULTIPLIER))
        self._peaks_distance_spin.setValue(200.0)  # 2 frames at 10fps = 200ms
        self._peaks_prominence_multiplier_spin.setValue(1)
        self._burst_wdg.setValue(
            (
                DEFAULT_BURST_THRESHOLD,
                DEFAULT_MIN_BURST_DURATION,
                DEFAULT_BURST_GAUSS_SIGMA,
            )
        )
        self._n_clusters_spin.setValue(DEFAULT_CLUSTER_N_CLUSTERS)
        self._max_k_spin.setValue(DEFAULT_CLUSTER_MAX_K)

    # PRIVATE METHODS -----------------------------------------------------------------

    def _on_n_clusters_changed(self, value: int) -> None:
        """Enable Max K only when auto-detection is active (N Clusters = 0)."""
        enabled = value == 0
        self._max_k_spin.setEnabled(enabled)
        self._max_k_lbl.setEnabled(enabled)


class _SpikeThresholdWidget(QWidget):
    """Widget to select the spike threshold multiplier."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setToolTip(
            "Spike detection threshold for identifying spikes in OASIS-denoised "
            "inferred spike traces.\n\n"
            "Two modes:\n"
            "• Global Minimum: Same absolute threshold applied to ALL ROIs across "
            "ALL FOVs. Spike amplitudes below this value are rejected (set to 0) "
            "everywhere.\n\n"
            "• Noise Multiplier: Adaptive threshold computed individually for EACH "
            "ROI in EACH FOV.\n"
            "  Threshold = median + multiplier * MAD-based noise estimate.\n"
            "  Noise is estimated from the lower half of positive amplitudes.\n"
            "  If the MAD-based noise is zero, median * multiplier is used.\n"
            "  Fewer than five positive amplitudes disables detection."
        )

        self._spike_threshold_lbl = QLabel("Spike Detection Threshold:", self)
        self._spike_threshold_lbl.setSizePolicy(*FIXED)

        self._spike_threshold_spin = QDoubleSpinBox(self)
        self._spike_threshold_spin.setDecimals(4)
        self._spike_threshold_spin.setRange(0.0, 10000.0)
        self._spike_threshold_spin.setSingleStep(0.1)
        self._spike_threshold_spin.setValue(DEFAULT_SPIKE_THRESHOLD)

        self._global_spike_threshold = QRadioButton("Global threshold")

        self._threshold_multiplier = QRadioButton("Noise multiplier")
        self._threshold_multiplier.setChecked(True)

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._spike_threshold_lbl, 0, 0)
        layout.addWidget(self._spike_threshold_spin, 0, 1)
        layout.addWidget(self._threshold_multiplier, 1, 0)
        layout.addWidget(self._global_spike_threshold, 1, 1)
        layout.setColumnStretch(1, 1)

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

    def setValue(self, value: tuple[float | None, str]) -> None:
        """Set the value of the spike threshold widget."""
        threshold, mode = value
        if threshold is None:
            raise ValueError("OASIS requires an explicit threshold value.")
        self._spike_threshold_spin.setValue(threshold)
        self._global_spike_threshold.setChecked(mode == GLOBAL_SPIKE_THRESHOLD)
        self._threshold_multiplier.setChecked(mode == MULTIPLIER)


class _CascadeThresholdWidget(QWidget):
    """CASCADE uses a model-derived AP threshold or an explicit spikes/frame value."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._spike_threshold_lbl = QLabel("CASCADE Threshold:", self)
        self._mode = QComboBox(self)
        self._mode.addItem("Model AP Threshold", CASCADE_AP_THRESHOLD)
        self._mode.addItem("Global Minimum (spikes/frame)", GLOBAL_SPIKE_THRESHOLD)
        self._global = QLineEdit(self)
        self._global.setPlaceholderText("Enter an explicit threshold in spikes/frame")
        self._fraction = QLineEdit(str(1 / math.e), self)
        self._fraction.setToolTip(
            "Fraction of the peak prediction for one action potential under the "
            "selected model's smoothing kernel. Default: 1/e. Must be in (0, 1]."
        )
        self._fraction_label = QLabel("AP peak fraction:", self)
        self._global_label = QLabel("Global value (spikes/frame):", self)
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self._spike_threshold_lbl, 0, 0)
        layout.addWidget(self._mode, 0, 1)
        layout.addWidget(self._fraction_label, 1, 0)
        layout.addWidget(self._fraction, 1, 1)
        layout.addWidget(self._global_label, 2, 0)
        layout.addWidget(self._global, 2, 1)
        layout.setColumnStretch(1, 1)
        self._mode.currentIndexChanged.connect(self._update_mode)
        self._update_mode()

    def _update_mode(self) -> None:
        ap_mode = self._mode.currentData() == CASCADE_AP_THRESHOLD
        self._fraction.setVisible(ap_mode)
        self._fraction_label.setVisible(ap_mode)
        self._global.setVisible(not ap_mode)
        self._global_label.setVisible(not ap_mode)

    def value(self) -> tuple[float | None, str]:
        mode = self._mode.currentData()
        threshold = None
        if mode == GLOBAL_SPIKE_THRESHOLD:
            if not self._global.text().strip():
                raise ValueError(
                    "CASCADE global threshold needs an explicit spikes/frame value."
                )
            threshold = float(self._global.text())
        return threshold, mode

    def ap_fraction(self) -> float | None:
        return (
            float(self._fraction.text())
            if self._mode.currentData() == CASCADE_AP_THRESHOLD
            else None
        )

    def setValue(self, value: tuple[float | None, str]) -> None:
        threshold, mode = value
        index = self._mode.findData(mode)
        if index < 0:
            raise ValueError(f"Invalid CASCADE threshold mode: {mode}")
        self._global.setText("" if threshold is None else str(threshold))
        self._mode.setCurrentIndex(index)


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
            "• Burst Min Duration (ms):\n"
            "   Minimum duration (in milliseconds) for a detected burst to be "
            "considered valid.\n"
            "   Example: At 10 fps, 300ms = 3 frames minimum burst duration.\n"
            "   Higher values ensure only sustained bursts are detected.\n\n"
            "• Burst Gaussian Blur Sigma:\n"
            "   Gaussian smoothing applied to population activity before "
            "burst detection.\n"
            "   Reduces noise and connects nearby activity peaks into "
            "coherent bursts.\n"
            "   Higher values provide more smoothing, merging closer events.\n"
            "   Lower values preserve temporal precision but may "
            "fragment bursts.\n"
            "   Set to 0 to disable smoothing."
        )

        self._burst_threshold_lbl = QLabel("Burst Threshold (%):", self)
        self._burst_threshold_lbl.setSizePolicy(*FIXED)
        self._burst_threshold = QDoubleSpinBox(self)
        self._burst_threshold.setDecimals(2)
        self._burst_threshold.setRange(0.0, 100.0)
        self._burst_threshold.setSingleStep(1)
        self._burst_threshold.setValue(DEFAULT_BURST_THRESHOLD)

        self._burst_min_threshold_label = QLabel("Burst Min Duration:", self)
        self._burst_min_threshold_label.setSizePolicy(*FIXED)
        self._burst_min_duration_ms = QDoubleSpinBox(self)
        self._burst_min_duration_ms.setSuffix(" ms")
        self._burst_min_duration_ms.setDecimals(2)
        self._burst_min_duration_ms.setRange(0.0, 100000.0)
        self._burst_min_duration_ms.setSingleStep(100.0)
        self._burst_min_duration_ms.setValue(DEFAULT_MIN_BURST_DURATION)

        self._burst_blur_label = QLabel("Burst Gaussian Blur Sigma:", self)
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
        burst_layout.addWidget(self._burst_min_duration_ms, 1, 1)
        burst_layout.addWidget(self._burst_blur_label, 2, 0)
        burst_layout.addWidget(self._burst_blur_sigma, 2, 1)

    # PUBLIC METHODS ------------------------------------------------------------------

    def value(self) -> tuple[float, float, float]:
        """Return the burst detection parameters."""
        return (
            self._burst_threshold.value(),
            self._burst_min_duration_ms.value(),
            self._burst_blur_sigma.value(),
        )

    def setValue(self, value: tuple[float, float, float]) -> None:
        """Set the value of the burst widget."""
        threshold, duration, sigma = value
        self._burst_threshold.setValue(threshold)
        self._burst_min_duration_ms.setValue(duration)
        self._burst_blur_sigma.setValue(sigma)


class _SpikeWidget(QWidget):
    """Widget to select the spike detection settings."""

    def __init__(
        self, parent: QWidget | None = None, *, method: SpikeMethod = "oasis"
    ) -> None:
        super().__init__(parent)
        self._method = method
        self._loaded_data: SpikeData | None = None
        self._displayed_data: SpikeData | None = None

        # spikes threshold
        self._spike_threshold_wdg = (
            _CascadeThresholdWidget(self)
            if method == "cascade"
            else _SpikeThresholdWidget(self)
        )

        # burst detection settings
        self._burst_wdg = _BurstWidget(self)

        # spike synchrony max lag settings
        self._spike_max_lag_wdg = QWidget(self)
        self._spike_max_lag_wdg.setToolTip(
            "Inferred Spike Max-Lag Cross-Correlation Settings\n\n"
            "Temporal Tolerance Parameter:\n"
            "Controls the maximum time window (in milliseconds) for cross-correlation "
            "analysis between ROI pairs.\n\n"
            "How it works:\n"
            "• Value = 500 ms: Checks correlations within ±500 ms window\n"
            "• Algorithm slides one spike train over another, looking for "
            "best match within this range\n"
            "• Takes the MAXIMUM correlation found within the lag window\n"
            "• Larger values: More permissive, detects more synchrony but may "
            "include false positives\n"
            "• Smaller values: More strict, may miss genuine synchrony with "
            "slight timing offsets\n\n"
            "Example with Max Lag = 500 ms @ 10 fps:\n"
            "ROI 1 spikes: [1000ms, 2500ms, 4000ms]  ROI 2 spikes: [1200ms, 2400ms, "
            "4100ms]\n"
            "Algorithm finds high correlation at lag +200ms and -100ms\n"
            "Result: High synchrony score based on best alignment."
        )
        self._spikes_max_lag_lbl = QLabel("CCG Max Lag:", self._spike_max_lag_wdg)
        self._spikes_max_lag_lbl.setSizePolicy(*FIXED)
        self._spikes_sync_cross_corr_max_lag = QDoubleSpinBox(self._spike_max_lag_wdg)
        self._spikes_sync_cross_corr_max_lag.setSuffix(" ms")
        self._spikes_sync_cross_corr_max_lag.setDecimals(2)
        self._spikes_sync_cross_corr_max_lag.setRange(0.0, 10000.0)  # 0 to 10 seconds
        self._spikes_sync_cross_corr_max_lag.setSingleStep(10.0)
        self._spikes_sync_cross_corr_max_lag.setValue(DEFAULT_SPIKE_SYNCHRONY_MAX_LAG)
        spike_max_lag_layout = QHBoxLayout(self._spike_max_lag_wdg)
        spike_max_lag_layout.setContentsMargins(0, 0, 0, 0)
        spike_max_lag_layout.setSpacing(5)
        spike_max_lag_layout.addWidget(self._spikes_max_lag_lbl)
        spike_max_lag_layout.addWidget(self._spikes_sync_cross_corr_max_lag)

        # spike jitter synchrony settings
        self._spike_jitter_wdg = QWidget(self)
        self._spike_jitter_wdg.setToolTip(
            "Inferred Spike Jitter Synchrony Settings\n\n"
            "Temporal Tolerance Parameter:\n"
            "Controls the maximum time window (in milliseconds) for detecting "
            "synchronous spikes between ROI pairs.\n\n"
            "How it works:\n"
            "• Value = 200: Spikes within ±200 ms are considered synchronous\n"
            "• Compares timing of spikes between all ROI pairs\n"
            "• Larger values: more permissive, detects more synchrony but may "
            "include false positives\n"
            "• Smaller values: more strict, may miss genuine synchrony with "
            "slight timing offsets\n\n"
            "Example with Jitter = 200 ms @ 10 fps:\n"
            "ROI 1 spikes: [1000ms, 2500ms, 4000ms]  ROI 2 spikes: [1200ms, 2400ms, "
            "4100ms]\n"
            "Result: All pairs are synchronous (differences ≤ 200 ms)."
        )
        self._spike_jitter_lbl = QLabel("Synchrony Jitter:", self._spike_jitter_wdg)
        self._spike_jitter_lbl.setSizePolicy(*FIXED)
        self._spike_jitter_spin = QDoubleSpinBox(self._spike_jitter_wdg)
        self._spike_jitter_spin.setSuffix(" ms")
        self._spike_jitter_spin.setDecimals(2)
        self._spike_jitter_spin.setRange(0.0, 10000.0)  # 0 to 10 seconds
        self._spike_jitter_spin.setSingleStep(10.0)
        self._spike_jitter_spin.setValue(DEFAULT_SPIKE_SYNC_JITTER_WINDOW)
        spike_jitter_layout = QHBoxLayout(self._spike_jitter_wdg)
        spike_jitter_layout.setContentsMargins(0, 0, 0, 0)
        spike_jitter_layout.setSpacing(5)
        spike_jitter_layout.addWidget(self._spike_jitter_lbl)
        spike_jitter_layout.addWidget(self._spike_jitter_spin)

        # CCG baseline shuffles setting
        self._ccg_shuffles_wdg = QWidget(self)
        self._ccg_shuffles_wdg.setToolTip(
            "CCG Baseline Correction Shuffles\n\n"
            "Controls the number of circular shift surrogates used for baseline "
            "correction in cross-correlogram (CCG) analysis.\n\n"
            "How it works:\n"
            "• The shift predictor method circularly shifts one spike train to "
            "create surrogate pairs\n"
            "• This breaks precise timing while preserving overall firing rates\n"
            "• The baseline mean and std are computed from these shuffled CCGs\n"
            "• Z-scores: (raw_CCG - baseline_mean) / baseline_std\n\n"
            "Trade-offs:\n"
            "• More shuffles: More accurate baseline, but slower\n"
            "• Fewer shuffles: Faster, but noisier baseline estimates\n\n"
        )
        self._ccg_shuffles_lbl = QLabel(
            "CCG Baseline Shuffles:", self._ccg_shuffles_wdg
        )
        self._ccg_shuffles_lbl.setSizePolicy(*FIXED)
        self._ccg_shuffles_spin = QSpinBox(self._ccg_shuffles_wdg)
        self._ccg_shuffles_spin.setRange(1, 500)
        self._ccg_shuffles_spin.setSingleStep(1)
        self._ccg_shuffles_spin.setValue(DEFAULT_CCG_N_SHUFFLES)
        ccg_shuffles_layout = QHBoxLayout(self._ccg_shuffles_wdg)
        ccg_shuffles_layout.setContentsMargins(0, 0, 0, 0)
        ccg_shuffles_layout.setSpacing(5)
        ccg_shuffles_layout.addWidget(self._ccg_shuffles_lbl)
        ccg_shuffles_layout.addWidget(self._ccg_shuffles_spin)

        # Rising edge analysis setting
        self._rising_edge_wdg = QWidget(self)
        self._rising_edge_wdg.setToolTip(
            "Rising Edge Analysis\n\n"
            "When enabled, performs additional CCG analysis using spike onset times "
            "(rising edges) instead of the full spike duration.\n\n"
            "How it works:\n"
            "• Rising edges: transitions from 0→1 in binary spike train\n"
            "• Captures precise spike onset timing\n"
            "• Useful for detecting fine-scale temporal coordination\n\n"
            "Trade-offs:\n"
            "• Enabled: More detailed analysis, ~2x CCG computation time\n"
            "• Disabled: Faster analysis using thresholded binary spikes\n\n"
            "Note: Rising edge results are stored separately and can be exported "
            "independently."
        )
        self._rising_edge_lbl = QLabel("Rising Edge Analysis:", self._rising_edge_wdg)
        self._rising_edge_lbl.setSizePolicy(*FIXED)
        self._rising_edge_checkbox = QCheckBox(self._rising_edge_wdg)
        self._rising_edge_checkbox.setChecked(DEFAULT_ENABLE_RISING_EDGE_ANALYSIS)
        rising_edge_layout = QHBoxLayout(self._rising_edge_wdg)
        rising_edge_layout.setContentsMargins(0, 0, 0, 0)
        rising_edge_layout.setSpacing(5)
        rising_edge_layout.addWidget(self._rising_edge_lbl)
        rising_edge_layout.addWidget(self._rising_edge_checkbox)

        # main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(
            settings_section(
                "Activity threshold",
                "Convert the inferred trace into active/inactive "
                "frames for raster, burst and synchrony analysis.",
                self._spike_threshold_wdg,
            )
        )
        layout.addWidget(
            settings_section(
                "Synchrony between ROIs",
                "Compare the timing of activity between pairs "
                "of ROIs. Lag and jitter set how far apart activity can occur "
                "and still "
                "be counted together.",
                self._spike_max_lag_wdg,
                self._spike_jitter_wdg,
            )
        )
        advanced = QCollapsible("Advanced synchrony: shuffles and onset analysis")
        advanced.addWidget(
            guidance(
                "More shuffles improve the baseline estimate but take longer. Onset "
                "analysis compares activity starts in addition to active frames.",
                advanced,
            )
        )
        advanced.addWidget(self._ccg_shuffles_wdg)
        advanced.addWidget(self._rising_edge_wdg)
        layout.addWidget(advanced)
        layout.addWidget(
            settings_section(
                "Network bursts",
                "Detect periods when a large fraction of ROIs is "
                "active together. These settings apply only to this method.",
                self._burst_wdg,
            )
        )
        if method == "cascade":
            self.reset()

    # PUBLIC METHODS ------------------------------------------------------------------

    def set_labels_width(self, width: int) -> None:
        """Set the width of the labels."""
        self._spike_threshold_wdg._spike_threshold_lbl.setFixedWidth(width)
        self._burst_wdg._burst_threshold_lbl.setFixedWidth(width)
        self._burst_wdg._burst_min_threshold_label.setFixedWidth(width)
        self._burst_wdg._burst_blur_label.setFixedWidth(width)
        self._spikes_max_lag_lbl.setFixedWidth(width)
        self._spike_jitter_lbl.setFixedWidth(width)
        self._ccg_shuffles_lbl.setFixedWidth(width)
        self._rising_edge_lbl.setFixedWidth(width)

    def value(self) -> SpikeData:
        data = self._read_value()
        if self._loaded_data is None or self._displayed_data is None:
            return data
        # Spin-box precision is a display choice, not a change to stored science.
        preserved = {
            name: getattr(self._loaded_data, name)
            for name in SpikeData.__dataclass_fields__
            if getattr(data, name) == getattr(self._displayed_data, name)
        }
        return replace(data, **preserved)

    def _read_value(self) -> SpikeData:
        """Get the current values of the widget."""
        spike_threshold, spike_threshold_mode = self._spike_threshold_wdg.value()
        burst_threshold, burst_min_duration, burst_blur_sigma = self._burst_wdg.value()
        synchrony_lag = self._spikes_sync_cross_corr_max_lag.value()
        synchrony_jitter = self._spike_jitter_spin.value()
        ccg_n_shuffles = self._ccg_shuffles_spin.value()
        enable_rising_edge = self._rising_edge_checkbox.isChecked()

        return SpikeData(
            method=self._method,
            cascade_ap_threshold_fraction=(
                self._spike_threshold_wdg.ap_fraction()
                if isinstance(self._spike_threshold_wdg, _CascadeThresholdWidget)
                else None
            ),
            spike_threshold=spike_threshold,
            spike_threshold_mode=spike_threshold_mode,
            burst_threshold=burst_threshold,
            burst_min_duration=burst_min_duration,
            burst_blur_sigma=burst_blur_sigma,
            synchrony_lag=synchrony_lag,
            synchrony_jitter=synchrony_jitter,
            ccg_n_shuffles=ccg_n_shuffles,
            enable_rising_edge_analysis=enable_rising_edge,
        )

    def setValue(self, value: SpikeData) -> None:
        """Set the values of the widget."""
        if value.method != self._method:
            raise ValueError("Spike analysis settings must match their method tab.")
        value.to_model()
        tr = (value.spike_threshold, value.spike_threshold_mode)
        self._spike_threshold_wdg.setValue(tr)
        if isinstance(self._spike_threshold_wdg, _CascadeThresholdWidget):
            self._spike_threshold_wdg._fraction.setText(
                str(value.cascade_ap_threshold_fraction or 1 / math.e)
            )
        bst = (value.burst_threshold, value.burst_min_duration, value.burst_blur_sigma)
        self._burst_wdg.setValue(bst)
        self._spikes_sync_cross_corr_max_lag.setValue(value.synchrony_lag)
        self._spike_jitter_spin.setValue(value.synchrony_jitter)
        self._ccg_shuffles_spin.setValue(value.ccg_n_shuffles)
        self._rising_edge_checkbox.setChecked(value.enable_rising_edge_analysis)
        self._loaded_data = value
        self._displayed_data = self._read_value()

    def reset(self) -> None:
        """Reset the widget to default values."""
        self._loaded_data = self._displayed_data = None
        self._spike_threshold_wdg.setValue(
            (None, CASCADE_AP_THRESHOLD)
            if self._method == "cascade"
            else (DEFAULT_SPIKE_THRESHOLD, MULTIPLIER)
        )
        if isinstance(self._spike_threshold_wdg, _CascadeThresholdWidget):
            self._spike_threshold_wdg._fraction.setText(str(1 / math.e))
        self._burst_wdg.setValue(
            (
                DEFAULT_BURST_THRESHOLD,
                DEFAULT_MIN_BURST_DURATION,
                DEFAULT_BURST_GAUSS_SIGMA,
            )
        )
        self._spikes_sync_cross_corr_max_lag.setValue(DEFAULT_SPIKE_SYNCHRONY_MAX_LAG)
        self._spike_jitter_spin.setValue(DEFAULT_SPIKE_SYNC_JITTER_WINDOW)
        self._ccg_shuffles_spin.setValue(DEFAULT_CCG_N_SHUFFLES)
        self._rising_edge_checkbox.setChecked(DEFAULT_ENABLE_RISING_EDGE_ANALYSIS)


class _MetadataWidget(QWidget):
    """Widget for metadata settings - frame rate only for analysis."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Frame Rate widget
        self._frame_rate_wdg = QWidget(self)
        self._frame_rate_wdg.setToolTip(
            "True acquisition rate in frames per second (fps).\n\n"
            "Converts event spacing, burst duration and synchrony windows to frames. "
            "Use acquisition timestamps or verified metadata. Exposure time alone "
            "does not include readout or gaps between frames."
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

        # FromMetaButton
        self._from_meta_btn = FromMetaButton(self, "Load From Metadata")
        self._from_meta_btn.setToolTip(
            "Try to load frame rate from the acquisition metadata."
        )
        self._from_meta_btn.setSizePolicy(
            QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed
        )

        # Main layout: frame rate field + button horizontally
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(self._frame_rate_wdg)
        layout.addWidget(self._from_meta_btn)

    # PUBLIC METHODS ------------------------------------------------------------------

    @property
    def from_metadata(self) -> None:
        """Signal emitted when the 'Load From Metadata' button is clicked."""
        return self._from_meta_btn.clicked  # type: ignore

    def set_labels_width(self, width: int) -> None:
        """Set the width of the labels."""
        self._frame_rate_lbl.setFixedWidth(width)

    def value(self) -> float:
        """Get the current frame rate value."""
        return self._frame_rate_spin.value()  # type: ignore

    def setValue(self, value: float) -> None:
        """Set the frame rate value."""
        self._frame_rate_spin.setValue(value)

    def reset(self) -> None:
        """Reset the widget to default values."""
        self._frame_rate_spin.setValue(DEFAULT_FRAME_RATE)
