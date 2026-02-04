"""Constants used throughout the cali package."""

from __future__ import annotations

from typing import Literal

# ==================== Metadata Keys ====================
EVENT_KEY = "mda_event"
PYMMCW_METADATA_KEY = "pymmcore_widgets"
RUNNER_TIME_KEY = "runner_time_ms"

# ==================== Experiment Types ====================
SPONTANEOUS = "Spontaneous Activity"
EVOKED = "Evoked Activity"

# ==================== Writer Types and File Extensions ====================
ZARR_TESNSORSTORE = "tensorstore-zarr"
OME_ZARR = "ome-zarr"
# dict with writer name and extension
WRITERS: dict[str, list[str]] = {
    ZARR_TESNSORSTORE: [".tensorstore.zarr"],
    OME_ZARR: [".ome.zarr"],
}
TS = WRITERS[ZARR_TESNSORSTORE][0]
OZ = WRITERS[OME_ZARR][0]
HCS = "hcs"

# ==================== Colors ====================
RED = "#C33"
GREEN = "#00FF00"
UNSELECTABLE_COLOR = "#404040"

# ==================== File Names ====================
GENOTYPE_MAP = "genotype_plate_map.json"
TREATMENT_MAP = "treatment_plate_map.json"
STIMULATION_MASK = "stimulation_mask.tif"
SETTINGS_PATH = "settings.json"

# ==================== Condition Keys ====================
COND1 = "condition_1"
COND2 = "condition_2"
PLATE_PLAN = "plate_plan"

# ==================== Units ====================
MWCM = "mW/cm²"

# ==================== Analysis Settings Keys ====================
LED_POWER_EQUATION = "led_power_equation"
PEAKS_HEIGHT_VALUE = "peaks_height_value"
PEAKS_HEIGHT_MODE = "peaks_height_mode"
SPIKE_THRESHOLD_VALUE = "spike_threshold_value"
SPIKE_THRESHOLD_MODE = "spike_threshold_mode"
PEAKS_PROMINENCE_MULTIPLIER = "peaks_prominence_multiplier"
PEAKS_DISTANCE = "peaks_distance"
DFF_WINDOW = "dff_window"
BURST_THRESHOLD = "burst_threshold"
BURST_MIN_DURATION = "burst_min_duration"
BURST_GAUSSIAN_SIGMA = "burst_gaussian_sigma"
DECAY_CONSTANT = "decay constant"
SPIKE_SYNCHRONY_METHOD = "cross_correlation"
SPIKES_SYNC_CROSS_CORR_MAX_LAG = "spikes_sync_cross_corr_lag"
CALCIUM_SYNC_JITTER_WINDOW = "calcium_sync_jitter_window"
NEUROPIL_INNER_RADIUS = "neuropil_inner_radius"
NEUROPIL_MIN_PIXELS = "neuropil_min_pixels"
NEUROPIL_CORRECTION_FACTOR = "neuropil_correction_factor"

# ==================== Analysis Categories ====================
EVK_STIM = "evk_stim"
EVK_NON_STIM = "evk_non_stim"

# ==================== Output Suffixes ====================
MEAN_SUFFIX = "_Mean"
SEM_SUFFIX = "_SEM"
N_SUFFIX = "_N"

# ==================== Analysis Thresholds ====================
EXCLUDE_AREA_SIZE_THRESHOLD = 50  # µm² threshold for excluding small ROIs
STIMULATION_AREA_THRESHOLD = 0.1  # 10% overlap threshold for stimulated ROIs
MAX_FRAMES_AFTER_STIMULATION = 5

# ==================== Global Settings Modes ====================
GLOBAL_HEIGHT = "global"
GLOBAL_SPIKE_THRESHOLD = "global"
MULTIPLIER = "multiplier"

# ==================== Default Values ====================
DEFAULT_BURST_THRESHOLD = 65.0
DEFAULT_CALCIUM_BURST_THRESHOLD = 25.0
DEFAULT_MIN_BURST_DURATION = 3
DEFAULT_BURST_GAUSS_SIGMA = 0.3
DEFAULT_FRAME_RATE = 10.0  # frames per second (fps)
DEFAULT_DFF_WINDOW = 10.0  # seconds
DEFAULT_PEAKS_DISTANCE = 200.0  # milliseconds (2 frames at 10 fps)
DEFAULT_HEIGHT = 3
DEFAULT_SPIKE_THRESHOLD = 3
DEFAULT_SPIKE_SYNCHRONY_MAX_LAG = 500.0  # milliseconds
DEFAULT_SPIKE_SYNC_JITTER_WINDOW = 200.0  # milliseconds
DEFAULT_CCG_N_SHUFFLES = 20  # number of shuffles for CCG baseline correction
DEFAULT_ENABLE_RISING_EDGE_ANALYSIS = False  # whether to compute CCG on rising edges
DEFAULT_NEUROPIL_INNER_RADIUS = 0
DEFAULT_NEUROPIL_MIN_PIXELS = 0
DEFAULT_NEUROPIL_CORRECTION_FACTOR = 0.7
DEFAULT_DFF_PERCENTILE = 10  # percentile for ΔF/F baseline calculation

# ==================== Database ====================
DEFAULT_CALI_DB_NAME = "results.cali"

# ==================== Extraction ====================
# Type for trace data export - must be defined before constants for type checking
TraceDataType = Literal[
    "Raw Calcium Traces",  # RAW_CALCIUM_TRACES
    "Neuropil Traces",  # NEUROPIL_TRACES
    "Neuropil Corrected Traces",  # NEUROPIL_CORRECTED_TRACES
    "ΔF/F Traces",  # DFF_TRACES
    "OASIS Deconvolved ΔF/F Traces",  # DEC_DFF_TRACES
    "OASIS Inferred Spikes Traces",  # INFERRED_SPIKES_TRACES
    "OASIS Thresholded Inferred Spikes (Binary)",  # INFERRED_SPIKES_THRESHOLDED_BINARY
]

CorrelationDataType = Literal[
    # Calcium correlations
    "ΔF/F Correlation Matrix",  # CALCIUM_DFF_CORRELATION
    "Deconvolved ΔF/F Correlation Matrix",  # CALCIUM_DEC_DFF_CORRELATION
    # Inferred Spikes - Thresholded Binary
    "Inferred Spikes Synchrony Matrix",  # INFERRED_SPIKES_SYNCHRONY
    "Inferred Spikes Cross-Correlation Matrix",  # INFERRED_SPIKES_CROSS_CORRELATION
    "Inferred Spikes Cross-Correlation Lags Matrix",  # INFERRED_SPIKES_CROSS_CORRELATION_LAGS  # noqa: E501
    "Inferred Spikes CCG Z-Score Matrix",  # INFERRED_SPIKES_CCG_ZSCORE
    # Inferred Spikes - Thresholded Rising Edges
    "Inferred Spikes Synchrony Matrix (Rising Edges)",  # INFERRED_SPIKES_SYNCHRONY_RISING_EDGES  # noqa: E501
    "Inferred Spikes Cross-Correlation Matrix (Rising Edges)",  # INFERRED_SPIKES_CROSS_CORRELATION_RISING_EDGES  # noqa: E501
    "Inferred Spikes Cross-Correlation Lags Matrix (Rising Edges)",  # INFERRED_SPIKES_CROSS_CORRELATION_LAGS_RISING_EDGES  # noqa: E501
    "Inferred Spikes CCG Z-Score Matrix (Rising Edges)",  # INFERRED_SPIKES_CCG_ZSCORE_RISING_EDGES  # noqa: E501
]

# fmt: off
RAW_CALCIUM_TRACES: TraceDataType = "Raw Calcium Traces"
NEUROPIL_TRACES: TraceDataType = "Neuropil Traces"
NEUROPIL_CORRECTED_TRACES: TraceDataType = "Neuropil Corrected Traces"
DFF_TRACES: TraceDataType = "ΔF/F Traces"
DEC_DFF_TRACES: TraceDataType = "OASIS Deconvolved ΔF/F Traces"
INFERRED_SPIKES_TRACES: TraceDataType = "OASIS Inferred Spikes Traces"
INFERRED_SPIKES_THRESHOLDED_BINARY: TraceDataType = "OASIS Thresholded Inferred Spikes (Binary)"  # noqa: E501
# CALCIUM_PEAKS = "Calcium Peaks"

# Calcium correlations
CALCIUM_DFF_CORRELATION: CorrelationDataType = "ΔF/F Correlation Matrix"
CALCIUM_DEC_DFF_CORRELATION: CorrelationDataType = "Deconvolved ΔF/F Correlation Matrix"
# Inferred Spikes - Thresholded Binary
INFERRED_SPIKES_SYNCHRONY: CorrelationDataType = "Inferred Spikes Synchrony Matrix"
INFERRED_SPIKES_CROSS_CORRELATION: CorrelationDataType = "Inferred Spikes Cross-Correlation Matrix"  # noqa: E501
INFERRED_SPIKES_CROSS_CORRELATION_LAGS: CorrelationDataType = "Inferred Spikes Cross-Correlation Lags Matrix"  # noqa: E501
INFERRED_SPIKES_CCG_ZSCORE: CorrelationDataType = "Inferred Spikes CCG Z-Score Matrix"
# Inferred Spikes - Thresholded Rising Edges
INFERRED_SPIKES_SYNCHRONY_RISING_EDGES: CorrelationDataType = "Inferred Spikes Synchrony Matrix (Rising Edges)"  # noqa: E501
INFERRED_SPIKES_CROSS_CORRELATION_RISING_EDGES: CorrelationDataType = "Inferred Spikes Cross-Correlation Matrix (Rising Edges)"  # noqa: E501
INFERRED_SPIKES_CROSS_CORRELATION_LAGS_RISING_EDGES: CorrelationDataType = "Inferred Spikes Cross-Correlation Lags Matrix (Rising Edges)"  # noqa: E501
INFERRED_SPIKES_CCG_ZSCORE_RISING_EDGES: CorrelationDataType = "Inferred Spikes CCG Z-Score Matrix (Rising Edges)"  # noqa: E501
# fmt: on
