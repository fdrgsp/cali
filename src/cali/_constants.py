"""Constants used throughout the cali package."""

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
ZR = WRITERS[OME_ZARR][0]
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
CALCIUM_PEAKS_SYNCHRONY_METHOD = "jitter_window"
CALCIUM_SYNC_JITTER_WINDOW = "calcium_sync_jitter_window"
CALCIUM_NETWORK_THRESHOLD = "calcium_network_threshold"
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
DEFAULT_BURST_THRESHOLD = 30.0
DEFAULT_MIN_BURST_DURATION = 3
DEFAULT_BURST_GAUSS_SIGMA = 2.0
DEFAULT_DFF_WINDOW = 30
DEFAULT_PEAKS_DISTANCE = 2
DEFAULT_HEIGHT = 3
DEFAULT_SPIKE_THRESHOLD = 1
DEFAULT_SPIKE_SYNCHRONY_MAX_LAG = 5
DEFAULT_CALCIUM_SYNC_JITTER_WINDOW = 2
DEFAULT_CALCIUM_NETWORK_THRESHOLD = 90.0
DEFAULT_NEUROPIL_INNER_RADIUS = 0
DEFAULT_NEUROPIL_MIN_PIXELS = 0
DEFAULT_NEUROPIL_CORRECTION_FACTOR = 0.7
DEFAULT_FRAME_RATE = 10.0  # in frames per second (fps)
