from unittest.mock import MagicMock, patch

import numpy as np

from cali.plot._util import (
    _compute_jitter_synchrony_matrix_numba,
    _get_calcium_peaks_event_synchrony,
    _get_calcium_peaks_event_synchrony_matrix,
    _get_calcium_peaks_events_from_rois,
    _get_data_analysis_for_run,
    _get_spikes_over_threshold,
    _get_traces_for_run,
    equation_from_str,
    separate_stimulated_vs_non_stimulated_peaks,
)
from cali.sqlmodel import ROI, DataAnalysis, Traces


def test_equation_from_str() -> None:
    # Linear
    eq = equation_from_str("y = 2*x + 3")
    assert eq is not None
    assert eq(2) == 7

    # Quadratic
    eq = equation_from_str("y = 1*x^2 + 2*x + 1")
    assert eq is not None
    assert eq(2) == 9

    # Exponential
    eq = equation_from_str("y = 2*exp(0.1*x) + 1")
    assert eq is not None
    assert np.isclose(eq(10), 2 * np.exp(1) + 1)

    # Power
    eq = equation_from_str("y = 2*x^2 + 1")
    assert eq is not None
    assert eq(3) == 19

    # Logarithmic
    eq = equation_from_str("y = 2*log(x) + 1")
    assert eq is not None
    assert np.isclose(eq(np.e), 3)

    # Invalid
    assert equation_from_str("invalid") is None
    assert equation_from_str("") is None


def test_get_calcium_peaks_event_synchrony() -> None:
    # Empty or None
    assert _get_calcium_peaks_event_synchrony(None) is None
    assert _get_calcium_peaks_event_synchrony(np.array([])) is None

    # Too small
    assert _get_calcium_peaks_event_synchrony(np.zeros((1, 1))) is None

    # Valid matrix (3x3)
    # 1 0.5 0.2
    # 0.5 1 0.8
    # 0.2 0.8 1
    matrix = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.8], [0.2, 0.8, 1.0]])

    # Off-diagonal sums:
    # Row 0: 0.7 -> mean 0.35
    # Row 1: 1.3 -> mean 0.65
    # Row 2: 1.0 -> mean 0.5
    # Median of [0.35, 0.65, 0.5] is 0.5

    score = _get_calcium_peaks_event_synchrony(matrix)
    assert score is not None
    assert np.isclose(score, 0.5)


def test_get_calcium_peaks_event_synchrony_matrix() -> None:
    # Empty dict
    assert _get_calcium_peaks_event_synchrony_matrix({}) is None

    # Single ROI
    assert _get_calcium_peaks_event_synchrony_matrix({"roi1": [1, 0]}) is None

    # Two ROIs, perfect sync
    data = {"roi1": [1.0, 0.0, 1.0, 0.0], "roi2": [1.0, 0.0, 1.0, 0.0]}
    matrix = _get_calcium_peaks_event_synchrony_matrix(
        data, method="jitter_window", jitter_window=0
    )
    assert matrix is not None
    assert matrix.shape == (2, 2)
    assert np.allclose(matrix, np.ones((2, 2)))

    # Two ROIs, no sync
    data = {"roi1": [1.0, 0.0, 0.0, 0.0], "roi2": [0.0, 0.0, 1.0, 0.0]}
    matrix = _get_calcium_peaks_event_synchrony_matrix(
        data, method="jitter_window", jitter_window=0
    )
    assert matrix is not None
    assert matrix[0, 1] == 0.0

    # Jitter window sync
    data = {"roi1": [1.0, 0.0, 0.0, 0.0], "roi2": [0.0, 1.0, 0.0, 0.0]}
    # With jitter=1, these should match
    matrix = _get_calcium_peaks_event_synchrony_matrix(
        data, method="jitter_window", jitter_window=1
    )
    # Total peaks = 2. Coincidences: roi1->roi2 (yes), roi2->roi1 (yes) -> 2.
    # Score = 2/2 = 1.0
    assert matrix is not None
    assert matrix[0, 1] == 1.0

    # Correlation method
    data = {"roi1": [1.0, 0.0, 1.0, 0.0], "roi2": [1.0, 0.0, 1.0, 0.0]}
    matrix = _get_calcium_peaks_event_synchrony_matrix(data, method="correlation")
    assert matrix is not None
    assert matrix[0, 1] == 1.0


def test_separate_stimulated_vs_non_stimulated_peaks() -> None:
    dec_dff = np.array([0.1, 0.5, 0.2, 0.8, 0.3, 0.6])
    peaks_dec_dff = np.array([1, 3, 5])  # Peaks at frames 1, 3, 5
    # Amplitudes: 0.5, 0.8, 0.6

    # Stimulation at frame 2 with power 50
    pulse_on_frames_and_powers = {"2": 50}

    # Peak at 1 is before stim -> ignored by this function
    # (it only looks for peaks >= stim_frame)

    # i = bisect_left(peaks, stim_frame) -> index of first peak >= 2.
    # peaks are [1, 3, 5]. bisect_left(..., 2) returns index 1 (value 3).
    # So it considers peak at 3.
    # Check if peak_idx (3) <= stim_frame (2) + MAX_FRAMES (assume > 1)
    # If MAX_FRAMES is e.g. 30, then 3 <= 32. Yes.
    # So peak at 3 (amp 0.8) is stimulated.

    # Next iteration? Only one stim frame.

    # What about peak at 5?
    # The loop iterates over stim frames.
    # For stim frame 2, it finds peak at 3.
    # It doesn't loop over peaks. It finds the *first* peak after stim.
    # So peak at 5 is not associated with this stim event.

    # Case 1: ROI is stimulated
    stim, non_stim = separate_stimulated_vs_non_stimulated_peaks(
        dec_dff, peaks_dec_dff, pulse_on_frames_and_powers, is_roi_stimulated=True
    )

    assert "50%_unknown" in stim
    assert stim["50%_unknown"] == [0.8]
    assert len(non_stim) == 0

    # Case 2: ROI is not stimulated
    stim, non_stim = separate_stimulated_vs_non_stimulated_peaks(
        dec_dff, peaks_dec_dff, pulse_on_frames_and_powers, is_roi_stimulated=False
    )

    assert len(stim) == 0
    assert "50%_unknown" in non_stim
    assert non_stim["50%_unknown"] == [0.8]


def test_compute_jitter_synchrony_matrix_numba() -> None:
    # 3 ROIs
    # ROI 0: [1, 0, 0]
    # ROI 1: [0, 1, 0]
    # ROI 2: [0, 0, 1]

    peak_array = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

    # Jitter 0: Identity matrix
    matrix = _compute_jitter_synchrony_matrix_numba(peak_array, 0)
    assert np.allclose(matrix, np.eye(3))

    # Jitter 1:
    # 0-1: dist 1 <= 1 -> sync
    # 1-2: dist 1 <= 1 -> sync
    # 0-2: dist 2 > 1 -> no sync
    matrix = _compute_jitter_synchrony_matrix_numba(peak_array, 1)
    assert matrix[0, 1] == 1.0
    assert matrix[1, 2] == 1.0
    assert matrix[0, 2] == 0.0


def test_get_spike_synchrony_matrix() -> None:
    from cali.plot._util import _get_spike_synchrony_matrix

    # Empty dict
    assert _get_spike_synchrony_matrix({}) is None

    # Single ROI
    assert _get_spike_synchrony_matrix({"roi1": [1, 0]}) is None

    # Two ROIs, perfect sync
    data = {"roi1": [1.0, 0.0, 1.0, 0.0], "roi2": [1.0, 0.0, 1.0, 0.0]}
    matrix = _get_spike_synchrony_matrix(data, method="jitter_window", jitter_window=0)
    assert matrix is not None
    assert matrix.shape == (2, 2)
    assert np.allclose(matrix, np.ones((2, 2)))

    # Two ROIs, no sync
    data = {"roi1": [1.0, 0.0, 0.0, 0.0], "roi2": [0.0, 0.0, 1.0, 0.0]}
    matrix = _get_spike_synchrony_matrix(data, method="jitter_window", jitter_window=0)
    assert matrix is not None
    assert matrix[0, 1] == 0.0

    # Cross correlation method
    data = {"roi1": [1.0, 0.0, 0.0, 0.0], "roi2": [0.0, 1.0, 0.0, 0.0]}
    # Lag 1 should catch it
    matrix = _get_spike_synchrony_matrix(data, method="cross_correlation", max_lag=1)
    assert matrix is not None
    assert matrix[0, 1] > 0.0


def test_get_spike_synchrony() -> None:
    from cali.plot._util import _get_spike_synchrony

    # Empty or None
    assert _get_spike_synchrony(None) is None
    assert _get_spike_synchrony(np.array([])) is None

    # Too small
    assert _get_spike_synchrony(np.zeros((1, 1))) is None

    # Valid matrix (3x3)
    matrix = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.8], [0.2, 0.8, 1.0]])

    score = _get_spike_synchrony(matrix)
    assert score is not None
    assert np.isclose(score, 0.5)


def test_calculate_cross_correlation_synchrony() -> None:
    from cali.plot._util import _calculate_cross_correlation_synchrony

    # Perfect sync
    events_i = np.array([1.0, 0.0, 1.0])
    events_j = np.array([1.0, 0.0, 1.0])
    score = _calculate_cross_correlation_synchrony(events_i, events_j, max_lag=0)
    assert np.isclose(score, 1.0)

    # Lagged sync
    events_i = np.array([1.0, 0.0, 0.0])
    events_j = np.array([0.0, 1.0, 0.0])
    # Lag 0 -> 0
    score = _calculate_cross_correlation_synchrony(events_i, events_j, max_lag=0)
    assert np.isclose(score, 0.0)
    # Lag 1 -> >0
    score = _calculate_cross_correlation_synchrony(events_i, events_j, max_lag=1)
    assert score > 0.0

    # No signal
    events_i = np.array([0.0, 0.0, 0.0])
    events_j = np.array([0.0, 0.0, 0.0])
    score = _calculate_cross_correlation_synchrony(events_i, events_j, max_lag=1)
    assert np.isclose(score, 0.0)


def test_create_connectivity_matrix() -> None:
    from cali.plot._util import _create_connectivity_matrix

    # 3x3 matrix
    # 1.0 0.9 0.1
    # 0.9 1.0 0.2
    # 0.1 0.2 1.0
    corr_matrix = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.2], [0.1, 0.2, 1.0]])

    # Threshold 50th percentile of off-diagonals [0.9, 0.1, 0.9, 0.2, 0.1, 0.2]
    # Sorted: 0.1, 0.1, 0.2, 0.2, 0.9, 0.9
    # Median (50%) is 0.2
    # So >= 0.2 should be 1, < 0.2 should be 0

    conn_matrix = _create_connectivity_matrix(corr_matrix, threshold_percentile=50.0)
    assert conn_matrix[0, 1] == 1  # 0.9 >= 0.2
    assert conn_matrix[0, 2] == 0  # 0.1 < 0.2
    assert conn_matrix[1, 2] == 1  # 0.2 >= 0.2

    # Empty/Single
    single = np.array([[1.0]])
    conn = _create_connectivity_matrix(single)
    assert np.allclose(conn, np.eye(1))


def test_get_traces_and_analysis_for_run() -> None:
    # Setup objects
    roi = ROI(id=1, fov_id=1, label_value=1)
    trace1 = Traces(roi_id=1, analysis_result_id=10)
    trace2 = Traces(roi_id=1, analysis_result_id=20)
    roi.traces_history = [trace1, trace2]

    da1 = DataAnalysis(roi_id=1, analysis_result_id=10)
    da2 = DataAnalysis(roi_id=1, analysis_result_id=20)
    roi.data_analysis_history = [da1, da2]

    # Test _get_traces_for_run
    assert _get_traces_for_run(roi, 10) == trace1
    assert _get_traces_for_run(roi, 20) == trace2
    assert _get_traces_for_run(roi, 99) == trace1  # Fallback
    assert _get_traces_for_run(roi, None) == trace1

    # Test _get_data_analysis_for_run
    assert _get_data_analysis_for_run(roi, 10) == da1
    assert _get_data_analysis_for_run(roi, 20) == da2
    assert _get_data_analysis_for_run(roi, 99) == da1  # Fallback
    assert _get_data_analysis_for_run(roi, None) == da1

    # Empty history
    roi_empty = ROI(id=2, fov_id=1, label_value=2)
    assert _get_traces_for_run(roi_empty, 10) is None
    assert _get_data_analysis_for_run(roi_empty, 10) is None


def test_get_stimulated_amplitudes_from_roi_data() -> None:
    from cali.plot._util import get_stimulated_amplitudes_from_roi_data
    from cali.sqlmodel._util import ROIData

    # Mock ROIData
    roi_data = ROIData(
        well_fov_position="test",
        active=True,
        stimulated=True,
        evoked_experiment=True,
        dec_dff=[0.1, 0.5, 0.2, 0.8, 0.3, 0.6],
        peaks_dec_dff=[1, 3, 5],
        stimulations_frames_and_powers={"2": 50},
        led_pulse_duration="10ms",
    )

    stim, non_stim = get_stimulated_amplitudes_from_roi_data(roi_data)
    assert "50%_10ms" in stim
    assert stim["50%_10ms"] == [0.8]
    assert len(non_stim) == 0

    # Test with power equation
    def power_eq(x: float) -> float:
        return x * 2

    stim, non_stim = get_stimulated_amplitudes_from_roi_data(
        roi_data, led_power_equation=power_eq
    )
    # 50 * 2 = 100.000
    # MWCM is imported from constants, let's assume it's "mW/cm²" or similar
    # The key format is f"{power_val:.3f}{MWCM}_{led_pulse_duration}"
    # We can check if any key contains "100.000"
    key = next(k for k in stim.keys() if "100.000" in k)
    assert key is not None
    assert stim[key] == [0.8]

    # Test missing data
    roi_data_empty = ROIData(well_fov_position="empty")
    stim, non_stim = get_stimulated_amplitudes_from_roi_data(roi_data_empty)
    assert stim == {}
    assert non_stim == {}


@patch("cali.plot._util.Session")
def test_get_spikes_over_threshold(mock_session_cls: MagicMock) -> None:
    # Mock engine
    engine = MagicMock()

    # Mock session and query result
    mock_session = MagicMock()
    mock_exec = MagicMock()
    mock_first = MagicMock()

    # Setup ROI with DataAnalysis and Traces
    da = DataAnalysis(inferred_spikes_threshold=0.5)
    trace = Traces(inferred_spikes=[0.1, 0.8, 0.2, 0.9])
    roi = ROI(id=1, fov_id=1, label_value=1)
    roi.data_analysis_history = [da]
    roi.traces_history = [trace]

    mock_first.return_value = roi
    mock_exec.all.return_value = [roi]
    mock_exec.first = mock_first
    mock_session.exec.return_value = mock_exec
    mock_session.__enter__.return_value = mock_session

    # Configure the mock class to return our mock session
    mock_session_cls.return_value = mock_session

    # Test raw=False
    spikes = _get_spikes_over_threshold(engine, "fov1", 1, raw=False)
    assert spikes is not None
    # [0.1, 0.8, 0.2, 0.9] threshold 0.5 -> [0.0, 0.8, 0.0, 0.9]
    assert spikes == [0.0, 0.8, 0.0, 0.9]

    # Test raw=True
    spikes_raw = _get_spikes_over_threshold(engine, "fov1", 1, raw=True)
    assert spikes_raw == [0.1, 0.8, 0.2, 0.9]

    # Test ROI not found
    mock_first.return_value = None
    assert _get_spikes_over_threshold(engine, "fov1", 99) is None


@patch("cali.plot._util.Session")
def test_get_calcium_peaks_events_from_rois(mock_session_cls: MagicMock) -> None:
    # Mock engine
    engine = MagicMock()

    # Mock session
    mock_session = MagicMock()
    mock_exec = MagicMock()

    # Setup data
    # ROI 1: peaks at 1, 3. Max frame 5.
    roi1 = ROI(id=1, fov_id=1, label_value=1, active=True)
    traces1 = Traces(corrected_trace=[0.0] * 5)
    da1 = DataAnalysis(peaks_dec_dff=[1, 3])

    # ROI 2: peaks at 2, 4. Max frame 5.
    roi2 = ROI(id=2, fov_id=1, label_value=2, active=True)
    traces2 = Traces(corrected_trace=[0.0] * 5)
    da2 = DataAnalysis(peaks_dec_dff=[2, 4])

    # Return list of tuples (ROI, Traces, DataAnalysis)
    results = [(roi1, traces1, da1), (roi2, traces2, da2)]

    mock_exec.all.return_value = results
    mock_session.exec.return_value = mock_exec
    mock_session.__enter__.return_value = mock_session

    # Configure the mock class to return our mock session
    mock_session_cls.return_value = mock_session

    # Test
    events = _get_calcium_peaks_events_from_rois(engine, "fov1", run_id=1)
    assert events is not None
    assert len(events) == 2
    assert "1" in events
    assert "2" in events

    # Check event trains
    # ROI 1: peaks at 1, 3 -> [0, 1, 0, 1, 0]
    assert np.allclose(events["1"], [0, 1, 0, 1, 0])
    # ROI 2: peaks at 2, 4 -> [0, 0, 1, 0, 1]
    assert np.allclose(events["2"], [0, 0, 1, 0, 1])

    # Test with too few ROIs
    mock_exec.all.return_value = [(roi1, traces1, da1)]
    assert _get_calcium_peaks_events_from_rois(engine, "fov1", run_id=1) is None


def test_calculate_jitter_window_synchrony() -> None:
    from cali.plot._util import _calculate_jitter_window_synchrony

    # Perfect sync
    events_i = np.array([1.0, 0.0, 1.0])
    events_j = np.array([1.0, 0.0, 1.0])
    score = _calculate_jitter_window_synchrony(events_i, events_j, jitter_window=0)
    assert np.isclose(score, 1.0)

    # Jittered sync
    events_i = np.array([1.0, 0.0, 0.0])
    events_j = np.array([0.0, 1.0, 0.0])
    score = _calculate_jitter_window_synchrony(events_i, events_j, jitter_window=1)
    assert score > 0.0


def test_equation_from_str_invalid() -> None:
    from cali.plot._util import equation_from_str

    # Invalid format
    assert equation_from_str("invalid equation") is None


@patch("cali.plot._util.Session")
def test_get_calcium_peaks_events_from_rois_extra_cases(
    mock_session_cls: MagicMock,
) -> None:
    # Mock engine
    engine = MagicMock()
    mock_session = MagicMock()
    mock_exec = MagicMock()

    # Setup data
    roi1 = ROI(id=1, fov_id=1, label_value=1, active=True)
    traces1 = Traces(corrected_trace=[0.0] * 5)
    da1 = DataAnalysis(peaks_dec_dff=[1])

    roi2 = ROI(id=2, fov_id=1, label_value=2, active=True)
    traces2 = Traces(corrected_trace=[0.0] * 5)
    da2 = DataAnalysis(peaks_dec_dff=[2])

    results = [(roi1, traces1, da1), (roi2, traces2, da2)]

    mock_exec.all.return_value = results
    mock_session.exec.return_value = mock_exec
    mock_session.__enter__.return_value = mock_session
    mock_session_cls.return_value = mock_session

    # Test run_id=None (logs warning)
    events = _get_calcium_peaks_events_from_rois(engine, "fov1", run_id=None)
    assert events is not None

    # Test with rois filtering
    events = _get_calcium_peaks_events_from_rois(engine, "fov1", run_id=1, rois=[1, 2])
    assert events is not None
