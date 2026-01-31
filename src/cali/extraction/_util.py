from __future__ import annotations

import numpy as np
from numba import njit


def calculate_dff(
    data: np.ndarray,
    window_sec: float = 10,
    frame_rate: float = 10.0,
    percentile: int = 10,
) -> np.ndarray:
    """Calculate the delta F/F using a sliding window and a percentile.

    Parameters
    ----------
    data : np.ndarray
        Array representing the fluorescence trace.
    window_sec : float
        Size of the moving window for the background calculation in seconds.
        Default is 10.0 seconds.
    frame_rate : float
        Acquisition frame rate in frames per second.
        Default is 10.0 fps (100ms exposure time).
    percentile : int
        Percentile to use for the background calculation. Default is 10.

    Returns
    -------
    np.ndarray
        Array representing the delta F/F.
    """
    # Convert window from seconds to frames
    # window_sec * frame_rate = window_frames
    window_frames = int(window_sec * frame_rate)
    # Ensure at least 1 frame
    window_frames = max(1, window_frames)

    dff: np.ndarray = np.array([])
    bg: np.ndarray = _calculate_bg_numba(data, window_frames, percentile)
    # make sure we don't divide by zero
    eps = np.finfo(float).eps
    bg_safe = np.maximum(bg, eps)
    dff = (data - bg_safe) / bg_safe
    return dff


@njit(cache=True)  # type: ignore
def _calculate_bg_numba(
    trace: np.ndarray, window: int, percentile: float
) -> np.ndarray:
    """
    Numba-accelerated rolling percentile to calculate background.

    It uses a centered sliding window to compute the specified percentile
    for each point in the trace.
    """
    T = trace.shape[0]
    half = window // 2
    bg = np.empty(T, dtype=np.float64)

    for t in range(T):
        start = t - half
        if start < 0:
            start = 0
        end = t + half + 1
        if end > T:
            end = T

        # slice window and copy into temp array
        size = end - start
        temp = np.empty(size, dtype=np.float64)
        for i in range(size):
            temp[i] = trace[start + i]

        # sort and pick percentile index
        temp.sort()
        # percentile in [0,100]; map to [0, size-1]
        k = round((percentile / 100.0) * (size - 1))
        bg[t] = temp[k]

    return bg


def _calculate_bg(data: np.ndarray, window: int, percentile: int = 10) -> np.ndarray:
    """
    Calculate the background using a moving window and a specified percentile.

    Parameters
    ----------
    data : np.ndarray
        Array representing the fluorescence trace.
    window : int
        Size of the moving window.
    percentile : int
        Percentile to use for the background calculation. Default is 10.

    Returns
    -------
    np.ndarray
        Array representing the background.
    """
    # Initialize background array
    background: np.ndarray = np.zeros_like(data)

    # Use a centered sliding window to calculate background from percentile
    # This provides symmetric context around each point and reduces edge artifacts
    for y in range(len(data)):
        start = max(0, y - window // 2)
        end = min(len(data), y + window // 2 + 1)
        lower_percentile = np.percentile(data[start:end], percentile)
        background[y] = lower_percentile

    return background


def get_iei(peaks: np.ndarray, elapsed_time_list_ms: list[float]) -> list[float] | None:
    """Calculate the interevent interval."""
    # if less than 2 peaks or framerate is negative
    if len(peaks) < 2 or len(elapsed_time_list_ms) <= 1:
        return None

    peaks_time_stamps = [elapsed_time_list_ms[i] for i in peaks]  # ms

    # calculate the difference in time between two consecutive peaks
    iei_ms = np.diff(np.array(peaks_time_stamps))  # ms

    return [float(iei_peak / 1000) for iei_peak in iei_ms]
