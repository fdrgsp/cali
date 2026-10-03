import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from oasis.functions import GetSn, deconvolve, estimate_parameters

from cali.extraction._extraction_runner import ExtractionRunner
from cali.extraction._frame_window import ExtractionFrameWindow
from cali.extraction._spike_inference import InferenceCancelled, OasisBackend
from cali.sqlmodel import FOV, ROI, AnalysisSettings, ExtractionSettings


def _source_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    data = rng.normal(100, 1, (256, 4, 4))
    for t in (30, 90, 155, 210):
        data[t:, :2, :2] += 25 * np.exp(-np.arange(256 - t) / 8)[:, None, None]
    mask = np.zeros((4, 4), bool)
    mask[:2, :2] = True
    return data, mask


@pytest.mark.parametrize("case_index", [0, 1])
def test_oasis_matches_pre_refactor_extraction_fixture(case_index: int) -> None:
    # Captured before moving OASIS out of _process_roi_trace. Covers estimated
    # and fixed AR coefficients, neuropil correction, inline metrics and flags.
    cases = json.loads(
        (Path(__file__).parent / "fixtures/oasis_extraction_baseline.json").read_text()
    )
    expected = cases[case_index]
    data, mask = _source_data()
    settings = ExtractionSettings(
        decay_constant=expected["tau"], dff_window=5, frame_rate=10
    )
    runner = ExtractionRunner()
    parts = runner._compute_roi_dff(data, [], "baseline", settings, 7, mask, ~mask, 0.7)
    assert parts is not None
    result = OasisBackend().infer_all(
        parts.dff[None, :],
        256 / 25.5,
        decay_constant=settings.decay_constant,
        roi_labels=[7],
        fov_name="baseline",
    )
    actual = runner._finalize_roi(
        parts,
        result.den_dff[0],
        result.spikes[0],
        float(result.sn_by_roi[0]),
        AnalysisSettings(frame_rate=10),
        25.5,
        (np.arange(256) * 100.0).tolist(),
        "ms",
        ExtractionFrameWindow(0, 0, 256, 256, 0, "runner_time"),
    )
    assert actual is not None
    trace, analysis, active, stimulated, size, units = actual
    assert analysis is not None
    for name, value in expected["traces"].items():
        actual_value = getattr(trace, name)
        if isinstance(value, list):
            np.testing.assert_allclose(actual_value, value, rtol=0, atol=0)
        else:
            assert actual_value == value
    for name, value in expected["analysis"].items():
        assert getattr(analysis, name) == value
    assert (active, stimulated, size, units) == (
        expected["active"],
        expected["stimulated"],
        expected["size"],
        expected["units"],
    )
    if expected["tau"]:
        g = (float(np.exp(-1 / ((256 / 25.5) * expected["tau"]))),)
        sn = GetSn(parts.dff, range_ff=[0.25, 0.5], method="median")
    else:
        g_arr, sn = estimate_parameters(
            parts.dff,
            p=1,
            range_ff=[0.25, 0.5],
            method="median",
            lags=10,
            fudge_factor=0.98,
        )
        g = tuple(np.atleast_1d(g_arr))
    np.testing.assert_allclose(result.sn_by_roi, [sn], rtol=0, atol=0)
    np.testing.assert_allclose(result.g_by_roi, [g], rtol=0, atol=0)


def test_oasis_estimation_and_deconvolution_fallbacks() -> None:
    dff = np.linspace(0, 1, 64)[None, :]
    original = deconvolve
    calls = []

    def first_call_fails(*args: object, **kwargs: object) -> tuple:
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("invalid AR")
        return original(*args, **kwargs)

    with (
        patch(
            "cali.extraction._spike_inference._oasis.estimate_parameters",
            side_effect=np.linalg.LinAlgError("singular"),
        ),
        patch(
            "cali.extraction._spike_inference._oasis.deconvolve",
            side_effect=first_call_fails,
        ),
    ):
        result = OasisBackend().infer_all(dff, 10)
    assert len(calls) == 2
    assert calls[1]["g"] == (0.95,)
    assert calls[0]["sn"] == calls[1]["sn"]
    np.testing.assert_array_equal(result.g_by_roi, [[0.95]])


def test_oasis_cancellation_between_rows() -> None:
    runner = ExtractionRunner()
    original = OasisBackend._infer_row
    count = 0

    def cancel_after_row(*args: object, **kwargs: object) -> tuple:
        nonlocal count
        result = original(*args, **kwargs)
        count += 1
        runner.cancel()
        return result

    with patch.object(
        OasisBackend, "_infer_row", side_effect=cancel_after_row, autospec=True
    ):
        with pytest.raises(InferenceCancelled):
            OasisBackend().infer_all(
                np.random.default_rng(1).normal(size=(3, 64)),
                10,
                cancel=runner._check_for_abort_requested,
            )
    assert count == 1


@pytest.mark.parametrize("cancel_phase", [None, "A", "B", "C"])
def test_fov_batches_once_and_attaches_only_complete_results(
    cancel_phase: str | None,
) -> None:
    data, mask = _source_data()
    dataset = MagicMock()
    dataset.isel.return_value = (
        data,
        [{"runner_time_ms": i * 100.0, "exposure_ms": 100.0} for i in range(len(data))],
    )
    fov = FOV(
        name="batch",
        position_index=0,
        fov_number=0,
        rois=[ROI(label_value=7), ROI(label_value=8)],
    )
    runner = ExtractionRunner()
    original_compute = runner._compute_roi_dff
    original_infer = OasisBackend().infer_all
    original_finalize = runner._finalize_roi

    def compute(*args: object, **kwargs: object) -> object:
        result = original_compute(*args, **kwargs)
        if cancel_phase == "A":
            runner.cancel()
        return result

    def infer(*args: object, **kwargs: object) -> object:
        if cancel_phase == "B":
            runner.cancel()
        return original_infer(*args, **kwargs)

    def finalize(*args: object, **kwargs: object) -> object:
        result = original_finalize(*args, **kwargs)
        if cancel_phase == "C":
            runner.cancel()
        return result

    with (
        patch.object(runner, "_get_label_mask", return_value={7: mask, 8: ~mask}),
        patch.object(runner, "_compute_roi_dff", side_effect=compute),
        patch(
            "cali.extraction._extraction_runner.OasisBackend.infer_all",
            side_effect=infer,
        ) as batch,
        patch.object(runner, "_finalize_roi", side_effect=finalize),
    ):
        result = runner._extract_trace_data_per_position(
            dataset,
            ExtractionSettings(dff_window=5, neuropil_inner_radius=0),
            None,
            fov,
        )
    if cancel_phase is None:
        assert result is fov
        batch.assert_called_once()
        assert batch.call_args.args[0].shape == (2, 256)
        assert all(len(roi._new_traces) == 1 for roi in fov.rois)
    else:
        assert result is None
        assert all(not hasattr(roi, "_new_traces") for roi in fov.rois)


def test_oasis_reports_coefficient_used_by_retry() -> None:
    dff = np.linspace(0, 1, 64)[None, :]
    with (
        patch(
            "cali.extraction._spike_inference._oasis.estimate_parameters",
            return_value=(np.array([1.2]), 0.05),
        ),
        patch(
            "cali.extraction._spike_inference._oasis.deconvolve",
            side_effect=[
                RuntimeError("invalid AR"),
                (np.zeros(64), np.zeros(64), 0, (0.95,), 0),
            ],
        ),
    ):
        result = OasisBackend().infer_all(dff, 10)
    np.testing.assert_array_equal(result.g_by_roi, [[0.95]])
    assert np.isfinite(result.den_dff).all()
