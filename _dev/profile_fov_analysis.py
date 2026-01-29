"""Profile compute_fov_analysis to identify performance bottlenecks."""

import time
from pathlib import Path

import numpy as np
from sqlmodel import Session, create_engine, select

from cali.sqlmodel._model import FOV, Experiment, AnalysisSettings


def profile_compute_fov_analysis(database_path: str, fov_names: list[str]):
    """Profile FOV analysis computation with detailed timing.

    Parameters
    ----------
    database_path : str
        Path to the .cali database
    fov_names : list[str]
        Names of FOVs to profile (e.g., ['p0', 'p4'])
    """
    engine = create_engine(
        f"sqlite:///{database_path}",
        connect_args={"timeout": 30.0, "check_same_thread": False},
        pool_pre_ping=True,
    )

    with Session(engine) as session:
        # Get the experiment and analysis settings
        experiment = session.exec(select(Experiment)).first()
        if experiment is None:
            print("No experiment found in database")
            return

        analysis_settings = session.exec(select(AnalysisSettings)).first()
        if analysis_settings is None:
            print("No analysis settings found")
            return

        print(f"Analysis Settings:")
        print(f"  Frame rate: {analysis_settings.frame_rate} fps")
        print(f"  CCG max lag: {analysis_settings.spikes_sync_cross_corr_lag} ms")
        print(f"  CCG n_shuffles: {analysis_settings.ccg_n_shuffles}")
        print(f"  Jitter window: {analysis_settings.spikes_sync_jitter_window} ms")
        print(
            f"  Rising edge analysis: {analysis_settings.enable_rising_edge_analysis}"
        )
        print()

        for fov_name in fov_names:
            print(f"\n{'='*60}")
            print(f"Profiling FOV: {fov_name}")
            print(f"{'='*60}")

            # Get FOV
            fov = session.exec(select(FOV).where(FOV.name == fov_name)).first()
            if fov is None:
                print(f"FOV {fov_name} not found")
                continue

            # Count active ROIs
            active_rois = [roi for roi in fov.rois if roi.active]
            print(f"Active ROIs: {len(active_rois)}")

            if len(active_rois) < 2:
                print("Not enough active ROIs for analysis")
                continue

            # Import here to add timing instrumentation
            from cali.analysis._util import (
                _compute_zero_lag_corr_matrix,
                _detect_calcium_population_bursts,
                _detect_spikes_population_bursts,
                _get_spike_correlations_matrix,
                _get_spike_synchrony,
            )

            # Manually collect data and time each step
            roi_labels = []
            dff_traces = []
            dec_dff_traces = []
            spike_trains = []
            spike_data_dict = {}
            spike_data_dict_rising_edges = {}

            t0 = time.perf_counter()
            for roi in active_rois:
                if roi.label_value is None:
                    continue

                # Get traces
                traces = None
                if hasattr(roi, "_new_traces") and roi._new_traces:
                    traces = roi._new_traces[-1]
                elif roi.traces_history:
                    traces = roi.traces_history[-1]

                if traces is None or traces.dec_dff is None:
                    continue

                # Get analysis data
                data_analysis = None
                if hasattr(roi, "_new_data_analysis") and roi._new_data_analysis:
                    data_analysis = roi._new_data_analysis[-1]
                elif roi.data_analysis_history:
                    data_analysis = roi.data_analysis_history[-1]

                dff = np.asarray(traces.dff, dtype=float)
                if dff.ndim != 1 or dff.size == 0:
                    continue

                dec_dff = np.asarray(traces.dec_dff, dtype=float)
                if dec_dff.ndim != 1 or dec_dff.size == 0:
                    continue

                roi_labels.append(int(roi.label_value))
                dff_traces.append(dff)
                dec_dff_traces.append(dec_dff)

                # Build spike data
                if traces.inferred_spikes is not None:
                    spikes = np.asarray(traces.inferred_spikes, dtype=float)
                    spike_threshold = (
                        data_analysis.inferred_spikes_threshold
                        if data_analysis is not None
                        else None
                    )

                    if spike_threshold is not None:
                        spikes_binary = spikes.copy()
                        spikes_binary[spikes_binary <= spike_threshold] = 0.0
                        spike_train = (spikes_binary > 0.0).astype(float)
                        spike_trains.append(spike_train)
                        spike_data_dict[str(roi.label_value)] = spike_train.tolist()

                        # Rising edges
                        positive_vals = spike_train > 0
                        rising = positive_vals & ~np.concatenate(
                            ([False], positive_vals[:-1])
                        )
                        spike_train_rising_edges = np.zeros_like(spike_train, dtype=float)
                        spike_train_rising_edges[rising] = 1.0
                        spike_data_dict_rising_edges[str(roi.label_value)] = (
                            spike_train_rising_edges.tolist()
                        )

            t_data_collection = time.perf_counter() - t0
            print(f"\n1. Data collection: {t_data_collection:.4f} s")
            print(f"   - Valid ROIs: {len(roi_labels)}")
            print(f"   - Spike trains: {len(spike_trains)}")
            print(f"   - Trace length: {len(dff_traces[0]) if dff_traces else 0} frames")

            # Helper function
            def ms_to_frames(ms: float) -> int:
                return max(0, int((ms / 1000.0) * analysis_settings.frame_rate))

            # 1. Zero-lag correlation on dff
            t0 = time.perf_counter()
            calcium_dff_corr_matrix = _compute_zero_lag_corr_matrix(dff_traces)
            t_dff_corr = time.perf_counter() - t0
            print(f"\n2. Calcium dff correlation: {t_dff_corr:.4f} s")

            # 2. Zero-lag correlation on dec_dff
            t0 = time.perf_counter()
            calcium_dec_dff_corr_matrix = _compute_zero_lag_corr_matrix(dec_dff_traces)
            t_dec_dff_corr = time.perf_counter() - t0
            print(f"3. Calcium dec_dff correlation: {t_dec_dff_corr:.4f} s")

            if len(spike_data_dict) >= 2:
                # 3. Max lag correlation on spikes (CCG)
                max_lag_ms = analysis_settings.spikes_sync_cross_corr_lag
                max_lag_frames = ms_to_frames(max_lag_ms)
                n_shuffles = analysis_settings.ccg_n_shuffles
                print(
                    f"\n4. Spike CCG (max_lag={max_lag_frames} frames, "
                    f"n_shuffles={n_shuffles}):"
                )

                t0 = time.perf_counter()
                (
                    spike_max_lag_corr_matrix,
                    spike_max_lag_values_matrix,
                    spike_ccg_zscore_matrix,
                ) = _get_spike_correlations_matrix(
                    spike_data_dict,
                    method="cross_correlation",
                    max_lag=max_lag_frames,
                    n_shuffles=n_shuffles,
                )
                t_ccg = time.perf_counter() - t0
                print(f"   - CCG computation: {t_ccg:.4f} s")

                if spike_max_lag_corr_matrix is not None:
                    t0 = time.perf_counter()
                    global_spike_max_lag_corr = _get_spike_synchrony(
                        spike_max_lag_corr_matrix
                    )
                    t_sync = time.perf_counter() - t0
                    print(f"   - Global synchrony: {t_sync:.4f} s")

                # 3b. Rising edges CCG
                if (
                    analysis_settings.enable_rising_edge_analysis
                    and len(spike_data_dict_rising_edges) >= 2
                ):
                    print(f"\n5. Spike CCG (rising edges):")
                    t0 = time.perf_counter()
                    (
                        spike_max_lag_corr_matrix_rising_edges,
                        spike_max_lag_values_matrix_rising_edges,
                        spike_ccg_zscore_matrix_rising_edges,
                    ) = _get_spike_correlations_matrix(
                        spike_data_dict_rising_edges,
                        method="cross_correlation",
                        max_lag=max_lag_frames,
                        n_shuffles=n_shuffles,
                    )
                    t_ccg_rising = time.perf_counter() - t0
                    print(f"   - CCG computation: {t_ccg_rising:.4f} s")

                # 4. Jitter synchrony
                jitter_window_ms = analysis_settings.spikes_sync_jitter_window
                jitter_window_frames = ms_to_frames(jitter_window_ms)
                print(
                    f"\n6. Spike jitter synchrony "
                    f"(window={jitter_window_frames} frames):"
                )

                t0 = time.perf_counter()
                spike_jitter_sync_matrix, _, _ = _get_spike_correlations_matrix(
                    spike_data_dict,
                    method="jitter_window",
                    jitter_window=jitter_window_frames,
                )
                t_jitter = time.perf_counter() - t0
                print(f"   - Jitter computation: {t_jitter:.4f} s")

                # 4b. Jitter rising edges
                if (
                    analysis_settings.enable_rising_edge_analysis
                    and len(spike_data_dict_rising_edges) >= 2
                ):
                    print(f"\n7. Spike jitter synchrony (rising edges):")
                    t0 = time.perf_counter()
                    spike_jitter_sync_matrix_rising_edges, _, _ = (
                        _get_spike_correlations_matrix(
                            spike_data_dict_rising_edges,
                            method="jitter_window",
                            jitter_window=jitter_window_frames,
                        )
                    )
                    t_jitter_rising = time.perf_counter() - t0
                    print(f"   - Jitter computation: {t_jitter_rising:.4f} s")

            # 5. Burst detection (spikes)
            if len(spike_trains) >= 2:
                print(f"\n8. Spike burst detection:")
                t0 = time.perf_counter()
                (
                    spike_burst_count,
                    spike_burst_avg_duration,
                    spike_burst_avg_interval,
                    spike_burst_starts,
                    spike_burst_ends,
                    spike_population_activity,
                    spike_population_activity_raw,
                ) = _detect_spikes_population_bursts(
                    spike_trains=spike_trains,
                    frame_rate=analysis_settings.frame_rate,
                    burst_threshold_percent=analysis_settings.burst_threshold,
                    min_duration_ms=analysis_settings.burst_min_duration,
                    gaussian_sigma_sec=analysis_settings.burst_gaussian_sigma,
                )
                t_spike_burst = time.perf_counter() - t0
                print(f"   - Burst detection: {t_spike_burst:.4f} s")

            # Burst detection (calcium)
            if len(dec_dff_traces) >= 2:
                print(f"\n9. Calcium burst detection:")
                t0 = time.perf_counter()
                (
                    calcium_burst_count,
                    calcium_burst_avg_duration,
                    calcium_burst_avg_interval,
                    calcium_burst_starts,
                    calcium_burst_ends,
                    calcium_population_activity,
                    calcium_population_activity_raw,
                ) = _detect_calcium_population_bursts(
                    dec_dff_traces=dec_dff_traces,
                    frame_rate=analysis_settings.frame_rate,
                    burst_threshold_percent=analysis_settings.calcium_burst_threshold,
                    min_duration_ms=analysis_settings.calcium_burst_min_duration,
                    gaussian_sigma_sec=analysis_settings.calcium_burst_gaussian_sigma,
                )
                t_calcium_burst = time.perf_counter() - t0
                print(f"   - Burst detection: {t_calcium_burst:.4f} s")

            # Summary
            print(f"\n{'='*60}")
            print(f"TIMING SUMMARY for {fov_name}")
            print(f"{'='*60}")
            total_time = (
                t_data_collection
                + t_dff_corr
                + t_dec_dff_corr
                + (t_ccg if "t_ccg" in locals() else 0)
                + (t_ccg_rising if "t_ccg_rising" in locals() else 0)
                + (t_jitter if "t_jitter" in locals() else 0)
                + (t_jitter_rising if "t_jitter_rising" in locals() else 0)
                + (t_spike_burst if "t_spike_burst" in locals() else 0)
                + (t_calcium_burst if "t_calcium_burst" in locals() else 0)
            )

            def print_timing(name, t):
                pct = (t / total_time * 100) if total_time > 0 else 0
                print(f"{name:40s}: {t:8.4f} s ({pct:5.1f}%)")

            print_timing("Data collection", t_data_collection)
            print_timing("Calcium dff correlation", t_dff_corr)
            print_timing("Calcium dec_dff correlation", t_dec_dff_corr)
            if "t_ccg" in locals():
                print_timing("Spike CCG", t_ccg)
            if "t_ccg_rising" in locals():
                print_timing("Spike CCG (rising edges)", t_ccg_rising)
            if "t_jitter" in locals():
                print_timing("Spike jitter synchrony", t_jitter)
            if "t_jitter_rising" in locals():
                print_timing("Spike jitter synchrony (rising edges)", t_jitter_rising)
            if "t_spike_burst" in locals():
                print_timing("Spike burst detection", t_spike_burst)
            if "t_calcium_burst" in locals():
                print_timing("Calcium burst detection", t_calcium_burst)
            print(f"{'-'*60}")
            print_timing("TOTAL", total_time)


if __name__ == "__main__":
    database_path = "/Users/fdrgsp/Desktop/cali/results_new.cali"
    # Use the FOV names from the database (B2_0000 and B3_0000 are the ones with active ROIs)
    fov_names = ["B2_0000", "B3_0000"]

    profile_compute_fov_analysis(database_path, fov_names)
