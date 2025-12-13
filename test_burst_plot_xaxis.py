"""Test script to check burst plot x-axis values."""

import sys
from pathlib import Path

import numpy as np
from sqlmodel import Session, create_engine, select

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cali.sqlmodel._model import FOV

# Open the database
db_path = "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/results_new.cali"
engine = create_engine(f"sqlite:///{db_path}")

print("=" * 80)
print("CHECKING BURST PLOT X-AXIS DATA")
print("=" * 80)

with Session(engine) as session:
    # Get all FOVs and find one with analysis
    stmt = select(FOV)
    fovs = session.exec(stmt).all()

    if not fovs:
        print("No FOVs found!")
        sys.exit(1)

    print(f"\nFound {len(fovs)} FOVs")

    # Find FOV with analysis
    fov = None
    for f in fovs:
        if f.fov_analysis_history:
            fov = f
            break

    if not fov:
        print("No FOV with analysis found!")
        sys.exit(1)

    print(f"Using FOV: {fov.name}")

    fov_analysis = fov.fov_analysis_history[-1]

    print(f"FOV Analysis ID: {fov_analysis.id}")
    print(f"Analysis Result ID: {fov_analysis.analysis_result_id}")

    # Check calcium burst data
    if fov_analysis.calcium_population_activity:
        pop_activity = np.array(fov_analysis.calcium_population_activity)
        print(f"\nCalcium population activity length: {len(pop_activity)}")
        print(
            f"Calcium population activity range: "
            f"[{pop_activity.min():.4f}, {pop_activity.max():.4f}]"
        )

        if fov_analysis.calcium_burst_starts:
            print(f"Number of calcium bursts: {len(fov_analysis.calcium_burst_starts)}")
            print(f"Burst start frames: {fov_analysis.calcium_burst_starts[:5]}...")
            print(f"Burst end frames: {fov_analysis.calcium_burst_ends[:5]}...")

    # Now check what time axis we would get
    print("\n" + "=" * 80)
    print("CHECKING TIME AXIS CALCULATION")
    print("=" * 80)

    # Get ROIs with traces
    if fov.rois:
        print(f"\nNumber of ROIs: {len(fov.rois)}")

        rois_with_traces = [r for r in fov.rois if r.traces_history]
        print(f"ROIs with traces: {len(rois_with_traces)}")

        if rois_with_traces:
            # Check first ROI's trace data
            roi = rois_with_traces[0]
            traces = roi.traces_history[-1]

            print(f"\nFirst ROI (label {roi.label_value}):")
            if traces.dec_dff:
                print(f"  dec_dff length: {len(traces.dec_dff)}")

            if traces.x_axis:
                print(f"  x_axis length: {len(traces.x_axis)}")
                print(f"  x_axis units: {traces.x_axis_units}")
                print(f"  x_axis range: [{traces.x_axis[0]}, {traces.x_axis[-1]}]")

                if traces.x_axis_units == "ms":
                    time_in_seconds = traces.x_axis[-1] / 1000.0
                    print(f"  Time in seconds: {time_in_seconds:.2f} s")
                else:
                    print(f"  Time (assuming 10 Hz): {traces.x_axis[-1] / 10.0:.2f} s")

            # Check data_analysis for recording time
            if roi.data_analysis_history:
                data_analysis = roi.data_analysis_history[-1]
                if data_analysis.total_recording_time_sec:
                    print(
                        f"  Recording time from data_analysis: "
                        f"{data_analysis.total_recording_time_sec:.2f} s"
                    )

        # Calculate what _get_population_calcium_data would return
        print("\n" + "=" * 80)
        print("SIMULATING _get_population_calcium_data TIME AXIS")
        print("=" * 80)

        rois_rec_time = []
        calcium_traces = []

        for roi in rois_with_traces[:10]:  # Check first 10
            traces_obj = roi.traces_history[-1]

            if traces_obj.dec_dff and len(traces_obj.dec_dff) > 0:
                calcium_traces.append(np.array(traces_obj.dec_dff))

                # Get recording time from data_analysis (NEW LOGIC)
                if roi.data_analysis_history:
                    data_analysis = roi.data_analysis_history[-1]
                    if data_analysis.total_recording_time_sec:
                        rois_rec_time.append(data_analysis.total_recording_time_sec)

        if calcium_traces:
            lengths = [len(t) for t in calcium_traces]
            max_length = max(lengths)

            print(
                f"\nTrace lengths: min={min(lengths)}, max={max_length}, "
                f"avg={np.mean(lengths):.1f}"
            )

            if rois_rec_time:
                avg_rec_time = float(np.mean(rois_rec_time))
                max_rec_time = max(rois_rec_time)
                time_axis = np.linspace(0, max_rec_time, max_length)

                print(f"\nRecording times found: {len(rois_rec_time)}")
                print(
                    f"Recording time range: "
                    f"[{min(rois_rec_time):.2f}, {max_rec_time:.2f}] s"
                )
                print(f"Average recording time: {avg_rec_time:.2f} s")
                print("\nTime axis created:")
                print(f"  Length: {len(time_axis)}")
                print(f"  Range: [{time_axis[0]:.2f}, {time_axis[-1]:.2f}] s")
                print("  Should be around 200 seconds!")
            else:
                print("\nNo recording times found - would use frame indices / 10.0")
                time_axis = np.arange(max_length) / 10.0
                print(f"Time axis: [{time_axis[0]:.2f}, {time_axis[-1]:.2f}] s")

print("\n" + "=" * 80)
print("Analysis complete!")
print("=" * 80)
