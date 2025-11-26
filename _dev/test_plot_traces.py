"""Test script to debug trace plotting - mimics GUI plotting logic."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sqlalchemy.engine import create_engine
from sqlmodel import Session, col, select

from cali.sqlmodel._model import FOV, ROI, DataAnalysis, Traces

# Database path
db_path = Path(
    "/Volumes/T7 Shield/for FG/TSC_hSynLAM77_ACTX250730_D36/results.cali"
)

if not db_path.exists():
    print(f"❌ Database not found: {db_path}")
    exit(1)

# Test parameters (same as GUI)
fov_name = "B2_0000"
run_id = 1
rois = None  # Plot all ROIs
raw = False
dff = False
dec = False
normalize = False
with_peaks = False
active_only = False

print(f"Testing trace plotting for FOV: {fov_name}, run_id: {run_id}")
print("=" * 80)

# Create engine
engine = create_engine(f"sqlite:///{db_path}", echo=False)

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

COUNT_INCREMENT = 1
P1 = 5
P2 = 100

with Session(engine) as session:
    # Query database for ROI data (EXACT same query as GUI)
    stmt = (
        select(ROI, Traces, DataAnalysis)
        .join(FOV, ROI.fov_id == FOV.id)
        .join(
            Traces,
            (Traces.roi_id == ROI.id) & (Traces.analysis_result_id == run_id),
        )
        .outerjoin(
            DataAnalysis,
            (DataAnalysis.roi_id == ROI.id)
            & (DataAnalysis.analysis_result_id == run_id),
        )
        .where(col(FOV.name) == fov_name)
    )

    # Filter by specific ROIs if requested
    if rois is not None:
        stmt = stmt.where(col(ROI.label_value).in_(rois))

    # Filter by active if requested
    if active_only:
        stmt = stmt.where(col(ROI.active) == True)  # noqa: E712

    # Order by label_value for consistent plotting
    stmt = stmt.order_by(col(ROI.label_value))

    results = session.exec(stmt).all()
    roi_data = results

print(f"\n📊 Found {len(roi_data)} results")

if not roi_data:
    print("❌ No data to plot!")
    exit(1)

# Compute percentiles (same as GUI)
p1 = p2 = 0.0
if normalize:
    all_values = []
    for _, trace_obj, _ in roi_data:
        trace = None
        if trace_obj:
            if dff:
                trace = trace_obj.dff
            elif dec:
                trace = trace_obj.dec_dff
            elif raw:
                trace = trace_obj.raw_trace
            else:
                trace = trace_obj.corrected_trace

        if trace:
            all_values.extend(trace)

    if all_values:
        percentiles = np.percentile(all_values, [P1, P2])
        p1, p2 = float(percentiles[0]), float(percentiles[1])
    else:
        p1, p2 = 0.0, 1.0

count = 0
rois_rec_time = []
last_trace = None
traces_plotted = 0

print("\nProcessing traces:")
print("=" * 80)

for roi_model, trace_obj, data_analysis in roi_data:
    # Extract trace data (same logic as GUI)
    trace = None
    if trace_obj:
        if dff:
            trace = trace_obj.dff
        elif dec:
            trace = trace_obj.dec_dff
        elif raw:
            trace = trace_obj.raw_trace
        else:
            trace = trace_obj.corrected_trace

    if not trace:
        print(f"  ⚠️  ROI {roi_model.label_value}: No trace data!")
        continue

    print(
        f"  ✓ ROI {roi_model.label_value}: Trace length={len(trace)}, "
        f"type={type(trace)}"
    )

    # Get recording time from data_analysis if available
    if data_analysis and data_analysis.total_recording_time_sec is not None:
        rois_rec_time.append(data_analysis.total_recording_time_sec)

    # Plot trace (same as GUI)
    offset = count * 1.1  # vertical offset

    if normalize:
        tr = np.array(trace) if isinstance(trace, list) else trace
        denom = p2 - p1
        if denom == 0:
            normalized = np.zeros_like(tr)
        else:
            normalized = (tr - p1) / denom
            normalized = np.clip(normalized, 0, 1)
        trace_to_plot = normalized + offset
    else:
        trace_to_plot = trace

    ax.plot(trace_to_plot, label=f"ROI {roi_model.label_value}")
    traces_plotted += 1

    # Get peaks data from data_analysis if available
    if with_peaks and data_analysis and data_analysis.peaks_dec_dff:
        peaks_indices = [int(p) for p in data_analysis.peaks_dec_dff]
        ax.plot(peaks_indices, np.array(trace_to_plot)[peaks_indices], "x")

    last_trace = trace
    count += COUNT_INCREMENT

print(f"\n✅ Plotted {traces_plotted} traces")

# Set graph title and labels (same as GUI)
if dff:
    title = "Normalized Calcium Traces (ΔF/F)" if normalize else "Calcium Traces (ΔF/F)"
    y_lbl = "ROIs" if normalize else "ΔF/F"
elif dec:
    title = (
        "Normalized Calcium Traces (Deconvolved ΔF/F)"
        if normalize
        else "Calcium Traces (Deconvolved ΔF/F)"
    )
    y_lbl = "ROIs" if normalize else "Deconvolved ΔF/F"
else:
    title = "Normalized Calcium Traces" if normalize else "Raw Calcium Traces"
    y_lbl = "ROIs" if normalize else "Fluorescence Intensity"
if with_peaks:
    title += " with Peaks"

ax.set_title(title)
ax.set_ylabel(y_lbl)

# Update time axis (same as GUI)
if last_trace is None or sum(rois_rec_time) <= 0:
    ax.set_xlabel("Frames")
else:
    avg_rec_time = int(np.mean(rois_rec_time))
    total_frames = len(last_trace) if last_trace is not None else 1
    tick_interval = avg_rec_time / total_frames
    x_ticks = np.linspace(0, total_frames, num=5, dtype=int)
    x_labels = [str(int(t * tick_interval)) for t in x_ticks]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Time (s)")

fig.tight_layout()

# Save the plot
output_path = Path("_dev/test_plot_output.png")
plt.savefig(output_path, dpi=150)
print(f"\n💾 Plot saved to: {output_path}")

# Show the plot
plt.show()

engine.dispose()
print("\n✅ Test complete!")
