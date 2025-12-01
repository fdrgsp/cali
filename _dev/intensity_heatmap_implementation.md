# Intensity Heatmap Plot Implementation

## Summary

Added two new intensity heatmap plots to visualize full trace data color-coded by signal intensity. These complement the existing raster plots which only show discrete peaks.

## New Plots Added

### 1. Calcium Intensity Heatmap (Deconvolved ΔF/F)
- **File**: `_plot_calcium_peaks_raster_plots.py`
- **Function**: `_generate_intensity_heatmap()`
- **Data Source**: `traces.dec_dff` (full deconvolved ΔF/F trace)
- **Visualization**: Each ROI is displayed as a horizontal row, with color representing signal intensity
- **Colormap**: Viridis (5th to 95th percentile normalization for outlier robustness)
- **Colorbar**: Shows intensity range in "dec ΔF/F" units

### 2. Inferred Spikes Intensity Heatmap
- **File**: `_plot_inferred_spike_raster_plots.py`
- **Function**: `_generate_spike_intensity_heatmap()`
- **Data Source**: `traces.inferred_spikes` (full inferred spike signal)
- **Visualization**: Each ROI as a row, color-coded spike intensity
- **Colormap**: Viridis (5th to 95th percentile normalization)
- **Colorbar**: Shows spike intensity range

## Key Features

### Robust Normalization
Uses percentile-based normalization instead of min/max to handle outliers:
```python
vmin = float(np.percentile(traces_array, 5))
vmax = float(np.percentile(traces_array, 95))
```

### Efficient Data Handling
- Stacks all traces into 2D numpy array: `(n_rois × n_frames)`
- Uses PyQtGraph ImageItem for fast rendering
- Automatic colorbar with viridis colormap

### Interactive Features
- **Click to select**: Click on any row to select that ROI
- **Time axis**: Shows time in seconds (if recording time available) or frames
- **ROI ordering**: Ordered by `label_value` for consistency with other plots

### Error Handling
- Graceful handling of missing data
- Clear error messages when no trace data available
- Uses `outerjoin` for DataAnalysis (not all runs may have analysis)

## Implementation Details

### Database Query Pattern
Both functions use optimized join queries:
```python
select(ROI, Traces, DataAnalysis)
    .join(FOV, ROI.fov_id == FOV.id)
    .join(Traces, ...)
    .outerjoin(DataAnalysis, ...)  # Optional for analysis metadata
    .where(col(FOV.name) == fov_name)
    .order_by(col(ROI.label_value))
```

### Heatmap Creation
```python
# Stack traces into 2D array
traces_array = np.vstack(traces_list)  # (n_rois, n_frames)

# Create PyQtGraph ImageItem
img = pg.ImageItem(traces_array)

# Apply viridis colormap with percentile normalization
cmap = pg.colormap.get("viridis")
img.setLookupTable(cmap.getLookupTable(vmin, vmax, 256))
img.setLevels((vmin, vmax))
```

### Click Handler
```python
def _attach_click_handlers_intensity(widget, plot, active_roi_labels):
    """Map clicked Y row → ROI label."""
    def _on_mouse_clicked(ev):
        p = vb.mapSceneToView(ev.scenePos())
        y = float(p.y())
        idx = round(y)
        if 0 <= idx < len(active_roi_labels):
            widget.roiSelected.emit(str(active_roi_labels[idx]))
```

## Registration

### Constants Added
```python
INTENSITY_HEATMAP = "Calcium Intensity Heatmap (Deconvolved ΔF/F)"
SPIKE_INTENSITY_HEATMAP = "Inferred Spikes Intensity Heatmap"
```

### Analysis Products
Both plots registered in `_main_plot.py`:
```python
AnalysisProduct(
    name=INTENSITY_HEATMAP,
    group=AnalysisGroup.SINGLE_WELL,
    analyzer=_generate_intensity_heatmap,
    category="Raster Plots",
    pipeline_stage=PipelineStage.ANALYSIS,
)
```

## Testing

### Test Coverage
- ✅ All 95 tests passing (added 2 new tests automatically via parametrization)
- ✅ Both plots render without errors
- ✅ ROI subset functionality works correctly
- ✅ Click-to-select ROI functionality verified

### Test Command
```bash
pytest tests/test_plot.py -k "Intensity" -xvs
# Output: 2 passed in 1.14s
```

## Comparison with Raster Plots

| Feature | Raster Plot | Intensity Heatmap |
|---------|-------------|-------------------|
| **Data Shown** | Discrete peaks only | Full continuous trace |
| **Visualization** | Scatter points | Continuous heatmap |
| **Temporal Resolution** | Peak timing only | Frame-by-frame intensity |
| **Use Case** | Event timing analysis | Signal dynamics visualization |
| **Color Encoding** | Peak amplitude (optional) | Trace intensity (always) |
| **Colorbar** | Only with amplitude coloring | Always present |

## Use Cases

### Calcium Intensity Heatmap
- **Visualize calcium dynamics** across multiple ROIs simultaneously
- **Identify coordinated activity** patterns visually
- **Quality control** - spot artifacts, baseline drift, bleaching
- **Temporal patterns** - see when activity increases/decreases

### Spike Intensity Heatmap
- **Spike train visualization** - see inferred spike patterns
- **Compare spike magnitudes** across ROIs
- **Network activity** - identify bursts or synchronized spiking
- **Pre/post-threshold comparison** - verify spike detection quality

## Files Modified

1. **`src/cali/plot/_main_plot.py`**
   - Added constants: `INTENSITY_HEATMAP`, `SPIKE_INTENSITY_HEATMAP`
   - Added imports: `_generate_intensity_heatmap`, `_generate_spike_intensity_heatmap`
   - Registered 2 new analysis products

2. **`src/cali/plot/_single_wells_plots/raster/_plot_calcium_peaks_raster_plots.py`**
   - Added `_generate_intensity_heatmap()` function (~180 lines)
   - Added `_add_intensity_colorbar_to_widget()` helper
   - Added `_attach_click_handlers_intensity()` helper

3. **`src/cali/plot/_single_wells_plots/raster/_plot_inferred_spike_raster_plots.py`**
   - Added `_generate_spike_intensity_heatmap()` function (~190 lines)
   - Added `_add_spike_intensity_colorbar_to_widget()` helper
   - Added `_attach_click_handlers_spike_intensity()` helper

## Code Quality

- ✅ **Ruff**: All checks passed
- ✅ **Tests**: 95/95 passing (2.12s)
- ✅ **Type hints**: Fully typed with proper annotations
- ✅ **Documentation**: Comprehensive docstrings
- ✅ **Error handling**: Graceful degradation with clear messages
- ✅ **Consistency**: Follows existing code patterns and conventions

## Future Enhancements (Optional)

1. **Alternative colormaps**: Add option to choose different colormaps (hot, plasma, etc.)
2. **Normalization options**: Per-ROI vs global normalization toggle
3. **Zoom/pan synchronization**: Link multiple heatmaps for comparison
4. **Export functionality**: Save heatmap as image or data matrix
5. **Overlay annotations**: Mark stimulus times, events, or regions of interest
