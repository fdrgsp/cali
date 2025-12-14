# cali

A Gui for Calcium Imaging Data Visualization, Segmentation and Analysis

[![CI](https://github.com/fdrgsp/cali/actions/workflows/ci.yml/badge.svg)](https://github.com/fdrgsp/cali/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/fdrgsp/cali/branch/main/graph/badge.svg)](https://codecov.io/gh/fdrgsp/cali)

[🚧 WIP 🚧]

`cali` is package that provides a gui to load calcium imaging timelapse data (1-photon neuronal cultures), segment neurons using Cellpose, extract and analyse fluorescence traces and visualize them. It was originally designed to work in combination with [micromanager-gui](https://github.com/fdrgsp/micromanager-gui), an open-source software to control microscopes through `Micro-Manager]` and [pymmcore-plus](https://github.com/pymmcore-plus).

## To Run

If you have [uv](https://docs.astral.sh/uv/) installed, you can run `cali` directly without installing it using:

`uvx "git+https://github.com/fdrgsp/cali"`

**Note:** Cellpose is an optional dependency. To use segmentation features, install with:

- `uvx -p 3.12 "git+https://github.com/fdrgsp/cali[cp4]"` for Cellpose 4.x (cellpose-sam) (use python 3.11 or greater)

- `uvx -p 3.12 "git+https://github.com/fdrgsp/cali[cp3]"` for Cellpose 3.x (use python 3.11 or greater)

## To install

### Using uv (pip)

Create a new virtual environment and install `cali` using `uv` (pip):

`uv pip install git+https://github.com/fdrgsp/cali`

To install with Cellpose:

- `uv pip install git+https://github.com/fdrgsp/cali[cp4]` for Cellpose 4.x (cellpose-sam)
- `uv pip install git+https://github.com/fdrgsp/cali[cp3]` for Cellpose 3.x

### NOTE

#### Building on macOS

If you encounter build errors with `oasis-deconv` (especially SDK-related errors), set these environment variables before installing:

```bash
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
export LDFLAGS="-L${SDKROOT}/usr/lib"
```

Then run your installation command.

## Overview

### File Formats

`cali` currently supports the following file formats:
- `tensorstore.zarr` and `ome.zarr` from [micromanager-gui](https://github.com/fdrgsp/micromanager-gui)
- 3D tiff stacks (t, y, x). In this case the gui will open a widget to assign each tiff file as a fov to a well plate. If no plate was used, options like 35mm dish are available (the gui requires a [useq-schema plate](https://github.com/pymmcore-plus/useq-schema/blob/772418587b06331eb20a27dc7061c12a68143bcd/src/useq/_plate.py#L35) in order to work).

### Open a File

In the GUI, go to `File -> Select Data Source...`. Two options are available:

- to re-open a previously saved `cali` project (`.cali` file), select the `From Database` tab and input the path to the database file and the path of the actual data within the database (either a tensorstore.zarr or ome.zarr or a folder of tiff files).
- to create a new project, select the `From Directories` tab and input the path to the data (either a tensorstore.zarr or ome.zarr or a folder of tiff files), the output path for the `cali` project database file and the name of the project (if you omit `.cali` it will be added automatically).

[screenshot of the file open dialog]

NOTE: If you are loading a folder of tiff files, a plate assignment widget will open where the plate type can be selected and each tiff file assigned (as fov) to a well. After confirming the plate assignment, the main `cali` window will open and next time the project can be opened directly from the database file.

[screenshot of the plate assignment dialog]

### Main Window

The main window contains the following sections:

- Left panel: shows the plate layout with wells and fovs ans the image viewer. Selecting a well/fov will update the image viewer to visualize the corresponding data. By double-clicking a well/fov, the full fov data can be visualize in a new [ndv](https://pyapp-kit.github.io/ndv/latest/) window.
- Center panel: contains tabs for the ROI detection, trace extraction and analysis steps as well as the visualization tab for displaying the results.
- Right panel: contains the list of *Runs* the user has performed: `cali` is structured in a way that each time the user chenge the pipeline settings and runs the analysis, a new "Run" is created so that the user can compare different settings and results. Selecting a run will update the center panel tabs to show the settings and results of that run.

[screenshot of the main window]

### Pipeline Tabs

Hovering over each parameter in the detection, extraction and analysis tabs will show a tooltip with a description of the parameter.

#### Detection Tab

The detection tab allows the user to set the paramanter that will be used to segment cells and thus select the ROIs for trace extraction. Currently, only Cellpose is supported as segmentation method. The user can set the Cellpose parameters and run the segmentation.

[screenshot of the detection tab]

#### Extraction Tab

The extraction tab allows the user to set the parameters for fluorescence trace extraction from the segmented ROIs. The parameteers include:

- **neuropil correction**: enable/disable and set the neuropil mask parameters
- **DF/F and OASIS deconvolution**: set the window size for DF/F calculation and the parameters for OASIS deconvolution (or leave on `auto` to use default parameters).
- **Metadata**: set the frame rate and pixel size
- **Number of Threads**: set the number of threads to use for parallel extraction across wells/fovs. Keep tis number low if you experience memory issues during extraction.

[screenshot of the extraction tab]

#### Analysis Tab

The analysis tab allows the user to set parameters for further analysis of the extracted traces, including:

- **Experiment Type**: since `cali` was designed to work with [micromanager-gui](https://github.com/fdrgsp/micromanager-gui), which supports spatio-temporal optogenetic stimulation, the user can select the `Evoked Activity` experiment type that will allow to input the stimulation metadata used during the acquisition to then divide the analysis results between stimulate vs non-stimulated ROIs.
- **Calcium Traces and Peaks Analysis**: set parameters for calcium peak detection and analysis of calcium traces.
- **Inferred Spikes**: set parameters for analysis of inferred spikes obtained from OASIS deconvolution. This includes detection thresholds, parameters to detect bursts of spikes and parameter for correlation analysis between ROIs.
- **Metadata**: set additional metadata for the experiment (e.g. frame-rate, which is directly linked to the one input in the extraction tab, changing one will auto-change the other).
- **Number of Threads**: set the number of threads to use for parallel extraction across wells/fovs. Keep tis number low if you experience memory issues during extraction.

[screenshot of the analysis tab]

### Run the Pipeline

After setting the parameters in each tab, the user can run the pipeline using the run pipeline panel at the center bottom of the main window.

[screenshot of the run pipeline panel]

The user can select which steps to run (detection, extraction, analysis) using the `Run Options` dropdown and then click on the `Run` button. This will run the selected pipeline on all the wells/fovs in the plate and create a new *Run* in the run panel on the right.

If the user wants to first explore and optimize the parameters, a subset of wells/fovs can be selected in the `Positions to Extract` field: by inserting a comma-separated list of well/fov position index (which can be obained looking at the fov name in the FOV table under the plate layout - e.g. 1, 3, 4 or 3-7 for a range of fovs), only those positions will be processed. This is useful to quickly test and optimize the parameters before running the full pipeline on all positions. Once the user is satisfied with the parameters, they can clear the `Positions to Extract` field and run the pipeline on the full dataset.

The segmentation results and neuropil masks (if any) will be displayed in the image viewer by clicking on the "Labels, the rest of the results can be visualized in the Visualization tab.

### Visualization Tab

The visualization tab allows the user to explore the results of the analysis for the selected *Run*.

Two tabs are available, the `Single Well` tab to visualize the results for a single well/fov and the `Multi well` tab to visualize summary metrics across all wells/fovs.

The plots are interactive and can be zoomed/panned and by clicking on a trace or data point, the corresponding ROI will be highlighted in the image viewer (and vice versa).

#### Single Well Tab

Several options are available to visualize the data, including:

- **Calcium Traces**: raw, DF/F, Deconvolved DF/F and neuropil traces for each ROI and detected calcium peaks.
- **Inferred Spikes**: raw and thresholded inferred spike trains from OASIS deconvolution.
- **Raster Plots**: raster plots of both calcium peaks and inferred spikes across all ROIs.
- **Calcium metrics**: amplitude, frequency and other metrics for each ROI.
- **Calcium and Inferred Spikes Bursts**: burst metrics on calcium and inferred spikes.
- **Correlation Metrics**: pairwise pearson correlation on calcium traces and inferred spikes traces as well as jitter synchrony and max-lag cross-correlation on inferred spikes.
- **Stimulated vs Non-Stimulated Analysis**: if the `Evoked Activity` experiment type was selected, the user can visualize and compare the metrics between stimulated and non-stimulated ROIs.

[screenshot of the single well visualization tab]

#### Multi Well Tab

The multi well tab allows the user to visualize summary metrics across all wells/fovs in the plate. If the user trated the well plate with different conditions (e.g. drug vs control), the user can click on the `Shoe/Edit Plate Map` button under the plate layout to open a plate map editor where each well can be assigned to a condition. Currently, only two conditions are supported (e.g. genotype and treatment). This information will then be used to group the wells/fovs in the multi well visualization tab. If none is provided, data will be shown in a per-well basis.

[screenshot of the multi well visualization tab]

## Analysis Details

### Extraction

#### DF/F Calculation

**Purpose**: ΔF/F (Delta F over F) is a standard fluorescence normalization method in calcium imaging that represents relative changes in fluorescence intensity. This normalization accounts for baseline differences in fluorescence between cells and provides a measure of relative activity.

**Calculation**:

$\Delta F/F(t) = \frac{F(t) - F_0(t)}{F_0(t)}$

where:

- $F(t)$ is the raw fluorescence at time $t$
- $F_0(t)$ is the baseline fluorescence estimated from a sliding window

The baseline $F_0(t)$ is computed by taking the 10th percentile of the fluorescence within a sliding window centered at each time point.

**GUI Parameters**:

- **Window Size** (ms): Size of the sliding window for baseline calculation

#### OASIS Deconvolution

**Purpose**: This implementation uses the [OASIS algorithm](https://github.com/j-friedrich/OASIS) (Friedrich et al., 2017) to deconvolve the calcium signal (ΔF/F) to infer the underlying spike activity.

`OASIS` is used on each ROI to estimate the ΔF/F calcium traces noise level (later used for calcium peaks detection) and to both obtain a deconvolved (denoised) calcium trace and an inferred spike train trace.

**GUI Parameters**:
Currently, only the following parameters are exposed in the GUI:

- **Decay Constant** ($\tau$, seconds): Time constant for calcium decay. If set to 0 (auto), `OASIS` will estimate it from the data.

The other `OASIS` parameters are for now set to default values:

- for noise estimation:

  - **AR Model**: 1 (first-order autoregressive model)
  - **Method**: median
  - **Lags**: 10
  - **Fudge Factor**: 0.98

- for deconvolution:

   - **Penalty**: 1 (L1 penalty for spike inference)

### Analysis

#### Calcium Peak Detection

**Purpose**: Identify significant calcium transients (peaks) in the deconvolved ΔF/F trace.

**Calculation**: Peaks are detected using [scipy.signal.find_peaks](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html).

We use height, prominence and minimum distance thresholds to identify peaks.

There are two modes to determine the height thresholds:

- **MULTIPLIER**: height threshold is computed dynamically for each ROI as a multiple of the noise level estimated during OASIS deconvolution (recommended).

- **GLOBAL**: fixed absolute height value specified by the user. The exact same value is used for all ROIs in all wells/fovs. This can be useful mainly for testing purposes as it does not adapt to different noise levels across ROIs.

The prominence threshold is always computed as a multiple of the noise level estimated during OASIS deconvolution.

Minimum distance between peaks is specified in milliseconds and determines how close in time two peaks can be to be considered separate events.

After detection, the following metrics are computed for each ROI:

- Peak amplitudes (Deconvolved ΔF/F values at peak locations (a.u.))
- Calcium Peaks Event frequency: number of peaks per second
- Inter-event intervals (IEI): time between consecutive peaks

After detection, the following metrics are computed for each ROI:

- Peak amplitudes (Deconvolved ΔF/F values at peak locations (a.u.))
- Calcium Peaks Event frequency: number of peaks per second
- Inter-event intervals (IEI): time between consecutive peaks

**GUI Parameters**:

- **Height Mode**:  MULTIPLIER (× noise level) or GLOBAL (absolute value)
- **Height Value**: Threshold for peak amplitude (value * noise if MULTIPLIER mode, absolute value if GLOBAL)
- **Prominence Multiplier**: Minimum prominence relative to noise
- **Min Distance** (ms): Minimum time between consecutive peaks

#### Calcium Burst Detection

**Purpose**: Detect periods of sustained elevated population activity in calcium signals. Bursts represent synchronized network events where many cells are co-active.

**Calculation**: Burst detection operates on population-level activity:

1. **Population Activity**: Compute mean deconvolved ΔF/F across all active ROIs
2. **Smoothing**: Apply Gaussian filter to reduce noise (optional))
3. **Threshold**: Detect periods where smoothed activity exceeds a fraction of maximum
4. **Duration Filter**: Keep only bursts lasting at least minimum duration

**GUI Parameters**:

- **Burst Threshold** (%): Percentage of maximum smoothed activity
- **Min Duration** (ms): Minimum burst duration to retain
- **Gaussian Sigma** (s): Smoothing parameter for population activity

**Computed Metrics**:

- Burst count
- Average burst duration
- Average inter-burst interval
- Burst onset/offset times

#### Inferred Spikes Thresholding

**Purpose**: Convert continuous spike probability traces from OASIS into binary spike trains by applying an adaptive threshold.

**Calculation**: Threshold is computed adaptively based on the distribution of positive (non-zero) spike values:

There are two modes to determine the spike threshold:

- **MULTIPLIER**: threshold is computed dynamically for each ROI as a multiple of the noise level estimated as Median Absolute Deviation (MAD) of the positive spike values (recommended).

- **GLOBAL**: fixed absolute threshold value specified by the user. The exact same value is used for all ROIs in all wells/fovs. This can be useful mainly for testing purposes as it does not adapt to different noise levels across ROIs.

**GUI Parameters**:

- **Threshold Mode**: MULTIPLIER (× MAD estimated noise level) or GLOBAL (absolute value)
- **Spike Threshold Value**: Threshold for spike detection (value * noise if MULTIPLIER mode, absolute value if GLOBAL)

#### Inferred Spikes Burst Detection

**Purpose**: Detect periods of sustained elevated population spiking activity. Spike bursts represent synchronized network firing events.

**Calculation**: Similar to calcium burst detection, but operates on binary spike trains:

1. **Population Spike Rate**: Compute fraction of active ROIs per frame
2. **Smoothing**: Apply Gaussian filter with standard deviation $\sigma$ (optional)
3. **Threshold**: Detect periods where smoothed rate exceeds percentage threshold
4. **Duration Filter**: Keep bursts lasting at least minimum duration

**GUI Parameters**:

- **Burst Threshold** (%): Percentage of ROIs that must be active
- **Min Duration** (ms): Minimum burst duration
- **Gaussian Sigma** (s): Smoothing parameter

**Computed Metrics**:

- Number of network bursts
- Average burst duration
- Average inter-burst interval
- Population firing rate during bursts

#### Inferred Spikes Max-Lag Cross-Correlation

**Purpose**: Quantify temporal relationships between spike trains by computing cross-correlograms (CCGs).

**Calculation**:

1. **Input**: Two binary spike trains (arrays of 0s and 1s where 1 = spike, 0 = no spike)
2. **For each lag**: Shift one spike train relative to the other by ± lag frames and compute normalized dot product (spike coincidence count, normalized by the geometric mean of spike counts)
3. **Find maximum**: Return the correlation value and lag that gives the highest correlation

**Output**: Two heatmaps are generated:

- **Correlation Matrix**: Maximum correlation values (range: 0 to 1, where 1 = perfect synchrony at optimal lag, 0 = no temporal relationship)
- **Lag Matrix**: Lag values in frames (± frame shifts where maximum correlation occurs)
   - Positive lag: ROI j spikes lag behind ROI i
   - Negative lag: ROI j spikes lead ROI i
   - Lag = 0: Synchronous activity

**GUI Parameters**:

- **Max Lag** (ms): Maximum time offset to search

**Summary Metric**: Global synchrony = median of row means (excluding diagonal)

#### Inferred Spikes Jitter Synchrony

**Purpose**: Measure synchrony between spike trains within a temporal tolerance window.

**Calculation**: For each pair of ROIs, compute bidirectional jitter synchrony:

1. Input: Two binary spike trains (arrays of 0s and 1s representing spike times).
2. For each spike in neuron i, check whether neuron j fires within a small time tolerance window (e.g., ±2 frames). If yes, count this as a coincident spike.
3. Repeat in the opposite direction: For each spike in neuron j, check whether neuron i fires within the same tolerance window.
4. Combine coincidences: add the coincidences from both directions.
5. Normalize: divide the total coincidences by the total number of spikes across both neurons. This yields a synchrony score between 0 and 1, where:
   - 0 → no spikes occur near each other
   - 1 → every spike from both neurons has a partner within the jitter window

**GUI Parameters**:

- **Jitter Window** (ms): Temporal tolerance for spike coincidence

**Summary Metric**: Global synchrony = median of row means (excluding diagonal)
