# cali

A Gui for Calcium Imaging Data Visualization, Segmentation and Analysis (🚧 WIP 🚧)

[![CI](https://github.com/fdrgsp/cali/actions/workflows/ci.yml/badge.svg)](https://github.com/fdrgsp/cali/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/fdrgsp/cali/branch/main/graph/badge.svg)](https://codecov.io/gh/fdrgsp/cali)

`cali` is package that provides a gui to load calcium imaging timelapse data (1-photon neuronal cultures), segment neurons using Cellpose, extract and analyse fluorescence traces and visualize them. It was originally designed to work in combination with [micromanager-gui](https://github.com/fdrgsp/micromanager-gui), an open-source software to control microscopes through `Micro-Manager]` and [pymmcore-plus](https://github.com/pymmcore-plus).

<img width="1736" height="1093" alt="Screenshot 2025-12-17 at 10 10 43 AM" src="https://github.com/user-attachments/assets/aac1188b-180e-49dc-b095-3d3a3e350750" />

## To Run

If you have [uv](https://docs.astral.sh/uv/) installed, you can run `cali` directly without installing it using:

`uvx "git+https://github.com/fdrgsp/cali"`

**Note:** Cellpose is an optional dependency. To use segmentation features, install with:

`uvx -p 3.12 "git+https://github.com/fdrgsp/cali[cp4]"` for Cellpose 4.x (cellpose-sam) (use python 3.11 or greater)

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

---

## Overview

### File Formats

`cali` currently supports the following file formats:

- `tensorstore.zarr` and `ome.zarr` from [micromanager-gui](https://github.com/fdrgsp/micromanager-gui)
- 3D TIFF stacks (`t, y, x`). In this case, the GUI opens a widget to assign each TIFF file as a FOV to a well in a plate. If no plate was used, options like a 35 mm dish are available. The GUI requires a [useq-schema plate](https://github.com/pymmcore-plus/useq-schema/blob/772418587b06331eb20a27dc7061c12a68143bcd/src/useq/_plate.py#L35) in order to work.

### Open a File

In the GUI, go to `File -> Select Data Source...`. Two options are available:

- To re-open a previously saved `cali` project (`.cali` file), select the **From Database** tab and input:
  - the path to the database file
  - the path of the actual data referenced in the database (either a `tensorstore.zarr`, `ome.zarr`, or a folder of TIFF files).

- To create a new project, select the **From Directories** tab and input:
  - the path to the data (either a `tensorstore.zarr`, `ome.zarr`, or a folder of TIFF files),
  - the output path for the `cali` project database file, and
  - the name of the project (if you omit `.cali`, it will be added automatically).

<img width="600" alt="Screenshot 2025-12-14 at 10 03 57 AM" src="https://github.com/user-attachments/assets/46e989b5-9c0e-451d-a5e0-98af878bb32b" />

**NOTE**: If you are loading a folder of TIFF files, a plate assignment widget will open where you can:

- select the plate type
- assign each TIFF file (as a FOV) to a well.

After confirming the plate assignment, the main `cali` window will open. Next time, the project can be loaded directly from the database file.

<img width="800" alt="Screenshot 2025-12-14 at 10 05 20 AM" src="https://github.com/user-attachments/assets/97cb92d0-6749-4f8f-9f42-8530e59692af" />

### Main Window

The main window contains the following sections:

- **Left panel**: shows the plate layout with wells and FOVs and the image viewer. Selecting a well/FOV updates the image viewer to show the corresponding data. Double-clicking a well/FOV opens the full FOV in a new [ndv](https://pyapp-kit.github.io/ndv/latest/) window.
- **Center panel**: contains tabs for ROI detection, trace extraction, analysis, and a visualization tab for displaying results.
- **Right panel**: contains the list of *Runs* the user has performed. `cali` is structured so that each time the user changes pipeline settings and runs the analysis, a new *Run* is created. This allows comparing different settings and results. Selecting a run updates the center panel tabs to show the settings and results for that run.

<img width="1739" height="1095" alt="Screenshot 2025-12-17 at 9 56 06 AM" src="https://github.com/user-attachments/assets/5192b1a5-783e-4636-95ba-a8408d5b6a44" />

### Pipeline Tabs

Hovering over each parameter in the Detection, Extraction, and Analysis tabs shows a tooltip with a description of that parameter.

#### Detection Tab

The Detection tab allows the user to set the parameters used to segment cells and define ROIs for trace extraction. Currently, only **Cellpose** is supported as the segmentation method. The user can set Cellpose parameters and run the segmentation.

<img width="800" alt="Screenshot 2025-12-17 at 9 58 19 AM" src="https://github.com/user-attachments/assets/d4a9a17f-6688-4d90-a0ce-d71f86033750" />

#### Extraction Tab

The Extraction tab allows the user to configure fluorescence trace extraction from the segmented ROIs. Parameters include:

- **Neuropil correction**: enable/disable and set neuropil mask parameters
- **ΔF/F and OASIS deconvolution**:
  - window size for ΔF/F calculation
  - parameters for OASIS deconvolution (or leave on `auto` to use default parameters)
- **Metadata**: frame rate and pixel size
- **Number of Threads**: number of threads used for parallel extraction across wells/FOVs. Keep this number low if you experience memory issues during extraction.

<img width="800" alt="Screenshot 2025-12-17 at 9 58 26 AM" src="https://github.com/user-attachments/assets/a93c48e6-f688-41f9-ab79-75d008e366b6" />

#### Analysis Tab

The Analysis tab allows the user to configure analysis of the extracted traces, including:

- **Experiment Type**: since `cali` was designed to work with [micromanager-gui](https://github.com/fdrgsp/micromanager-gui), which supports spatio-temporal optogenetic stimulation, the user can select the `Evoked Activity` experiment type. This enables input of stimulation metadata used during acquisition and allows splitting results into stimulated vs non-stimulated ROIs.
- **Calcium Traces and Peaks Analysis**: parameters for calcium peak detection and analysis of calcium traces.
- **Inferred Spikes**: parameters for analysis of inferred spikes obtained from OASIS deconvolution. This includes detection thresholds, burst detection parameters, and correlation / synchrony analysis between ROIs.
- **Metadata**: additional experiment metadata (e.g. frame rate). The frame rate here is linked to the one in the Extraction tab; changing one will update the other.
- **Number of Threads**: number of threads for running the analysis across wells/FOVs. Keep this low if you experience memory issues.

<img width="800" alt="Screenshot 2025-12-17 at 9 58 30 AM" src="https://github.com/user-attachments/assets/4c8c5080-dae1-450c-95f3-c9d90b81ba2f" />

### Run the Pipeline

After setting parameters in each tab, the user can run the pipeline using the run panel at the bottom center of the main window.

<br>
<img width="800" alt="Screenshot 2025-12-14 at 7 01 27 PM" src="https://github.com/user-attachments/assets/95d08357-b284-486a-9160-b430d4a9247c" />
<br>
<br>

The user can select which steps to run (Detection, Extraction, Analysis) using the **Run Options** dropdown and then click **Run**. This runs the selected pipeline steps on all wells/FOVs in the plate and creates a new *Run* in the run panel on the right.

If the user wants to first explore and optimize parameters, a subset of wells/FOVs can be specified in the **Positions to Extract** field. By entering a comma-separated list of FOV position indices (obtained from the FOV table under the plate layout, e.g. `1, 3, 4` or `3-7` for a range), only those positions will be processed. This is useful to quickly test and optimize parameters before running the full pipeline. Once satisfied, the user can clear **Positions to Extract** and run the pipeline on the full dataset.

Segmentation results and neuropil masks (if enabled) are displayed in the image viewer by clicking on “Labels”. The rest of the results can be visualized in the **Visualization** tab.

The full pipeline settings can be saved and loaded through the `save` and `load` button next to the `Run` and `Cancel` buttons.

### Visualization Tab

The Visualization tab allows the user to explore analysis results for the selected *Run*.

Two sub-tabs are available:

- **Single Well**: visualize results for a single well/FOV.
- **Multi Well**: visualize summary metrics across all wells/FOVs.

Plots are interactive (zoom/pan). Clicking on a trace or data point highlights the corresponding ROI in the image viewer (and vice versa).

#### Single Well Tab

Available visualizations include:

- **Calcium Traces**: raw, ΔF/F, deconvolved ΔF/F, neuropil traces, and detected calcium peaks per ROI.
- **Inferred Spikes**: raw and thresholded inferred spike trains from OASIS.
- **Raster Plots**: raster plots of calcium peaks and inferred spikes across all ROIs.
- **Calcium Metrics**: amplitude, frequency, and other per-ROI metrics.
- **Calcium and Inferred Spikes Bursts**: burst metrics based on calcium peaks and inferred spikes (for inferred spikes raster, events are defined as rising edges in the thresholded binary spike train).
- **Correlation Metrics**:
  - pairwise Pearson correlation on calcium traces
  - jitter synchrony and max-lag cross-correlation on inferred spikes.
- **Stimulated vs Non-Stimulated Analysis** (for `Evoked Activity`): visualize and compare metrics between stimulated and non-stimulated ROIs.

<img width="927" height="1041" alt="Screenshot 2025-12-17 at 10 20 23 AM" src="https://github.com/user-attachments/assets/2620b629-c718-46d3-9c3a-eb83ca9ed3f4" />

#### Multi Well Tab

The Multi Well tab visualizes summary metrics across all wells/FOVs.

If the plate was treated with different conditions (e.g. drug vs control), clicking the **Show/Edit Plate Map** button under the plate layout opens a plate map editor where each well can be assigned to a condition. Currently, two condition dimensions are supported (e.g. genotype and treatment). This information is used to group wells/FOVs in the Multi Well tab. If no plate map is defined, data are shown on a per-well basis.

[screenshot of the multi well visualization tab]

---

## Analysis Details

### Extraction

#### ΔF/F Calculation

**Purpose**: ΔF/F (Delta F over F) is a standard fluorescence normalization method in calcium imaging that represents relative changes in fluorescence intensity. It accounts for baseline differences between cells and provides a normalized measure of activity.

**Calculation**:

$\Delta F/F(t) = \frac{F(t) - F_0(t)}{F_0(t)}$

where:

- \(F(t)\) is the raw fluorescence at time \(t\)
- \(F_0(t)\) is the baseline fluorescence estimated from a sliding window

The baseline \(F_0(t)\) is computed by taking the 10th percentile of the fluorescence within a sliding window centered at each time point.

**GUI Parameters**:

- **Window Size** (ms): size of the sliding window for baseline calculation.

#### OASIS Deconvolution

**Purpose**: `cali` uses the [OASIS algorithm](https://github.com/j-friedrich/OASIS) (Friedrich et al., 2017) to deconvolve the ΔF/F signal and infer the underlying spike activity.

For each ROI, `OASIS` is used to:

- estimate the noise level of the ΔF/F trace (later used for calcium peak detection)
- obtain both a deconvolved (denoised) ΔF/F trace and an inferred spike train.

**GUI Parameters**  
Currently, only the following parameter is exposed in the GUI:

- **Decay Constant** (\(\tau\), seconds): calcium decay time constant (depends on the calcium indicator and cell type).
  - If set to **0 (auto)**, `OASIS` estimates it from the data.

All other `OASIS` parameters are currently kept at default values:

- **Noise estimation**:
  - **AR Model**: 1 (first-order autoregressive model)
  - **Method**: `median`
  - **Lags**: 10
  - **Fudge Factor**: 0.98

- **Deconvolution**:
  - **Penalty**: 1 (L1 penalty for spike inference)

### Analysis

#### Calcium Peak Detection

**Purpose**: Identify significant calcium transients (peaks) in the deconvolved ΔF/F trace.

**Calculation**: Peaks are detected using [`scipy.signal.find_peaks`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html).

We use **height**, **prominence**, and **minimum distance** thresholds to identify peaks.

There are two modes to determine the peak *height* threshold:

- **MULTIPLIER** (recommended):  
  The height threshold is computed separately for each ROI as a multiple of the noise level estimated during `OASIS`-based noise estimation.

- **GLOBAL**:  
  A fixed absolute height value specified by the user. The same value is used for all ROIs in all wells/FOVs. This is mainly useful for testing, as it does *not* adapt to different noise levels across ROIs.

The **prominence** threshold is always computed as a multiple of the noise level estimated during `OASIS`.

The **minimum distance** between peaks is specified in milliseconds and determines how close in time two peaks can be while still being considered distinct events.

After detection, the following metrics are computed per ROI:

- Peak amplitudes (deconvolved ΔF/F values at peak locations; a.u.)
- Calcium peak event frequency: number of peaks per second
- Inter-event intervals (IEI): time between consecutive peaks

**GUI Parameters**:

- **Height Mode**: `MULTIPLIER` (× noise level) or `GLOBAL` (absolute value)
- **Height Value**:  
  - if `MULTIPLIER`: height threshold = value × noise  
  - if `GLOBAL`: height threshold = absolute value
- **Prominence Multiplier**: minimum prominence relative to noise
- **Min Distance** (ms): minimum time between consecutive peaks

#### Calcium Burst Detection

**Purpose**: Detect periods of sustained, elevated population activity in calcium signals. Bursts represent synchronized events where many cells are co-active.

**Calculation** (population-level, based on deconvolved ΔF/F):

1. **Population Activity**: compute the mean deconvolved ΔF/F across all active ROIs.
2. **Smoothing**: apply a Gaussian filter to the population trace (optional, to reduce noise).
3. **Threshold**: detect periods where the smoothed activity exceeds a fraction of its maximum.
4. **Duration Filter**: keep only bursts lasting at least a minimum duration.

**GUI Parameters**:

- **Burst Threshold** (%): percentage of maximum smoothed activity used as the detection threshold.
- **Min Duration** (ms): minimum burst duration to be retained.
- **Gaussian Sigma** (s): temporal smoothing (standard deviation) for the population activity.

**Computed Metrics**:

- Burst count  
- Average burst duration  
- Average inter-burst interval  
- Burst onset/offset times

#### Inferred Spikes Thresholding

**Purpose**: Convert continuous inferred spike traces from OASIS into binary spike trains by applying an adaptive threshold.

**Calculation**: The threshold is computed adaptively from the distribution of positive (non-zero) spike values for each ROI.

Two modes are available:

- **MULTIPLIER** (recommended):  
  - Estimate a noise level from the positive spikes (e.g. via Median Absolute Deviation (MAD) or a low percentile).  
  - The spike threshold is set as *threshold = value × noise* (per ROI).

- **GLOBAL**:  
  - Use a fixed, absolute threshold for all ROIs and wells/FOVs.  
  - Mainly useful for diagnostic/testing purposes; it does not adapt to per-ROI noise.

**GUI Parameters**:

- **Threshold Mode**: `MULTIPLIER` (× estimated noise level) or `GLOBAL` (absolute value).
- **Spike Threshold Value**:  
  - if `MULTIPLIER`: threshold = value × noise  
  - if `GLOBAL`: threshold = absolute value

#### Inferred Spikes Burst Detection

**Purpose**: Detect periods of sustained elevated population spiking activity. These spike bursts represent network-wide firing events.

**Calculation** (population-level, based on binary spike trains):

1. **Population Spike Rate**: compute the fraction of active ROIs per frame.
2. **Smoothing**: apply a Gaussian filter with standard deviation \(σ\) (optional).
3. **Threshold**: detect periods where the smoothed population rate exceeds a percentage threshold.
4. **Duration Filter**: retain bursts that last at least a minimum duration.

**GUI Parameters**:

- **Burst Threshold** (%): percentage of ROIs that must be active (after smoothing) to define a burst.
- **Min Duration** (ms): minimum burst duration.
- **Gaussian Sigma** (s): temporal smoothing of the population spike rate.

**Computed Metrics**:

- Number of network bursts  
- Average burst duration  
- Average inter-burst interval  
- Population firing rate during bursts

### Correlation Analysis

#### Pairwise Pearson Correlation on Calcium Traces

**Purpose**: Measure linear relationships between calcium activity patterns across ROIs using ΔF/F or deconvolved ΔF/F traces.

**Calculation**:

1. **Input**: ΔF/F traces from all ROIs (continuous raw calcium signals or deconvolved (denoised) by OASIS)
2. **Z-score normalization**: Each trace is mean-centered and divided by its standard deviation
3. **Compute correlation**: Standard Pearson correlation coefficient is calculated between all pairs of z-scored traces at zero lag
4. **Output**: NxN correlation matrix where each element represents the correlation between ROI pairs

**Output**:

- **Correlation Matrix**: Values range from -1 to 1
  - 1: perfect positive correlation (synchronized calcium activity)
  - 0: no linear relationship
  - -1: perfect negative correlation (anti-correlated activity)

**Summary Metric**: Global synchrony = median of row means (excluding diagonal)

#### Max-Lag Cross-Correlation on Inferred Spikes

**Purpose**: Quantify temporal relationships between spike trains by computing cross-correlograms (CCGs).

**Calculation**:

1. **Input**: Two binary spike trains (arrays of 0s and 1s where 1 = spike, 0 = no spike)
2. Spike events are defined as the rising edges in the binary spike trains.
3. **For each lag**: Shift one spike train relative to the other by ± lag frames and compute normalized dot product (spike coincidence count, normalized by the geometric mean of spike counts)
4. **Find maximum**: Return the correlation value and lag that gives the highest correlation

**Important Note on Methodology**:

Unlike continuous signal correlation (e.g., on ΔF/F traces), this analysis uses a **normalized dot product without mean centering**, not Pearson correlation.

**Why not Pearson correlation for spike trains?**

In Pearson correlation, data are mean-centered before computing the correlation coefficient. This is appropriate for continuous signals where values fluctuate around a mean. However, for binary spike trains:

- **0 means "no spike"** — this is real information about the absence of activity, not a missing or neutral value
- **Mean-centering would convert 0 → negative values**, which has no biological interpretation (a neuron cannot have "negative spike absence")
- **Pearson correlation on spike trains would penalize non-coincident spikes**, treating silent periods as anti-correlated rather than simply unrelated

**What we use instead: Normalized Dot Product**

The normalized dot product counts coincident spikes (where both neurons fire at the same time) and normalizes by the geometric mean of spike counts:

$\text{Correlation}(i, j) = \frac{\sum_{t} s_i(t) \cdot s_j(t)}{\sqrt{\sum_{t} s_i(t)^2 \cdot \sum_{t} s_j(t)^2}}$

where $s_i(t)$ and $s_j(t)$ are binary spike trains (0 or 1 at each time point).

This formula:
- Counts only when **both neurons spike together** (1 × 1 = 1)
- Ignores when **either or both are silent** (0 × 0 = 0, 0 × 1 = 0)
- Normalizes by spike counts to make the measure independent of firing rate
- Results are in **[0, 1]** where 1 = perfect synchrony, 0 = no temporal relationship

**Example with Lag**:

Consider two neurons over 10 time bins:

```
Neuron A: [0, 1, 0, 1, 0, 0, 1, 0, 0, 1]  (4 spikes)
Neuron B: [0, 1, 0, 0, 1, 0, 1, 0, 0, 1]  (4 spikes)
```

**At lag = 0** (no shift):
- **Coincident spikes** (both = 1): time bins 2, 7, 10 → **3 coincidences**
- **Correlation**: $\frac{3}{\sqrt{4 \cdot 4}} = \frac{3}{4} = 0.75$

**At lag = +1** (shift B forward by 1 frame):
```
Neuron A: [0, 1, 0, 1, 0, 0, 1, 0, 0, 1]
Neuron B: [_, 0, 1, 0, 0, 1, 0, 1, 0, 0]  (shifted, last value wraps/drops)
```
- **Coincidences**: time bins 3, 6, 8 → **0 coincidences** (in this example)
- **Correlation**: $\frac{0}{\sqrt{4 \cdot 4}} = 0$

**At lag = -1** (shift B backward by 1 frame):
```
Neuron A: [0, 1, 0, 1, 0, 0, 1, 0, 0, 1]
Neuron B: [1, 0, 0, 1, 0, 1, 0, 0, 1, _]  (shifted)
```
- **Coincidences**: time bins 4, 6, 9 → **1 coincidence** 
- **Correlation**: $\frac{1}{\sqrt{4 \cdot 4}} = 0.25$

**The algorithm tests all lags from -max_lag to +max_lag** and returns:
- **Maximum correlation value**: 0.75 (found at lag = 0)
- **Optimal lag**: 0 frames (indicating synchronous firing)

If the maximum had been at lag = +2, it would indicate Neuron B consistently fires ~2 frames after Neuron A.

**Why this matters**: Neurons with delayed coupling (e.g., A → B) might have low correlation at zero lag but high correlation at positive lags, revealing directional relationships that zero-lag analysis would miss.

**Comparison to Pearson correlation**:

**Pearson correlation** would give a different answer because it would:
1. Mean-center both trains (converting many 0s to negative values)
2. Penalize non-overlapping spikes as negative correlation
3. Produce values in [-1, 1] range, which is harder to interpret for spike coincidence

**Output**: Two heatmaps are generated:

- **Correlation Matrix**: Maximum correlation values (range: 0 to 1, where 1 = perfect synchrony at optimal lag, 0 = no temporal relationship)
- **Lag Matrix**: Lag values in frames (± frame shifts where maximum correlation occurs)

  - Positive lag: ROI j spikes lag behind ROI i
  - Negative lag: ROI j spikes lead ROI i
  - Lag = 0: Synchronous activity

**GUI Parameters**:

- **Max Lag** (ms): maximum time offset (converted to frames) over which to search for correlations.

**Summary Metric**:  
Global synchrony = median of the mean correlation per ROI (row means), excluding the diagonal.

#### Jitter Synchrony on Inferred Spikes

**Purpose**: Measure spike-time synchrony between spike trains within a small temporal tolerance window, independent of exact frame alignment.

**Calculation** (bidirectional jitter-based synchrony):

1. **Input**: Two binary spike trains (arrays of 0s and 1s representing spike times).
2. Spike events are defined as the rising edges in the binary spike trains.
3. For each spike in neuron $i$, check whether neuron $j$ fires within a temporal tolerance window of $±w$ frames. If yes, count this as a coincident spike.
4. Repeat in the opposite direction: for each spike in neuron $j$, check for a spike in neuron $i$ within $±w$ frames.
5. Count coincidences in both directions:
   - $C_{i \to j}$: coincidences found starting from spikes in $i$
   - $C_{j \to i}$: coincidences found starting from spikes in $j$
6. Combine and normalize:
   $S_{ij} = \frac{C_{i \to j} + C_{j \to i}}{N_i + N_j}$
   where $N_i$ and $N_j$ are the total number of spikes in neurons $i$ and $j$, respectively.

This yields a synchrony score between 0 and 1:

- **0** → no spikes occur near each other in time  
- **1** → every spike from both neurons has a partner within the jitter window

**GUI Parameters**:

- **Jitter Window** (ms): temporal tolerance for spike coincidence (converted to ± frames).

**Summary Metric**:  
Global synchrony = median of the mean synchrony per ROI (row means), excluding the diagonal.
