# cali

[![CI](https://github.com/fdrgsp/cali/actions/workflows/ci.yml/badge.svg)](https://github.com/fdrgsp/cali/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/fdrgsp/cali/branch/main/graph/badge.svg)](https://codecov.io/gh/fdrgsp/cali)

A GUI for Calcium Imaging Data Segmentation, Analysis and Visualization (🚧 WIP 🚧)

`cali` is package that provides a GUI to load calcium imaging timelapse data (1-photon neuronal cultures), segment neurons using Cellpose, extract and analyse fluorescence traces and visualize them. It was originally designed to work in combination with [micromanager-gui](https://github.com/fdrgsp/micromanager-gui), an open-source software to control microscopes through `Micro-Manager` and [pymmcore-plus](https://github.com/pymmcore-plus).

<img width="1750" height="1103" alt="Screenshot 2026-03-09 at 10 58 41 PM" src="https://github.com/user-attachments/assets/b9189d48-a2c1-494f-8fc4-7ceb9bd4012e" />

<br>

## Table of Contents

- [Getting Started](#getting-started)
  - [Quick Run (without installing)](#quick-run-without-installing)
  - [Installation](#installation)
    - [Using (uv) pip](#-using-uv-pip)
    - [Using uv](#-using-uv)
  - [Notes](#-notes)
- [CLI](#cli)
  - [`cali tree`](#cali-tree-to-inspect-a-database)
  - [Logger](#logger)
- [GUI Overview](#gui-overview)
  - [File Formats](#file-formats)
  - [Open a File](#open-a-file)
  - [Main Window](#main-window)
  - [Pipeline Tabs](#pipeline-tabs)
    - [Detection Tab](#detection-tab)
    - [Extraction Tab](#extraction-tab)
    - [Analysis Tab](#analysis-tab)
  - [Run the Pipeline](#run-the-pipeline)
  - [Visualization Tab](#visualization-tab)
    - [Single Well Tab](#single-well-tab)
    - [Multi Well Tab](#multi-well-tab)
- [Analysis Details](#analysis-details)
  - [Extraction](#extraction)
    - [ΔF/F Calculation](#δff-calculation)
    - [OASIS Denoising and Deconvolution](#oasis-denoising-and-deconvolution)
  - [Analysis](#analysis)
    - [Calcium Peak Detection](#calcium-peak-detection)
    - [Calcium Burst Detection](#calcium-burst-detection)
    - [Inferred Spikes Thresholding](#inferred-spikes-thresholding)
    - [Inferred Spikes Burst Detection](#inferred-spikes-burst-detection)
  - [Correlation Analysis](#correlation-analysis)
    - [Pairwise Pearson Correlation on Calcium Traces](#pairwise-pearson-correlation-on-calcium-traces)
    - [Cross-Correlogram (CCG) Analysis on Inferred Spikes](#cross-correlogram-ccg-analysis-on-inferred-spikes)
    - [Jitter Synchrony on Inferred Spikes](#jitter-synchrony-on-inferred-spikes)
  - [Cluster Analysis](#cluster-analysis)
  - [Multi-Well Condition Analysis](#multi-well-condition-analysis)
    - [Condition Labels](#condition-labels)
    - [Two-Level Aggregation](#two-level-aggregation)
    - [Percentage Metrics](#percentage-metrics)
    - [FOV-Level Scalar Metrics (Network Metrics)](#fov-level-scalar-metrics-network-metrics)

<br>

---

<br>

## Getting Started

### Quick Run (without installing)

If you have [uv](https://docs.astral.sh/uv/) installed, you can run `cali` directly without installing it using:

```bash
uvx -p 3.13 "git+https://github.com/fdrgsp/cali[cp4]"
```

***NOTE**: use python 3.11 or greater. Swap `cp4` with `cp3` to use Cellpose 3.x.*

<br>

To use Cellpose with GPU (e.g. on Windows), the correct version of CUDA has to be installed:

```bash
uvx -p 3.13 --index pytorch-cu130=https://download.pytorch.org/whl/cu130 "git+https://github.com/fdrgsp/cali[cp4,cu130]"
```

***NOTE**: swap `cu130` with `cu126` or `cu128` for different CUDA versions and `cp4` with `cp3` to use Cellpose 3.x.*

***NOTE**: [NVIDIA Drivers](https://www.nvidia.com/en-us/drivers/) should be already installed. You can check what version of cuda you need by running:*

```bash
nvidia-smi
```

### Installation

#### ▶ Using (uv) pip

Install `cali` directly without cloning (use python 3.11 or greater):

```bash
(uv) pip install "git+https://github.com/fdrgsp/cali[cp4]"
```

***NOTE**: swap `cp4` with `cp3` to use Cellpose 3.x.*

<br>

To use Cellpose with GPU (e.g. on Windows), the correct version of CUDA has to be installed:

```bash
(uv) pip install --index pytorch-cu130=https://download.pytorch.org/whl/cu130 "git+https://github.com/fdrgsp/cali[cp4,cu130]"
```

***NOTE**: swap `cu130` with `cu126` or `cu128` for different CUDA versions and `cp4` with `cp3` to use Cellpose 3.x.*

***NOTE**: [NVIDIA Drivers](https://www.nvidia.com/en-us/drivers/) should be already installed. You can check what version of cuda you need by running:*

```bash
nvidia-smi
```

#### ▶ Using uv

Clone the repository and install with `uv sync`:

```bash
git clone https://github.com/fdrgsp/cali
cd cali
```

To install with Cellpose:

```bash
uv sync --extra cp4
```

***NOTE**: swap `cp4` with `cp3` to use Cellpose 3.x.*

<br>

To use Cellpose with GPU (e.g. on Windows), the correct version of CUDA has to be installed. Add a CUDA extra alongside your Cellpose extra:

```bash
uv sync --extra cp4 --extra cu130   # Cellpose 4.x + CUDA 13.0
```

***NOTE**: swap `cu130` with `cu126` or `cu128` for different CUDA versions.*

***NOTE**: [NVIDIA Drivers](https://www.nvidia.com/en-us/drivers/) should be already installed. You can check what version of cuda you need by running:*

```bash
nvidia-smi
```

#### ▶ NOTES

##### Building on macOS

If you encounter build errors with `oasis-deconv` (especially SDK-related errors), set these environment variables before installing:

```bash
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
export LDFLAGS="-L${SDKROOT}/usr/lib"
```

Then run your installation command.

<br>

---

<br>

## CLI

### `cali tree` to inspect a database

Print a tree view of the contents of a `.cali` database file without opening the GUI:

```bash
cali tree <path/to/database.cali> [-l LEVEL] [-s {True,False}]
```

**Arguments:**

| Argument | Short | Default | Description |
|---|---|---|---|
| `db` | — | *(required)* | Path to the `.cali` database file |
| `--level` | `-l` | `roi` | Maximum tree depth: `experiment`, `plate`, `well`, `fov`, or `roi` |
| `--show-settings` | `-s` | `True` | Show detailed analysis settings (`True` or `False`) |

**Examples:**

```bash
# Full tree including ROI details
cali tree my_experiment.cali

# Show only up to FOV level, without settings details
cali tree my_experiment.cali -l fov -s False

# Full level names also work
cali tree my_experiment.cali --level well --show-settings False
```
<img width="833" alt="Screenshot 2026-03-09 at 9 59 02 PM" src="https://github.com/user-attachments/assets/b5482401-245f-4814-9acd-c516d54c7755" />


### Logger

| Option | Short | Default | Description |
|---|---|---|---|
| `--logger` | `-lg` | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, or `CRITICAL` |

**Examples:**

```bash
# set the logging level to DEBUG
cali --logger DEBUG
```

<br>

---

<br>

## GUI Overview

<https://github.com/user-attachments/assets/4fdcabb9-6d7b-4c4f-ae84-8dffc8487376>

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

<img width="600" alt="Screenshot 2026-03-09 at 10 01 00 PM" src="https://github.com/user-attachments/assets/76bc82ee-7d19-478e-b95e-0b02c72c90ee" />

**NOTE**: If you are loading a folder of TIFF files, a plate assignment widget will open where you can:

- select the plate type
- assign each TIFF file (as a FOV) to a well. Files are **auto-matched** to wells by filename pattern (e.g., a file containing `A1` in its name is automatically assigned to well A1). You can manually adjust assignments using the Add/Remove buttons if needed.

After confirming the plate assignment, the main `cali` window will open. Next time, the project can be loaded directly from the database file.

<img width="800" alt="Screenshot 2026-03-09 at 10 02 10 PM" src="https://github.com/user-attachments/assets/5edd79e4-250f-4b94-8acf-003b863ca60b" />


### Main Window

The main window contains the following sections:

- **Left panel**: shows the plate layout with wells and FOVs and the image viewer. Selecting a well/FOV updates the image viewer to show the corresponding data. Double-clicking a well/FOV opens the full FOV in a new [ndv](https://pyapp-kit.github.io/ndv/latest/) window.
- **Center panel**: contains tabs for ROI detection, trace extraction, analysis, and a visualization tab for displaying results.
- **Right panel**: contains the list of *Runs* the user has performed. `cali` is structured so that each time the user changes pipeline settings and runs the analysis, a new *Run* is created. This allows comparing different settings and results. Selecting a run updates the center panel tabs to show the settings and results for that run.

<img width="1750" height="1103" alt="Screenshot 2026-03-09 at 10 10 25 PM" src="https://github.com/user-attachments/assets/3af9fccb-4412-4865-ae1a-8e24fa1262af" />


### Pipeline Tabs

Hovering over each parameter in the Detection, Extraction, and Analysis tabs shows a tooltip with a description of that parameter.

#### Detection Tab

The Detection tab allows the user to set the parameters used to segment cells and define ROIs for trace extraction. Two detection methods are available:

- **Cellpose**: run Cellpose segmentation with configurable parameters directly from the GUI.
- **Imported Labels**: import pre-existing label TIFF files (e.g., from manual segmentation or another tool) and assign them to specific FOVs. Clicking "Import Labels..." opens a dedicated dialog where you can:
  - browse for a folder containing label TIFF files
  - view the plate map and select individual wells
  - assign each label file to a specific FOV (files are **auto-matched** to FOVs by filename pattern when possible)
  - manually adjust assignments using Assign/Unassign/Reset buttons

  On import, each label TIFF is read and converted into ROI/Mask objects in the database. The user can then proceed with extraction and analysis as usual.
  
<img width="800" alt="Screenshot 2026-03-09 at 10 11 24 PM" src="https://github.com/user-attachments/assets/ecd650d5-6450-49e8-abb5-cee35341c5f7" />


#### Extraction Tab

The Extraction tab allows the user to configure fluorescence trace extraction from the segmented ROIs. Parameters include:

- **Neuropil correction**: enable/disable and set neuropil mask parameters
- **ΔF/F and OASIS Deconvolution**:
  - window size for ΔF/F calculation
  - percentile for ΔF/F baseline
  - parameters for OASIS deconvolution (or leave on `auto` to use default parameters)
- **Metadata**: frame rate and pixel size
- **Number of Threads**: number of threads used for parallel extraction across wells/FOVs. Keep this number low if you experience memory issues during extraction.
- **CSV Export**: raw traces, ΔF/F, deconvolved ΔF/F, inferred spikes (raw and thresholded), neuropil traces, and neuropil-corrected traces.

<img width="800" alt="Screenshot 2026-03-09 at 10 11 33 PM" src="https://github.com/user-attachments/assets/d6a6bcd0-137d-441c-a3f5-0e7b9c292e82" />


#### Analysis Tab

The Analysis tab allows the user to configure analysis of the extracted traces, including:

- **Experiment Type**: since `cali` was designed to work with [micromanager-gui](https://github.com/fdrgsp/micromanager-gui), which supports spatio-temporal optogenetic stimulation, the user can select the `Evoked Activity` experiment type. This enables input of stimulation metadata used during acquisition and allows splitting results into stimulated vs non-stimulated ROIs.
- **Calcium Traces and Peaks Analysis**: parameters for calcium peak detection and analysis of calcium traces.
- **Inferred Spikes**: parameters for analysis of inferred spikes obtained from OASIS deconvolution. This includes detection thresholds, burst detection parameters, and correlation / synchrony analysis between ROIs.
- **Cluster Analysis**: groups ROIs into functional clusters based on their pairwise denoised ΔF/F correlation patterns using Hierarchical clustering (average/UPGMA linkage), with automatic or fixed number of clusters.
- **Metadata**: additional experiment metadata (e.g. frame rate). The frame rate here is linked to the one in the Extraction tab; changing one will update the other.
- **Number of Threads**: number of threads for running the analysis across wells/FOVs. Keep this low if you experience memory issues.
- **CCG Worker Processes**: number of worker processes for parallel CCG (Cross-Correlogram) computation. CCG computation is the most time-consuming part of FOV analysis and uses multiprocessing to parallelize across ROI pairs. Default is CPU count - 2. Higher values speed up computation but use more memory.
- **CSV Export**:
  - Pairwise correlation matrices (calcium ΔF/F, denoised ΔF/F, spike synchrony, spike cross-correlation, and cross-correlation lags).
  - Multi-well aggregated data: exports all multi-well bar plot data to CSV files in a `multi_well/` subdirectory. Each CSV contains condition means, SEMs, and individual FOV values for every available metric.

<img width="800" alt="Screenshot 2026-03-09 at 10 11 44 PM" src="https://github.com/user-attachments/assets/9d4d613e-92d1-40ee-9b9c-ad0742c9a815" />


### Run the Pipeline

After setting parameters in each tab, the user can run the pipeline using the run panel at the bottom center of the main window.

<br>
<img width="800" alt="Screenshot 2025-12-14 at 7 01 27 PM" src="https://github.com/user-attachments/assets/95d08357-b284-486a-9160-b430d4a9247c" />
<br>
<br>

The user can select which steps to run (Detection, Extraction, Analysis) using the **Run Options** dropdown and then click **Run**. This runs the selected pipeline steps on all wells/FOVs in the plate and creates a new *Run* in the run panel on the right.

If the user wants to first explore and optimize parameters, a subset of wells/FOVs can be specified in the **Positions to Extract** field. By entering a comma-separated list of FOV position indices (obtained from the FOV table under the plate layout, e.g. `1, 3, 4` or `3-7` for a range), only those positions will be processed. This is useful to quickly test and optimize parameters before running the full pipeline. Once satisfied, the user can clear **Positions to Extract** and run the pipeline on the full dataset.

Segmentation results and neuropil masks (if enabled) are displayed in the image viewer by clicking on "Labels". The rest of the results can be visualized in the **Visualization** tab.

The full pipeline settings can be saved and loaded through the `save` and `load` button next to the `Run` and `Cancel` buttons.

### Visualization Tab

The Visualization tab allows the user to explore analysis results for the selected *Run*.

Two sub-tabs are available:

- **Single Well**: visualize results for a single well/FOV.
- **Multi Well**: visualize summary metrics across all wells/FOVs.

Plots are interactive (zoom/pan). Clicking on a trace or data point highlights the corresponding ROI in the image viewer (and vice versa).

#### Single Well Tab

Available visualizations include:

- **Calcium Traces**: raw, ΔF/F, denoised ΔF/F, neuropil traces, and detected calcium peaks per ROI.
- **Inferred Spikes**: raw and thresholded inferred spike trains from OASIS.
- **Raster Plots**: raster plots of calcium peaks and inferred spikes across all ROIs.
- **Calcium Metrics**: amplitude, frequency, and other per-ROI metrics.
- **Inferred Spikes Metrics**: frequency (on thresholded and/or thresholded rising edges).
- **Calcium and Inferred Spikes Bursts**: burst metrics based on calcium peaks and inferred spikes (for inferred spikes raster, events are defined as rising edges in the thresholded binary spike train).
- **Correlation Metrics**:
  - pairwise Pearson correlation on calcium traces
  - jitter synchrony and max-lag cross-correlation on inferred spikes.
- **Stimulated vs Non-Stimulated Analysis** (for `Evoked Activity`): visualize and compare metrics between stimulated and non-stimulated ROIs.
- **Cluster Analysis**: cluster-sorted correlation heatmap, cluster-colored raster plot, and cluster-colored trace overlay, showing how ROIs are grouped into functional clusters based on correlated activity.

<img width="927" height="1041" alt="Screenshot 2025-12-17 at 10 20 23 AM" src="https://github.com/user-attachments/assets/2620b629-c718-46d3-9c3a-eb83ca9ed3f4" />

#### Multi Well Tab

The Multi Well tab visualizes summary metrics across all wells/FOVs, aggregated by condition. Data are grouped using a three-level hierarchical aggregation: ROI values are first averaged within each FOV, then FOV means within the same well are averaged to produce a single well mean (treating FOVs as technical replicates), and finally well means are combined into condition-level estimates treating each well as one independent biological replicate. Error bars represent the between-well SEM, and individual well means are overlaid as scatter points.

If the plate was treated with different conditions (e.g. drug vs control), clicking the **Show/Edit Plate Map** button under the plate layout opens a plate map editor where each well can be assigned to a condition. Currently, two condition dimensions are supported (e.g. genotype and treatment). This information is used to group wells/FOVs in the Multi Well tab. If no plate map is defined, data are shown on a per-well basis.

Available multi-well visualizations include:

- **Cell Properties**: % active cells, mean cell size, and per-condition stim/non-stim breakdowns (evoked experiments).
- **Calcium Peaks**: amplitude, frequency, and inter-event interval (IEI) bar plots; stim-split amplitude and frequency (evoked experiments).
- **Calcium Bursts**: population-level burst count, average duration, and average inter-burst interval.
- **Inferred Spike Frequency**: thresholded and rising-edge variants; stim-split frequency (evoked experiments).
- **Inferred Spike Bursts**: burst count, average duration, average interval, and burst rate.
- **Calcium Correlation**: global Pearson correlation on ΔF/F and denoised ΔF/F traces.
- **Spike Synchrony**: jitter-based synchrony (with rising-edge variant).
- **Spike Correlation**: max-lag cross-correlation from CCG analysis (with rising-edge variant).
- **Fraction Significant CCG Pairs**: fraction of ROI pairs with |z-score| > 2, measuring network connectivity density (with rising-edge variant).

<br>

---

<br>

## Analysis Details

### Extraction

#### ΔF/F Calculation

**Purpose**: ΔF/F (Delta F over F) is a standard fluorescence normalization method in calcium imaging that represents relative changes in fluorescence intensity. It accounts for baseline differences between cells and provides a normalized measure of activity.

**Calculation**:

$$\Delta F/F(t) = \frac{F(t) - F_0(t)}{F_0(t)}$$

where:

- $F(t)$ is the raw fluorescence at time $t$
- $F_0(t)$ is the baseline fluorescence estimated from a sliding window

The baseline $F_0(t)$ is computed by taking a user-defined percentile of the fluorescence within a sliding window centered at each time point.

**GUI Parameters**:

- **Window Size** (s): size of the sliding window for baseline calculation.
- **Percentile**: percentile of fluorescence values used as the baseline F₀ (default: 10).

#### OASIS Denoising and Deconvolution

**Purpose**: `cali` uses the [OASIS algorithm](https://github.com/j-friedrich/OASIS) (Friedrich et al., 2017) to denoise the ΔF/F signal and infer the underlying spike activity.

For each ROI, `OASIS` is used to:

- estimate the noise level of the ΔF/F trace (later used for calcium peak detection)
- obtain both a denoised ΔF/F trace and an inferred spike train.

**Understanding the inferred spike train:**

The inferred spike trace has the same length and time axis as the input ΔF/F, one value per acquired frame. Each value represents an inferred event magnitude (deconvolved activity) assigned to that frame. A larger value means more inferred activity, but the scale isn’t in physical units; interpret the values as relative and use them for comparison within your dataset.

Since the output is sampled once per frame, temporal resolution is limited by the acquisition rate. For example, at 10 Hz, each sample corresponds to a 100 ms bin. Events closer together than 100 ms cannot be cleanly separated and will generally appear as a larger inferred value in one frame or spread across adjacent frames. In practice, calcium indicator kinetics and noise can further reduce effective timing precision to multiple frames, so a single underlying spike/event may yield inferred activity spanning 2–3 frames. Higher frame rates help, but indicator dynamics and SNR often set the ultimate limit.

**GUI Parameters**
Currently, only the following parameter is exposed in the GUI:

- **Decay Constant** ($\tau$, seconds): calcium decay time constant (depends on the calcium indicator and cell type).
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

**Purpose**: Identify significant calcium transients (peaks) in the denoised ΔF/F trace.

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

- Peak amplitudes (denoised ΔF/F values at peak locations; a.u.)
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

**Calculation** (population-level, based on detected calcium peaks):

1. **Binary peak events**: for each ROI, detected calcium peaks are converted to binary arrays (1 at peak frames, 0 elsewhere).
2. **Population activity**: compute the fraction of active ROIs with peaks at each frame:

   ```
   ROI 1: [0, 0, 1, 0, 1, 0, 1, 0, ...]
   ROI 2: [0, 0, 1, 0, 0, 0, 1, 0, ...]
   ROI 3: [0, 0, 0, 0, 1, 0, 1, 0, ...]
   ─────────────────────────────────────
   Mean:  [0, 0, .67, 0, .67, 0, 1.0, 0, ...]
   ```

3. **Smoothing**: apply a Gaussian filter (default sigma=0.3s) to the population trace. This is needed because peak detection has limited temporal precision (~1-2 frames), so cells firing together may have peaks on nearby but not identical frames. Smoothing bridges these small gaps.
4. **Threshold**: detect periods where the smoothed fraction exceeds the threshold (e.g., 25% → 0.25).
5. **Duration Filter**: keep only bursts lasting at least a minimum duration.

**GUI Parameters**:

- **Burst Threshold** (%): percentage of active ROIs that must have peaks simultaneously (e.g., 25% means at least 25% of active cells must fire together).
- **Min Duration** (ms): minimum burst duration to be retained.
- **Gaussian Sigma** (s): temporal smoothing (standard deviation) for the population activity. Default 0.3s.

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
2. **Smoothing**: apply a Gaussian filter with standard deviation $\sigma$ (optional).
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

**Purpose**: Measure linear relationships between calcium activity patterns across ROIs using ΔF/F or denoised ΔF/F traces.

**Calculation**:

1. **Input**: ΔF/F traces from all ROIs (continuous raw calcium signals or denoised by OASIS)
2. **Z-score normalization**: Each trace is transformed to have zero mean and unit variance:

   $$z_i(t) = \frac{x_i(t) - \bar{x}_i}{\sigma_i}$$

   where $x_i(t)$ is the raw ΔF/F value at time $t$, $\bar{x}_i$ is the mean, and $\sigma_i$ is the standard deviation.

   *Example*: If an ROI has mean ΔF/F = 0.5 and std = 0.2, a raw value of 0.9 becomes z = (0.9 - 0.5) / 0.2 = 2.0. This standardization ensures that ROIs with different baseline fluorescence levels and noise amplitudes can be fairly compared — a correlation of 0.8 means the same thing regardless of whether the original signals ranged from 0-1 or 0-10.

3. **Compute correlation**: Standard Pearson correlation coefficient is calculated between all pairs of z-scored traces at zero lag
4. **Output**: NxN correlation matrix where each element represents the correlation between ROI pairs

**Output**:

- **Correlation Matrix**: Values range from -1 to 1
  - 1: perfect positive correlation (synchronized calcium activity)
  - 0: no linear relationship
  - -1: perfect negative correlation (anti-correlated activity)

**Summary Metric**: Global synchrony = median of row means (excluding diagonal)

#### Cross-Correlogram (CCG) Analysis on Inferred Spikes

**Purpose**: Quantify temporal relationships between spike trains using standard cross-correlogram (CCG) methodology with statistical significance testing.

**Overview**:

The CCG analysis computes the probability that neuron j fires at a given time lag relative to neuron i.

**Calculation**:

1. **Input**: Two binary spike trains (arrays of 0s and 1s where 1 = spike, 0 = no spike). Two spike event definitions are available:
   - **Thresholded**: Uses the thresholded binary spike values directly (each 1 in the binary array counts as a spike event)
   - **Rising Edges** (optional): Uses only the rising edges (0→1 transitions) of the thresholded binary spike trains, which better captures distinct spike onsets when spikes span multiple frames. Enable via "Enable Rising Edge Analysis" checkbox.

2. **Per-trigger probability normalization**: For each lag τ, compute:

   $$CCG(\tau) = P(\text{spike in } j \text{ at lag } \tau \mid \text{spike in } i)$$

   This is calculated as the count of coincidences at lag τ divided by the number of reference spikes in neuron i.

   *Example*: If neuron i has 10 spikes and 4 of them have a corresponding spike in neuron j at lag τ=2, then CCG(2) = 4/10 = 0.4, meaning there's a 40% probability that neuron j fires 2 frames after neuron i.

3. **Border correction**: At large lags, there are fewer overlapping time points between the two spike trains. Border correction scales the count to account for this reduced overlap, providing unbiased estimates across all lags.

   *Example*: With 100 frames and lag=10, only 90 frames overlap. If we count 3 coincidences in those 90 frames, border correction scales it to 3 × (100/90) ≈ 3.33 (~11% adjustment). With 1000 frames and the same lag, the correction is only 3 × (1000/990) ≈ 3.03 (~1% adjustment). The correction matters more for shorter recordings and larger lags.

4. **Baseline correction (shift predictor)**: To distinguish true functional connectivity from slow co-modulations (e.g., both neurons increasing activity over time), we compute a **null model** that represents what the CCG would look like if there were no precise timing relationships between the neurons.

   **What is a circular shift?**
   A circular shift wraps the spike train around itself: elements that "fall off" one end reappear at the other. For example, with a spike train `[0,0,1,0,1,1,0,0]` and a shift of 3 positions:
   - The last 3 elements `[1,0,0]` move to the front
   - Result: `[1,0,0,0,0,1,0,1]`

   This operation preserves the total number of spikes and the overall firing rate pattern, but destroys the precise frame-by-frame timing relationship between the two spike trains.

   **Why circular shifts create a valid null model:**
   - If two neurons have a true synaptic connection, their spikes will be precisely timed relative to each other (e.g., neuron j consistently fires 2-3 frames after neuron i)
   - If two neurons are merely co-modulated by a common input (e.g., both increase firing during arousal), they will have correlated activity but no precise timing relationship
   - By circularly shifting one spike train, we break precise timing while preserving the slow co-modulation pattern
   - The CCG computed on shifted data represents what we'd expect from co-modulation alone

   **Computing the baseline:**
   We repeat this process multiple times (configurable, default 20 shuffles), each time shifting by a different random amount:
   - **Baseline mean** ($\mu_{baseline}$): Average CCG across all shuffles — this captures the expected CCG due to slow co-modulations alone
   - **Baseline std** ($\sigma_{baseline}$): Standard deviation across shuffles — this measures the variability in the null model, used for significance testing

   *Example*: If two neurons both fire more frequently during the second half of a recording (slow co-modulation), the raw CCG might show elevated values. By circularly shifting one spike train (e.g., moving the last 200 frames to the beginning), we break the precise timing relationship while preserving the overall firing rates. Averaging CCG across 20 such shifts gives a baseline that captures this slow co-modulation, allowing us to identify whether the observed CCG is higher than expected by chance.

5. **Z-score calculation**: Using the baseline from step 4, we compute statistical significance:

   $$z = \frac{CCG_{raw} - \mu_{baseline}}{\sigma_{baseline}}$$

   The z-score tells us how many standard deviations the observed CCG is above (or below) the null model expectation:
   - **z > 2**: The observed spike coincidence is significantly *higher* than expected from co-modulation alone → suggests excitatory functional connectivity
   - **z < -2**: The observed spike coincidence is significantly *lower* than expected → suggests inhibitory functional connectivity or anti-correlation
   - **|z| < 2**: The observed CCG is within the range expected from slow co-modulations; no significant precise-timing relationship detected

   The threshold of |z| > 2 corresponds to roughly the 95th percentile of a normal distribution (p < 0.05), meaning there's less than a 5% chance the observed CCG would occur by chance under the null model.

   *Example*: If raw CCG = 0.35, baseline mean = 0.20, and baseline std = 0.05, then z = (0.35 - 0.20) / 0.05 = 3.0. Since |3.0| > 2, this pair shows significant functional connectivity — the observed spike coincidence is 3 standard deviations above what's expected from chance co-modulation.

6. **Find optimal lag**: Return the CCG value and lag where the raw CCG is maximum.

**Output**: Three heatmaps are generated:

- **Peak CCG Matrix**: Maximum CCG values at optimal lag (per-trigger probability, between [0, 1])
  - Higher values indicate stronger spike coincidence at the optimal lag
  - Values represent P(spike in j | spike in i) at the optimal lag

- **Lag Matrix**: Lag values in frames where maximum CCG occurs
  - Positive lag: ROI j lags behind ROI i (i fires first)
  - Negative lag: ROI j leads ROI i (j fires first)
  - Lag = 0: Synchronous activity

- **Z-Score Matrix**: Statistical significance of the CCG values
  - |z| > 2: Suggests significant functional connectivity
  - |z| < 2: Connectivity may be due to slow co-modulations or chance

**GUI Parameters**:

- **CCG Max Lag** (ms): Maximum time offset (converted to frames) over which to search for correlations.
- **CCG Baseline Shuffles**: Number of circular shift surrogates for baseline correction (default: 20). More shuffles = more accurate baseline but slower computation.
- **Enable Rising Edge Analysis**: When enabled, performs additional CCG analysis on spike onset times (0→1 transitions). Approximately doubles CCG computation time.

**Summary Metric**:
Global synchrony = median of the mean CCG per ROI (row means), excluding the diagonal.

#### Jitter Synchrony on Inferred Spikes

**Purpose**: Measure spike-time synchrony between spike trains within a small temporal tolerance window, independent of exact frame alignment.

**Calculation** (bidirectional jitter-based synchrony):

1. **Input**: Two binary spike trains (arrays of 0s and 1s representing spike times).
2. Spike events are defined as all non-zero frames in the binary spike trains (i.e., every frame where the value is 1). When rising-edge analysis is enabled, only the onset frames (transitions from 0 to 1) are used instead.
3. For each spike in neuron $i$, check whether neuron $j$ fires within a temporal tolerance window of $±w$ frames. If yes, count this as a coincident spike.
4. Repeat in the opposite direction: for each spike in neuron $j$, check for a spike in neuron $i$ within $±w$ frames.
5. Count coincidences in both directions:
   - $C_{i \to j}$: coincidences found starting from spikes in $i$
   - $C_{j \to i}$: coincidences found starting from spikes in $j$
6. Combine and normalize:

   $$S_{ij} = \frac{C_{i \to j} + C_{j \to i}}{N_i + N_j}$$

   where $N_i$ and $N_j$ are the total number of spikes in neurons $i$ and $j$, respectively.

This yields a synchrony score between 0 and 1:

- **0** → no spikes occur near each other in time
- **1** → every spike from both neurons has a partner within the jitter window

*Example*: Neuron i has 8 spikes and neuron j has 6 spikes. With a jitter window of ±2 frames: starting from neuron i's spikes, 5 of the 8 have a spike in neuron j within ±2 frames ($C_{i \to j}$ = 5). Starting from neuron j's spikes, 4 of the 6 have a spike in neuron i within ±2 frames ($C_{j \to i}$ = 4). The synchrony score is $S_{ij}$ = (5 + 4) / (8 + 6) = 9/14 ≈ 0.64. This means about 64% of all spikes from both neurons have a temporally coincident partner. If all spikes were perfectly synchronized (every spike in i matched one in j and vice versa), the score would be 1.0.

**GUI Parameters**:

- **Jitter Window** (ms): temporal tolerance for spike coincidence (converted to ± frames).

**Summary Metric**:
Global synchrony = median of the mean synchrony per ROI (row means), excluding the diagonal.

### Cluster Analysis

**Purpose**: Group ROIs into functional clusters based on the similarity of their pairwise denoised ΔF/F correlation patterns, revealing co-active cell assemblies within a FOV.

**Input**: The NxN pairwise Pearson correlation matrix computed from denoised ΔF/F traces.

Each ROI is represented by its row in the correlation matrix (a vector of length N describing how correlated it is with every other ROI). ROIs with similar correlation profiles — i.e., ROIs that are collectively correlated or anti-correlated with the same partners — are placed in the same cluster.

**Method — Hierarchical clustering (average/UPGMA linkage)**:

Builds a cluster hierarchy by iteratively merging the pair of clusters with the smallest average pairwise distance. The distance between ROIs is defined as $d_{ij} = 1 - r_{ij}$, where $r_{ij}$ is the Pearson correlation coefficient, so highly correlated ROIs are close together ($d \approx 0$) and anti-correlated ROIs are far apart ($d \approx 2$). A linkage tree is built and then cut at the level that produces the target number of clusters.

**Automatic cluster count selection**:

When "Number of Clusters" is set to 0 (Auto), the optimal number of clusters K is determined by scanning K = 2, 3, …, Max K and computing the **average silhouette score** for each:

$$s_i = \frac{b_i - a_i}{\max(a_i,\, b_i)}$$

where $a_i$ is the mean intra-cluster distance for ROI $i$ and $b_i$ is the mean distance to the nearest other cluster. The silhouette score ranges from −1 to +1; higher values indicate that ROIs are well-matched to their own cluster and clearly separated from neighboring clusters. The K with the highest average silhouette score is selected.

**GUI Parameters**:

- **Number of Clusters**: set to `0` (Auto) to detect the optimal K via silhouette scoring, or enter a positive integer to force a fixed cluster count.
- **Auto-detect Max K**: upper bound of the silhouette-score scan (only active when Number of Clusters = 0). The scan evaluates every K from 2 up to this value.

### Multi-Well Condition Analysis

The **Multi Well** tab displays condition-level summaries of analysis metrics, aggregating data from all wells and FOVs that share the same condition assignment. The aggregation follows a two-level hierarchical approach to correctly account for the nested structure of the data (ROIs within FOVs, FOVs within conditions).

#### Condition Labels

Conditions labels are defined via the plate map editor in the GUI. For `Evoked Activity` experiments, condition labels can optionally include the stimulation status of each ROI (e.g., `ctrl_stim` vs `ctrl_non_stim`), enabling a direct comparison between stimulated and non-stimulated populations within the same condition.

#### Three-Level Aggregation

For each metric, data are aggregated in three steps:

**Step 1 — ROI → FOV**

For each FOV, the values of all active ROIs (including flattened per-ROI lists such as individual peak amplitudes) are averaged to produce a single FOV mean.

**Step 2 — FOV → Well**

FOV means within the same well are averaged (unweighted) to produce a single well mean. Multiple FOVs from the same well are treated as technical replicates of the same biological sample.

**Step 3 — Well → Condition**

The condition mean is the unweighted mean of the well means, treating each well as one independent biological replicate. The condition SEM is computed across well means: `std(well_means, ddof=1) / sqrt(n_wells)`.

The individual well means are overlaid on each bar as scatter points, providing direct visibility into within-condition variability.

#### Percentage Metrics

For the **% Active Cells** metric, the same three-level hierarchy is used:

**Step 1 — ROI → FOV**: for each FOV, the percentage of active ROIs is computed as the number of active ROIs divided by the total number of ROIs, multiplied by 100.

**Step 2 — FOV → Well**: FOV percentages within the same well are averaged (unweighted) to produce a single well percentage.

**Step 3 — Well → Condition**: The condition mean and SEM are computed across well means: `std(well_means, ddof=1) / sqrt(n_wells)`. Using between-well SEM avoids pseudo-replication: cells within the same well are not independent, and collecting more ROIs from the same well should not shrink the error bar.

#### FOV-Level Scalar Metrics (Network Metrics)

Some metrics are inherently FOV-level scalars rather than per-ROI distributions — for example, global correlation, synchrony, and burst counts. These are computed from pairwise matrices or population-level analyses and yield a single value per FOV.

For these metrics, there are no within-FOV ROI values to average over, so the aggregation uses a two-step well-based hierarchy:

**Step 1 — FOV → Well**: FOV scalars within the same well are combined using a weighted mean (weight = number of unique ROI pairs `n*(n-1)/2` for correlation/synchrony metrics, or equal weight for burst metrics). FOVs with more cells produce more reliable pairwise estimates and therefore contribute proportionally more within the well.

**Step 2 — Well → Condition**: The condition mean is the unweighted mean of the well means; the condition SEM is `std(well_means, ddof=1) / sqrt(n_wells)`. Wells are treated as equal independent biological replicates.

The FOV-level scalar metrics available in the Multi-Well tab include:

- **Calcium ΔF/F Correlation**: global zero-lag Pearson correlation on ΔF/F traces (median of off-diagonal row-means)
- **Calcium Denoised ΔF/F Correlation**: same as above but on denoised ΔF/F traces
- **Spike Jitter Synchrony**: global synchrony of inferred spikes (jitter window method)
- **Spike Max-Lag Correlation**: global max-lag cross-correlation of inferred spikes (CCG method)
- **Spike Jitter Synchrony (Rising Edges)**: same as above but on rising-edge-filtered spike trains
- **Spike Max-Lag Correlation (Rising Edges)**: same as above but on rising-edge-filtered spike trains
- **Fraction Significant CCG Pairs**: fraction of ROI pairs with |z-score| > 2 in the CCG analysis, measuring network connectivity density
- **Fraction Significant CCG Pairs (Rising Edges)**: same as above but on rising-edge-filtered spike trains
- **Burst Count, Duration, Interval, Rate**: population-level burst statistics (both spike-based and calcium-based)
