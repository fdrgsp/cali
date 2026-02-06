# cali

A Gui for Calcium Imaging Data Segmentation, Analysis and Visualization (🚧 WIP 🚧)

[![CI](https://github.com/fdrgsp/cali/actions/workflows/ci.yml/badge.svg)](https://github.com/fdrgsp/cali/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/fdrgsp/cali/branch/main/graph/badge.svg)](https://codecov.io/gh/fdrgsp/cali)

`cali` is package that provides a gui to load calcium imaging timelapse data (1-photon neuronal cultures), segment neurons using Cellpose, extract and analyse fluorescence traces and visualize them. It was originally designed to work in combination with [micromanager-gui](https://github.com/fdrgsp/micromanager-gui), an open-source software to control microscopes through `Micro-Manager` and [pymmcore-plus](https://github.com/pymmcore-plus).

<img width="1736" height="1093" alt="Screenshot 2025-12-17 at 10 10 43 AM" src="https://github.com/user-attachments/assets/aac1188b-180e-49dc-b095-3d3a3e350750" />

## To Run

If you have [uv](https://docs.astral.sh/uv/) installed, you can run `cali` directly without installing it using:

```bash
uvx -p 3.13 "git+https://github.com/fdrgsp/cali[cp4]"
```

***NOTE**: use python 3.11 or greater. Swap `cp4` with `cp3` to use Cellpose 3.x.*

To use Cellpose with GPU (e.g. on Windows), the correct version of CUDA has to be installed:

```bash
uvx -p 3.13 --index pytorch-cu130=https://download.pytorch.org/whl/cu130 "git+https://github.com/fdrgsp/cali[cp4,cu130]"
```

***NOTE**: swap `cu130` with `cu126` or `cu128` for different CUDA versions and `cp4` with `cp3` to use Cellpose 3.x.*

## To install

### Using (uv) pip

Install `cali` directly without cloning (use python 3.11 or greater):

```bash
(uv) pip install "git+https://github.com/fdrgsp/cali[cp4]"
```

***NOTE**: swap `cp4` with `cp3` to use Cellpose 3.x.*

To use Cellpose with GPU (e.g. on Windows), the correct version of CUDA has to be installed:

```bash
(uv) pip install --index pytorch-cu130=https://download.pytorch.org/whl/cu130 "git+https://github.com/fdrgsp/cali[cp4,cu130]"
```

***NOTE**: swap `cu130` with `cu126` or `cu128` for different CUDA versions and `cp4` with `cp3` to use Cellpose 3.x.*

***NOTE**: [NVIDIA Drivers](https://www.nvidia.com/en-us/drivers/) should be already installed. You can check what version of cuda you need by running:*

```bash
nvidia-smi
```

### Using uv

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

To use Cellpose with GPU (e.g. on Windows), the correct version of CUDA has to be installed. Add a CUDA extra alongside your Cellpose extra:

```bash
uv sync --extra cp4 --extra cu130   # Cellpose 4.x + CUDA 13.0
```

***NOTE**: swap `cu130` with `cu126` or `cu128` for different CUDA versions.*

***NOTE**: [NVIDIA Drivers](https://www.nvidia.com/en-us/drivers/) should be already installed. You can check what version of cuda you need by running:*

```bash
nvidia-smi
```

### NOTES

#### Building on macOS

If you encounter build errors with `oasis-deconv` (especially SDK-related errors), set these environment variables before installing:

```bash
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
export LDFLAGS="-L${SDKROOT}/usr/lib"
```

Then run your installation command.

---

## Overview

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

<img width="800" alt="Screenshot 2026-01-31 at 4 00 05 PM" src="https://github.com/user-attachments/assets/fbd211f2-918e-49a5-b10f-700e8c4bd265" />

#### Extraction Tab

The Extraction tab allows the user to configure fluorescence trace extraction from the segmented ROIs. Parameters include:

- **Neuropil correction**: enable/disable and set neuropil mask parameters
- **ΔF/F and OASIS deconvolution**:
  - window size for ΔF/F calculation
  - percentile for ΔF/F baseline
  - parameters for OASIS deconvolution (or leave on `auto` to use default parameters)
- **Metadata**: frame rate and pixel size
- **Number of Threads**: number of threads used for parallel extraction across wells/FOVs. Keep this number low if you experience memory issues during extraction.

<img width="800" alt="Screenshot 2026-01-31 at 4 00 12 PM" src="https://github.com/user-attachments/assets/639a7c5a-84e1-4f9c-8fc4-c99bff962094" />

#### Analysis Tab

The Analysis tab allows the user to configure analysis of the extracted traces, including:

- **Experiment Type**: since `cali` was designed to work with [micromanager-gui](https://github.com/fdrgsp/micromanager-gui), which supports spatio-temporal optogenetic stimulation, the user can select the `Evoked Activity` experiment type. This enables input of stimulation metadata used during acquisition and allows splitting results into stimulated vs non-stimulated ROIs.
- **Calcium Traces and Peaks Analysis**: parameters for calcium peak detection and analysis of calcium traces.
- **Inferred Spikes**: parameters for analysis of inferred spikes obtained from OASIS deconvolution. This includes detection thresholds, burst detection parameters, and correlation / synchrony analysis between ROIs.
- **Metadata**: additional experiment metadata (e.g. frame rate). The frame rate here is linked to the one in the Extraction tab; changing one will update the other.
- **Number of Threads**: number of threads for running the analysis across wells/FOVs. Keep this low if you experience memory issues.
- **CCG Worker Processes**: number of worker processes for parallel CCG (Cross-Correlogram) computation. CCG computation is the most time-consuming part of FOV analysis and uses multiprocessing to parallelize across ROI pairs. Default is CPU count - 2. Higher values speed up computation but use more memory.

<img width="800" alt="Screenshot 2026-01-31 at 4 00 21 PM" src="https://github.com/user-attachments/assets/fe17ae9b-410f-4579-b532-1342a0598ca1" />

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
- **Inferred Spikes Metrics**: frequency (on thresholded and/or thresholded rising edges).
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

$$\Delta F/F(t) = \frac{F(t) - F_0(t)}{F_0(t)}$$

where:

- $F(t)$ is the raw fluorescence at time $t$
- $F_0(t)$ is the baseline fluorescence estimated from a sliding window

The baseline $F_0(t)$ is computed by taking the 10th percentile of the fluorescence within a sliding window centered at each time point.

**GUI Parameters**:

- **Window Size** (s): size of the sliding window for baseline calculation.
- **Percentile**: percentile of fluorescence values used as the baseline F₀ (default: 10).

#### OASIS Deconvolution

**Purpose**: `cali` uses the [OASIS algorithm](https://github.com/j-friedrich/OASIS) (Friedrich et al., 2017) to deconvolve the ΔF/F signal and infer the underlying spike activity.

For each ROI, `OASIS` is used to:

- estimate the noise level of the ΔF/F trace (later used for calcium peak detection)
- obtain both a deconvolved (denoised) ΔF/F trace and an inferred spike train.

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

**Purpose**: Measure linear relationships between calcium activity patterns across ROIs using ΔF/F or deconvolved ΔF/F traces.

**Calculation**:

1. **Input**: ΔF/F traces from all ROIs (continuous raw calcium signals or deconvolved (denoised) by OASIS)
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
   A circular shift (also called circular permutation) wraps the spike train around itself: elements that "fall off" one end reappear at the other. For example, with a spike train `[0,0,1,0,1,1,0,0]` and a shift of 3 positions:
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
- **CCG Baseline Shuffles**: Number of circular shift surrogates for baseline correction (default: 30). More shuffles = more accurate baseline but slower computation.
- **Enable Rising Edge Analysis**: When enabled, performs additional CCG analysis on spike onset times (0→1 transitions). Approximately doubles CCG computation time.

**Summary Metric**:
Global synchrony = median of the mean CCG per ROI (row means), excluding the diagonal.

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
