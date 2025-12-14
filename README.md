# cali

A Gui for Calcium Imaging Data Visualization, Segmentation and Analysis

[![CI](https://github.com/fdrgsp/cali/actions/workflows/ci.yml/badge.svg)](https://github.com/fdrgsp/cali/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/fdrgsp/cali/branch/main/graph/badge.svg)](https://codecov.io/gh/fdrgsp/cali)

[🚧 WIP 🚧]

`cali` is package that provides a gui to load calcium imaging timelapse data, segment cells using Cellpose, extract and analyse fluorescence traces and visualize them. It was originally designed to work in combination with [micromanager-gui](https://github.com/fdrgsp/micromanager-gui), an open-source software to control microscopes through `Micro-Manager]` and [pymmcore-plus](https://github.com/pymmcore-plus).

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

#### OASIS Deconvolution

### Analysis

#### Calcium Peak Detection

#### Calcium Burst Detection

#### Inferred Spikes Thresholding

#### Inferred Spikes Burst Detection

#### Inferred Spikes Max-Lag Cross-Correlation

#### Inferred Spikes Jitter Synchrony

