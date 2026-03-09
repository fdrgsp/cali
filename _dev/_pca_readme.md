#### PCA Scatter

The PCA Scatter plots reduce the high-dimensional per-FOV activity profile to two dimensions (PC1 and PC2) so that FOVs with similar overall activity patterns cluster together visually.

**Feature matrix**

One row is built per FOV. The columns are:

- Per-ROI scalars averaged to FOV level: calcium peak amplitude, calcium peak frequency, inter-event interval (IEI), inferred spike frequency (thresholded), inferred spike frequency (rising edges), and mean cell size.
- Percentage of active ROIs (% active cells).
- FOV-level burst statistics from inferred spikes: burst count, average burst duration, and average inter-burst interval.

**Preprocessing**

Before PCA is applied, the feature matrix is cleaned and normalised in two steps:

1. Any column where every row is missing (all-NaN) is dropped entirely — this typically removes features that were not computed for the current run (e.g. burst metrics when no bursts were detected).
2. Remaining missing values are imputed with the column median, so that FOVs that individually lack a metric (e.g. no peaks detected) still contribute to the scatter.
3. All remaining columns are z-scored (zero mean, unit variance) so that features on very different numerical scales contribute equally to the PCA.

**PCA**

A two-component PCA is fitted to the z-scored feature matrix. Each FOV is projected onto the first two principal components (PC1 and PC2). The axis labels in the scatter plot show the percentage of total variance explained by each component.

**Visualization**

Each FOV appears as one dot in the scatter plot, coloured by its condition assignment. The legend identifies each condition.

**Stim vs NonStim variant**

The **PCA Scatter (Stim vs NonStim)** plot is available for `Evoked Activity` runs. In this variant, each FOV is split into two rows before PCA: one computed from its stimulated ROIs only, and one from its non-stimulated ROIs only. This produces two scatter points per FOV, and the colour encodes the condition plus stimulation status — allowing a direct visual comparison of how the overall activity profile of stimulated and non-stimulated sub-populations differs across the FOV population.
