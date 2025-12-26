# TODO

- fix csv export of analysis results

- CaImAn

- Cascade (instead of OASIS)
    - https://github.com/HelmchenLabSoftware/Cascade
    - https://www.scientifica.uk.com/learning-zone/how-to-compute-δf-f-from-calcium-imaging-data?utm_source=chatgpt.com

ask to claude:

- ‼ Multi-Plot ‼

- I want to add to add to #sym:_ImageViewer the fact that if I keep pressed ctrl whyle clicking on the roi, I sulect multiple of them. I guess this would mean to update the #sym:valueChanged signal...what do you think? This will allow to hoghlight the traces for exaole in the plots of all the roi I highlight.

- code and widget to load own label images for segmentation

- export csv tables

- export labels

- is good to have a button with a lock symbol in the _SingleWellGraphWidget that will apply the same roi to all the widgets.

- add "database only" tab in the init widget to open a database without the actual data loading so only the plotting and analysis can be done.

- have a new gui that allows to only open a list of databases and plot the same metrics/results for all of them...like for instance if I have 10 databases form 10 different recordings on the same plate but done in different day we can see the variation over time.

- link image viewer with plots

- the run option should be updated depending on the position input in the position to analyze widget

- fix: titles correlation are too long

- add stim vs non-stim calcium peaks raster
- add sotrted stim vs non-stim calcium and spikes heatmap to plots

- add plot similar to plt.stem for spikes
- add plot similar to plt.stem for spikes to stim vs non-stim plots
    is it possible to add in #file:spikes and in #file:_plot_evoked_experiment_data_plots.py another type of inferred spike traces plot similar to matplotlib.stem?
    the one in n #file:spikes should be one with raw and one thresholded traces and should be added to category="Inferred Spikes Traces".

    In #file:_plot_evoked_experiment_data_plots.py  make it colored as for the other green stim magenta non-stim. do it only on thresholded data. the add it to the category="Evoked Experiment".
