# TODO

- fix csv export of analysis results

- CaImAn

- Cascade (instead of OASIS)
    - https://github.com/HelmchenLabSoftware/Cascade
    - https://www.scientifica.uk.com/learning-zone/how-to-compute-δf-f-from-calcium-imaging-data?utm_source=chatgpt.com

ask to claude:

- ‼ Multi-Plot ‼

- I want to add to add to #sym:_ImageViewer the fact that if I keep pressed ctrl whyle clicking on the roi, I sulect multiple of them. I guess this would mean to update the #sym:valueChanged signal...what do you think? This will allow to hoghlight the traces for exaole in the plots of all the roi I highlight.

- general improvement of how the database is handled during cali pipeline

- menu to save and load settings in CaliGui, they are dataclasses so we can save as json

- code and widget to load own label images for segmentation

- ⁇ to simplify the code, is it worth to have the runs in a different db???

- export tab to export tables and labels

- add a button to reset settings in each tab to default values

- is good to have a button with a lock symbol in the _SingleWellGraphWidget that will apply the same roi to all the widgets.

- add "database only" tab in the init widget to open a database without the actual data loading so only the plotting and analysis can be done.

- the plate map is in the analysis settings gui but is not in the analysis setting object. this means that we cannot have different runs with different plate maps because the run will always use the last one saved in the analysis settings. How can we enable this? should we modify the model? or have the plate map as settings as well? what is the best approach? please evaluate the best options and fix it.

- I do not see any plot when I select these options in the multi-plot combo: #file:_multi_well_bar_plot.py:790-945. we need to fix them. How come the test pass? tests shpould fail if we do not see any data in the plot. If they do not fail it means they do not cover correctly the functionality therefore please mofdify all plot tests to actually assert data in the graph.
