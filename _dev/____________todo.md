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

- ⚡️ menu to save and load settings in CaliGui, they are dataclasses so we can save as json

- code and widget to load own label images for segmentation

- ⁇ to simplify the code, is it worth to have the runs in a different db???

- ⚡️ export tab to export tables and labels

- add a button to reset settings in each tab to default values

- is good to have a button with a lock symbol in the _SingleWellGraphWidget that will apply the same roi to all the widgets.

- add "database only" tab in the init widget to open a database without the actual data loading so only the plotting and analysis can be done.

- have a new gui that allows to only open a list of databases and plot the same metrics/results for all of them...like for instance if I have 10 databases form 10 different recordings on the same plate but done in different day we can see the variation over time.

- is there something wrong with the multithreading? it seems slower if I multithread...

- ask to carefully evaluate correlation code and make a detailed markdown with description

- make overall markdown of cali runner pipeline

- inferred spike raster is not continuous...even if there are no gap within the timepoint...