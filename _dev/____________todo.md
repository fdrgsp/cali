# TODO

- numba?
- fix csv export of analysis results

- CaImAn

- Cascade (instead of OASIS)
    - https://github.com/HelmchenLabSoftware/Cascade
    - https://www.scientifica.uk.com/learning-zone/how-to-compute-δf-f-from-calcium-imaging-data?utm_source=chatgpt.com


ask to claude:

- the logic in CaliGui to load the data is solid but the code is repetitive. first create a test that cover all the logic, then try to not copy paste the same code multiple times.

- I want to add to add to #sym:_ImageViewer the fact that if I keep pressed ctrl whyle clicking on the roi, I sulect multiple of them. I guess this would mean to update the #sym:valueChanged signal...what do you think? This will allow to hoghlight the traces for exaole in the plots of all the roi I highlight.

- general improvement of how the database is handled during cali pipeline

- menu to save and load settings in CaliGui, they are dataclasses so we can save as json

- replece red color in plot combo with unclickable item

- add correlation plots to evoked experiments (one established is correct)

- ⚠️ fix bug: after deleting one run, I get some unique trace error...

- add a loading bar when switching between runs in CaliGui

- code and widget to load own label images for segmentation

- ⁇ to simplify the code, is it worth to have the runs in a different db???

- export tab to export tables and labels
