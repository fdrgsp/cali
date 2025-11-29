# TODO

- numba?
- fix csv export of analysis results

- CaImAn

- Cascade (instead of OASIS)
    - https://github.com/HelmchenLabSoftware/Cascade
    - https://www.scientifica.uk.com/learning-zone/how-to-compute-δf-f-from-calcium-imaging-data?utm_source=chatgpt.com


ask to claude:
- the logic in CaliGui to load the data is dolid but the code is repetitive. first create a test that cover all the logic, then try to not copy paste the same code multiple times.
- I want to add to add to #sym:_ImageViewer the fact that if I keep pressed ctrl whyle clicking on the roi, I sulect multiple of them. I guess this would mean to update the #sym:valueChanged signal...what do you think? This will allow to hoghlight the traces for exaole in the plots of all the roi I highlight.
- when we cancel, we need to make sure that the position not detected/extracted/analyzed should be removed form the #sym:CaliResult no?
- how to fix laggy hover cursor in plots when there are many rois/traces/lines...
- general improvement of how the database is handled durynt cali pipeline