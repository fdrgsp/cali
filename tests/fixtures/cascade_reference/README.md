This small oracle contains the first two ROIs and first 256 frames of the
Allen Brain Observatory example distributed with CascadeTorch. The original
MAT file, immutable upstream revision, selected slice, model manifest, package
source identity, CPU environment, and fixture checksum are recorded in
`manifest.json`. Preserve the upstream GPL-3.0 repository provenance for this
derived test data.

`real_excerpt.npz` stores fractional DFF inputs, model-rate noise estimates, and
direct upstream predictions before float32 storage conversion. Its 30 Hz model
uses the half-open valid interval `[32, 224)`. No pretrained weights or GPL
implementation code are bundled here.

Regenerate with an installed `cali[cascade]` environment, the exact source MAT,
and a separately downloaded model:

```bash
cali cascade-download Global_EXC_30Hz_smoothing25ms --model-dir /tmp/cascade-models \
  --expected-manifest ac8954174ba0a01a2d929a7e8b3fc7e3a4365d5c01f5e262d2b597822fcae184
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python _dev/generate_cascade_reference.py \
  /path/to/Experiment_552195520_excerpt.mat --model-dir /tmp/cascade-models
```

The generator calls external upstream APIs directly. The software gate compares
the adapter with this fixture and fresh implicit-noise upstream predictions at
`rtol=1e-5, atol=1e-6`; current same-environment predictions also agree exactly
after float32 conversion. This checks numerical equivalence, not biological
calibration on a particular experimental preparation.
