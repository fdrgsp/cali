# CASCADE extraction and storage measurements

These are controlled CPU measurements for Step 10, not a completed release gate.
The machine-readable results are in `cascade_full_mode_cpu_benchmark.json`.
The upstream reference remains the default; the cached service is experimental.

## Scope and reproduction

`benchmark_cascade_extraction.py` starts each output/backend combination in a
fresh process. It runs actual image-to-DFF extraction, the mandatory OASIS pass,
the selected CASCADE backend, ROI/FOV calcium analysis, and the normal FOV staging
and SQLite commit functions. It starts from known masks and initialized databases.
Cold means one FOV with a newly prepared backend; warm means four concurrent FOVs
reusing that backend in a second batch. Image generation/loading is inside the
extraction timer. Detection, database/schema setup, optional-package initialization,
artifact checks/writes, CSV export and GUI rendering are outside that timer.
Package initialization is reported separately. Aggregate stage timings can overlap
across FOV workers; caller timings include inference queue/lock waiting. ROI calcium
function timings are a subset of trace-finalization time.

Images use seed 9183, 30 Hz uniform runner timestamps, separate image buffers per
caller, nonoverlapping known masks, no neuropil correction and a configured 1 s
OASIS decay constant. Fixing tau is deliberate: upstream automatic AR estimation
randomizes invalid coefficients, which prevents exact comparisons across fresh
processes and concurrent FOV scheduling. This benchmark does not change that
application behavior. Every FOV is checked, with exact comparisons across output
selections and backends for denoised calcium, calcium noise and stored spike arrays.

The reference, service and benchmark-only lock alternative use the same pinned
30 Hz model/manifest. The service owns Torch on one worker with one queued request;
the lock alternative serializes the cached predictor across FOV threads. RSS is
the process lifetime peak and includes validation artifacts and result graphs.
The reported retained-payload lower bound includes each blocked caller's source
image, ROI masks, Phase-A raw/DFF arrays, stacked DFF and mandatory OASIS denoised/
spike arrays. It excludes Python/container overhead and inference temporaries.

Run from the repository with a Python environment containing `cali[cascade]` and
the verified models already downloaded:

```sh
export PYTEST_RUNNING=1
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR=/tmp/cali-mpl-cache
export CALI_CASCADE_MODELS=/tmp/cali-cascade-real-models
cascade_bench_python=/tmp/cali-cascade-clean/bin/python

"$cascade_bench_python" _dev/benchmark_cascade_extraction.py \
  --model-dir "$CALI_CASCADE_MODELS" --output-dir /tmp/cali-full-short-release \
  --rois 8 --frames 256 --fovs 4 --workers 4

"$cascade_bench_python" _dev/benchmark_cascade_extraction.py \
  --model-dir "$CALI_CASCADE_MODELS" --output-dir /tmp/cali-full-long-release \
  --rois 32 --frames 2048 --fovs 4 --workers 4

"$cascade_bench_python" _dev/benchmark_cascade_storage.py \
  --workload short --output-dir /tmp/cali-storage-short-release --fovs 4

"$cascade_bench_python" _dev/benchmark_cascade_storage.py \
  --workload long --output-dir /tmp/cali-storage-long-release --fovs 4 \
  --prediction /tmp/cali-cascade-p3b-benchmark/service-prediction.npz \
  --reference-prediction /tmp/cali-cascade-p3b-benchmark/reference-prediction.npz
```

Use fresh output directories: the scripts refuse to overwrite existing databases.
Run the measured workloads sequentially, without concurrent test/build jobs.
The long storage inputs can be recreated using `benchmark_cascade_inference.py`
with its default 100 ROIs, 6000 frames and seed 9183. Storage checks require cached
predictions to equal the independent reference after its float32 conversion and
record both NPZ checksums. The short workload uses the checked real upstream golden
excerpt, repeated to 100 ROIs. Storage FOVs also repeat these arrays; projections
to 96 FOVs are arithmetic estimates, not measurements on an independent plate.

## Bugs found by the complete extraction workload

The coordinator previously passed already-completed futures back to
`wait(..., FIRST_COMPLETED)`. Once the first FOV completed, the coordinator spun
and competed with unfinished workers. It now waits only on pending futures.
A regression test blocks a remaining worker and verifies the coordinator sleeps
instead of repeatedly polling. The pre-fix service sample used the same short
workload, fixed tau and thread environment; its warm four-FOV time was 23.26 s.
The artifact retains that sample and identifies it separately from acceptance runs.

Calcium-only FOV analysis also read OASIS-specific CCG settings unconditionally.
With at least two active ROIs, CASCADE-only settings therefore raised an error even
when spike analysis was disabled. Those reads now happen inside the enabled spike
branch. A regression exercises all three extraction modes with active cells and
requires identical calcium correlation/burst products and no spike analyses.

## Complete-mode CPU results

macOS arm64, Python 3.13, Torch 2.14.1; one Torch thread was verified inside
inference, not just on the calling thread. The short workload is 8 ROIs × 256
frames; the longer workload is 32 ROIs × 2048 frames. Cold columns cover one FOV;
warm columns cover four FOVs with four extraction workers. All timings are seconds.
These are single measurements, without confidence intervals.

| Output | Backend | Short cold | Short warm | Longer cold | Longer warm | Longer peak RSS MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| OASIS | reference | 0.285 | 0.143 | 0.574 | 1.564 | 565.89 |
| CASCADE | reference | 0.573 | 1.592 | 10.440 | 40.386 | 731.61 |
| dual | reference | 0.498 | 1.567 | 10.608 | 41.242 | 699.23 |
| CASCADE | service | 0.435 | 1.039 | 9.738 | 37.893 | 634.41 |
| dual | service | 0.412 | 1.069 | 10.072 | 38.126 | 578.59 |
| CASCADE | cached lock | 0.487 | 1.022 | 9.770 | 39.292 | 571.39 |
| dual | cached lock | 0.402 | 0.984 | 10.103 | 38.231 | 575.47 |

All stored-array differences are exactly zero. The short reference loads 25 weights
cold and 100 warm; the longer reference loads 40 cold and 160 warm. Both cached
alternatives load the cold ensembles once and load **zero** weights warm. All
CASCADE paths reach four retained callers; the service queue peaks at its bound of
one. The longer workload retains at least 11.52 MiB of source/trace/mask arrays per
caller before inference temporaries or Python overhead.

The service improves longer warm CASCADE-only wall time by 6.2% and dual by 7.6%
against the reference. Service versus lock differences are smaller and these single
samples do not establish a stable ordering. Both cached alternatives stay below
the earlier 256 MiB incremental-RSS target on these workloads; the longer reference
exceeds it. This is useful partial evidence, not a material full-plate speedup or
the required 100-ROI/6000-frame extraction acceptance.

## Storage decision and remaining gates

The storage benchmark writes actual `SpikeTrace.values` JSON through SQLModel,
measures payload bytes with SQLite, records vacuumed `.cali` sizes and maintenance
cost separately, then measures ORM reads, actual raw CSV/metadata exports and
NumPy conversion/valid-interval slicing. The plot timing is data preparation only;
CASCADE plot integration and GUI rendering remain pending. Compatibility duplication
copies the canonical OASIS JSON into the physical legacy `trace.inferred_spikes`
column, without changing historical migrations.

The proposed release budget is **512 MiB for all retained spike arrays**, including
temporary compatibility duplication, at 96 FOVs × 100 ROIs × 6000 frames. Base
calcium traces, indices and analysis products are additional. This caps the added
spike storage at roughly 5.3 MiB per FOV rather than permitting gigabyte-scale JSON
growth. The numerical budget is **zero additional error** relative to already-stored
CASCADE float32 and OASIS float64 values. Six-decimal rounding is measured but is
not selected: small per-sample error can still change a configurable AP-threshold
crossing. The benchmark reports common AP fractions and constructs a fraction
between an original and rounded sample to demonstrate the sensitivity.

A benchmark-only candidate replaces each JSON payload with a zlib-compressed BLOB
and metadata containing version, dtype, shape and SHA-256 of uncompressed bytes.
CASCADE stays float32 and OASIS stays float64. Every decoded sample must equal its
original JSON value exactly; sums, rates and threshold crossings consequently do
not change. Candidate files are not application-readable databases. Their SQL
read/decode timings exclude the separately reported comparisons to original JSON;
they are not ORM/GUI acceptance timings.

Long storage results below are **four FOVs × 100 ROIs × 6000 frames**. Projected
payload columns scale the measured arrays to 96 FOVs and include legacy duplication
where applicable. Database columns are measured four-FOV files, not projections.

| Output | JSON payload, 96 FOVs MiB | Candidate payload, 96 FOVs MiB | JSON DB MiB | Candidate DB MiB | JSON write s | ORM read s | Export s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OASIS | 300.19 | 14.75 | 190.74 | 178.82 | 3.313 | 0.120 | 20.476 |
| CASCADE | 1144.26 | 189.07 | 225.96 | 186.05 | 3.811 | 0.418 | 24.312 |
| dual | 1444.45 | 203.83 | 238.72 | 187.00 | 4.237 | 0.565 | 29.816 |
| dual + legacy OASIS JSON | 1744.64 | 504.02 | 251.22 | 199.50 | 4.088 | 0.558 | 29.723 |

Legacy duplication write and file maintenance are additional, reported separately
in the artifact. Plot-array preparation costs 0.043–0.088 s after the ORM read.
The short golden workload's JSON files are 8.93–10.89 MiB; candidates are
8.26–9.07 MiB. Their projected spike payloads all fit the budget. The long workload
fails for every CASCADE-containing JSON mode. The lossless candidate fits, but
504.02 MiB with compatibility duplication leaves only about 8 MiB of headroom:
independent data may have different sparsity/compressibility and must be checked.

Six-decimal rounding on the long source matrix has maximum per-value error
`5.0e-7`, per-ROI sum error `6.13e-5` and mean-rate error `3.10e-7 Hz`.
The sampled AP fractions 0.1, 0.2, 0.5 and 1.0 do not change crossings, but a valid
fraction of `0.01388311736` changes three samples. Lossless compression avoids this
entire tradeoff and preserves the existing float32 golden agreement.

If the measured long JSON workload exceeds this budget, release stays blocked until
the application has a versioned codec migration, legacy JSON reads, integrity and
rollback tests, and consumer reads through one abstraction. Repeat the storage gate
against that implemented codec before proceeding to GUI exposure.

The bundled eight-position imaging fixture contains only two unique biological
positions repeated across wells. Its timestamps have roughly 300, 400 and 500 ms
intervals at stimulation boundaries, versus a 100 ms median; it fails the 1%
uniform-timing requirement. The other real fixture has only ten frames, below the
model's minimum. Neither is an acceptable representative real-plate benchmark.
No timestamps were replaced or resampled to bypass preflight.

CASCADE spike analysis does not exist yet, so its ROI/FOV and dual-method analysis
costs cannot be certified at this step. Measure those after P6 and before release.
An independently recorded plate, the 100 × 6000 end-to-end workload and GPU
acceptance also remain pending. These partial CPU results do not promote the
cached backend to the default.

Validation: **1901 passed, 13 skipped** in the full base/GUI suite; **95 passed**
against a freshly built and installed wheel with actual pretrained CASCADE tests
enabled. Ruff passes, and mypy remains at 353 pre-existing diagnostics with no
additions. The two test-migrated database fixtures were restored after the suite.
