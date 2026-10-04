# CASCADE extraction, analysis and storage measurements

These are controlled CPU measurements for Step 10, not completed release acceptance.
The historical JSON/prototype results are in `cascade_full_mode_cpu_benchmark.json`
(commit `0ed3443`); the production codec measurements are in
`cascade_trace_codec_cpu_benchmark.json`.
The upstream reference remains the default; the cached service is experimental.

The current full-analysis measurements are in `cascade_analysis_cpu_benchmark.json`.
Its database audits supersede the historical extraction benchmark's warm persistence,
database-size and complete-wall-time figures: those archived warm databases retained
only 8–9 of 32 requested short traces and 33 of 128 requested long traces. The separate
production codec storage/migration benchmark retains its independently checked row counts.

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

## Historical JSON/prototype storage decision

The original storage benchmark at `0ed3443` writes `SpikeTrace.values` JSON through SQLModel,
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

At this historical step CASCADE spike analysis was unavailable. P6d now enables it;
the complete controlled analysis measurements below supersede that limitation.
An independently recorded plate, the 100 × 6000 end-to-end workload and GPU
acceptance also remain pending. These partial CPU results do not promote the
cached backend to the default.

Validation: **1901 passed, 13 skipped** in the full base/GUI suite; **95 passed**
against a freshly built and installed wheel with actual pretrained CASCADE tests
enabled. Ruff passes, and mypy remains at 353 pre-existing diagnostics with no
additions. The two test-migrated database fixtures were restored after the suite.

## Complete method-bound analysis measurements

The benchmark also accepts `--sample-memory` for one-second simultaneous RSS
sampling of the measured process and its descendants, excluding the sampler's own
`ps` subprocess. It separates extraction/analysis, validation, persistence and offline
stages. Summed RSS counts shared pages in each process and can miss short peaks;
the parent lifetime high-water mark remains recorded separately. Resource trackers
are included and can exist even when no FOV analysis pool is created. The sampler
requires the Unix `ps` command, consistent with this benchmark's `resource` usage.

Every new report includes persistence row counts and exact comparisons of stored
denoised/spike samples and scalar calcium noise. Independently replay these checks
without opening a migration-capable database connection:

```sh
"$cascade_bench_python" _dev/audit_cascade_extraction.py \
  --input-dir /tmp/cali-analysis-long-full \
  --output /tmp/cali-analysis-long-full-audit.json
```

The auditor supports all-case and single-case output directories, opens SQLite
read-only, checks canonical inference ownership, decodes the production spike
codec, compares every sample with the saved NPZ arrays and records full database
checksums. Missing rows and a changed sample with a valid codec checksum both fail.

`benchmark_cascade_extraction.py --analysis full` now includes default OASIS/CASCADE
ROI analysis, independent FOV populations, CCG/jitter/bursts and offline re-analysis
of stored traces. The matching `--analysis calcium` runs use the same inputs and
physical output selections. Defaults are 30 Hz, method-specific default thresholds,
20 CCG shuffles, rising-edge analysis disabled, one FOV analysis process, four
extraction workers and one offline ROI analysis thread. All settings appear in the
artifact. Both workloads use one cold FOV and four warm FOVs; short is 8 × 256,
long is 32 × 2048. Each combination runs in a fresh process, sequentially so cases
do not contend with other benchmarks. These are single samples without uncertainty
intervals, and cold refers to inference preparation rather than an empty disk/Numba cache.

ROI timers intercept the shared helpers, including one timer per spike method;
FOV timers separate calcium from each spike population. FOV spike timers include
the real multiprocessing pool startup/join costs. Aggregate ROI times overlap across
extraction workers, and helper times are included in their enclosing finalization/FOV
timers. Offline timing includes the analysis runner and ROI/FOV calculations, while
database loading, result checks and offline persistence are excluded. It performs no
image reads, OASIS inference, CASCADE inference or model loads.

The benchmark now detaches worker inputs as the public runner does. Its earlier
attached inputs let one FOV commit expire another FOV's ROI collection and lose
private staged products. Every current database is audited for all requested FOVs,
ROI traces, analysis rows and method children. Persisted arrays must exactly match
the extracted arrays. Shared calcium and each method's deterministic ROI/FOV products
must match across single/dual outputs, reference/service/lock dispatch, persistence
and offline re-analysis. Only shuffled CCG z-scores/significant-pair fractions are
excluded from comparisons; production random shuffles remain unchanged. Raw CCG,
lags, jitter, membership, valid intervals and all other scientific fields are checked.

Measurements also exposed a legitimate flush-order bug: the trace normalizer can
merge a transient inference row before its staged FOV result is attached. FOV binding
now accepts only a compatible transient alias from the same extraction and rebinds
it to the stored canonical row; different owners, persisted identities and known
metadata conflicts still fail. The regression tests cover both flush orders and
negative provenance cases.

The table uses the corrected long workload: **four FOVs × 32 ROIs × 2048 frames**.
It records complete warm wall time (extraction/analysis plus persistence), the
separate persisted-graph commit time, aggregate method-bound FOV spike analysis,
and offline re-analysis. All times are seconds.

| Output | Backend | Calcium-only complete | Full complete | Persistence | FOV spike analysis | Offline full |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| OASIS | reference | 1.841 | 4.153 | 0.549 | 2.197 | 2.333 |
| CASCADE | reference | 39.284 | 41.352 | 0.550 | 2.204 | 2.336 |
| dual | reference | 39.292 | 43.917 | 0.703 | 4.407 | 4.594 |
| CASCADE | service | 36.278 | 38.683 | 0.566 | 2.204 | 2.335 |
| dual | service | 36.511 | 41.175 | 0.718 | 4.407 | 4.597 |
| CASCADE | cached lock | 36.293 | 38.637 | 0.575 | 2.207 | 2.334 |
| dual | cached lock | 36.492 | 41.074 | 0.711 | 4.403 | 4.592 |

Long populations contain **31 OASIS-active ROIs / 465 pairs** and
**32 CASCADE-active ROIs / 496 pairs** per FOV. Short populations contain
8 active ROIs / 28 pairs for each method. Long aggregate ROI spike helper times
are 0.033–0.170 s for single outputs and 0.069–0.087 s for dual outputs; they
overlap across extraction threads. Population/CCG work dominates spike analysis.

The first complete measurements started a one-worker pool for every large
population, despite the default `n_processes=1`. That default now invokes the same
pair workers directly, with identical ordering, normalization and matrix assembly;
two or more workers retain multiprocessing. The separate small-population algorithm
is unchanged. The before/after long reference comparisons are:

| Output | Offline before | Offline after | Complete before | Complete after |
| --- | ---: | ---: | ---: | ---: |
| OASIS | 8.249 | 2.333 | 10.030 | 4.153 |
| CASCADE | 8.285 | 2.336 | 47.658 | 41.352 |
| dual | 16.356 | 4.594 | 55.439 | 43.917 |

This reduces default offline analysis by about **72%** in these samples, with
identical deterministic scientific fingerprints in every backend and phase. A
regression with unequal ROI event counts checks the pair normalization and lag
conventions. The stochastic CCG fields remain computed and retain their existing
random-shuffle behavior. The discarded diagnostic that used the small-population
algorithm is excluded from the artifact because it changed large-population matrices.

The artifact audits **70 measured databases**, including the before-optimization
baseline. Every requested trace, analysis row and method child is present, inference
rows share their canonical owner/method identity, and all stored spike/denoised/noise
values equal the extracted arrays exactly. Shared calcium fingerprints match with
spike analysis enabled and disabled in all cases. The separate archived audit records
the incomplete warm databases and their byte counts/checksums; their historical
complete/persistence/storage figures must not be used as four-FOV acceptance.

Reproduce final controlled measurements with the environment above, choosing a
fresh output directory each time. Each command runs all seven cases sequentially:

```sh
for cascade_analysis in calcium full; do
  "$cascade_bench_python" _dev/benchmark_cascade_extraction.py \
    --analysis "$cascade_analysis" --analysis-processes 1 --ccg-shuffles 20 \
    --model-dir "$CALI_CASCADE_MODELS" \
    --output-dir "/tmp/cali-analysis-short-$cascade_analysis" \
    --rois 8 --frames 256 --fovs 4 --workers 4
  "$cascade_bench_python" _dev/benchmark_cascade_extraction.py \
    --analysis "$cascade_analysis" --analysis-processes 1 --ccg-shuffles 20 \
    --model-dir "$CALI_CASCADE_MODELS" \
    --output-dir "/tmp/cali-analysis-long-$cascade_analysis" \
    --rois 32 --frames 2048 --fovs 4 --workers 4
done
```

Before-optimization results retain their runtime module hashes and settings in
`benchmarks.long.full_before_single_worker`. Final runs use the direct single-worker
implementation; they do not recreate that historical pool-startup baseline.

Parent peak RSS includes artifact validation and offline result graphs. The final
one-process path spawns no FOV workers, while the before-optimization measurements
exclude their worker RSS. Long full service increments are **313.81 MiB CASCADE** and
**334.66 MiB dual**, above the earlier 256 MiB incremental comparison target. Reference
increments are 440.28/468.84 MiB and the lock alternative is 422.80/444.23 MiB. These
complete-graph samples supersede the historical incomplete-graph RSS evidence; they
do not isolate inference-only allocations or certify a many-process memory budget.
The independent uniformly timed real plate, 100 × 6000 complete image workload,
GPU acceptance and independent codec compressibility gates remain pending. The cached
service stays experimental and the released GUI remains gated.

Final regression validation is recorded in the migration plan.


## Production codec storage and migration

Schema 11 now uses the production lossless codec for `SpikeTrace.values`. Its inline
BLOB header carries version, dtype, shape and a checksum covering metadata plus raw
array bytes, followed by zlib-compressed data. Every ORM consumer receives a numeric
list through the same decoding boundary; portable JSON snapshots stay ordinary lists.
Legacy JSON reads remain supported. Float32 is used only when every sample round-trips
exactly; doubles that need float64 and historical nonfinite values retain float64.
The inference provenance's dtype remains unchanged by the physical storage decision.

The table measures the actual codec through SQLModel, normal FOV commit, ORM reads,
CSV/sidecar export and valid-interval NumPy preparation. All source samples are checked
exactly in every FOV. These are single CPU samples on the same environment/workload as
above, without confidence intervals. Projections include legacy copies when present;
base calcium arrays and other records are additional.

| Output | Codec payload, 96 FOVs MiB | Actual 4-FOV DB MiB | Write s | ORM read s | CSV export s |
| --- | ---: | ---: | ---: | ---: | ---: |
| OASIS | 14.85 | 178.82 | 4.121 | 0.052 | 21.001 |
| CASCADE | 189.18 | 186.05 | 4.299 | 0.070 | 25.041 |
| dual | 204.03 | 187.01 | 5.604 | 0.121 | 28.335 |
| dual + legacy OASIS JSON | 504.22 | 199.52 | 5.198 | 0.119 | 28.534 |

The codec passes the controlled **512 MiB** spike-payload budget with **zero** added
sample, sum, rate or threshold-crossing error. All 400 CASCADE rows use float32. Of
400 OASIS rows, 372 require float64 and 28 round-trip exactly as float32. Canonical
writes take longer than the historical JSON samples because compression/validation
cost is included; dual ORM reads fall from 0.565 s to 0.121 s in these samples. CSV
export still emits numeric text and remains a substantial cost. Array preparation
is 0.043–0.088 s after ORM loading. Short workload measurements also pass and appear
in the artifact. The compatibility case leaves only **7.78 MiB** of payload headroom;
independent input sparsity/compressibility must still be checked.

The migration measurement copies the real schema-10 controlled dual/legacy database
from the earlier benchmark and opens the copy through `create_cali_engine()`. The
800 canonical spike rows upgrade in **7.31 s**, with the version update in the same
transaction. Fingerprints normalize only array encoding and require all decoded
float64 bytes and every other stored field—including physical legacy JSON and
unknown provenance—to match exactly. All four CSV files match byte-for-byte, and
both JSON sidecars have identical content (object key ordering is immaterial).
Migration does not shrink the file immediately: freed SQLite pages are reusable.
A separately timed `VACUUM` takes **1.16 s** and reduces the file to **199.52 MiB**.
Original schema-10 inputs and exports remain untouched.

Reproduce after the setup above, using fresh output directories and sequential runs:

```sh
"$cascade_bench_python" _dev/benchmark_cascade_storage.py \
  --storage codec --workload short --output-dir /tmp/cali-codec-storage-short --fovs 4

"$cascade_bench_python" _dev/benchmark_cascade_storage.py \
  --storage codec --workload long --output-dir /tmp/cali-codec-storage-long --fovs 4 \
  --prediction /tmp/cali-cascade-p3b-benchmark/service-prediction.npz \
  --reference-prediction /tmp/cali-cascade-p3b-benchmark/reference-prediction.npz

"$cascade_bench_python" _dev/benchmark_trace_array_migration.py \
  /tmp/cali-storage-long-release/long-dual-legacy-duplication.cali \
  --output-dir /tmp/cali-codec-migrated-long-verified
```

`--storage json` (the benchmark default) selects the historical JSON binding in an
isolated benchmark process, so the earlier JSON reproduction commands still work.
The application always writes the production codec. Prototype candidate sidecars
are retained only for comparison and remain application-unreadable artifacts.
The migration command requires an actual v10 file and its earlier export directory.

Tests cover version/dtype/shape and metadata integrity, bounded decompression,
corrupt/truncated/concatenated streams, precision, interrupted migration rollback
and repair/retry, mixed encodings, ORM writes, snapshots and exports. The optional
installed-wheel CI job now includes the codec/consumer tests. Remaining release
gates above still apply; P6 method-specific CASCADE spike analysis is the next step.

Validation: the full base/GUI suite passes **1944 tests, 13 skipped in 237.55 s**.
A freshly built and installed wheel passes **138 tests in 9.18 s**, including
codec/migration/consumer tests and real pretrained reference/cache/extraction
acceptance. Ruff lint/format pass; mypy remains at 353 existing diagnostics with
no additions. Both test-migrated tracked database fixtures were restored afterward.
