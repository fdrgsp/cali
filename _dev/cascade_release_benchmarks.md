# CASCADE extraction, analysis and storage measurements

These are controlled CPU measurements for Step 10, not completed release acceptance.
The historical JSON/prototype results are in `cascade_full_mode_cpu_benchmark.json`
(commit `0ed3443`); the production codec measurements are in
`cascade_trace_codec_cpu_benchmark.json`.
The upstream reference remains the default; the cached service is experimental.

**Scope (2026-10-05):** the independent real-plate benchmark and CUDA hardware
validation are deferred and are not current release gates. Device coverage is
CPU and MPS. Later "pending" notes about real-plate data or CUDA in this file
are superseded by the Scope decisions in
[`cascade_migration_plan.md`](cascade_migration_plan.md).

## Independent recording command (deferred)

`benchmark_cascade_real_plate.py` prepares the deferred real-plate check using an
existing recording and matching saved detection masks. It opens the source database
read-only, makes a consistent SQLite backup including committed WAL contents, and
migrates only that temporary backup. It never runs detection. Measured outputs go
to fresh databases containing the selected masks and new products; historical results
are not included in their storage figures.

First run the preflight, selecting the experiment, detection settings, extraction
settings and acquisition positions explicitly:

```sh
PYTEST_RUNNING=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 MPLCONFIGDIR=/tmp/cali-mpl-cache \
python _dev/benchmark_cascade_real_plate.py \
  --database /path/to/recording.cali \
  --dataset /path/to/recording.tensorstore.zarr \
  --experiment-id 1 --detection-settings-id 1 --extraction-settings-id 2 \
  --positions 0 1 2 3 \
  --recording-description 'Recording origin, indicator, plate and acquisition protocol' \
  --model Global_EXC_30Hz_smoothing25ms \
  --model-dir /path/to/verified-model-cache \
  --output-dir /tmp/cali-real-plate-preflight --preflight-only
```

Run on macOS/Linux with a Python environment containing `cali[cascade]`; process
memory sampling uses Unix `ps`/`resource` APIs. The command also supports the
production OME-Zarr reader and TIFF collections configured in the selected experiment.
Preflight checks every position's masks, trusted timing, retained length and selected
model rate. Non-uniform acquisition timestamps remain an error even when settings
contain a verified rate. A failure writes `preflight-rejected.json` and runs no inference.
Pixel and metadata hashes bind the actual reader calls to the preflight inputs; each
case must use the same database snapshot, masks, settings, recording and model manifest.
Descriptions record the recording's claimed origin; they do not certify biological
representativeness.

Remove `--preflight-only` and choose a new output directory to run all seven reference,
service and cached-lock cases in fresh CPU processes. Cold uses the first selected FOV;
warm uses all selected FOVs, including that first position, through the same backend.
Use at least four distinct representative positions with `--workers 4` for the planned
concurrency gate. `--analysis-processes 2` selects the additional multiprocessing scope.
Run measurements sequentially without concurrent benchmark/test jobs.

The selected extraction settings are preserved, including neuropil correction,
startup discard and decay constant. Analysis uses full calcium/spike
benchmark defaults, independent method settings, `--ccg-shuffles` (default 20), and
rising-edge analysis disabled. It does not reproduce a saved evoked-analysis protocol.
Both settings objects are saved in each case report. Automatic OASIS AR estimation can
randomize invalid coefficients; a saved zero decay constant can therefore cause exact
cross-process parity to fail. The command does not silently replace that setting.

Each phase records extraction plus complete ROI/FOV analysis, persistence, offline
re-analysis, summed parent/descendant RSS, actual checkpoint loads and runtime/source
identities. The extraction timer includes image loading and provenance hashing;
schema/mask setup, optional-package initialization, artifact verification, exports and
GUI rendering are excluded. Package initialization has a separate duration. Lifetime
peak RSS includes preflight and validation; stage samples have the same summed-RSS
limitations as the controlled benchmark. A warm cached case can legitimately load
additional noise ensembles when a different FOV first needs them.

Every position and ROI label is checked, including non-contiguous labels and differing
ROI counts. Independent SQLite audits verify saved calcium/spike samples, foreign keys
and inference ownership. Offline analysis must match the original deterministic products
and make zero image, model-loading or inference calls. Raw CCG/lag/jitter products are
compared; the four stochastic significance fields are listed as exclusions. Per-phase
noise-QC CSVs keep calcium and CASCADE estimators separate. Storage reports actual
canonical spike bytes plus an arithmetic projection including legacy OASIS JSON to
96 FOVs against the existing 512 MiB budget; this is not a complete-plate measurement.

`cascade_real_plate_harness_validation.json` records a file-reader smoke test of this
command, not independent real-plate acceptance. A suitable independently acquired
recording and review of its performance, compressibility and memory scope are
deferred until such a recording is available. The command never promotes cached
inference or enables CASCADE in the GUI.

## Controlled workloads

### Full-frame CPU/MPS memory scope (2026-10-10)

`cascade_full_frame_memory_validation.json` records new installed-wheel reference
measurements using **6,000 × 512 × 512 uint16 images**, 100 known ROI masks, dual
OASIS/CASCADE output, one extraction worker and one analysis process. Each device
runs one cold and one warm FOV, with full calcium/spike analysis, 20 CCG shuffles,
normal SQLite persistence, and separately timed inference-free offline re-analysis.
The Apple M2 Pro/16 GiB host, package pin and verified model are the same as the
GUI installed-wheel acceptance. CPU threads are fixed at one. Synthetic pixel
generation is included in image loading; these are controlled measurements with
known masks, not independently acquired data or disk-reader throughput tests.

| Device | Cold complete s | Warm complete s | Warm offline s | Parent peak MiB | Sampled summed RSS peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU reference | 141.348 | 140.451 | 36.520 | 4498.48 | 4081.30 |
| MPS reference | 80.254 | 80.696 | 37.549 | 3693.77 | 3686.44 |

Complete time includes preparation, extraction, online analysis and persistence;
it excludes validation and the separately timed offline analysis. Parent lifetime
RSS includes those later stages. The one-second sampler sums simultaneous parent
and descendant RSS, may miss brief peaks, and counts shared pages in each process.
Neither figure certifies unique physical memory or MPS allocator/driver peaks.
Detection and GUI rendering are excluded. These are single cold/warm measurements,
not a repeated performance distribution.

Independent read-only audits verify all four complete databases, each with 100
calcium traces, 200 method-owned spike traces, complete ROI/FOV analyses, and exact
saved samples. CPU/MPS raw/DFF/denoised/time arrays, noise inputs, calcium/OASIS
products, binary spike decisions and deterministic FOV products are exact. CASCADE
sample deviation is at most **2.38 × 10⁻⁷**, within `rtol=1e-5, atol=1e-6`.
The historical device auditor excludes seven additive noise-QC fields; an additional
same-schema check compares all **218** such values exactly. Four stochastic CCG
significance fields remain excluded from scientific parity, as in prior benchmarks.
29/100 controlled ROI noise estimates fall outside the model's [2, 9] coverage and
use the nearest ensembles; this evidence does not establish biological accuracy.
Each phase stores a 42.51 MiB database and reference inference loads 40 checkpoints.
Offline re-analysis makes zero image-loading, inference or checkpoint-loading calls.

The original 40 × 40 images remain useful for trace-level parity, but do not cover
retaining larger frames. One 512 × 512 uint16 stack alone is **2.93 GiB**; four
concurrent stacks require **11.72 GiB** before masks, trace arrays, analysis or
inference. This new evidence covers **one extraction worker and one analysis
process only**. It is not a many-worker memory allowance. On 2026-10-10 the user
accepted **6 GiB parent RSS** for this measured scope and approved normal development
GUI activation using the single-worker profile. Detection and GUI rendering remain
excluded; the target is not a runtime memory limit or physical-RAM minimum. Dedicated
CASCADE CI and installed-wheel GUI checks passed. Distribution licensing review
remains pending. The reference stays default and cached inference stays opt-in
pending real-data evidence. See `cascade_gui_release_profile.json` for the decision.

Reproduce sequentially in an installed `cali[cascade]` environment with fresh
output directories and a verified model cache:

```sh
export PYTEST_RUNNING=1 QT_QPA_PLATFORM=offscreen
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
cascade_bench_python=/path/to/cali-cascade/bin/python

"$cascade_bench_python" _dev/benchmark_cascade_extraction.py \
  --mode dual --backend reference --device cpu \
  --model-dir /path/to/verified-model-cache \
  --output-dir /tmp/cali-full-frame-cpu-new \
  --image-side 512 --image-dtype uint16 --rois 100 --frames 6000 \
  --fovs 1 --workers 1 --analysis full --analysis-processes 1 \
  --ccg-shuffles 20 --sample-memory

"$cascade_bench_python" _dev/benchmark_cascade_extraction.py \
  --mode dual --backend reference --device mps \
  --model-dir /path/to/verified-model-cache \
  --output-dir /tmp/cali-full-frame-mps-new \
  --image-side 512 --image-dtype uint16 --rois 100 --frames 6000 \
  --fovs 1 --workers 1 --analysis full --analysis-processes 1 \
  --ccg-shuffles 20 --sample-memory

"$cascade_bench_python" _dev/audit_cascade_extraction.py \
  --input-dir /tmp/cali-full-frame-cpu-new \
  --output /tmp/cali-full-frame-cpu-new-audit.json
"$cascade_bench_python" _dev/audit_cascade_extraction.py \
  --input-dir /tmp/cali-full-frame-mps-new \
  --output /tmp/cali-full-frame-mps-new-audit.json
```

The recorded device comparison used temporary `report.json` files that wrapped
each completed `dual-reference.json` in a `results` list with its device in
`comparison_policy`, so the existing device auditor could consume the single-case
outputs. The wrappers are metadata, not extra runs. The durable evidence binds
the original single-case reports, installed wheel and production source hashes.
Default small-image float64 samples remain exact when the new flags are omitted.

### Explicit GPU validation

The pretrained tests accept `CALI_CASCADE_TEST_DEVICES=cpu,mps` (or `cpu,cuda`;
CUDA hardware validation is currently deferred).
The default remains CPU. Explicitly requested unavailable devices fail; they are
not skipped and do not fall back. The same tests cover the bundled real trace,
synthetic traces, chunks of 1/37/1024 windows, concurrent service calls, resolved
device provenance, calcium parity across output selections, persistence and
inference-free offline analysis. CPU/GPU predictions use the plan's unchanged
`rtol=1e-5, atol=1e-6`; CPU noise estimates and ensemble selections remain exact.
A pretrained cancellation test stops after the first completed chunk, retries
without reloading the ensemble, and checks cache cleanup on the selected device.

Run the installed wheel with the verified model cache and single CPU threads:

```sh
PYTEST_RUNNING=1 QT_QPA_PLATFORM=offscreen \
CALI_CASCADE_REFERENCE_TESTS=1 CALI_CASCADE_TEST_DEVICES=cpu,mps \
CALI_CASCADE_MODELS=/path/to/verified-model-cache \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
python -I -Werror \
  '-Wignore:Call to deprecated function (or staticmethod) _destroy.:DeprecationWarning:sys' \
  -m pytest --noconftest -q \
  tests/test_cascade_reference.py tests/test_cascade_cached.py \
  tests/test_cascade_extraction.py tests/test_cascade_runner_analysis.py -k pretrained
```

The narrow command-line warning exception above is for the installed NumCodecs
0.15.1 `atexit.register(blosc.destroy)` callback: its deprecated native `_destroy`
wrapper warns from `sys` after pytest restores its filters. This occurs independently
of Torch/MPS inference. No project warning policy or inference-warning filter is changed.
It can be omitted with dependency versions whose shutdown callback is warning-free.

`benchmark_cascade_inference.py --device cpu|mps|cuda` now selects an explicit
device for each isolated reference/cached/service/lock process. Capture a fresh CPU
reference before a GPU comparison:

```sh
# Use the same environment variables as above, running measurements sequentially.
python _dev/benchmark_cascade_inference.py \
  --model-dir /path/to/verified-model-cache \
  --output-dir /tmp/cali-cpu-oracle --mode reference --device cpu \
  --rois 100 --frames 6000 --fovs 1 --workers 1

python _dev/benchmark_cascade_inference.py \
  --model-dir /path/to/verified-model-cache \
  --output-dir /tmp/cali-mps-inference --device mps \
  --cpu-reference /tmp/cali-cpu-oracle \
  --rois 100 --frames 6000 --fovs 4 --workers 4
```

CPU oracle comparison checks matching input DFF hashes, model/manifest/package
identities, dimensions, CPU device and prediction checksum before comparing every
backend at the same tolerances. Every warm FOV also matches its cold result; cached
GPU repetition uses numerical tolerance and CPU repetition stays exact. Existing
prediction/report files are protected against replacement.

Host RSS and GPU allocator figures have separate scopes. MPS reports synchronized
tensor, driver and recommended-memory snapshots at baseline/cold/warm/close; these
are **not MPS peak-memory measurements**. CUDA reports allocator current/peak/reserved
bytes, excluding driver overhead and other processes. Cached parameter counts remain
a separate cap. Host RSS and MPS driver counters can overlap on unified-memory
hardware; adding them does not yield physical peak memory. The existing 256 MiB
host RSS comparison is unchanged and cannot
certify GPU device memory or representative-image retention. Timings include retained
caller payload construction and warm parity checks, while snapshot checks are outside
the warm timer; cold includes device initialization and its baseline snapshot.

This is controlled Phase-B inference with 32 × 32 uint16 retained source images,
not complete image extraction/analysis/persistence or independent plate acceptance.
Keep the reference production default and experimental cached flag until all their
remaining gates pass. GPU numerical checks do not enable the GUI option.

Local MPS validation is recorded in `cascade_mps_validation.json` on the same Apple
M2 Pro/16 GiB host, with Python 3.13 and Torch 2.14.1. All **121 CPU/MPS backend and
runner tests** pass, including real chunk cancellation and retry. Four independent
100 × 6000 prediction audits confirm finite nonnegative float32 samples, zero edge
padding and model/input/file identities. Maximum deviation from the fresh CPU oracle
is **2.68 × 10⁻⁷**; cached/service/lock deviation from MPS reference is **5.96 × 10⁻⁸**.
All warm FOVs match their cold predictions.

| MPS path | Cold FOV (s) | Four warm FOVs (s) | Incremental host peak (MiB) | Within existing 256 MiB comparison |
| --- | ---: | ---: | ---: | --- |
| Reference | 24.275 | 90.065 | 1042.98 | No |
| Cached, one caller | 9.149 | 37.398 | 220.77 | Yes |
| Service, four callers | 10.483 | 37.596 | 182.05 | Yes |
| Cached lock, four callers | 9.400 | 35.511 | 317.22 | No |

The service is **2.396×** faster than the MPS reference here. The lock path takes
about **5.5%** less warm time than the service and fails this host comparison.
Cached paths load 30 checkpoints cold and zero warm, retaining six selected noise ensembles.
MPS cached driver snapshots are 50.72 MiB warm and after close; tensor allocation
returns to zero on close. Retained allocator memory is separate from retained models.
Reference driver snapshots retain about 1034.72 MiB after calls; snapshots do not
reveal the transient peak. The CPU oracle's single warm FOV takes 95.155 s, with
808.97 MiB incremental host peak; its concurrency scope differs from the GPU batch.
No complete-pipeline CPU/GPU speedup or representative memory acceptance is claimed.
Explicit unavailable CUDA selection fails before creating a prediction. Representative
complete GPU and memory scopes and remaining release evidence are still pending; CUDA
hardware and independent real-plate inputs are deferred.

### Complete MPS extraction, analysis and persistence

The complete-image GPU command uses the same installed wheel, model, 100 ROI
masks, 6,000 frames, 40 × 40 float64 images, four extraction workers and one FOV
analysis worker as the controlled CPU workload. It runs all seven output/backend
combinations in fresh processes, with one cold FOV and four warm FOVs. Full spike
analysis uses 20 CCG shuffles and no rising-edge analysis. Image loading, ROI/FOV
analysis and normal SQLite persistence are included; offline re-analysis is timed
separately and must make zero inference/checkpoint-load calls.

```sh
"$cascade_bench_python" _dev/benchmark_cascade_extraction.py \
  --mode all --device mps --model-dir /tmp/cali-cascade-real-models \
  --output-dir /tmp/cali-mps-complete-large \
  --rois 100 --frames 6000 --fovs 4 --workers 4 \
  --analysis full --analysis-processes 1 --sample-memory

"$cascade_bench_python" _dev/audit_cascade_device_parity.py \
  --cpu /tmp/cali-p8-large-full --gpu /tmp/cali-mps-complete-large \
  --output /tmp/cali-mps-complete-device-audit.json
```

The device audit opens both SQLite inputs read-only. It binds extraction/analysis
settings, model/package/runtime identities, every raw/DFF/denoised/time sample,
noise values, frame windows and all stored model provenance except the intentionally
different device. Every ROI's binary threshold decisions, activity, ordering,
lag choices, synchrony, bursts and deterministic FOV population products must be
exact. Only CASCADE prediction samples and ROI expected count/rate allow the
existing `rtol=1e-5, atol=1e-6`. A rounding error within that tolerance still fails
if it changes a threshold crossing. Finite/nonnegative predictions and exact zero
padding are also verified. Four random CCG significance fields remain excluded
from comparisons, as in the CPU benchmark.

The archived CPU oracle predates additive schema-14 noise summaries. The audit
excludes those seven fields from cross-version scientific products, while comparing
all persisted calcium/model noise inputs exactly. This is numerical evidence;
the old CPU and new GPU complete times are not a same-version speed comparison.
Database audits require exact persisted samples and valid run/FOV ownership.
Host RSS includes the complete result graph and validation/offline stages;
one-second simultaneous parent/descendant samples can miss short peaks and count
shared pages more than once. These are host measurements, not GPU allocator peaks.

| Output / backend | Cold complete s | Four warm FOVs s | Warm offline s | Parent peak MiB | Summed RSS peak MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| OASIS / reference (CPU) | 22.243 | 88.373 | 72.534 | 1530.81 | 1551.34 |
| CASCADE / reference | 43.780 | 171.871 | 75.169 | 2141.34 | 2153.50 |
| dual / reference | 61.114 | 251.646 | 145.560 | 2151.16 | 2156.22 |
| CASCADE / service | 30.610 | 120.351 | 74.077 | 1644.53 | 1665.30 |
| dual / service | 48.797 | 193.296 | 146.992 | 1837.50 | 1858.22 |
| CASCADE / cached lock | 31.200 | 121.671 | 74.991 | 1652.11 | 1672.86 |
| dual / cached lock | 50.071 | 195.545 | 144.066 | 1819.67 | 1840.52 |

The service improves complete warm time **1.428×** for CASCADE-only and **1.302×**
for dual output against MPS reference. Its observed times are about 1.1% shorter
than the lock baseline; this single sample does not establish a stable advantage.
All CASCADE paths resolve to MPS in persisted provenance. Reference loads 40
checkpoints cold and 160 warm; cached paths load 40 cold and zero warm, retaining
eight ensembles / 5.25 MiB of parameters with chunks capped at 1,024 windows.
Optimized/reference MPS prediction differences are at most **5.96 × 10⁻⁸**.
CPU/MPS differences are at most **2.38 × 10⁻⁷**, with every binary decision and
deterministic FOV product exact. All 14 large databases contain the complete
requested rows and lossless samples. Dual/legacy spike payload projections are
**493.934 MiB per 96 FOVs**, below the unchanged controlled 512 MiB comparison.
The final strict 12 × 256 smoke adds 14 audited databases; the updated installed
CI test list passes **478 tests in 31.11 s**, including 19 acceptance-guard tests.
Explicit unavailable CUDA fails without a database or prediction.

The complete graph does not pass the earlier **256 MiB inference-only** host
comparison: four callers' retained image/trace/mask lower bound alone is
**385.13 MiB**, before inference and Python/container overhead. The parameter
cache cap is not a whole-pipeline memory budget. A representative image scope,
concurrency policy and accepted complete-pipeline budget still need evaluation.
No cached-default promotion or GUI release gate is opened by this controlled
measurement; remaining release checks stay pending (independent recording deferred).

The final comparator was tightened after timing to require exact FOV products
instead of allowing tolerance in four floating-point FOV fields. Timed extraction,
analysis and persistence code did not change. The original measurement script hash
is retained in the raw report; the final strict smoke and CPU/device audits bind
the final script separately in `cascade_mps_complete_validation.json`.

### Complete CPU extraction and storage

The short/long full-analysis measurements are in `cascade_analysis_cpu_benchmark.json`;
the completed 100 × 6000 image workload and worker-memory comparison are in
`cascade_large_mode_cpu_benchmark.json`. Its v1 spike-payload projection exceeds
512 MiB including legacy OASIS duplication. Schema 13 repairs this controlled
workload with lossless byte-shuffle storage; production migration and fresh-write
evidence is in `cascade_trace_shuffle_cpu_benchmark.json`.
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
The 100 × 6000 complete image workload is recorded below. GPU acceptance and scoped
memory acceptance remain pending; independent uniformly timed real-plate data and its
codec compressibility are deferred. The cached service stays experimental and the released GUI remains gated.

Final regression validation is recorded in the migration plan.


## 100 × 6000 complete image workload and worker memory

The [large-workload artifact](cascade_large_mode_cpu_benchmark.json) records all seven
output/backend cases plus dual/service with two FOV analysis processes. Each runs in a
fresh process, sequentially, with one cold and four warm FOVs. Inputs are deterministic
**6000 × 40 × 40 float64 images**, 100 known four-pixel masks, 30 Hz, fixed one-second
OASIS decay, default method thresholds, 20 CCG shuffles and no rising-edge analysis.
Four extraction workers and one offline ROI analysis thread match the previous setup.
The CPU environment is Apple M2 Pro with 16 GiB RAM, Python 3.13, Torch 2.14.1,
NumPy 2.5.3, SciPy 1.16.3 and Numba 0.68.0. Detection, GUI rendering, offline
persistence and database-load time are excluded. These are controlled single samples, not representative biological-plate acceptance.

The default populations contain 99 OASIS-active ROIs / 4851 pairs and 100 CASCADE-active
ROIs / 4950 pairs per FOV. CASCADE's common valid interval is `[32, 5968)`; OASIS uses
all 6000 retained frames. The model's eight noise levels are 2–9; 27 of 100 synthetic
ROI noise estimates fall outside that coverage and use the recorded nearest ensemble.
This is a workload cost check, not validation of biological accuracy outside training
coverage. Model/config/package identities and all stage settings appear in the artifact.

Warm times below cover four FOVs and are seconds. Complete time includes extraction,
full ROI/FOV analysis and persistence; aggregate FOV spike times are already included
in complete time. Offline timing excludes loading and persistence.

| Output | Backend | Complete | Persistence | FOV spike analysis | Offline |
| --- | --- | ---: | ---: | ---: | ---: |
| OASIS | reference | 88.494 | 4.643 | 71.541 | 72.326 |
| CASCADE | reference | 501.170 | 5.344 | 73.972 | 75.451 |
| dual | reference | 571.307 | 6.458 | 150.419 | 153.257 |
| CASCADE | service | 482.629 | 5.056 | 76.076 | 77.452 |
| dual | service | 509.473 | 6.272 | 143.764 | 144.977 |
| CASCADE | cached lock | 468.678 | 4.972 | 76.198 | 77.265 |
| dual | cached lock | 510.106 | 5.997 | 144.601 | 146.096 |
| dual, two FOV processes | service | 449.884 | 5.754 | 89.181 | 90.358 |

Reference CASCADE cases load 40 checkpoints cold and 160 across the warm batch.
Cached cases load 40 cold and none warm, retaining eight ensembles with 5,500,960
parameter bytes and chunks of at most 1024 windows. Each CASCADE-containing selection
also performs mandatory OASIS calcium denoising. Offline reuse performs no inference
or model loads. Single/dual and reference/service/lock arrays and all deterministic
scientific ROI/FOV fingerprints match exactly in both phases; the stochastic CCG
fields remain computed but are excluded from comparisons.

The independent read-only audit verifies **16 complete databases**, all expected
ROI/FOV/trace/method rows, foreign keys and canonical method/owner bindings. Every
stored denoised/spike sample and scalar calcium-noise value matches the saved NPZ
arrays. Full database checksums are recorded. All warm databases contain 400 base
traces and analysis rows; dual cases have 800 spike traces and ROI spike analyses,
eight FOV spike analyses and two shared canonical inference rows.

Memory figures are MiB. Parent peaks include extraction, persistence validation and
offline graphs. The sampled tree sums parent and simultaneous descendant RSS every
second, including the resource tracker; shared pages count in each process and brief
peaks may be missed. These are neither unique physical-memory measurements nor
inference-only allocations.

| Output | Backend | Parent peak | Growth above startup peak | Sampled tree peak |
| --- | --- | ---: | ---: | ---: |
| OASIS | reference | 1419.89 | 946.95 | 1433.33 |
| CASCADE | reference | 1711.88 | 1239.48 | 1723.11 |
| dual | reference | 1767.42 | 1295.47 | 1777.70 |
| CASCADE | service | 1207.91 | 732.89 | 1219.16 |
| dual | service | 1310.44 | 837.05 | 1319.84 |
| CASCADE | cached lock | 1101.91 | 630.92 | 1113.03 |
| dual | cached lock | 1381.41 | 910.20 | 1392.23 |
| dual, two FOV processes | service | 2365.56 | 1892.92 | 3558.41 |

Two processes reduce dual/service offline time from 144.977 to 90.358 s, with matching
deterministic fingerprints, but increase sampled summed RSS from 1319.84 to 3558.41
MiB. Default one-process runs include a resource tracker and create no FOV analysis
pool; the two-process run captures two analysis workers plus that tracker. Model
parameter bytes remain below the 128 MiB cache cap, which is not a process-memory cap.
The harness's retained image/trace/mask lower bound is 96.28 MiB per blocked CASCADE
caller, or 385.13 MiB for four callers before prediction buffers/results. The prior
256 MiB inference-workload comparison cannot certify this complete-workload memory
scope. A documented representative-image/concurrency budget remains necessary.

Storage reveals a new failed gate. Measured warm four-FOV files are 184.49 MiB OASIS,
191.64 MiB CASCADE and 195.80 MiB dual. Projecting measured canonical spike BLOBs to
96 FOVs gives 16.33, 195.59 and 211.92 MiB respectively. Serializing the exact OASIS
samples as hypothetical migrated legacy JSON adds 306.04 MiB: **517.96 MiB total**,
**5.96 MiB over the 512 MiB budget**. This is a spike-payload projection, not a measured
96-FOV file; base calcium traces, analyses, indexes and SQLite space are additional.
It supersedes the earlier claim that the controlled storage margin was sufficient
for every longer input.

`benchmark_trace_array_shuffle.py` evaluates a proposed v2 layout that groups each
sample's bytes by byte position before zlib compression, then reverses the shuffle
and checks every original sample bit and metadata-bound checksum. It selects the
smaller raw/shuffled full payload, preserving float32/double choices and including
header/checksum overhead. On these exact arrays, the candidate plus legacy projection
is **493.93 MiB**, leaving **18.07 MiB** of budget headroom with zero sample changes.
Raising zlib's level alone barely improved the representative traces. The candidate
does not write databases, and the v1 reader at this checkpoint rejects its version;
reader/migration, rollback/integrity and consumer acceptance must land before this
becomes production storage evidence. No rounding or quantization is proposed.

Reproduce the measurements and independent audits in fresh directories, using the
same environment above and Unix `ps` access:

```sh
"$cascade_bench_python" _dev/benchmark_cascade_extraction.py \
  --analysis full --analysis-processes 1 --ccg-shuffles 20 --sample-memory \
  --model-dir "$CALI_CASCADE_MODELS" --output-dir /tmp/cali-large-full \
  --rois 100 --frames 6000 --fovs 4 --workers 4
"$cascade_bench_python" _dev/benchmark_cascade_extraction.py \
  --analysis full --analysis-processes 2 --ccg-shuffles 20 --sample-memory \
  --mode dual --backend service \
  --model-dir "$CALI_CASCADE_MODELS" --output-dir /tmp/cali-large-two-processes \
  --rois 100 --frames 6000 --fovs 4 --workers 4
"$cascade_bench_python" _dev/audit_cascade_extraction.py \
  --input-dir /tmp/cali-large-full --output /tmp/cali-large-audit.json
"$cascade_bench_python" _dev/audit_cascade_extraction.py \
  --input-dir /tmp/cali-large-two-processes --output /tmp/cali-large-worker-audit.json
"$cascade_bench_python" _dev/benchmark_trace_array_shuffle.py \
  --input /tmp/cali-large-full/dual-reference-warm.npz \
  --output /tmp/cali-large-byte-shuffle-candidate.json
```

The controlled 100 × 6000 workload and multi-worker memory measurement are now
recorded. The production storage repair follows below; GPU and scoped memory
acceptance remain pending, while independent uniformly timed real-plate
performance/compressibility is deferred. The reference remains the production default, the cached service remains
experimental, and GUI exposure remains gated.

## Production v2 byte-shuffle codec and schema-13 migration

[`cascade_trace_shuffle_cpu_benchmark.json`](cascade_trace_shuffle_cpu_benchmark.json)
records the production repair of the larger v1 storage failure. The reader accepts
v1/v2 BLOBs and historical JSON. New ORM writes use v2, selecting the smaller full
raw/shuffled zlib payload, including its header. Checksums bind version, compression,
dtype, shape and the original unshuffled bytes. Float32 remains conditional on exact
round trips; migration retains existing BLOB dtype and every sample bit, including
NaN payloads and signed zero. No quantization is introduced.

Schema 13 streams canonical spike rows within one versioned transaction. A corrupt
row, interrupted write or read-back mismatch rolls back the row changes and version
update. The earlier schema-11 step explicitly continues writing v1, so interruption
before schema 13 cannot leave v2 bytes under schema 11/12. Historical physical spike
copies and inference provenance are preserved. The previous schema-12 wheel rejects
a schema-13 copy without changing any database bytes.

The installed production wheel upgrades a copy of the complete-image four-FOV dual
database (800 spike rows) from schema 12 to 13 in **0.776 s**. Every original database
field remains exact; existing BLOB dtype/raw bytes are checked separately. All **11**
exported files match the prior-wheel exports (CSV bytes and JSON object contents).
ORM reads take **0.150 s**, export **33.233 s**, and a separately requested `VACUUM`
**0.877 s**. The file is **195.80 MiB** before migration, **195.81 MiB** immediately
after and **195.30 MiB** after vacuum. SQLite page allocation means logical payload
savings need not translate directly to file-size savings.

All 400 CASCADE float32 rows select byte-shuffle; OASIS retains raw zlib with 396
float64 and four exactly compact float32 rows. The projected 96-FOV canonical payload
is **187.90 MiB** (16.33 OASIS + 171.57 CASCADE). Hypothetical exact legacy OASIS JSON
adds **306.04 MiB**, giving **493.93 MiB**, below the unchanged **512 MiB** budget by
**18.07 MiB**. This closes the controlled storage/numerical gate for this input;
independent biological input compressibility remains unverified. The projection
excludes calcium traces, analyses, indexes and SQLite space; the copied source itself
has no physical legacy spike duplication.

The independent read-only audit verifies the migrated large database against the
saved NPZ arrays. A separate installed-wheel full-runner smoke uses 12 ROIs × 256
frames, one cold and two warm FOVs for all seven output/backend selections. Its
**14** deterministic scientific comparisons and **14** independent database audits
pass with fresh v2 writes and offline reuse. These small checks establish consumer
correctness; the larger full-analysis timing table above still describes v1 and has
not been retimed end to end for v2.

Validation: **2301 passed, 15 skipped in 284.64 s** in the full regression; **543
passed in 39.59 s** against the installed wheel with actual pretrained models. The
standalone pre-upgrade golden BLOB test was added after full-suite collection and
is covered by the installed checks and final **110 passed in 6.55 s** focused run.
Strict mypy passes all three changed source files. Codec corruption, float32/float64
shuffle, raw-layout selection, nonfinite bits, migration ordering and rollback/retry
checks pass. Test-modified checked-in database fixtures were restored.

Reproduce with the current schema-13 wheel and the previously generated schema-12
large database plus its prior-wheel baseline exports:

```sh
"$cascade_bench_python" _dev/benchmark_trace_array_migration.py \
  /tmp/cali-large-full/dual-reference-warm.cali \
  --output-dir /tmp/cali-v2-migration \
  --expected-exports /tmp/cali-v1-baseline/dual-reference-warm_exports
"$cascade_bench_python" _dev/benchmark_cascade_extraction.py \
  --analysis full --analysis-processes 1 --ccg-shuffles 20 \
  --model-dir "$CALI_CASCADE_MODELS" --output-dir /tmp/cali-v2-full-smoke \
  --rois 12 --frames 256 --fovs 2 --workers 4
"$cascade_bench_python" _dev/audit_cascade_extraction.py \
  --input-dir /tmp/cali-v2-full-smoke --output /tmp/cali-v2-full-smoke-audit.json
```

GPU and scoped memory acceptance remain pending; independent real-plate
performance/compressibility is deferred. Reference inference remains the default, cached inference remains
experimental, and GUI exposure remains gated by that release evidence.

## Method-qualified noise QC and schema-14 preservation

[`cascade_noise_qc_validation.json`](cascade_noise_qc_validation.json) records the
planned per-FOV noise summaries. New analysis stores the actual calcium/GetSn noise
used, plus calcium and CASCADE FOV median, linear IQR and known ROI count on separate
scales. Samples include inactive ROIs. Known zero remains zero; missing, negative
or nonfinite values are excluded. A FOV with no active events can still retain QC
without populating activity/correlation/burst metrics.

Completed batch warnings use Q1/Q3 ± 3 IQR with at least four FOV medians, separately
for calcium and CASCADE. CASCADE groups match model, weights manifest and model
sampling rate; unknown identities and zero-IQR groups are skipped. These checks are
descriptive, leave results/thresholds/model selection unchanged and do not certify
biological accuracy. Normal trace exports include `noise_qc.csv` when summaries are
stored, with estimator, units, selected run and CASCADE model/rate metadata.

An installed-wheel full-runner smoke repeats the seven 12-ROI × 256-frame selections
with one cold and two warm FOVs. The independent auditor verifies **14** pre/post
comparisons against the previous schema-13 run: every original sample array and
deterministic scientific field is exact after excluding only seven additive QC
fields. Stored summaries match independently recomputed selected-ROI inputs; **14**
complete database audits pass. A temporary modified noise median is rejected.

The installed schema-14 migration of the large schema-13 copy (800 spike rows)
takes **0.006 s**. All original fields, sample bits/dtypes and 11 exports remain exact;
all new QC fields are NULL, and the file remains **195.30 MiB**. Its existing spike
payload remains **493.93 MiB** projected with hypothetical legacy duplication,
below 512 MiB. Migration adds no historical estimates; explicit re-analysis produces
summaries. The measurement is a controlled single CPU sample, not a new plate gate.

Validation: **2316 passed, 15 skipped in 314.15 s** full regression and **558 passed
in 47.21 s** installed-wheel tests with pretrained models. Final history-source,
automatic-export, CSV-writer and empty-selection cleanup checks are covered by
the installed suite and a 70-test focused export run. Empty selections remove stale
QC CSVs at the requested path. Seven
targeted sources pass strict mypy; commit hooks pass. Reproduce the independent
comparison on completed before/after smoke directories:

```sh
"$cascade_bench_python" _dev/audit_cascade_noise_qc.py \
  --before /tmp/cali-v2-full-smoke --after /tmp/cali-qc-full-smoke \
  --output /tmp/cali-noise-qc-audit.json
```

GPU, scoped memory and remaining release evidence remain pending; independent
real-plate performance/compressibility is deferred. GUI exposure stays gated.

## Production codec storage and migration

Schema 11 introduced the original production lossless codec for `SpikeTrace.values`. Its inline
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

On this earlier input, the codec passes the controlled **512 MiB** spike-payload
budget with **zero** added sample, sum, rate or threshold-crossing error. The larger
complete-image input above exceeds this budget and requires a storage repair. All 400 CASCADE rows use float32. Of
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
