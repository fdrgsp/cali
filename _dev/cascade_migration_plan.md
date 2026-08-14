# Adding CASCADE spike inference to `cali`

**Branch**: `cascade` (branched from `main` at `336e7b5`)
**Status**: plan only — no code changes yet
**Date**: 2026-08-13

---

## 0. TL;DR

| Question | Answer |
| --- | --- |
| CascadeTorch or the TensorFlow Cascade? | **CascadeTorch.** The TF version pins `tensorflow==2.3` / Python 3.7–3.8 and cannot be installed alongside `cali` (`requires-python >=3.11`). CascadeTorch uses the *same weights*, mechanically converted, verified to `max_abs_diff < 1e-5` against Keras. Torch 2.10 is already in the venv via `cellpose`. |
| Replace OASIS or keep both? | **Keep both — decided.** Not a preference, a constraint. Cascade emits *only* a spike-rate trace; it does **not** produce a denoised ΔF/F trace, and `Traces.den_dff` is the input to cali's entire calcium pillar (peak detection, amplitudes, IEI, calcium bursts, ΔF/F correlation, cluster analysis). The target configuration is **OASIS for `den_dff` + the calcium pillar, Cascade for `inferred_spikes` + the spike pillar**. |
| How is the choice exposed? | The MVP adds `spike_method` (`"oasis"` \| `"cascade"`), an explicit `cascade_model`, and `cascade_device`. **There is no `denoise_method="none"` in this migration**: OASIS always produces `den_dff`, so calcium analysis and legacy behaviour remain intact. |
| Biggest code changes? | First, every database path moves behind one versioned-migration engine factory so old `.cali` files remain readable after schema expansion. Second, deconvolution moves from **per-ROI** to **per-FOV batched**; CASCADE must never be called once per ROI. |
| Biggest scientific change? | The *meaning* of `inferred_spikes` changes: OASIS gives sparse, arbitrary-unit event amplitudes; CASCADE gives smooth **expected spikes per frame** under the model's calibration domain. Every downstream threshold in cali was tuned for the former. See §6. |
| Call `cascade.predict()` or reimplement it? | **Never call it per ROI.** Keep one direct, once-per-FOV call as the numerical reference and safe fallback. The intended production path is a small cached predictor that preserves upstream semantics while reusing loaded ensembles and processing windows in chunks. Ship that path only after golden equivalence and performance/RSS gates pass. |

---

## 1. Which Cascade

### 1.1 CascadeTorch vs `HelmchenLabSoftware/Cascade`

Local checkout inspected: `/Users/fdrgsp/Documents/git/CascadeTorch`.

| | CascadeTorch | Cascade (TF) |
| --- | --- | --- |
| Framework | PyTorch (tested 2.4–2.9; 2.10 works locally) | `tensorflow==2.3` + `keras==2.3.1` |
| Python | ≥3.8 | 3.7 / 3.8 recommended |
| Weights | `.pth`, converted from the `.h5` originals | `.h5` |
| Model count | 156 pretrained models in `available_models_CascadeTorch.yaml` (updated 2026-07-19) | same catalogue upstream |
| Apple Silicon | works (CPU / MPS) | Rosetta workarounds |
| Docs / FAQ | *"Check the parent CASCADE repository. FAQs and updates are updated only there."* | canonical |
| PyPI | **not published** — install from git or vendor | not published |

**Numerical equivalence.** `etc/model_conversion.py` transposes Keras kernels into `nn.Conv1d`/`nn.Linear` layout and its `verify_conversion()` asserts `max_diff < 1e-5` on random input. The parent repo's README states CascadeTorch "uses identical pre-trained models converted to PyTorch format, producing identical inference results". CascadeTorch commit `9a59b23` is literally *"Delete old files (Tensorflow models)"* — the Torch fork is the actively-maintained artefact (latest commit adds the 2026 *Nature Methods* GCaMP8 reference).

**Conclusion**: no reason to consider the TF version. It is unusable in this environment and offers nothing extra.

### 1.2 The API you actually need

The reusable upstream primitives are:

```python
utils.calculate_noise_levels(traces, frame_rate)  # → per-neuron noise, in % ΔF/F / √Hz
utils.preprocess_traces(traces, before_frac, window_size)  # → (n, T, W) sliding windows
utils.define_model(filter_sizes, filter_numbers, dense_expansion,
                   windowsize, loss_function, optimizer)   # → nn.Module
```

`cascade.predict(model_name, traces, model_folder, threshold, padding, trace_noise_levels,
verbosity, device)` remains the **correctness oracle**. It already accepts an explicit device,
`threshold=0`, `padding=0`, caller-supplied noise levels, and `verbosity=0`. A single call per FOV is
therefore a valid first integration and fallback. Its real production limitations are repeated
model construction/loading, whole-FOV window materialisation, no mid-call cancellation, and a few
unconditional warnings/prints. P3 defines the evidence required before replacing its
orchestration.

Explicitly out of scope: `utils_discrete_spikes.infer_discrete_spikes()` (end of §5).

---

## 2. What Cascade can and cannot replace

OASIS currently produces **two** outputs per ROI, in one call
(`_extraction_runner.py:666`):

```python
den_dff, spikes, _b, _g, _lam = deconvolve(dff, g=g, sn=sn, penalty=1, optimize_g=optimize_g)
```

| Output | Consumed by | Cascade equivalent |
| --- | --- | --- |
| `den_dff` | calcium peak detection, peak amplitudes, IEI, calcium-burst detection, `CALCIUM_DEN_DFF_CORRELATION`, cluster analysis, CSV export | **none** |
| `inferred_spikes` | spike thresholding, spike frequency, raster, synchrony/CCG, spike bursts, dimensionality reduction, CSV export | **yes** — CASCADE estimates expected spikes per frame, subject to model/data matching |
| `sn` (noise, via `GetSn`) | peak height + prominence thresholds (`_analysis_runner.py:312`) | different estimator, different scale (see §6.4) |

So "swap OASIS for Cascade" is really **"swap the spike half of OASIS for Cascade"**.

### 2.1 The decided configuration

**OASIS keeps producing `den_dff`; CASCADE takes over `inferred_spikes`.** In settings terms:
`spike_method="cascade"`; denoising is not configurable in this migration.

The trace dependencies form a clean scientific cut:

| | reads | gated by | producer changes? |
| --- | --- | --- | --- |
| Calcium pillar | `den_dff` (peaks), `dff` (noise via `GetSn`) | `AnalysisSettings.enable_calcium` — [`_analysis_runner.py:310`](../src/cali/analysis/_analysis_runner.py#L310) | no — bit-for-bit identical to today |
| Spike pillar | `inferred_spikes` | `AnalysisSettings.enable_spikes` — [`_analysis_runner.py:344`](../src/cali/analysis/_analysis_runner.py#L344) | **yes — OASIS → Cascade** |

The implementation is **not yet fully independent**. `_fov_analysis_parallel.py:141` requires
`den_dff` before collecting any ROI, and `_analysis_runner.py:399` assigns the single `ROI.active`
flag from calcium first when both pillars are enabled. P6 therefore makes FOV eligibility/activity
pillar-specific before CASCADE is exposed. The calcium calculation itself remains unchanged because
it continues to read the same OASIS `den_dff`, `dff`, and `sn`.

**What this costs**: `oasis-deconv==0.2.0` stays a permanent dependency, along with its from-source
C build (README:134), the `scipy<1.17` pin it forces (`pyproject.toml:49`), and the cvxpy warning
filters. This buys freedom from OASIS as a *spike inference method*, not from OASIS as a
*dependency*. `_oasis_suite2p.py` is **not** a substitute here — see §3.7; it returns spikes only,
which is precisely the half being retired.

**What this preserves**: the Decay Constant widget stays fully live — τ still drives OASIS's AR(1)
fit for `den_dff` for both spike methods.

### 2.2 If you ever want OASIS gone entirely

Two possible replacement strategies to evaluate in a separate OASIS-removal project:

1. **Reconvolve** the Cascade rate with an exponential kernel of τ → a calcium-like trace. Reproduces the *shape* of a denoised trace, but the scale is arbitrary (spikes-per-frame convolved) unless a per-ROI gain is fitted, so all historical amplitude comparisons break.
2. **Low-pass the ΔF/F** (Savitzky–Golay / Gaussian). Keeps ΔF/F units, but is a much weaker denoiser than OASIS's AR(1) fit, so peak *counts* shift.

Do not assume only amplitudes would change. A different smoother can shift peak locations, merge or
split peaks, and therefore alter counts, frequency, IEI, activity classification, calcium bursts,
and downstream FOV membership even if its output remains in ΔF/F units. Dropping OASIS requires a
separate end-to-end calcium revalidation.

Do not add a `denoise_method="none"` path in this work. Despite the apparent `enable_calcium=False`
gate, `_fov_analysis_parallel.py` currently drops any ROI whose trace lacks `den_dff`, and
`ROI.active` is calcium-first when both pillars are enabled. A spike-only denoising mode is therefore
not independent today. If OASIS removal becomes a separate goal, first make FOV eligibility and
activity state pillar-specific, then validate a replacement calcium trace.

---

## 3. Current OASIS integration map

Every touchpoint, verified by grep on `main`.

### 3.1 Compute (must change)

| Location | What |
| --- | --- |
| [`_extraction_runner.py:11`](../src/cali/extraction/_extraction_runner.py#L11) | `from oasis.functions import GetSn, deconvolve, estimate_parameters` |
| [`_extraction_runner.py:630-692`](../src/cali/extraction/_extraction_runner.py#L630-L692) | the whole deconvolution block: τ→g conversion, `estimate_parameters`, `GetSn`, `deconvolve`, two `try/except` fallbacks to `g=(0.95,)` |
| [`_extraction_runner.py:351-436`](../src/cali/extraction/_extraction_runner.py#L351-L436) | the per-ROI `for label_value in tqdm(...)` loop that calls `_process_roi_trace` — **this is the loop that has to become batched** |
| [`_extraction_runner.py:512-717`](../src/cali/extraction/_extraction_runner.py#L512-L717) | `_process_roi_trace`: does mask→trace→neuropil→dff→deconvolve→`Traces` in one function |
| [`_analysis_runner.py:14,312`](../src/cali/analysis/_analysis_runner.py#L14) | `GetSn(dff, range_ff=[0.25, 0.5], method="median")` for calcium peak thresholds — the only OASIS use in the *analysis* stage |

### 3.2 Settings / schema (must extend)

| Location | What |
| --- | --- |
| [`_model.py:964`](../src/cali/sqlmodel/_model.py#L964) | `ExtractionSettings.decay_constant` |
| [`_model.py:972-1006`](../src/cali/sqlmodel/_model.py#L972-L1006) | `__eq__` / `__hash__` — **easy to miss**; the runner uses settings equality to decide whether to re-run extraction. New fields must be added to both. |
| [`_model.py:1702-1703`](../src/cali/sqlmodel/_model.py#L1702-L1703) | `Traces.den_dff`, `Traces.inferred_spikes` |
| [`_util.py:29-54`](../src/cali/sqlmodel/_util.py#L29-L54) | `migrate_analysis_settings()` — the current one-off migration that must be replaced by the centralized, versioned P2a system |

### 3.3 GUI (must extend)

| Location | What |
| --- | --- |
| [`_extraction_gui.py:76-86`](../src/cali/gui/_extraction_gui.py#L76-L86) | `TraceExtractionData` dataclass |
| [`_extraction_gui.py:465-493`](../src/cali/gui/_extraction_gui.py#L465-L493) | `_TraceExtractionWidget` decay-constant spinbox + tooltip naming OASIS |
| [`_extraction_gui.py:503-527`](../src/cali/gui/_extraction_gui.py#L503-L527) | `value()` / `setValue()` / `reset()` |
| [`_cali_gui.py:516,531,2335`](../src/cali/gui/_cali_gui.py#L516) | settings.json ⇄ `TraceExtractionData` round-trip |
| [`_analysis_gui.py:805,1048`](../src/cali/gui/_analysis_gui.py#L805) | tooltips describing OASIS noise/threshold semantics |
| [`_pygraph_plot_widgets.py:981`](../src/cali/gui/_pygraph_plot_widgets.py#L981) | plot-selector tooltip |

### 3.4 Labels and strings (change carefully)

[`_constants.py:128-131, 157-159`](../src/cali/_constants.py#L128-L131):

```python
DEN_DFF_TRACES: TraceDataType = "OASIS Denoised ΔF/F Traces"
INFERRED_SPIKES_TRACES: TraceDataType = "OASIS Inferred Spikes Traces"
INFERRED_SPIKES_THRESHOLDED_BINARY: TraceDataType = "OASIS Thresholded Inferred Spikes (Binary)"
```

⚠️ **These string *values* are persisted data, not just labels.** They are the keys of the
`export_options` dict written to `settings.json`, the lookup keys in
`_database_to_csv.py:1048-1050, 1169-1177`, and they are asserted literally in
`tests/test_export_group.py`, `test_runner_csv_export.py`, `test_export_only_option.py`,
`test_gui_export.py`. Renaming them silently invalidates every saved `settings.json`.

**Recommendation**: keep the literal values unchanged in this branch. Provenance belongs in
`ExtractionSettings.spike_method` (queryable, per-run) rather than in a display string. If you do
want backend-neutral labels ("Denoised ΔF/F Traces"), do it as a separate commit with an
`_LEGACY_TRACE_LABELS: dict[str, str]` alias map applied when loading `settings.json`.

### 3.5 Packaging

| Location | What |
| --- | --- |
| [`pyproject.toml:34`](../pyproject.toml#L34) | `"oasis-deconv==0.2.0"` — built from source, needs a C toolchain (README:134 documents SDK workarounds) |
| [`pyproject.toml:49`](../pyproject.toml#L49) | `"scipy<1.17"` — **pinned only because of OASIS** |
| [`pyproject.toml:144-153`](../pyproject.toml#L144) | pytest treats warnings as errors, with a small explicit ignore list; the CASCADE job must obey the same policy |
| [`__init__.py:8-9`](../src/cali/__init__.py#L8-L9) | the same cvxpy filter, applied before importing cali |

### 3.6 Docs

`README.md` §"OASIS Denoising and Deconvolution" (L413-443), plus L273-279, L290, L335, L458, L463, L518, L571. `_dev/TODO.md:5` already lists *"Cascade (instead of OASIS)"* — this plan closes that item.

### 3.7 Pre-existing repo state worth knowing

- `src/cali/extraction/_oasis_suite2p.py` **stays in the tree** (owner's decision, 2026-08-13). It was briefly deleted in the working tree; that deletion has been reverted. Nothing in `src/` imports it — only `tests/test_oasis_suite2p.py` does — so it costs 3 tests and ~5 KB and is otherwise inert. Keep it that way.
- It is a numba port of suite2p's OASIS — no C extension, no `cvxpy`, no `scipy<1.17` pin. **Kept deliberately, but not used.** It is tempting as a way off the `oasis-deconv` C build, and it is the wrong tool for the job this branch needs:

  | | `oasis-deconv` (current) | `_oasis_suite2p.py` |
  | --- | --- | --- |
  | Returns `den_dff` | yes | **no** — `oasis_trace` writes only `s`; the denoised trace is never formed |
  | τ estimation | `estimate_parameters()` per ROI when Decay Constant = Auto | **none** — `tau` is a required argument, one global value |
  | Sparsity | `penalty=1`, λ derived from `sn` | **none** — "No sparsity constraints" |
  | `GetSn` | yes | no |

  The denoised trace *is* recoverable from the pool representation — within pool `p`,
  `c[t[p] + k] = v[p] * exp(g * k)` for `k in range(l[p])`, about ten lines inside `oasis_trace`.
  But even patched, the absent L1 penalty produces **different `den_dff` values**, hence different
  peak heights, counts and amplitudes — breaking exactly the calcium comparability §2.1 exists to
  protect. Keep the file as a documented escape hatch in case the C build breaks on a future Python
  or macOS SDK; that is its whole value.

---

## 4. Target architecture

### 4.1 New package

```text
src/cali/extraction/_spike_inference/
├── __init__.py        # construct the selected spike backend
├── _base.py           # result dataclasses + SpikeInferenceBackend protocol
├── _oasis.py          # today's OASIS calculation, moved verbatim
├── _cascade.py        # reference adapter + cached predictor
└── _cascade_models.py # catalogue, download, manifests, frame-rate compatibility
```

### 4.2 The seam

```python
# _base.py
from dataclasses import dataclass
import numpy as np
from typing import Protocol

@dataclass(frozen=True)
class OasisResult:
    """All outputs needed to preserve today's OASIS and calcium behaviour."""
    spikes: np.ndarray            # (n_rois, T) float, non-negative, no NaNs
    den_dff: np.ndarray           # (n_rois, T) float
    sn_by_roi: np.ndarray         # preserve the value used by inline calcium thresholds
    g_by_roi: np.ndarray          # fitted/fallback AR parameter for diagnostics


@dataclass(frozen=True)
class SpikeInferenceResult:
    """Per-FOV spike result. Row i corresponds to input row i."""
    spikes: np.ndarray            # (n_rois, T), finite and non-negative
    units: str                    # "a.u." (OASIS) | "spikes/frame" (CASCADE)
    valid_start: int              # inclusive; 0 for OASIS
    valid_stop: int               # exclusive; T for OASIS
    provenance: "SpikeInferenceProvenance"
    noise_by_roi: np.ndarray | None = None
    selected_noise_level_by_roi: np.ndarray | None = None


class SpikeInferenceBackend(Protocol):
    name: str

    def infer(
        self,
        dff: np.ndarray,          # (n_rois, T) ΔF/F as a *fraction*, not percent
        frame_rate: float,        # Hz
        *,
        cancel: "Callable[[], bool] | None" = None,
    ) -> SpikeInferenceResult: ...
```

`OasisBackend.infer_all(..., cancel=...)` loops over rows, checks cancellation before each ROI, and
returns `OasisResult`; its spike output is wrapped as a `SpikeInferenceResult` when selected.
`CascadeBackend.infer()` returns only
`SpikeInferenceResult`. This split is deliberate: a generic `DeconvolutionResult` with an optional
`den_dff` loses OASIS's `sn`, even though the current inline calcium threshold consumes that exact
value. It also falsely suggests the two algorithms produce interchangeable outputs.

`provenance` is not merely a log dictionary, but it also must not duplicate run-constant strings and
hashes on every ROI. Add a normalized `SpikeInferenceRun` table
(`__tablename__ = "spike_inference_run"`), with a unique `analysis_result_id` foreign key so there is
one row per extraction `CaliResult`. It contains method, units, backend package/revision, resolved
model, catalogue revision, config/ordered-weight manifest SHA-256, resolved device/dtype, model
sampling rate, smoothing, kernel type, and a provenance schema version. Add the corresponding
one-to-one `CaliResult.spike_inference_run` relationship. Do not put
the resolved manifest on `ExtractionSettings`: those settings are deduplicated/reused, so a hash can
become stale if files under the same model name change.

Each `Traces` row stores only a `spike_inference_run_id` foreign key plus genuinely trace/ROI-varying
values: `inferred_spikes_valid_start`, nullable `inferred_spikes_valid_stop`, timing source,
observed frame rate/jitter, per-ROI noise estimate, and selected model noise level. Timing belongs
here because it can differ by FOV within one extraction run. Configure the
`Traces.spike_inference_run` relationship for eager `selectin` loading so detached traces remain
self-describing during
`AnalysisRunner.run(fovs, analysis_settings)`. Audit every trace-loading query to ensure the
relationship is populated before `expunge_all()`; do not solve detached access by copying the full
JSON blob onto every trace.

The legacy migration creates one OASIS/a.u. provenance row per historical extraction result and
backfills its traces. Rows without an `analysis_result_id` retain a documented fallback of
OASIS/a.u./full-trace-valid. A nullable `valid_stop` means `len(inferred_spikes)` for legacy data.

### 4.3 Data flow after the change

```text
ExtractionRunner._extract_trace_data_per_position(fov)
│
├─ Phase A  per ROI (loop, threaded per-FOV as today)
│     mask → raw trace → neuropil correction → calculate_dff()
│     └─ collect _RoiParts(label_value, raw, corrected, neuropil, dff, roi_size, ...)
│
├─ Phase B  per FOV (ONE call, batched)
│     dff_matrix = np.vstack([p.dff for p in parts])        # (n_rois, T)
│     oasis = OasisBackend.infer_all(dff_matrix, frame_rate, decay_constant=τ, cancel=cancel)
│     spikes = oasis.spikes if spike_method == "oasis" else CascadeBackend.infer(...)
│
└─ Phase C  per ROI
      build Traces(dff=..., den_dff=oasis.den_dff[i], inferred_spikes=spikes[i],
                   persisted spike provenance...)
      + optional per-ROI DataAnalysis, passing oasis.sn_by_roi[i]
```

`AnalysisRunner` must become provenance-aware before CASCADE is exposed. It re-reads
`Traces.inferred_spikes` from the DB without extraction settings; it must dispatch OASIS vs CASCADE
threshold/rate semantics from the eagerly loaded `SpikeInferenceRun` relationship plus trace-level
fields and crop every spike analysis to the stored valid interval. Extraction-time analysis and
analysis-only re-runs must share one helper and produce identical results.

---

## 5. Implementation phases

Each phase should leave `pytest` green.

### P0 — Housekeeping (30 min)

1. ~~Restore `_oasis_suite2p.py`~~ — **done.** The working-tree deletion was reverted; the module
   and its 3 tests are back and collection is healthy. Owner's decision (2026-08-13): the file
   **stays in the tree indefinitely, as a dormant escape hatch**. It is not used by this work and
   must not be wired into the extraction path — see §3.7.
2. Baseline, measured on this branch with `_oasis_suite2p.py` restored:

   ```text
   $ python -m pytest -q
   1507 passed, 1 skipped in 258.64s
   ```

   Every later phase must match or beat that.

### P1 — Extract the backend seam, no behaviour change (½ day)

1. Create `_spike_inference/` per §4.1.
2. Move `_extraction_runner.py:630-692` into `_oasis.py` as `OasisBackend.infer_all()`, looping rows.
   Keep both `try/except` fallbacks and the `cali_logger.debug(f"OASIS params ROI ...")` line
   and return `sn_by_roi` and `g_by_roi` in `OasisResult`. Accept `cancel=` and check it before each
   row so batching does not regress today's per-ROI cancellation granularity.
3. Split `_process_roi_trace` (`:512-717`) into:
   - `_compute_roi_dff(...) -> _RoiParts | None` — everything through line 628.
   - `_finalize_roi(parts, den_dff, spikes, sn, provenance, ...) -> tuple[Traces, DataAnalysis|None, bool, bool, float, str]` — lines 698 to end, unchanged logic. Passing `sn` is required for exact inline calcium-threshold behaviour.
4. Rewrite the loop at `:351-436` into Phase A / B / C (§4.3). Keep the `tqdm` description and all
   `_check_for_abort_requested()` calls; add one between Phase A and B, and pass `cancel=` into
   `infer()` so a long batch can bail.
5. Keep `_NUMBA_LOCK` around `calculate_dff` exactly where it is.
6. Phase A now retains `_RoiParts` for a full FOV. Keep only arrays needed for Phase C, release them
   immediately after finalization, and include Phase-A retention multiplied by concurrent FOV
   workers in the RSS benchmark (roughly 20–25 MB for 100 ROIs × 6000 frames × four float64 traces,
   before Python/object overhead).

**Acceptance**: full suite green, and a re-run of an existing dataset produces byte-identical
   `den_dff` / `inferred_spikes`, OASIS `sn`, per-ROI metrics, and active flags. Use a checked-in
   regression fixture rather than a one-off-only script and assert `rtol=0, atol=0`.

### P2 — Centralized migrations, settings, and provenance schema (1–2 days)

#### P2a — one database-opening choke point (blocker before mapped columns)

There are dozens of direct `create_engine()` call sites across model loaders, the runs panel, GUI,
runner, exports, utilities, and plots. Today only `AnalysisSettings.load_from_database()` calls the
existing migration. Once mapped columns/relationships are added to `trace`, `data_analysis`, or the
settings tables, an un-migrated ORM `SELECT` against an old `.cali` file fails before application
fallback code can run.

Create one public `cali.sqlmodel.create_cali_engine(...)` factory and refactor every production
database-path call site to use it. The factory creates the SQLAlchemy engine and calls
`migrate_database(engine)` before returning it. Public functions that accept an already-created
`Engine` must call an idempotent `ensure_schema_current(engine)` at their boundary. Do not use a
`first_connect` listener that opens a second connection recursively; migration must be explicit,
testable, and complete before the first ORM query.

Replace the one-off `migrate_analysis_settings()` scheme with ordered, versioned migrations using
SQLite `PRAGMA user_version` (version 0 means a legacy unversioned database). Each migration runs in
a transaction, validates the expected source schema, advances the version only after success, and
is safe to retry after interruption. New databases created via `SQLModel.metadata.create_all()` are
marked at the current version. Keep a temporary compatibility alias only until all internal callers
move to the new factory.

Land P2a as its own commit before defining the new mapped columns below.

#### P2b — settings and normalized provenance

`ExtractionSettings` gains only the MVP selection inputs:

```python
spike_method: str = "oasis"      # "oasis" | "cascade"
cascade_model: str | None = None # required for cascade; explicit, never nearest-rate auto
cascade_device: str = "auto"     # "auto" | "cpu" | "cuda" | "mps"
```

- Add all three to `ExtractionSettings.__eq__` **and** `__hash__`. Validate that
  `cascade_model` is present for CASCADE and ignored/cleared for OASIS. Resolve the exact model
  before persisting/reusing settings; `None` must never mean "whatever the catalogue chooses today".
- Add `CASCADE_AP_THRESHOLD = "cascade_ap"` as a real persisted
  `AnalysisSettings.spike_threshold_mode` value, plus
  `cascade_ap_threshold_fraction: float = 1 / e`. Add the fraction to
  `AnalysisSettings.__eq__` **and** `__hash__`; equality gates whether re-analysis runs.
- Validate threshold-mode legality against the trace provenance:

  ```text
  OASIS:   global | multiplier
  CASCADE: global | cascade_ap
  ```

  `cascade_ap_threshold_fraction` is used only for `cascade_ap`. If the user selects a global
  CASCADE threshold, require an explicit value and label it in spikes/frame; do not silently reuse
  the OASIS-oriented default `3.0`.
- Add nullable `DataAnalysis.estimated_spike_rate_hz`, `estimated_spike_count`,
  `calcium_active`, and `spike_active`. Keep legacy `inferred_spikes_frequency` OASIS-only.
- Keep `DataAnalysis.inferred_spikes_threshold`, but add
  `inferred_spikes_threshold_mode_applied` and `inferred_spikes_threshold_units`. Without them the
  same numeric/CSV column silently mixes OASIS arbitrary units and CASCADE spikes/frame.
- Add the normalized `SpikeInferenceRun` and trace-level fields specified in §4.2. The migration
  order is: create `spike_inference_run`; add settings/analysis/trace columns; insert one
  OASIS/a.u. provenance row per distinct historical trace `analysis_result_id`; backfill
  `trace.spike_inference_run_id`; verify counts/foreign keys; then advance `user_version`.

**Acceptance**: take an unmodified pre-migration `.cali` fixture and exercise it through the runs
panel queries, experiment/result/settings direct loaders, trace plotting, CSV export, analysis-only
re-run, and save/reopen. Assert that every route migrates before its first ORM query, repeated opens
are no-ops, an interrupted migration resumes safely, provenance relations remain available after
detachment, and all legacy calcium results remain unchanged.

### P3 — CASCADE backend (1–2 days) — reference first, then cache

The repeated model loading is a real reason to own a small production inference loop: a plate with
many FOVs can otherwise reconstruct and `torch.load()` the same five ensemble members for every
noise level used by every FOV. It is **not** a reason to begin without a numerical oracle.

#### P3a — direct upstream reference adapter

Implement a private adapter that calls upstream exactly once per FOV:

```python
cfg = read_config(model_folder / model_name / "config.yaml")
noise_by_roi = utils.calculate_noise_levels(dff_matrix, cfg["sampling_rate"])
reference = cascade.predict(
    model_name,
    dff_matrix,
    model_folder=model_folder,
    threshold=0,
    padding=0,
    trace_noise_levels=noise_by_roi,
    verbosity=0,
    device=device,
)
```

The model-selection noise calculation must use `cfg["sampling_rate"]`, exactly like an upstream
call with `trace_noise_levels=None`. The observed recording rate is used to validate compatibility
and report QC, not to change this estimator. Use the same `noise_by_roi` array in the oracle and
cached paths, and include a test showing that the explicit-noise oracle equals upstream's implicit
noise calculation.

This is the correctness reference and safe fallback, not the final performance architecture. It
also proves the end-to-end seam before introducing custom windowing/cache behaviour. Reject empty,
too-short, non-2D, or non-finite inputs before the call; do not silently convert input NaNs to zero.
Record its output on a small real pretrained CASCADE example as a golden fixture.

#### P3b — reusable cached predictor

Implement `CachedCascadePredictor` as a small orchestration layer over the pinned package's public
model definition and noise estimator. Do not copy/paste upstream `predict()`. Preserve its model
selection, ensemble averaging, `threshold=0`, and padding semantics, but add:

- persistent, eval-mode model ensembles across FOVs;
- bounded ROI/time-window chunks and cancellation checks between chunks;
- explicit `torch.inference_mode()` and device placement;
- cali logging and structured diagnostics;
- deterministic cleanup through `close()` / `clear_cache()` for tests and device changes.

Ship the cached path only if all of these gates pass:

1. CPU output matches the direct upstream call on synthetic traces **and a real bundled example**
   at `rtol=1e-5, atol=1e-6` (tighten if observed equivalence permits).
2. OASIS `den_dff` is bit-identical regardless of the selected spike method.
3. Two successive FOVs cause no additional `torch.load()` for an unchanged model manifest/device.
4. Peak resident memory stays within a documented budget and cancellation completes after the
   current chunk, not after the whole FOV.
5. Representative-plate wall time materially improves over the once-per-FOV reference path. If it
   does not, keep the simpler upstream adapter as production and retain the cached implementation
   only behind an experimental flag or remove it.

```python
@dataclass(frozen=True)
class ModelCacheKey:
    model_dir: Path
    model_manifest_sha256: str   # config + ordered weights; invalidates replaced files
    noise_level: float
    device_type: str
    device_index: int | None
    dtype: str

_MODEL_CACHE: dict[ModelCacheKey, tuple[torch.nn.Module, ...]] = {}
_CASCADE_LOCK = threading.RLock()  # safe MVP policy for the FOV ThreadPool
```

Keying only by `(model_dir, noise_level)` is incorrect: it can return CPU modules for an MPS/CUDA
request and misses in-place weight/config changes. Cache population and inference run under the
same resource policy. Start with one global lock for correctness; benchmark a per-device lock or
single dedicated inference worker later instead of assuming concurrent Torch FOVs help.

Non-negotiable details:

1. **`padding=0`, plus a persisted valid interval.** Match `predict()`'s effective padding rule
   exactly:

   ```python
   valid_start = int(before_frac * windowsize)
   valid_stop = T - int((1 - before_frac) * windowsize)
   ```

   `preprocess_traces()` begins materializing valid windows one index earlier
   (`int(before_frac * windowsize) - 1`), but `predict()` pads through that sample. The optimized
   path must match `predict()`, not expose the preprocessing-only sample. With `windowsize=64` and
   `before_frac=0.5`, that is 32 frames at each end. Store finite zeros for compatibility, but treat
   them as missing in *every* spike metric, binary train, burst, synchrony, CCG, and export—not only
   in the firing-rate denominator. Suppress a synthetic rising edge at `valid_start`.
2. **`threshold=0`.** Clip negative model outputs to zero. Upstream `threshold=1` applies a second,
   dilated AP mask; cali must retain the unthresholded expected-rate trace and derive binary events
   only at analysis time. Require every value inside the valid interval to be finite; fail rather
   than hiding unexpected model NaNs/Infs with `nan_to_num`.
3. **ΔF/F is a fraction.** `calculate_dff` already returns `(F - F0) / F0`; the noise function
   performs its own `×100` conversion.
4. **Do not call `utils.preprocess_traces()` for the optimized path.** It first allocates the full
   `(n_rois, T, W)` array as float64; casting afterward does not prevent that peak. Implement a
   tested float32 chunk/window iterator (or Torch `unfold`) whose windows match upstream exactly.
5. **Device.** `"auto"` resolves once to CUDA, then MPS, then CPU, and the resolved device is
   persisted. Test CPU against MPS/CUDA with device-specific tolerances; do not promise bitwise
   cross-device equality. Avoid changing global `torch.set_num_threads()` inside worker calls.
6. **Diagnostics.** Log/store the per-ROI noise estimates and selected model noise levels. Surface
   out-of-range coverage as a warning/error policy rather than relying on upstream stdout.
7. **Licensing.** A cali-local implementation must be independently written from documented
   behaviour and call the external package; do not transplant GPL source into BSD files. Prefer an
   upstream contribution or a pinned GPL fork if substantial upstream code must be modified, and
   obtain a project licensing review before distribution.

### P4 — Model catalogue and download (½ day)

`_cascade_models.py`.

- **Cache dir**: `Path.home() / ".cali" / "cascade_models"` (override via `CALI_CASCADE_MODELS`).
  Do **not** bundle models in the wheel: the catalogue has 156 entries and the user should be able
  to pick any of them. (Precedent for bundling exists — `detection/cellpose_models/` ships a 26 MB
  file — but one hard-coded Cascade model would be the wrong trade-off here.)
- **Download**: adapt the upstream `cascade.download_model()` behaviour to use `cali_logger`, a real
  timeout, and atomic extract-then-rename. Models are ~5.5 MB (40 `.pth` files + `config.yaml`).
- **CLI**: implement and test the advertised `cali cascade-download <name>` command in this phase,
  including `--model-dir`, offline failure, interrupted download cleanup, checksum/manifest output,
  and a non-zero exit code on failure.
- **Explicit selection, rate-filtered UI**: parse the catalogue and show only exact-rate-compatible
  choices after measuring the recording's timestamps. Do not silently choose the nearest model.
  For a verified 10 Hz recording, examples include:

  ```text
  Global_EXC_10Hz_smoothing50ms       Global_EXC_10Hz_smoothing50ms_causalkernel
  Global_EXC_10Hz_smoothing100ms      Global_EXC_10Hz_smoothing100ms_causalkernel
  Global_EXC_10Hz_smoothing200ms      Global_EXC_10Hz_smoothing200ms_causalkernel
  ```

  `Global_EXC_10Hz_smoothing200ms` is a reasonable **validation candidate**, not an automatic
  scientific default. Model family, smoothing, causal/acausal kernel, cell type, indicator, and
  measured noise must be chosen deliberately and recorded. If there is no compatible model, stop
  with guidance; resampling is a separate, explicitly validated feature.

- **Offline / air-gapped**: if the model dir is missing and download fails, raise a
  `CascadeModelNotFound` with the exact `cali cascade-download <name>` command to run.
- **Immutable provenance**: pin the catalogue to a known revision, hash the downloaded
  `config.yaml` and ordered weight files into a manifest, verify it when loading, and persist the
  catalogue revision plus manifest SHA-256. A model name alone is not reproducible.

#### The catalogue (156 models, 12 families)

| Family | N | Notes |
| --- | --- | --- |
| `Global_EXC_*` | 72 | universal excitatory; 17 rates (1, 2, 2.5, 3, 4.25, 5, 6, 7, 7.5, 10, 12.5, 15, 17.5, 20, 25, 30, 40 Hz) |
| `GC8_EXC_*` / `GC8s` / `GC8f` / `GC8m` | 17/13/8/6 | GCaMP8, combined and per-variant |
| `Online_model_*` | 20 | causal kernel, real-time inference |
| `OGB_zf_pDp_*` | 5 | zebrafish OGB1, pDp |
| `Global_IN_*` | 3 | **inhibitory** interneurons |
| `Spinal_cord_excitatory/inhibitory_*` | 3/3 | 2025 J.Neurosci models |
| `Zebrafish_*`, `GCaMP6f_mouse_30Hz_smoothing200ms` | 1/1 | the only indicator-specific GCaMP6 model |

#### Indicator choice — `Global_EXC` is not "the GCaMP6 model"

It is multi-indicator by construction. The bundled `Global_EXC_30Hz_smoothing25ms` trains on **18
datasets** spanning OGB1, Cal520, GCaMP5k, GCaMP6f/6s **and three red indicators**
(`DS18-R-CaMP-m-CA3`, `DS19-R-CaMP-m-S1`, `DS20-jRCaMP1a-m-V1`). Use `Global_EXC` as the broad
starting candidate; prefer a preparation-specific family when the indicator/cell class is actually
represented, then validate on the target data.

**For jRCaMP1b specifically**: no dedicated model and no jRCaMP1b ground truth (the database has
jRCaMP1**a**). Red GECIs in `Global_EXC` make it the best starting candidate, but not a validated
jRCaMP1b calibration. Validate rate/noise matching and known biological controls before treating
its absolute output as comparable across experiments.

⚠️ **Known gap: there is no 10 Hz `_high_noise` Global_EXC model.** The 11 that exist are at 2.5, 3,
4.25, 7.5, 15, 30 and 40 Hz. Standard models cover noise levels **2–9**; red indicators are dimmer
than GCaMP and may exceed that. Do **not** substitute the 7.5 Hz high-noise model at 10 Hz — that
trips §6.1. Measure first, on real ΔF/F:

  ```python
  model_rate = cfg["sampling_rate"]
  noise = np.median(np.abs(np.diff(dff, axis=1)), axis=1) / np.sqrt(model_rate) * 100
  ```

  If the median exceeds 9, request a 10 Hz high-noise model from the authors — the README offers
  "additional models upon request" and the warning in upstream `cascade.predict()` points users at
  <https://github.com/HelmchenLabSoftware/Cascade/issues/61> for exactly this. Long lead time;
  ask early. This check belongs in the P3 diagnostics and should run before the first real batch.

### P5 — Wire extraction and persist provenance (½ day)

In `_extract_trace_data_per_position`, Phase B always runs OASIS once, then selects only the spike
producer:

```python
oasis = OasisBackend().infer_all(
    dff_matrix,
    frame_rate,
    decay_constant=decay_constant,
    cancel=cancel,
)
spike_result = (
    SpikeInferenceResult.from_oasis(oasis, trace_length=dff_matrix.shape[1])
    if extraction_settings.spike_method == "oasis"
    else cascade_backend.infer(dff_matrix, measured_frame_rate, cancel=cancel)
)
```

Phase C always stores `oasis.den_dff`; it passes `oasis.sn_by_roi[i]` to the shared analysis helper,
creates/reuses one `SpikeInferenceRun` for the extraction result, and stores
`spike_result.spikes[i]` plus the provenance FK, valid interval, per-ROI noise, and selected noise
level, together with the FOV timing source/rate/jitter. Do not use a `CompositeBackend`: denoising
is no longer an interchangeable backend choice in this scope.

Also update:

- Add a worker regression test that reads all three settings after `session.expunge()`. Ordinary
  scalar columns are loaded with the row, so do not claim the existing eager-access tuple is needed
  to prevent `DetachedInstanceError`; extend it only if an actual expiry/deferred-load test proves
  necessary.
- `_visualize_experiment.py:257` — print the spike method, resolved model/manifest, units, device,
  and valid interval alongside the OASIS decay constant.
- Exports — include backend/units/model/manifest/valid interval in metadata without renaming the
  persisted trace-label constants in §3.4.

Keep CASCADE inaccessible from the released GUI until P6 and P7 pass.

### P6 — Make spike analysis provenance-aware (1 day)

This phase precedes GUI exposure and is part of the CASCADE feature, not follow-up cleanup.

1. Move extraction-time and re-analysis ROI calculations into one shared function. Both paths must
   consume the `Traces` provenance fields and produce identical `DataAnalysis` objects.
2. For CASCADE, compute `estimated_spike_count = sum(valid_rate)` and
   `estimated_spike_rate_hz = mean(valid_rate) * measured_frame_rate`. Leave legacy
   `inferred_spikes_frequency` unset (or explicitly documented as legacy-only); do not silently
   change its meaning from OASIS event/frame counts to expected spikes.
3. Validate the persisted `AnalysisSettings.spike_threshold_mode` against the trace's
   `SpikeInferenceRun.method`. For `cascade_ap`, apply the model-derived AP threshold from §6.2 only
   inside the valid interval using `cascade_ap_threshold_fraction`; for `global`, require an explicit
   spikes/frame value. Store the applied mode and threshold units on `DataAnalysis`. Suppress an
   edge at the first valid sample if the signal is already above threshold there.
4. Crop/mask to the common valid interval before raster, burst, synchrony, CCG, and dimensionality
   reduction. Invalid zero padding must never be treated as inactivity or exported as observed rate.
5. Store `calcium_active` and `spike_active` separately. For backward-compatible display,
   `ROI.active = calcium_active or spike_active` when both pillars are enabled; do not retain the
   current calcium-first `if/elif` behaviour.
6. Refactor `_fov_analysis_parallel.py` so it does not globally require `den_dff` or a single
   `ROI.active` flag. Build calcium and spike ROI collections independently, requiring only the
   traces/metrics each pillar consumes. This preserves today's calcium ROI set while allowing the
   spike pillar to select spike-active ROIs.
7. CSV/export code must emit method, threshold mode, and threshold units beside
   `inferred_spikes_threshold`; a bare numeric threshold cannot be compared across OASIS and
   CASCADE runs.

**Acceptance**: extraction-time analysis equals analysis-only re-runs for both methods; selecting
CASCADE cannot change any `den_dff`, calcium metric, or calcium FOV membership.

### P7 — Make the optional dependency installable (½–1 day)

CascadeTorch is not on PyPI, and the currently inspected checkout at
`c6978d5ff33edad8792c76e040412f0636913092` is **not installable as written**: `setup.py` uses
`find_packages()`, but `cascade2p/` has no `__init__.py`, so `find_packages()` returns no packages.
Do not add a direct reference until an upstream commit or maintained fork fixes packaging.

Once fixed, pin a full immutable SHA and enable Hatch direct references:

```toml
[tool.hatch.metadata]
allow-direct-references = true

[project.optional-dependencies]
cascade = [
  "CascadeTorch @ git+https://github.com/PTRRupprecht/CascadeTorch@<full-commit-sha>",
]
```

Notes:

- Do not use `@master`. Record the package commit alongside model/catalogue provenance.
- The upstream distribution already declares `torch` and `ruamel.yaml`; reading configs with
  PyYAML inside cali does **not** remove `ruamel.yaml` from installation while that metadata remains.
  If dependency cleanup matters, fix it in the pinned fork/upstream package.
- **Import lazily.** `cali` must remain importable without Torch or `cascade2p`; raise a clear
  `ImportError("Install cali[cascade] to use CASCADE spike inference")` only on backend use.
- Add a clean-environment CI smoke test that installs the built wheel with `[cascade]`, imports
  `cascade2p`, loads a tiny model fixture, and runs one prediction. A source checkout on
  `PYTHONPATH` is not an installation test.
- Run that job under cali's existing `filterwarnings = ["error", ...]` policy and budget time to
  audit warnings from Torch/CASCADE import, `torch.load()`, model construction, and inference.
  Fix warnings where possible; otherwise add only narrow, documented filters tied to a specific
  upstream warning/category/module. Do not add a blanket Torch/CASCADE warning suppression.
- **Licensing**: CascadeTorch and `oasis-deconv` are GPL-3.0 while cali is BSD-3-Clause. Do not
  vendor or copy CASCADE source into `src/cali`; document the optional dependency and obtain the
  project's licensing review for distribution. Do not claim that an existing GPL dependency
  automatically settles every new packaging/derivative-work question.
- OASIS remains required for `den_dff`; therefore its C-build instructions, SciPy pin, and warning
  filters remain in this branch.

### P8 — GUI, docs, and release gate (½ day)

`_extraction_gui.py`:

- `TraceExtractionData` gains `spike_method`, `cascade_model`, and `cascade_device`.
- Add a "Spike Inference" combo (OASIS / CASCADE), an explicit compatible-model combo, model
  download/status controls, and device selection. There is no denoising combo; Decay Constant
  remains enabled because OASIS always produces `den_dff`.
- Populate model choices from the pinned catalogue/local cache, filtered by measured frame rate;
  show family/smoothing/noise/kernel metadata rather than silently choosing a default.
- `value()` / `setValue()` / `reset()` and `_cali_gui.py` settings JSON must round-trip all three
  fields. Read old JSON via `.get(key, default)`.
- Make `_SpikeThresholdWidget` trace-method-aware: retain GLOBAL/MULTIPLIER controls for OASIS and
  expose the persisted GLOBAL/CASCADE_AP modes for CASCADE. Round-trip the AP fraction through
  `AnalysisSettings`; label a CASCADE global threshold in spikes/frame and require an explicit value
  instead of carrying over the OASIS default `3.0`. Correct the existing tooltip, which describes a
  percentile rule while the code uses MAD.
- Retitle "ΔF/F0 and Deconvolution" to "ΔF/F0 and Spike Inference" and make plot/export tooltips
  backend-neutral. Keep the literal persisted keys in §3.4 unchanged.
- Update README installation, offline download, units, valid-edge, model-selection, provenance,
  and cross-backend-comparison guidance; update `_dev/TODO.md` only after the release gates pass.

The GUI option is enabled only when P1–P7, the dedicated CASCADE CI job, a clean install, and a
representative-dataset benchmark are all green.

### Out of scope — discrete-spike post-processing

Do not include `utils_discrete_spikes.infer_discrete_spikes()` in this migration. Upstream cautions
that discrete predictions are generally not recommended except for exceptional data quality and
can imply false precision. If evaluated later, retain the calibrated expected-rate trace in
`Traces.inferred_spikes` and store discrete estimates in a **separate column/table with separate
provenance**; never overwrite the rate trace or label point estimates as ground-truth spike times.

---

## 6. Semantics: what actually changes in the numbers

**This section matters more than the plumbing.** cali's spike analysis was tuned against OASIS
output; Cascade output has different statistics, and several thresholds silently change meaning.

> **Scope decision (owner, 2026-08-13):** the existing inferred-spike logic is **not** to be
> preserved. It may be redesigned around Cascade. The binding constraint is the *calcium* side:
> old `.cali` files must open, and re-running calcium analysis must reproduce existing numbers.
> That is guaranteed by §2.1 (OASIS keeps producing `den_dff`, untouched) plus the P2 migration.
>
> Practical consequence: build §6.2(a) and §6.2(b) as the new default spike pipeline. Do **not**
> spend effort making `MULTIPLIER` behave sensibly on a Cascade trace — retire it for that backend.

### 6.1 Frame rate must match the model

`cascade.predict()` computes noise levels with **the model's** `cfg["sampling_rate"]` in its
noise-estimation block, not the recording's. If a 30 Hz model is applied to a 10 Hz recording, both the
noise matching and the temporal scaling are wrong, and the failure is silent — the output looks
plausible.

Do not trust `ExtractionSettings.frame_rate` alone. Change `_get_elapsed_time_ms_list()` to return a
timing descriptor with its source (`RUNNER_TIME_KEY` timestamps versus exposure-derived synthetic
times). For real timestamps, require finite, strictly increasing, approximately uniform intervals
and derive observed rate as `1000 / median(diff_ms)`—not the current `len(dff) / (last-first)`, which
is off by one sample interval. For synthetic times, validate the exposure-derived rate against the
setting but record that jitter could not be measured. Do not alter the legacy calcium-duration
calculation in this migration if doing so changes historical calcium metrics.

Verify that the measured/derived rate agrees with both extraction settings and model sampling rate
within a documented tight tolerance (start at 1%, informed by timestamp precision). Reject zero
exposure, irregular sampling, and incompatible models with a clear error. Resampling, if ever added,
must be explicit and tested for its effect on noise and calibration. Persist timing source, observed
rate, model rate, interval jitter (when available), and the validation result.

### 6.2 The trace is no longer sparse

| | OASIS | Cascade |
| --- | --- | --- |
| Support | sparse — most frames are exactly `0.0` | dense — smooth non-negative rate, few exact zeros |
| Units | arbitrary (ΔF/F event magnitude) | **expected spikes per frame** under the model's calibration domain |
| Edges | full length | first/last 32 frames unpredictable → 0-padded |
| Value of `sum(trace)` | meaningless | ≈ total number of spikes |

Consequences:

- **`compute_inferred_spike_threshold`** ([`_trace_analysis.py:22-71`](../src/cali/analysis/_trace_analysis.py#L22-L71)) in `MULTIPLIER` mode takes `spikes[spikes > 0]`, splits at the median, and computes a MAD. With OASIS that samples the *noise floor of a sparse* signal. With Cascade `spikes > 0` is nearly the whole trace, so the "noise" estimate becomes a statistic of the *signal* — the threshold will scale with activity, and a highly active ROI gets a *higher* bar than a quiet one. **This mode is not usable as-is for Cascade.**

- **`count_thresholded_spike_events`** ([`_trace_analysis.py:229-259`](../src/cali/analysis/_trace_analysis.py#L229-L259)) returns `sum(spike_train > 0)` — the number of frames above threshold. On a sparse OASIS trace that approximates an event count. On a smooth Cascade rate it measures **time spent active**, so `inferred_spikes_frequency` becomes a duty cycle, not a rate. The existing `num_rising_edges` is the meaningful count for Cascade.

**The redesign.** Three replacements, in priority order. `MULTIPLIER` is retired for the Cascade
backend (it stays for OASIS, where it is correct).

#### (a) Threshold-free rate metrics — build first

CASCADE's output is an expected spike count per frame, so the headline summaries need no binary
threshold:

```python
valid = spikes[valid_start:valid_stop]
estimated_spike_rate_hz = float(np.mean(valid)) * observed_frame_rate
estimated_spike_count   = float(np.sum(valid))
```

Add `DataAnalysis.estimated_spike_rate_hz` and `DataAnalysis.estimated_spike_count` (both nullable
with migration entries). These are substantially more comparable than OASIS amplitudes, but they
are not automatically interchangeable across indicators, preparations, or experiments. Transfer
calibration can vary across datasets and individual neurons; report the chosen model and validation
domain beside the values and avoid an unqualified "absolute ground truth" claim.

The valid interval is binding for all consumers, not just this denominator. Cropping only the mean
while allowing padded zeros into event boundaries, bursts, CCG, or synchrony would still produce
biased results.

#### (b) A principled binary threshold — build second

The population analyses (§6.3) still need a binary train. Replace the data-derived guess with the
model-derived amplitude of one action potential, which CASCADE computes in
`cascade.predict()`'s `threshold == 1` block:

```python
# _constants.py
CASCADE_AP_THRESHOLD = "cascade_ap"   # persisted AnalysisSettings.spike_threshold_mode

single = np.zeros(1001); single[500] = 1.0
ap_peak = gaussian_filter(single, sigma=smoothing * sampling_rate).max()
threshold = cascade_ap_threshold_fraction * ap_peak
```

Use `cascade_ap_threshold_fraction = 1/e` as the upstream-derived default, separate from OASIS's
existing default value `3.0`. The threshold is model-specific and ROI-independent. Read smoothing
and sampling rate from persisted provenance, not from a model file that might be unavailable or
changed during later re-analysis. `cascade_ap` is a real user-selectable, persisted mode; provenance
determines whether that mode is legal, not which mode was selected.

Also switch the Cascade path's event count from `num_thresholded` to **`num_rising_edges`**, which
`count_thresholded_spike_events` already computes — frames-above-threshold is a duty cycle on a
smoothed trace, not an event count.

#### (c) Discrete spikes — not in this migration

See "Out of scope" in §5. They are model-derived point estimates, not observed "real spikes", and
must not replace the calibrated expected-rate trace.

**GUI implication (P8)**: the threshold-mode radio buttons in `_SpikeThresholdWidget`
([`_analysis_gui.py:1041-1094`](../src/cali/gui/_analysis_gui.py#L1041)) become backend-dependent —
`GLOBAL`/`MULTIPLIER` for OASIS, `GLOBAL`/`CASCADE_AP` for CASCADE. Persist the selected mode and
enforce the same legal-mode matrix in non-GUI analysis code. Its tooltip is also **already
wrong today** (it describes a 10th-percentile rule; the code implements a MAD rule). Fix while there.

### 6.3 Downstream analyses that consume the binary train

`threshold_spike_train` → `compute_rising_edges` feeds raster plots, synchrony, CCG
(`_fov_analysis.py:147-166`), spike bursts, and dimensionality reduction. All of them keep working
mechanically, but their inputs shift:

- Cascade's Gaussian smoothing (25–200 ms) **broadens** each event, which inflates zero-lag
  synchrony and narrows CCG peaks' apparent lag structure. Compare backends on the same dataset
  before trusting a cross-backend comparison.
- `spikes_sync_jitter_window` (default 200 ms) and `spikes_sync_cross_corr_lag` (default 500 ms)
  are of the same order as the model smoothing — likely need re-tuning per model.
- A future discrete-spike estimate would have different error modes; it is not assumed to solve
  these concerns without independent validation.

### 6.4 Noise estimation for calcium peaks

`GetSn` ([`_analysis_runner.py:312`](../src/cali/analysis/_analysis_runner.py#L312)) is a PSD-based
estimator; CASCADE's `calculate_noise_levels` is
`median(|diff|)/√model_sampling_rate × 100`. **Different scale, different units—never substitute
one for the other.** OASIS always supplies `den_dff` in this
migration, so `GetSn` stays and calcium results are unchanged. If OASIS is ever fully removed, note that
`compute_calcium_peak_detection_thresholds` already has a MAD fallback for `noise=None`
([`_trace_analysis.py:97-99`](../src/cali/analysis/_trace_analysis.py#L97-L99)) — switching to it
will shift every peak-height threshold, so it needs re-validation against known data, not a
drop-in swap.

**Both backends estimate noise per ROI — keep it that way.** Cascade's batching is a *computational*
detail, not a pooling one: `calculate_noise_levels` on an `(n_rois, T)` matrix returns shape
`(n_rois,)`, and that per-ROI value is what selects each neuron's noise-level network
(`best_model_for_each_neuron` inside `cascade.predict()`). Cascade ships 8 models per family precisely so a
dim and a bright ROI in the same FOV get different weights; collapsing to one FOV-wide number would
defeat that.

Per-ROI is also correct for the calcium side, for a reason specific to the estimators:

- `GetSn` averages the PSD only over `0.25 < f < 0.5` × Nyquist — the top half of the spectrum. Calcium transients are seconds-scale, so their power is all at low frequency; the estimator looks where the signal is not.
- Cascade's `median(|diff|)` is dominated by frame-to-frame shot noise, and the median is robust to transients.

So the §6.2 pathology (*active ROI → inflated "noise" → higher threshold → busiest cells penalised*)
**does not apply to either noise estimator**. Pooling across a FOV would discard genuine per-ROI
differences (brightness, area, focus, neuropil contamination) to solve a problem that does not
exist — and would move `den_dff` and every peak threshold, breaking §2.1.

**Do add a per-FOV noise QC metric** (additive, changes no existing number). Store
`median`/`IQR` of the per-ROI noise on `FOVAnalysis`, and warn when a FOV's median sits far from the
plate's. This is a real gap for 96-well work — an out-of-focus or bleached FOV is currently
invisible until the plots look wrong. Upstream does the equivalent via
`plot_noise_level_distribution` and its "models cannot match the noise levels" warning; fold both
into the P3 diagnostics rather than inventing a separate mechanism.

### 6.5 Performance

Rough expectation for one 96-well plate, ~100 ROIs/FOV, 6000 frames, CPU:

- OASIS: ~10–50 ms per ROI, embarrassingly parallel across the FOV ThreadPool.
- CASCADE reference: one batched call per FOV, but it reconstructs/reloads ensembles and allocates
  whole-FOV float64 windows on every call.
- CASCADE cached: the same numerical computation with weights reused and float32 chunks; initially
  serialized by `_CASCADE_LOCK` for predictable memory use.

Never call either path per ROI. Benchmark reference versus cached inference on representative ROI
counts/trace lengths and one real plate. Record wall time, model-load count, peak RSS, device, Torch
thread settings, chunk size, and numerical difference in the PR; use those measurements to decide
which path ships as the default.

---

## 7. Testing plan

| Test | File | Purpose |
| --- | --- | --- |
| Backend/result conformance | `tests/test_spike_inference_backends.py` (new) | batched shapes, finite/non-negative values, structured OASIS `sn/g`, valid interval, normalized run/ROI provenance |
| OASIS regression | same | refactored OASIS `den_dff`, spikes, `sn`, metrics, and active flags equal the pre-refactor path at `rtol=0, atol=0` |
| Method isolation | same | selecting CASCADE changes only `inferred_spikes` + spike provenance/metrics; `dff`, `den_dff`, calcium metrics, and calcium FOV membership remain exactly equal |
| Upstream golden equivalence | `tests/test_cascade_reference.py` | cached CPU predictor matches direct `cascade.predict()` on synthetic inputs and a pinned real example/model; explicit model-rate noise equals upstream implicit noise |
| Input validation | same | empty, 1D, too-short, NaN/Inf, timestamp mismatch, and irregular sampling fail clearly before inference |
| Valid edges | analysis + FOV tests | effective interval uses `predict()`'s padding formula (not preprocessing's one-earlier window); stored edges are finite zero, but all consumers use only the valid intersection and create no edge at `valid_start` |
| Model cache | same | repeated FOVs load each required ensemble once; model manifest, device/index, or dtype changes miss the cache; `clear_cache()` releases entries |
| Chunk/cancel | same | several chunk sizes equal the reference; CASCADE cancels between chunks and batched OASIS cancels between ROIs, with no partial `Traces` rows committed |
| Device coverage | dedicated CI where available | CPU deterministic; MPS/CUDA compared to CPU using measured, documented tolerances; cache never crosses devices |
| Thread safety / memory | extend `tests/test_analysis_threading.py` | concurrent FOV extraction is deterministic; Phase-A `_RoiParts`, window chunks, and model cache remain within the documented RSS budget |
| Extraction/re-analysis parity | new | extraction-time and analysis-only runs produce identical OASIS and CASCADE `DataAnalysis` results |
| Pillar-specific activity | FOV tests | calcium-active and spike-active sets are independent; spike FOV analysis does not inherit calcium-first `ROI.active` selection |
| Settings/provenance round-trip | settings + SQLModel tests | extraction fields, real `cascade_ap` mode, AP fraction, threshold mode/units, normalized run provenance, and trace-level values survive GUI/JSON/DB/detachment; both settings classes' equality/hash change appropriately |
| Threshold legality | analysis tests | OASIS accepts global/multiplier; CASCADE accepts explicit global/cascade_ap; invalid method/mode pairs fail in GUI and headless paths |
| Legacy migration | new | every production DB-open route migrates an old `.cali` before ORM access; `user_version` advances atomically/idempotently, interrupted migration resumes, provenance is backfilled, and calcium re-analysis is unchanged |
| Download/offline | model tests + CLI tests | atomic install, checksum, interrupted cleanup, cache override, actionable offline error, and advertised CLI command |
| Clean install | packaging CI | build/install `cali[cascade]` in an empty environment, import `cascade2p`, load a model, and infer once under warnings-as-errors |

Default CI must remain network-free and green without the optional dependency, using fakes to test
dispatch/provenance/analysis. Do **not** skip all real CASCADE coverage: add a separate `cascade` CI
job pinned to the exact package/catalogue/model manifests, with dependency/model caching. Treat the
real-model golden comparison and clean-install smoke test as release gates.

A synthetic trace with known spikes is useful as a scientific validation plot, but a loose
"within 30%" assertion is not a reliable unit test across model families. Numerical equivalence to
the upstream implementation is the software gate; recovery accuracy on representative labeled or
biologically controlled data is the scientific gate.

---

## 8. Risks and decisions

| Risk | Severity | Mitigation |
| --- | --- | --- |
| CASCADE's smooth rate breaks `MULTIPLIER` thresholding | **high** — silently wrong frequencies | §6.2: separate AP-fraction mode + threshold-free expected-rate metrics |
| Old databases reach ORM before migration | **high** — ordinary UI/export/plot paths fail | central engine factory, versioned P2a migration, legacy end-to-end matrix |
| Frame-rate/model mismatch or irregular timestamps | **high** — plausible but invalid output | §6.1: measured-timestamp validation and no nearest-model auto-selection |
| Provenance absent during detached re-analysis | **high** — wrong units/thresholds | normalized `SpikeInferenceRun` plus eagerly loaded relation and trace-level valid/noise fields |
| Threshold mode incompatible with spike method | **high** — plausible but meaningless binary train | persist `cascade_ap` as a real mode and enforce the legal method/mode matrix in GUI and headless paths |
| Cached implementation drifts from upstream numerics | **high** | direct once-per-FOV oracle, real-model golden test, safe fallback |
| Optional package currently installs no `cascade2p` package | **high** — feature unusable | upstream/fork packaging fix, immutable SHA, Hatch flag, clean-install CI |
| Memory blow-up from retained ROI parts/full float64 windows | medium | release Phase-A parts promptly; float32 chunk iterator; concurrent-FOV RSS gate |
| Zero-padded receptive-field edges bias analyses | medium | persist valid interval and crop/mask every spike consumer and export |
| Torch/thread interaction with the FOV ThreadPool | medium | global inference lock first; benchmark per-device worker later; avoid per-call global thread changes |
| MPS/CUDA numerical drift vs CPU | medium | device-specific golden tolerances and cache keys; persist resolved device/dtype |
| Model files/catalogue change under the same name | medium | config + ordered-weight manifest hash and catalogue revision |
| Torch/CASCADE warnings fail CI unexpectedly | medium | warnings-as-errors import/inference job; fix or narrowly document/filter each warning |
| GPL-3 CascadeTorch vs BSD-3 cali | medium | no source transplant/vendor; optional dependency documentation and licensing review |
| Renaming `*_TRACES` constants breaks saved settings + CSV keys | medium | §3.4: don't rename in this branch |

**Decisions / required inputs before implementation:**

1. **Decided for this migration:** OASIS remains the sole producer of `den_dff`; only the selected
   `inferred_spikes` producer changes.
2. Measure the real acquisition timestamps for the target datasets before choosing a model. The
   10 Hz examples in this document are illustrative, not an assumption the code may make.
3. The MVP stores one selected spike trace with complete provenance. If simultaneous OASIS and
   CASCADE spike traces are required for side-by-side comparison, add a second trace product/table
   rather than making one column ambiguous; decide that before freezing the P2 schema.

---

## 9. Suggested commit sequence

```text
0. docs: this plan                                                (P0 — done)
1. refactor: extract spike-inference backend seam, batch per FOV  (P1)  ← behaviour-identical
2. refactor(db): central engine factory + versioned migrations     (P2a blocker)
3. feat(db): normalized provenance, settings, metrics, backfill    (P2b)
4. build: fix/pin/install cali[cascade] + warning-clean CI         (P7 prerequisite)
5. feat: pinned model catalogue, manifest, download CLI            (P4)
6. test: once-per-FOV upstream adapter + model-rate golden oracle  (P3a)
7. perf: cached/chunked predictor, equivalence + RSS benchmark      (P3b)
8. feat: wire spike selection; preserve OASIS den_dff/sn/cancel    (P5)
9. feat(analysis): modes, provenance, edges, rates, pillar activity (P6)
10. feat(gui): expose backend/model only after release gates       (P8)
11. docs: README spike-inference section, TODO.md                  (§3.6)
```

This landing order is binding even though §5 groups some work by subsystem. In particular, packaging
and the numerical oracle land before the optimized predictor, and analysis semantics land before
the GUI option. Step 1 remains independently useful if CASCADE is later abandoned.

---

## 10. Primary references used for implementation decisions

- [Official CASCADE repository and FAQ](https://github.com/HelmchenLabSoftware/Cascade) — output
  interpretation, model matching, edge padding, threshold modes, and the caution about discrete
  spike inference.
- [CASCADE method paper, *Nature Neuroscience* (2021)](https://www.nature.com/articles/s41593-021-00895-5)
  — calibrated inference and cross-dataset evaluation.
- [GCaMP8 transfer/calibration study, *Nature Methods* (2026)](https://www.nature.com/articles/s41592-026-03183-x)
  — motivation for qualifying cross-indicator/cross-neuron absolute-rate claims.
- [Hatch metadata documentation](https://hatch.pypa.io/latest/config/metadata/) — direct-reference
  opt-in required for the git dependency.

Pin exact package, catalogue, and model revisions in the implementation/CI; these links explain the
design but are not substitutes for immutable build inputs.
