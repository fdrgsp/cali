# Adding CASCADE spike inference to `cali`

**Branch**: `cascade` (branched from `main` at `336e7b5`)
**Status**: plan only — no code changes yet
**Date**: 2026-08-14

---

## 0. TL;DR

| Question | Answer |
| --- | --- |
| CascadeTorch or the TensorFlow Cascade? | **CascadeTorch.** The TF version pins `tensorflow==2.3` / Python 3.7–3.8 and cannot be installed alongside `cali` (`requires-python >=3.11`). CascadeTorch uses the *same weights*, mechanically converted, verified to `max_abs_diff < 1e-5` against Keras. Torch 2.10 is already in the venv via `cellpose`. |
| Replace OASIS or keep both? | **Keep both, independently selectable — decided.** OASIS always produces `den_dff`; the user can retain/analyze OASIS spikes, CASCADE spikes, or both. CASCADE emits only expected spikes/frame and cannot replace the calcium trace. |
| How is the choice exposed? | Add extraction-owned **Spike Outputs** checkboxes: **CASCADE** (checked by default for a new GUI setup) and **OASIS** (optional comparison output); require at least one. Legacy databases/settings and bare headless `ExtractionSettings()` select OASIS only. CASCADE never silently falls back: if selected it requires the optional package, a verified frame rate, and an explicit compatible model. |
| Can startup bleaching be excluded? | **Yes.** Add a zero-default **Discard at Start** extraction setting with Frames/Seconds radio buttons. Resolve it independently for every source position/file, crop before all trace calculations, and persist the source-to-retained frame/time transform so analysis, stimulation metadata, plots, and exports stay aligned. |
| What do plots show? | Plot/export choices follow stored provenance. A results-level backend selector chooses OASIS or CASCADE when both exist; shared spike plots use that backend's units and method-specific metrics are filtered. OASIS denoised-ΔF/F/calcium plots remain available in every mode because OASIS still produces `den_dff`. |
| Biggest code changes? | First, every database path moves behind one versioned-migration engine factory. Singular spike arrays/settings/ROI/FOV metrics become method-bound child tables so both outputs cannot overwrite each other. Deconvolution also moves from **per-ROI** to **per-FOV batched**; CASCADE must never be called once per ROI. |
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

**OASIS always produces `den_dff`; the user chooses one or both spike outputs to retain and
analyze.** In settings terms, `spike_methods` is a canonical non-empty tuple containing `"oasis"`,
`"cascade"`, or both; denoising is not configurable in this migration. The existing OASIS spike
path remains supported and regression-tested, not merely a migration fallback. "OASIS unchecked"
means its already-computed spike array is not retained or analyzed—it does not skip the OASIS call
needed for `den_dff`.

The trace dependencies form a clean scientific cut:

| | reads | gated by | producer changes? |
| --- | --- | --- | --- |
| Calcium pillar | `den_dff` (peaks), `dff` (noise via `GetSn`) | `AnalysisSettings.enable_calcium` — [`_analysis_runner.py:310`](../src/cali/analysis/_analysis_runner.py#L310) | no — bit-for-bit identical to today |
| Spike pillar | one or more method-bound `SpikeTrace.values` arrays | `AnalysisSettings.enable_spikes` — [`_analysis_runner.py:344`](../src/cali/analysis/_analysis_runner.py#L344) | **selectable — OASIS, CASCADE, or both** |

The implementation is **not yet fully independent**. `_fov_analysis_parallel.py:141` requires
`den_dff` before collecting any ROI, and `_analysis_runner.py:399` assigns the single `ROI.active`
flag from calcium first when both pillars are enabled. P6 therefore makes FOV eligibility/activity
pillar-specific before CASCADE is exposed. The calcium calculation itself remains unchanged because
it continues to read the same OASIS `den_dff`, `dff`, and `sn`.

**What this costs**: `oasis-deconv==0.2.0` stays a permanent dependency, along with its from-source
C build (README:134), the `scipy<1.17` pin it forces (`pyproject.toml:49`), and the cvxpy warning
filters. Any selection containing CASCADE performs both computations—OASIS for `den_dff` and
CASCADE for spikes—so its user-visible wall time is the combined pipeline, not CASCADE inference
alone. Selecting both spike outputs does **not** add a second OASIS inference call: it retains the
spikes already returned with `den_dff`. It does add persistence, per-ROI analysis, and per-FOV
analysis work. Quantify all three selection modes in §6.5. `_oasis_suite2p.py` is not a substitute
for `den_dff`; see §3.7.

**What this preserves**: the Decay Constant widget stays fully live — τ still drives OASIS's AR(1)
fit for `den_dff` for every spike-output selection and also its retained spike output when selected.

### 2.2 Discarding startup bleaching

Add a zero-default extraction setting that excludes the beginning of **each source position/file**:

```python
discard_initial_value: float = 0.0
discard_initial_unit: Literal["frames", "seconds"] = "frames"
```

This is scientifically useful for rapid indicator bleaching, illumination settling, focus settling,
or acquisition startup artefacts. It is an extraction transform, not an analysis display filter:
resolve the cutoff immediately after `dataset.isel(p=..., metadata=True)` and crop the image stack,
per-frame metadata, and timing vector **before** raw/neuropil traces, ΔF/F baselines, OASIS,
CASCADE, extraction-time analysis, or export. Detection and stored ROI masks remain based on the
original source data; only extraction products and downstream analysis are cropped.

The unit semantics are binding:

- **Frames**: require a non-negative integer and discard exactly that many leading samples.
- **Seconds with trusted per-frame timestamps**: compute relative source times and keep the first sample whose
  timestamp is `>= requested_seconds`; equivalently use `searchsorted(relative_ms, cutoff_ms,
  side="left")`. This handles irregular but otherwise valid timestamps without assuming a rate.
- **Seconds without per-frame timestamps**: require either trusted explicit frame-period metadata or
  a user-verified acquisition frame rate, then resolve with `ceil(seconds * frame_rate)` so at least
  the requested duration is excluded. Exposure duration alone is not necessarily frame period and
  must not silently authorize the conversion.

The resolved crop can differ by FOV in seconds mode. Persist one `ExtractionFrameWindow` per
extraction result/FOV with requested value/unit, timing source, original/retained frame counts,
zero-based `source_start_frame`, original `source_start_time_ms`, actual excluded duration, and the
conversion rule. Store retained `Traces.x_axis` rebased to zero, while the window preserves exact
mapping back to the source. All derived event indices are retained-trace-relative; exports include
the source offset (and source-frame columns where event indices are exported). CASCADE
`SpikeTrace.valid_start/valid_stop` are also retained-relative; source-valid bounds are obtained by
composing them with the frame window, not by storing a second independently maintained interval.

Do not mutate global `AnalysisSettings.led_pulse_on_frames`: those values remain in source-file
coordinates. Route every evoked consumer through one frame/time-transform helper. Fully discarded
pulses are omitted, retained pulses are shifted, and a pulse interval crossing the cutoff is clipped
to retained time and explicitly marked. Normalize the current user-facing one-based pulse-frame
convention to zero-based indices inside that helper rather than scattering additional `-1` logic.

Validate every FOV before Phase A: the cutoff must leave enough samples for DFF, OASIS parameter/noise
estimation, and every selected CASCADE model's window/padding requirements. Reject a cutoff that
removes all or too many samples with the file/FOV name, original count, resolved cutoff, retained
count, and limiting requirement. `0 frames` is a strict no-op and must remain bit-identical to the
pre-feature pipeline.

Current readers expose one independent temporal sequence per dataset position, so the cutoff is
resolved per position—not once for a plate. If a future reader concatenates multiple acquisition
files into one position, it must expose segment boundaries and apply the cutoff to each segment
before concatenation; silently applying one cutoff to the combined sequence would violate the
per-file contract.

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
| [`_extraction_runner.py:314-351, 851-875`](../src/cali/extraction/_extraction_runner.py#L314-L351) | loads one position stack, constructs elapsed times, and calculates duration before ROI extraction; the startup crop belongs here |
| [`_analysis_runner.py:14,312`](../src/cali/analysis/_analysis_runner.py#L14) | `GetSn(dff, range_ff=[0.25, 0.5], method="median")` for calcium peak thresholds — the only OASIS use in the *analysis* stage |

### 3.2 Settings / schema (must extend)

| Location | What |
| --- | --- |
| [`_model.py:964`](../src/cali/sqlmodel/_model.py#L964) | `ExtractionSettings.decay_constant` |
| [`_model.py:972-1006`](../src/cali/sqlmodel/_model.py#L972-L1006) | `__eq__` / `__hash__` — **easy to miss**; the runner uses settings equality to decide whether to re-run extraction. New fields must be added to both. |
| [`_model.py:1702-1703`](../src/cali/sqlmodel/_model.py#L1702-L1703) | `Traces.den_dff`, `Traces.inferred_spikes` |
| [`_model.py:1623`](../src/cali/sqlmodel/_model.py#L1623), [`_model.py:1771-1800`](../src/cali/sqlmodel/_model.py#L1771-L1800) | `ROI.active` is singular and `DataAnalysis` stores one threshold/frequency result per ROI, so two spike methods cannot be represented independently |
| [`_model.py:1803-2023`](../src/cali/sqlmodel/_model.py#L1803-L2023) | `FOVAnalysis` mixes calcium fields with one set of spike matrices/bursts and one `active_roi_labels` ordering; spike fields must become method-bound |
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
| [`_main_plot.py:1126-1183`](../src/cali/plot/_main_plot.py#L1126-L1183) | `get_available_plots()` filters only by pipeline stage/experiment type; it needs stored-method and active-backend context |
| [`_analysis_gui.py` / evoked plots](../src/cali/gui/_analysis_gui.py) | `led_pulse_on_frames` is persisted in source/user frame coordinates and several consumers currently subtract `1` independently; startup cropping needs one shared transform |

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
`test_gui_export.py`. Renaming them without a compatibility reader silently invalidates every saved
`settings.json`.

Do the naming cleanup in this migration, but first separate stable machine identifiers from display
labels:

```python
TraceExportId = Literal["raw", "neuropil", "corrected", "dff", "den_dff",
                        "inferred_spikes", "inferred_spikes_binary"]
```

- Persist `TraceExportId`, never a user-facing label. Add a settings-schema version and a
  `_LEGACY_TRACE_OPTION_IDS` map that converts all current strings above to the new IDs on load;
  unknown keys produce a warning rather than being silently discarded. Save only canonical IDs
  after conversion. Database-to-CSV dispatch also keys on the IDs.
- Keep backend-neutral internal names such as `inferred_spikes`, `_plot_inferred_spikes`, and
  `INFERRED_SPIKES_TRACES`: both backends satisfy that contract. Keep explicit implementation names
  such as `OasisBackend` and `CascadeBackend` where the code really is method-specific.
- Resolve display labels from the **stored trace provenance and active results-backend selector**,
  not from the extraction checkboxes' current values. `den_dff` is always labelled
  **OASIS Denoised ΔF/F** for every spike-output selection.
  The shared spike export/plot is labelled **OASIS Inferred Spikes (a.u.)** or
  **CASCADE Inferred Spike Rate (expected spikes/frame)**. The binary form is labelled
  **OASIS Thresholded Inferred Spikes** or **CASCADE Supra-threshold Excursions**.
- Do not create separate persisted export IDs for OASIS and CASCADE. The stable `inferred_spikes`
  ID addresses a semantic product; when both outputs exist, interactive views use the selected
  backend and batch export writes one method-qualified file/table partition per backend so neither
  can overwrite the other.

Method-specific results move out of the one-per-ROI `DataAnalysis` row into one `SpikeAnalysis` row
per stored `SpikeTrace` and analysis run. Because the method is a required relationship, column
names need semantic precision rather than redundant backend prefixes. In the P2 migration map the
existing columns as follows and update queries/exports:

```text
DataAnalysis.inferred_spikes_threshold             -> SpikeAnalysis.threshold
DataAnalysis.inferred_spikes_frequency             -> SpikeAnalysis.suprathreshold_sample_rate_hz
DataAnalysis.inferred_spikes_rising_edge_frequency -> SpikeAnalysis.suprathreshold_rising_edge_rate_hz
```

The first rate counts above-threshold *samples* per second, not biological spikes, and is populated
only on migrated/new OASIS rows. CASCADE rows instead use `expected_spike_rate_hz`,
`expected_spike_count`, and `suprathreshold_excursion_rate_hz`. `SpikeAnalysis.threshold`, applied
mode, and units are backend-neutral fields whose owning `SpikeTrace → SpikeInferenceRun` makes the
method unambiguous. Remove internal reliance on the old `DataAnalysis` spike attributes; if they
are public API, expose deprecated read-only accessors only when exactly one spike result exists and
raise an explicit ambiguity error when both do.

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
├── __init__.py        # construct the selected spike backends
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
returns `OasisResult`; its spike output is wrapped as a `SpikeInferenceResult` when OASIS retention
is selected. `CascadeBackend.infer()` returns only `SpikeInferenceResult`. This split is deliberate:
a generic `DeconvolutionResult` with an optional `den_dff` loses OASIS's `sn`, even though the
current inline calcium threshold consumes that exact value. It also falsely suggests the two
algorithms produce interchangeable outputs.

`provenance` is not merely a log dictionary, but it also must not duplicate run-constant strings and
hashes on every ROI. Add these normalized tables and relationships:

- **`ExtractionFrameWindow`** (`__tablename__ = "extraction_frame_window"`): one row per extraction
  result/FOV, enforced by `UNIQUE(extraction_result_id, fov_id)`. It stores the requested startup
  discard value/unit, timing/conversion source, original and retained counts, zero-based source start
  frame, original start time, actual excluded duration, and schema version. Each base `Traces` row
  references its window so detached analysis/export can translate retained indices back to source
  coordinates without consulting mutable settings.
- **`SpikeInferenceRun`** (`__tablename__ = "spike_inference_run"`): one row per extraction
  `CaliResult` **and method**, enforced by `UNIQUE(extraction_result_id, method)`. It contains method,
  units, backend package/revision, resolved model, catalogue revision, config/ordered-weight
  manifest SHA-256, resolved device/dtype, model sampling rate, smoothing, kernel type, and a
  provenance schema version. `CaliResult.spike_inference_runs` is one-to-many. Do not put the
  resolved manifest on reusable `ExtractionSettings`, where it can become stale if files under the
  same model name change.
- **`SpikeTrace`** (`__tablename__ = "spike_trace"`): one row per base `Traces` row and inference
  run, enforced by `UNIQUE(trace_id, spike_inference_run_id)`. It owns `values`, `valid_start`,
  nullable `valid_stop`, per-ROI noise estimate, and selected model noise level. The base `Traces`
  row keeps the shared DFF/OASIS-denoised arrays plus the rebased retained x-axis and frame-window
  FK; it exposes `spike_traces: list[SpikeTrace]` rather than one ambiguous spike array.
- **`SpikeAnalysisSettings`**: one child of `AnalysisSettings` per method, enforced by
  `UNIQUE(analysis_settings_id, method)`. It owns that method's threshold mode/value/AP fraction and
  spike-specific burst/synchrony parameters. Equality/hash use a sorted tuple of child semantic
  keys so child order cannot trigger or suppress re-analysis.
- **`SpikeAnalysis`**: one ROI-level result per `SpikeTrace` and analysis `CaliResult`, enforced by
  `UNIQUE(spike_trace_id, analysis_result_id)`. It owns the applied threshold/mode/units,
  method-specific rates/counts, and `spike_active`. `DataAnalysis` remains the one-per-ROI home for
  calcium metrics and `calcium_active`.
- **`SpikeFOVAnalysis`**: one method-bound FOV result per FOV, inference run, and analysis result,
  enforced by `UNIQUE(fov_id, spike_inference_run_id, analysis_result_id)`. It owns the spike
  active-ROI ordering, binary/rising-edge matrices, CCG/synchrony values, population bursts, and
  spike-derived dimensionality-reduction products. `FOVAnalysis` retains calcium
  correlations, calcium bursts, and calcium-based clustering; rename its ambiguous
  `active_roi_labels` to `calcium_active_roi_labels` during migration.

Configure `Traces.extraction_frame_window`, `Traces.spike_traces → SpikeTrace.inference_run`, and
the analysis relationships for eager `selectin` loading so detached traces remain self-describing during
`AnalysisRunner.run(fovs, analysis_settings)`. Audit every trace-loading query before `expunge_all()`;
do not solve detached access by copying provenance onto every ROI. Internal callers must use an
explicit `get_spike_trace(method)`/iteration API. A deprecated singular `inferred_spikes` accessor,
if retained for one release, works only when exactly one child exists and raises on dual-output data.

The legacy migration creates one OASIS/a.u. `SpikeInferenceRun` per historical extraction result,
copies each existing spike array into an OASIS `SpikeTrace`, and moves ROI/FOV spike results into the
new method-bound analysis tables. It also creates a zero-discard `ExtractionFrameWindow` for each
historical extraction/FOV (`source_start_frame=0`, original=retained count inferred from the trace).
Rows without an extraction result retain documented synthetic provenance. A nullable `valid_stop`
means `len(values)` for migrated data.

### 4.3 Data flow after the change

```text
ExtractionRunner._extract_trace_data_per_position(fov)
│
├─ Phase 0  once per source position/file
│     full_data, full_meta = dataset.isel(p=position, metadata=True)
│     timing = build_timing_descriptor(full_meta, len(full_data))
│     window = resolve_initial_discard(settings, timing)
│     data = full_data[window.source_start_frame:]
│     meta = full_meta[window.source_start_frame:]
│     elapsed_ms = rebase(timing.timestamps_ms[window.source_start_frame:])
│
├─ Phase A  per ROI (loop, threaded per-FOV as today)
│     mask → raw trace → neuropil correction → calculate_dff()
│     └─ collect _RoiParts(label_value, raw, corrected, neuropil, dff, roi_size, ...)
│
├─ Phase B  per FOV (one OASIS call; optional one CASCADE call, both batched)
│     dff_matrix = np.vstack([p.dff for p in parts])        # (n_rois, T)
│     oasis = OasisBackend.infer_all(dff_matrix, frame_rate, decay_constant=τ, cancel=cancel)
│     spike_results = {}
│     if "oasis" in spike_methods:  spike_results["oasis"] = from_oasis(oasis)
│     if "cascade" in spike_methods: spike_results["cascade"] = CascadeBackend.infer(...)
│
└─ Phase C  per ROI
      build Traces(dff=..., den_dff=oasis.den_dff[i], x_axis=elapsed_ms,
                   extraction_frame_window=window)
      + one SpikeTrace(values=result.spikes[i], provenance...) per selected method
      + optional calcium DataAnalysis, passing oasis.sn_by_roi[i]
      + optional SpikeAnalysis per SpikeTrace
```

`AnalysisRunner` must become provenance-aware before CASCADE is exposed. It re-reads
all `Traces.spike_traces` from the DB without extraction settings; it must dispatch OASIS and/or
CASCADE threshold/rate semantics from each eagerly loaded `SpikeInferenceRun` plus trace-level
fields and crop each analysis to its stored valid interval. Extraction-time analysis and
analysis-only re-runs must share one helper and produce identical method-bound results. Phase B/C
is atomic per FOV: cancellation or failure of any selected backend stores neither backend, rather
than silently degrading a requested comparison run.

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
7. Keep Phase B behind the synchronous backend interface, but do not bake a global lock into the
   runner. The CASCADE backend may delegate that call to the bounded, device-owning inference
   service in P3b; this keeps a later post-pass/service implementation possible without reshaping
   Phases A/C again.

**Acceptance**: full suite green, and a re-run of an existing dataset produces byte-identical
   `den_dff` / `inferred_spikes`, OASIS `sn`, per-ROI metrics, and active flags. Use a checked-in
   regression fixture rather than a one-off-only script and assert `rtol=0, atol=0`.

### P2 — Centralized migrations, settings, and multi-output schema (2–4 days)

#### P2a — one database-opening choke point (blocker before mapped columns)

There are dozens of direct `create_engine()` call sites across model loaders, the runs panel, GUI,
runner, exports, utilities, and plots. The existing migration is called by
`create_database_and_tables()` and `AnalysisSettings.load_from_database()`, but ordinary read paths
still bypass it. Once mapped columns/relationships are added to `trace`, `data_analysis`, or the
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

**Normalized child tables do not remove the migration requirement.** This revision deliberately
uses `SpikeTrace`, `SpikeAnalysis`, and `SpikeFOVAnalysis` because their cardinality is one per
method, but it still adds relationships plus extraction settings, source-result linkage, and calcium
activity/ROI-ordering fields. Old databases must be upgraded before any ORM query touches that
schema. Avoiding P2a would require keeping *all* new mappings invisible to ordinary loaders and
hydrating them conditionally forever. Keep P2a and land it independently so the repository has a
general schema-evolution path beyond CASCADE.

#### P2b — settings and normalized provenance

`ExtractionSettings` gains the multi-output, startup-discard, and CASCADE inputs:

```python
SpikeMethod = Literal["oasis", "cascade"]
spike_methods: tuple[SpikeMethod, ...] = ("oasis",)  # canonical non-empty subset
cascade_model: str | None = None              # required iff cascade is selected
cascade_device: str = "auto"                  # "auto" | "cpu" | "cuda" | "mps"
frame_rate_verified: bool = False  # shared by seconds-mode discard and CASCADE validation
discard_initial_value: float = 0.0
discard_initial_unit: Literal["frames", "seconds"] = "frames"
```

- Add all six fields to `ExtractionSettings.__eq__` **and** `__hash__`. Parse list/tuple method input, reject an
  empty or unknown selection, remove duplicates, and store methods in one canonical order before
  comparing/hashing. Persist the selection as a JSON array and expose an immutable tuple at the
  model boundary. Require and validate CASCADE settings whenever `"cascade" in spike_methods`;
  ignore/clear them only when CASCADE is absent. Resolve the exact model before persisting/reusing
  settings; `None` must never mean "whatever the catalogue chooses today". The SQLModel/migration
  default remains OASIS-only so old and headless workflows work without the optional dependency.
  P8 deliberately checks CASCADE only for a brand-new GUI setup.
- Validate startup discard as specified in §2.2: finite and non-negative; integral in Frames mode;
  Seconds mode requires trusted timestamps/frame-period metadata or `frame_rate_verified=True`.
  Migrate old settings to `0 frames`, which is an exact no-op. Add the new frame-window table and
  ensure settings equality causes re-extraction whenever requested value, unit, or verified
  timebase changes.
- Add `SpikeAnalysisSettings` rows per §4.2. Migrate the current scalar spike fields from each
  `AnalysisSettings` row into an OASIS child without changing their values. A new CASCADE child uses
  `CASCADE_AP_THRESHOLD = "cascade_ap"` and `cascade_ap_threshold_fraction = 1 / e` by default.
  Threshold, burst, synchrony/CCG, and rising-edge settings are method-bound because dual analysis
  must not make one backend overwrite the other's configuration. Include a sorted tuple of all
  child semantic keys in `AnalysisSettings.__eq__` and `__hash__`; equality gates re-analysis.
  A bare headless `AnalysisSettings()` materializes the legacy OASIS child defaults. The combined
  GUI creates children for its selected/stored outputs, and headless analysis validates that every
  method being analyzed has exactly one child configuration.
- Validate each child threshold mode against its matching trace provenance:

  ```text
  OASIS:   global | multiplier
  CASCADE: global | cascade_ap
  ```

  `cascade_ap_threshold_fraction` is used only for `cascade_ap`. If the user selects a global
  CASCADE threshold, require an explicit value and label it in spikes/frame; do not silently reuse
  the OASIS-oriented default `3.0`. Dual runs therefore carry two independent threshold configs.
- Add `DataAnalysis.calcium_active`; move every spike threshold, metric, and active flag to
  `SpikeAnalysis` using §3.4's mapping. CASCADE rows add nullable `expected_spike_rate_hz`,
  `expected_spike_count`, and `suprathreshold_excursion_rate_hz`; OASIS rows populate the two legacy
  sample/rising-edge rates. A CASCADE rising edge is an excursion, not a spike.
- Split the spike-specific columns out of `FOVAnalysis` into `SpikeFOVAnalysis`, one row per method;
  keep calcium correlations/bursts/clustering in `FOVAnalysis` and rename its ROI ordering to
  `calcium_active_roi_labels`. Add
  `CaliResult.source_extraction_result_id` so every future analysis-only result identifies the exact
  trace generation it analyzed rather than relying on "latest trace" ordering. A combined
  extraction+analysis result points to itself; an analysis-only result points to the explicitly
  selected extraction result.
- Add all normalized tables/constraints specified in §4.2. The migration order is: create the new
  tables; add/normalize `spike_methods`, discard settings, frame windows, and source-extraction
  linkage; insert one OASIS/a.u.
  `SpikeInferenceRun` per historical extraction result; copy legacy trace arrays into `SpikeTrace`;
  create OASIS `SpikeAnalysisSettings`; move ROI and FOV spike results into `SpikeAnalysis` and
  `SpikeFOVAnalysis`; verify row counts, value checksums, uniqueness, and foreign keys; then advance
  `user_version`. For a historical analysis-only result lacking an explicit source, resolve the
  unique latest compatible extraction result at or before its timestamp and persist
  `legacy_trace_resolution="latest_preceding_inferred"`. If resolution is absent or ambiguous, do
  not guess or make the whole database unusable: record an auditable migration issue, leave only the
  affected spike-analysis rows on the read-only compatibility path, warn in the UI, and block
  re-analysis/comparison for those rows until the user chooses a source extraction.
- Stop all new writes to the old singular spike columns in `Traces`, `DataAnalysis`, and
  `FOVAnalysis`. Retain them physically, read-only, for one compatibility release after verified
  backfill; new code reads only normalized tables. Budget the temporary duplication in §6.6 and
  remove the columns in a later versioned migration.

**Acceptance**: take an unmodified pre-migration `.cali` fixture and exercise it through the runs
panel queries, experiment/result/settings direct loaders, trace plotting, CSV export, analysis-only
re-run, and save/reopen. Assert that every route migrates before its first ORM query, repeated opens
are no-ops, an interrupted migration resumes safely, provenance relations remain available after
detachment, every legacy trace/ROI/FOV spike value is exactly represented by one OASIS child row,
every legacy FOV has an exact zero-discard frame window, and all legacy calcium results remain
unchanged. Also create a new dual-output/nonzero-discard fixture and assert all cardinality,
uniqueness, source-index mapping, and explicit source-extraction constraints.

#### P2c — Initial-frame exclusion and source-index transform (1–2 days)

1. Replace `_get_elapsed_time_ms_list()` with the general `TimingDescriptor` required by §6.1 and
   resolve an `ExtractionFrameWindow` once per loaded source position. This common timing path serves
   both seconds-mode discard and later CASCADE frame-rate validation.
2. Crop `data`, per-frame metadata, and the timing vector once, before Phase A. Rebase retained time
   to zero and pass only cropped arrays into ROI extraction. Do not independently slice each derived
   trace; one upstream crop guarantees equal lengths across raw, neuropil, corrected, DFF, OASIS,
   and every `SpikeTrace`.
3. Persist `ExtractionFrameWindow` and attach it to every base trace. Keep source data immutable;
   this is a view/copy used by the extraction run, not a rewrite of TIFF/Zarr input.
4. Add a single `SourceFrameTransform` helper for retained ↔ source frames/times and route LED
   intervals, evoked peak classification, plot bands, event tables, and frame-index exports through
   it. Transform intervals by intersection with the retained window; never mutate/re-save the global
   source-coordinate `AnalysisSettings` values.
5. Preflight retained length against all enabled consumers and selected backends before ROI work.
   Surface a per-FOV error instead of letting DFF/OASIS/CASCADE fail later on a short array.
6. Keep analysis indices and the default plot x-axis relative to the retained trace (`0` is the
   first retained frame/time). Show the source offset in plot metadata/tooltips and include both
   retained and source indices in event-oriented exports.

**Acceptance**: `0 frames` is byte-for-byte identical to the baseline; frame and seconds modes
resolve documented boundary cases without off-by-one errors; every stored trace has the retained
length; extraction-time and later re-analysis use identical rebased time/event coordinates; and two
FOVs with different trusted timestamp spacing may resolve different frame counts from the same
seconds request while preserving the requested duration.

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

Run the cached predictor inside a long-lived, single-worker `CascadeInferenceService` that owns the
resolved device and model cache. FOV workers call a synchronous `infer()` facade that submits to a
bounded queue and waits for the result; this preserves the P1 backend seam while avoiding arbitrary
FOV threads owning Torch modules. Start with queue capacity 1 and include blocked callers' retained
`_RoiParts`/image data in the RSS measurement. Keep the global-lock implementation only as a simple
reference/fallback, not the assumed production architecture.

Ship the cached path only if all of these gates pass:

1. CPU output matches the direct upstream call on synthetic traces **and a real bundled example**
   at `rtol=1e-5, atol=1e-6` (tighten if observed equivalence permits).
2. OASIS `den_dff` is bit-identical for OASIS-only, CASCADE-only, and dual-output selections.
3. Two successive FOVs cause no additional `torch.load()` for an unchanged model manifest/device.
4. Peak resident memory stays within a documented budget and cancellation completes after the
   current chunk, not after the whole FOV.
5. Representative-plate wall time materially improves over the once-per-FOV reference path. If it
   does not, keep the simpler upstream adapter as production and retain the cached implementation
   only behind an experimental flag or remove it.
6. Measure the numbers users actually experience: complete OASIS-only, CASCADE-only
   (**OASIS denoising + CASCADE spikes**), and dual-output extraction, cold and warm. Separate
   inference, persistence, ROI analysis, and FOV analysis time so the added cost of retaining and
   analyzing the already-computed OASIS spike result is visible. Do not present predictor-only
   speedups as total-pipeline speedups.
7. Compare the dedicated inference service with the global-lock fallback under the real FOV pool.
   The service ships only if its ownership/cancellation behavior is correct and its queue does not
   create unacceptable retained-FOV memory.

```python
@dataclass(frozen=True)
class ModelCacheKey:
    model_dir: Path
    model_manifest_sha256: str   # config + ordered weights; invalidates replaced files
    noise_level: float
    device_type: str
    device_index: int | None
    dtype: str

class CascadeInferenceService:
    # One device-owning worker, bounded request queue, and service-owned cache.
    _model_cache: dict[ModelCacheKey, tuple[torch.nn.Module, ...]]
```

Keying only by `(model_dir, noise_level)` is incorrect: it can return CPU modules for an MPS/CUDA
request and misses in-place weight/config changes. Cache population and inference run under the
same service/resource policy. A future multi-device implementation may own one service per device;
do not assume concurrent FOV calls into the same Torch model improve throughput.

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

#### The catalogue (156 models, 13 families)

| Family | N | Notes |
| --- | --- | --- |
| `Global_EXC_*` | 72 | universal excitatory; 17 rates (1, 2, 2.5, 3, 4.25, 5, 6, 7, 7.5, 10, 12.5, 15, 17.5, 20, 25, 30, 40 Hz) |
| `GC8_EXC_*` / `GC8s` / `GC8f` / `GC8m` | 17/13/8/6 | GCaMP8, combined and per-variant |
| `Online_model_*` | 20 | causal kernel, real-time inference |
| `OGB_zf_pDp_*` | 5 | zebrafish OGB1, pDp |
| `Global_IN_*` | 3 | **inhibitory** interneurons |
| `Interneurons_GC8+_*` | 4 | GCaMP8+ interneuron high-noise models at 7.5/30 Hz |
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

### P5 — Wire extraction and persist one/both outputs (1 day)

In `_extract_trace_data_per_position`, Phase B always runs OASIS once, retains its spike result when
selected, and additionally runs CASCADE when selected:

Preflight the canonical selection, optional package, model manifest/checksum, device, and frame-rate
compatibility before plate/FOV work begins. A dual run with an unavailable CASCADE component fails
up front rather than doing OASIS work and discovering the problem late.

```python
oasis = OasisBackend().infer_all(
    dff_matrix,
    frame_rate,
    decay_constant=decay_constant,
    cancel=cancel,
)
spike_results: dict[SpikeMethod, SpikeInferenceResult] = {}
if "oasis" in extraction_settings.spike_methods:
    spike_results["oasis"] = SpikeInferenceResult.from_oasis(
        oasis, trace_length=dff_matrix.shape[1]
    )
if "cascade" in extraction_settings.spike_methods:
    spike_results["cascade"] = cascade_backend.infer(
        dff_matrix, measured_frame_rate, cancel=cancel
    )
```

Phase C always stores `oasis.den_dff`; it passes `oasis.sn_by_roi[i]` to the shared analysis helper,
creates/reuses one `SpikeInferenceRun` per selected method, and stores one `SpikeTrace` child per
method/ROI with values, provenance FK, valid interval, per-ROI noise, and selected noise level.
Timing source/rate/jitter and the retained-to-source mapping are stored once in the FOV's persisted
`ExtractionFrameWindow`, referenced by each base trace. Do not use a `CompositeBackend`:
denoising is not an interchangeable choice, while spike results are an explicit method-keyed
collection. Flush/commit only after all selected methods and analysis rows for the FOV succeed; a
CASCADE error in a dual run must not quietly leave an OASIS-only result.

Also update:

- Add a worker regression test that reads every dispatch, timing, and startup-discard setting after
  `session.expunge()`. Ordinary scalar columns are loaded with the row, so do not claim the existing
  eager-access tuple is needed to prevent `DetachedInstanceError`; extend it only if an actual
  expiry/deferred-load test proves necessary.
- `_visualize_experiment.py:257` — print every retained spike method and its resolved
  model/manifest, units, device, and valid interval alongside the OASIS decay constant; also print
  each FOV's requested discard, resolved source start, retained count, and timing source.
- Exports — include backend/units/model/manifest/valid interval in metadata, dispatch through the
  stable trace/export IDs in §3.4, render labels from stored provenance, and include the discard
  request plus resolved frame-window/source offsets. When both spike methods are present, write
  method-qualified files/table partitions in a deterministic order and never overwrite one.

Keep CASCADE inaccessible from the released GUI until P6 and P7 pass.

`AnalysisSettings.enable_spikes` controls whether **all retained spike traces** are analyzed; it does
not choose which outputs extraction retains. Extraction-only runs must still be reusable for a later
spike-enabled re-analysis. The extraction-owned checkboxes are the only backend selection control;
do not overload the analysis checkbox or silently analyze just the graph selector's current method.

### P6 — Make ROI/FOV spike analysis method-bound (2–3 days)

This phase precedes GUI exposure and is part of the CASCADE feature, not follow-up cleanup.

1. Move extraction-time and re-analysis ROI calculations into shared calcium and spike helpers. The
   spike helper consumes one `SpikeTrace` plus its matching `SpikeAnalysisSettings` and produces one
   `SpikeAnalysis`; callers iterate all retained methods in deterministic order. Running both must
   produce the same per-method rows as running each method alone.
2. For CASCADE, compute `expected_spike_count = sum(valid_rate)` and
   `expected_spike_rate_hz = mean(valid_rate) * measured_frame_rate`. Leave the OASIS-only
   sample/rising-edge rates unset; do not mix their meanings. Store CASCADE threshold crossings as
   `suprathreshold_excursion_rate_hz`. OASIS rows keep semantically renamed sample/rising-edge rates
   and leave CASCADE fields unset.
3. Validate each persisted `SpikeAnalysisSettings.threshold_mode` against its
   `SpikeInferenceRun.method`. For `cascade_ap`, apply the model-derived AP threshold from §6.2 only
   inside the valid interval using `cascade_ap_threshold_fraction`; for `global`, require an explicit
   spikes/frame value. Store applied mode and units on `SpikeAnalysis`. Suppress an edge at the first
   valid sample if the signal is already above threshold there.
4. Crop/mask each method to its own valid interval before raster, burst, synchrony, CCG, and
   dimensionality reduction. Invalid CASCADE padding must never be treated as inactivity or exported
   as observed rate. A comparison plot uses the intersection of both methods' valid intervals.
5. Store `DataAnalysis.calcium_active` once and `SpikeAnalysis.spike_active` per method. For the
   backward-compatible summary only, set `ROI.active = calcium_active or any(spike_active_by_method)`
   across enabled pillars. Calcium consumers use `calcium_active`; every spike plot/FOV computation
   uses the active set for its selected method. Neither pillar uses the union summary flag.
6. Refactor `_fov_analysis_parallel.py` so it does not globally require `den_dff` or a singular spike
   trace. Build the calcium collection once, then build one spike collection and one
   `SpikeFOVAnalysis` per method. This preserves today's calcium ROI set while preventing OASIS and
   CASCADE matrices, active-label orderings, correlations, or bursts from overwriting each other.
7. CSV/export code must emit method, threshold mode, and threshold units beside every threshold and
   result. Dual export is long-form with a method column or separate method-qualified files; a bare
   numeric threshold or metric must never mix methods.
8. Make the plot registry provenance-aware. Give each `AnalysisProduct` a stable ID plus optional
   `supported_spike_methods`/`required_metrics` metadata. `get_available_plots()` receives the
   methods stored on the selected run and the active results-backend selection. Filtering in the GUI
   is not sufficient: every plot/compute function validates its method and required metric for
   headless use. Multi-run aggregation must select/facet by method and reject pooled cross-method
   values.

**Acceptance**: extraction-time analysis equals analysis-only re-runs for OASIS-only,
CASCADE-only, and dual-output runs; each method's dual-run results equal its single-method results;
adding or removing retained spike outputs cannot change any `den_dff`, calcium metric, or calcium
FOV membership.

### P7 — Make the optional dependency installable (½–1 day)

CascadeTorch is not on PyPI, and the currently inspected checkout at
`c6978d5ff33edad8792c76e040412f0636913092` is **not installable as written**: `setup.py` uses
`find_packages()`, but `cascade2p/` has no `__init__.py`, so `find_packages()` returns no packages.
Do not leave this as an external wait state. Create a small maintained GPL fork, add
`cascade2p/__init__.py`, verify the built wheel contains/imports `cascade2p`, pin that fork's full
SHA, and submit the packaging fix upstream separately. Move back to upstream only after an immutable
upstream commit passes the same clean-install/golden tests.

Enable Hatch direct references and use the maintained fork initially:

```toml
[tool.hatch.metadata]
allow-direct-references = true

[project.optional-dependencies]
cascade = [
  "CascadeTorch @ git+https://github.com/<maintained-fork-owner>/CascadeTorch@<full-commit-sha>",
]
```

Notes:

- Do not use `@master`. Record the package commit alongside model/catalogue provenance.
- The upstream distribution already declares `torch` and `ruamel.yaml`; reading configs with
  PyYAML inside cali does **not** remove `ruamel.yaml` from installation while that metadata remains.
  It also declares Matplotlib, seaborn, and h5py. Do not simply delete them from metadata:
  `cascade2p.utils` imports Matplotlib unconditionally today. In the fork, first make plotting/
  training-only imports lazy, then trim only dependencies proven unnecessary by clean inference,
  download, and golden tests (seaborn/h5py are candidates). Keep fork changes minimal and documented.
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

### P8 — GUI, plots, docs, and release gate (1–2 days)

`_extraction_gui.py`:

- `TraceExtractionData` gains `spike_methods`, `cascade_model`, `cascade_device`,
  `frame_rate_verified`, `discard_initial_value`, and `discard_initial_unit`.
- Add **Discard at Start** beside the trace-extraction/timebase controls: a non-negative value plus
  **Frames** / **Seconds** radio buttons. Frames mode uses an integer editor and `frames` suffix;
  Seconds mode uses a decimal editor and `s` suffix. Default/reset is `0 frames`. The tooltip states
  that the exclusion is applied independently to every position/file before all trace calculations,
  while detection/ROI masks are unchanged.
- In Seconds mode, show the timing source and resolved frame count when the selected data make it
  available. If only an unverified setting/exposure is available, require the general
  `frame_rate_verified` confirmation; do not silently round from the default 10 fps. Surface the
  per-FOV resolved count in the run summary/log because it may differ across files.
- Add a **Spike Outputs** group with independent **CASCADE** and **OASIS** checkboxes and require at
  least one. CASCADE is checked for a brand-new GUI setup; OASIS is an optional comparison output.
  Its tooltip must say that OASIS denoising still runs for `den_dff` when the OASIS *spike output* is
  unchecked. There is no denoising checkbox; Decay Constant remains enabled.
- The editable output checkboxes are owned by `ExtractionSettings`, even if the combined-run UI
  places them beside the existing **Inferred Spikes** analysis checkbox. In re-analysis-only views,
  show all stored methods read-only. Adding/removing a retained output requires re-extraction and
  must never masquerade as an analysis-only threshold change.
- If selected CASCADE dependencies/models are unavailable, keep CASCADE visibly checked with an
  install/download action and block the run; never silently uncheck it or fall back to OASIS. Old
  singular `spike_method` JSON migrates to a one-element `spike_methods` list. Old settings and bare
  headless construction remain OASIS-only; the widget's new/reset state is CASCADE-only.
- Populate model choices from the pinned catalogue/local cache, filtered by measured frame rate;
  show family/smoothing/noise/kernel metadata rather than silently choosing a default. Enable and
  validate model/download/device controls only while CASCADE is checked.
- `value()` / `setValue()` / `reset()` and `_cali_gui.py` settings JSON must round-trip the canonical
  method list, discard value/unit, verification flag, and all CASCADE fields. The compatibility
  reader accepts both old `spike_method` and new `spike_methods`, defaults missing discard fields to
  `0 frames`, rejects invalid values/units or an empty/unknown method result, and writes only the new
  form.
- Make `_SpikeThresholdWidget` render one method-labelled section/tab per stored output: retain
  GLOBAL/MULTIPLIER for OASIS and expose GLOBAL/CASCADE_AP for CASCADE. Round-trip each
  `SpikeAnalysisSettings` child independently; label a CASCADE global threshold in spikes/frame and
  require an explicit value instead of carrying over the OASIS default `3.0`. Correct the existing
  OASIS tooltip, which describes a percentile rule while the code uses MAD.
- Retitle "ΔF/F0 and Deconvolution" to "ΔF/F0 and Spike Inference". Migrate plot/export
  settings to stable IDs and make labels, titles, tooltips, and units provenance-aware as in §3.4.
- Filter graphs from the **selected stored run**, not the editable checkboxes for a future
  extraction. With exactly one stored method, hide the backend selector and omit every incompatible
  spike product. If both methods exist, show a results-level **Spike Backend** selector (CASCADE
  first) and keep shared inferred-trace/raster/synchrony/CCG/burst products single in the combo; they
  render the selected backend's data, title, and units. Add a small explicit **Backend Comparison**
  category only for dual-output runs. OASIS calcium plots remain visible for every selection because
  `den_dff` is always OASIS-produced.
- Replace hard-coded `"Inferred Spikes (a.u.)"` axes throughout `plot/` with a provenance-derived
  unit label. For the combined `den_dff` + inferred-spike plot, use normalized traces or separate
  y-axes; never place CASCADE spikes/frame and ΔF/F on one axis labelled `a.u.`.
- Rebased trace plots start at retained time/frame zero and display the discarded source offset in
  metadata/tooltips. Evoked overlays use transformed/clipped pulse intervals from P2c rather than
  reading `led_pulse_on_frames` directly.
- Update README with startup-discard semantics/source coordinates plus CASCADE installation, offline
  download, units, valid-edge, model-selection, provenance, end-to-end runtime/storage expectations,
  and cross-backend-comparison guidance; update `_dev/TODO.md` only after the release gates pass.

The visible product matrix is:

| Plot/export family | Active backend = OASIS | Active backend = CASCADE | Dual-only comparison |
| --- | --- | --- | --- |
| OASIS denoised ΔF/F and calcium metrics | show | show | not backend-dependent |
| Shared inferred trace, binary raster, synchrony, CCG, bursts, dimensionality reduction | show with OASIS label/units | show with CASCADE label/units | side-by-side/normalized comparison only |
| OASIS supra-threshold sample/rising-edge rates | show | hide | separate OASIS panel/series |
| CASCADE expected spike rate/count and supra-threshold excursion rate | hide | show | separate CASCADE panel/series |

Do not duplicate every shared graph as an "OASIS graph" and a "CASCADE graph." The backend selector
chooses its method-bound data. Comparison products must use aligned valid intervals and separate
axes/panels or clearly normalized traces; they must not imply that OASIS amplitudes and CASCADE
spikes/frame are directly interchangeable.

The GUI option is enabled only when P1–P7, the dedicated CASCADE CI job, a clean install, and the
end-to-end performance/storage gates in §6.5–6.6 are all green.

### Out of scope — discrete-spike post-processing

Do not include `utils_discrete_spikes.infer_discrete_spikes()` in this migration. Upstream cautions
that discrete predictions are generally not recommended except for exceptional data quality and
can imply false precision. If evaluated later, retain the calibrated expected-rate trace in
its CASCADE `SpikeTrace` and store discrete estimates as a **separate trace product with separate
provenance**; never overwrite the expected-rate trace or label point estimates as ground-truth spike
times.

---

## 6. Semantics: what actually changes in the numbers

**This section matters more than the plumbing.** cali's spike analysis was tuned against OASIS
output; Cascade output has different statistics, and several thresholds silently change meaning.

> **Scope decision (updated 2026-08-14):** preserve the existing OASIS spike producer and its
> analysis semantics as the legacy selectable path. Add a separate CASCADE-aware spike-analysis
> path rather than forcing CASCADE output through OASIS thresholds/frequencies. The binding
> constraint on the calcium side remains unchanged: old `.cali` files must open, and re-running
> calcium analysis must reproduce existing numbers. That is guaranteed by §2.1 plus P2.
>
> Practical consequence: build §6.2(a) and §6.2(b) as the new default spike pipeline. Do **not**
> spend effort making `MULTIPLIER` behave sensibly on a Cascade trace — retire it for that backend.

### 6.1 Frame rate must match the model

`cascade.predict()` computes noise levels with **the model's** `cfg["sampling_rate"]` in its
noise-estimation block, not the recording's. If a 30 Hz model is applied to a 10 Hz recording, both the
noise matching and the temporal scaling are wrong, and the failure is silent — the output looks
plausible.

Do not trust `ExtractionSettings.frame_rate` alone. P2c replaces `_get_elapsed_time_ms_list()` with a
shared timing descriptor whose source enum is acquisition timestamps (`RUNNER_TIME_KEY`), explicit
metadata frame period, user-verified settings, or exposure-only synthetic times. Startup discard in
seconds needs only finite, strictly increasing timestamps; CASCADE additionally requires
approximately uniform intervals. Derive its observed rate as
`1000 / median(diff_ms)`—not the current `len(dff) / (last-first)`, which does exist in extraction
and is off by one sample interval. Do not alter the legacy calcium-duration calculation in this
migration if doing so changes historical calcium metrics.

Exposure duration is not necessarily frame period because camera readout and inter-frame overhead
may be omitted. Therefore an exposure-only synthetic axis must **not** authorize CASCADE model
selection by itself. When real timestamps/frame-period metadata are unavailable, require the user
to confirm the actual acquisition rate (`frame_rate_verified=True`) and persist the source
as `user_verified`; otherwise block CASCADE with an actionable message. Jitter is unknown—not zero—
for that source.

Verify that the measured or explicitly verified rate agrees with both extraction settings and model
sampling rate within a documented tight tolerance (start at 1%, informed by timestamp precision). Reject zero
exposure, irregular sampling, and incompatible models with a clear error. Resampling, if ever added,
must be explicit and tested for its effect on noise and calibration. Persist timing source,
source/retained window, observed rate, model rate, interval jitter (when available), and the
validation result.

### 6.2 The trace is no longer sparse

| | OASIS | Cascade |
| --- | --- | --- |
| Support | sparse — most frames are exactly `0.0` | dense — smooth non-negative rate, few exact zeros |
| Units | arbitrary (ΔF/F event magnitude) | **expected spikes per frame** under the model's calibration domain |
| Edges | full length | first/last 32 frames unpredictable → 0-padded |
| Value of `sum(trace)` | meaningless | ≈ total number of spikes |

Consequences:

- **`compute_inferred_spike_threshold`** ([`_trace_analysis.py:22-71`](../src/cali/analysis/_trace_analysis.py#L22-L71)) in `MULTIPLIER` mode takes `spikes[spikes > 0]`, splits at the median, and computes a MAD. With OASIS that samples the *noise floor of a sparse* signal. With Cascade `spikes > 0` is nearly the whole trace, so the "noise" estimate becomes a statistic of the *signal* — the threshold will scale with activity, and a highly active ROI gets a *higher* bar than a quiet one. **This mode is not usable as-is for Cascade.**

- **`count_thresholded_spike_events`** ([`_trace_analysis.py:229-259`](../src/cali/analysis/_trace_analysis.py#L229-L259)) returns `sum(spike_train > 0)` — the number of frames above threshold. On a sparse OASIS trace that approximates an event count. On a smooth Cascade rate it measures **time spent active**, so the current `inferred_spikes_frequency` is not a spike rate. P2 migrates that legacy OASIS value to `SpikeAnalysis.suprathreshold_sample_rate_hz`; CASCADE rows leave it unset. The existing `num_rising_edges` is the meaningful excursion count for Cascade.

**The redesign.** Three replacements, in priority order. `MULTIPLIER` is retired for the Cascade
backend (it stays for OASIS, where it is correct).

#### (a) Threshold-free rate metrics — build first

CASCADE's output is an expected spike count per frame, so the headline summaries need no binary
threshold:

```python
valid = spikes[valid_start:valid_stop]
expected_spike_rate_hz = float(np.mean(valid)) * observed_frame_rate
expected_spike_count   = float(np.sum(valid))
```

Add `SpikeAnalysis.expected_spike_rate_hz` and `SpikeAnalysis.expected_spike_count` (both nullable
and valid only for a CASCADE-owned row). These are substantially more comparable than OASIS
amplitudes, but they are not automatically interchangeable across indicators, preparations, or
experiments. Transfer calibration can vary across datasets and individual neurons; report the
chosen model and validation domain beside the values and avoid an unqualified "absolute ground
truth" claim.

The valid interval is binding for all consumers, not just this denominator. Cropping only the mean
while allowing padded zeros into event boundaries, bursts, CCG, or synchrony would still produce
biased results.

#### (b) A model-informed binary excursion threshold — build second

The population analyses (§6.3) still need a binary train. Replace the data-derived guess with the
model-derived amplitude of one action potential, which CASCADE computes in
`cascade.predict()`'s `threshold == 1` block:

```python
# _constants.py
CASCADE_AP_THRESHOLD = "cascade_ap"   # persisted SpikeAnalysisSettings.threshold_mode

single = np.zeros(1001); single[500] = 1.0
ap_peak = gaussian_filter(single, sigma=smoothing * sampling_rate).max()
threshold = cascade_ap_threshold_fraction * ap_peak
```

Use `cascade_ap_threshold_fraction = 1/e` as the upstream-derived default, separate from OASIS's
existing default value `3.0`. The threshold is model-specific and ROI-independent. Read smoothing
and sampling rate from persisted provenance, not from a model file that might be unavailable or
changed during later re-analysis. `cascade_ap` is a real user-selectable, persisted mode; provenance
determines whether that mode is legal, not which mode was selected.

This deliberately does **not** reproduce upstream `threshold=1`. Upstream thresholds the rate,
applies `binary_dilation(iterations=int(smoothing * sampling_rate))`, and uses that mask to retain a
wider portion of the rate trace. cali keeps the unmasked expected-rate trace and applies a plain
binary cutoff at analysis time without dilation. Document and test this difference; never claim
parity with upstream's masked-rate output.

For CASCADE, a rising edge is one **supra-threshold excursion**, not an estimated spike count.
Several action potentials inside one smoothing kernel/burst can form a single excursion, so this
metric systematically undercounts at high rates. Store it as
`SpikeAnalysis.suprathreshold_excursion_rate_hz`; the threshold-free expected count/rate in §6.2(a)
remains the headline quantitative output. Frames-above-threshold remains a duty cycle and is not an
event count.

#### (c) Discrete spikes — not in this migration

See "Out of scope" in §5. They are model-derived point estimates, not observed "real spikes", and
must not replace the calibrated expected-rate trace.

**GUI implication (P8)**: `_SpikeThresholdWidget`
([`_analysis_gui.py:1041-1094`](../src/cali/gui/_analysis_gui.py#L1041)) renders one method-bound
configuration per stored output—`GLOBAL`/`MULTIPLIER` for OASIS and `GLOBAL`/`CASCADE_AP` for
CASCADE. Persist both `SpikeAnalysisSettings` children in a dual run and enforce the same legal-mode
matrix in non-GUI analysis code. The OASIS tooltip is also **already wrong today** (it describes a
10th-percentile rule; the code implements a MAD rule). Fix while there.

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
migration, so `GetSn` stays and calcium results are unchanged.

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

**Do add method-qualified per-FOV noise QC metrics** (additive, changes no existing number). Store
OASIS/`GetSn` median/IQR with the calcium `FOVAnalysis` and CASCADE model-noise median/IQR with the
CASCADE `SpikeFOVAnalysis`; never combine their scales. Warn when a FOV's median sits far from the
plate's distribution for the same method. This is a real gap for 96-well work—an out-of-focus or
bleached FOV is currently invisible until the plots look wrong. Upstream does the equivalent via
`plot_noise_level_distribution` and its "models cannot match the noise levels" warning; fold both
into the P3 diagnostics rather than inventing a separate mechanism.

### 6.5 Performance

Rough expectation for one 96-well plate, ~100 ROIs/FOV, 6000 frames, CPU:

- OASIS: ~10–50 ms per ROI, embarrassingly parallel across the FOV ThreadPool.
- CASCADE reference: one batched call per FOV, but it reconstructs/reloads ensembles and allocates
  whole-FOV float64 windows on every call.
- CASCADE cached: the same numerical computation with weights reused and float32 chunks; initially
  serialized by the device-owning inference service for predictable model/device ownership.

Never call either path per ROI. Benchmark reference versus cached inference on representative ROI
counts/trace lengths and one real plate. Record wall time, model-load count, peak RSS, device, Torch
thread settings, chunk size, and numerical difference in the PR; use those measurements to decide
which path ships as the default. Also report end-to-end cold/warm extraction for OASIS-only,
CASCADE-only, and dual-output modes; both CASCADE-containing modes include the mandatory OASIS
`den_dff` pass, while dual mode additionally persists/analyzes the already-computed OASIS spikes.
Include the dedicated service versus global-lock fallback and account for FOV workers blocked while
retaining `_RoiParts`.

### 6.6 Storage footprint

The legacy `Traces.inferred_spikes` is JSON; new values live in one or two `SpikeTrace` rows. OASIS
output is sparse and serializes mostly short `0.0` tokens; CASCADE is dense and full-precision JSON
can grow several-fold. At 100 ROIs × 6000 frames per FOV, the CASCADE rows can add several megabytes
per FOV and plausibly hundreds of megabytes to more than a gigabyte per 96-well plate. Dual mode
adds the OASIS rows plus method-bound ROI/FOV analyses, and the compatibility release temporarily
duplicates migrated legacy OASIS arrays. Storage is therefore a release gate, not an afterthought.

For representative short/long plates, record OASIS-only, CASCADE-only, dual-output, and migrated
legacy-duplication sizes for the `SpikeTrace` payloads and final `.cali` file, SQLite write/read
time, and export/plot load time. Persist CASCADE arrays from float32, but do not assume conversion
alone makes JSON compact: measure the actual serialized representation. Do not blindly round to six
decimal places; first bound the effect on per-sample values, `sum(valid_rate)`, mean rate,
AP-threshold crossings, and the upstream golden comparison.

Define an explicit size and numerical-error budget from those measurements. If dense JSON exceeds
the budget, block release and add a versioned trace-array codec (prefer a compressed binary/BLOB
sidecar with dtype/shape/checksum metadata) while retaining backward-compatible reads of legacy JSON.
All plotting, analysis, and export code should read through one codec abstraction rather than
depending directly on the SQL JSON representation.

---

## 7. Testing plan

| Test | File | Purpose |
| --- | --- | --- |
| Backend/result conformance | `tests/test_spike_inference_backends.py` (new) | batched shapes, finite/non-negative values, structured OASIS `sn/g`, valid interval, normalized `SpikeInferenceRun`/`SpikeTrace` provenance and uniqueness |
| OASIS regression | same | refactored OASIS `den_dff`, spikes, `sn`, metrics, and active flags equal the pre-refactor path at `rtol=0, atol=0` |
| Method isolation | same | OASIS-only, CASCADE-only, and dual runs have identical base traces/calcium results; each method's `SpikeTrace`/ROI/FOV analysis in dual mode equals its single-method result |
| Dual-output atomicity | runner + DB tests | both methods create exactly one child per ROI; cancellation/error in either backend commits neither; a singular compatibility accessor raises instead of choosing when two exist |
| Upstream golden equivalence | `tests/test_cascade_reference.py` | cached CPU predictor matches direct `cascade.predict()` on synthetic inputs and a pinned real example/model; explicit model-rate noise equals upstream implicit noise |
| Input/timing validation | same | empty, 1D, too-short, NaN/Inf, timestamp mismatch, and unconfirmed exposure-only timing fail clearly; monotonic irregular timestamps may resolve seconds discard but are rejected for CASCADE; timing sources persist correctly |
| Startup-discard resolver | new extraction/window tests | exact frame mode; timestamp `searchsorted` seconds mode; verified-rate `ceil` fallback; zero/negative/fractional-frame/all-removed/too-short cases; per-FOV resolutions and persisted actual duration |
| Crop propagation/no-op | extraction + analysis tests | `0 frames` is byte-identical; nonzero crop gives equal retained lengths for x-axis/raw/neuropil/corrected/DFF/denoised/all spike traces; every ROI/FOV metric excludes startup samples |
| Source-index/event transform | evoked plot/export tests | one-based pulse inputs normalize once; pulses before/after/across cutoff are omitted/shifted/clipped correctly; retained/source event indices round-trip and re-analysis matches extraction-time coordinates |
| Valid edges | analysis + FOV tests | effective interval uses `predict()`'s padding formula (not preprocessing's one-earlier window); stored edges are finite zero, but all consumers use only the valid intersection and create no edge at `valid_start` |
| Model cache | same | repeated FOVs load each required ensemble once; model manifest, device/index, or dtype changes miss the cache; `clear_cache()` releases entries |
| Chunk/cancel | same | several chunk sizes equal the reference; CASCADE cancels between chunks and batched OASIS cancels between ROIs, with no partial base or `SpikeTrace` rows committed |
| Device coverage | dedicated CI where available | CPU deterministic; MPS/CUDA compared to CPU using measured, documented tolerances; cache never crosses devices |
| Inference service / memory | extend `tests/test_analysis_threading.py` | one device-owning worker, bounded queue, cancellation, and cache cleanup are deterministic; blocked FOVs, Phase-A `_RoiParts`, chunks, and models stay within the RSS budget; compare global-lock fallback |
| Extraction/re-analysis parity | new | extraction-time and analysis-only runs produce identical per-method `SpikeAnalysis` and `SpikeFOVAnalysis` rows for all three selections |
| Pillar/method-specific activity | FOV tests | calcium-active and per-method spike-active sets are independent; `ROI.active` is only a union summary and never selects method-bound FOV inputs |
| Settings/provenance round-trip | settings + SQLModel tests | canonical methods, discard value/unit/timebase verification, frame windows, both method-bound analysis settings, provenance, and child traces survive GUI/JSON/DB/detachment; equality/hash are complete |
| Dual-backend/discard GUI | GUI/settings tests | new GUI checks CASCADE only and defaults to 0 frames; old JSON migrates to OASIS/0 frames; either/both and Frames/Seconds states round-trip; unavailable selected CASCADE or unverified seconds mode blocks clearly |
| Backend-aware plots/exports | plot, GUI, export tests | old OASIS-labelled keys migrate to stable IDs; the stored-method/results selector drives visibility, labels, and units; dual exports cannot overwrite; comparison plots align valid intervals; mixed-backend aggregation is rejected/faceted |
| Threshold legality | analysis tests | OASIS accepts global/multiplier; CASCADE accepts explicit global/cascade_ap; invalid method/mode pairs fail in GUI and headless paths |
| Excursion semantics | analysis tests | `cascade_ap` plain cutoff is explicitly different from upstream dilated `threshold=1`; closely spaced synthetic APs demonstrate excursion undercount while expected count/rate remain the headline metrics |
| Legacy migration | new | every DB-open route migrates before ORM access; old arrays/results move to OASIS child rows and exact zero-discard windows with checksums/counts; source extraction is resolved/audited; interrupted migration resumes; calcium results are unchanged |
| Download/offline | model tests + CLI tests | atomic install, checksum, interrupted cleanup, cache override, actionable offline error, and advertised CLI command |
| Clean install | packaging CI | build/install `cali[cascade]` in an empty environment, import `cascade2p`, load a model, and infer once under warnings-as-errors |
| End-to-end performance | benchmark job/script | cold/warm OASIS-only, CASCADE-only, and dual extraction; cached versus reference and service versus lock; separate inference/persistence/ROI/FOV analysis costs |
| Storage | benchmark + codec tests | compare all three modes plus temporary legacy duplication; float32/quantization error cannot materially change sums/rates/crossings; trigger compressed codec work if budget is exceeded |

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
| Frame-rate/model mismatch, exposure-only timing, or irregular timestamps | **high** — plausible but invalid output | §6.1: trusted/explicitly verified timing source and no nearest-model auto-selection |
| Startup crop shifts pulse/peak/source indices | **high** — evoked analyses look valid but are misaligned | one persisted `ExtractionFrameWindow`, centralized one-based→zero-based transform, interval clipping, retained+source export columns |
| Seconds-mode discard uses exposure/default fps as true frame period | **high** — wrong amount removed | trusted timestamps/metadata or explicit `frame_rate_verified`; show and persist conversion source/resolved count |
| Provenance absent during detached re-analysis | **high** — wrong units/thresholds | normalized `SpikeInferenceRun`/`SpikeTrace` plus eagerly loaded relationships and explicit source-extraction linkage |
| Threshold mode incompatible with spike method | **high** — plausible but meaningless binary train | persist `cascade_ap` as a real mode and enforce the legal method/mode matrix in GUI and headless paths |
| Dual results overwrite singular ROI/FOV spike fields | **high** — one backend silently wins | method-bound `SpikeAnalysis` and `SpikeFOVAnalysis` rows with uniqueness constraints; no singular internal accessor |
| OASIS and CASCADE metrics are pooled as comparable values | **high** — scientifically misleading aggregate | results-backend selector, method-qualified exports, explicit comparison plots, reject/facet mixed-method aggregation |
| Cached implementation drifts from upstream numerics | **high** | direct once-per-FOV oracle, real-model golden test, safe fallback |
| Optional package currently installs no `cascade2p` package | **high** — feature unusable | maintained minimal fork now, upstream PR, immutable SHA, Hatch flag, clean-install CI |
| Dense/dual `SpikeTrace` rows cause plate-scale DB growth | **high** — storage/I/O regression | §6.6 three-mode budget, float32/error tests, compressed versioned codec if required |
| Any CASCADE selection pays for OASIS denoising + CASCADE | medium | report complete cold/warm three-mode wall time, not predictor-only speedup |
| Dual mode doubles spike analysis/FOV work | medium | benchmark inference, persistence, ROI, and FOV stages separately; user opts in via OASIS comparison checkbox |
| Memory blow-up from retained ROI parts/full float64 windows | medium | release Phase-A parts promptly; float32 chunk iterator; concurrent-FOV RSS gate |
| Zero-padded receptive-field edges bias analyses | medium | persist valid interval and crop/mask every spike consumer and export |
| Startup discard leaves too few frames for DFF/OASIS/CASCADE | medium | per-FOV preflight against every enabled consumer; fail before ROI work with counts and limiting requirement |
| Torch/thread interaction with the FOV ThreadPool | medium | bounded single-worker inference service, service-vs-lock benchmark, retained-caller RSS gate, no per-call global thread changes |
| MPS/CUDA numerical drift vs CPU | medium | device-specific golden tolerances and cache keys; persist resolved device/dtype |
| Model files/catalogue change under the same name | medium | config + ordered-weight manifest hash and catalogue revision |
| Torch/CASCADE warnings fail CI unexpectedly | medium | warnings-as-errors import/inference job; fix or narrowly document/filter each warning |
| GPL-3 CascadeTorch vs BSD-3 cali | medium | no source transplant/vendor; optional dependency documentation and licensing review |
| Renaming OASIS-labelled trace/export strings breaks saved settings + CSV keys | medium | §3.4: stable IDs, settings-schema version, exhaustive legacy-key map, unknown-key warnings, and round-trip fixtures |

**Decisions / required inputs before implementation:**

1. **Decided for this migration:** OASIS remains the sole producer of `den_dff`; spike outputs are a
   non-empty selection of OASIS, CASCADE, or both. OASIS spikes are already computed with `den_dff`
   and are retained/analyzed only when selected.
2. Measure the real acquisition timestamps for the target datasets before choosing a model. The
   10 Hz examples in this document are illustrative, not an assumption the code may make.
3. Dual storage is first-class in P2: `Traces` has zero/one `SpikeTrace` per method, and ROI/FOV
   spike analyses are method-bound child rows. Never put two methods into one array or one set of
   metric columns.
4. New GUI workflows check CASCADE only by default and allow either/both output checkboxes. Migrated
   databases/settings and bare headless settings select OASIS only. If CASCADE is checked, missing
   dependencies/models block explicitly and never cause silent unchecking or backend substitution.
5. The **Inferred Spikes** checkbox is an analysis gate for all stored outputs. The editable output
   checkboxes belong to extraction—even if placed nearby—and changing them requires re-extraction.
6. Packaging proceeds through a minimal maintained fork immediately, with an upstream PR in
   parallel; it is not blocked waiting for an external repository change.
7. Plot/export choices follow stored provenance. A results-backend selector chooses a method when
   both exist; method-specific metrics never mix, while explicit comparison products may show both
   with aligned intervals and honest units. OASIS `den_dff`/calcium plots remain available for all
   selections.
8. Startup discard is an extraction transform applied independently to every source position/file
   before all trace calculations. Detection and ROI masks remain unchanged. Zero is the legacy/no-op
   default; Frames is exact, while Seconds uses timestamps or an explicitly verified rate.
9. Stored arrays, analyses, and default plots use retained-relative coordinates rebased to zero.
   `ExtractionFrameWindow` preserves source coordinates, and one transform handles stimulation
   intervals plus retained/source event export; global analysis settings are never rewritten.

---

## 9. Suggested commit sequence

```text
0. docs: this plan                                                (P0 — done)
1. refactor: extract spike-inference backend seam, batch per FOV  (P1)  ← behaviour-identical
2. refactor(db): central engine factory + versioned migrations     (P2a blocker)
3. feat(db): multi-output traces/settings/analyses + backfill      (P2b)
4. feat(extraction): startup frame/time exclusion + transforms     (P2c)
5. build: minimal CascadeTorch fork, pin/install + warning-clean CI (P7 prerequisite)
6. feat: pinned model catalogue, manifest, download CLI            (P4)
7. test: once-per-FOV upstream adapter + model-rate golden oracle  (P3a)
8. perf: cached/chunked inference service, equivalence + RSS gates  (P3b)
9. feat: wire one/both spike outputs; preserve OASIS den_dff/sn    (P5)
10. perf: full-mode wall time + DB storage/codec release gate      (§6.5–6.6)
11. feat(analysis): per-method ROI/FOV results/comparison safety   (P6)
12. feat(gui): discard controls, output checkboxes, result selector (P8)
13. docs: README extraction/spike-inference sections, TODO.md      (§3.6)
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
