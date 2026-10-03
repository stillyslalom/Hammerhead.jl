```@meta
CurrentModule = Hammerhead
```

# Experiment recipes and run records

Version 1 is a bounded, file-based planar PIV format. It snapshots explicit
pass schedules, built-in preprocessing, a static mask in the original image
frame, ROI, scale, CPU/KA backend, Float32/Float64 precision, and execution
options. It records input bytes and software provenance separately from run
history. Existing native result files and `save_results`/`load_results` remain
independent. See [Save and replay a planar experiment](../howto/experiments.md).
Frozen-camera stereo sequences use the separate
[stereo experiment format](stereo_experiments.md).

## Recipe settings and comparisons

`PIVRecipe` exposes copied public settings: `passes`, `preprocessing`,
`external_preprocess`, `mask`, `roi`, `scale`, `backend`, `image_type`,
`threaded`, `predictor_smoothing`, `mask_threshold`, and
`uncertainty_backend`. Its `recipe_id` is immutable provenance; modifying a
contained vector/matrix/dictionary is detected on save/replay. Construct a new
recipe to record a processing revision.

`recipe_diff(before, after)` compares verified `PIVRecipe` snapshots and returns
a `RecipeDiff` with their identities and ordered `RecipeChange` entries. Each
change has a field `path` and detached `before`/`after` values. Dictionary fields
are traversed in sorted order; ordered passes, preprocessing and validators use
1-based positional paths, such as `passes[2].window_size` and
`preprocessing[1].options.sigma`. Reorders appear as changes at the affected
positions, rather than inferred move operations. The report supports iteration,
integer indexing, `length`, and `isempty`, and has readable text/plain display.

Small settings remain scalars/tuples; added/removed nested items become named
tuples. A `missing` value means an absent sequence item or mapping field, while
`nothing` remains a real optional-setting value. Embedded arrays always become
`RecipeArraySummary(size, element_type, sha256)` values, even inside added or
removed preprocessing steps. Digests include shape, precision and column-major
content through the experiment identity encoding. Reports retain no mask or
background array, and their summaries use chunked canonical hashing. Snapshot
verification still uses `recipe_identity`, including its existing temporary
encoding allocations. Reports compare by value and do not mutate either recipe.

Only scientific recipe settings participate: referenced script digest and
entrypoint are included; locators, experiment inputs, environments, run history
and output paths are excluded. Relocated identical scripts compare equal even
if their files are unavailable. Comparison verifies saved snapshot integrity,
not current script-file bytes; recreate a `ScriptReference` for a new snapshot
or use replay's content preflight to check the current file. Compare `input_id`
separately when comparing experiments. This API performs no numerical execution
and does not assess the impact of changes on representative image pairs.

## Saved-file structure

| Native experiment field | Meaning |
|:------------------------|:--------|
| `experiment_format_version` | Integer `1`; unknown versions are rejected before decoding settings. |
| `experiment/recipe` | Explicit primitive mapping of every `PIVParameters` field and every recipe option; defaults are stored rather than rediscovered. |
| `experiment/recipe_id` | SHA-256 of the typed recipe settings, embedded mask/background arrays, script content digest and entrypoint. |
| `experiment/script_path` | Optional script locator; not executable code or a serialized function. |
| `experiment/input_files` | Locators, SHA-256 file-byte digests, byte counts, and decoded full-image dimensions. Images are not embedded. |
| `experiment/pairs` | Ordered two-index references into the input-file list. |
| `experiment/input_id` | SHA-256 of the ordered pair sequence's content/dimension descriptors, independent of paths and locator deduplication. |
| `experiment/creation_environment` | Julia/core versions, core source digest, resolved package versions/tree hashes, platform/thread settings, and Project/Manifest text. |
| `experiment/runs` | Run metadata: UUID, identities, Unix times, status/count, output locator/digest, actual environment, and optional error summary. |

The table describes mappings inside a JLD2 payload, not HDF5 dataset paths:
`experiment` is one dictionary dataset. Application-defined Julia types and
functions are not used to persist recipes. Canonical SHA-256 encoding orders
dictionary keys, records scalar types, and records array element type, shape,
and column-major values. Hashes detect changed data/settings; they are not
authentication signatures. Paths are locators/provenance and are excluded
from recipe/input identities. Exact file bytes matter: re-encoding an image
with equal decoded pixels creates a new input identity.

## Processing and input checks

Built-in preprocessing supports background subtraction, intensity cap,
highpass, CLAHE, percentile stretch, inversion, and local-variance normalization.
Backgrounds are embedded snapshots. Operations run in saved order on the full
image, before ROI extraction. Original-size static masks are retained in full.
The four built-in validators are supported; other validator objects are
rejected. Numeric pass thresholds must be finite; the velocity-magnitude
validator permits `max=Inf` for an unbounded upper limit.

Replay checks recipe/input identities, dimensions, masks/ROI/search windows,
referenced script bytes, supported backend/options, and environment compatibility
before opening result output. It protects inputs, scripts, known experiment
paths, and the run-record destination using paths and filesystem same-file
checks. Existing experiment files also cannot be used as result output. Each
image's bytes are checked again around its load; concurrent mutation of input
files is unsupported, so the checks do not provide an atomic filesystem
snapshot. Saving records validates their structure/identities but does not
require images or scripts to be present; replay verifies their actual content.

## Software environment

`allow_environment_change=true` permits an explicit rerun in another
environment. Every run records its actual environment separately from the
record's creation environment; the override option is not a separate persisted
flag. By default compatibility compares Julia/core versions, core source
content, resolved package versions/tree hashes, OS/architecture, and Julia/FFTW
thread counts. Project/Manifest text is retained for inspection, never activated
or instantiated automatically. Local edits to other development dependencies,
provider preferences, device/hardware state, and arbitrary callback external
state are not fully captured. Matching metadata does not prove bitwise
reproducibility on all hardware.

## Custom preprocessing

An optional `ScriptReference` records a custom preprocessing script's content
and named entrypoint. The library never `include`s or `eval`s that script,
resolves its entrypoint, or serializes a closure. Replay requires an explicit
`custom_preprocess` callable from the caller and verifies the script bytes;
the caller must establish that the supplied function corresponds to the
reference. It runs after built-in steps and must return finite values in the
saved precision with unchanged full-image dimensions. Callback state and
external inputs remain the caller's responsibility.

## Progress, failures and saved history

Every replay starts at the first pair and writes an ordinary native result
file with `collect_results=false`. A successful `ExperimentRun` does not retain
results. `completed_pairs` counts result writes that finished before the next
pair, rather than promising durable checkpoints. Processing failures preserve
the native completed prefix and rethrow the original exception. Supplying
`run_record` also writes failed-run metadata after the result writer closes;
a secondary failure saving that metadata is logged while preserving the
original processing exception. A failure to save successful-run metadata can
throw after the result file has been completed. Preflight rejection leaves
both files unchanged. The two files are not an atomic transaction, and saved
run history is not a resume instruction. Use distinct output paths to retain
earlier runs; overwriting an ordinary result file makes an earlier recorded
output digest stale.

`replay_experiment(...; progress=(written,total)->...)` notifies on the calling
task after a pair's native result, requested companions and source labels have
been written. There is no zero/preflight or within-pass notification. The count
is captured before calling user code; a throwing callback leaves a failed run
with that count, including when it throws after the final write. Pending
prefetch drains and output closes before failure-history handling. Callback
selection does not enter recipe identity. The GUI adds cooperative cancellation
at these boundaries without changing version-1 persisted statuses or providing
checkpoint resume.

## Supported workflows

Stereo calibration/dewarping/self-calibration, timestamp persistence, per-pair
delays, dynamic masks, PTV/tracking, GPU execution, automatic custom-script
replay, and resume are outside version 1. A scalar `PhysicalScale` is stored;
`PlanarTransform` export calibration is not part of this processing record yet.
The GUI can bridge its settings to this core schema, but GUI state/workflows
are not automatically embedded.

## Functions and types

```@index
Pages = ["experiments.md"]
```

```@autodocs
Modules = [Hammerhead]
Pages = ["src/experiments.jl", "experiment_comparison.jl"]
Private = false
```
