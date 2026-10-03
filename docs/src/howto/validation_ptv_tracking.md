# Evaluate annotated PTV and tracking

Run the controlled, identity-aware scorecard from a fresh Julia process with
frozen sources:

```bash
julia --project=. --threads=1 bench/validation_ptv_tracking.jl --output=bench/profile-output/ptv-tracking-run1
```

Choose a new output directory under `bench/profile-output` when working inside
the checkout. The command creates it exclusively and protects consumed inputs
from output aliases. If publication leaves a partial directory, inspect it and
choose a fresh destination for the retry.

The default evaluates annotated synthetic clips with production detector and
matching settings. PIV predictors are calculated from the images; the prescribed
positions are used for scoring.

## Fixed evidence matrix

The default matrix contains two deterministic placements, seeds 7321 and 7322,
and four eight-frame clips per placement. Each image is 128 × 128 pixels.

| Clip | Prescribed condition |
|---|---|
| `clean` | 25 jittered-grid targets moving by (0.8, 0.4) px per frame |
| `one_frame_gap` | Every fourth target deliberately invisible in frame 3 |
| `two_frame_gap` | The same targets deliberately invisible in frames 3 and 4 |
| `encounter_noise_clutter` | Close approaching targets, faint targets, additive centered uniform noise, a nuisance peak and a boundary exit |

IDs and positions are defined before rendering. The complete ledger includes
every identity in every selected frame, including invisible and outside-image
samples. The renderer is `SyntheticData.generate_gaussian_particle!`: diameter
3 px means 4σ, with σ = 0.75 px. Its square bounding box uses
`round(Int, center ± ceil(3σ))`, clipped to the image, with nearest-even rounding.
Pixel-center intensities use the Gaussian values directly.
The mixed stress condition has unclipped noise in [−0.01, 0.01]; the other clips
are noiseless. Read this clip as the combined response to encounter, noise,
clutter and boundary effects.

Each clip uses the production detector, `run_ptv` on every adjacent pair and
`track_particles` with `max_gap = 0, 1, 2` and `min_track_length = 2`.
The full command uses default `PTVParameters`, `predictor = :piv`, and the
complete default `[64, 32]` PIV pass schedule. Inputs and processing use Float64,
CPU, full-frame raw pixels, and ordinal frame intervals; mask, ROI, preprocessing
and scale are disabled. Every
parameter and the actual process thread count are recorded. Internal PIV uses
its production default, `threaded = Threads.nthreads() > 1`; the recipe and
environment record that Boolean independently of whether the chosen predictor
requires PIV. Use that Boolean to identify the driver's threading policy.

The budget is 64 standalone detector calls, 56 two-frame PTV calls and 24
tracking calls. Including repeated detection inside these APIs, that is 368
detector invocations and 224 matcher transitions. There are at most 80 full
two-pass PIV predictions: 56 pair predictions and 24 first tracking predictions.
Later tracking predictions use the production binned/smoothed field and
constant-velocity heads. The artifacts cover correspondence and identity
metrics. The default generator processes one clip at a time; its original and detached
eight-frame image copies occupy about 2 MiB, excluding workspaces and artifacts.
Report/CSV metadata grows with the number of observations.

## Read the recorded outcomes

The frozen Windows CPU run in
`bench/profile-output/validation-ptv-tracking-refrozen/` completed the fixed
matrix with stable source/environment identities. The focused regression suite
passed 2,197 checks. An independent persisted-artifact audit passed 36,571 checks,
including exact scientific report equality and all 144 CSVs byte-identical to
the preceding run. Both historical runs used one Julia thread, but their
recorded PIV-threading Boolean was incorrectly hardcoded to `true`. Use the
corrected run below for threading provenance.
The production default is `false` with one thread and `true` with multiple
threads. Separate one- and four-thread regression processes check the recorded
values, actual CPU correlator fan-out and production PIV/PTV equivalence.

The corrected one-thread run in
`bench/profile-output/validation-ptv-tracking-threading-corrected/` completes
all eight clips with stable sources and records both threading Booleans as
`false`. Its audit verifies 51 source hashes and passes 36,600 checks: scientific
report fields remain identical after excluding environment, generation time and
the corrected recipe Boolean, and all 144 scientific CSVs match prior bytes.
The focused regression suite passes 2,204 parent assertions plus 22 assertions
in separate one- and four-thread children. Use this fresh report for threading
provenance; retain historical reports with their original values.

For each seed, all six scheduled one-frame gaps were recovered with
`max_gap ≥ 1`, and all six two-frame gaps with `max_gap = 2`. With a smaller
allowed gap, those events remained unrecovered and the six affected truth
identities were split across returned tracks. Clean clips had complete
localization/correspondence in this bounded condition.

| Mixed stress seed | Detection TP / predictions / visible targets | Raw pair correct / wrong / unmapped / ambiguous | Accepted precision | Full accepted-pair recall | Ambiguous visible samples resetting track identity memory |
|---|---:|---:|---:|---:|---:|
| 7321 | 216 / 224 / 218 | 184 / 0 / 7 / 3 | 170/170 | 170/190 | 4 |
| 7322 | 216 / 224 / 218 | 184 / 0 / 7 / 3 | 168/168 | 168/190 | 4 |

Read accepted precision alongside recall and identity retention. In these stress
clips, `max_gap = 0` retained 198/218 and 196/218 confirmed visible observations;
allowing gaps retained 195/218 and 194/218. Longer-gap policies recovered correct
same-ID jumps across visible misses, with lower full consecutive-visible edge
recall in this subset. Confirmed same-track identity changes were zero;
ambiguous observations retain their separate population. Compare gap policies
using these tradeoffs for each clip.

## Read populations before fractions

Per-frame localization uses an independently implemented one-to-one assignment:
maximize cardinality within 0.75 px, then minimize total squared distance.
Tiny exhaustive partial-injection tests verify this objective independently of
the Hungarian implementation and the production greedy matcher.

Detection TP is this operational localized-object count. FP includes all
unassigned detections, including nuisance peaks; FN includes every unlocalized
**visible target**. Deliberately invisible samples have a separate population.
All faint visible targets remain in the detection denominator.

An association is conservatively ambiguous when it has multiple in-gate target
IDs, competing detections, or an in-gate annotated nuisance contributor. A
deterministic tie-break selects an operational assignment; a merged intensity
can still have several candidate identities. Candidate IDs, the operational
assignment and confirmed truth ID are recorded separately.

Every reported correspondence/trajectory edge enters one of four populations:
confirmed correct, confirmed wrong, unmapped or ambiguous. All-prediction
precision is reported as a strict lower bound `correct / N` and a conservative
upper bound `(correct + ambiguous) / N`. Unmapped endpoints remain in `N`.
These identification bounds retain ambiguity as potentially correct. Their
width reflects the unresolved identity population.

For two-frame PTV, raw matches and non-outlier accepted matches are scored
separately. Full recall divides confirmed correct pairs by all same-ID visible
truth pairs. A separately labelled conditional recall uses truth pairs localized
at both endpoints by the operational assignment; ambiguity remains explicit.
Displacement bias/RMS uses confirmed correct pairs; all-match precision/recall
describes the wider correspondence population. Zero denominators are unavailable.

Tracking metrics describe the returned tracks, whose minimum-length filter
discards singletons. Detection-to-returned-observation loss therefore combines
linking and minimum-length retention. These metrics score returned observations
and edges.

| Metric | Exact convention |
|---|---|
| Same-track identity changes | Consecutive returned observations with distinct confirmed truth IDs |
| Target-centric output-ID changes | Different returned trajectory IDs at successive identifiable observations of one truth ID; memory persists across misses/absence and resets at ambiguous visible samples |
| Fragmentation | Extra distinct confirmed returned trajectory IDs per truth ID, `max(0, K − 1)` |
| Restart episodes | Tracked → untracked → tracked across visible truth samples; ambiguity resets episode memory |
| Full edge recall | Recovered exact consecutive-visible same-ID truth edges divided by all such truth edges |
| Gap recovery | One exact consecutive returned edge connects the correct visible bracketing endpoints of a declared invisible interval |

Read these identifiable-subset counts alongside ambiguous samples/transitions
and full populations. The table defines this scorecard's tracking conventions.

Gap recovery retains every eligible declared absence event, grouped by missing
frame count and by scheduled/annotated/mixed absence. Full recall, both-localized
endpoint recall, and the subgroup with at least two prior visible truth samples
are separate. Actual retained two-observation prefixes have their own count;
the full denominator includes every eligible event, including gaps longer than
the configured `max_gap`.

Bridge precision counts **every** returned frame jump larger than one, including
wrong, unmapped and ambiguous endpoints. A correct same-ID bridge across a
missed **visible** intermediate detection/retained observation is separated from
declared invisible-gap recovery. The bridge is one returned edge; intervening
consecutive-visible truth edges remain separate recall opportunities. Trajectories
contain observed positions and explicit frame gaps.

Pooled fractions use summed counts. Correct-pair error summaries remain per
clip. Interpret the two placements together with their shared-frame and
shared-track dependence.

## Inspect artifacts

`ptv_tracking.toml` contains complete recipes, input/annotation/source/environment
identities, per-clip and pooled counts, ambiguity bounds, per-identity summaries,
and exact gap events. `ptv_tracking.md` shows correspondence, tracking and gap
tables alongside their denominators.

Per-clip files include a complete truth CSV, detector/track association CSVs and
classified track-edge CSVs. Candidate-list cells contain a reversible TOML
assignment such as `ids = ["a", "b"]`; CSV quoting preserves punctuation,
Unicode, quotes and newlines in IDs. These are bench annotation/scoring schemas.

Ordinary pair and track CSVs are written through `export_table` with its existing
language-neutral schema. `trajectory_id` indexes a **returned** trajectory;
the association table connects it to confirmed or ambiguous truth IDs. Track
CSVs retain observed frame indices, explicit gaps, validity and units.

## Supply an annotated clip

The importer is available as a bench API and CLI route:

```bash
julia --project=. --threads=1 bench/validation_ptv_tracking.jl --manifest=/local/path/clip.toml --output=/local/path/new-scorecard
```

Its exact version is `hammerhead-annotated-particle-clip-1`. A manifest contains
`clip_id`, `annotation_kind` (`independent_synthetic` or `manual_real`), nonempty
`source_uri`, `citation`, `license`, `coordinate_convention =
"one_based_pixel_centers_x_columns_y_rows"`, `length_unit = "px"`, `image_shape`,
an ordered `frames` array and a `truth` array.

Each frame contains exactly `ordinal`, `locator`, `bytes`, `sha256`,
`decoded_sha256`, and `decoded_processing_type = "Float64"`. Ordinals are 1…N.
File SHA256 identifies source bytes; decoded SHA256 uses the scorecard's Float64
little-endian column-major processing-pixel encoding. Decoded images must be
finite and share the declared shape.

Each truth row contains `id` (nonempty string), `frame`, finite `x`/`y`, Boolean
`visible`, `visibility_reason` (`visible`, `scheduled_absence`, `outside_image`
or `annotated_absence`), and `role` (`target` or `nuisance`). Optional `intensity`
is a finite nonnegative value or the literal string `"unknown"`. Omitting it
normalizes to `"unknown"`. Every identity needs an explicit row/status and finite
position for every selected frame, including invisible samples. Visible positions
must be inside the image, and outside-image reasons must agree with positions.
Each ID keeps the same role.

Relative image locators resolve beneath the manifest directory, including
resolved-parent checks. Foreign/absolute or escaping locators require explicit
receiving-host relocation through the bench API:

```julia
include("bench/validation_ptv_tracking.jl")
using .ValidationPTVTracking
clip = ValidationPTVTracking.load_clip("/local/path/clip.toml";
    relocation = Dict(1 => "/local/path/relocated-first-frame.png"))
bundle = ValidationPTVTracking.run_study(; clips=[clip])
ValidationPTVTracking.write_report("/local/path/new-scorecard", bundle;
    protected_paths=clip.protected_paths)
```

Original locators remain provenance; relocation must match both file and
processing-pixel identities. Supply the supported version, complete unique rows
and compatible images with matching file identities. Choose a fresh output
destination. Input and manifest identities are rechecked after processing and
before publication, together with source/environment stability. Review the
annotation's accuracy and provenance when selecting a clip.

For sparse ID/position annotations, see the [VSJ301 study](validation_vsj301.md).
It evaluates a fixed independent synthetic clip and records unknown visibility
and alternative coordinate origins. Use the complete manifest above for clips
with a reviewed visibility status in every selected frame.
