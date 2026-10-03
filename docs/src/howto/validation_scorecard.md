# Regenerate the validation scorecard

Compare a processing recipe with synthetic displacement truth, inspect the
experimental workflows and record local CPU timings. From a checkout with its
dependencies installed, run:

```bash
julia --project=. -t 4 bench/validation_scorecard.jl
```

The command writes `scorecard.toml` and `scorecard.md` under the gitignored
`bench/profile-output/validation-scorecard/`. The Markdown table is a summary;
the TOML file contains the full pass schedule (every `PIVParameters` field),
driver settings, data identities, coordinate/reference conventions, validity
rules, environment, and individual timing samples. Keep reports inside the
checkout under `bench/profile-output`; choose another destination when report
names already belong to unrelated files.

For component uncertainty coverage and signed normalized errors across seeded
conditions, run the separate [synthetic uncertainty scorecard](validation_uncertainty.md).
It uses a controlled final-output recipe and reports selection effects, zero
uncertainty and arithmetic failures explicitly. Use this baseline for
displacement errors and processing time, and the uncertainty scorecard for
component coverage.

## Read the baseline populations

The default evaluates deterministic 128 × 128 px translation and linear-shear
particle pairs, an intentionally empty particle scene, the committed Challenge
A pair cropped to rows 257:768 and columns 385:896, and a coarse stereo
reconstruction from the committed 4E slice. Synthetic identities include the
specified generator, seed, particle/noise/dropout settings and SHA256 of both
rendered images. A fixed SplitMix64 counter stream avoids changes to Julia's
default random generator. Rendering/software versions can still affect exact
pixels; hashes identify the actual inputs used. Experimental files, fixture
readmes, package source, and the scorecard script are also hashed. The active
Project/Manifest contents and Julia/CPU/thread settings accompany each report.
Run in a fresh Julia process with the checkout unchanged throughout. Source
hashes are captured before evaluation and checked afterward; an unstable
checkout is reported explicitly and requires regenerating the final evidence.

Synthetic particles move with one forward-Euler step. For the selected linear
flow, `u = du + shear*(y_launch-center)` and `v = dv`; vector positions refer
to trajectory midpoints, so the exact reference is evaluated at
`y_launch = y_vector - dv/2`. The prescribed motion supplies the reference.
The report records signed bias and population RMS of measured-minus-reference
displacement. RMS includes bias.

The A and 4E rows report validity/yield and timing as **real-data smoke checks**.
Bias and RMS error require a numerical motion reference and are unavailable for
these rows. The 4E recipe uses full calibration images, a 0.5 mm dewarp grid,
and the fitted cameras directly; it exercises a coarse reconstruction workflow.
Use the full real-data tutorial to explore that recording in more detail.
Known-motion experiments need a recording with an independently measured motion
reference; their report entries remain unavailable until those inputs are supplied.

Error metrics use unmasked, unflagged nodes with all components finite.
Flags remain part of this selection after replacement. Valid yield divides
that count by all unmasked nodes, including rejected/nonfinite nodes. The report
records each count and selection rule. Check the empty-scene status:
`no_valid_vectors` is the expected outcome; an unexpected-valid-vectors status
identifies admitted output that needs investigation.
The initial baseline exposed this latter failure: the empty pair produced an
apparently valid field with artificial displacement. After adding exact-constant
input and non-informative-plane guards, the expanded Windows CPU rerun on
2026-10-02 reports `no_valid_vectors` (0/225) for this pair. Translation/shear
error metrics and A/4E smoke yields remain unchanged. This fixes the wholly empty
pair. [Original-stencil contrast checks](../explanation/noninformative_windows.md)
also guard originally flat patches under nonzero deformation; their separate
regressions describe the scientific convention and weak-contrast behavior.
Use the generated status to assess later changes.

The expanded source-stencil rerun on 2026-10-02 retained the translation/shear
errors and A/4E smoke yields. The sparse synthetic case changed from 205/225
to 195/225 accepted nodes: ten previously accepted nodes lacked contrast in
at least one sampled original-source stencil union. Predictor and validation
effects also changed some retained vectors. Its reported RMS changed from
0.037549/0.020701 px to 0.032888/0.020575 px (u/v). Compare these errors together
with the changed accepted populations.

## Compare processing time and expand the conditions

```bash
julia --project=. -t 4 bench/validation_scorecard.jl --expanded --samples=5
```

The expanded mode adds sparse/dense seeding, larger particle diameter, uniform
additive noise, stronger linear shear, independent frame-B particle dropout,
and a 32 px final-window comparison. Compare each condition with its recorded
recipe and accepted-node population.

Each exact recipe runs once for warmup, then the requested number of measured
calls. Processing uses `backend = :cpu`, `threaded = false`; the report also
records Julia, BLAS, and FFTW thread counts. Fixture loading, synthetic generation,
calibration, hashing, warmup, and report writing are excluded from processing
time; preparation is reported separately. Stereo call time includes dewarping
and reconstruction. Garbage collection runs before each sample, outside its
timer; GC during the call is included and reported. Compare individual samples
and median/minimum/maximum on the same otherwise idle machine, Julia version,
environment, and thread settings.

`julia_allocated_bytes_samples` measures cumulative Julia allocations during
each call. Native-library allocations can fall outside this counter. Measure
peak host/device memory separately when sizing a workload. Use the per-sample
timings to compare recipes under matched local conditions.
