# Saved planar ensemble experiments

The dedicated version-1 ensemble record stores primitive builtin settings,
ordered file-pair content identities and creation environment. It does not
change the planar sequence or stereo record formats. See
[the workflow guide](../howto/ensemble_experiments.md) for creating and replaying one.

`input_pairs × scheduled_passes` is the contribution budget. Progress counts
joined pair contributions; even its final event precedes peak analysis and
publication. `completed_pools` and `published_results` refer to the single
finished output, not passes. They remain zero for cancelled/failed attempts.
Requested iteration/tolerance settings are saved, but the ensemble driver
executes exactly one sweep per pass and ignores those iteration settings.

The native artifact retains result format 1. A separate
`ensemble_experiment_run_format_version = 1` marker and
`ensemble_experiment_run` primitive envelope bind its one `results/000001`
entry to recipe/input/run identities, the full creation-record snapshot,
actual execution-environment signature, contribution/publication counts,
diagnostic recording policy and raw measurement digest. `ensemble_sources`
stores ordered file labels as `[A₁, B₁, A₂, B₂, …]`. Source IDs are byte-content
identities, not authentication. Any recorded ensemble execution companion
retains its independent schema and measurement binding.

Metadata-only run inspection checks the whole artifact hash, exact entry
mapping, source labels, association and requested companion schema/settings.
It does not inspect raw field dimensions/values; `verify_results=true` does.
The measurement digest includes axes, displacement, quality metrics,
uncertainty, mask/outlier flags and scale. Final parameters and retained-plane
type/shape/presence are independently checked against the recipe. Correlation
plane contents are covered by the artifact hash rather than this field digest.
No calibration, stationarity, uncertainty applicability, source authenticity
or estimator-accuracy claim follows from successful verification.

Failed/cancelled attempts assert no output identity: their recorded destination
may still contain a previous result. Metadata inspection does not read that
destination and raw verification rejects. Current input-byte verification is
optional when inspecting a run, and mandatory for replay.

Replay prepares the complete output in a sibling temporary file and uses
native same-directory replacement without a copy/delete fallback. Processing
and cancellation preserve the previous destination; filesystem failure,
concurrent mutation, process-kill recovery and durability are not guaranteed.
Run-history saving is separate. `EnsembleRunRecordError` exposes accurate
completed/cancelled metadata if that requested save fails; callers can retain
the run and inspect any completed output. An ordinary processing/callback
failure remains the original exception even if saving failed history also
fails. There is no atomic output/history transaction.

```@autodocs
Modules = [Hammerhead]
Pages = ["src/ensemble_experiments.jl"]
Private = false
```
