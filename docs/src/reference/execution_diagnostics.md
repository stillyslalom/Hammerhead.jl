```@meta
CurrentModule = Hammerhead
```

# Planar execution diagnostics

An opt-in companion API for actual planar sweeps and stopping decisions.
See [Inspect actual planar pass execution](../howto/execution_diagnostics.md).
Native result format 1 and existing result/run structs remain unchanged.

`run_piv(...; on_diagnostics=callback)` delivers one
`PIVExecutionDiagnostics`. Sequence/replay callbacks receive `(i, diagnostics)`;
`record_diagnostics=true` additionally requests native persistence. Callbacks
run on the processing task before result delivery/persistence/progress.
Callback exceptions propagate before the current pair is persisted. Previously
completed pairs remain readable; failed numerical executions have no completed
diagnostics. A callback is not a durable-commit notification.

The immutable root contains `execution_id` (a generated UUID), `backend`,
`image_type`, `processing_size` (the processed image dimensions, after ROI),
`core_source_sha256`, optional `pair_index`, optional `association` and the
`passes` tuple. Association is `nothing` for ordinary calls or an immutable
`(recipe_id, input_id)` from verified replay snapshots. Pair indices are
positions in the driver's supplied sequence, not inferred acquisition frame
numbers. Replay's whole-output digest covers companion data. These are recorded
execution associations, not cryptographic authentication or vector history.
`core_source_sha256` describes the on-disk checkout at report generation; a
fresh process and unchanged sources are required to relate it to loaded code.

Each `PassDiagnostics` contains:

| Field | Meaning |
|:--|:--|
| `pass_index` | Contiguous one-based processing-pass position |
| `requested_iterations` | Requested maximum sweeps |
| `executed_iterations` | Sweeps actually completed |
| `requested_tolerance` | Requested nonnegative tolerance in processing pixels; positive infinity is allowed |
| `stop_reason` | `single_sweep`, `iteration_budget` or `tolerance_condition_met` |
| `checks` | Actual tolerance evaluations; not the sweep count |
| `last_check` | Immutable last comparison, or `nothing` if never evaluated |
| `residual` | Immutable final-sweep primary correlation-output summary |

`last_check` records `sweep`, `value_state` (`finite`, `infinite` or `nan`),
`value` (a finite q95 value, otherwise `nothing`), `included_count`,
`finite_count`, `infinite_count`, `excluded_count` and `tolerance_met`.
The criterion is q95 of `max(abs(du), abs(dv))` between successive
post-validation/replacement fields, excluding the grid mask. Existing
`field_change` rules are preserved: nonfinite differences with matching source
NaN patterns are omitted, while a change of NaN pattern contributes infinity.
For example, unchanged `Inf-Inf` differences produce NaN and can be omitted.
Finite/infinite counts describe contributions to this comparison, not valid
measured vectors. With no contributing nodes, the actual value is zero; the
report retains empty support even when the tolerance condition is met.

The final budgeted sweep is never tolerance-checked. A two-sweep budget has
zero checks. For longer exhausted budgets, `last_check.sweep` refers to the
earlier evaluated sweep; it must not be presented as a final convergence test.
Tolerance zero disables evaluation. A tolerance condition met after internal
filling describes solver stopping, not measurement validity or accuracy.

Residual data contains `finite_count`, `nonfinite_count`, `masked_count`,
`mean_magnitude`, `rms_magnitude`, `maximum_magnitude`, `predictor_present`,
`unit="px"` and `value_basis="primary_peak_before_validation"`. Magnitudes
are `hypot(primary_du, primary_dv)` in Float64. Masked nodes are excluded;
nonfinite components or magnitude count as nonfinite. Statistics use only the
finite unmasked magnitudes, with RMS `sqrt(mean(magnitude^2))`; computation
uses a stable online form. No finite samples means all three statistics are
`nothing`. Stored observations are primary-peak values before predictor
addition, substitution and filling; they are not associated with changed
output measurements. No physical conversion is applied.

Native companions use root `execution_diagnostics_format_version=1` and
`execution_diagnostics/<result-entry-name>`. Each mapping contains exact
`result_key` (for example `results/000007`) and `diagnostics`, the detached
primitive root plus `diagnostics_format_version=1`. Scalar named tuples become
primitive mappings, tuples become arrays and symbols become strings. `nothing`
denotes explicit absence. A persistence-requested sequence declares the root
version before processing, even if its first pair fails. Missing per-entry
companions mean not recorded. Result keys bind companion entries, while
per-pair files additionally retain their absolute sequence index.

Readers validate versions, identities, exact keys, support counts, stopping
consistency and numeric states, opening/closing per access and checking the
`ResultFile` stamps. They read only selected companion metadata and do not
deserialize results. Unknown companion versions do not affect ordinary result
readers. Ordinary result-only copies omit companions. No payload arrays or
sweep traces are retained: observations use O(passes) memory per execution;
native indexes still retain O(entries) keys. This is a completed-file API and
adds no atomic write, concurrent-writer or checkpoint guarantees.

Stereo, PTV and ensemble diagnostics are outside this version. Stereo would
require separately labeled camera observations in dewarped pixels; ensemble
would require pooled observations with ignored `max_iterations`, rather than
invented repeated sweeps or individual-pair residuals.

```@autodocs
Modules = [Hammerhead]
Pages = ["execution_diagnostics.jl"]
```
