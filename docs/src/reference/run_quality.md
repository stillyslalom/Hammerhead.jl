```@meta
CurrentModule = Hammerhead
```

# Run-quality reports

Format version 1 is language-neutral TOML containing primitive mappings,
arrays, strings, integers, booleans and finite floating-point values. It reports
stored planar/stereo PIV arrays. PTV/tracking results are rejected.
See [Save a run-quality report](../howto/run_quality.md) for usage.

The root fields are `quality_report_format_version`, `generated_at_unix_s`,
`generator`, `provenance`, `protected_locators`, `groups`, and `unavailable`.
Generation time is Unix seconds and varies between otherwise identical reports.
The generator records Julia/package versions, `core_source_sha256`,
`value_basis="stored_arrays"` and `weighting="node_weighted"`. This reporting
source hash is distinct from the recorded run environment identity.

Generic sequences have `provenance.association="unassociated"`. A known
`ResultFile` adds absolute `source_path`, `source_sha256`, `source_index_entries`
and `source_selection` (`whole_file` or `provided_array` for array wrappers).
An associated report uses `recorded_output_verified` and adds `recipe_id`,
`input_id`, `run_id`, `run_environment_id`, and `completed_pairs`. The selected
run must be completed, its metadata must agree with the record, and its current
native output must match the recorded byte digest and result count. This is
recorded provenance, not cryptographic authentication or a guarantee about
individual measured vectors. Loading a saved report does not reverify files.

Each present `groups.planar` or `groups.stereo` contains `counts` and
`fractions`. Empty input has no groups. Nodes are counted across all entries;
the group entry count is not a fraction denominator. Masks exclude nodes from
all counters except `nodes` and `masked`. A finite output has every velocity
component finite; current flags are counted independently of that condition.

| Fraction | Numerator count | Denominator count |
|:--|:--|:--|
| `masked_fraction` | `masked` | `nodes` |
| `current_outlier_flag_fraction` | `outlier_flagged_unmasked` | `unmasked` |
| `finite_output_fraction` | `finite_output_unmasked` | `unmasked` |
| `unflagged_finite_output_fraction` | `unflagged_finite_output_unmasked` | `unmasked` |
| `stored_uq_all_numerically_available_fraction` | `uq_all_available_unmasked` | `unmasked` |
| `stored_uq_on_unflagged_finite_output_fraction` | `unflagged_finite_output_with_uq_available` | `unflagged_finite_output_unmasked` |
| `stored_uq_unavailable_when_requested_fraction` | `uq_all_unavailable_when_requested_unmasked` | `uncertainty_requested_unmasked_nodes` |
| `stored_uq_C_numerically_available_fraction` | `uq_C_available_unmasked` | `unmasked` |

`C` is `u`/`v`, plus `w` for stereo. Numerical UQ availability means finite and
nonnegative in the stored array. Per-component counters partition unmasked
nodes into `uq_C_available_unmasked`, `uq_C_negative_finite_unmasked` and
`uq_C_nonfinite_unmasked`; joint availability requires every component
available. `uncertainty_requested_entries` and
`uncertainty_requested_unmasked_nodes` use the saved final parameters'
`uncertainty` setting. The requested-unavailable numerator counts unmasked
nodes in those entries where at least one stored component is unavailable.
Availability in other fractions reflects the arrays even if the setting is
disabled. No values are rescaled or averaged.

Every fraction stores `numerator`, `denominator`, `denominator_count`,
`unit="1"`, and boolean `available`. Positive denominators additionally store
`value=numerator/denominator`; zero denominators omit `value`. These are
node-weighted fractions, not averages of per-frame percentages. Counter sums
are checked for overflow and persisted fractions must agree with their counts.

`flagged_finite_output_unmasked` counts finite outputs whose current flag is
set. It is **not** a replacement count: filling can retain flags and
alternative-peak substitution can clear them. The `unavailable` mapping
explicitly labels `rejection_events`, `replacement_history`,
`alternative_peak_history`, and `uncertainty_measurement_association` as
`not_persisted`; `accuracy`, `uncertainty_coverage` and `peak_locking` as
`not_evaluated`; and `recipe_sensitivity` as `not_compared`. Each has
`available=false`. Current finite/nonnegative UQ counts establish neither
measurement association nor calibrated uncertainty coverage.

Report snapshots retain no payload arrays and dictionary access returns a
copy. Known protected locators are authoritative when saving, with normalized
path and filesystem same-file checks. Anonymous iterators need explicit
`protected_paths` for dependencies the report cannot discover. Schema version,
types, identities, counts, denominators and unavailable reasons are validated
before destination opening; this does not promise atomic writes, concurrent
writer safety, resumption or cryptographic authenticity.

```@autodocs
Modules = [Hammerhead]
Pages = ["run_quality.jl"]
```
