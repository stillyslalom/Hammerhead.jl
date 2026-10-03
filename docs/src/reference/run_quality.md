```@meta
CurrentModule = Hammerhead
```

# Run-quality reports

Format version 1 is language-neutral TOML containing primitive mappings,
arrays, strings, integers, booleans and finite floating-point values. It reports
stored planar/stereo PIV arrays by default. PTV/tracking results are rejected
by the default format-1 report.
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

This version summarizes result fields only. Its legacy `not_persisted` reason
means that the needed history is absent from those fields; the report does not
inspect optional [measurement-history companions](measurement_history.md) or
execution diagnostics, even when the native file contains them. Use the
companion readers for those observations, or explicitly request format 2 below.

## Opt-in format 2

`include_measurement_history=true` adds `entry_kinds` and `measurement_history`
to the root, with generator
`value_basis="stored_arrays_and_verified_final_sweep_history"`. The loader accepts
all three report versions. The opt-in requires a whole direct native index or the
verified record/run overload; anonymous/converted iterators and array views are
refused. Existing current-array groups retain their definitions. `entry_kinds`
counts planar, stereo, PTV and tracking entries; PTV/tracking contribute no
current-array group. Whole-file provenance counts all entry kinds.

`measurement_history` records `scope="final_pass_final_executed_sweep"`,
`binding="raw_measurement_digest_verified"`, fixed-size `counts`, and `fractions`.
Coverage counters are `planar_entries`, `recorded_entries`, `missing_entries`,
and `unsupported_stereo_entries`/`unsupported_ptv_entries`/
`unsupported_tracking_entries`. `nodes`, `masked` and `unmasked` count only
verified recorded planar packets. Missing packets reduce coverage; corrupt
packets or packets on unsupported result kinds cause errors. Present packets
in associated reports must match recipe/input IDs and absolute pair index;
`association=nothing` is refused separately from an absent packet.

Event counters are `first_rejected`, `pre_substitution_flagged`,
`accepted_alternative`, `fill_attempted`, `fill_assigned` and `primary_restored`.
First-rejection counters have fixed buckets `rejection_nonfinite_primary`,
`rejection_implicit_uod`, `rejection_implicit_peak_ratio`,
`rejection_configured_builtin`, `rejection_configured_custom`, and
`rejection_unclassified`. Only recognized generated version-1 stage labels/kinds
are classified; unfamiliar labels are never guessed. These count first observed
flag transitions, not every failed predicate. A custom validator can clear a
flag later, so first-rejection and pre-substitution counts need not agree.

Final-origin counters are `origin_unavailable`, `origin_primary`,
`origin_alternative`, `origin_fill` and `origin_custom_unclassified`. Their sum
is covered `unmasked`. An internal median assignment followed by restoration
increases `fill_assigned` and `primary_restored` while final origin is primary or
unavailable. Assignment does not certify a finite or valid output.

Each event, rejection-bucket and final-origin fraction uses covered `unmasked`
as denominator and is named `<counter>_fraction`. `recorded_entry_fraction` uses
`recorded_entries / planar_entries` (entry coverage); `masked_fraction` uses
covered `masked / nodes`. The same available/value/zero-denominator convention
applies as in format 1. Missing and unsupported entry nodes are excluded from
event denominators, never treated as zero-event observations.

Format 2 replaces the three generic unavailable event-history entries with this
scoped section, adds `earlier_pass_and_sweep_history: not_recorded`, and labels
`uncertainty_measurement_association: applicability_not_established`. Accuracy,
coverage, peak locking and sensitivity remain unevaluated. Finite stored
uncertainty never establishes applicability, including for primary-origin nodes.
Execution tolerance counts/outcomes are not quality metrics in formats 1 or 2.

## Opt-in format 3

`include_execution_diagnostics=true` emits version 3, adding `entry_kinds` and
`execution_diagnostics`. It requires the same direct whole-file native mapping
as format 2. `include_measurement_history=true` can additionally retain the
unchanged history section. The generator's `value_basis` is
`stored_arrays_and_recorded_execution`, or
`stored_arrays_recorded_execution_and_verified_final_sweep_history` when both
are requested. `weighting="field_nodes_and_execution_observations"` separates
the existing node-weighted field counts from execution entry/pass observations.
Defaults continue to emit formats 1 and 2. Older report loaders reject version 3;
the native result format and companion versions do not change.

The execution section has fixed fields:

| Field | Value or meaning |
|:--|:--|
| `scope` | `recorded_passes_and_final_pass_primary_support` |
| `aggregation_basis` | `entry_and_pass_counts` |
| `verification_time` | `report_generation`, never fresh verification on TOML loading |
| `last_check_scope` | `last_recorded_check_per_pass` |
| `primary_support_scope` | `final_pass_before_validation` |
| `unsupported_entries` | PTV/tracking native-entry counts |
| `groups` | Fixed `planar`, `cam1`, `cam2` groups, including empty groups |

Each execution group has `coordinate_basis`, `residual_unit="px"`, `binding`,
`counts`, and `fractions`. Planar coordinates are
`planar_processing_pixels_x_columns_y_rows` with `binding="entry_key_only"`.
Camera coordinates are `dewarped_pixels_x_columns_y_rows` with
`binding="raw_measurement_fields_checked_at_report_generation"`. Camera binding
independently checks reconstructed/camera dimensions, world-grid mapping and
captured measurement fields against the already loaded raw result. Parameters
and correlation planes remain excluded. Calibration, source bytes,
synchronization and scientific accuracy are not verified. A report loader opens
no recorded locators and preserves these statements as past generation checks.

All counters are nonnegative `Int` values, with checked additions and partitions:

| Counter family | Scope |
|:--|:--|
| `eligible_entries`, `recorded_entries`, `missing_entries` | Planar native entries for `planar`; stereo native entries independently for each camera. Eligible = recorded + missing. |
| `passes` | All recorded passes; no unique-execution-ID deduplication is implied. |
| `requested_sweeps`, `executed_sweeps`, `tolerance_checks` | Actual recorded requested budgets, sweeps and checks across passes. |
| `stop_single_sweep`, `stop_iteration_budget`, `stop_tolerance_condition_met` | A partition of recorded passes. A tolerance condition is not measurement acceptance. |
| `last_checks_present`, `last_checks_absent` | A partition of passes. A budgeted final sweep can be unchecked. |
| `last_check_finite`, `last_check_nan`, `last_check_infinite` | A partition of present last checks; not a histogram of every check in a pass. |
| `last_check_empty` | Present last checks with zero included support; the finite zero-valued comparison may satisfy tolerance. |
| `last_check_included`, `last_check_finite_support`, `last_check_infinite_support`, `last_check_excluded` | Sum support over each pass's last recorded check; included = finite + infinite. |
| `final_primary_nodes`, `final_primary_masked`, `final_primary_unmasked` | Final-pass primary support only; nodes = masked + unmasked. |
| `final_primary_finite`, `final_primary_nonfinite` | A partition of final primary unmasked nodes using finite residual magnitude, before validation/substitution/filling. |

`recorded_entry_fraction` uses recorded/eligible entries.
`final_primary_finite_fraction` uses final finite/unmasked primary support.
Both retain the existing explicit denominator/availability convention. Missing
entries contribute neither support nor zero events. Camera groups remain
separate from reconstructed `groups.stereo` field quality and from each other.
The two camera groups must agree on associated entry coverage, requested pass
budgets and final grid-node counts; their residual states and actual stopping
behavior can differ.

No residual means/RMS/max values, world-norm conversions or reconstructed 3C
residuals are aggregated. Processing geometry can vary across entries; read
individual packets for their amplitude statistics. The additional unavailable
reason is `pooled_execution_residual_amplitudes: not_aggregated`. Without history,
the existing unavailable event-history reasons remain; with history, the scoped
format-2 changes still apply. Finite primary support never establishes UQ
applicability or final-vector measurement origin.

Both execution companion root markers are checked even for empty files.
Unknown versions, missing markers for present groups, wrong result-key linkage,
crossed planar/stereo companions and companions attached to PTV/tracking are
refused. A native PIV result does not identify whether an unrecorded driver was
an ensemble, so workflow categories are not inferred from missing metadata.
Associated planar execution packets must match the selected recipe/input IDs
and absolute pair index; generic files remain unassociated. Stereo packets have
no supported experiment recipe association and are refused in that associated
mode. Each raw payload is loaded once, and only fixed-size counters survive;
indexes and protected locators retain their existing documented memory costs.

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
