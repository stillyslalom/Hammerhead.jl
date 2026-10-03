# Reports for pooled results

[`quality_report`](@ref) emits strict version 4 only when
`include_ensemble_execution_diagnostics=true`. Generation requires a direct
whole-file native `ResultFile`. The existing history and ordinary execution
flags may add their unchanged sections. Versions 1/2/3 and their validation
semantics remain supported; neither a present root marker nor a bare
`PIVResult` silently opts into format 4.
The supplied `ResultFile.entry_keys` must exactly match the independently sorted
native result keys. Preflight refuses omissions, duplicates, reordering and
malformed results roots, then captures a detached O(entries) key snapshot before
creating provenance or counts. No raw payload is loaded by a valid mapping check.

The root adds `ensemble_execution_diagnostics` and `entry_kinds` to the standard
report fields. Optional `measurement_history` and `execution_diagnostics`
sections have their existing schemas and denominators. `provenance.association`
is `unassociated`: this direct-file format does not attach a saved recipe or run.
The separate [associated ensemble format](ensemble_experiment_quality.md)
verifies a completed `EnsembleExperimentRecord` run and emits version 5.
The generator's `value_basis` appends
`_and_verified_recorded_ensemble_execution` to the corresponding existing
basis; weighting is `field_nodes_and_execution_observations`.

## Classification and verification

The ensemble section's `classification` contains four nonnegative integer counts:

| Field | Meaning |
|:--|:--|
| `planar_entries_examined` | All raw PIV entries in the native index |
| `recorded_ensemble_entries` | Present, structurally valid, raw-bound ensemble packets |
| `recorded_planar_iteration_entries` | Entries with ordinary planar iteration packets |
| `entries_without_execution_metadata` | Entries carrying neither execution family |

The last three partition the first. Absence of execution metadata does not imply
a missing expected ensemble. `unsupported_entries` separately counts stereo,
PTV and tracking entries. Ensemble and ordinary/history/stereo companions on
one entry are refused; separate entries can coexist. Empty files still require
valid companion markers. All four inspected families must have native group
roots and every child key must map to an indexed result; orphan metadata is
refused. Selected sibling entries must carry matching explicit `result_key`
linkage and pass their family's schema/integrity checks.

The section records these fixed interpretation fields:

| Field | Value |
|:--|:--|
| `scope` | `recorded_ensemble_pools_not_expected_workflow_population` |
| `aggregation_basis` | `all_pass_window_pair_observations_and_final_pass_node_observations` |
| `coordinate_basis` | `planar_processing_pixels_x_columns_y_rows` |
| `residual_unit`, `uncertainty_unit` | `px` |
| `binding` | `raw_measurement_fields_and_geometry_checked_at_report_generation` |
| `verification_time` | `report_generation` |
| `source_support_convention` | `original_sampled_4x4_stencil_union_not_nonlocal_bspline_support` |
| `primary_support_basis` | `final_pass_primary_peak_before_predictor_addition_and_validation` |
| `uncertainty_basis` | `final_pass_pooled_statistics_before_validation_and_availability_cleanup` |

The measurement digest covers raw axes, fields, peak/UQ metrics, flags and scale.
Independent geometry validation checks array shapes and final-pass axes/mask
support. Parameters, retained planes and source bytes are excluded. Loading a
saved report validates only the saved report, without reopening its source or
promoting generation-time verification to a fresh check.

## Count scopes

`counts` contains fixed scalar observations rather than per-entry packets:

| Fields | Scope |
|:--|:--|
| `pair_observations` | Sum of contributing pair counts once per recorded pool |
| `passes`, `executed_pooling_sweeps` | Equal totals: one pooled sweep per scheduled pass |
| `ignored_requested_iterations` | Sum of requested pass budgets, not executed work |
| `passes_with_positive_ignored_tolerance`, `tolerance_checks` | Positive ignored tolerance requests; checks are always zero |
| `all_pass_pair_observations` | Pair count repeated for each pass |
| `all_pass_window_opportunities`, `all_pass_masked_window_pairs`, `all_pass_source_gated_window_pairs`, `all_pass_accumulated_window_pairs` | Opportunity partition across all passes |
| `all_pass_finite_zero_planes`, `all_pass_finite_flat_nonzero_planes`, `all_pass_finite_nonflat_planes`, `all_pass_nonfinite_planes` | Partition of accumulated numerical planes |
| `all_pass_no_predictor_pairs`, `all_pass_evaluated_pairs`, `all_pass_disabled_nonfinite_source_pairs` | Partition of all-pass pair/source-support observations |
| `final_nodes`, `final_masked`, `final_unmasked` | Final grid once per recorded pool |
| `final_zero_finite_nonzero_count`, `final_some_finite_nonzero_count`, `final_all_finite_nonzero_count` | Partition of final unmasked nodes by contributor count |
| `final_nodes_with_nonfinite_plane` | Final unmasked nodes with at least one nonfinite contribution |
| `final_primary_finite`, `final_primary_nonfinite` | Partition of final unmasked primary peaks before predictor addition/validation |
| `final_uq_requested_entries`, `final_uq_evaluated_entries`, `final_uq_evaluated_unmasked_nodes` | Requested/evaluated final pools and evaluated node denominator |
| `final_uq_admitted_window_pair_updates`, `final_uq_source_gated_window_pairs` | Update/skip observations in evaluated final pooled UQ |
| `final_uq_C_finite_nonnegative_count`, `final_uq_C_finite_negative_count`, `final_uq_C_nonfinite_count` | Per-component partition of evaluated final unmasked nodes, `C=u/v` |

Finite nonzero contributors include flat nonzero planes. Counts do not establish
independent sample size, stationarity, valid peaks, or estimator applicability.
No residual magnitudes, contributor extrema or UQ amplitudes are averaged across
grids. Unevaluated UQ pools do not enter evaluated component denominators.
Checked integer arithmetic rejects overflow, and validation enforces count
partitions, support subsets and fully covered mask/grid agreement.

`fractions` uses the ordinary dimensionless fraction structure (`numerator`,
`denominator`, `denominator_count`, `available`, `unit="1"`, and `value` only for
positive denominators):

| Fraction | Denominator |
|:--|:--|
| `recorded_ensemble_fraction_of_planar_entries` | All examined planar entries |
| `accumulated_window_pair_fraction`, `source_gated_window_pair_fraction` | All-pass full-grid window/pair opportunities, including masks |
| `final_primary_finite_fraction`, `final_zero_finite_nonzero_contributor_fraction` | Final unmasked nodes in recorded ensembles |
| `final_uq_u_numeric_availability`, `final_uq_v_numeric_availability` | Final unmasked nodes in evaluated pooled UQ |

`unavailable` retains the existing unsupported scientific diagnostics and adds
`ensemble_independent_sample_size`, `ensemble_common_displacement_assumption`
and `ensemble_per_node_measurement_history`. Pooled residual amplitudes remain
explicitly `not_aggregated`. The
[how-to](../howto/ensemble_quality_reports.md) describes generation and GUI use.
