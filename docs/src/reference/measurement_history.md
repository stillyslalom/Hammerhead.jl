# Recorded vector history

Native result format 1 and the existing result structs remain unchanged. The
optional `measurement_history_format_version=1` declares a sibling group
`measurement_history/<entry>` whose envelope contains `result_key`, `history`,
and `history_sha256`. Each envelope names the exact `results/<entry>` key.
Missing companions mean not recorded, never inferred history.

The `history` primitive schema contains:

- Context: `history_format_version`, independent `history_id` UUID, `backend`,
  `image_type`, `processing_size`, current on-disk `core_source_sha256`, optional
  `pair_index` and verified recipe/input `association`.
- Scope: final `pass_index`, actual final `sweep_index`, `n_peaks`,
  `coordinate_basis="original_image_pixels"`, `displacement_unit="px"`, and
  raw `x`/`y` axes. Matrix rows correspond to `y`, columns to `x`.
- Primary measurements: `primary_u`, `primary_v`, `primary_residual_u`,
  `primary_residual_v`, and `mask`. Masked residuals are unavailable (`NaN`).
- Validation and events: `first_rejection_stage`, ordered `rejection_stages`,
  `pre_substitution_outliers`, `accepted_peak_rank`, `fill_attempted`,
  `fill_assigned`, `primary_restored`, `final_origin`, `origin_codes`,
  `final_outliers`, and `custom_validators_present`.
- Stored uncertainty: per-component `uncertainty_u_status` and
  `uncertainty_v_status`, `uncertainty_status_codes`, and explicit
  `uncertainty_basis` limitations.
- Binding: `measurement_sha256` of numerical result axes, u/v, primary peak
  metrics, stored uncertainty, outlier/mask flags and scale metadata. Parameter
  objects and correlation planes are outside this binding. Replay's whole-file
  output SHA additionally covers the native result and companion bytes.

Origin codes use zero-based indices into `origin_codes`: unavailable, primary,
alternative, fill, masked, custom unclassified. A fill-origin value may itself
be nonfinite: assignment is an event, not a validity claim. Restored primary
values take precedence over fill events; masked cells take precedence over all
measurement events. Accepted alternative ranks are at least two; zero means no
alternative acceptance. Rejection-stage indices are one-based, with zero for
none. Their names describe the first observed flag transition at an actual
stage, not every predicate failure or a calibrated physical failure cause.

The uncertainty codes similarly use zero-based indices: not requested,
finite nonnegative, negative, nonfinite, masked. Availability does not certify
association with an accepted alternative or replacement. Primary origin does
not certify the estimator's near-zero residual assumption either.

The packet digest checks detached snapshot integrity. Companion-only loading
checks the schema, event/origin invariants, exact key and packet digest without
deserializing results. `verify_result=true` additionally reads just the selected
result to check the measurement binding. Neither digest authenticates a file
against deliberate rewriting. `ResultFile` stamps detect ordinary completed-file
changes; no concurrent-writer or resume guarantee is supplied. Ordinary result
readers ignore companion groups, and `save_results` copies results alone.

`history_id` is independent of `PIVExecutionDiagnostics.execution_id`; pair/key
and recipe/input association relate separately recorded companions. Source
provenance describes files on disk when the history is completed. Use a fresh
process and keep the checkout unchanged to associate it with loaded code.

Memory is O(final-grid nodes): four numeric primary/residual matrices, integer
reason/rank matrices, compact event/status/origin arrays, and grid axes. Work
arrays are reset each final-pass sweep; no earlier pass/sweep or sequence-wide
history is retained. Explicit data copies and retained callback packets have
their own storage cost. No observation arrays or hashes are computed when
history is not requested.

`verify_measurement_history(history, raw_result)` checks an already-loaded raw
payload without reading it again. Perform this before physical conversion.
`measurement_history_at(history, CartesianIndex(row, column))` returns detached
scalar observations without copying the entire packet. Both integrity checks
scan/hash packet data; the scalar accessor is O(grid nodes) in verification
work but constant-size in returned storage. These are useful for streaming
[history-aware reports](run_quality.md) and
[GUI inspection](../howto/gui_companions.md).

```@autodocs
Modules = [Hammerhead]
Pages = ["measurement_history.jl"]
```
