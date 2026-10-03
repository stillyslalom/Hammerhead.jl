```@meta
CurrentModule = Hammerhead
```

# Ensemble result reports

Summarize a completed saved ensemble run with its record:

```julia
record = load_ensemble_experiment("ensemble-record.jld2")
report = quality_report(record, last(record.runs); verify_inputs=true)
display(report)
save_quality_report("ensemble-quality.toml", report)
```

Follow [Save and replay an ensemble experiment](../howto/ensemble_experiments.md)
for a runnable example, or [Check an ensemble result](../howto/ensemble_quality_reports.md)
for interpreting contribution counts. The details below define the saved report
and its verification limits.

## Options and saved format

`quality_report(record::EnsembleExperimentRecord, run::EnsembleExperimentRun)`
uses version 5 of the TOML report schema. Existing generic and planar/stereo
overloads retain formats 1–4. This overload defaults to
`include_ensemble_execution_diagnostics=true`; setting it to `false` keeps
format 5 and omits the pooled execution section. History and ordinary iteration
sections are unsupported for this associated ensemble workflow.

The root contains the existing report fields, `entry_kinds`, and optionally
`ensemble_execution_diagnostics`. Stored-array groups retain their existing
numerators, denominators and numerical uncertainty availability rules. Pooled
execution uses the unchanged [format-4 scientific counters](ensemble_quality_reports.md).
The native source must contain exactly one planar result. Input pairs are not
counted as individually measured or independently informative results.

## Input pairs, processing work and results

The provenance mapping has `association="recorded_ensemble_output_verified"`,
`workflow="planar_ensemble"`, `verification_time="report_generation"`, the
usual source path/hash/index count/selection, and recipe/input/run/environment
identities. It has no `completed_pairs` field.

| Field | Completed-run contract |
|:--|:--|
| `input_pairs` | Positive number of ordered input pairs |
| `scheduled_passes` | Positive number of requested pooling passes |
| `total_contributions` | Checked product of input pairs and scheduled passes |
| `completed_contributions` | Equals the total, including gated or masked pair work |
| `completed_pools` | Exactly one completed pooled estimate, not one per pass |
| `published_results` | Exactly one native result |
| `record_diagnostics` | Boolean requested recording policy, not inferred from missing metadata |

## What is verified

`input_bytes_checked` records whether current input files were checked during
generation. `raw_measurement_fields_checked`, `recipe_grid_mask_scale_checked`
and `requested_companions_checked` are strictly `true`. The source index is
checked against actual sorted native keys and independently captured. Completed
run identity, output SHA, raw measurement binding, final recipe geometry/mask/
scale and requested companions are verified before and after aggregation.

When a pooled section is requested, its recorded coverage agrees with the run's
recording policy. Recorded packet pair/pass/contribution counts agree with the
run counts. With recording disabled, a requested section retains the missing
execution-metadata population; it does not invent observations.

Loading TOML validates the saved schema and arithmetic without opening any
source or record. These flags describe **past generation-time verification**.
Hashes do not establish source authenticity, numerical rerun correctness,
stationarity, independent samples, effective sample size or uncertainty coverage.
Per-node ensemble measurement history remains unavailable.

## Relocation and saving

`output=local_path` explicitly locates a relocated native artifact; historical
locators remain unchanged. `verify_inputs=true` additionally checks current input
bytes at both verification boundaries. Known local inputs, records, historical
results and the consumed artifact are protected by `save_quality_report`.
Foreign locators are never silently converted to local paths. Caller-supplied
`protected_paths` can add dependencies. Ordinary report destinations may be
overwritten; this is not atomic publication or concurrent-writer safety.

## API

```@autodocs
Modules = [Hammerhead]
Pages = ["src/ensemble_experiment_quality.jl"]
```
