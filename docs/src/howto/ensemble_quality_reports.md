```@meta
CurrentModule = Hammerhead
```

# Check an ensemble result

A pooled vector field tells you the estimated displacement. A quality report
also shows how much recorded processing contributed to it and which stored
vectors are finite, masked or flagged.

## Record and inspect a small pool

Here two image pairs pass through two pooling passes. The left strip is masked.
The requested iteration budget is deliberately larger than one so the output
makes the ensemble's one-sweep-per-pass behavior visible.

```@example ensemble_report
using Hammerhead, Random

rng = MersenneTwister(742)
pairs = [begin
    a = rand(rng, Float32, 48, 48)
    (a, circshift(a, (1, 2)))
end for _ in 1:2]
p = PIVParameters(window_size=16, overlap=8, padding=true,
    max_iterations=8, convergence_tol=1e-3, uod_enable=false)
mask = falses(48, 48)
mask[:, 1:16] .= true

mktempdir() do directory
    source = joinpath(directory, "pooled.jld2")
    run_piv_ensemble(pairs, [p, p]; mask,
        threaded=false, progress=false, output=source, record_diagnostics=true)
    report = quality_report(ResultFile(source);
        include_ensemble_execution_diagnostics=true)
    saved = save_quality_report(joinpath(directory, "quality.toml"), report)
    data = quality_report_data(load_quality_report(saved))
    c = data["ensemble_execution_diagnostics"]["counts"]
    @assert c["pair_observations"] == 2
    @assert c["executed_pooling_sweeps"] == 2 && c["tolerance_checks"] == 0
    (; pooled_results=data["groups"]["planar"]["counts"]["entries"],
       input_pair_observations=c["pair_observations"],
       pooling_sweeps=c["executed_pooling_sweeps"],
       pair_contributions_across_passes=c["all_pass_pair_observations"],
       tolerance_checks=c["tolerance_checks"],
       masked_windows=c["all_pass_masked_window_pairs"],
       accumulated_windows=c["all_pass_accumulated_window_pairs"])
end
```

Expect **one result, two pooling sweeps, four pair contributions and zero
convergence checks**. Masked windows did not contribute correlation planes.
The requested eight iterations did not produce eight sweeps per pass.

## Read the useful numbers

| Question | Look at |
|:--|:--|
| Are the stored vectors usable? | `groups["planar"]`: finite, masked and outlier counts |
| Did a window contribute? | `all_pass_*_window_pairs`: masked, source-gated or accumulated |
| How many pairs supported the final grid? | `final_zero/some/all_finite_nonzero_count` |
| Was pooled uncertainty evaluated? | `final_uq_evaluated_entries` and the separate u/v availability fractions |

Finite nonzero planes can still be flat or have an unhelpful peak. Counts of
contributions do not establish independent sample size or uncertainty coverage.
Residual and uncertainty observations stay in processing pixels, even when the
stored field has a physical scale. Individual packets retain residual
amplitudes; the report does not average them across different grids.

## Choose the right report call

For a native result file:

```julia
report = quality_report(ResultFile("pooled.jld2");
    include_ensemble_execution_diagnostics=true)
```

For a saved recipe and completed run:

```julia
record = load_ensemble_experiment("ensemble-record.jld2")
report = quality_report(record, last(record.runs); verify_inputs=true)
```

The second call additionally verifies the saved run association; see
[Save and replay an ensemble experiment](ensemble_experiments.md).
Both inspect raw fields before summarizing. Reopening a saved TOML report
validates past report data and does not recheck the native result file.

A missing packet means **no recorded execution evidence**. A bare planar result
cannot tell you whether it was produced by an ensemble. In mixed native files,
ensemble, ordinary execution and history sections can be requested separately;
conflicting companion families on the same entry are refused.

See [Reports for pooled results](../reference/ensemble_quality_reports.md)
for populations, count partitions, file checks and supported inputs, or
[Ensemble result reports](../reference/ensemble_experiment_quality.md)
for saved-run association. The [GUI companion guide](gui_companions.md)
shows inspection and report saving in the result explorer.
