```@meta
CurrentModule = Hammerhead
```

# Inspect actual planar pass execution

Use `on_diagnostics` to observe the sweeps actually executed by planar PIV.
Requested `max_iterations` is a budget; it is not an executed-sweep count.
Diagnostics are opt-in companions, leaving the result object unchanged.

```@example execution_diagnostics
using Hammerhead, Random

a = rand(MersenneTwister(821), 40, 40)
b = circshift(a, (1, 2))
parameters = PIVParameters(window_size=16, overlap=8, padding=true,
    max_iterations=3, convergence_tol=1e6)
observed = Ref{Any}(nothing)
result = run_piv(a, b, parameters; threaded=false,
    on_diagnostics=diagnostics -> (observed[] = diagnostics))
pass = only(observed[].passes)
(requested=pass.requested_iterations, executed=pass.executed_iterations,
 stop=pass.stop_reason, checks=pass.checks,
 residual=pass.residual)
```

The callback receives an immutable [`PIVExecutionDiagnostics`](@ref) after
numerical completion. Its tuple of [`PassDiagnostics`](@ref) records each pass
and its final primary residual summary. Nested observations contain immutable
scalar named tuples. For GUI/script consumption,
`execution_diagnostics_data(observed[])` returns a detached primitive dictionary.
Display it with `show(stdout, MIME"text/plain"(), observed[])`.

Preserve the distinction between tolerance outcomes and measurements. The
solver compares post-validation/filling fields using the 95th percentile of
the largest absolute component change at each contributing unmasked node.
It always exits the last budgeted sweep before evaluating tolerance. Thus a
two-sweep pass has zero comparisons; a budget-exhausted pass reports the last
earlier check, if any. A zero tolerance disables checking; a positive infinite
tolerance is supported. Empty comparison support produces the solver's actual
zero-valued decision and is explicitly counted, not evidence of valid vectors.

Primary residuals are the correlation output before predictor addition,
alternative-peak substitution or vector filling. Their magnitude summaries
remain processing pixels even when a physical scale is attached. A first
sweep without a predictor measures displacement relative to zero; it is not
automatically a small corrective residual. These summaries cannot be assigned
to substituted or filled output vectors.

To store indexed companions without retaining results, enable native output:

```@example execution_diagnostics
mktempdir() do directory
    output = joinpath(directory, "vectors.jld2")
    run_piv_sequence([(a, b), (a, b)], parameters;
        output, record_diagnostics=true, collect_results=false,
        progress=false, threaded=false)
    index = ResultFile(output)
    diagnostics = load_execution_diagnostics(index, 2)
    (pair=diagnostics.pair_index,
     sweeps=only(diagnostics.passes).executed_iterations,
     stop=only(diagnostics.passes).stop_reason)
end
```

Sequence `on_diagnostics(i, diagnostics)` runs before `on_result`, writing and
progress. It indicates numerical completion, not publication. Exceptions
propagate before that pair is written; pending frame loading is joined, and
the existing completed prefix remains. The driver retains only the current
summary; callbacks may deliberately retain summaries themselves. Persisting
requires an output path or function. One-result-per-file output retains the
absolute position in the supplied sequence as `pair_index`, while its native
result key remains `results/000001`.

For experiment replay, use
`replay_experiment(record; output="vectors.jld2", record_diagnostics=true)`.
Its companions record the verified recipe/input identities; the run's output
byte hash includes them. Observing execution does not change scientific recipe
identity. Loading metadata later validates its schema without inferring it from
requested settings or deserializing result payloads. Missing metadata returns
`nothing`. Ordinary `save_results` copies numerical results only and omits these
companions; retain the original file when its execution observations matter.

Run in a fresh Julia process and keep source files unchanged. The reporting
source hash describes the on-disk checkout at reporting time and cannot certify
loaded modules after editing source in that process. Native files must remain
unchanged during indexed reads; these observations do not add concurrent-writer
or atomic-publication guarantees.

This planar companion covers single runs, sequences and replay. Stereo single
runs and sequences use a separate [per-camera companion](stereo_execution_diagnostics.md).
PTV requests are rejected before frame loading/output opening; ensemble
diagnostics are unsupported. Ensemble still ignores `max_iterations`, and no
diagnostics fabricate an iteration outcome for it. Checkpoint capture/export and
pooled ensemble summaries remain separate work. With diagnostics disabled, no
new observations, residual summaries or diagnostics source hashes are computed.
