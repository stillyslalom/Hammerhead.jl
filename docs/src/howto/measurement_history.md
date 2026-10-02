# Record the origin of returned planar vectors

Enable measurement history when you need to distinguish the final primary peak,
an accepted alternative, and a median assignment. This records only the final
pass's final executed sweep. It does not reconstruct history from old result
files or trace vectors through earlier grids and sweeps.

```@example history_pair
using Hammerhead, Random
rng = MersenneTwister(204)
a = rand(rng, Float32, 48, 48)
b = circshift(a, (1, 2))
params = PIVParameters(window_size=16, overlap=8, padding=true,
    n_peaks=3, max_iterations=2, uncertainty=true)
captured = Ref{Any}()
result = run_piv(a, b, params;
    on_measurement_history = h -> (captured[] = h))
data = measurement_history_data(captured[])
@assert data["x"] == result.x && data["y"] == result.y
@assert data["sweep_index"] == 2
alternative_count = count(>(0), data["accepted_peak_rank"])
fill_count = count(data["fill_assigned"])
(alternative_count, fill_count)
```

Every matrix uses `[row, column]` on the result's `y` and `x` axes. Coordinates
remain original-image pixels, including ROI offsets; displacement and primary
residual components remain pixels even with an attached physical scale.
`accepted_peak_rank` is zero when no alternative was accepted, and at least two
for an actual accepted ordered alternative. Peak ratio and correlation moment
in the result still describe the primary correlation measurement: alternatives
are accepted by the existing local median rule, without rerunning all validators.

`first_rejection_stage` records the first actual transition to a flag; zero
means no observed rejection. Index `rejection_stages` with positive entries to
identify that stage. A later validator may also fail an already flagged node,
so this is not an exhaustive list of failed criteria. Custom validators execute
once in their existing order. Their arbitrary field effects are not classified
as primary measurements; `custom_unclassified` identifies this limitation.

`fill_attempted` and `fill_assigned` describe real median branch events.
Assignment can produce a nonfinite value or leave a value numerically unchanged.
When iterations use fills internally but `replace_outliers=false`,
`primary_restored` identifies flagged nodes restored to their original primary
values. `final_origin` reflects that restoration rather than calling the
returned value a fill. Masked cells have no measurement events.

Persist a streaming sequence without retaining all result or history payloads:

```@example history_sequence
using Hammerhead, Random
rng = MersenneTwister(205)
a = rand(rng, Float32, 48, 48)
b = circshift(a, (1, 2))
params = PIVParameters(window_size=16, overlap=8, padding=true, n_peaks=3)
mktempdir() do directory
    path = joinpath(directory, "results.jld2")
    run_piv_sequence([(a,b), (a,b)], params; output=path,
        record_measurement_history=true, collect_results=false, progress=false)
    index = ResultFile(path)
    history = load_measurement_history(index, 2; verify_result=true)
    @assert measurement_history_data(history)["pair_index"] == 2
    show(stdout, MIME"text/plain"(), history)
end
```

`on_measurement_history(i,h)` runs before `on_result`, persistence and progress.
Exceptions propagate and prefetched loading finishes before the driver returns.
The driver retains one current history packet and clears it on success/failure;
callbacks that retain packets explicitly increase memory use. A function-valued
`output` writes one file per pair, retaining the absolute input-sequence index
in each companion. `record_measurement_history=true` requires a native output.
`replay_experiment` accepts the same history options and records verified recipe
and input identities without changing scientific recipe identity. Checkpoint,
stereo, ensemble and PTV history are not implemented.

History owns detached mutable arrays. `measurement_history_data` returns another
detached copy; modifying that copy is safe. Changing private packet storage is
detected by its frozen digest. Before writing a result with history, the driver
checks that callbacks have changed neither the packet nor the bound numerical
result fields. A failed check leaves the current pair unwritten; earlier pairs
remain readable. This is not an atomic transaction or a concurrent-writer API.

Stored uncertainty status reports only numerical availability per component:
finite/nonnegative, negative, nonfinite, not requested, or masked. It uses the
final deformed windows' zero-shift correlation statistics. Alternatives are not
reestimated and fill uncertainty is not propagated. Even a finite sigma attached
to a primary-origin output does not establish estimator applicability, accuracy,
coverage or convergence. Primary residual arrays make the measured residual
available for separate review without inventing those conclusions.
