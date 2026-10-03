```@meta
CurrentModule = Hammerhead
```

# Check and save a result summary

Use a quality report to count finite, masked and flagged vectors in a result
file. Start here when you want a compact record of what the saved fields
contain, without rerunning PIV.

## Try a field with a mask and a missing vector

This tiny example deliberately includes one masked node and one flagged node
whose horizontal component is `NaN`.

```@example run_quality
using Hammerhead

parameters = PIVParameters(window_size=4, overlap=2, uncertainty=true)
result = PIVResult([1., 2.], [1., 2.],
    [1. NaN; 3. 4.], ones(2, 2), ones(2, 2), ones(2, 2),
    [0.1 0.2; -0.1 NaN], fill(0.2, 2, 2),
    BitMatrix([false true; false false]),
    BitMatrix([false false; false true]), parameters)

mktempdir() do directory
    source = save_results(joinpath(directory, "vectors.jld2"), [result])
    report = quality_report(ResultFile(source))
    destination = save_quality_report(joinpath(directory, "quality.toml"), report)
    data = quality_report_data(load_quality_report(destination))
    c = data["groups"]["planar"]["counts"]
    f = data["groups"]["planar"]["fractions"]["finite_output_fraction"]
    @assert c["masked"] == 1 && c["unmasked"] == 3
    @assert c["finite_output_unmasked"] == 2 && f["value"] == 2/3
    (; grid_nodes=c["nodes"], masked=c["masked"], unmasked=c["unmasked"],
       finite_unmasked=c["finite_output_unmasked"], finite_fraction=f["value"])
end
```

There are four nodes: one is masked, leaving three for inspection. Two of
those three have finite vectors, so the finite fraction is **2/3**, not 2/4.
The report keeps masks and outlier flags separate from numerical finiteness.
A finite vector can still be flagged.

## Summarize your file

```julia
report = quality_report(ResultFile("results.jld2"))
display(report)
save_quality_report("quality.toml", report)
```

Planar and reconstructed stereo fields have separate groups. Node counts are
summed across entries, so larger grids carry more weight. Physical scales do
not affect these dimensionless counts. A zero denominator produces an
unavailable fraction, rather than zero percent.

Stored uncertainty is numerically available when finite and nonnegative.
This says nothing about its accuracy or its applicability after a vector was
replaced. Use the [metric reference](../reference/run_quality.md) when choosing
a denominator or interpreting uncertainty counts.

## Check a saved run against its record

```julia
record = load_experiment("experiment.jld2")
run = last(record.runs)
report = quality_report(record, run)
save_quality_report("run-quality.toml", report)
```

Only completed runs can be reported this way. The report verifies their
recorded output identity and stores recipe, input and run identities.
Use `load_stereo_experiment` for a saved stereo record; use
`load_ensemble_experiment` for a saved ensemble record. A direct file report
has no experiment association.

Stereo and ensemble report overloads also accept `verify_inputs=true` to check
current input bytes, and `output="local/moved.jld2"` to locate a moved output.
The planar sequence overload uses the recorded output path and does not reopen
input images. No preprocessing script is evaluated by reporting.

## Include recorded processing events

Choose the evidence you recorded alongside the native result:

| Evidence | Report option | What it adds |
|:--|:--|:--|
| Final planar measurement history | `include_measurement_history=true` | Rejection, alternative-peak, fill and restoration events |
| Planar or per-camera stereo execution | `include_execution_diagnostics=true` | Passes, sweeps, tolerance checks and stop reasons |
| Pooled ensemble execution | `include_ensemble_execution_diagnostics=true` | Pair/window contributions and pooled support |

Missing companions reduce evidence coverage; they are not zero-event
observations. Camera residuals remain in dewarped pixels. These summaries do
not establish measurement accuracy. See the focused guides for
[measurement history](measurement_history.md),
[execution diagnostics](execution_diagnostics.md), or
[ensemble results](ensemble_quality_reports.md), including supported inputs.

Loading a TOML report checks its saved schema, not current source files.
Regenerate it for a fresh check. Keep the native file unchanged while reporting;
see the [reference](../reference/run_quality.md) for verification, supported
inputs and save protections.
