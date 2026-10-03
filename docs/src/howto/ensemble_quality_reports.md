# Summarize recorded ensemble execution

Save the ensemble companion alongside its raw result, then summarize the
completed native file with an explicit format-4 opt-in:

```julia
using Hammerhead, Random

A = rand(MersenneTwister(741), 48, 48)
B = circshift(A, (1, 2))
p = PIVParameters(window_size=16, overlap=8, padding=true,
    max_iterations=8, convergence_tol=1e-3, uncertainty=true)

run_piv_ensemble([(A, B), (A, B)], [p, p];
    threaded=false, progress=false,
    output="ensemble-results.jld2", record_diagnostics=true)

report = quality_report(ResultFile("ensemble-results.jld2");
    include_ensemble_execution_diagnostics=true)
display(report)
save_quality_report("ensemble-quality.toml", report)
section = quality_report_data(report)["ensemble_execution_diagnostics"]
```

The result file contains **one result per pool**, rather than one per contributing
image pair. Each recorded pass performed one pooled sweep; requested iteration
budgets and tolerance values were ignored. Repeated passes do not demonstrate
convergence. The report leaves residual amplitudes in the individual packets,
where their processing grid and pixel basis remain explicit.

## Interpret coverage and contributions

Every planar entry is classified as a recorded ensemble, a recorded ordinary
iteration workflow, or an entry without execution metadata. A missing ensemble
packet cannot establish whether that entry was an ensemble at all. Consequently
the reported ensemble fraction uses **all examined planar entries** as its
denominator; it is not coverage of an independently known ensemble population.

All-pass window/pair opportunities partition into masked, original-source-gated
and accumulated contributions. Accumulated numerical planes partition into
finite zero, finite flat nonzero, finite nonflat and nonfinite planes. Source
support observations use the original sampled 4×4 stencil convention described
in [non-informative windows](../explanation/noninformative_windows.md).
Finite nonzero includes flat nonzero planes; it does not certify a useful peak.
Pair observations can repeat the same acquisition in different pools or passes;
they are not counts of unique inputs or independent samples.

Final-pass node populations distinguish zero, some and all finite nonzero
contributions, nonfinite planes, masks, and finite/nonfinite primary peaks.
These are node observations across recorded pools, not correspondence across
different interrogation grids. The primary support precedes predictor addition
and validation. Current stored output quality remains in the ordinary `groups`
section; replacement or validation can change the final fields.

Final pooled UQ counts cover only pools whose statistics were actually
evaluated. The two component populations separately partition eligible nodes
into finite nonnegative, finite negative and nonfinite values, before validation
and availability cleanup. Disabled, intermediate or entirely masked cases do
not invent evaluated estimates. Zero denominators remain unavailable. These
counts are distinct from the stored output's UQ availability and do not establish
estimator applicability, common displacement, effective sample size or coverage.
Residual and UQ observations retain processing-pixel units even when a result
has a physical scale.

## Combine supported observations

A completed native file can contain ensemble and ordinary planar/stereo entries.
Request their separate sections explicitly:

```julia
report = quality_report(ResultFile("mixed-results.jld2");
    include_ensemble_execution_diagnostics=true,
    include_execution_diagnostics=true,
    include_measurement_history=true)
```

The ordinary execution and history sections retain their existing denominators.
An ensemble entry therefore lacks ordinary pair-iteration/history coverage;
the format-4 classification explains that absence. Different entries may carry
different companion families. An ensemble packet combined with ordinary
iteration, stereo diagnostics or measurement history **on the same entry** is
refused. Wrong-kind packets, unsupported root markers, malformed companion
roots and orphan entries are refused across all four inspected families,
including empty native files.

The default report remains format 1, history opt-in remains format 2, and
ordinary execution alone remains format 3 with its existing ensemble refusal.
Format 4 requires a direct whole-file `ResultFile`; bare/physical result arrays,
views, checkpoints and experiment-record association are unsupported for this
opt-in. No replayable ensemble recipe association is inferred from planar or
stereo experiment records.

## Verification and memory

Generation loads each raw native result once, checks any ensemble packet's
measurement-field digest and independent grid/array/flag geometry against that
payload, and retains fixed-count summaries. Binding excludes parameter objects,
retained correlation planes and source bytes. The completed file is checked and
hashed before/after reading. There is no concurrent-writer guarantee or claim
about calibration, input authenticity or scientific assumptions.
Before recording provenance or entry populations, the supplied index keys must
exactly match the independently sorted native result keys. Omissions, duplicates
and reordering are refused. The report uses a detached metadata snapshot for its
scan; changing a caller-owned index vector cannot change that selected request.

`load_quality_report` validates saved structure, count partitions and fractions.
It describes checks **when the report was generated** and never freshly verifies
the recorded source. Save guards protect known local source destinations. Metric
storage is bounded independently of entry count; the native index and protected
locator metadata retain O(entries) strings. One loaded result can include any
saved correlation planes.

The [GUI companion guide](gui_companions.md) describes raw verification before
physical display and native-explorer report saving. See the
[format-4 schema](../reference/ensemble_quality_reports.md) for exact fields.
