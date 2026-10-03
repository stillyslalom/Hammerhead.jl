# Inspect ensemble execution diagnostics

Ensemble PIV pools correlation planes across pairs before locating a displacement
peak. Its execution observations have a separate companion from ordinary
single-pair/sequence diagnostics: one pooling sweep executes per scheduled pass,
and `max_iterations` and `convergence_tol` are ignored. Repeating a final pass
does not establish convergence or common displacement across pairs.

```julia
using Hammerhead, Random

A = rand(MersenneTwister(741), 48, 48)
B = circshift(A, (1, 2))
p = PIVParameters(window_size=16, overlap=8, padding=true,
    max_iterations=8, convergence_tol=1e-3, uncertainty=true)
packet = Ref{Any}()
result = run_piv_ensemble([(A, B), (A, B)], [p, p];
    threaded=false, progress=false,
    on_diagnostics=d -> (packet[] = d),
    output="ensemble-results.jld2", record_diagnostics=true)

display(packet[])
loaded = load_ensemble_execution_diagnostics("ensemble-results.jld2";
    verify_result=true)
data = execution_diagnostics_data(loaded)
```

Both explicit-schedule and effort overloads support `on_diagnostics(d)`,
`output=nothing` and `record_diagnostics=false`. Recording requires an output
path; callback-only capture can also accompany an ordinary native result save.
Capture supports CPU and the KA CPU proving backend with Float32/64. CUDA/AMDGPU
capture rejects explicitly before loading; it does not fall back to CPU. Existing
backend option restrictions still apply, including CPU-only retained planes and
enlarged search areas. No vendor hardware validation is implied.

## Interpret the populations

Each pass retains scalar summaries with its actual processing precision, image
size, interrogation grid and predictor presence. The opportunity count is pairs
times grid nodes, partitioned into masked, original-source-gated and accumulated
window/pair contributions. Accumulated planes are partitioned into finite exact
zero, finite flat nonzero, finite nonflat and nonfinite planes. These describe the
values used by the existing addition, without discarding nonfinite inputs or
changing accumulation arithmetic. A nonzero/nonflat plane is not a verified
individual displacement measurement.
Later passes can still have no predictor when the prior field has no usable
values; observations report the actual presence rather than infer it from the
pass index.

Per-node summaries report zero/some/all finite nonzero contributions and their
minimum/maximum counts. Masked nodes are excluded. Temporary contributor counts
cost O(grid nodes), independently of pair count; neither per-pair traces nor
planes are retained in the packet. Weak contributions are not thresholded away.
Processing-precision underflow can still yield an exact zero numerical plane.

Source-support status distinguishes no predictor, evaluated original-stencil
support and a guard disabled by nonfinite source pixels. It uses the existing
[original-source convention](../explanation/noninformative_windows.md): sampled
clipped raw 4×4 stencils, masks and the union of supported deformation arithmetic.
This is not a mathematical claim about nonlocal B-spline coefficient tails.
Gate skips omit the CPU contribution or stage numerical zeros in the KA path;
they are distinct from an admitted exact-zero plane.

Primary residual summaries come directly from the pooled peak analysis before
adding the shared predictor, validation, alternative substitution or filling.
Their unit remains processing pixels even when a physical scale is attached.
They are not mean pairwise residuals, reconstructed world residuals or residuals
associated with every final output vector.

Pooled uncertainty is evaluated only on the final scheduled pass when requested
and eligible windows exist. Its admitted update count describes window/pair inputs
fed to the additive statistics, including admitted zero/nonfinite planes. That
count is neither usable-statistics count nor an effective independent sample size.
Component numerical populations are recorded separately before validation and
unavailable-UQ cleanup; they may differ from final output availability. The
estimator's common-displacement assumption, bias, temporal independence and
coverage are not checked by these observations.

## Persistence and verification

The completed raw result and optional packet are written after the callback.
Known selected file/TIFF aliases and invalid recording options reject before
effort sizing or processing loads. Selection containers, pass schedule and static
mask are snapshotted for capture/output before loading/preprocessor callbacks;
caller-owned pixel contents and arbitrary custom
preprocessor dependencies are not locked or authenticated. Callback exceptions
and detected measurement edits occur before opening the output destination.
Writing result and companion within the native file is not transactional crash
recovery. No incomplete-execution packet is invented after a failed computation.

The default reader checks metadata schema, scalar partitions, digest and exact
result-key linkage. `verify_result=true` reads and discards one selected raw
result, checking its precision/shape/grid and measurement-field binding. It does
not retain arrays or a file handle. Alternatively:

```julia
packet = load_ensemble_execution_diagnostics("ensemble-results.jld2")
raw = only(load_results("ensemble-results.jld2"))
data = execution_diagnostics_data(packet; result=raw)
```

Supplied-result inspection verifies without a second payload read and leaves the
packet's original runtime state unchanged. Binding covers axes, displacement,
quality metrics, UQ, masks/outlier flags and attached scale. Parameters and
correlation planes are excluded. Converted/edited measurement fields reject.
Metadata integrity is not source-byte authenticity or scientific validation.
The source digest records on-disk core source: use a fresh Julia process with
unchanged sources to relate it to loaded code.

Native result layout/version and planar/stereo companion schemas remain
unchanged. `save_results` on a bare result cannot recreate execution observations.
Stereo ensemble, experiment/checkpoint integration and timing/history companions
remain unsupported. [Ensemble quality reports](ensemble_quality_reports.md)
use explicit `include_ensemble_execution_diagnostics=true` opt-in and a separate
format-4 section for pooled populations. Earlier execution-report formats still
refuse the ensemble marker; ordinary default result reports remain available.
Lazy native GUI inspection verifies ensemble raw fields before physical display
and shows pooled counts without inventing per-node measurement history.
