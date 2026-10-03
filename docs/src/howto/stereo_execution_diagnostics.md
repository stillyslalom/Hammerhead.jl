# Inspect per-camera stereo execution

Use `on_diagnostics` to inspect actual per-camera passes from stereo single-pair
or sequence processing. The callback receives one immutable
`StereoPIVExecutionDiagnostics` **after both camera analyses, reconstruction and
scale attachment**. It does not receive two independent camera callbacks.
Requested settings alone cannot recover these observations from older results.

```@example stereo_execution_diagnostics
using Hammerhead, Random

# Synthetic camera models agree on the z=0 image plane and have different rays.
camera1 = PinholeCamera([100. 0 15. 0; 0 100. 0 0; 0 0 1. 100.])
camera2 = PinholeCamera([100. 0 -15. 0; 0 100. 0 0; 0 0 1. 100.])
grid = DewarpGrid(x=1.:48., y=1.:48.)
dw1 = ImageDewarper(camera1, grid, (48,48))
dw2 = ImageDewarper(camera2, grid, (48,48))
A = rand(MersenneTwister(13), 48, 48)
B = circshift(A, (1,2))
params = PIVParameters(window_size=16, overlap=8, uod_enable=false,
    max_iterations=4, convergence_tol=1e6)
packet = Ref{Any}(nothing)
result = run_piv_stereo(A, B, A, B, dw1, dw2, params;
    threaded=false, on_diagnostics=d -> (packet[]=d))
@assert packet[].cam1.execution_id != packet[].cam2.execution_id
@assert packet[].cam1.passes[1].executed_iterations == 2
data = execution_diagnostics_data(packet[])
@assert data["verification"]["measurement_field_binding_checked"]
data["cameras"][1]["diagnostics"]["passes"][1]
```

Each child uses the existing planar contract: actual sweep counts, actual
tolerance checks and stop reason, and the final primary-peak residual summary
before validation/filling. Camera execution IDs differ; both have an explicit
parent execution ID and camera role. A sequence packet and both children carry
the absolute acquisition index, including one-result-per-file output. Camera
roles are driver argument positions, not authenticated camera identities.

## Interpret coordinates and convergence

Residuals and tolerance comparisons remain in **dewarped processing pixels**:
x runs along columns and y along rows. The packet records common world-grid
endpoints, counts, z and signed steps; a descending world axis keeps its negative
step. A common ROI can crop planar processing: each child's `processing_size`
describes that crop, while the world geometry describes the original dewarp grid.
Both cameras must share their processing and output grids. Separate per-camera
ROIs are not introduced by this API.

These scalar residual magnitudes cannot be converted into an anisotropic world
norm without their lost directional information. No reconstructed 3C residual or
calibration uncertainty is invented. Attaching a scale does not convert these
diagnostics. The scale is included in measurement-field binding, but it does not
establish a physical unit for the provided calibration.

A tolerance condition describes the predictor after validation/filling, not
measurement acceptance. An empty comparison can satisfy it. Residuals describe
primary peaks before validation and are not associated with substituted or filled
final vectors. Read the [planar execution guide](execution_diagnostics.md) for
these shared stopping and residual conventions.

## Record and read sequence companions

```@example stereo_execution_diagnostics
path = joinpath(mktempdir(), "stereo.jld2")
run_piv_stereo_sequence(fill((A,B,A,B), 2), dw1, dw2, params;
    threaded=false, progress=false, output=path, record_diagnostics=true,
    collect_results=false, on_diagnostics=(i,d) -> nothing)

index = ResultFile(path)
metadata_only = load_stereo_execution_diagnostics(index, 2)
verified = load_stereo_execution_diagnostics(index, 2; verify_result=true)
@assert metadata_only.pair_index == 2
@assert !execution_diagnostics_data(metadata_only)["verification"]["measurement_field_binding_checked"]
@assert execution_diagnostics_data(verified)["verification"]["inspection_state"] == "loaded_measurement_fields_verified"
verified
```

`record_diagnostics=true` requires a native output path or function. Callback-only
capture needs no output. Sequence order is diagnostics, `on_result`, output,
progress; delivery runs on the caller's task. A function output is resolved before
the final binding check and before opening that destination. Callback errors
propagate and outstanding prefetch work finishes before returning. The current
packet and camera capture references are cleared on success and failure.

Measurement-field binding includes reconstructed coordinates, u/v/w, stored UQ,
flags and scale, plus both cameras' planar coordinates/components, peak ratio,
correlation moment, stored UQ, flags and scale. Each camera also has a separate
role-specific binding. **Parameter objects and correlation planes are excluded**;
this is not full result serialization identity. Captured fields are rechecked
after diagnostics, result, output-path and progress callbacks. Mutation is
refused even when capture is callback-only. A progress callback runs after
persistence, so a failure there can leave an already completed entry on disk.
Later caller edits can invalidate a previously recorded binding; verification is
a check at capture/read time, not a continuing guarantee.

Native result format 1 and `StereoPIVResult` layouts stay unchanged. Optional
companions use a separate versioned `stereo_execution_diagnostics` sibling group.
Ordinary eager/lazy readers still return results; planar companion readers stay
unchanged. Bare `save_results(result)` does not invent or copy diagnostics.

The default stereo companion reader reads metadata only and checks schema,
counts, geometry, camera/parent associations, checksum and exact result-key
linkage. `verify_result=true` reads/discards one selected stereo payload and checks
its measurement-field binding, independent dimensions and grid mapping. The
inspection status is runtime-only and is not persisted as a verification claim.
Missing metadata returns `nothing`; unsupported versions or malformed metadata
raise errors. File size/mtime checks do not provide concurrent-writer safety.

When a consumer already loaded the raw payload, verify it without loading it
again:

```@example stereo_execution_diagnostics
raw = index[2]
checked = execution_diagnostics_data(metadata_only; result=raw)
@assert checked["verification"]["inspection_state"] == "supplied_measurement_fields_verified"
@assert checked["verification"]["measurement_field_binding_checked"]
@assert execution_diagnostics_data(metadata_only)["verification"]["inspection_state"] == "metadata_only"
```

The returned dictionary is detached; the packet remains unchanged and retains
no supplied result. The fields must match the captured **raw** values and scale,
before any physical display conversion. This verifies measurement fields and
geometry, without checking native source bytes or the excluded parameters/planes.
For bounded aggregate counts, use [execution-aware quality reports](run_quality.md).

The packet retains O(camera passes) scalar observations, not field/image arrays or
sweep traces. Noncollecting execution retains the current result and bounded
prefetched images; retaining packets alone does not retain earlier results. Opt-in
binding checks scan current measurement arrays and hash the on-disk core sources.
Run a fresh Julia process and keep sources unchanged to relate that checkout hash
to loaded code; it does not identify edited code already loaded into a process.

Neither reader nor packet verifies calibration, input bytes, synchronization,
physical accuracy or final-vector measurement origin. Checksums are integrity
checks, not signatures. Ensemble diagnostics, stereo measurement history
and checkpoint/replay integration remain unsupported. Lazy native GUI inspection
uses the same supplied-raw verification before physical conversion; see
[recorded processing details](gui_companions.md). Stored-field quality metrics
remain separate from the opt-in execution report section.

See the [stereo diagnostics reference](../reference/stereo_execution_diagnostics.md).
