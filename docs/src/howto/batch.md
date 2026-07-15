# Batch processing and result files

**Goal:** process a whole recording to a results file, survive crashes
mid-batch, and read everything back later.

## Build the pair list

[`image_pairs`](@ref) groups an ordered frame list (file paths and/or
in-memory matrices) into correlation pairs. Build the list by filtering a
directory with `readdir` and Julia's own predicates — no extra dependency:

```julia
using Hammerhead

dir = "run_042"
# Every .tif in the directory, in sorted order (readdir already sorts):
files = filter(endswith(".tif"), readdir(dir; join = true))
# Or a stricter pattern — frames B00001.tif … B09999.tif only:
files = filter(f -> occursin(r"^B\d{5}\.tif$", basename(f)),
               readdir(dir; join = true))

pairs = image_pairs(files)                   # double-frame: (1,2), (3,4), ...
pairs = image_pairs(files; mode = :chained)  # time-resolved: (1,2), (2,3), ...
```

For sampled or multi-delay recordings, use `stride`, zero-based `offset`, and
one or more `deltas`:

```julia
pairs = image_pairs(files; mode=:chained, stride=2, offset=1, deltas=(1, 4))
```

`FrameSource(n, i -> read_vendor_frame(i))` adapts a proprietary CINE, MRAW,
SEQ, video, or camera reader without a package dependency. Pair construction
does not read frames. Optional `timestamps` travel with the pairs; when a
`PhysicalScale` is passed to the sequence driver, each result receives the
pair's actual `dt`. Use `TIFFStack("recording.tif")` for multi-page TIFFs.

For true shell-style `*` globbing across nested directories, add
[Glob.jl](https://github.com/vtjnash/Glob.jl) (`Glob.glob("cam1/*.tif")`);
the predicate filter above covers most batch needs without a dependency.

## Run the sequence with incremental output

```julia
passes = multipass_parameters([64, 32, 16]; padding = true, apodization = :gauss)
results = run_piv_sequence(pairs, passes;
    preprocess = img -> intensity_cap!(img),
    output = "run_042_piv.jld2",
    mask = mask,                  # optional, shared by all pairs
)
```

With `output` set, each result is written to the JLD2-format Julia data file
**as it completes** — a crashed batch keeps its finished pairs. For file-path pairs
the source image paths are stored alongside the results. Threading applies
within each pair (the window grid is split across tasks); results are
bitwise identical to serial processing.

To halve memory traffic on large recordings, load frames in single
precision with `image_type = Float32` (see the
[precision policy](../explanation/precision.md)).

For synchronized stereo recordings, pass one pair list per camera and reuse
the calibrated dewarpers across the run:

```julia
stereo = run_piv_stereo_sequence(cam1_pairs, cam2_pairs, dw1, dw2, passes;
    preprocess = (preprocess_cam1, preprocess_cam2),
    output = "run_042_stereo.jld2",
    progress = (done, total) -> update_progress(done, total),
    cancel = () -> cancellation_requested[],
)
```

`mask` may also be a vector with one entry per pair or a callback
`(i, frameA, frameB) -> mask`. Return `(maskA, maskB)` when geometry moves
between exposures; Hammerhead excludes their union (`true` always means
excluded). Pass `roi=ROI(101:900, 201:1200)` to analyze a crop while keeping
returned coordinates in the original image frame.

The driver stops cleanly between acquisitions when `cancel()` becomes true
and returns the completed prefix. Each completed `StereoPIVResult` is already
persisted. A single shared preprocessing function is also accepted. The
equivalent one-list layout is `(A1, B1, A2, B2)` per acquisition. Timestamped
camera `FramePair`s must have matching intervals and attach their actual `dt`
to each scaled stereo result.

## Run the sequence on a GPU

Load the device package and pass its selector:

```julia
using AMDGPU
results = run_piv_sequence(pairs, passes;
    backend = :amdgpu,
    image_type = Float32,
    output = "run_042_piv.jld2",
)
```

The sequence driver reuses one device workspace across pairs while the next
pair's load and preprocessing are prefetched on the CPU. Output remains
incremental and results remain ordinary host-side `PIVResult` objects. See
[Run PIV on a GPU](gpu.md) for installation, supported options, and when
offload pays for itself.

## One file per pair

To write each pair to its **own** JLD2 file instead of one combined file,
pass `output` a function `(i, pair) -> outpath` (`i` is the 1-based pair
index, `pair` the original 2-tuple). [`frame_index_strings`](@ref) pulls the
differing frame-index substrings out of a path pair, so the outputs can
mirror the inputs:

```julia
outdir = "run_042_piv"
run_piv_sequence(pairs, passes;
    output = (i, pair) -> begin
        a, b = frame_index_strings(pair...)      # "img_0001.tif","img_0002.tif" → "0001","0002"
        joinpath(outdir, "piv_$(a)_$(b).jld2")
    end,
)
```

Parent directories are created automatically, and each file is a standalone
single-result file (with source paths recorded for file-path pairs) readable
with [`load_results`](@ref). Fall back to the index `i` when frames aren't
named by index. The same `output` function works for
[`run_ptv_sequence`](@ref).

## Read results back

```julia
results = load_results("run_042_piv.jld2")   # Vector of results, in order
```

[`load_results`](@ref) returns [`PIVResult`](@ref) and/or
[`StereoPIVResult`](@ref) entries in sequence order. The stored source
paths, when present, are retrievable directly with JLD2:

```julia
using JLD2
sources = JLD2.load("run_042_piv.jld2", "sources/000001")
```

Standalone results (e.g. from interactive work) round-trip with
[`save_results`](@ref) / [`load_results`](@ref):

```julia
save_results("snapshot.jld2", result)        # single result or a vector
```

`TrackingResult` is supported by the same lossless JLD2 persistence. For
language-neutral exchange, `export_table("field.csv", result)` writes the
stable `hammerhead-table-1` long-form schema. Its fixed columns are
`TABLE_COLUMNS`; planar, stereo, and PTV rows use the same superset and leave
inapplicable values empty. Identifiers, flags, quality values, uncertainties,
and unit strings are included, and attached scaling is applied. Planar and
stereo grids can also be written for ParaView with
`export_vtk("field.vtk", result)` (legacy ASCII structured-grid VTK).

## Post-process the sequence

Sequences of same-grid results feed the statistics utilities directly:

```julia
validate_temporal!(results)                  # flag temporal outliers in place
stats = field_statistics(results)            # planar or stereo mean/RMS/stresses/counts
f, psd = power_spectrum([r.u[12, 8] for r in results]; dt = 1/10_000)
```

## A note on failure

If a pair fails (unreadable file, size mismatch), `run_piv_sequence` logs
which pair and rethrows — the incremental output file retains everything
processed up to that point, so you can fix the offending frame and resume
from a trimmed pair list.
