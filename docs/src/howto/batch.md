# Batch processing and result files

**Goal:** pair frames in acquisition order, save completed results as a batch
runs, and read them back for validation and statistics.

## Build the pair list

[`image_pairs`](@ref) groups an ordered frame list (file paths and/or
in-memory matrices) into correlation pairs. Build the list by filtering a
directory with `readdir` and Julia's predicates:

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

Check the first and last few pairs before processing. `readdir` sorts names
lexically, so filenames such as `frame_1.tif`, `frame_10.tif`, and
`frame_2.tif` are out of acquisition order unless their indices are padded
or you sort them numerically.

For sampled or multi-delay recordings, use `stride`, zero-based `offset`, and
one or more `deltas` (frame-index differences):

```julia
pairs = image_pairs(files; mode=:chained, stride=2, offset=1, deltas=(1, 4))
```

`FrameSource(n, i -> read_vendor_frame(i))` adapts a proprietary CINE, MRAW,
SEQ, video, or camera reader without a package dependency. Pair construction
does not read frames. Optional `timestamps` travel with the pairs; when a
`PhysicalScale` is passed to the sequence driver, each result receives the
pair's actual `dt`. Use `TIFFStack("recording.tif")` for multi-page TIFFs.

For shell-style `*` matching across nested directories, you can use
[Glob.jl](https://github.com/vtjnash/Glob.jl) (`Glob.glob("cam1/*.tif")`).
Sort and inspect its output before pairing, too.

## Run the sequence with incremental output

```julia
passes = multipass_parameters([64, 32, 16]; padding = true, apodization = :gauss)
results = run_piv_sequence(pairs, passes;
    preprocess = img -> intensity_cap!(img),
    output = "run_042_piv.jld2",
    mask = mask,                  # optional, shared by all pairs
)
```

To keep these settings for another recording, put them in a recipe and run it
with `apply_recipe`, which also stores the settings in the output file; see
[Save settings and reuse them](recipes.md).

With `output` set, each result is written to the JLD2-format Julia data file
**as it completes**. If a later pair fails, the file keeps the finished
prefix. For file-path pairs
the source image paths are stored alongside the results. Threading applies
within each pair (the window grid is split across tasks); results are
bitwise identical to serial processing.

For file-path pairs, `image_type = Float32` loads frames in single precision
and reduces memory traffic. In-memory matrices keep their existing element
type. See the [precision policy](../explanation/precision.md).

For synchronized stereo recordings, pass one pair list per camera and reuse
the calibrated dewarpers across the run:

```julia
stop_requested = Ref(false)
stereo = run_piv_stereo_sequence(cam1_pairs, cam2_pairs, dw1, dw2, passes;
    preprocess = (preprocess_cam1, preprocess_cam2),
    output = "run_042_stereo.jld2",
    progress = true,
    cancel = () -> stop_requested[],
)
```

Change `stop_requested[]` to `true` from your control code to stop between
acquisitions. For a blocking script
without a stop control, omit `cancel`.

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

[`load_results`](@ref) returns result entries in sequence order; planar,
stereo, PTV, and tracking results can share a file. An empty saved vector, or
a batch stopped before its first result, loads as an empty vector. Files with
an unknown `format_version` raise an error. Stored source paths, when present,
are retrievable directly with JLD2:

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
`TABLE_COLUMNS`; planar, stereo, PTV, and tracking rows use the same superset
and leave inapplicable values empty. Identifiers, flags, quality values, uncertainties,
and unit strings are included, and attached scaling is applied. Tracking rows
preserve observed frame indices and gaps, with derived elapsed time and
numerical validity flags; see the [I/O reference](../reference/io.md). Planar and stereo grids can also be written for
ParaView with
`export_vtk("field.vtk", result)` (legacy ASCII structured-grid VTK).
The VTK `FIELD` metadata records coordinate and vector-component unit labels;
without an attached scale, stereo labels use `world_unit` for the calibration
grid's unnamed world units.

## Post-process the sequence

### Consume results without collecting the recording in RAM

All three sequence drivers (`run_piv_sequence`, `run_ptv_sequence`, and
`run_piv_stereo_sequence`) accept `collect_results = false`. They return
`nothing`; completed results still reach the `on_result` callback and output
file before progress is reported. The driver then releases its reference to
that result. For example, save a large planar run while keeping only a small
summary in memory:

```julia
accepted_counts = Int[]
run_piv_sequence(pairs, passes;
    collect_results = false,
    output = "large_run.jld2",
    on_result = (i, r) -> push!(accepted_counts, count(.!(r.mask .| r.outliers))))
```

Use a callback that processes each result without retaining it for constant
result-storage memory. Input lists, accumulated summaries, output-file metadata,
workspaces, and memory retained by the callback have their own costs. Loading
the saved file with the default `load_results(path)` materializes the entire
result vector; `load_results(path; lazy = true)` indexes a completed file
without retaining its payloads. See [completed-file GUI browsing](gui.md).

Exceptions and prefetch cleanup follow the same contract as collecting runs.
Stereo cancellation returns `nothing` in this mode; completed acquisitions
remain available through the callback or configured output.

### Accumulate field statistics as results arrive

[`FieldStatisticsAccumulator`](@ref) keeps only grid-sized online moments
and valid counts. It accepts planar and stereo results, and returns the
same statistics as `field_statistics(results)` when finalized:

```julia
acc = FieldStatisticsAccumulator()
run_piv_sequence(pairs, passes;
    collect_results = false,
    on_result = (i, r) -> update_statistics!(acc, r))
stats = field_statistics(acc)  # independent means/RMS/stresses/counts snapshot
```

The first update establishes the grid and result kind. Later updates must
have matching coordinates, component/mask/outlier dimensions, and attached
scale factors and unit labels; these checks happen before changing the
accumulator. An incompatible field leaves the completed prefix's statistics
intact. Finalizing before the first update raises an error. Snapshots do not
reset the accumulator and share no mutable arrays with it, so you can inspect
progress snapshots during processing.

Statistics use the stored components. To calculate velocities, supply a
scale to the driver and update with `physical(r)`:

```julia
acc = FieldStatisticsAccumulator()
run_piv_sequence(pairs, passes;
    scale = PhysicalScale(pixel_size = 0.02, dt = 0.001,
                          length_unit = "mm", time_unit = "s"),
    collect_results = false,
    on_result = (i, r) -> update_statistics!(acc, physical(r)))
velocity_stats = field_statistics(acc)
```

Conversion before updating also supports pair-specific delays, provided
the resulting physical grid and unit labels agree. The accumulator checks
scale compatibility more strictly than the vector statistics API; it does
not combine absent scales with attached ones or different numeric factors.
Masks and nonfinite components always exclude a sample. Outliers are
excluded by default; construct with `include_invalid = true` to include
finite flagged vectors. Each node has its own valid count, and an unsampled
node has `NaN` moments. Fluctuation RMS includes measurement noise. Use the same callback pattern
with `run_piv_stereo_sequence`; one accumulator must hold only one result
kind. Callbacks run serially, which suits the accumulator's serial updates.

### Replay a completed file without collecting its fields

[`ResultFile`](@ref), also returned by `load_results(path; lazy = true)`,
loads one result per indexed access or iteration and closes the file after
each read. For a completed file containing same-grid planar or stereo fields:

```julia
saved = load_results("large_run.jld2"; lazy = true) # or ResultFile("large_run.jld2")
replay = FieldStatisticsAccumulator()
for r in saved
    update_statistics!(replay, physical(r))
end
velocity_stats = field_statistics(replay)
```

This retains only the grid-sized accumulator and the field currently being
read, plus O(number of results) entry-key metadata; it does not retain a
payload for every saved result. Open the index after the writer closes,
and keep the file unchanged while reading. For interactive access,
see [completed-file GUI browsing](gui.md).

### Analyze collected results

For a worked example of sampling intervals, valid counts, and convergence
of a mean field, follow [From image pairs to flow statistics](../tutorials/sequence_statistics.md).

Sequences of same-grid results feed the statistics utilities directly:

```julia
validate_temporal!(results)                  # flag temporal outliers in memory
stats = field_statistics(results)            # mean/RMS/stresses/counts on valid vectors
spectrum = result_spectrum(results, 12, 8; dt = 1/10_000)
```

For this spectrum, `dt` is the time between successive velocity results,
not the delay within an image pair. [`result_spectrum`](@ref) checks masks
and outlier flags and errors on invalid samples by default; choose an
explicit `invalid` policy only if filling those samples is justified.
`validate_temporal!` changes the in-memory flags. Save the checked results
to a new file if you need those flags later:

```julia
save_results("run_042_checked.jld2", results)
```

## A note on failure

If a pair fails (unreadable file, size mismatch), `run_piv_sequence` logs
which pair and rethrows — the incremental output file retains everything
processed up to that point, so you can fix the offending frame and resume
from a trimmed pair list. The driver does not automatically resume or append
to an existing output file, so choose a new output path for the remaining
pairs and keep the original completed prefix.
The driver waits for any started frame prefetch before it returns, including
when processing or a callback fails.
