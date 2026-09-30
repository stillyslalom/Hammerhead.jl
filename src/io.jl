# File I/O and batch processing: image loading via FileIO/ImageIO, result
# serialization via JLD2, and a sequence driver over image-pair lists.

"""
    load_image(path)          -> Matrix{Float64}
    load_image(T, path)       -> Matrix{T}

Read one image as a grayscale matrix for [`run_piv`](@ref) or preprocessing.
Color images are converted to grayscale. Normalized fixed-point pixels from
the image decoder convert to values in `[0, 1]`; numeric matrices are converted
directly to `T`. No per-image contrast normalization is applied.

The element type defaults to `Float64`; call `load_image(Float32, path)`
for a `Float32` matrix and single-precision image processing.

Formats are dispatched through FileIO: TIFF (including 16-bit) and PNG are
supported by the installed image decoders. The decoded image must be a 2D
matrix. Use [`TIFFStack`](@ref) for a multi-page recording.
"""
function load_image(::Type{T}, path::AbstractString) where {T<:AbstractFloat}
    isfile(path) || throw(ArgumentError("no such image file: $path"))
    return image_to_matrix(T, _load_raw(FileIO.query(path)), path)
end

load_image(path::AbstractString) = load_image(Float64, path)

# TIFFs are loaded through a Stream: TiffImages' path-based entry point runs a
# full GC.gc() before every open on Windows (to release handles of previously
# mmapped TIFFs — we never mmap), costing a full-heap pause per frame.
_load_raw(f::FileIO.File{FileIO.DataFormat{:TIFF}}) = open(FileIO.filename(f)) do io
    FileIO.load(FileIO.Stream{FileIO.DataFormat{:TIFF}}(io, FileIO.filename(f)))
end
_load_raw(f::FileIO.File) = FileIO.load(f)

image_to_matrix(::Type{T}, img::AbstractMatrix{<:Real}, path) where {T} = T.(img)
image_to_matrix(::Type{T}, img::AbstractMatrix{<:Colorant}, path) where {T} = T.(Gray.(img))
image_to_matrix(::Type, img, path) =
    throw(ArgumentError("$path did not load as a single 2D image " *
                        "(got $(summary(img))); load and slice it manually"))

"""
    load_mask(path; threshold = 0.5, invert = false) -> BitMatrix

Load an analysis mask from an image file: pixels with grayscale value
`>= threshold` become `true` (excluded from analysis; see `mask` in
[`run_piv`](@ref)). Use `invert = true` when dark pixels mark the excluded
region instead.
"""
function load_mask(path::AbstractString; threshold::Real = 0.5, invert::Bool = false)
    img = load_image(path)
    return invert ? BitMatrix(img .< threshold) : BitMatrix(img .>= threshold)
end

"""
    image_pairs(files; mode=:paired, stride=1, offset=0, deltas=(1,)) -> Vector{Tuple}

Group an ordered list of frame paths or matrices into pairs for
[`run_piv_sequence`](@ref). This function does not sort or load frames;
verify acquisition order before calling it.

- `mode = :paired` selects start indices `1 + offset:2*stride:length(files)`.
  The defaults produce `(1, 2), (3, 4), …` and require an even frame count.
- `mode = :chained` selects start indices `1 + offset:stride:length(files)`.
  The defaults produce `(1, 2), (2, 3), …` and require at least two frames.
- `offset` is a nonnegative count of leading frames to skip; `stride` is
  positive. `deltas` is a positive integer or collection of frame separations.
  Pairs are grouped by delta, then start index; incomplete pairs are omitted.

For example, `mode=:chained, stride=2, deltas=(1, 3)` first pairs frames
1 and 2, 3 and 4, and so on, then frames 1 and 4, 3 and 6, and so on.
See [`FrameSource`](@ref) for lazy loading and per-pair timestamps.
"""
function image_pairs(files::AbstractVector; mode::Symbol = :paired,
                     stride::Integer = 1, offset::Integer = 0, deltas = (1,))
    # Preserve the original strict paired-mode diagnostic for the default.
    if mode === :paired && stride == 1 && offset == 0 && deltas == (1,) && isodd(length(files))
        throw(ArgumentError("mode = :paired requires an even number of frames, got $(length(files))"))
    end
    mode === :chained && length(files) < 2 &&
        throw(ArgumentError("mode = :chained requires at least 2 frames, got $(length(files))"))
    return [(files[a], files[b]) for (a,b) in
            _pair_indices(length(files); mode, stride, offset, deltas)]
end

"""
    frame_index_strings(pathA, pathB) -> (strA, strB)

Extract frame indices from two differing path stems, without their
directories or extensions. Leading zeros in an index are preserved.
Use the returned strings to name per-pair output files; see `output` in
[`run_piv_sequence`](@ref).

```jldoctest
julia> frame_index_strings("path/to/img_0001.tif", "path/to/img_0002.tif")
("0001", "0002")

julia> frame_index_strings("a/f_099.png", "a/f_100.png")
("099", "100")
```

Throws an `ArgumentError` if the two stems are identical (no differing field).
"""
function frame_index_strings(pathA::AbstractString, pathB::AbstractString)
    a = collect(splitext(basename(String(pathA)))[1])
    b = collect(splitext(basename(String(pathB)))[1])
    la, lb = length(a), length(b)
    p = 0                                        # longest common prefix
    while p < min(la, lb) && a[p + 1] == b[p + 1]
        p += 1
    end
    s = 0                                        # longest common suffix (past the prefix)
    while s < min(la, lb) - p && a[la - s] == b[lb - s]
        s += 1
    end
    # Don't split a numeric field: pull shared digits adjacent to the differing
    # region out of the common prefix/suffix and back into the returned middles.
    while p >= 1 && isdigit(a[p]) &&
          ((p < la - s && isdigit(a[p + 1])) || (p < lb - s && isdigit(b[p + 1])))
        p -= 1
    end
    while s >= 1 && isdigit(a[la - s + 1]) &&
          ((la - s > p && isdigit(a[la - s])) || (lb - s > p && isdigit(b[lb - s])))
        s -= 1
    end
    midA = String(a[(p + 1):(la - s)])
    midB = String(b[(p + 1):(lb - s)])
    (isempty(midA) || isempty(midB)) &&
        throw(ArgumentError("no differing frame index between \"$pathA\" and \"$pathB\" " *
                            "(identical stems)"))
    return (midA, midB)
end

# Version 1 is the first public format: results/000001… (PIVResult,
# StereoPIVResult, or PTVResult entries, each carrying its optional
# PhysicalScale) plus optional sources/…. The pre-registration development
# formats (old versions 1–5, whose result structs lacked the scale field)
# were retired without a load shim when the scale field landed — no files
# existed outside the test suite.
const RESULTS_FORMAT_VERSION = 1

result_key(i::Integer) = "results/" * lpad(i, 6, '0')
source_key(i::Integer) = "sources/" * lpad(i, 6, '0')

"""
    save_results(path, results) -> path

Write one result or a vector of results to a JLD2 file at `path`, replacing
an existing file. The vector may mix [`PIVResult`](@ref),
[`StereoPIVResult`](@ref), [`PTVResult`](@ref), and
[`TrackingResult`](@ref). Read it with [`load_results`](@ref).
"""
function save_results(path::AbstractString,
                      results::AbstractVector{<:Union{PIVResult,StereoPIVResult,PTVResult,TrackingResult}})
    jldopen(path, "w") do f
        f["format_version"] = RESULTS_FORMAT_VERSION
        for (i, r) in enumerate(results)
            f[result_key(i)] = r
        end
    end
    return path
end

save_results(path::AbstractString, result::Union{PIVResult,StereoPIVResult,PTVResult,TrackingResult}) =
    save_results(path, [result])

"""
    load_results(path) -> Vector

Read results in saved order from a Hammerhead JLD2 file. The returned
vector may contain planar PIV, stereo PIV, PTV, and tracking results.
An empty saved sequence returns an empty vector. Unknown file-format
versions raise an `ArgumentError`.

Sequence drivers also save source labels when available; this function
returns result objects only. To inspect the first saved pair's labels, load
`"sources/000001"` from the same file with JLD2.
"""
function load_results(path::AbstractString)
    jldopen(path, "r") do f
        haskey(f, "format_version") ||
            throw(ArgumentError("$path has no format_version; not a Hammerhead results file"))
        version = f["format_version"]
        version == RESULTS_FORMAT_VERSION ||
            throw(ArgumentError("$path has unsupported format_version $version (supported: $RESULTS_FORMAT_VERSION)"))
        # An empty save or a batch stopped before its first result has no
        # `results` group: JLD2 creates groups only when a child is written.
        haskey(f, "results") || return PIVResult[]
        g = f["results"]
        return [g[k] for k in sort!(keys(g))]
    end
end

"""
    run_piv_sequence(pairs, params = PIVParameters();
                     preprocess = nothing, output = nothing,
                     progress = true, backend = :cpu, kwargs...) -> Vector{PIVResult}
    run_piv_sequence(pairs; effort = :low/:medium/:high, kwargs...) -> Vector{PIVResult}

Analyze image pairs in order and return one [`PIVResult`](@ref) per pair.
`pairs` is a nonempty vector of path or matrix 2-tuples, or lazy
[`FramePair`](@ref)s. Use [`image_pairs`](@ref) to build it.
`params` is a single `PIVParameters` or a multi-pass schedule; alternatively,
omit it and pass `effort = :low`, `:medium`, or `:high` to use the built-in
effort schedules from [`run_piv`](@ref). Remaining `kwargs`, including `roi`
and parameter overrides when `effort` is set, go to [`run_piv`](@ref).

- `preprocess`: function applied to each frame after loading, e.g.
  `img -> clahe!(subtract_background!(img, bg))`. Frames loaded from file
  paths are fresh buffers, so mutating preprocessors are safe; in-memory
  matrix pairs are passed through as-is; use allocating preprocessors there
  to leave the caller's arrays untouched.
- `output`: either one JLD2 path (overwritten) receiving completed results
  incrementally, or a function `(i, pair) -> outpath` mapping a 1-based pair
  index and the original pair to a per-pair output path
  (each written as its own single-result JLD2 file as that pair completes;
  parent directories are created). For file-path pairs the source paths are
  stored alongside in either mode. Read any of these back with
  [`load_results`](@ref); see [`frame_index_strings`](@ref) for building
  per-pair names from the frame paths.
- `progress`: show a progress meter (`true`/`false`), or a function
  `(i, n) -> nothing` called after each completed pair (for driving an
  external progress display). Throwing from the callback aborts the batch;
  pairs finished before the abort stay in `output`.
- `on_result`: optional function `(i, result) -> nothing` called with each
  pair's result immediately after it completes, before `output` and
  `progress`. Use it to consume results during a batch. Runs on the calling
  task, in pair order. Throwing aborts the batch like a throwing `progress`
  callback; pairs already persisted stay in `output`.
- `backend`: execution backend selector. The core provides `:cpu` and `:ka`;
  package extensions add device selectors (see [Run PIV on a GPU](@ref)).
- `image_type`: element type frames are loaded as (default `Float64`);
  `Float32` selects single precision for the main image-processing buffers.
  In-memory matrices and frame-source outputs keep their existing types.
- `mask`: a shared exclusion mask, a mask per pair, or a callback
  `(i, imgA, imgB) -> mask` applied to the loaded, preprocessed images.
  A returned `(maskA, maskB)` tuple excludes the union. Lazy frame pairs also
  accept one mask per source frame.
- `scale`: optional [`PhysicalScale`](@ref). For timestamped `FramePair`s,
  their `dt` replaces the supplied scale's delay; timestamps alone do not
  attach a physical scale.

Pairs are analyzed serially while the next pair is loaded and preprocessed
on a background task. Preprocessors must be safe to call from that task.
A shared [`PIVWorkspace`](@ref) reuses buffers across pairs. If processing or
a callback throws, the exception propagates after pending loading finishes;
results already written to `output` are retained.
"""
function run_piv_sequence(pairs::AbstractVector,
                          params::Union{PIVParameters,AbstractVector{PIVParameters}};
                          effort::Union{Nothing,Symbol} = nothing,
                          backend::Symbol = :cpu,
                          preprocess = nothing,
                          output::Union{Nothing,AbstractString,Function} = nothing,
                          progress::Union{Bool,Function} = true,
                          on_result::Union{Nothing,Function} = nothing,
                          image_type::Type{<:AbstractFloat} = Float64,
                          mask = nothing,
                          scale::Union{Nothing,PhysicalScale} = nothing,
                          kwargs...)
    effort === nothing ||
        throw(ArgumentError("effort cannot be combined with explicit PIVParameters or pass schedules"))
    workspace = piv_workspace(; backend)
    _run_sequence((imgA, imgB, i, pair, mask, scale) -> run_piv(imgA, imgB, params; backend, workspace, mask, scale, kwargs...),
                  PIVResult, pairs;
                  preprocess, output, progress, on_result, image_type, mask, scale, label = "PIV")
end

function run_piv_sequence(pairs::AbstractVector; effort::Union{Nothing,Symbol} = nothing,
                          backend::Symbol = :cpu,
                          preprocess = nothing,
                          output::Union{Nothing,AbstractString,Function} = nothing,
                          progress::Union{Bool,Function} = true,
                          on_result::Union{Nothing,Function} = nothing,
                          image_type::Type{<:AbstractFloat} = Float64,
                          mask = nothing,
                          scale::Union{Nothing,PhysicalScale} = nothing,
                          kwargs...)
    workspace = piv_workspace(; backend)
    process = effort === nothing ?
        ((imgA, imgB, i, pair, mask, scale) -> run_piv(imgA, imgB, PIVParameters(); backend, workspace, mask, scale, kwargs...)) :
        ((imgA, imgB, i, pair, mask, scale) -> run_piv(imgA, imgB; effort, backend, workspace, mask, scale, kwargs...))
    _run_sequence(process, PIVResult, pairs;
                  preprocess, output, progress, on_result, image_type, mask, scale, label = "PIV")
end

"""
    run_ptv_sequence(pairs, params = PTVParameters();
                     preprocess = nothing, output = nothing,
                     progress = true, kwargs...) -> Vector{PTVResult}

Analyze image pairs in order with [`run_ptv`](@ref), returning one
[`PTVResult`](@ref) per pair. Pairs may contain file paths or matrices.
`params` controls detection and matching. `preprocess`, `output`,
`progress`, `on_result`, and `image_type` follow
[`run_piv_sequence`](@ref); remaining keywords such as `predictor`,
`mask`, and `scale` go to `run_ptv`. If `output` is supplied, completed
results are saved incrementally for [`load_results`](@ref).
"""
function run_ptv_sequence(pairs::AbstractVector, params::PTVParameters = PTVParameters();
                          preprocess = nothing,
                          output::Union{Nothing,AbstractString,Function} = nothing,
                          progress::Union{Bool,Function} = true,
                          on_result::Union{Nothing,Function} = nothing,
                          image_type::Type{<:AbstractFloat} = Float64,
                          mask = nothing,
                          scale::Union{Nothing,PhysicalScale} = nothing,
                          kwargs...)
    _run_sequence((imgA, imgB, i, pair, mask, scale) -> run_ptv(imgA, imgB, params; mask, scale, kwargs...), PTVResult, pairs;
                  preprocess, output, progress, on_result, image_type, mask, scale, label = "PTV")
end

# Shared sequence driver: iterate `pairs`, load/preprocess each frame, run
# `process(imgA, imgB)` (a PIV or PTV closure), persist incrementally, and
# log-and-rethrow per pair. `R` is the result element type; `label` names the
# analysis in progress/error messages. `output` is either a single JLD2 path
# (all results in one file) or a function `(i, pair) -> outpath` (one
# single-result file per pair); both write incrementally as pairs complete.
#
# The next pair's load+preprocess runs on a background task (`load_pair`) while
# the current pair's `process` runs, so slow-source IO (network/disk) overlaps
# compute. Results stay bitwise identical to a serial run: same `process` calls
# in the same order on the same images. The overlap only materializes with ≥2
# threads — `process` is CPU-bound with no yield points, so under `-t 1` the
# prefetch task cannot run until `process` returns.
function _run_sequence(process, ::Type{R}, pairs::AbstractVector;
                       preprocess = nothing,
                       output::Union{Nothing,AbstractString,Function} = nothing,
                       progress::Union{Bool,Function} = true,
                       on_result::Union{Nothing,Function} = nothing,
                       image_type::Type{<:AbstractFloat} = Float64,
                       label::AbstractString = "PIV",
                       mask = nothing,
                       scale = nothing) where {R}
    isempty(pairs) && throw(ArgumentError("pairs must not be empty"))
    results = Vector{R}(undef, length(pairs))
    file = output isa AbstractString ? jldopen(output, "w") : nothing
    load_pair(pair) = Threads.@spawn begin
        imgA = load_frame(pair[1], image_type)
        imgB = load_frame(pair[2], image_type)
        preprocess === nothing ? (imgA, imgB) : (preprocess(imgA), preprocess(imgB))
    end
    pending = nothing
    failed = false
    try
        file === nothing || (file["format_version"] = RESULTS_FORMAT_VERSION)
        meter = Progress(length(pairs); desc = "$label sequence: ", enabled = progress === true)
        pending = load_pair(pairs[1])
        for (i, pair) in enumerate(pairs)
            frameA, frameB = pair
            try
                imgA, imgB = fetch_frames(pending)
                i < length(pairs) && (pending = load_pair(pairs[i + 1]))
                pmask = pair_mask(mask, i, pair, imgA, imgB)
                results[i] = process(imgA, imgB, i, pair, pmask, pair_scale(scale, pair))
            catch
                @error "$label sequence failed on pair $i of $(length(pairs))" frameA = frame_label(frameA) frameB = frame_label(frameB)
                rethrow()
            end
            # Live-consumer hook. `results[i]` is stored on this (serial)
            # task — the prefetch task only loads frames — so the callback
            # runs on the caller's task, in pair order.
            on_result === nothing || on_result(i, results[i])
            if file !== nothing
                file[result_key(i)] = results[i]
                labels = pair_source_labels(frameA, frameB)
                if labels !== nothing
                    file[source_key(i)] = labels
                end
            elseif output isa Function
                write_pair_file(String(output(i, pair)), results[i], frameA, frameB)
            end
            progress isa Function ? progress(i, length(pairs)) : next!(meter)
        end
    catch
        failed = true
        rethrow()
    finally
        try
            if pending !== nothing
                try
                    fetch_frames(pending)
                catch
                    failed || rethrow()
                end
            end
        finally
            if file !== nothing
                try
                    close(file)
                catch
                    failed || rethrow()
                end
            end
        end
    end
    return results
end

# One-result-per-file writer for the function-`output` sequence mode: a
# standalone results file (readable by `load_results`) recording the pair's
# source paths when the pair entries are file paths.
function write_pair_file(path::AbstractString, result, frameA, frameB)
    dir = dirname(path)
    isempty(dir) || mkpath(dir)
    jldopen(path, "w") do f
        f["format_version"] = RESULTS_FORMAT_VERSION
        f[result_key(1)] = result
        labels = pair_source_labels(frameA, frameB)
        if labels !== nothing
            f[source_key(1)] = labels
        end
    end
    return path
end

# Await a prefetch task, unwrapping a failed load so the original exception
# (e.g. the `ArgumentError` from a missing file) propagates rather than the
# `TaskFailedException` wrapper `fetch` would otherwise raise.
function fetch_frames(task)
    try
        return fetch(task)
    catch err
        err isa TaskFailedException ? rethrow(err.task.exception) : rethrow()
    end
end

frame_label(x::AbstractString) = x
frame_label(x) = summary(x)
source_label(x::AbstractString) = String(x)
source_label(x) = nothing
function pair_source_labels(a, b)
    la, lb = source_label(a), source_label(b)
    la === nothing || lb === nothing ? nothing : [la, lb]
end

load_frame(x::AbstractString, ::Type{T}) where {T} = load_image(T, x)
load_frame(x::AbstractMatrix{<:Real}, ::Type) = x
load_frame(x, ::Type) =
    throw(ArgumentError("sequence entries must be file paths or real-valued matrices, got $(typeof(x))"))

function pair_mask(spec, i, pair, imgA, imgB)
    spec === nothing && return nothing
    value = if spec isa Function
        spec(i, imgA, imgB)
    elseif spec isa AbstractMatrix{Bool}
        spec
    elseif spec isa AbstractVector
        # A vector may describe pairs directly, or frames when a FramePair
        # exposes source indices. Frame-wise masks are unioned across A/B.
        a, b = pair[1], pair[2]
        if hasproperty(a, :index) && hasproperty(b, :index) &&
           hasproperty(a, :source) && length(spec) == length(getproperty(a, :source))
            (spec[getproperty(a, :index)], spec[getproperty(b, :index)])
        else
            spec[i]
        end
    else
        spec
    end
    if value isa Tuple && length(value) == 2
        a, b = value
        size(a) == size(b) || throw(DimensionMismatch("pair masks must have equal size"))
        return BitMatrix(a .| b)
    end
    value isa AbstractMatrix{Bool} ||
        throw(ArgumentError("mask must be a Bool matrix, a sequence, a callback, or a pair of masks"))
    return value
end

function pair_scale(scale, pair)
    scale === nothing && return nothing
    dt = hasproperty(pair, :dt) ? getproperty(pair, :dt) : nothing
    dt === nothing && return scale
    PhysicalScale(scale.pixel_size, dt, scale.length_unit, scale.time_unit)
end
