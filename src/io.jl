# File I/O and batch processing: image loading via FileIO/ImageIO, result
# serialization via JLD2, and a sequence driver over image-pair lists.

"""
    load_image(path)          -> Matrix{Float64}
    load_image(T, path)       -> Matrix{T}

Load an image file as a grayscale floating-point matrix, ready for
[`run_piv`](@ref) or the preprocessing functions. Color images are converted
to grayscale; integer-valued images are scaled by their type's full range
(e.g. 16-bit TIFF values map to `[0, 1]` as `n / 65535`).

The element type defaults to `Float64`; pass `T = Float32` to run the
downstream pipeline in single precision (see [`PIVResult`](@ref)).

Formats are dispatched through FileIO: TIFF (including 16-bit) and PNG are
supported out of the box; BMP and other exotic formats work when ImageMagick
is installed (`using ImageMagick`).
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
`>= threshold` become `true` (excluded from analysis — see `mask` in
[`run_piv`](@ref)). Use `invert = true` when dark pixels mark the excluded
region instead.
"""
function load_mask(path::AbstractString; threshold::Real = 0.5, invert::Bool = false)
    img = load_image(path)
    return invert ? BitMatrix(img .< threshold) : BitMatrix(img .>= threshold)
end

"""
    image_pairs(files; mode = :paired) -> Vector{Tuple}

Group an ordered list of frames (file paths or matrices) into correlation
pairs for [`run_piv_sequence`](@ref):

- `mode = :paired` — double-frame recordings: `(1, 2), (3, 4), …`
  (requires an even number of frames).
- `mode = :chained` — time-resolved recordings: `(1, 2), (2, 3), …`
"""
function image_pairs(files::AbstractVector; mode::Symbol = :paired)
    if mode === :paired
        iseven(length(files)) ||
            throw(ArgumentError("mode = :paired requires an even number of frames, got $(length(files))"))
        return [(files[i], files[i + 1]) for i in 1:2:(length(files) - 1)]
    elseif mode === :chained
        length(files) >= 2 ||
            throw(ArgumentError("mode = :chained requires at least 2 frames, got $(length(files))"))
        return [(files[i], files[i + 1]) for i in 1:(length(files) - 1)]
    else
        throw(ArgumentError("mode must be :paired or :chained, got :$mode"))
    end
end

"""
    frame_index_strings(pathA, pathB) -> (strA, strB)

Extract the differing frame-index substrings from a pair of frame paths:
strip each path's directory and extension, then return the portions of the
two stems that differ, with any shared leading/trailing **digits** of that
field kept intact (so a zero-padded index is not truncated). Handy for
naming per-pair output files — see the `output` keyword of
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

# Version 2 added the uncertainty_u/uncertainty_v fields to PIVResult;
# version 3 allows StereoPIVResult entries alongside PIVResult;
# version 4 allows PTVResult entries;
# version 5 added the max_iterations/convergence_tol fields to the embedded
# PIVParameters.
const RESULTS_FORMAT_VERSION = 5

result_key(i::Integer) = "results/" * lpad(i, 6, '0')
source_key(i::Integer) = "sources/" * lpad(i, 6, '0')

"""
    save_results(path, results) -> path

Save a [`PIVResult`](@ref), [`StereoPIVResult`](@ref), or [`PTVResult`](@ref)
(or a vector of them, possibly mixed) to a JLD2 file (conventionally `*.jld2`),
overwriting `path` if it exists. Read back with [`load_results`](@ref).
"""
function save_results(path::AbstractString,
                      results::AbstractVector{<:Union{PIVResult,StereoPIVResult,PTVResult}})
    jldopen(path, "w") do f
        f["format_version"] = RESULTS_FORMAT_VERSION
        for (i, r) in enumerate(results)
            f[result_key(i)] = r
        end
    end
    return path
end

save_results(path::AbstractString, result::Union{PIVResult,StereoPIVResult,PTVResult}) =
    save_results(path, [result])

"""
    load_results(path) -> Vector

Load the results (`PIVResult` and/or `StereoPIVResult` entries) stored in a
JLD2 file written by [`save_results`](@ref) or [`run_piv_sequence`](@ref), in
sequence order. Files written by `run_piv_sequence(...; output = path)` from
file-path pairs also carry the source image paths, retrievable with
`JLD2.load(path, "sources/000001")` etc.
"""
function load_results(path::AbstractString)
    jldopen(path, "r") do f
        haskey(f, "results") ||
            throw(ArgumentError("$path has no \"results\" group; not a Hammerhead results file"))
        g = f["results"]
        return [g[k] for k in sort!(keys(g))]
    end
end

"""
    run_piv_sequence(pairs, params = PIVParameters();
                     preprocess = nothing, output = nothing,
                     progress = true, kwargs...) -> Vector{PIVResult}
    run_piv_sequence(pairs; effort = :low/:medium/:high, kwargs...) -> Vector{PIVResult}

Run PIV over a sequence of image pairs. `pairs` is a vector of 2-tuples whose
entries are file paths (loaded with [`load_image`](@ref)) and/or real-valued
matrices — see [`image_pairs`](@ref) for building it from a frame list.
`params` is a single `PIVParameters` or a multi-pass schedule; alternatively,
omit it and pass `effort = :low`, `:medium`, or `:high` to use the built-in
effort schedules from [`run_piv`](@ref). Remaining `kwargs` (e.g. `mask` for a
static analysis mask shared by all pairs, or PIV-parameter overrides when
`effort` is set) are forwarded to [`run_piv`](@ref).

- `preprocess`: function applied to each frame after loading, e.g.
  `img -> clahe!(subtract_background!(img, bg))`. Frames loaded from file
  paths are fresh buffers, so mutating preprocessors are safe; in-memory
  matrix pairs are passed through as-is — use the allocating versions there
  to leave the caller's arrays untouched.
- `output`: either a path of a single JLD2 file (overwritten) to which all
  results are written incrementally as they complete — so a crashed batch
  keeps its finished pairs — or a function `(i, pair) -> outpath` mapping the
  1-based pair index and the original pair tuple to a per-pair output path
  (each written as its own single-result JLD2 file as that pair completes;
  parent directories are created). For file-path pairs the source paths are
  stored alongside in either mode. Read any of these back with
  [`load_results`](@ref); see [`frame_index_strings`](@ref) for building
  per-pair names from the frame paths.
- `progress`: show a progress meter (`true`/`false`), or a function
  `(i, n) -> nothing` called after each completed pair (for driving an
  external progress display). Throwing from the callback aborts the batch;
  pairs finished before the abort stay in `output`.
- `image_type`: element type frames are loaded as (default `Float64`);
  `Float32` runs the pipeline in single precision. In-memory matrix pairs are
  used as-is, so convert those yourself.

The interpolant, deformation, and correlator scratch is reused across pairs via
a single [`PIVWorkspace`](@ref) (the pairs share an image size), so a batch pays
those allocations once; results are bitwise identical to calling [`run_piv`](@ref)
per pair. Because pairs are processed serially — while the *next* pair's frames
are prefetched on a background task that never touches the workspace — this stays
race-free.
"""
function run_piv_sequence(pairs::AbstractVector,
                          params::Union{PIVParameters,AbstractVector{PIVParameters}};
                          effort::Union{Nothing,Symbol} = nothing,
                          preprocess = nothing,
                          output::Union{Nothing,AbstractString,Function} = nothing,
                          progress::Union{Bool,Function} = true,
                          image_type::Type{<:AbstractFloat} = Float64,
                          kwargs...)
    effort === nothing ||
        throw(ArgumentError("effort cannot be combined with explicit PIVParameters or pass schedules"))
    workspace = piv_workspace()
    _run_sequence((imgA, imgB) -> run_piv(imgA, imgB, params; workspace, kwargs...),
                  PIVResult, pairs;
                  preprocess, output, progress, image_type, label = "PIV")
end

function run_piv_sequence(pairs::AbstractVector; effort::Union{Nothing,Symbol} = nothing,
                          preprocess = nothing,
                          output::Union{Nothing,AbstractString,Function} = nothing,
                          progress::Union{Bool,Function} = true,
                          image_type::Type{<:AbstractFloat} = Float64,
                          kwargs...)
    workspace = piv_workspace()
    process = effort === nothing ?
        ((imgA, imgB) -> run_piv(imgA, imgB, PIVParameters(); workspace, kwargs...)) :
        ((imgA, imgB) -> run_piv(imgA, imgB; effort, workspace, kwargs...))
    _run_sequence(process, PIVResult, pairs;
                  preprocess, output, progress, image_type, label = "PIV")
end

"""
    run_ptv_sequence(pairs, params = PTVParameters();
                     preprocess = nothing, output = nothing,
                     progress = true, kwargs...) -> Vector{PTVResult}

Run PTV over a sequence of image pairs, mirroring [`run_piv_sequence`](@ref):
`pairs` entries are file paths (loaded with [`load_image`](@ref)) and/or
real-valued matrices, `params` is a [`PTVParameters`](@ref), and the same
`preprocess`, `output` (incremental JLD2), `progress`, and `image_type`
options apply. Remaining `kwargs` (e.g. `predictor` or `mask`) are forwarded to
[`run_ptv`](@ref). Results are [`PTVResult`](@ref)s; when `output` is a path
they are persisted incrementally as they complete (with source paths for
file-path pairs), readable with [`load_results`](@ref).
"""
function run_ptv_sequence(pairs::AbstractVector, params::PTVParameters = PTVParameters();
                          preprocess = nothing,
                          output::Union{Nothing,AbstractString,Function} = nothing,
                          progress::Union{Bool,Function} = true,
                          image_type::Type{<:AbstractFloat} = Float64,
                          kwargs...)
    _run_sequence((imgA, imgB) -> run_ptv(imgA, imgB, params; kwargs...), PTVResult, pairs;
                  preprocess, output, progress, image_type, label = "PTV")
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
                       image_type::Type{<:AbstractFloat} = Float64,
                       label::AbstractString = "PIV") where {R}
    isempty(pairs) && throw(ArgumentError("pairs must not be empty"))
    results = Vector{R}(undef, length(pairs))
    file = output isa AbstractString ? jldopen(output, "w") : nothing
    load_pair(pair) = Threads.@spawn begin
        imgA = load_frame(pair[1], image_type)
        imgB = load_frame(pair[2], image_type)
        preprocess === nothing ? (imgA, imgB) : (preprocess(imgA), preprocess(imgB))
    end
    try
        file === nothing || (file["format_version"] = RESULTS_FORMAT_VERSION)
        meter = Progress(length(pairs); desc = "$label sequence: ", enabled = progress === true)
        pending = load_pair(pairs[1])
        for (i, pair) in enumerate(pairs)
            frameA, frameB = pair
            try
                imgA, imgB = fetch_frames(pending)
                i < length(pairs) && (pending = load_pair(pairs[i + 1]))
                results[i] = process(imgA, imgB)
            catch
                @error "$label sequence failed on pair $i of $(length(pairs))" frameA = frame_label(frameA) frameB = frame_label(frameB)
                rethrow()
            end
            if file !== nothing
                file[result_key(i)] = results[i]
                if frameA isa AbstractString && frameB isa AbstractString
                    file[source_key(i)] = [String(frameA), String(frameB)]
                end
            elseif output isa Function
                write_pair_file(String(output(i, pair)), results[i], frameA, frameB)
            end
            progress isa Function ? progress(i, length(pairs)) : next!(meter)
        end
    finally
        file === nothing || close(file)
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
        if frameA isa AbstractString && frameB isa AbstractString
            f[source_key(1)] = [String(frameA), String(frameB)]
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

load_frame(x::AbstractString, ::Type{T}) where {T} = load_image(T, x)
load_frame(x::AbstractMatrix{<:Real}, ::Type) = x
load_frame(x, ::Type) =
    throw(ArgumentError("sequence entries must be file paths or real-valued matrices, got $(typeof(x))"))
