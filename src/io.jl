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
    return image_to_matrix(T, FileIO.load(path), path)
end

load_image(path::AbstractString) = load_image(Float64, path)

image_to_matrix(::Type{T}, img::AbstractMatrix{<:Real}, path) where {T} = T.(img)
image_to_matrix(::Type{T}, img::AbstractMatrix{<:Colorant}, path) where {T} = T.(Gray.(img))
image_to_matrix(::Type, img, path) =
    throw(ArgumentError("$path did not load as a single 2D image " *
                        "(got $(summary(img))); load and slice it manually"))

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

const RESULTS_FORMAT_VERSION = 1

result_key(i::Integer) = "results/" * lpad(i, 6, '0')
source_key(i::Integer) = "sources/" * lpad(i, 6, '0')

"""
    save_results(path, results) -> path

Save a `PIVResult` or vector of `PIVResult`s to a JLD2 file (conventionally
`*.jld2`), overwriting `path` if it exists. Read back with
[`load_results`](@ref).
"""
function save_results(path::AbstractString, results::AbstractVector{<:PIVResult})
    jldopen(path, "w") do f
        f["format_version"] = RESULTS_FORMAT_VERSION
        for (i, r) in enumerate(results)
            f[result_key(i)] = r
        end
    end
    return path
end

save_results(path::AbstractString, result::PIVResult) = save_results(path, [result])

"""
    load_results(path) -> Vector{PIVResult}

Load the results stored in a JLD2 file written by [`save_results`](@ref) or
[`run_piv_sequence`](@ref), in sequence order. Files written by
`run_piv_sequence(...; output = path)` from file-path pairs also carry the
source image paths, retrievable with `JLD2.load(path, "sources/000001")` etc.
"""
function load_results(path::AbstractString)
    jldopen(path, "r") do f
        haskey(f, "results") ||
            throw(ArgumentError("$path has no \"results\" group; not a Hammerhead results file"))
        g = f["results"]
        return PIVResult[g[k] for k in sort!(keys(g))]
    end
end

"""
    run_piv_sequence(pairs, params = PIVParameters();
                     preprocess = nothing, output = nothing,
                     progress = true, kwargs...) -> Vector{PIVResult}

Run PIV over a sequence of image pairs. `pairs` is a vector of 2-tuples whose
entries are file paths (loaded with [`load_image`](@ref)) and/or real-valued
matrices — see [`image_pairs`](@ref) for building it from a frame list.
`params` is a single `PIVParameters` or a multi-pass schedule; remaining
`kwargs` are forwarded to [`run_piv`](@ref).

- `preprocess`: function applied to each frame after loading, e.g.
  `img -> clahe!(subtract_background!(img, bg))`. Frames loaded from file
  paths are fresh buffers, so mutating preprocessors are safe; in-memory
  matrix pairs are passed through as-is — use the allocating versions there
  to leave the caller's arrays untouched.
- `output`: path of a JLD2 file (overwritten) to which results are written
  incrementally as they complete, so a crashed batch keeps its finished pairs.
  For file-path pairs the source paths are stored alongside. Read back with
  [`load_results`](@ref).
- `progress`: show a progress meter.
- `image_type`: element type frames are loaded as (default `Float64`);
  `Float32` runs the pipeline in single precision. In-memory matrix pairs are
  used as-is, so convert those yourself.
"""
function run_piv_sequence(pairs::AbstractVector,
                          params::Union{PIVParameters,AbstractVector{PIVParameters}} = PIVParameters();
                          preprocess = nothing,
                          output::Union{Nothing,AbstractString} = nothing,
                          progress::Bool = true,
                          image_type::Type{<:AbstractFloat} = Float64,
                          kwargs...)
    isempty(pairs) && throw(ArgumentError("pairs must not be empty"))
    results = Vector{PIVResult}(undef, length(pairs))
    file = output === nothing ? nothing : jldopen(output, "w")
    try
        file === nothing || (file["format_version"] = RESULTS_FORMAT_VERSION)
        meter = Progress(length(pairs); desc = "PIV sequence: ", enabled = progress)
        for (i, pair) in enumerate(pairs)
            frameA, frameB = pair
            try
                imgA = load_frame(frameA, image_type)
                imgB = load_frame(frameB, image_type)
                if preprocess !== nothing
                    imgA = preprocess(imgA)
                    imgB = preprocess(imgB)
                end
                results[i] = run_piv(imgA, imgB, params; kwargs...)
            catch
                @error "PIV sequence failed on pair $i of $(length(pairs))" frameA = frame_label(frameA) frameB = frame_label(frameB)
                rethrow()
            end
            if file !== nothing
                file[result_key(i)] = results[i]
                if frameA isa AbstractString && frameB isa AbstractString
                    file[source_key(i)] = [String(frameA), String(frameB)]
                end
            end
            next!(meter)
        end
    finally
        file === nothing || close(file)
    end
    return results
end

frame_label(x::AbstractString) = x
frame_label(x) = summary(x)

load_frame(x::AbstractString, ::Type{T}) where {T} = load_image(T, x)
load_frame(x::AbstractMatrix{<:Real}, ::Type) = x
load_frame(x, ::Type) =
    throw(ArgumentError("sequence entries must be file paths or real-valued matrices, got $(typeof(x))"))
