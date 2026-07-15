# Language-neutral result export and lazy frame sources.  This file deliberately
# uses only Base IO; CSV and legacy VTK are simple enough not to justify a hard
# dependency in the numerical package.

"""
    ROI(rows, cols)

Rectangular image region used by `run_piv`, `run_ptv`, and the sequence
drivers. `rows` and `cols` are integer unit ranges in the original image.
Returned coordinates remain in that original image coordinate system.
"""
struct ROI
    rows::UnitRange{Int}
    cols::UnitRange{Int}
    function ROI(rows::UnitRange{<:Integer}, cols::UnitRange{<:Integer})
        isempty(rows) && throw(ArgumentError("ROI rows must not be empty"))
        isempty(cols) && throw(ArgumentError("ROI columns must not be empty"))
        first(rows) >= 1 && first(cols) >= 1 ||
            throw(ArgumentError("ROI indices must be positive"))
        new(Int(first(rows)):Int(last(rows)), Int(first(cols)):Int(last(cols)))
    end
end

ROI(t::Tuple{UnitRange{<:Integer},UnitRange{<:Integer}}) = ROI(t...)

function roi_views(imgA, imgB, mask, roi::ROI)
    axes(imgA, 1) == axes(imgB, 1) && axes(imgA, 2) == axes(imgB, 2) ||
        throw(DimensionMismatch("images must have matching axes before applying ROI"))
    last(roi.rows) <= size(imgA, 1) && last(roi.cols) <= size(imgA, 2) ||
        throw(BoundsError(imgA, (roi.rows, roi.cols)))
    cmask = mask === nothing ? nothing :
        size(mask) == size(imgA) ? view(mask, roi.rows, roi.cols) :
        size(mask) == (length(roi.rows), length(roi.cols)) ? mask :
        throw(DimensionMismatch("mask must match the full image or ROI size"))
    return view(imgA, roi.rows, roi.cols), view(imgB, roi.rows, roi.cols), cmask
end

function offset_result(r::PIVResult{T}, roi::ROI) where {T}
    PIVResult{T}(r.x .+ T(first(roi.cols) - 1), r.y .+ T(first(roi.rows) - 1),
        r.u, r.v, r.peak_ratio, r.correlation_moment, r.uncertainty_u,
        r.uncertainty_v, r.outliers, r.mask, r.parameters, r.correlation_planes, r.scale)
end

function offset_result(r::PTVResult{T}, roi::ROI) where {T}
    dx, dy = T(first(roi.cols) - 1), T(first(roi.rows) - 1)
    pa = Particles{T}(r.particles_a.x .+ dx, r.particles_a.y .+ dy,
        r.particles_a.intensity, r.particles_a.diameter)
    pb = Particles{T}(r.particles_b.x .+ dx, r.particles_b.y .+ dy,
        r.particles_b.intensity, r.particles_b.diameter)
    PTVResult{T}(r.x .+ dx, r.y .+ dy, r.u, r.v, r.match_residual,
        r.outliers, r.index_a, r.index_b, pa, pb, r.parameters, r.scale)
end

"""Abstract interface for lazily addressable image recordings."""
abstract type AbstractFrameSource end

"""
    FrameSource(n, loader; timestamps=nothing, labels=nothing)

Create a lazy source around `loader(i)`. Camera-format adapters only need to
provide a frame count and an index loader; optional timestamps are carried into
`FramePair.dt` by [`image_pairs`](@ref).
"""
struct FrameSource{F,TS,L} <: AbstractFrameSource
    n::Int
    loader::F
    timestamps::TS
    labels::L
end

function FrameSource(n::Integer, loader; timestamps=nothing, labels=nothing)
    n >= 0 || throw(ArgumentError("frame count must be nonnegative"))
    timestamps === nothing || length(timestamps) == n ||
        throw(DimensionMismatch("timestamps length must equal frame count"))
    labels === nothing || length(labels) == n ||
        throw(DimensionMismatch("labels length must equal frame count"))
    FrameSource(Int(n), loader, timestamps, labels)
end

Base.length(s::FrameSource) = s.n
Base.getindex(s::FrameSource, i::Integer) = (checkbounds(1:s.n, i); s.loader(i))
frame_timestamp(s::FrameSource, i) = s.timestamps === nothing ? nothing : s.timestamps[i]
frame_source_label(s::FrameSource, i) = s.labels === nothing ? string(i) : string(s.labels[i])

"""A frame reference. Calling `load_frame` materializes only this frame."""
struct FrameRef{S<:AbstractFrameSource}
    source::S
    index::Int
end

"""Pair descriptor carrying source indices and optional timestamp difference."""
struct FramePair{A,B,D}
    first::A
    second::B
    dt::D
end
Base.length(::FramePair) = 2
Base.getindex(p::FramePair, i::Int) = i == 1 ? p.first : i == 2 ? p.second : throw(BoundsError(p, i))
Base.iterate(p::FramePair, st=1) = st > 2 ? nothing : (p[st], st + 1)

"""
    TIFFStack(path; image_type=Float64)

Open a multi-page TIFF as a frame source. The TIFF container is opened once;
individual pages are converted to matrices only when indexed. Single-page
TIFFs are valid one-frame stacks.
"""
struct TIFFStack{T,R} <: AbstractFrameSource
    path::String
    raw::R
end

function TIFFStack(path::AbstractString; image_type::Type{T}=Float64) where {T<:AbstractFloat}
    isfile(path) || throw(ArgumentError("no such image file: $path"))
    raw = _load_raw(FileIO.query(path))
    ndims(raw) in (2, 3) || throw(ArgumentError("$path is not a 2D or multi-page TIFF"))
    TIFFStack{T,typeof(raw)}(String(path), raw)
end
Base.length(s::TIFFStack) = ndims(s.raw) == 2 ? 1 : size(s.raw, 3)
function Base.getindex(s::TIFFStack{T}, i::Integer) where {T}
    checkbounds(1:length(s), i)
    page = ndims(s.raw) == 2 ? s.raw : view(s.raw, :, :, i)
    image_to_matrix(T, page, "$(s.path) page $i")
end
frame_timestamp(::TIFFStack, i) = nothing
frame_source_label(s::TIFFStack, i) = "$(s.path)#$i"

function _pair_indices(n::Int; mode::Symbol=:paired, stride::Integer=1,
                       offset::Integer=0, deltas=(1,))
    stride >= 1 || throw(ArgumentError("stride must be positive"))
    offset >= 0 || throw(ArgumentError("offset must be nonnegative"))
    ds = deltas isa Integer ? (Int(deltas),) : Tuple(Int.(deltas))
    !isempty(ds) && all(>(0), ds) || throw(ArgumentError("deltas must be positive"))
    starts = if mode === :paired
        (1 + offset):(2 * stride):n
    elseif mode === :chained
        (1 + offset):stride:n
    else
        throw(ArgumentError("mode must be :paired or :chained, got :$mode"))
    end
    [(a, a+d) for d in ds for a in starts if a + d <= n]
end

"""
    image_pairs(source; mode=:paired, stride=1, offset=0, deltas=(1,))

Build lazy pairs from a frame source, with arbitrary start `offset`, sampling
`stride`, and one or more frame separations (`deltas`). Timestamped sources
attach the actual time difference to each returned `FramePair`.
"""
function image_pairs(s::AbstractFrameSource; mode::Symbol=:paired, stride::Integer=1,
                     offset::Integer=0, deltas=(1,))
    [begin
        ta, tb = frame_timestamp(s, a), frame_timestamp(s, b)
        FramePair(FrameRef(s, a), FrameRef(s, b),
                  ta === nothing || tb === nothing ? nothing : tb - ta)
     end for (a,b) in _pair_indices(length(s); mode, stride, offset, deltas)]
end

load_frame(x::FrameRef, ::Type) = x.source[x.index]
frame_label(x::FrameRef) = frame_source_label(x.source, x.index)
source_label(x::FrameRef) = frame_source_label(x.source, x.index)

const TABLE_SCHEMA_VERSION = "hammerhead-table-1"
const TABLE_COLUMNS = ("schema_version", "result_type", "frame_id", "source_a", "source_b",
    "point_id", "i", "j", "x", "y", "z", "u", "v", "w", "masked", "outlier",
    "peak_ratio", "correlation_moment", "uncertainty_u", "uncertainty_v", "uncertainty_w",
    "match_residual", "index_a", "index_b", "length_unit", "time_unit", "velocity_unit")

_csv(x::Missing) = ""
_csv(::Nothing) = ""
_csv(x::Bool) = x ? "true" : "false"
_csv(x::Real) = string(x)
_csv(x) = '"' * replace(string(x), '"' => "\"\"") * '"'

function _export_units(r)
    s = r.scale
    s === nothing ? ("px", "frame", "px/frame", r) :
        (s.length_unit, s.time_unit, velocity_unit(s), physical(r))
end

"""
    export_table(path, result; frame_id="", source_a="", source_b="")

Write a long-form UTF-8 CSV using the stable `hammerhead-table-1` schema.
Planar, stereo, and PTV exports share the same columns; non-applicable values
are empty. Attached physical scaling is applied and its units are recorded.
"""
function export_table(path::AbstractString, r::Union{PIVResult,StereoPIVResult,PTVResult};
                      frame_id="", source_a="", source_b="")
    lu, tu, vu, q = _export_units(r)
    open(path, "w") do io
        println(io, join(TABLE_COLUMNS, ','))
        emit(vals) = println(io, join(_csv.(vals), ','))
        if q isa PIVResult
            k = 0
            for j in eachindex(q.x), i in eachindex(q.y)
                k += 1
                emit((TABLE_SCHEMA_VERSION,"planar",frame_id,source_a,source_b,k,i,j,
                    q.x[j],q.y[i],missing,q.u[i,j],q.v[i,j],missing,q.mask[i,j],q.outliers[i,j],
                    q.peak_ratio[i,j],q.correlation_moment[i,j],q.uncertainty_u[i,j],
                    q.uncertainty_v[i,j],missing,missing,missing,missing,lu,tu,vu))
            end
        elseif q isa StereoPIVResult
            k = 0
            for j in eachindex(q.x), i in eachindex(q.y)
                k += 1
                emit((TABLE_SCHEMA_VERSION,"stereo",frame_id,source_a,source_b,k,i,j,
                    q.x[j],q.y[i],q.z,q.u[i,j],q.v[i,j],q.w[i,j],q.mask[i,j],q.outliers[i,j],
                    missing,missing,q.uncertainty_u[i,j],q.uncertainty_v[i,j],q.uncertainty_w[i,j],
                    missing,missing,missing,lu,tu,vu))
            end
        else
            for k in eachindex(q.x)
                emit((TABLE_SCHEMA_VERSION,"ptv",frame_id,source_a,source_b,k,missing,missing,
                    q.x[k],q.y[k],missing,q.u[k],q.v[k],missing,false,q.outliers[k],missing,missing,
                    missing,missing,missing,q.match_residual[k],q.index_a[k],q.index_b[k],lu,tu,vu))
            end
        end
    end
    path
end

"""
    export_vtk(path, result)

Write a planar or stereo result as an ASCII VTK legacy structured grid. The
file contains point coordinates, a three-component `velocity` vector, mask and
outlier flags, and available uncertainty/quality scalar arrays.
"""
function export_vtk(path::AbstractString, r::Union{PIVResult,StereoPIVResult})
    _, _, _, q = _export_units(r)
    nx, ny = length(q.x), length(q.y)
    z = q isa StereoPIVResult ? q.z : 0
    open(path, "w") do io
        println(io, "# vtk DataFile Version 3.0\nHammerhead result\nASCII\nDATASET STRUCTURED_GRID")
        println(io, "DIMENSIONS $nx $ny 1\nPOINTS $(nx*ny) double")
        for y in q.y, x in q.x
            println(io, "$x $y $z")
        end
        println(io, "POINT_DATA $(nx*ny)\nVECTORS velocity double")
        for i in eachindex(q.y), j in eachindex(q.x)
            w = q isa StereoPIVResult ? q.w[i,j] : 0
            println(io, "$(q.u[i,j]) $(q.v[i,j]) $w")
        end
        function scalar(name, a)
            println(io, "SCALARS $name double 1\nLOOKUP_TABLE default")
            for i in eachindex(q.y), j in eachindex(q.x)
                value = a[i,j]
                println(io, value isa Bool ? Int(value) : value)
            end
        end
        scalar("masked", q.mask); scalar("outlier", q.outliers)
        scalar("uncertainty_u", q.uncertainty_u); scalar("uncertainty_v", q.uncertainty_v)
        q isa StereoPIVResult && scalar("uncertainty_w", q.uncertainty_w)
        q isa PIVResult && (scalar("peak_ratio", q.peak_ratio); scalar("correlation_moment", q.correlation_moment))
    end
    path
end
