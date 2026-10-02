# Language-neutral result export and lazy frame sources.  This file deliberately
# uses only Base IO; CSV and legacy VTK are simple enough not to justify a hard
# dependency in the numerical package.

"""
    ROI(rows, cols)

Limit [`run_piv`](@ref), [`run_ptv`](@ref), or a sequence driver to a
rectangular part of each image. `rows` and `cols` are nonempty, positive
integer unit ranges in the original image. Returned particle positions or
vector centers retain the original image coordinates.
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

"""Common type for indexed frame sources used by [`image_pairs`](@ref)."""
abstract type AbstractFrameSource end

"""
    FrameSource(n, loader; timestamps=nothing, labels=nothing,
                source_id=nothing, frame_ids=nothing, time_unit=nothing, clock_id=nothing)

Wrap an indexed frame loader without materializing the whole recording.
`loader(i)` must return the image matrix for one-based frame index `i`.
Optional `timestamps` and `labels` must each have `n` entries. Use numeric
timestamps in the same time unit as any [`PhysicalScale`](@ref) you attach:
[`image_pairs`](@ref) subtracts paired timestamps into `FramePair.dt`.
`labels` identify frames in saved sequence results.

Optional nonempty string `source_id` and per-frame `frame_ids` are opaque caller
identifiers, not verified content hashes. `time_unit` and `clock_id` label the
provided timestamps without conversion or synchronization certification. They
are additive metadata for [`PairTiming`](@ref); omitted labels remain unknown.
The existing timestamp/physical-scale same-unit contract still applies.
"""
struct FrameSource{F,TS,L} <: AbstractFrameSource
    n::Int
    loader::F
    timestamps::TS
    labels::L
    source_id::Union{Nothing,String}
    frame_ids::Union{Nothing,Vector{String}}
    time_unit::Union{Nothing,String}
    clock_id::Union{Nothing,String}
end

function _source_metadata_string(value, name)
    value === nothing && return nothing
    value isa AbstractString && !isempty(value) || throw(ArgumentError("$name must be a nonempty string or nothing"))
    String(value)
end
function FrameSource(n::Integer, loader; timestamps=nothing, labels=nothing,
                     source_id=nothing, frame_ids=nothing, time_unit=nothing, clock_id=nothing)
    n >= 0 || throw(ArgumentError("frame count must be nonnegative"))
    timestamps === nothing || length(timestamps) == n ||
        throw(DimensionMismatch("timestamps length must equal frame count"))
    labels === nothing || length(labels) == n ||
        throw(DimensionMismatch("labels length must equal frame count"))
    frame_ids === nothing || length(frame_ids) == n || throw(DimensionMismatch("frame_ids length must equal frame count"))
    ids = frame_ids === nothing ? nothing : [_source_metadata_string(v, "frame_ids entries") for v in frame_ids]
    ids === nothing || all(v -> v !== nothing, ids) || throw(ArgumentError("frame_ids entries must be nonempty strings"))
    FrameSource(Int(n), loader, timestamps, labels, _source_metadata_string(source_id, "source_id"),
        ids === nothing ? nothing : String[ids...], _source_metadata_string(time_unit, "time_unit"), _source_metadata_string(clock_id, "clock_id"))
end

# Preserve the original public positional construction and type-parameter arity.
FrameSource(n::Integer, loader, timestamps, labels) = FrameSource(n, loader; timestamps, labels)
FrameSource{F,TS,L}(n::Int, loader::F, timestamps::TS, labels::L) where {F,TS,L} =
    FrameSource{F,TS,L}(n, loader, timestamps, labels, nothing, nothing, nothing, nothing)

Base.length(s::FrameSource) = s.n
Base.getindex(s::FrameSource, i::Integer) = (checkbounds(1:s.n, i); s.loader(i))
frame_timestamp(s::FrameSource, i) = s.timestamps === nothing ? nothing : s.timestamps[i]
frame_source_label(s::FrameSource, i) = s.labels === nothing ? string(i) : string(s.labels[i])

"""Reference to one indexed frame of an [`AbstractFrameSource`](@ref). The
source loader runs when a driver requests the frame; constructing the
reference does not load image pixels."""
struct FrameRef{S<:AbstractFrameSource}
    source::S
    index::Int
end

"""Two paired frames and optional `dt` between their timestamps. The
`dt` field is `nothing` without timestamps. In sequence drivers it
overrides the delay of a supplied [`PhysicalScale`](@ref); timestamps
alone do not attach a scale or convert displacement to velocity."""
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

Expose the pages of a TIFF as an indexed frame source. Construction decodes
the TIFF through FileIO and may hold the whole decoded stack in memory;
indexing a page converts only that page to `image_type` (`Float64` by
default). A single-page TIFF has one frame. Use [`image_pairs`](@ref) to
form lazy page references for a sequence driver.
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

Return [`FramePair`](@ref)s of lazy references to an indexed source.
`mode = :paired` selects starts spaced by `2 * stride`; `:chained` uses
starts spaced by `stride`. `offset` and `deltas` follow the vector overload of
[`image_pairs`](@ref). When both source timestamps are present, each pair
stores their difference as `dt`; check its units before attaching a scale.
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
    "match_residual", "index_a", "index_b", "length_unit", "time_unit", "velocity_unit",
    "trajectory_id", "observation_id", "frame_index", "elapsed_time", "gap_before",
    "position_valid", "velocity_valid", "time_provenance")

# Trajectory-only columns are an additive extension of hammerhead-table-1.
const _EMPTY_TRACKING_COLUMNS = ntuple(_ -> missing, 8)

_csv(x::Missing) = ""
_csv(::Nothing) = ""
_csv(x::Bool) = x ? "true" : "false"
_csv(x::Real) = string(x)
_csv(x) = '"' * replace(string(x), '"' => "\"\"") * '"'

function _export_units(r)
    s = r.scale
    s === nothing ? (r isa StereoPIVResult ? "world_unit" : "px", "frame",
                     r isa StereoPIVResult ? "world_unit/frame" : "px/frame", r) :
        (s.length_unit, s.time_unit, velocity_unit(s), physical(r))
end

# Prepare the optional calibrated grid once for both language-neutral writers.
# Keep the no-transform path untouched, including its physical() conversion.
function _export_grid(r; transform=nothing, length_unit=nothing, dt=nothing,
                      time_unit=nothing, uncertainty_assumption::Symbol=:unknown)
    if transform === nothing
        (length_unit === nothing && dt === nothing && time_unit === nothing &&
         uncertainty_assumption === :unknown) ||
            throw(ArgumentError("length_unit, dt, time_unit, and uncertainty_assumption require transform"))
        lu, tu, vu, q = _export_units(r)
        return lu, tu, vu, q, nothing
    end
    transform isa PlanarTransform || throw(ArgumentError("transform must be a PlanarTransform"))
    r isa PIVResult || throw(ArgumentError("PlanarTransform export supports planar PIVResult grids only"))
    r.scale === nothing ||
        throw(ArgumentError("PlanarTransform requires raw pixel data with no attached PhysicalScale; " *
                            "remove metadata only from an unconverted result, and supply dt explicitly"))
    length_unit isa AbstractString && !isempty(length_unit) ||
        throw(ArgumentError("transform requires a nonempty length_unit label"))
    if dt === nothing
        time_unit === nothing || throw(ArgumentError("time_unit requires an explicit dt"))
        tu = "frame"
    else
        dt isa Real && isfinite(dt) && dt > 0 ||
            throw(ArgumentError("dt must be a positive finite interval"))
        time_unit isa AbstractString && !isempty(time_unit) ||
            throw(ArgumentError("dt requires a nonempty time_unit label"))
        tu = String(time_unit)
    end
    uncertainty_assumption in (:unknown, :independent) ||
        throw(ArgumentError("uncertainty_assumption must be :unknown or :independent"))
    all(isfinite, transform.matrix) && all(isfinite, transform.offset) &&
        isfinite(det(transform.matrix)) && !iszero(det(transform.matrix)) ||
        throw(ArgumentError("transform must be finite and nonsingular"))
    T = promote_type(eltype(r.u), eltype(transform.matrix),
                     dt === nothing ? eltype(r.u) : typeof(float(dt)))
    interval = dt === nothing ? one(T) : T(dt)
    nx, ny = length(r.x), length(r.y)
    x, y, u, v, su, sv = ntuple(_ -> Matrix{T}(undef, ny, nx), 6)
    for j in eachindex(r.x), i in eachindex(r.y)
        x[i,j], y[i,j] = transform_point(transform, (r.x[j], r.y[i]))
        ui, vi = transform_vector(transform, (r.u[i,j], r.v[i,j]))
        u[i,j], v[i,j] = ui / interval, vi / interval
        su[i,j] = _export_component_uncertainty(transform.matrix[1,1], transform.matrix[1,2],
            r.uncertainty_u[i,j], r.uncertainty_v[i,j], interval, uncertainty_assumption)
        sv[i,j] = _export_component_uncertainty(transform.matrix[2,1], transform.matrix[2,2],
            r.uncertainty_u[i,j], r.uncertainty_v[i,j], interval, uncertainty_assumption)
    end
    lu = String(length_unit)
    return lu, tu, lu * "/" * tu, r, (; x, y, u, v, su, sv)
end

function _export_component_uncertainty(a, b, su, sv, dt, assumption)
    # Zero coefficients do not require an unavailable marginal from that axis.
    iszero(a) && return sv >= 0 ? abs(b) * sv / dt : NaN
    iszero(b) && return su >= 0 ? abs(a) * su / dt : NaN
    assumption === :unknown && return NaN
    su >= 0 && sv >= 0 || return NaN
    return hypot(a * su, b * sv) / dt
end

"""
    export_table(path, result; frame_id="", source_a="", source_b="",
                 transform=nothing, length_unit=nothing, dt=nothing,
                 time_unit=nothing, uncertainty_assumption=:unknown)

Write one planar PIV, stereo PIV, PTV, or tracking result to a long-form UTF-8 CSV
using schema `hammerhead-table-1`, replacing `path` and returning it.
Every grid node, matched particle, or observed trajectory point is
written, including masked and flagged entries; use the `masked` and
`outlier` columns when filtering. Non-applicable fields are empty.
If a [`PhysicalScale`](@ref) is attached, positions and components are
converted with [`physical`](@ref) and the units are recorded. Otherwise
planar PIV and PTV values remain in pixels per pair, and stereo values in
unlabeled world length units per pair.

For [`TrackingResult`](@ref), `trajectory_id` is the one-based position in
`result.trajectories`, `observation_id` is the one-based position in that
trajectory, and `point_id` counts all written observations. These IDs are
local to this result. `frame_index` preserves the observed input frame index;
`gap_before` counts missed frames since the preceding observation (zero for
the first observation). No interpolated or predicted gap rows are written.
`frame_id`, `source_a`, and `source_b` are caller-supplied labels for the whole
result, not per-observation frame numbers or source filenames.

`elapsed_time` is measured from input frame 1: `(frame_index - 1)` without a
scale (`time_provenance = "frame_index"`), or `(frame_index - 1) * scale.dt`
with one (`time_provenance = "physical_scale"`). The latter assumes `dt` is
the uniform interval between input frames. These are derived elapsed times,
not acquisition timestamps; original timestamps are not stored in a tracking
result. `u` and `v` use [`trajectory_velocities`](@ref), accounting for gaps
and retaining the scale's interval even after physical conversion.

`position_valid` means both coordinates are finite; `velocity_valid` means
the position and both derived components are finite. These are numerical
validity checks, not uncertainty or tracking-quality estimates. Tracking
has no retained mask/outlier flags, so those columns are empty. A singleton
has empty `u`/`v` and `velocity_valid = false`. Empty trajectories have no
rows (their IDs are not reused); an empty result writes only the header.
Malformed trajectories (mismatched arrays, non-increasing or out-of-range
frames, or inconsistent `start_frame`) are rejected before replacing `path`.

For a raw, unscaled planar [`PIVResult`](@ref), `transform = PlanarTransform(...)`
exports coordinates as `A * (x, y) + b` and components as `A * (u, v)` in
the transformed basis. Supply `length_unit` explicitly. Without `dt`, components
remain displacements per frame interval; supplying positive finite `dt` and
`time_unit` converts them to velocities. A transform and an attached
[`PhysicalScale`](@ref) cannot be combined, including a scale left by
[`physical`](@ref); stripping metadata does not undo conversion. Stereo, PTV,
and tracking transforms are unsupported and rejected before replacing `path`.

For transformed uncertainties, `:unknown` (default) writes `NaN` for a component
mixing both input axes, because the cross-component covariance is not stored.
Rows depending on just one axis transform its standard deviation exactly.
`uncertainty_assumption = :independent` explicitly assumes zero input
cross-component covariance and exports
`hypot(A[k,1] * uncertainty_u, A[k,2] * uncertainty_v) / dt` (unit interval
when `dt` is absent). The induced output covariance is not exported, and
calibration/timing errors are not included. Mask/outlier flags and pixel-native
quality metrics retain their values. See [Scale results to physical units](@ref).
"""
function export_table(path::AbstractString, r::Union{PIVResult,StereoPIVResult,PTVResult};
                      frame_id="", source_a="", source_b="", transform=nothing,
                      length_unit=nothing, dt=nothing, time_unit=nothing,
                      uncertainty_assumption::Symbol=:unknown)
    lu, tu, vu, q, grid = _export_grid(r; transform, length_unit, dt, time_unit,
                                      uncertainty_assumption)
    open(path, "w") do io
        println(io, join(TABLE_COLUMNS, ','))
        emit(vals) = println(io, join(_csv.((vals..., _EMPTY_TRACKING_COLUMNS...)), ','))
        if q isa PIVResult
            k = 0
            for j in eachindex(q.x), i in eachindex(q.y)
                k += 1
                x, y, u, v, su, sv = grid === nothing ?
                    (q.x[j], q.y[i], q.u[i,j], q.v[i,j], q.uncertainty_u[i,j], q.uncertainty_v[i,j]) :
                    (grid.x[i,j], grid.y[i,j], grid.u[i,j], grid.v[i,j], grid.su[i,j], grid.sv[i,j])
                emit((TABLE_SCHEMA_VERSION,"planar",frame_id,source_a,source_b,k,i,j,
                    x,y,missing,u,v,missing,q.mask[i,j],q.outliers[i,j],
                    q.peak_ratio[i,j],q.correlation_moment[i,j],su,sv,
                    missing,missing,missing,missing,lu,tu,vu))
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

function _check_tracking_table(r::TrackingResult)
    r.n_frames >= 0 || throw(ArgumentError("tracking frame count must be nonnegative"))
    for (id, t) in enumerate(r.trajectories)
        length(t.x) == length(t.y) == length(t.frames) ||
            throw(ArgumentError("trajectory $id has mismatched coordinate/frame lengths"))
        isempty(t.frames) && continue
        t.start_frame == first(t.frames) ||
            throw(ArgumentError("trajectory $id start_frame differs from its first observation"))
        all(f -> 1 <= f <= r.n_frames, t.frames) ||
            throw(ArgumentError("trajectory $id has a frame outside 1:$(r.n_frames)"))
        all(k -> t.frames[k] > t.frames[k - 1], 2:length(t.frames)) ||
            throw(ArgumentError("trajectory $id frame indices must be strictly increasing"))
    end
    nothing
end

function export_table(path::AbstractString, r::TrackingResult;
                      frame_id="", source_a="", source_b="", transform=nothing,
                      length_unit=nothing, dt=nothing, time_unit=nothing,
                      uncertainty_assumption::Symbol=:unknown)
    _check_tracking_table(r)
    lu, tu, vu, q, _ = _export_grid(r; transform, length_unit, dt, time_unit,
                                   uncertainty_assumption)
    dt = q.scale === nothing ? 1 : q.scale.dt
    provenance = q.scale === nothing ? "frame_index" : "physical_scale"
    open(path, "w") do io
        println(io, join(TABLE_COLUMNS, ','))
        point_id = 0
        for (id, t) in enumerate(q.trajectories)
            isempty(t.x) && continue
            u, v = length(t) >= 2 ? trajectory_velocities(t, q.scale) : (nothing, nothing)
            for k in eachindex(t.x)
                point_id += 1
                uk, vk = u === nothing ? (missing, missing) : (u[k], v[k])
                position_valid = isfinite(t.x[k]) && isfinite(t.y[k])
                velocity_valid = u !== nothing && position_valid && isfinite(uk) && isfinite(vk)
                gap = k == 1 ? 0 : t.frames[k] - t.frames[k - 1] - 1
                vals = (TABLE_SCHEMA_VERSION,"tracking",frame_id,source_a,source_b,
                    point_id,missing,missing,t.x[k],t.y[k],missing,uk,vk,missing,
                    missing,missing,missing,missing,missing,missing,missing,missing,
                    missing,missing,lu,tu,vu,id,k,t.frames[k],(t.frames[k] - 1) * dt,
                    gap,position_valid,velocity_valid,provenance)
                println(io, join(_csv.(vals), ','))
            end
        end
    end
    path
end

"""
    export_vtk(path, result; transform=nothing, length_unit=nothing, dt=nothing,
               time_unit=nothing, uncertainty_assumption=:unknown)

Write a planar or stereo grid to a legacy ASCII VTK file, replacing `path`
and returning it. The file includes all grid nodes, including masked and
flagged ones, plus mask, outlier, uncertainty, and available quality arrays.
Its vector array is named `velocity` by the VTK writer; values are still
displacements per pair when no [`PhysicalScale`](@ref) is attached. With a
scale, [`physical`](@ref) converts them to velocity. `FIELD` metadata stores
`coordinate_unit` and `component_unit`; unscaled stereo uses the placeholder
`world_unit` because the calibration's length-unit name is unavailable.
PTV and tracking results are not structured grids and are not supported.
For raw planar PIV grids, the transform/unit/time/uncertainty keywords follow
[`export_table`](@ref), including refusal of attached scales and unavailable
mixed-axis uncertainties under the default `:unknown` assumption. The affine
map changes every point and its vector basis while retaining grid topology;
rotation or reflection need not preserve separable coordinate axes.
"""
function export_vtk(path::AbstractString, r::Union{PIVResult,StereoPIVResult};
                    transform=nothing, length_unit=nothing, dt=nothing, time_unit=nothing,
                    uncertainty_assumption::Symbol=:unknown)
    coordinate_unit, _, component_unit, q, grid = _export_grid(r;
        transform, length_unit, dt, time_unit, uncertainty_assumption)
    nx, ny = length(q.x), length(q.y)
    z = q isa StereoPIVResult ? q.z : 0
    open(path, "w") do io
        println(io, "# vtk DataFile Version 3.0\nHammerhead result\nASCII\nDATASET STRUCTURED_GRID")
        println(io, "DIMENSIONS $nx $ny 1\nPOINTS $(nx*ny) double")
        for i in eachindex(q.y), j in eachindex(q.x)
            x, y = grid === nothing ? (q.x[j], q.y[i]) : (grid.x[i,j], grid.y[i,j])
            println(io, "$x $y $z")
        end
        println(io, "FIELD FieldData 2")
        for (name, unit) in (("coordinate_unit", coordinate_unit),
                             ("component_unit", component_unit))
            bytes = codeunits(unit)
            println(io, "$name 1 $(length(bytes)) unsigned_char")
            println(io, join(bytes, ' '))
        end
        println(io, "POINT_DATA $(nx*ny)\nVECTORS velocity double")
        for i in eachindex(q.y), j in eachindex(q.x)
            w = q isa StereoPIVResult ? q.w[i,j] : 0
            u, v = grid === nothing ? (q.u[i,j], q.v[i,j]) : (grid.u[i,j], grid.v[i,j])
            println(io, "$u $v $w")
        end
        function scalar(name, a)
            println(io, "SCALARS $name double 1\nLOOKUP_TABLE default")
            for i in eachindex(q.y), j in eachindex(q.x)
                value = a[i,j]
                println(io, value isa Bool ? Int(value) : value)
            end
        end
        scalar("masked", q.mask); scalar("outlier", q.outliers)
        scalar("uncertainty_u", grid === nothing ? q.uncertainty_u : grid.su)
        scalar("uncertainty_v", grid === nothing ? q.uncertainty_v : grid.sv)
        q isa StereoPIVResult && scalar("uncertainty_w", q.uncertainty_w)
        q isa PIVResult && (scalar("peak_ratio", q.peak_ratio); scalar("correlation_moment", q.correlation_moment))
    end
    path
end
