# Scale-tool controller: derive a PhysicalScale from a calibration line —
# two clicked points on an image plus the known physical separation — and
# hand it to a BatchRunner. This is the practical planar-calibration path
# for single-camera setups (a full PlanarTransform tool is out of scope).

"""
    ScaleTool(image)
    ScaleTool(path::AbstractString)

Controller for the calibration-line scale tool: an `image` (matrix, or a
path loaded with `Hammerhead.load_image`), the clicked line endpoints
(`points`, at most two), and the physical inputs `separation` (the known
distance between the two points), `length_unit`, `dt`, and `time_unit` — all
as `Observables`.

Click two points along a feature of known size ([`click!`](@ref); a third
click starts a new line), enter the separation and units, and read the
derived [`pixel_size`](@ref) / [`physical_scale`](@ref); hand the result to
a batch with [`apply_scale!`](@ref).
"""
struct ScaleTool
    image::Matrix{Float64}
    points::Observable{Vector{NTuple{2,Float64}}}
    separation::Observable{Float64}
    length_unit::Observable{String}
    dt::Observable{Float64}
    time_unit::Observable{String}
end

ScaleTool(image::AbstractMatrix{<:Real}) =
    ScaleTool(Matrix{Float64}(image), Observable(NTuple{2,Float64}[]),
              Observable(1.0), Observable("mm"),
              Observable(1.0), Observable("frame"))
ScaleTool(path::AbstractString) = ScaleTool(load_image(path))

function Base.show(io::IO, st::ScaleTool)
    print(io, "ScaleTool($(size(st.image)) image, $(length(st.points[])) point",
          length(st.points[]) == 1 ? "" : "s", ")")
end

"""
    click!(st::ScaleTool, x::Real, y::Real)

Place a calibration-line endpoint at data-space `(x, y)`: the first two
clicks set the endpoints, a third click starts a new line at `(x, y)`.
"""
function click!(st::ScaleTool, x::Real, y::Real)
    pts = st.points[]
    length(pts) >= 2 && empty!(pts)
    push!(pts, (Float64(x), Float64(y)))
    notify(st.points)
    return st
end

"""
    clear_points!(st::ScaleTool)

Drop the calibration-line endpoints.
"""
clear_points!(st::ScaleTool) = (empty!(st.points[]); notify(st.points); st)

"""
    set_separation!(st::ScaleTool, value)
    set_dt!(st::ScaleTool, value)

Set the physical separation between the two clicked points / the frame
interval, from a positive number or its string form.
"""
set_separation!(st::ScaleTool, v::Real) =
    (v > 0 || throw(ArgumentError("separation must be positive, got $v"));
     st.separation[] = Float64(v); st)
set_separation!(st::ScaleTool, s::AbstractString) =
    (st.separation[] = _parse_positive(s, "separation"); st)
set_dt!(st::ScaleTool, v::Real) =
    (v > 0 || throw(ArgumentError("dt must be positive, got $v")); st.dt[] = Float64(v); st)
set_dt!(st::ScaleTool, s::AbstractString) = (st.dt[] = _parse_positive(s, "dt"); st)

"""
    pixel_distance(st::ScaleTool) -> Union{Nothing,Float64}

Length of the calibration line in pixels, or `nothing` until two points are
placed (or when they coincide).
"""
function pixel_distance(st::ScaleTool)
    pts = st.points[]
    length(pts) == 2 || return nothing
    d = hypot(pts[2][1] - pts[1][1], pts[2][2] - pts[1][2])
    return d > 0 ? d : nothing
end

"""
    pixel_size(st::ScaleTool) -> Union{Nothing,Float64}

The derived pixel size, `separation / pixel_distance` (physical units per
pixel), or `nothing` until the line is defined.
"""
function pixel_size(st::ScaleTool)
    d = pixel_distance(st)
    return d === nothing ? nothing : st.separation[] / d
end

"""
    physical_scale(st::ScaleTool) -> Union{Nothing,PhysicalScale}

The [`PhysicalScale`](@ref) defined by the calibration line and the entered
`dt`/units, or `nothing` until the line is defined.
"""
function physical_scale(st::ScaleTool)
    ps = pixel_size(st)
    ps === nothing && return nothing
    return PhysicalScale(ps, st.dt[], st.length_unit[], st.time_unit[])
end

"""
    apply_scale!(bc::BatchRunner, st::ScaleTool)

Copy the tool's derived scale into a [`BatchRunner`](@ref)'s scale form
(see [`set_scale!`](@ref)), so the batch outputs carry it. Throws when the
calibration line is not defined yet.
"""
function apply_scale!(bc::BatchRunner, st::ScaleTool)
    ps = pixel_size(st)
    ps === nothing &&
        throw(ArgumentError("place two calibration points first"))
    set_scale!(bc; pixel_size = ps, dt = st.dt[],
               length_unit = st.length_unit[], time_unit = st.time_unit[])
    return bc
end

"""
    scale_summary(st::ScaleTool) -> String

One-line status: the line length and derived pixel size, or a prompt while
points are missing.
"""
function scale_summary(st::ScaleTool)
    d = pixel_distance(st)
    d === nothing && return "click two points of known separation"
    ps = st.separation[] / d
    return string(@sprintf("%.4g", d), " px = ", @sprintf("%.4g", st.separation[]),
                  " ", st.length_unit[], " → ", @sprintf("%.4g", ps), " ",
                  st.length_unit[], "/px")
end
