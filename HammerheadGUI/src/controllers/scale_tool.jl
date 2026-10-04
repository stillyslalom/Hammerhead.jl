# Scale-tool controller: derive a PhysicalScale from a calibration line —
# two clicked points on an image plus the known physical separation. This is
# the practical planar-calibration path for single-camera setups (a full
# PlanarTransform tool is out of scope).

"""
    ScaleTool(size::Dims{2})
    ScaleTool(image::AbstractMatrix)

Derive a planar pixel scale from two points on images of `size`
`(rows, cols)` (or the size of `image`; the tool keeps only the size).
`points` holds at most two clicked endpoints. `separation` is their known
physical distance; `dt` is the interval between the two PIV images.
`length_unit` and `time_unit` label those measurements. These values are
`Observables`.

Click two points along a feature of known size ([`click!`](@ref); a third
click starts a new line), enter the separation and units, and read the
derived [`pixel_size`](@ref) and [`physical_scale`](@ref).
"""
struct ScaleTool
    size::Dims{2}
    points::Observable{Vector{NTuple{2,Float64}}}
    separation::Observable{Float64}
    length_unit::Observable{String}
    dt::Observable{Float64}
    time_unit::Observable{String}
end

ScaleTool(sz::Dims{2}) =
    ScaleTool(sz, Observable(NTuple{2,Float64}[]), Observable(1.0), Observable("mm"),
              Observable(1.0), Observable("frame"))
ScaleTool(image::AbstractMatrix) = ScaleTool(size(image))

function Base.show(io::IO, st::ScaleTool)
    print(io, "ScaleTool($(st.size) image, $(length(st.points[])) point",
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
    undo_point!(st::ScaleTool) -> Bool

Drop the last endpoint (`false` when there is none).
"""
undo_point!(st::ScaleTool) =
    isempty(st.points[]) ? false : (pop!(st.points[]); notify(st.points); true)

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

"""
    scale_description(scale::Union{Nothing,PhysicalScale}) -> String

A readable sentence for a physical scale, e.g. "0.005882 mm per pixel ·
1 frame between exposures", or what happens without one.
"""
function scale_description(sc::Union{Nothing,PhysicalScale})
    sc === nothing && return "no scale: results stay in pixels and frames"
    tu = sc.time_unit == "frame" && sc.dt != 1 ? "frames" : sc.time_unit
    return string(@sprintf("%.4g", sc.pixel_size), " ", sc.length_unit, " per pixel · ",
                  @sprintf("%.4g", sc.dt), " ", tu, " between exposures")
end
