# Mask-editor controller: polygon drawing/editing state for an image size,
# exporting the package mask convention (image-sized Bool, `true` =
# excluded). Framework-free — a canvas forwards clicks and key presses into
# the gesture API below.

"""
    MaskEditor(size::Dims{2}; polygons = [], holes = falses(length(polygons)), raster = nothing)
    MaskEditor(image::AbstractMatrix; kwargs...)

Edit an exclusion mask for images of `size` `(rows, cols)` (or the size of
`image`; the editor keeps only the size). Editing state is held in
`Observables`: `polygons` (committed polygons, each a vector of `(x, y)`
vertices in pixel coordinates), `holes` (per polygon: `true` restores an
area instead of excluding it), `active` (the in-progress polygon),
`selected` (index of the selected polygon or `nothing`), `show_mask`
(overlay toggle), and `raster` (a mask the polygons are drawn on top of —
a loaded mask, or the result of [`grow_mask!`](@ref)/[`shrink_mask!`](@ref)).

Gestures ([`click!`](@ref), [`alt_click!`](@ref)) implement the editing
model: click to add vertices (a click on empty background starts a new
polygon; a click inside an existing polygon selects it), alt/right-click to
close the active polygon. Export the combined mask with
`polygon_mask(editor)` or [`save_mask`](@ref). Use [`begin_hole!`](@ref) to
draw a region that is restored inside an exclusion polygon. Seed `polygons`
to resume editing an existing set.
"""
struct MaskEditor
    size::Dims{2}
    polygons::Observable{Vector{Vector{Tuple{Float64,Float64}}}}
    holes::Observable{Vector{Bool}}
    active::Observable{Vector{Tuple{Float64,Float64}}}
    hole_mode::Observable{Bool}
    selected::Observable{Union{Nothing,Int}}
    show_mask::Observable{Bool}
    raster::Observable{Union{Nothing,BitMatrix}}
end

function MaskEditor(sz::Dims{2}; polygons = Vector{Tuple{Float64,Float64}}[],
                    holes = falses(length(polygons)), raster = nothing)
    all(>(0), sz) || throw(ArgumentError("mask editor needs a nonempty image size, got $sz"))
    polys = [[(Float64(v[1]), Float64(v[2])) for v in p] for p in polygons]
    all(p -> length(p) >= 3, polys) ||
        throw(ArgumentError("every seeded polygon needs at least 3 vertices"))
    length(holes) == length(polys) ||
        throw(ArgumentError("holes must have one entry per seeded polygon"))
    raster === nothing || size(raster) == sz ||
        throw(DimensionMismatch("raster mask size $(size(raster)) does not match the editor size $sz"))
    return MaskEditor(sz, Observable(polys), Observable(Bool.(holes)),
                      Observable(Tuple{Float64,Float64}[]), Observable(false),
                      Observable{Union{Nothing,Int}}(nothing), Observable(false),
                      Observable{Union{Nothing,BitMatrix}}(raster === nothing ? nothing : BitMatrix(raster)))
end

MaskEditor(image::AbstractMatrix; kwargs...) = MaskEditor(size(image); kwargs...)

function Base.show(io::IO, me::MaskEditor)
    nr, nc = me.size
    print(io, "MaskEditor($(nc)×$(nr) image, $(length(me.polygons[])) polygon",
          length(me.polygons[]) == 1 ? "" : "s",
          isempty(me.active[]) ? ")" : ", drawing)")
end

"""
    add_vertex!(me::MaskEditor, x, y)

Append a vertex to the active (in-progress) polygon.
"""
function add_vertex!(me::MaskEditor, x::Real, y::Real)
    push!(me.active[], (Float64(x), Float64(y)))
    notify(me.active)
    return me
end

"""
    undo_vertex!(me::MaskEditor)

Remove the last vertex of the active polygon (no-op when not drawing;
removing the only vertex cancels the polygon).
"""
function undo_vertex!(me::MaskEditor)
    isempty(me.active[]) && return me
    pop!(me.active[])
    notify(me.active)
    return me
end

"""
    close_active!(me::MaskEditor) -> Bool

Commit the active polygon (returns `true`) when it has ≥ 3 vertices;
otherwise discard it (cancel). No-op returning `false` when not drawing.
"""
function close_active!(me::MaskEditor)
    verts = me.active[]
    isempty(verts) && return false
    committed = length(verts) >= 3
    if committed
        push!(me.polygons[], copy(verts))
        push!(me.holes[], me.hole_mode[])
        notify(me.polygons)
        notify(me.holes)
    end
    empty!(verts)
    me.hole_mode[] = false
    notify(me.active)
    return committed
end

"""
    cancel_active!(me::MaskEditor) -> Bool

Discard the active polygon (and a pending hole); `false` when not drawing.
"""
function cancel_active!(me::MaskEditor)
    drawing = !isempty(me.active[]) || me.hole_mode[]
    drawing || return false
    empty!(me.active[])
    me.hole_mode[] = false
    notify(me.active)
    return true
end

"""Begin drawing a polygon that removes an area from the exclusion mask."""
function begin_hole!(me::MaskEditor)
    isempty(me.active[]) || throw(ArgumentError("finish the active polygon first"))
    me.hole_mode[] = true
    me.selected[] = nothing
    return me
end

# Even-odd point-in-polygon (matches Hammerhead.polygon_mask's fill rule).
function _inside(p::AbstractVector{Tuple{Float64,Float64}}, x::Real, y::Real)
    inside = false
    n = length(p)
    for k in 1:n
        x1, y1 = p[k]
        x2, y2 = p[mod1(k + 1, n)]
        ((y1 <= y < y2) || (y2 <= y < y1)) || continue
        x < x1 + (y - y1) / (y2 - y1) * (x2 - x1) && (inside = !inside)
    end
    return inside
end

"""
    polygon_at(me::MaskEditor, x, y) -> Union{Nothing,Int}

Index of the topmost (most recently drawn) committed polygon containing
`(x, y)`, or `nothing`.
"""
polygon_at(me::MaskEditor, x::Real, y::Real) =
    findlast(p -> _inside(p, x, y), me.polygons[])

"""
    click!(me::MaskEditor, x, y)

Primary-click gesture: while drawing, add a vertex; otherwise select the
polygon under the click, or (on empty background) deselect and start a new
polygon at `(x, y)`.
"""
function click!(me::MaskEditor, x::Real, y::Real)
    if !isempty(me.active[])
        return add_vertex!(me, x, y)
    end
    if me.hole_mode[]
        return add_vertex!(me, x, y)
    end
    hit = polygon_at(me, x, y)
    if hit === nothing
        me.selected[] === nothing || (me.selected[] = nothing)
        add_vertex!(me, x, y)
    else
        me.selected[] = hit
    end
    return me
end

"""
    alt_click!(me::MaskEditor)

Secondary-click gesture: close the active polygon when drawing, otherwise
drop the selection.
"""
function alt_click!(me::MaskEditor)
    if !isempty(me.active[])
        close_active!(me)
    else
        me.selected[] === nothing || (me.selected[] = nothing)
    end
    return me
end

"""
    delete_selected!(me::MaskEditor)

Delete the selected polygon (no-op without a selection).
"""
function delete_selected!(me::MaskEditor)
    i = me.selected[]
    i === nothing && return me
    deleteat!(me.polygons[], i)
    deleteat!(me.holes[], i)
    me.selected[] = nothing
    notify(me.polygons)
    notify(me.holes)
    return me
end

"""
    clear_polygons!(me::MaskEditor)

Delete all polygons and cancel any active drawing.
"""
function clear_polygons!(me::MaskEditor)
    empty!(me.polygons[])
    empty!(me.holes[])
    empty!(me.active[])
    me.raster[] = nothing
    me.hole_mode[] = false
    me.selected[] = nothing
    notify(me.polygons)
    notify(me.holes)
    notify(me.active)
    return me
end

"""
    polygon_mask(me::MaskEditor) -> BitMatrix

Return the image-sized `BitMatrix` mask (`true` means excluded), including
holes and any rasterized edits. Pass it as `mask` to `run_piv`. If no edits
are present, all values are `false`.
"""
function Hammerhead.polygon_mask(me::MaskEditor)
    mask = me.raster[] === nothing ? falses(me.size) : copy(me.raster[])
    for (p, hole) in zip(me.polygons[], me.holes[])
        pm = polygon_mask(me.size, p)
        hole ? (mask .&= .!pm) : (mask .|= pm)
    end
    return mask
end


"""
    set_raster!(me::MaskEditor, mask)

Replace the editor's content with a raster `mask` (`nothing` clears it):
polygons are dropped, and new polygons are drawn on top of the raster.
"""
function set_raster!(me::MaskEditor, mask::Union{Nothing,AbstractMatrix{Bool}})
    mask === nothing || size(mask) == me.size ||
        throw(DimensionMismatch("mask size $(size(mask)) does not match the editor size $(me.size)"))
    empty!(me.polygons[]); empty!(me.holes[]); empty!(me.active[])
    me.hole_mode[] = false
    me.selected[] = nothing
    me.raster[] = mask === nothing ? nothing : BitMatrix(mask)
    notify(me.polygons); notify(me.holes); notify(me.active)
    return me
end

"""
    has_mask(me::MaskEditor) -> Bool

Whether the editor holds any committed polygon or raster.
"""
has_mask(me::MaskEditor) = me.raster[] !== nothing || !isempty(me.polygons[])

"""Rasterize the mask, expand excluded pixels by `radius`, and return `me`.
Existing polygons become a raster mask and can no longer be edited as polygons.
"""
function grow_mask!(me::MaskEditor, radius::Integer = 1)
    me.raster[] = Hammerhead.grow_mask(polygon_mask(me), radius)
    empty!(me.polygons[]); empty!(me.holes[])
    notify(me.polygons); notify(me.holes)
    return me
end

"""Rasterize the mask, reduce excluded pixels by `radius`, and return `me`.
Existing polygons become a raster mask and can no longer be edited as polygons.
"""
function shrink_mask!(me::MaskEditor, radius::Integer = 1)
    me.raster[] = Hammerhead.shrink_mask(polygon_mask(me), radius)
    empty!(me.polygons[]); empty!(me.holes[])
    notify(me.polygons); notify(me.holes)
    return me
end

"""
    save_mask(me::MaskEditor, path) -> path

Write the combined mask as a grayscale image (white = excluded), the
convention `Hammerhead.load_mask` reads back by default.
"""
function save_mask(me::MaskEditor, path::AbstractString)
    FileIO.save(path, Gray.(polygon_mask(me)))
    return path
end

"""
    status_text(me::MaskEditor) -> String

One-line summary of the editing state (view status bar).
"""
function status_text(me::MaskEditor)
    n = length(me.polygons[])
    parts = ["$n polygon" * (n == 1 ? "" : "s")]
    isempty(me.active[]) || push!(parts, "drawing ($(length(me.active[])) vertices)")
    me.hole_mode[] && isempty(me.active[]) && push!(parts, "next polygon is a hole")
    me.selected[] === nothing || push!(parts, "selected: $(me.selected[])")
    return join(parts, " · ")
end
