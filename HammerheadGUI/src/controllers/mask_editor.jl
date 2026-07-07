# Mask-editor controller: polygon drawing/editing state over a reference
# image, exporting the package mask convention (image-sized Bool,
# `true` = excluded). Framework-free — the view forwards clicks and key
# presses into the gesture API below.

"""
    MaskEditor(image; polygons = [])
    MaskEditor(path::AbstractString)

Controller for the mask editor: a reference image (displayed as the drawing
background; the string form loads it with `Hammerhead.load_image`) plus the
editing state as `Observables` — `polygons` (committed exclusion polygons,
each a vector of `(x, y)` vertices in pixel coordinates), `active` (the
in-progress polygon), `selected` (index of the selected polygon or
`nothing`), and `show_mask` (overlay toggle).

Gestures ([`click!`](@ref), [`alt_click!`](@ref)) implement the editing
model: click to add vertices (a click on empty background starts a new
polygon; a click inside an existing polygon selects it), alt/right-click to
close the active polygon. Export the combined mask with
`polygon_mask(editor)` or [`save_mask`](@ref); seed `polygons` to resume
editing an existing set.
"""
struct MaskEditor
    image::Matrix{Float64}
    polygons::Observable{Vector{Vector{Tuple{Float64,Float64}}}}
    active::Observable{Vector{Tuple{Float64,Float64}}}
    selected::Observable{Union{Nothing,Int}}
    show_mask::Observable{Bool}
end

function MaskEditor(image::AbstractMatrix{<:Real}; polygons = Vector{Tuple{Float64,Float64}}[])
    polys = [[(Float64(v[1]), Float64(v[2])) for v in p] for p in polygons]
    all(p -> length(p) >= 3, polys) ||
        throw(ArgumentError("every seeded polygon needs at least 3 vertices"))
    return MaskEditor(Matrix{Float64}(image), Observable(polys),
                      Observable(Tuple{Float64,Float64}[]),
                      Observable{Union{Nothing,Int}}(nothing), Observable(false))
end

MaskEditor(path::AbstractString; kwargs...) = MaskEditor(load_image(path); kwargs...)

function Base.show(io::IO, me::MaskEditor)
    nr, nc = size(me.image)
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
        notify(me.polygons)
    end
    empty!(verts)
    notify(me.active)
    return committed
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
    me.selected[] = nothing
    notify(me.polygons)
    return me
end

"""
    clear_polygons!(me::MaskEditor)

Delete all polygons and cancel any active drawing.
"""
function clear_polygons!(me::MaskEditor)
    empty!(me.polygons[])
    empty!(me.active[])
    me.selected[] = nothing
    notify(me.polygons)
    notify(me.active)
    return me
end

"""
    polygon_mask(me::MaskEditor) -> BitMatrix

The combined exclusion mask of all committed polygons, in the package mask
convention (image-sized, `true` = excluded) — pass it as `mask` to
`run_piv`. All-`false` when no polygons are committed.
"""
function Hammerhead.polygon_mask(me::MaskEditor)
    mask = falses(size(me.image))
    for p in me.polygons[]
        mask .|= polygon_mask(size(me.image), p)
    end
    return mask
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
    me.selected[] === nothing || push!(parts, "selected: $(me.selected[])")
    return join(parts, " · ")
end
