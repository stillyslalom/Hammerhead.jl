# Framework-free rectangular image selection; the processing contract is core ROI.

"""
    ROIEditor(image_or_path; roi = nothing)

Edit an inclusive rectangular [`ROI`](@ref) in original image pixels.
`roi` and the pending first corner (`anchor`) are observables. Two calls to
[`click!`](@ref) select opposite corners; numeric bounds can be supplied with
[`set_roi!`](@ref). Clearing selects the full image (`roi = nothing`).
The image is copied for display; selections never crop or mutate it.
"""
struct ROIEditor
    image::Matrix{Float64}
    roi::Observable{Union{Nothing,ROI}}
    anchor::Observable{Union{Nothing,NTuple{2,Int}}}
end

function ROIEditor(image::AbstractMatrix{<:Real}; roi = nothing)
    isempty(image) && throw(ArgumentError("ROI editor needs a nonempty image"))
    ed = ROIEditor(Matrix{Float64}(image),
                   Observable{Union{Nothing,ROI}}(nothing),
                   Observable{Union{Nothing,NTuple{2,Int}}}(nothing))
    set_roi!(ed, roi)
    return ed
end
ROIEditor(path::AbstractString; kwargs...) = ROIEditor(load_image(path); kwargs...)

_as_roi(::Nothing) = nothing
_as_roi(roi::ROI) = roi
_as_roi(roi::Tuple) = ROI(roi)

function _check_roi(image, roi::ROI)
    # Keep bounds and full-image/cropped-mask behavior defined by the core.
    Hammerhead.roi_views(image, image, nothing, roi)
    return roi
end

"""
    set_roi!(editor::ROIEditor, roi)
    set_roi!(editor::ROIEditor, row_first, row_last, col_first, col_last)

Set inclusive integer bounds, a core `ROI`, or a `(rows, cols)` range tuple.
Numeric bounds also accept integer strings. Invalid, reversed, or out-of-image
bounds throw without changing the selection. Pass `nothing` to reset.
"""
function set_roi!(ed::ROIEditor, roi)
    rr = _as_roi(roi)
    rr === nothing || _check_roi(ed.image, rr)
    ed.anchor[] = nothing
    ed.roi[] = rr
    return ed
end

_roi_index(n::Integer) = Int(n)
function _roi_index(s::AbstractString)
    n = tryparse(Int, strip(s))
    n === nothing && throw(ArgumentError("ROI bounds must be integers, got \"$s\""))
    return n
end
set_roi!(ed::ROIEditor, r1, r2, c1, c2) =
    set_roi!(ed, ROI(_roi_index(r1):_roi_index(r2), _roi_index(c1):_roi_index(c2)))

"""
    click!(editor::ROIEditor, x::Real, y::Real)

Select opposite corners with two clicks (`x` = column, `y` = row).
Coordinates snap to the nearest pixel and clamp to the image boundary.
A third click begins a new rectangle while retaining the current ROI until
the second corner is placed. Nonfinite coordinates throw.
"""
function click!(ed::ROIEditor, x::Real, y::Real)
    isfinite(x) && isfinite(y) || throw(ArgumentError("ROI coordinates must be finite"))
    nr, nc = size(ed.image)
    col = round(Int, clamp(x, 1, nc))
    row = round(Int, clamp(y, 1, nr))
    first_corner = ed.anchor[]
    if first_corner === nothing
        ed.anchor[] = (col, row)
    else
        c, r = first_corner
        set_roi!(ed, ROI(min(r, row):max(r, row), min(c, col):max(c, col)))
    end
    return ed
end

"""
    clear_roi!(editor::ROIEditor)

Reset to the full image and discard a pending first corner.
"""
clear_roi!(ed::ROIEditor) = set_roi!(ed, nothing)

"""
    roi_summary(editor::ROIEditor) -> String

Describe the selected inclusive pixel bounds, or the full-image selection.
"""
function roi_summary(ed::ROIEditor)
    roi = ed.roi[]
    text = roi === nothing ? "full image" :
        "rows $(first(roi.rows)):$(last(roi.rows)), columns $(first(roi.cols)):$(last(roi.cols)) ($(length(roi.rows)) × $(length(roi.cols)) px)"
    return ed.anchor[] === nothing ? text : text * "\nclick the opposite corner"
end
