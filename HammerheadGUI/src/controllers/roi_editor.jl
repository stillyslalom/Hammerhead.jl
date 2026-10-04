# Framework-free rectangular image selection; the processing contract is core ROI.

"""
    ROIEditor(size::Dims{2}; roi = nothing)
    ROIEditor(image::AbstractMatrix; roi = nothing)

Edit an inclusive rectangular [`ROI`](@ref) in original image pixels, for
images of `size` `(rows, cols)` (or the size of `image`; the editor keeps
only the size). `roi` and the pending first corner (`anchor`) are
observables. Two calls to [`click!`](@ref) select opposite corners; numeric
bounds can be supplied with [`set_roi!`](@ref). Clearing selects the full
image (`roi = nothing`).
"""
struct ROIEditor
    size::Dims{2}
    roi::Observable{Union{Nothing,ROI}}
    anchor::Observable{Union{Nothing,NTuple{2,Int}}}
end

function ROIEditor(sz::Dims{2}; roi = nothing)
    all(>(0), sz) || throw(ArgumentError("ROI editor needs a nonempty image size, got $sz"))
    ed = ROIEditor(sz, Observable{Union{Nothing,ROI}}(nothing),
                   Observable{Union{Nothing,NTuple{2,Int}}}(nothing))
    set_roi!(ed, roi)
    return ed
end
ROIEditor(image::AbstractMatrix; kwargs...) = ROIEditor(size(image); kwargs...)

_as_roi(::Nothing) = nothing
_as_roi(roi::ROI) = roi
_as_roi(roi::Tuple) = ROI(roi)

# The core ROI constructor rejects empty and nonpositive ranges.
function _check_roi(sz::Dims{2}, roi::ROI)
    last(roi.rows) <= sz[1] ||
        throw(ArgumentError("ROI rows end at $(last(roi.rows)), beyond the image's $(sz[1]) rows"))
    last(roi.cols) <= sz[2] ||
        throw(ArgumentError("ROI columns end at $(last(roi.cols)), beyond the image's $(sz[2]) columns"))
    return roi
end

"""
    set_roi!(editor::ROIEditor, roi)
    set_roi!(editor::ROIEditor, row_first, row_last, col_first, col_last)

Set inclusive integer bounds, a core `ROI`, or a `(rows, cols)` range tuple.
Numeric bounds also accept integer strings. Invalid, reversed, or out-of-image
bounds throw `ArgumentError` without changing the selection. Pass `nothing` to reset.
"""
function set_roi!(ed::ROIEditor, roi)
    rr = _as_roi(roi)
    rr === nothing || _check_roi(ed.size, rr)
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
    nr, nc = ed.size
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
    cancel_corner!(editor::ROIEditor) -> Bool

Discard a pending first corner (`false` when there is none).
"""
cancel_corner!(ed::ROIEditor) =
    ed.anchor[] === nothing ? false : (ed.anchor[] = nothing; true)

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
