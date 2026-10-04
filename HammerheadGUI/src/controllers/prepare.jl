# State of the workflow's Prepare step: which sub-page is open and the four
# editors behind them. The workflow's `preprocessing`, `mask`, `roi`, and
# `scale` stay the single source for the recipe; prepare_workflow.jl keeps
# the editors and those fields in sync and routes canvas gestures.

"""
Sub-pages of the planar Prepare step, in order.
"""
const PREPARE_PAGES = (:preprocess, :mask, :roi, :scale)

"""
    PrepareState(; runner = _inline_runner)

The Prepare step of a workflow (`wf.prepare`). `page` is the open sub-page
(one of `prepare_pages(wf)`). `preview` is a `PreprocessPreview` of the
representative pair; `mask`, `roi`, and `scale` hold a `MaskEditor`,
`ROIEditor`, and `ScaleTool` built for the current frame size (`nothing`
before a frame size is known). `show_processed` selects the processed
frame in the viewer on the Preprocess page.

In a `StereoWorkflow` the mask editor is sized to the dewarped grid
(`nothing` until there are dewarpers), and `roi` and `scale` stay `nothing`
(stereo has no ROI, and its scale has no measured line).

`revision` changes whenever any editor changes (canvases redraw on it).
`status` reports the background estimate (`background_running` while it
runs); `roi_error`/`scale_error` hold the last rejected ROI or scale edit.
`ruler` is an optional separate image (with `ruler_name`) on which the Scale
page measures the pixel size instead of the frames (see
[`load_ruler!`](@ref)).
"""
struct PrepareState
    page::Observable{Symbol}
    preview::PreprocessPreview
    mask::Observable{Union{Nothing,MaskEditor}}
    roi::Observable{Union{Nothing,ROIEditor}}
    scale::Observable{Union{Nothing,ScaleTool}}
    show_processed::Observable{Bool}
    revision::Observable{Int}
    status::Observable{String}
    background_running::Observable{Bool}
    roi_error::Observable{String}
    scale_error::Observable{String}
    background_generation::Base.RefValue{Int}
    syncing::Base.RefValue{Bool}           # set while one side writes the other
    ruler::Observable{Union{Nothing,Matrix{Float32}}}
    ruler_name::Observable{String}
end

PrepareState(; runner = _inline_runner) =
    PrepareState(Observable(:preprocess), PreprocessPreview(; runner),
                 Observable{Union{Nothing,MaskEditor}}(nothing),
                 Observable{Union{Nothing,ROIEditor}}(nothing),
                 Observable{Union{Nothing,ScaleTool}}(nothing),
                 Observable(false), Observable(0), Observable(""), Observable(false),
                 Observable(""), Observable(""), Ref(0), Ref(false),
                 Observable{Union{Nothing,Matrix{Float32}}}(nothing), Observable(""))

function Base.show(io::IO, ps::PrepareState)
    me = ps.mask[]
    print(io, "PrepareState(:", ps.page[], ", ", pipeline_summary(ps.preview),
          me === nothing ? ", no frame size)" : ", $(me.size[1])×$(me.size[2]) px)")
end

_bump!(ps::PrepareState) = (ps.revision[] += 1; ps)

# Run `f` with the sync guard set, so observers of the written side do not
# write back.
function _syncing(f, ps::PrepareState)
    ps.syncing[] && return nothing
    ps.syncing[] = true
    try
        return f()
    finally
        ps.syncing[] = false
    end
end
