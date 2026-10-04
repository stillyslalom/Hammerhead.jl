# Workflow canvases. Under Qt a canvas has a current GL context only while Qt
# renders it, and GLMakie creates or frees GPU objects as soon as a plot is
# added to or removed from a displayed figure. So every plot of a canvas is
# created before the figure is first shown, and afterwards only its inputs
# change, atomically through `update!` (applied lazily at render time).

const VALID_COLOR = RGBf(0.20, 0.82, 1.0)
const FLAGGED_COLOR = RGBf(1.0, 0.30, 0.30)
const BOX_COLORS = (RGBf(1.0, 0.80, 0.20), RGBf(0.55, 0.95, 0.45), RGBf(0.95, 0.45, 0.95),
                    RGBf(0.40, 0.75, 1.0))
const _NOPOINT = [Point2f(NaN, NaN)]

# Atomic update of a plot's positional arguments and attributes (positional
# data go in as `arg1…argN`; a lone positional would hit the Dict method).
_update!(plot, args...; attrs...) =
    Makie.update!(plot; (Symbol(:arg, i) => a for (i, a) in enumerate(args))..., attrs...)
const _EMPTY_IMAGE = fill(NaN32, 2, 2)

"""
    PlanarCanvas

The image canvas of a planar workflow window: `fig`/`ax`, the plots it
updates, and a `dirty` flag the shell reads to request a redraw.
"""
struct PlanarCanvas
    fig::Figure
    ax::Axis
    dirty::Base.RefValue{Bool}
    frame::Any
    mask::Any
    roi::Any
    boxes::Any
    box_labels::Any
    shafts::Any
    heads::Any
    frame_size::Base.RefValue{Union{Nothing,Dims{2}}}
end

"""
    planar_canvas(wf::PlanarWorkflow) -> PlanarCanvas

Build the canvas for `wf` and keep it in sync with the workflow.
"""
function planar_canvas(wf::PlanarWorkflow)
    fig = Figure(size = (900, 800), figure_padding = 6)
    ax = Axis(fig[1, 1]; aspect = DataAspect(), yreversed = true,
              xlabel = "x (px)", ylabel = "y (px)")
    frame = heatmap!(ax, 1:2, 1:2, _EMPTY_IMAGE; colormap = :grays, nan_color = :transparent)
    mask = heatmap!(ax, 1:2, 1:2, _EMPTY_IMAGE; colormap = [RGBAf(1, 0.35, 0.2, 0.45)],
                    colorrange = (0, 1), nan_color = :transparent)
    roi = lines!(ax, _NOPOINT; color = :white, linestyle = :dash, linewidth = 1.5)
    boxes = lines!(ax, _NOPOINT; color = [BOX_COLORS[1]], linewidth = 2)
    box_labels = text!(ax, _NOPOINT; text = [""], color = [BOX_COLORS[1]], fontsize = 13,
                       align = (:left, :bottom), offset = (2, 2))
    shafts = linesegments!(ax, [Point2f(NaN, NaN), Point2f(NaN, NaN)];
                           color = [VALID_COLOR, VALID_COLOR], linewidth = 1.4)
    heads = scatter!(ax, _NOPOINT; marker = :utriangle, rotation = [0.0f0], markersize = 8,
                     color = [VALID_COLOR])
    translate!(frame, 0, 0, -10)
    translate!(mask, 0, 0, -5)
    c = PlanarCanvas(fig, ax, Ref(true), frame, mask, roi, boxes, box_labels, shafts, heads,
                     Ref{Union{Nothing,Dims{2}}}(nothing))
    fs = wf.frames
    onany((_...) -> _draw_frame!(c, wf), fs.files, fs.pair_mode, fs.pair, fs.shown)
    onany((_...) -> _draw_geometry!(c, wf), wf.mask, wf.roi)
    onany((_...) -> _draw_boxes!(c, wf), wf.step, wf.passes.passes, fs.files, fs.pair, wf.roi)
    onany((_...) -> _draw_vectors!(c, wf), wf.step, wf.test.result, wf.run.completed)
    _draw_frame!(c, wf)
    _draw_geometry!(c, wf)
    _draw_vectors!(c, wf)
    return c
end

function _draw_frame!(c::PlanarCanvas, wf::PlanarWorkflow)
    img = try
        shown_image(wf.frames)
    catch
        nothing
    end
    if img === nothing
        _update!(c.frame, 1:2, 1:2, _EMPTY_IMAGE)
        c.frame_size[] = nothing
    else
        nr, nc = size(img)
        _update!(c.frame, 1:nc, 1:nr, permutedims(img))
        # new frame dimensions: show the whole frame (keep the zoom otherwise)
        if size(img) != c.frame_size[]
            c.frame_size[] = size(img)
            reset_limits!(c.ax)
        end
    end
    _draw_boxes!(c, wf)
    c.dirty[] = true
    return c
end

function _draw_geometry!(c::PlanarCanvas, wf::PlanarWorkflow)
    m = wf.mask[]
    if m === nothing
        _update!(c.mask, 1:2, 1:2, _EMPTY_IMAGE)
    else
        nr, nc = size(m)
        shade = permutedims(Float32.(m))
        shade[shade .== 0] .= NaN32
        _update!(c.mask, 1:nc, 1:nr, shade)
    end
    roi = wf.roi[]
    if roi === nothing
        _update!(c.roi, _NOPOINT)
    else
        r0, r1 = first(roi.rows) - 0.5f0, last(roi.rows) + 0.5f0
        c0, c1 = first(roi.cols) - 0.5f0, last(roi.cols) + 0.5f0
        _update!(c.roi, Point2f[(c0, r0), (c1, r0), (c1, r1), (c0, r1), (c0, r0)])
    end
    c.dirty[] = true
    return c
end

# Interrogation-window outlines for each distinct pass size, centered on the
# analysis region, so the user can judge them against the particle images.
function _draw_boxes!(c::PlanarCanvas, wf::PlanarWorkflow)
    sz = c.frame_size[]
    pts = Point2f[]; cols = RGBf[]; labels = String[]; label_pos = Point2f[]; label_cols = RGBf[]
    if wf.step[] === :passes && sz !== nothing
        roi = wf.roi[]
        cy = roi === nothing ? (sz[1] + 1) / 2 : (first(roi.rows) + last(roi.rows)) / 2
        cx = roi === nothing ? (sz[2] + 1) / 2 : (first(roi.cols) + last(roi.cols)) / 2
        seen = Int[]
        for p in wf.passes.passes[]
            w = p.window_size[1]
            w in seen && continue
            push!(seen, w)
            col = BOX_COLORS[mod1(length(seen), length(BOX_COLORS))]
            h = w / 2
            append!(pts, Point2f[(cx - h, cy - h), (cx + h, cy - h), (cx + h, cy + h),
                                 (cx - h, cy + h), (cx - h, cy - h), (NaN, NaN)])
            append!(cols, fill(col, 6))
            push!(labels, "$w px"); push!(label_pos, Point2f(cx - h, cy - h)); push!(label_cols, col)
        end
    end
    if isempty(pts)
        _update!(c.boxes, _NOPOINT; color = [BOX_COLORS[1]])
        _update!(c.box_labels, _NOPOINT; text = [""], color = [BOX_COLORS[1]])
    else
        _update!(c.boxes, pts; color = cols)
        _update!(c.box_labels, label_pos; text = labels, color = label_cols)
    end
    c.dirty[] = true
    return c
end

# Vectors of the test result (Test step) or the latest finished pair (Run).
function _draw_vectors!(c::PlanarCanvas, wf::PlanarWorkflow)
    r = wf.step[] === :test ? wf.test.result[] :
        wf.step[] === :run && !isempty(wf.run.completed[]) ? last(wf.run.completed[]) : nothing
    _update_arrows!(c.shafts, c.heads, r)
    c.dirty[] = true
    return c
end

# Quiver-style arrows: linesegment shafts + rotated triangle heads, colored
# by validity (static in data space, cheap to pan and zoom).
function _update_arrows!(shafts, heads, r; lengthscale = nothing,
                         valid_color = VALID_COLOR, flagged_color = FLAGGED_COLOR)
    d = r isa Controllers.GridResult ? vector_data(r) : nothing
    if d === nothing || isempty(d.x)
        _update!(shafts, [Point2f(NaN, NaN), Point2f(NaN, NaN)]; color = [valid_color, valid_color])
        _update!(heads, _NOPOINT; rotation = [0.0f0], color = [valid_color])
        return
    end
    ls = lengthscale === nothing ? auto_lengthscale(r, d) : lengthscale
    n = length(d.x)
    segs = Vector{Point2f}(undef, 2n)
    tips = Vector{Point2f}(undef, n)
    rots = Vector{Float32}(undef, n)
    for k in 1:n
        tip = Point2f(d.x[k] + ls * d.u[k], d.y[k] + ls * d.v[k])
        segs[2k - 1] = Point2f(d.x[k], d.y[k])
        segs[2k] = tip
        tips[k] = tip
        # screen-space CCW rotation on a y-reversed axis; :utriangle points up
        rots[k] = Float32(atan(-d.v[k], d.u[k]) - π / 2)
    end
    cols = [o ? flagged_color : valid_color for o in d.outlier]
    _update!(shafts, segs; color = repeat(cols, inner = 2))
    _update!(heads, tips; rotation = rots, color = cols)
    return
end
