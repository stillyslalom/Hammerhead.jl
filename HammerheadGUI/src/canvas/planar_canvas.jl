# Workflow canvases. Under Qt a canvas has a current GL context only while Qt
# renders it, and GLMakie creates or frees GPU objects as soon as a plot is
# added to or removed from a displayed figure. So every plot of a canvas is
# created before the figure is first shown, and afterwards only its inputs
# change, atomically through `update!` (applied lazily at render time).
# Overlays that are empty get a NaN placeholder instead of being removed.
#
# Gestures: clicks and keys only change controller state (`canvas_click!`,
# `canvas_alt_click!`, `canvas_key!`); the overlays follow the controller
# observables. A click edits, a drag navigates (left-drag zooms to a box,
# right-drag pans, scroll zooms).

const VALID_COLOR = RGBf(0.20, 0.82, 1.0)
const FLAGGED_COLOR = RGBf(1.0, 0.30, 0.30)
const BOX_COLORS = (RGBf(1.0, 0.80, 0.20), RGBf(0.55, 0.95, 0.45), RGBf(0.95, 0.45, 0.95),
                    RGBf(0.40, 0.75, 1.0))
const MASK_COLOR = RGBf(1.0, 0.45, 0.25)          # exclusion polygons
const HOLE_COLOR = RGBf(0.30, 0.85, 1.0)          # polygons restoring an area
const SELECTED_COLOR = RGBf(1.0, 0.92, 0.25)
const TOOL_COLOR = RGBf(1.0, 0.92, 0.25)          # ROI corner, scale line, probe
const PARTICLE_COLOR = RGBf(1.0, 0.85, 0.25)      # detected particles
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
    polygons::Any            # committed mask polygons (Mask page)
    active_line::Any         # polygon being drawn
    active_points::Any
    roi_corner::Any          # pending first ROI corner
    scale_line::Any
    scale_points::Any
    scale_label::Any
    probe_box::Any
    particles::Any           # detection preview (particle modes, Passes step)
    tracks::Any              # tracking test/run trajectories
    frame_size::Base.RefValue{Union{Nothing,Dims{2}}}
    shown::Base.RefValue{Any}  # the matrix the frame heatmap shows
    contrast::Base.RefValue{Bool}  # auto contrast (display only)
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
    polygons = lines!(ax, _NOPOINT; color = [MASK_COLOR], linewidth = 2)
    active_line = lines!(ax, _NOPOINT; color = MASK_COLOR, linewidth = 2, linestyle = :dash)
    active_points = scatter!(ax, _NOPOINT; color = MASK_COLOR, markersize = 9,
                             strokecolor = :black, strokewidth = 1)
    roi_corner = scatter!(ax, _NOPOINT; color = TOOL_COLOR, marker = :cross, markersize = 16)
    scale_line = lines!(ax, _NOPOINT; color = TOOL_COLOR, linewidth = 2)
    scale_points = scatter!(ax, _NOPOINT; color = TOOL_COLOR, markersize = 10,
                            strokecolor = :black, strokewidth = 1)
    scale_label = text!(ax, _NOPOINT; text = [""], color = TOOL_COLOR, fontsize = 14,
                        align = (:left, :bottom), offset = (6, 6))
    probe_box = lines!(ax, _NOPOINT; color = TOOL_COLOR, linewidth = 2)
    # circles the size of each detected particle (data-space marker size)
    particles = scatter!(ax, _NOPOINT; marker = Circle, markerspace = :data, markersize = [1.0f0],
                         color = :transparent, strokecolor = PARTICLE_COLOR, strokewidth = 1.2)
    tracks = lines!(ax, _NOPOINT; color = VALID_COLOR, linewidth = 1.2)
    translate!(frame, 0, 0, -10)
    translate!(mask, 0, 0, -5)
    c = PlanarCanvas(fig, ax, Ref(true), frame, mask, roi, boxes, box_labels, shafts, heads,
                     polygons, active_line, active_points, roi_corner, scale_line, scale_points,
                     scale_label, probe_box, particles, tracks, Ref{Union{Nothing,Dims{2}}}(nothing),
                     Ref{Any}(nothing), Ref(false))
    fs, ps = wf.frames, wf.prepare
    pp = ps.preview
    onany((_...) -> _draw_frame!(c, wf), fs.files, fs.pair_mode, fs.pair, fs.shown, fs.loaded,
          wf.step, ps.page, ps.show_processed, pp.processed, ps.ruler)
    fm = wf.frame_masks
    onany((_...) -> _draw_geometry!(c, wf), wf.mask, wf.roi, fm.files, fm.loaded, fs.pair)
    onany((_...) -> _draw_boxes!(c, wf), wf.step, wf.passes.passes, wf.passes.mode,
          wf.particles.predictor, fs.files, fs.pair, wf.roi)
    onany((_...) -> _draw_vectors!(c, wf), wf.step, wf.test.result, wf.run.completed)
    onany((_...) -> _draw_prepare!(c, wf), wf.step, ps.revision)
    onany((_...) -> _draw_particles!(c, wf), wf.step, wf.passes.mode, wf.particles.detected)
    _register_gestures!(c, wf)
    _draw_frame!(c, wf)
    _draw_geometry!(c, wf)
    _draw_vectors!(c, wf)
    _draw_prepare!(c, wf)
    _draw_particles!(c, wf)
    return c
end

# The detection preview: on the Passes step of a particle mode, a circle the
# size of each particle detected on the shown frame.
function _draw_particles!(c::PlanarCanvas, wf::PlanarWorkflow)
    p = wf.step[] === :passes && Controllers._particle_mode(wf.passes.mode[]) ?
        wf.particles.detected[] : nothing
    if p === nothing || length(p) == 0
        _update!(c.particles, _NOPOINT; markersize = [1.0f0])
    else
        _update!(c.particles, [Point2f(p.x[k], p.y[k]) for k in 1:length(p)];
                 markersize = [Float32(max(d, 2)) for d in p.diameter])
    end
    c.dirty[] = true
    return c
end

# Clicks become `canvas_click!`/`canvas_alt_click!` (consumed only when the
# controller used them, so the Axis keeps its own click behavior); a quick
# second click arrives as a double click and counts as another click. Keys
# go to `canvas_key!` while the canvas has focus. Shared by the workflow
# canvases (planar and stereo).
_register_gestures!(c::PlanarCanvas, wf::PlanarWorkflow) = (_register_workflow_gestures!(c.ax, c.fig, wf); c)

function _register_workflow_gestures!(ax::Axis, fig::Figure, wf::AbstractWorkflow)
    register_interaction!(ax, :workflow_gesture) do event::MouseEvent, _
        # a click with a modifier is Makie's (ctrl-click resets the zoom)
        _modifier_held(fig) && return Consume(false)
        t = event.type
        used = if t === MouseEventTypes.leftclick || t === MouseEventTypes.leftdoubleclick
            _gesture(() -> canvas_click!(wf, event.data[1], event.data[2]), wf)
        elseif t === MouseEventTypes.rightclick || t === MouseEventTypes.rightdoubleclick
            _gesture(() -> canvas_alt_click!(wf), wf)
        else
            false
        end
        return Consume(used)
    end
    keys = Dict(Keyboard.backspace => :backspace, Keyboard.escape => :escape,
                Keyboard.delete => :delete)
    on(events(fig).keyboardbutton) do ev
        (ev.action === Keyboard.press || ev.action === Keyboard.repeat) || return Consume(false)
        key = get(keys, ev.key, nothing)
        key === nothing && return Consume(false)
        return Consume(_gesture(() -> canvas_key!(wf, key), wf))
    end
    return ax
end

"""
View modes of the window's viewers, chosen in the viewer toolbar: `:edit`
(clicks work on the open step — probe, mask, profile …; left-drag zooms to a
box, right-drag pans), `:zoom` (drag a box to zoom; clicks do not edit), and
`:pan` (left-drag pans).
"""
const VIEW_MODES = (:edit, :zoom, :pan)

"""
    set_view_mode!(ax::Axis, mode::Symbol, gesture::Symbol)

Apply a [`VIEW_MODES`](@ref) mode to a viewer axis whose click gestures are
the interaction named `gesture`.
"""
function set_view_mode!(ax::Axis, mode::Symbol, gesture::Symbol)
    mode in VIEW_MODES || throw(ArgumentError("view mode must be one of $(VIEW_MODES), got :$mode"))
    if mode === :pan
        deactivate_interaction!(ax, :rectanglezoom)
        ax.panbutton = Mouse.left
    else
        activate_interaction!(ax, :rectanglezoom)
        ax.panbutton = Mouse.right
    end
    mode === :edit ? activate_interaction!(ax, gesture) : deactivate_interaction!(ax, gesture)
    return ax
end

const _MODIFIER_KEYS = (Keyboard.left_control, Keyboard.right_control, Keyboard.left_shift,
                        Keyboard.right_shift, Keyboard.left_alt, Keyboard.right_alt,
                        Keyboard.left_super, Keyboard.right_super)
_modifier_held(fig) = any(k -> k in events(fig).keyboardstate, _MODIFIER_KEYS)

"""
    set_contrast!(canvas, wf, auto::Bool)

Show frames on the image canvas with their full intensity range, or (`auto`)
stretched to the 0.5–99.5 % percentile band. Display only: the analysis
always uses the frames' values.
"""
function set_contrast!(c, wf::AbstractWorkflow, auto::Bool)
    c.contrast[] == auto && return c
    c.contrast[] = auto
    c.shown[] = nothing                     # redraw the current frame
    _draw_frame!(c, wf)
    return c
end

# The heatmap's color range for a frame: its extrema, or the 0.5–99.5 %
# band of a subsample (auto contrast).
function _frame_colorrange(img::AbstractMatrix, auto::Bool)
    if !auto
        lo, hi = Inf32, -Inf32
        for v in img
            isfinite(v) && ((lo, hi) = (min(lo, v), max(hi, v)))
        end
    else
        stride = max(1, length(img) ÷ 250_000)
        vals = Float32[img[k] for k in 1:stride:length(img) if isfinite(img[k])]
        isempty(vals) && return (0.0f0, 1.0f0)
        sort!(vals)
        n = length(vals)
        lo, hi = vals[clamp(round(Int, 0.005 * (n - 1)) + 1, 1, n)],
                 vals[clamp(round(Int, 0.995 * (n - 1)) + 1, 1, n)]
    end
    isfinite(lo) || return (0.0f0, 1.0f0)
    hi > lo || (hi = lo + 1)
    return (Float32(lo), Float32(hi))
end

# A failing gesture reports in the status line instead of breaking Makie's
# event handling.
function _gesture(f, wf::AbstractWorkflow)
    try
        return f()::Bool
    catch err
        wf.status[] = "error: " * Controllers._errmsg(err)
        return true
    end
end

# The frame the canvas shows: the processed preview on the Prepare step's
# Preprocess page when selected, otherwise the raw frame.
function _canvas_image(wf::PlanarWorkflow)
    ps, fs = wf.prepare, wf.frames
    if wf.step[] === :prepare && ps.page[] === :preprocess && ps.show_processed[]
        pp = ps.preview
        img = fs.shown[] === :a ? pp.processed[] : pp.processed2[]
        img === nothing || return img
    end
    wf.step[] === :prepare && ps.page[] === :scale && ps.ruler[] !== nothing && return ps.ruler[]
    return try
        shown_image(fs)
    catch
        nothing
    end
end

function _draw_frame!(c::PlanarCanvas, wf::PlanarWorkflow)
    img = _canvas_image(wf)
    # while the pair loads on a worker, keep showing the previous frame
    img === nothing && current_pair(wf.frames) !== nothing && pair_loading(wf.frames) && return c
    img === c.shown[] && return c
    c.shown[] = img
    if img === nothing
        _update!(c.frame, 1:2, 1:2, _EMPTY_IMAGE)
        c.frame_size[] = nothing
    else
        nr, nc = size(img)
        _update!(c.frame, 1:nc, 1:nr, Float32.(permutedims(img));
                 colorrange = _frame_colorrange(img, c.contrast[]))
        # new frame dimensions: show the whole frame (keep the zoom otherwise);
        # explicit limits, as the heatmap's new data applies only at render time
        if size(img) != c.frame_size[]
            c.frame_size[] = size(img)
            c.ax.limits[] = ((0.5, nc + 0.5), (0.5, nr + 0.5))
            reset_limits!(c.ax)
        end
    end
    _draw_boxes!(c, wf)
    c.dirty[] = true
    return c
end

# A polygon as a closed line, NaN-terminated so rings join into one plot.
function _closed_ring(p)
    pts = [Point2f(v...) for v in p]
    return push!(pts, pts[1], Point2f(NaN, NaN))
end

# Editing overlays of the Prepare step's open sub-page (NaN placeholders on
# every other page and step).
function _draw_prepare!(c::PlanarCanvas, wf::PlanarWorkflow)
    ps = wf.prepare
    page = wf.step[] === :prepare ? ps.page[] : :none
    _draw_mask_editor!(c.polygons, c.active_line, c.active_points, page === :mask ? ps.mask[] : nothing)

    ed = ps.roi[]
    corner = page === :roi && ed !== nothing ? ed.anchor[] : nothing
    _update!(c.roi_corner, corner === nothing ? _NOPOINT : [Point2f(corner...)])

    st = ps.scale[]
    pts = page === :scale && st !== nothing ? [Point2f(p...) for p in st.points[]] : Point2f[]
    _update!(c.scale_points, isempty(pts) ? _NOPOINT : pts)
    if length(pts) == 2
        d = pixel_distance(st)
        label = d === nothing ? "" : string(round(d; digits = 1), " px")
        _update!(c.scale_line, pts)
        _update!(c.scale_label, [(pts[1] + pts[2]) / 2]; text = [label])
    else
        _update!(c.scale_line, _NOPOINT)
        _update!(c.scale_label, _NOPOINT; text = [""])
    end

    probe = page === :preprocess || (wf.step[] === :passes && passes_probe_available(wf))
    _draw_probe!(c.probe_box, probe ? ps.preview : nothing, c.frame_size[])
    c.dirty[] = true
    return c
end

# The mask editor's committed polygons and the polygon being drawn
# (`me === nothing`: placeholders).
function _draw_mask_editor!(polygons, active_line, active_points, me)
    if me !== nothing
        pts = Point2f[]; cols = RGBf[]
        for (k, (p, hole)) in enumerate(zip(me.polygons[], me.holes[]))
            ring = _closed_ring(p)
            col = me.selected[] == k ? SELECTED_COLOR : hole ? HOLE_COLOR : MASK_COLOR
            append!(pts, ring); append!(cols, fill(col, length(ring)))
        end
        isempty(pts) ? _update!(polygons, _NOPOINT; color = [MASK_COLOR]) :
                       _update!(polygons, pts; color = cols)
        act = [Point2f(v...) for v in me.active[]]
        col = me.hole_mode[] ? HOLE_COLOR : MASK_COLOR
        _update!(active_line, length(act) >= 2 ? act : _NOPOINT; color = col)
        _update!(active_points, isempty(act) ? _NOPOINT : act; color = col)
    else
        _update!(polygons, _NOPOINT; color = [MASK_COLOR])
        _update!(active_line, _NOPOINT)
        _update!(active_points, _NOPOINT)
    end
    return
end

# The correlation probe's window outline (`pp === nothing`: placeholder).
function _draw_probe!(probe_box, pp, frame_size)
    rect = pp !== nothing && pp.probe[] !== nothing && frame_size !== nothing ?
           probe_rect(pp.probe[], pp.probe_window[], frame_size) : nothing
    if rect === nothing
        _update!(probe_box, _NOPOINT)
    else
        x0, y0 = rect.x0 - 0.5f0, rect.y0 - 0.5f0
        x1, y1 = x0 + rect.window, y0 + rect.window
        _update!(probe_box, Point2f[(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)])
    end
    return
end

function _draw_geometry!(c::PlanarCanvas, wf::PlanarWorkflow)
    _draw_raster!(c.mask, representative_mask(wf))
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

# A Bool raster as a shade over its `true` pixels (`nothing`: placeholder).
function _draw_raster!(hm, m)
    if m === nothing
        _update!(hm, 1:2, 1:2, _EMPTY_IMAGE)
    else
        nr, nc = size(m)
        shade = permutedims(Float32.(m))
        shade[shade .== 0] .= NaN32
        _update!(hm, 1:nc, 1:nr, shade)
    end
    return
end

# Interrogation-window outlines for each distinct pass size, centered on the
# analysis region, so the user can judge them against the particle images.
function _draw_boxes!(c::PlanarCanvas, wf::PlanarWorkflow)
    sz = c.frame_size[]
    center = nothing
    piv_windows = !Controllers._particle_mode(wf.passes.mode[]) || wf.particles.predictor[] === :piv
    if wf.step[] === :passes && sz !== nothing && piv_windows
        roi = wf.roi[]
        cy = roi === nothing ? (sz[1] + 1) / 2 : (first(roi.rows) + last(roi.rows)) / 2
        cx = roi === nothing ? (sz[2] + 1) / 2 : (first(roi.cols) + last(roi.cols)) / 2
        center = (cx, cy)
    end
    _draw_window_boxes!(c.boxes, c.box_labels, center === nothing ? () : wf.passes.passes[], center)
    c.dirty[] = true
    return c
end

# One outline per distinct window size of `passes`, centered on `center`
# (no passes: placeholders).
function _draw_window_boxes!(boxes, box_labels, passes, center)
    pts = Point2f[]; cols = RGBf[]; labels = String[]; label_pos = Point2f[]; label_cols = RGBf[]
    seen = Int[]
    for p in passes
        cx, cy = center
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
    if isempty(pts)
        _update!(boxes, _NOPOINT; color = [BOX_COLORS[1]])
        _update!(box_labels, _NOPOINT; text = [""], color = [BOX_COLORS[1]])
    else
        _update!(boxes, pts; color = cols)
        _update!(box_labels, label_pos; text = labels, color = label_cols)
    end
    return
end

# Vectors of the test result (Test step) or the latest finished pair (Run):
# arrows for PIV and PTV results, polylines for tracks.
function _draw_vectors!(c::PlanarCanvas, wf::PlanarWorkflow)
    r = wf.step[] === :test ? wf.test.result[] :
        wf.step[] === :run && !isempty(wf.run.completed[]) ? last(wf.run.completed[]) : nothing
    _update_arrows!(c.shafts, c.heads, r)
    if r isa TrackingResult && !isempty(r.trajectories)
        pts = Point2f[]
        for t in r.trajectories
            xs, ys = trajectory_points(t)
            append!(pts, Point2f.(xs, ys))
            push!(pts, Point2f(NaN, NaN))
        end
        _update!(c.tracks, pts)
    else
        _update!(c.tracks, _NOPOINT)
    end
    c.dirty[] = true
    return c
end

# Quiver-style arrows: linesegment shafts + rotated triangle heads, colored
# by validity (static in data space, cheap to pan and zoom).
function _update_arrows!(shafts, heads, r; lengthscale = nothing, kwargs...)
    d = r isa Union{Controllers.GridResult,PTVResult} ? vector_data(r) : nothing
    if d === nothing || isempty(d.x)
        _set_arrows!(shafts, heads, nothing, 1.0; kwargs...)
        return
    end
    ls = lengthscale === nothing ? auto_lengthscale(r, d) : lengthscale
    _set_arrows!(shafts, heads, d, ls; kwargs...)
    return
end

# Arrows of `d = (; x, y, u, v, outlier)` scaled by `ls` (`nothing`:
# placeholders). Head rotation is screen-space CCW; `yreversed` says whether
# the axis's y grows downward.
function _set_arrows!(shafts, heads, d, ls; valid_color = VALID_COLOR,
                      flagged_color = FLAGGED_COLOR, yreversed::Bool = true)
    if d === nothing || isempty(d.x)
        _update!(shafts, [Point2f(NaN, NaN), Point2f(NaN, NaN)]; color = [valid_color, valid_color])
        _update!(heads, _NOPOINT; rotation = [0.0f0], color = [valid_color])
        return
    end
    n = length(d.x)
    segs = Vector{Point2f}(undef, 2n)
    tips = Vector{Point2f}(undef, n)
    rots = Vector{Float32}(undef, n)
    for k in 1:n
        tip = Point2f(d.x[k] + ls * d.u[k], d.y[k] + ls * d.v[k])
        segs[2k - 1] = Point2f(d.x[k], d.y[k])
        segs[2k] = tip
        tips[k] = tip
        # :utriangle points up; on a y-reversed axis +v points down the screen
        rots[k] = Float32(atan(yreversed ? -d.v[k] : d.v[k], d.u[k]) - π / 2)
    end
    cols = [o ? flagged_color : valid_color for o in d.outlier]
    _update!(shafts, segs; color = repeat(cols, inner = 2))
    _update!(heads, tips; rotation = rots, color = cols)
    return
end
