# Image canvas of the stereo workflow window. Same rules as the planar
# canvas (planar_canvas.jl): every plot exists before the figure is first
# shown and afterwards only its inputs change; empty overlays hold NaN
# placeholders.
#
# What it shows depends on the step:
#   Images        the shown camera's raw frame (camera pixels)
#   Calibration   the shown camera's selected plate image with the detected
#                 dots (coloured by reprojection error), the fiducial markers,
#                 and the reprojection residuals as magnified arrows
#   later steps   the shown camera's frame dewarped onto the common grid
#                 (`wf.dewarped`, or the processed preview on Prepare ›
#                 Preprocess), with the cameras' out-of-view union shaded; the
#                 mask, editing overlays, window outlines, and test/run
#                 vectors are drawn in dewarped pixels (grid nodes: x =
#                 column, y = row), where `canvas_click!` expects them.
# Without dewarpers the later steps show the raw frame instead.

const OUT_OF_VIEW_COLOR = RGBAf(0.30, 0.32, 0.42, 0.65)
const RESIDUAL_COLOR = RGBf(0.0, 0.85, 1.0)

"""
    StereoCanvas

The image canvas of a stereo workflow window: `fig`/`ax`, the plots it
updates, and a `dirty` flag the shell reads to request a redraw. `space` says
what the frame heatmap shows: `:camera` (camera pixels), `:grid` (dewarped
pixels), or `:plate` (a calibration plate image).
"""
struct StereoCanvas
    fig::Figure
    ax::Axis
    dirty::Base.RefValue{Bool}
    frame::Any
    out_of_view::Any
    mask::Any
    boxes::Any
    box_labels::Any
    shafts::Any
    heads::Any
    polygons::Any            # committed mask polygons (Mask page)
    active_line::Any         # polygon being drawn
    active_points::Any
    probe_box::Any
    dots::Any                # detected calibration dots, coloured by error
    residuals::Any           # magnified reprojection residuals
    square::Any              # fiducial markers
    triangle::Any
    frame_size::Base.RefValue{Union{Nothing,Dims{2}}}
    space::Base.RefValue{Symbol}
    shown::Base.RefValue{Any}  # the matrix the frame heatmap shows
    title::Base.RefValue{String}
    note::Base.RefValue{String}  # calibration overlay legend, appended to the title
    residual_gain::Base.RefValue{Float64}
end

"""
    stereo_canvas(wf::StereoWorkflow) -> StereoCanvas

Build the canvas for `wf` and keep it in sync with the workflow.
"""
function stereo_canvas(wf::StereoWorkflow)
    fig = Figure(size = (900, 800), figure_padding = 6)
    ax = Axis(fig[1, 1]; aspect = DataAspect(), yreversed = true, xlabel = "x (px)",
              ylabel = "y (px)", title = "", titlealign = :left, titlesize = 14,
              titlefont = :regular)
    frame = heatmap!(ax, 1:2, 1:2, _EMPTY_IMAGE; colormap = :grays, nan_color = :transparent)
    oov = heatmap!(ax, 1:2, 1:2, _EMPTY_IMAGE; colormap = [OUT_OF_VIEW_COLOR],
                   colorrange = (0, 1), nan_color = :transparent)
    mask = heatmap!(ax, 1:2, 1:2, _EMPTY_IMAGE; colormap = [RGBAf(1, 0.35, 0.2, 0.45)],
                    colorrange = (0, 1), nan_color = :transparent)
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
    probe_box = lines!(ax, _NOPOINT; color = TOOL_COLOR, linewidth = 2)
    residuals = linesegments!(ax, [Point2f(NaN, NaN), Point2f(NaN, NaN)];
                              color = RESIDUAL_COLOR, linewidth = 1.5)
    dots = scatter!(ax, _NOPOINT; color = [0.0f0], colormap = :plasma, colorrange = (0, 1),
                    markersize = 7, strokecolor = :black, strokewidth = 0.5)
    square = scatter!(ax, _NOPOINT; marker = :rect, color = :transparent,
                      strokecolor = :cyan, strokewidth = 2, markersize = 20)
    triangle = scatter!(ax, _NOPOINT; marker = :utriangle, color = :transparent,
                        strokecolor = :cyan, strokewidth = 2, markersize = 20)
    translate!(frame, 0, 0, -10)
    translate!(oov, 0, 0, -6)
    translate!(mask, 0, 0, -5)
    c = StereoCanvas(fig, ax, Ref(true), frame, oov, mask, boxes, box_labels, shafts, heads,
                     polygons, active_line, active_points, probe_box, dots, residuals, square,
                     triangle, Ref{Union{Nothing,Dims{2}}}(nothing), Ref(:none), Ref{Any}(nothing),
                     Ref(""), Ref(""), Ref(1.0))
    fs1, fs2, ps, cal = wf.frames1, wf.frames2, wf.prepare, wf.calibration
    pp = ps.preview
    onany((_...) -> _draw_frame!(c, wf), fs1.files, fs1.pair_mode, fs1.pair, fs1.shown, fs1.loaded,
          fs2.files, fs2.loaded, wf.camera, wf.step, ps.page, ps.show_processed, pp.processed,
          pp.processed2, wf.dewarped, cal.reviews[1], cal.reviews[2], cal.plates[1], cal.plates[2])
    onany((_...) -> _draw_geometry!(c, wf), wf.mask, cal.dewarpers)
    onany((_...) -> _draw_boxes!(c, wf), wf.step, wf.passes.passes)
    onany((_...) -> _draw_vectors!(c, wf), wf.step, wf.test.result, wf.run.completed, cal.dewarpers)
    onany((_...) -> _draw_prepare!(c, wf), wf.step, ps.revision)
    onany((_...) -> _draw_calibration!(c, wf), wf.step, wf.camera, cal.reviews[1], cal.reviews[2])
    # a review's plane selection and refits (a new model) redraw the plate
    function watch_review(cr)
        cr === nothing && return
        onany((_...) -> (_draw_frame!(c, wf); _draw_calibration!(c, wf)), cr.plane, cr.camera)
        return
    end
    for k in 1:2
        on(watch_review, cal.reviews[k])
        watch_review(cal.reviews[k][])
    end
    _register_workflow_gestures!(ax, fig, wf)
    _draw_frame!(c, wf)
    _draw_all_overlays!(c, wf)
    return c
end

function _draw_all_overlays!(c::StereoCanvas, wf::StereoWorkflow)
    _draw_geometry!(c, wf)
    _draw_boxes!(c, wf)
    _draw_vectors!(c, wf)
    _draw_prepare!(c, wf)
    _draw_calibration!(c, wf)
    return c
end

function _shown_or_nothing(fs)
    return try
        shown_image(fs)
    catch
        nothing
    end
end

# (image, space, title) of the current step.
function _stereo_view(wf::StereoWorkflow)
    k = wf.camera[]
    fs = shown_frames(wf)
    which = fs.shown[] === :a ? "frame A" : "frame B"
    step = wf.step[]
    step === :images && return (_shown_or_nothing(fs), :camera, "camera $k · $which")
    step === :calibration && return _plate_view(wf)
    ps, pp = wf.prepare, wf.prepare.preview
    gsz = grid_size(wf)
    if step === :prepare && ps.page[] === :preprocess && ps.show_processed[]
        img = fs.shown[] === :a ? pp.processed[] : pp.processed2[]
        if img !== nothing
            grid = gsz !== nothing && size(img) == gsz
            return (img, grid ? :grid : :camera,
                    "camera $k · $which · processed" * (grid ? ", dewarped" : ""))
        end
    end
    dw = wf.dewarped[]
    if dw !== nothing && gsz !== nothing && size(dw[1]) == gsz
        img = fs.shown[] === :a || dw[2] === nothing ? dw[1] : dw[2]
        return (img, :grid, "camera $k · $which · dewarped")
    end
    note = wf.calibration.dewarpers[] === nothing ? " · not dewarped (calibrate first)" : " · dewarping…"
    return (_shown_or_nothing(fs), :camera, "camera $k · $which" * note)
end

function _plate_view(wf::StereoWorkflow)
    cal, k = wf.calibration, wf.camera[]
    cr = cal.reviews[k][]
    if cr !== nothing
        i = cr.plane[]
        z = _num(cr.zs[i])
        return (cr.images[i], :plate,
                "camera $k · plate $i of $(nplanes(cr)) · z = $z $(cal.length_unit[])")
    end
    plates = cal.plates[k][]
    isempty(plates) && return (nothing, :plate, "camera $k · no calibration plates")
    img = plates[1].image
    img isa AbstractMatrix && return (img, :plate, "camera $k · plate 1 · not fitted yet")
    return (nothing, :plate, "camera $k · fit the calibration to show its plates")
end

_set_title!(c::StereoCanvas) =
    (t = c.title[] * c.note[]; c.ax.title[] == t || (c.ax.title = t); c)

function _draw_frame!(c::StereoCanvas, wf::StereoWorkflow)
    img, space, title = _stereo_view(wf)
    fs = shown_frames(wf)
    # while the pair loads on a worker, keep showing the previous frame
    img === nothing && space !== :plate && current_pair(fs) !== nothing && pair_loading(fs) && return c
    c.title[] = title
    _set_title!(c)
    if img === c.shown[] && space === c.space[]
        c.dirty[] = true
        return c
    end
    c.shown[] = img
    moved = space !== c.space[]
    c.space[] = space
    if space !== :grid
        c.ax.xlabel = "x (px)"; c.ax.ylabel = "y (px)"
    else
        c.ax.xlabel = "x (dewarped px)"; c.ax.ylabel = "y (dewarped px)"
    end
    if img === nothing
        _update!(c.frame, 1:2, 1:2, _EMPTY_IMAGE)
        c.frame_size[] = nothing
    else
        nr, nc = size(img)
        _update!(c.frame, 1:nc, 1:nr, Float32.(permutedims(img)))
        # new frame dimensions or space: show the whole frame (keep the zoom
        # otherwise). Explicit limits: the heatmap's new data applies only at
        # render time, so its data limits would still be the old frame's.
        if size(img) != c.frame_size[] || moved
            c.frame_size[] = size(img)
            c.ax.limits[] = ((0.5, nc + 0.5), (0.5, nr + 0.5))
            reset_limits!(c.ax)
        end
    end
    _draw_all_overlays!(c, wf)
    c.dirty[] = true
    return c
end

# Mask raster and the out-of-view shade: grid-sized, so only over a
# dewarped frame.
function _draw_geometry!(c::StereoCanvas, wf::StereoWorkflow)
    grid = c.space[] === :grid
    m = wf.mask[]
    _draw_raster!(c.mask, grid && m !== nothing && size(m) == c.frame_size[] ? m : nothing)
    oov = grid ? out_of_view(wf) : nothing
    _draw_raster!(c.out_of_view, oov !== nothing && size(oov) == c.frame_size[] ? oov : nothing)
    c.dirty[] = true
    return c
end

# Window outlines on the Passes step, centered on the frame.
function _draw_boxes!(c::StereoCanvas, wf::StereoWorkflow)
    sz = c.frame_size[]
    show = wf.step[] === :passes && sz !== nothing
    _draw_window_boxes!(c.boxes, c.box_labels, show ? wf.passes.passes[] : (),
                        show ? ((sz[2] + 1) / 2, (sz[1] + 1) / 2) : nothing)
    c.dirty[] = true
    return c
end

"""
    grid_vector_data(r::StereoPIVResult, grid::DewarpGrid) -> NamedTuple

The in-plane vectors of a stereo result in dewarped pixels of `grid`
(`(; x, y, u, v, outlier)` as `vector_data` returns them, with `x` = column
and `y` = row of the dewarped image and the displacements divided by the
grid steps, signs included), plus `spacing`, the vector spacing in dewarped
pixels.
"""
function grid_vector_data(r::StereoPIVResult, grid::DewarpGrid)
    d = vector_data(r)
    sx, sy = step(grid.x), step(grid.y)
    x0, y0 = first(grid.x), first(grid.y)
    spacing = min(minimum(abs.(diff(r.x)); init = Inf) / abs(sx),
                  minimum(abs.(diff(r.y)); init = Inf) / abs(sy))
    return (; x = @.(1 + (d.x - x0) / sx), y = @.(1 + (d.y - y0) / sy),
            u = d.u ./ sx, v = d.v ./ sy, outlier = d.outlier, spacing)
end

# In-plane vectors of the test result (Test step) or the latest finished
# pair (Run), on the dewarped frame.
function _draw_vectors!(c::StereoCanvas, wf::StereoWorkflow)
    r = wf.step[] === :test ? wf.test.result[] :
        wf.step[] === :run && !isempty(wf.run.completed[]) ? last(wf.run.completed[]) : nothing
    dws = wf.calibration.dewarpers[]
    if r isa StereoPIVResult && dws !== nothing && c.space[] === :grid
        d = grid_vector_data(r, dws[1].grid)
        dmax = maximum(k -> hypot(d.u[k], d.v[k]), eachindex(d.u); init = 0.0)
        ls = isfinite(d.spacing) && dmax > 0 ? 0.85 * d.spacing / dmax : 1.0
        _set_arrows!(c.shafts, c.heads, d, ls)
    else
        _set_arrows!(c.shafts, c.heads, nothing, 1.0)
    end
    c.dirty[] = true
    return c
end

# Mask editing overlays (dewarped frame only) and the probe outline.
function _draw_prepare!(c::StereoCanvas, wf::StereoWorkflow)
    ps = wf.prepare
    page = wf.step[] === :prepare ? ps.page[] : :none
    me = ps.mask[]
    show_mask = page === :mask && me !== nothing && c.space[] === :grid && me.size == c.frame_size[]
    _draw_mask_editor!(c.polygons, c.active_line, c.active_points, show_mask ? me : nothing)
    _draw_probe!(c.probe_box, page === :preprocess ? ps.preview : nothing, c.frame_size[])
    c.dirty[] = true
    return c
end

# A 1-2-5 magnification that draws the largest residual about `target` px long.
function _residual_gain(maxerr::Real, target::Real)
    (isfinite(maxerr) && maxerr > 0) || return 1.0
    g = target / maxerr
    e = floor(log10(g))
    m = g / 10^e
    return (m >= 5 ? 5 : m >= 2 ? 2 : 1) * 10^e
end

# Detected dots (coloured by reprojection error once fitted), fiducial
# markers, and the residual arrows of the shown plate.
function _draw_calibration!(c::StereoCanvas, wf::StereoWorkflow)
    cr = wf.step[] === :calibration ? wf.calibration.reviews[wf.camera[]][] : nothing
    note = ""
    if cr === nothing || c.shown[] !== cr.images[cr.plane[]]
        _update!(c.dots, _NOPOINT; color = [0.0f0], colorrange = (0, 1))
        _update!(c.residuals, [Point2f(NaN, NaN), Point2f(NaN, NaN)])
        _update!(c.square, _NOPOINT)
        _update!(c.triangle, _NOPOINT)
    else
        g = cr.grids[cr.plane[]]
        pts = [Point2f(p[1], p[2]) for p in g.pixels]
        res = plane_residuals(cr)
        if res === nothing
            _update!(c.dots, isempty(pts) ? _NOPOINT : pts; color = zeros(Float32, max(length(pts), 1)),
                     colorrange = (0, 1))
            _update!(c.residuals, [Point2f(NaN, NaN), Point2f(NaN, NaN)])
            note = " · no fit: dots as detected"
        else
            errs = Float32[hypot(r...) for r in res.residuals]
            emax = maximum(errs; init = 0.0f0)
            sz = size(cr.images[cr.plane[]])
            gain = _residual_gain(emax, 0.04 * maximum(sz))
            c.residual_gain[] = gain
            segs = Point2f[]
            for (p, r) in zip(res.pixels, res.residuals)
                push!(segs, Point2f(p[1], p[2]), Point2f(p[1] + gain * r[1], p[2] + gain * r[2]))
            end
            _update!(c.dots, isempty(pts) ? _NOPOINT : pts;
                     color = isempty(errs) ? [0.0f0] : errs, colorrange = (0, max(emax, eps(Float32))))
            _update!(c.residuals, isempty(segs) ? [Point2f(NaN, NaN), Point2f(NaN, NaN)] : segs)
            note = " · residual arrows ×$(_num(gain)) · dot colour: error " *
                   "0–$(Controllers.display_number(round(emax; sigdigits = 2))) px"
        end
        _update!(c.square, g.square === nothing ? _NOPOINT : [Point2f(g.square...)])
        _update!(c.triangle, g.triangle === nothing ? _NOPOINT : [Point2f(g.triangle...)])
    end
    c.note[] = note
    _set_title!(c)
    c.dirty[] = true
    return c
end
