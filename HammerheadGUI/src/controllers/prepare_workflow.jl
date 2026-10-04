# The Prepare step inside a workflow: keeps the editors and the workflow's
# `preprocessing`/`mask`/`roi`/`scale` fields in sync (both directions, with
# a guard against write-back loops), rebuilds the editors when the frame
# size (planar) or the dewarped grid (stereo) changes, and routes canvas
# gestures (clicks and keys) to the controller of the current step and
# sub-page. Hooks per workflow type:
#   _has_roi(wf), _has_scale_tool(wf)   which editors besides the mask exist
#   _mask_target(wf)                    (size, what) the mask must match
#   _current_scale(wf)                  the scale an edit starts from
#   _scale_fields(wf)                   editable scale fields

# Settings ↔ editors for every workflow: preprocessing, mask, and scale.
function _connect_settings!(wf::AbstractWorkflow)
    ps, pp = wf.prepare, wf.prepare.preview
    # editors → workflow
    on(pp.steps) do steps
        _syncing(ps) do
            steps == _edited_steps(wf) || _set_edited_steps!(wf, copy(steps))
        end
    end
    # workflow → editors (opening settings, or edits from outside)
    on(_ -> _syncing(() -> set_steps!(pp, _edited_steps(wf)), ps), wf.preprocessing)
    on(_ -> _syncing(() -> _seed_mask!(wf), ps), wf.mask)
    _has_roi(wf) && on(_ -> _syncing(() -> _seed_roi!(wf), ps), wf.roi)
    on(_ -> _syncing(() -> _seed_scale!(wf), ps), wf.scale)
    for obs in (ps.page, ps.show_processed, pp.steps, pp.probe, pp.probe_window, pp.probe_result,
                pp.processed, pp.status, pp.error)
        on(_ -> _bump!(ps), obs)
    end
    _syncing(() -> set_steps!(pp, _edited_steps(wf)), ps)
    return wf
end

function _connect_prepare!(wf::PlanarWorkflow)
    fs = wf.frames
    # representative pair → preview frames and editor sizes
    onany((_...) -> _frames_changed!(wf), fs.files, fs.pair_mode, fs.pair, fs.loaded)
    _connect_settings!(wf)
    _frames_changed!(wf)
    return wf
end

_has_roi(::PlanarWorkflow) = true
_has_scale_tool(::PlanarWorkflow) = true
_mask_target(wf::PlanarWorkflow) = (frame_size(wf.frames), "the frames are")

# New representative pair (or its delivery): hand the frames to the preview
# and rebuild the editors when the frame size changed. While a pair loads on
# a worker, everything keeps the previous pair.
function _frames_changed!(wf::PlanarWorkflow)
    ps, fs = wf.prepare, wf.frames
    imgs = try
        pair_images(fs)
    catch
        nothing
    end
    if imgs === nothing
        (pair_loading(fs) || current_pair(fs) !== nothing) && return wf
        set_frames!(ps.preview, nothing, nothing)
        _set_editors!(wf, nothing)
        return wf
    end
    a, b = imgs
    size(a) == size(b) ? set_frames!(ps.preview, a, b) : set_frames!(ps.preview, a, nothing)
    me = ps.mask[]
    (me === nothing || me.size != size(a)) && _set_editors!(wf, size(a))
    return wf
end

# Editors for a new size, seeded from the workflow's settings.
function _set_editors!(wf::AbstractWorkflow, sz::Union{Nothing,Dims{2}})
    ps = wf.prepare
    if sz === nothing
        ps.mask[] = nothing
        ps.roi[] = nothing
        ps.scale[] = nothing
        _bump!(ps)
        return wf
    end
    me = MaskEditor(sz)
    ed = _has_roi(wf) ? ROIEditor(sz) : nothing
    st = _has_scale_tool(wf) ? ScaleTool(sz) : nothing
    ps.mask[] = me
    ps.roi[] = ed
    ps.scale[] = st
    _syncing(ps) do
        _seed_mask!(wf)
        ed === nothing || _seed_roi!(wf)
        _seed_scale!(wf)
    end
    for obs in (me.polygons, me.holes, me.raster)
        on(_ -> _syncing(() -> _write_mask!(wf, me), ps), obs)
    end
    for obs in (me.polygons, me.holes, me.active, me.hole_mode, me.selected, me.raster)
        on(_ -> _bump!(ps), obs)
    end
    if ed !== nothing
        on(_ -> _syncing(() -> _write_roi!(wf, ed), ps), ed.roi)
        foreach(obs -> on(_ -> _bump!(ps), obs), (ed.roi, ed.anchor))
    end
    if st !== nothing
        for obs in (st.points, st.separation)
            on(_ -> _syncing(() -> _write_measurement!(wf, st), ps), obs)
        end
        foreach(obs -> on(_ -> _bump!(ps), obs), (st.points, st.separation))
    end
    _bump!(ps)
    return wf
end

# ---------------------------------------------------------------- mask

function _write_mask!(wf::AbstractWorkflow, me::MaskEditor)
    me === wf.prepare.mask[] || return
    m = has_mask(me) ? polygon_mask(me) : nothing
    isequal(m, wf.mask[]) || (wf.mask[] = m)
    return
end

# A mask from outside becomes the editor's raster (new polygons go on top).
# A mask of another size stays in the settings but cannot be edited.
function _seed_mask!(wf::AbstractWorkflow)
    me = wf.prepare.mask[]
    me === nothing && return
    m = wf.mask[]
    if m === nothing || size(m) != me.size
        has_mask(me) && set_raster!(me, nothing)
    elseif !has_mask(me) || !isequal(polygon_mask(me), m)
        set_raster!(me, m)
    end
    return
end

"""
    load_mask_file!(wf::AbstractWorkflow, path)

Use a mask image (`Hammerhead.load_mask`: white = excluded) as the mask. It
must match the frames (planar) or the dewarped grid (stereo).
"""
function load_mask_file!(wf::AbstractWorkflow, path::AbstractString)
    m = load_mask(path)
    sz, what = _mask_target(wf)
    sz === nothing || size(m) == sz ||
        throw(DimensionMismatch("the mask is $(size(m, 2))×$(size(m, 1)) px but $what $(sz[2])×$(sz[1]) px"))
    wf.mask[] = m
    wf.status[] = "mask: $(basename(path))"
    return wf
end

"""
    save_mask_file(wf::AbstractWorkflow, path) -> path

Write the mask as an image (white = excluded), as `load_mask` reads it.
"""
function save_mask_file(wf::AbstractWorkflow, path::AbstractString)
    wf.mask[] === nothing && throw(ArgumentError("there is no mask to save"))
    FileIO.save(path, Gray.(wf.mask[]))
    wf.status[] = "saved mask to $(basename(path))"
    return path
end

# ---------------------------------------------------------------- ROI

function _write_roi!(wf::PlanarWorkflow, ed::ROIEditor)
    ed === wf.prepare.roi[] || return
    isequal(ed.roi[], wf.roi[]) || (wf.roi[] = ed.roi[])
    isempty(wf.prepare.roi_error[]) || (wf.prepare.roi_error[] = "")
    return
end

# An ROI outside the frame stays in the settings; the editor shows none.
function _seed_roi!(wf::PlanarWorkflow)
    ed = wf.prepare.roi[]
    (ed === nothing || isequal(ed.roi[], wf.roi[])) && return
    try
        set_roi!(ed, wf.roi[])
    catch
        set_roi!(ed, nothing)
    end
    return
end

"""
    edit_roi!(wf::PlanarWorkflow, row_first, row_last, col_first, col_last) -> Bool

Set the ROI from (text) bounds through the ROI editor. A rejected entry
leaves the ROI unchanged, sets `wf.prepare.roi_error`, and returns `false`.
"""
function edit_roi!(wf::PlanarWorkflow, r1, r2, c1, c2)
    ps = wf.prepare
    ed = ps.roi[]
    try
        ed === nothing && throw(ArgumentError("add frames first"))
        set_roi!(ed, r1, r2, c1, c2)
    catch err
        ps.roi_error[] = _errmsg(err)
        return false
    end
    isempty(ps.roi_error[]) || (ps.roi_error[] = "")
    return true
end

# ---------------------------------------------------------------- scale

_current_scale(wf::PlanarWorkflow) = something(wf.scale[], PhysicalScale())

# The tool measures in the scale's length unit (a measurement in "px" would
# be meaningless, so a new scale measures in the tool's unit, mm by default).
function _seed_scale!(wf::AbstractWorkflow)
    st = wf.prepare.scale[]
    (st === nothing || wf.scale[] === nothing) && return
    sc = wf.scale[]
    sc.length_unit == "px" || st.length_unit[] == sc.length_unit || (st.length_unit[] = sc.length_unit)
    st.dt[] == sc.dt || (st.dt[] = sc.dt)
    st.time_unit[] == sc.time_unit || (st.time_unit[] = sc.time_unit)
    return
end

function _write_measurement!(wf::PlanarWorkflow, st::ScaleTool)
    st === wf.prepare.scale[] || return
    ps = pixel_size(st)
    ps === nothing && return
    cur = _current_scale(wf)
    new = PhysicalScale(ps, cur.dt, st.length_unit[], cur.time_unit)
    isequal(new, wf.scale[]) || (wf.scale[] = new)
    return
end

const SCALE_FIELDS = (:pixel_size, :dt, :length_unit, :time_unit)

_scale_fields(::PlanarWorkflow) = SCALE_FIELDS

"""
    set_scale_field!(wf::AbstractWorkflow, field, value)

Set one field of the physical scale (`:pixel_size`, `:dt`, `:length_unit`,
or `:time_unit`) from a value or its text form; the other fields keep their
values (unscaled defaults when there is no scale yet). Invalid values throw
and leave the scale unchanged. A stereo scale has no `:pixel_size`: its
lengths are the calibration's world units.
"""
function set_scale_field!(wf::AbstractWorkflow, field::Symbol, value)
    field in SCALE_FIELDS || throw(ArgumentError("unknown scale field :$field"))
    field in _scale_fields(wf) ||
        throw(ArgumentError("a stereo scale has no $field: lengths are the calibration's world units"))
    cur = _current_scale(wf)
    f = Dict{Symbol,Any}(k => getfield(cur, k) for k in SCALE_FIELDS)
    if field in (:pixel_size, :dt)
        what = field === :dt ? "dt" : "pixel size"
        f[field] = value isa AbstractString ? _parse_positive(value, what) :
                   (value isa Real && isfinite(value) && value > 0) ? Float64(value) :
                   throw(ArgumentError("$what must be a positive number, got $value"))
    else
        u = strip(String(value))
        isempty(u) && throw(ArgumentError("the unit must not be empty"))
        f[field] = String(u)
    end
    new = PhysicalScale(f[:pixel_size], f[:dt], f[:length_unit], f[:time_unit])
    isequal(new, wf.scale[]) || (wf.scale[] = new)
    st = wf.prepare.scale[]
    field === :length_unit && st !== nothing && (st.length_unit[] = new.length_unit)
    return wf
end

"""
    edit_scale!(wf::AbstractWorkflow, field, value) -> Bool

[`set_scale_field!`](@ref), or the measured line's `:separation` (see
`set_separation!`; planar only), reporting a rejected entry in
`wf.prepare.scale_error` (returns `false`) instead of throwing.
"""
function edit_scale!(wf::AbstractWorkflow, field::Symbol, value)
    ps = wf.prepare
    try
        if field === :separation
            _has_scale_tool(wf) || throw(ArgumentError("a stereo scale has no measured line"))
            st = ps.scale[]
            st === nothing && throw(ArgumentError("add frames first"))
            set_separation!(st, value isa Real ? value : String(value))
        else
            set_scale_field!(wf, field, value)
        end
    catch err
        ps.scale_error[] = _errmsg(err)
        return false
    end
    isempty(ps.scale_error[]) || (ps.scale_error[] = "")
    return true
end

"""
    clear_scale!(wf::AbstractWorkflow)

Remove the physical scale (results stay in measured units and frames) and
the measured line.
"""
function clear_scale!(wf::AbstractWorkflow)
    st = wf.prepare.scale[]
    st === nothing || isempty(st.points[]) || clear_points!(st)
    wf.scale[] === nothing || (wf.scale[] = nothing)
    isempty(wf.prepare.scale_error[]) || (wf.prepare.scale_error[] = "")
    return wf
end

# ---------------------------------------------------------------- preprocessing

"""
    edit_step_option!(wf::AbstractWorkflow, i, key, value) -> Bool

`set_step_option!` on the preview, reporting a rejected value in the
preview's `error`/`error_step` (returns `false`) instead of throwing.
"""
function edit_step_option!(wf::AbstractWorkflow, i::Integer, key, value)
    pp = wf.prepare.preview
    try
        set_step_option!(pp, i, key, value)
    catch err
        pp.error_step[] = Int(i)
        pp.error[] = _errmsg(err)
        return false
    end
    _clear_error!(pp)
    return true
end

"""
    estimate_background!(wf::PlanarWorkflow; frames = 10, method = :min)

Estimate the background from the first `frames` frames (`compute_background`)
and subtract it as the first preprocessing step. In a window this runs on a
worker task; `wf.prepare.status` reports progress. A `StereoWorkflow`
estimates each camera's background from its own frames (see
[`set_backgrounds!`](@ref)).
"""
function estimate_background!(wf::PlanarWorkflow; frames::Integer = 10, method::Symbol = :min)
    ps = wf.prepare
    frames >= 1 || throw(ArgumentError("use at least one frame for the background"))
    entries = wf.frames.files[][1:min(end, frames)]
    isempty(entries) && (ps.status[] = "add frames first"; return wf)
    g = (ps.background_generation[] += 1)
    n = length(entries)
    ps.status[] = "estimating the background from $n frame" * (n == 1 ? "" : "s") * "…"
    ps.background_running[] = true
    apply = function (out)
        g == ps.background_generation[] || return
        ps.background_running[] = false
        if out.err === nothing
            set_background!(ps.preview, out.value)
            ps.status[] = "background: $(method === :min ? "minimum" : "mean") of $n frame" *
                          (n == 1 ? "" : "s")
        else
            ps.status[] = "background failed: " * _errmsg(out.err)
        end
    end
    ps.preview.runner[](() -> estimate_background(entries; method), apply)
    return wf
end

"""
    background_note(wf::AbstractWorkflow) -> Union{Nothing,String}

`nothing` when [`estimate_background!`](@ref) can estimate a background for
`wf`, otherwise the reason it cannot. Windows show the note instead of the
background controls.
"""
background_note(::AbstractWorkflow) = nothing

# ---------------------------------------------------------------- page

"""
    set_prepare_page!(wf::AbstractWorkflow, page)

Open a Prepare sub-page (one of [`prepare_pages`](@ref)`(wf)`). Leaving a
page drops its unfinished gesture (a polygon being drawn, a pending ROI
corner).
"""
function set_prepare_page!(wf::AbstractWorkflow, page::Symbol)
    page in prepare_pages(wf) || throw(ArgumentError("unknown Prepare page :$page"))
    ps = wf.prepare
    ps.page[] == page && return wf
    me, ed = ps.mask[], ps.roi[]
    me === nothing || cancel_active!(me)
    ed === nothing || cancel_corner!(ed)
    ps.page[] = page
    return wf
end

# ---------------------------------------------------------------- gestures

"""
    canvas_click!(wf::AbstractWorkflow, x, y) -> Bool

A primary click on the image canvas at data coordinates `(x, y)`
(x = column, y = row). On the Prepare step it goes to the open sub-page —
Preprocess: place the correlation probe; Mask, ROI, Scale: `click!` on the
editor (a stereo workflow's canvas is the dewarped grid, and its Scale page
has no editor). Returns whether the click was used (otherwise the viewer
keeps it).
"""
function canvas_click!(wf::AbstractWorkflow, x::Real, y::Real)
    (wf.step[] === :prepare && isfinite(x) && isfinite(y)) || return false
    ps = wf.prepare
    page = ps.page[]
    if page === :preprocess
        ps.preview.image[] === nothing && return false
        click!(ps.preview, x, y)
        return true
    end
    ed = page === :mask ? ps.mask[] : page === :roi ? ps.roi[] : ps.scale[]
    ed === nothing && return false
    click!(ed, x, y)
    return true
end

"""
    canvas_alt_click!(wf::AbstractWorkflow) -> Bool

A secondary (right) click on the image canvas. Mask: close the polygon
being drawn, or drop the selection; ROI: drop a pending corner; Scale: drop
the measured line; Preprocess: remove the probe. Returns whether it was used.
"""
function canvas_alt_click!(wf::AbstractWorkflow)
    wf.step[] === :prepare || return false
    ps = wf.prepare
    page = ps.page[]
    if page === :preprocess
        ps.preview.probe[] === nothing && return false
        clear_probe!(ps.preview)
        return true
    elseif page === :mask
        me = ps.mask[]
        (me === nothing || (isempty(me.active[]) && me.selected[] === nothing)) && return false
        alt_click!(me)
        return true
    elseif page === :roi
        ed = ps.roi[]
        return ed !== nothing && cancel_corner!(ed)
    else
        st = ps.scale[]
        (st === nothing || isempty(st.points[])) && return false
        clear_points!(st)
        return true
    end
end

"""
    canvas_key!(wf::AbstractWorkflow, key::Symbol) -> Bool

A key pressed on the image canvas: `:backspace` (undo the last vertex or
point), `:escape` (cancel the polygon, pending corner, line, or probe), or
`:delete` (delete the selected polygon). Returns whether it was used.
"""
function canvas_key!(wf::AbstractWorkflow, key::Symbol)
    wf.step[] === :prepare || return false
    ps = wf.prepare
    page = ps.page[]
    if page === :mask
        me = ps.mask[]
        me === nothing && return false
        if key === :backspace
            isempty(me.active[]) && return false
            undo_vertex!(me)
            return true
        elseif key === :escape
            cancel_active!(me) && return true
            me.selected[] === nothing && return false
            me.selected[] = nothing
            return true
        elseif key === :delete
            me.selected[] === nothing && return false
            delete_selected!(me)
            return true
        end
    elseif page === :roi
        ed = ps.roi[]
        return ed !== nothing && key in (:escape, :backspace) && cancel_corner!(ed)
    elseif page === :scale
        st = ps.scale[]
        st === nothing && return false
        key === :backspace && return undo_point!(st)
        key === :escape && !isempty(st.points[]) && (clear_points!(st); return true)
    elseif key in (:escape, :delete) && ps.preview.probe[] !== nothing
        clear_probe!(ps.preview)
        return true
    end
    return false
end

# A frame set's pending load is forgotten (its result would go to a queue
# nobody drains any more).
function _abandon_load!(fs::FrameSet)
    fs.generation[] += 1
    fs.request[] = nothing
    fs.loading[] && (fs.loading[] = false)
    return fs
end

# A window closed with jobs in flight: their results go to a queue nobody
# drains any more. Forget them and bring the state up to date inline.
function _abandon_jobs!(wf::PlanarWorkflow)
    ps = wf.prepare
    _abandon_load!(wf.frames)
    if ps.background_running[]
        ps.background_generation[] += 1
        ps.background_running[] = false
        ps.status[] = ""
    end
    _request_preview!(ps.preview)
    _frames_changed!(wf)
    return wf
end
