# Qt Quick shell for the workflow windows (QML.jl + QMLMakie).
#
# Threading: Qt's event loop owns the main thread while a window is open.
# QML→Julia calls only change state and return. Background jobs hand their
# observable updates to `deliver`, which queues them; `hh_tick` (a QML Timer,
# ~60 Hz) drains the queue on the GUI thread and refreshes what QML shows.
#
# Canvases: QMLMakie canvases (`MakieArea`) are never destroyed while a
# window is open — destroying one leaves a dangling callback in jlqml that
# crashes at window teardown. The main window and the pop-out window each own
# one canvas for their lifetime; figures move between them instead.

const QML_DIR = joinpath(@__DIR__, "..", "qml")

# ---------------------------------------------------------------- Qt setup

const _QT_READY = Ref(false)

# GLFW contexts (GLMakie windows and offscreen `colorbuffer` screens) and the
# Qt windows' GL contexts must not share a process: once a GLFW context had
# existed, Qt's render thread crashed in the AMD driver. A Qt window refuses
# to open next to a GLFW screen; `_QT_OPENED` lets the standalone GLMakie
# views warn after a Qt window was opened.
const _QT_OPENED = Ref(false)

_glfw_screen_count() = count(s -> !(s isa GLMakie.Screen{QMLMakie.QMLWindow}), GLMakie.ALL_SCREENS)

function _check_no_glfw_screens()
    n = _glfw_screen_count()
    n == 0 && return
    throw(ArgumentError(
        "this Julia session has created $n GLMakie screen" * (n == 1 ? "" : "s") *
        " (a result_explorer, calibration_review, or selfcal_review window, or an " *
        "offscreen render), and GLMakie windows and the Qt workflow windows cannot " *
        "share a process (the graphics driver can crash). Restart Julia to open " *
        "this window; browse results in its Results step instead of a GLMakie window."))
end

function _warn_if_qt_opened(view::AbstractString)
    _QT_OPENED[] || return
    @warn "$view opens a GLMakie window, and a Qt workflow window was opened in this " *
          "Julia session: planar_window and stereo_window cannot open again in this " *
          "session once a GLMakie window exists (the graphics driver can crash). " *
          "Use the window's Results step, or a separate Julia session." maxlog = 1
    return
end

function _qt_init!()
    _QT_READY[] && return
    if Sys.iswindows()
        # The FluentWinUI3 style plugin depends on a library outside the DLL
        # search path; load it first so the native Windows 11 style is used.
        impl = joinpath(Qt6Declarative_jll.artifact_dir, "bin",
                        "Qt6QuickControls2FluentWinUI3StyleImpl.dll")
        isfile(impl) && Libdl.dlopen(impl)
        # Quick Controls reads the style when QML first imports it (after app
        # start). Qt reads msvcrt's copy of the environment, which Julia's ENV
        # (Win32 SetEnvironmentVariable) does not update, so set both.
        style = get(ENV, "QT_QUICK_CONTROLS_STYLE", "FluentWinUI3")
        ENV["QT_QUICK_CONTROLS_STYLE"] = style
        ccall((:_putenv_s, "msvcrt"), Cint, (Cstring, Cstring), "QT_QUICK_CONTROLS_STYLE", style)
    end
    _QT_READY[] = true
    return
end

# Glyphs a canvas may show (ASCII, the symbols of the controllers' labels,
# Makie's tick minus). A glyph first laid out while a window is open reaches
# the atlas texture through a callback that runs outside Qt's render, where
# no GL context is current, and draws as garbage; glyphs in the atlas before
# the first render upload with it.
const _CANVAS_GLYPHS = vcat(Char(32):Char(126), collect("°±²³·¼×÷ΓΔεπσωµ–—−…›→↔∘≤≥≈|"))

# Rendering these glyphs takes ~3 s, so the warmed atlas is cached on disk
# next to Makie's own atlas cache and loaded on the next start, provided no
# screen uses the session's atlas yet and the cache holds every glyph it has.
const _ATLAS_KEY = (2048, 64)                        # GLMakie's atlas
_atlas_cache_path() = Makie.get_cache_path(_ATLAS_KEY...) * ".hammerheadgui"

function _load_cached_atlas!(atlas)
    path = _atlas_cache_path()
    (isempty(atlas.font_render_callback) && isfile(path)) || return atlas
    cached = try
        Makie.load_texture_atlas(path)
    catch
        return atlas
    end
    all(k -> haskey(cached.mapping, k), keys(atlas.mapping)) || return atlas
    Makie.TEXTURE_ATLASES[_ATLAS_KEY] = cached
    return cached
end

function _warm_glyph_atlas!()
    atlas = _load_cached_atlas!(Makie.get_texture_atlas(_ATLAS_KEY...))
    n = length(atlas.mapping)
    fonts = Makie.theme(:fonts)
    for key in keys(fonts)
        font = Makie.to_font(fonts[key][])
        for c in _CANVAS_GLYPHS
            Makie.insert_glyph!(atlas, c, font)
        end
    end
    if length(atlas.mapping) > n
        path = _atlas_cache_path()
        try
            tmp = path * ".$(getpid()).tmp"
            Makie.store_texture_atlas(tmp, atlas)
            mv(tmp, path; force = true)
        catch
        end
    end
    return
end

# After a window closes Qt has destroyed its GL contexts: forget the canvases'
# screens so GLMakie neither renders to them nor frees them at exit.
function _release_qml_screens!()
    for s in GLMakie.ALL_SCREENS
        s isa GLMakie.Screen{QMLMakie.QMLWindow} && (s.glscreen.context.valid = false)
    end
    filter!(s -> !(s isa GLMakie.Screen{QMLMakie.QMLWindow}), GLMakie.ALL_SCREENS)
    for key in collect(keys(GLMakie.atlas_texture_cache))
        key[2] isa QMLMakie.QMLWindow || continue
        _, callback = pop!(GLMakie.atlas_texture_cache, key)
        Makie.remove_font_render_callback!(key[1], callback)
    end
    return
end

# QML FileDialog URLs → local paths.
function _url_to_path(url::AbstractString)
    s = String(url)
    startswith(s, "file://") || return s
    s = s[8:end]                                   # drop "file://"
    Sys.iswindows() && startswith(s, "/") && occursin(r"^/[A-Za-z]:", s) && (s = s[2:end])
    return _percent_decode(s)
end

function _percent_decode(s::String)
    io = IOBuffer()
    i = firstindex(s)
    while i <= lastindex(s)
        if s[i] == '%' && i + 2 <= lastindex(s)
            write(io, parse(UInt8, s[i+1:i+2]; base = 16))
            i += 3
        else
            write(io, s[i])
            i = nextind(s, i)
        end
    end
    return String(take!(io))
end

# ---------------------------------------------------------------- canvas host

"""
Which canvas shows the current figure, and the two-phase move between the
main window's canvas and the pop-out window's: the source canvas first draws
its own placeholder (GLMakie releases the figure in that canvas's GL
context), then the figure goes to the target.
"""
mutable struct CanvasHost
    props::JuliaPropertyMap
    placeholder::Dict{Symbol,Figure}
    content::Figure
    area::Symbol                                   # :main or :pop
    pending::Union{Nothing,Symbol}                 # target area of a move
end

function CanvasHost(props::JuliaPropertyMap, content::Figure)
    ph = Dict(:main => Figure(size = (10, 10)), :pop => Figure(size = (10, 10)))
    props["main"] = content
    props["pop"] = ph[:pop]
    props["popped"] = false
    return CanvasHost(props, ph, content, :main, nothing)
end

_area_key(a::Symbol) = a === :main ? "main" : "pop"

function show_figure!(h::CanvasHost, fig::Figure)
    fig === h.content && return h
    h.content = fig
    h.pending === nothing && (h.props[_area_key(h.area)] = fig)
    return h
end

function toggle_popout!(h::CanvasHost)
    h.pending === nothing || return h
    target = h.area === :main ? :pop : :main
    h.props[_area_key(h.area)] = h.placeholder[h.area]
    h.pending = target
    return h
end

# Called every tick; completes a move once the figure is free.
function advance!(h::CanvasHost)
    h.pending === nothing && return false
    isempty(Makie.get_scene(h.content).current_screens) || return true
    h.area = h.pending
    h.pending = nothing
    h.props[_area_key(h.area)] = h.content
    h.props["popped"] = h.area === :pop
    return true
end

# ---------------------------------------------------------------- bridge

mutable struct StepRow
    key::String
    label::String
    status::String                                 # :todo/:ok/:attention/:busy ("state" clashes with QML)
    summary::String
end

mutable struct PassRow
    number::Int                                    # "index" is reserved in QML delegates
    window::Int
    search::Int
    overlap::Float64
    iterations::Int
end

# One preprocessing step; options travel as "|"-joined keys, labels, values.
mutable struct PrepStepRow
    number::Int
    label::String
    optionKeys::String
    optionLabels::String
    optionValues::String
    error::String
end

"""
State shared between a workflow (`PlanarWorkflow` or `StereoWorkflow`) and
its QML window: the property map `app`, the item models, the image canvas
(`PlanarCanvas`/`StereoCanvas`), the results canvas, and the queue of
background updates that `hh_tick` drains. `PlanarShell` and `StereoShell`
are its two forms; `models` holds a window's extra item models by their QML
name (stereo: the plate lists).
"""
mutable struct WorkflowShell{W<:AbstractWorkflow,C}
    wf::W
    canvas::C
    host::CanvasHost
    results::ResultsCanvas
    app::JuliaPropertyMap
    step_rows::Vector{StepRow}
    step_model::JuliaItemModel
    pass_rows::Vector{PassRow}
    pass_model::JuliaItemModel
    prep_rows::Vector{PrepStepRow}
    prep_model::JuliaItemModel
    queue::Channel{Any}
    dirty::Base.RefValue{Bool}
    shown::Dict{String,Any}
    models::Dict{String,JuliaItemModel}
    rows::Dict{String,Any}                         # the extra models' rows (and caches)
end

const PlanarShell = WorkflowShell{PlanarWorkflow}

PlanarShell(wf::PlanarWorkflow; queue::Channel{Any} = Channel{Any}(Inf)) =
    WorkflowShell(wf, planar_canvas(wf); queue)

# `base`: the shell of the session this one replaces in an open window (a
# change of recording type). The objects QML is bound to (the property map,
# the item models, the canvas host, and the results canvas) carry over; the
# workflow and its image canvas are new.
function WorkflowShell(wf::AbstractWorkflow, canvas; queue::Channel{Any} = Channel{Any}(Inf),
                       base::Union{Nothing,WorkflowShell} = nothing)
    steps = [StepRow(String(s), step_label(wf, s), "todo", "") for s in workflow_steps(wf)]
    sh = if base === nothing
        app = JuliaPropertyMap()
        pass_rows, prep_rows = PassRow[], PrepStepRow[]
        models, rows = Dict{String,JuliaItemModel}(), Dict{String,Any}()
        for k in 1:2                  # the Calibration step's plate lists (stereo sessions)
            rows["plates$k"] = PlateRow[]
            models["plates$(k)Model"] = JuliaItemModel(rows["plates$k"])
        end
        WorkflowShell(wf, canvas, CanvasHost(app, canvas.fig), results_canvas(), app,
                      steps, JuliaItemModel(steps), pass_rows, JuliaItemModel(pass_rows),
                      prep_rows, JuliaItemModel(prep_rows), queue, Ref(true),
                      Dict{String,Any}(), models, rows)
    else
        _sync_rows!(base.step_rows, base.step_model, steps, (a, b) -> false)
        for k in 1:2
            _sync_rows!(base.rows["plates$k"], base.models["plates$(k)Model"], PlateRow[],
                        (a, b) -> false)
        end
        WorkflowShell(wf, canvas, base.host, base.results, base.app, base.step_rows,
                      base.step_model, base.pass_rows, base.pass_model, base.prep_rows,
                      base.prep_model, base.queue, Ref(true), base.shown, base.models, base.rows)
    end
    app, results = sh.app, sh.results
    mark = (_...) -> (sh.dirty[] = true)
    ps, pe, t, r = wf.prepare, wf.passes, wf.test, wf.run
    pp = ps.preview
    for obs in (wf.step, wf.preprocessing, wf.mask, wf.scale, wf.saved, wf.settings_path,
                wf.explorer, wf.results_path, wf.status,
                ps.revision, ps.status, ps.background_running, ps.roi_error, ps.scale_error,
                ps.ruler_name,
                pp.error_step, pp.status, pp.processed2,
                pe.passes, pe.preset, pe.mode, pe.image_type, pe.error,
                pe.backend, pe.gpu_loading, pe.gpu_status,
                t.result, t.running, t.status, r.output_path, r.running, r.progress, r.status,
                r.completed, r.finished_output)
        on(mark, obs)
    end
    _connect_window!(sh, mark)
    on(pe.passes) do _
        _sync_pass_rows!(sh)
    end
    onany((_...) -> _sync_prep_rows!(sh), pp.steps, pp.error, pp.error_step)
    function watch_explorer(ex)
        set_explorer!(results, ex)
        ex === nothing && return
        for obs in (ex.frame, ex.field, ex.selection, ex.color_mode, ex.show_vectors, ex.status,
                    ex.tool, ex.tool_points, ex.profile_data, ex.circulation_result,
                    ex.color_min, ex.color_max, ex.color_percentiles, ex.physical_units,
                    ex.include_flagged, ex.revalidation, ex.image_frame, ex.image_available,
                    ex.tool_selected)
            on(mark, obs)
        end
    end
    on(watch_explorer, wf.explorer)
    watch_explorer(wf.explorer[])
    # Every key QML binds to exists from the start (bindings read them at once).
    for (k, v) in ("grabPath" => "", "viewMode" => "edit", "resultFrame" => 1, "resultFrames" => 1, "resultFieldKeys" => "",
                   "resultFieldLabels" => "", "resultField" => "", "resultFieldLabel" => "", "resultColorMode" => "robust",
                   "resultColorScale" => "percentile", "resultColorLow" => "2", "resultColorHigh" => "98",
                   "resultHasScale" => false, "resultPhysical" => true, "resultIncludeFlagged" => false,
                   "resultIsPlanar" => false, "resultRevalidate" => false, "rvUodEnable" => true,
                   "rvUodThreshold" => "2", "rvUodNeighborhood" => 2, "rvMinPeakRatio" => "1",
                   "rvReplace" => true,
                   # the Particles page's keys (planar particle modes; other windows keep these)
                   "particleMode" => false, "ptvThreshold" => "auto", "ptvThresholdK" => "",
                   "ptvMinSeparation" => "", "ptvMinDiameter" => "", "ptvMaxDiameter" => "",
                   "ptvSearchRadius" => "", "ptvIntensityWeight" => "", "ptvDiameterWeight" => "",
                   "ptvUodThreshold" => "", "ptvUodEpsilon" => "", "ptvUodNeighbors" => "",
                   "ptvUodEnable" => true, "ptvPredictor" => "piv", "ptvMinTrackLength" => 3,
                   "ptvMaxGap" => 0, "ptvError" => "", "ptvDetectStatus" => "", "frameCount" => 0,
                   "resultVectors" => true, "selectionText" => "", "resultsStatus" => "",
                   "resultTool" => "inspect", "resultToolsAvailable" => false, "toolSummary" => "",
                   # the GPU switch: installed device packages are looked up once
                   "gpuInstalled" => !isempty(gpu_packages()),
                   "gpuPackages" => join((Controllers._gpu_package(b) for b in gpu_packages()), " and "),
                   # per-frame mask images (planar only)
                   "frameMaskCount" => 0, "frameMasksInfo" => "", "patternDir3" => "",
                   "pattern3" => "*.png", "patternCount3" => 0, "patternInfo3" => "",
                   "switchQuestion" => "")
        base === nothing && _set!(sh, k, v)
    end
    base === nothing && _seed_other_type!(sh)
    if base !== nothing                        # the toolbar's view mode carries over
        m = Symbol(get(sh.shown, "viewMode", "edit"))
        set_view_mode!(canvas.ax, m, :workflow_gesture)
    end
    _sync_pass_rows!(sh)
    _sync_prep_rows!(sh)
    _refresh!(sh)
    return sh
end

# The window loads the pages of both recording types, so every key either
# type binds must exist from the start: seed the other type's keys from a
# fresh session of it (the session's own refresh then sets its keys).
function _seed_other_type!(sh::WorkflowShell)
    other = new_workflow(recording_type(sh.wf) === :stereo ? :planar : :stereo)
    seed = WorkflowShell{typeof(other),Nothing}(other, nothing, sh.host, sh.results, sh.app,
                                                sh.step_rows, sh.step_model, sh.pass_rows,
                                                sh.pass_model, sh.prep_rows, sh.prep_model,
                                                sh.queue, Ref(false), sh.shown, sh.models, sh.rows)
    _refresh_frames!(seed)
    _refresh_region!(seed)
    _refresh_window!(seed)
    return sh
end

# The planar window's own observables: the frames and the ROI.
function _connect_window!(sh::PlanarShell, mark)
    wf = sh.wf
    fs = wf.frames
    for f in (fs, wf.frame_masks), obs in (f.files, f.pair_mode, f.pair, f.loading, f.loaded, f.load_error,
                                          f.pattern_dir, f.pattern, f.pattern_matches, f.pattern_error)
        on(mark, obs)
    end
    on(mark, wf.roi)
    on(mark, fs.shown)
    pt = wf.particles
    for obs in (pt.ptv, pt.predictor, pt.min_track_length, pt.max_gap, pt.error, pt.detect_status)
        on(mark, obs)
    end
    return sh
end

# Assign a QML property only when its value changed.
function _set!(sh::WorkflowShell, key::String, val)
    haskey(sh.shown, key) && isequal(sh.shown[key], val) && return
    sh.shown[key] = val
    sh.app[key] = val
    return
end

# Replace the rows of an item model: in place when the count is unchanged
# (only changed rows are re-sent), otherwise with a full reset.
function _sync_rows!(rows::Vector{R}, model::JuliaItemModel, new::Vector{R}, same) where {R}
    if length(new) == length(rows)
        for (i, r) in enumerate(new)
            same(rows[i], r) && continue
            model[i] = r
        end
    else
        empty!(rows)
        append!(rows, new)
        QML.force_model_update(model)
    end
    return
end

function _sync_pass_rows!(sh::WorkflowShell)
    new = [PassRow(i, r.window, r.search, r.overlap, r.iterations)
           for (i, r) in enumerate(pass_rows(sh.wf.passes))]
    _sync_rows!(sh.pass_rows, sh.pass_model, new,
                (a, b) -> (a.window, a.search, a.overlap, a.iterations) ==
                          (b.window, b.search, b.overlap, b.iterations))
    return
end

function _prep_row(pp, i::Int, step::PreprocessStep)
    opts = step_options(step)
    return PrepStepRow(i, preprocess_label(step.operation), join(first.(opts), "|"),
                       join((Controllers.OPTION_LABELS[k] for k in first.(opts)), "|"),
                       join(last.(opts), "|"), pp.error_step[] == i ? pp.error[] : "")
end

function _sync_prep_rows!(sh::WorkflowShell)
    pp = sh.wf.prepare.preview
    new = [_prep_row(pp, i, s) for (i, s) in enumerate(pp.steps[])]
    _sync_rows!(sh.prep_rows, sh.prep_model, new,
                (a, b) -> (a.label, a.optionKeys, a.optionValues, a.error) ==
                          (b.label, b.optionKeys, b.optionValues, b.error))
    return
end

# Numbers for text fields: integers plainly, others to ~6 significant digits
# (an unedited field never writes its rounded text back).
_num(x::Real) = isinteger(x) && abs(x) < 1e15 ? string(Int(x)) : Controllers.display_number(x)

# Prepare step fields of the property map.
function _refresh_prepare!(sh::WorkflowShell)
    wf = sh.wf
    ps = wf.prepare
    pp = ps.preview
    _set!(sh, "preparePage", String(ps.page[]))
    _set!(sh, "hasFrameSize", ps.mask[] !== nothing)
    _set!(sh, "prepareStatus", ps.status[])
    _set!(sh, "backgroundRunning", ps.background_running[])
    _set!(sh, "backgroundNote", something(background_note(wf), ""))
    _set!(sh, "pipelineSummary", pipeline_summary(pp))
    _set!(sh, "previewStatus", pp.status[])
    _set!(sh, "showProcessed", ps.show_processed[])
    _set!(sh, "probeWindow", pp.probe_window[])
    _set!(sh, "probeSummary", probe_summary(pp))

    me = ps.mask[]
    _set!(sh, "maskStatus", me === nothing ? _no_mask_editor(wf) :
                            Controllers.status_text(me) * (me.raster[] === nothing ? "" : " · raster mask"))
    _set!(sh, "maskDrawing", me !== nothing && !isempty(me.active[]))
    _set!(sh, "maskSelected", me !== nothing && me.selected[] !== nothing)
    _set!(sh, "hasMask", wf.mask[] !== nothing)

    _refresh_region!(sh)

    sc = wf.scale[]
    st = ps.scale[]
    _set!(sh, "hasScale", sc !== nothing)
    _set!(sh, "scalePixelSize", sc === nothing ? "" : _num(sc.pixel_size))
    _set!(sh, "scaleLengthUnit", sc === nothing ? "" : sc.length_unit)
    _set!(sh, "scaleDt", sc === nothing ? "" : _num(sc.dt))
    _set!(sh, "scaleTimeUnit", sc === nothing ? "" : sc.time_unit)
    _set!(sh, "scaleSummary", _scale_summary(wf))
    _set!(sh, "scaleSeparation", st === nothing ? "" : _num(st.separation[]))
    _set!(sh, "scaleMeasureUnit", st === nothing ? "" : st.length_unit[])
    _set!(sh, "scaleMeasure", st === nothing ? "" : Controllers.scale_summary(st))
    _set!(sh, "scaleError", ps.scale_error[])
    _set!(sh, "rulerName", ps.ruler_name[])
    _set!(sh, "probeOnPasses", passes_probe_available(wf))
    return
end

_no_mask_editor(::PlanarWorkflow) = "add frames to draw a mask"
_scale_summary(wf::PlanarWorkflow) = scale_description(wf.scale[])

# The planar Region page.
function _refresh_region!(sh::PlanarShell)
    wf = sh.wf
    ed = wf.prepare.roi[]
    roi = wf.roi[]
    sz = ed === nothing ? nothing : ed.size
    rows = roi !== nothing ? roi.rows : sz === nothing ? (1:0) : 1:sz[1]
    cols = roi !== nothing ? roi.cols : sz === nothing ? (1:0) : 1:sz[2]
    _set!(sh, "roiRowFirst", isempty(rows) ? "" : string(first(rows)))
    _set!(sh, "roiRowLast", isempty(rows) ? "" : string(last(rows)))
    _set!(sh, "roiColFirst", isempty(cols) ? "" : string(first(cols)))
    _set!(sh, "roiColLast", isempty(cols) ? "" : string(last(cols)))
    _set!(sh, "roiSummary", ed === nothing ? "add frames to select a region" : Controllers.roi_summary(ed))
    _set!(sh, "roiError", wf.prepare.roi_error[])
    _set!(sh, "hasRoi", roi !== nothing)
    return
end

_window_kind(::PlanarWorkflow) = "planar PIV"

function _refresh_frames!(sh::PlanarShell)
    fs = sh.wf.frames
    problem = frames_problem(fs)
    _set!(sh, "framesSummary", frames_summary(fs))
    _set!(sh, "framesProblem", problem === nothing ? "" : problem)
    _set!(sh, "pairMode", String(fs.pair_mode[]))
    _refresh_pattern!(sh, 0, fs)
    _set!(sh, "shown", String(fs.shown[]))
    fm = sh.wf.frame_masks
    _refresh_pattern!(sh, 3, fm)
    nm = length(fm.files[])
    problem = frame_masks_problem(sh.wf)
    _set!(sh, "frameMaskCount", nm)
    _set!(sh, "frameMasksInfo", problem !== nothing ? problem :
                                nm == 0 ? "none: every pair uses the mask above" :
                                "$nm mask image" * (nm == 1 ? "" : "s") * ", one per frame" *
                                (fm.loading[] ? " (loading the pair's masks…)" :
                                 isempty(fm.load_error[]) ? "" : "; " * fm.load_error[]))
    return
end

# The folder-and-pattern row of camera `k` (0: the planar frame set).
function _refresh_pattern!(sh::WorkflowShell, k::Int, fs::FrameSet)
    n = length(fs.pattern_matches[])
    _set!(sh, "patternDir$k", fs.pattern_dir[])
    _set!(sh, "pattern$k", fs.pattern[])
    _set!(sh, "patternCount$k", n)
    _set!(sh, "patternInfo$k", !isempty(fs.pattern_error[]) ? fs.pattern_error[] :
                               n == 0 ? "" :
                               "$n matching file" * (n == 1 ? "" : "s") * ": " *
                               basename(first(fs.pattern_matches[])) *
                               (n > 1 ? " … " * basename(last(fs.pattern_matches[])) : ""))
    return
end

# The Particles page (planar particle modes): every option as text.
function _refresh_window!(sh::PlanarShell)
    pt = sh.wf.particles
    p = pt.ptv[]
    _set!(sh, "particleMode", Controllers._particle_mode(sh.wf.passes.mode[]))
    _set!(sh, "ptvThreshold", p.threshold === :auto ? "auto" : _num(p.threshold))
    for (key, f) in (("ptvThresholdK", :threshold_k), ("ptvMinSeparation", :min_separation),
                     ("ptvMinDiameter", :min_diameter), ("ptvMaxDiameter", :max_diameter),
                     ("ptvSearchRadius", :search_radius), ("ptvIntensityWeight", :intensity_weight),
                     ("ptvDiameterWeight", :diameter_weight), ("ptvUodThreshold", :uod_threshold),
                     ("ptvUodEpsilon", :uod_epsilon), ("ptvUodNeighbors", :uod_neighbors))
        _set!(sh, key, _num(getfield(p, f)))
    end
    _set!(sh, "ptvUodEnable", p.uod_enable)
    _set!(sh, "ptvPredictor", String(pt.predictor[]))
    _set!(sh, "ptvMinTrackLength", pt.min_track_length[])
    _set!(sh, "ptvMaxGap", pt.max_gap[])
    _set!(sh, "ptvError", pt.error[])
    _set!(sh, "ptvDetectStatus", pt.detect_status[])
    _set!(sh, "frameCount", length(sh.wf.frames.files[]))
    return
end

function _refresh!(sh::WorkflowShell)
    wf = sh.wf
    pe, t, r = wf.passes, wf.test, wf.run
    name = isempty(wf.settings_path[]) ? "untitled settings" : basename(wf.settings_path[])
    modified = try
        settings_modified(wf)
    catch
        true
    end
    _set!(sh, "title", "Hammerhead $(_window_kind(wf)) | " * name * (modified ? " •" : ""))
    _set!(sh, "modality", String(recording_type(wf)))
    _set!(sh, "step", String(wf.step[]))
    _set!(sh, "status", wf.status[])

    _refresh_frames!(sh)
    # the pair bar: the representative pair, or on Results the shown result
    pos, count = pair_position(wf)
    _set!(sh, "pairIndex", pos)
    _set!(sh, "pairCount", count)
    _set!(sh, "autoContrast", sh.canvas.contrast[])
    problem = workflow_problem(wf)
    _set!(sh, "analysisProblem", problem === nothing ? "" : problem)

    _refresh_prepare!(sh)
    _set!(sh, "preset", pe.preset[] === nothing ? "custom" : String(pe.preset[]))
    _set!(sh, "passesError", pe.error[])
    _set!(sh, "passesSummary", passes_summary(pe))
    _set!(sh, "correlation", String(option_value(pe, :correlation)))
    _set!(sh, "subpixel", String(option_value(pe, :subpixel)))
    _set!(sh, "accuracy", option_value(pe, :accuracy))
    _set!(sh, "padding", option_value(pe, :padding))
    _set!(sh, "apodization", option_value(pe, :apodization))
    _set!(sh, "imageInterpolation", String(option_value(pe, :image_interpolation)))
    _set!(sh, "predictorInterpolation", String(option_value(pe, :predictor_interpolation)))
    _set!(sh, "uncertainty", option_value(pe, :uncertainty))
    _set!(sh, "uodEnable", option_value(pe, :uod_enable))
    _set!(sh, "uodThreshold", _num(option_value(pe, :uod_threshold)))
    _set!(sh, "uodNeighborhood", option_value(pe, :uod_neighborhood))
    _set!(sh, "minPeakRatio", _num(option_value(pe, :min_peak_ratio)))
    _set!(sh, "replaceOutliers", option_value(pe, :replace_outliers))
    _set!(sh, "mode", String(pe.mode[]))
    _set!(sh, "precision", string(pe.image_type[]))
    _set!(sh, "gpuOn", pe.backend[] !== :cpu || pe.gpu_loading[])
    _set!(sh, "gpuLoading", pe.gpu_loading[])
    _set!(sh, "gpuStatus", pe.gpu_status[])

    s = test_summary(t)
    _set!(sh, "testRunning", t.running[])
    _set!(sh, "testStatus", t.status[])
    _set!(sh, "testLines", s === nothing ? "" : join(summary_lines(s; previous = t.previous[]), "\n"))
    _set!(sh, "testStale", s !== nothing && test_stale(wf))
    _set!(sh, "testPair", t.pair[])

    _set!(sh, "outputPath", r.output_path[])
    _set!(sh, "runRunning", r.running[])
    _set!(sh, "runDone", r.progress[][1])
    _set!(sh, "runTotal", r.progress[][2])
    _set!(sh, "runStatus", r.status[])
    _set!(sh, "runEta", run_eta(r))
    _set!(sh, "runProgress", run_progress(r))

    ex = wf.explorer[]
    _set!(sh, "resultsLabel", step_status(wf, :results)[2])
    _set!(sh, "resultsFile", wf.results_path[] === nothing ? "" : wf.results_path[])
    if ex !== nothing
        res = current_result(ex)
        fields = available_fields(res)
        _set!(sh, "resultFrame", ex.frame[])
        _set!(sh, "resultFrames", nframes(ex))
        # newline-separated: labels such as "|velocity|" contain "|"
        _set!(sh, "resultFieldKeys", join(String.(fields), "\n"))
        _set!(sh, "resultFieldLabels", join([field_label(res, f) for f in fields], "\n"))
        _set!(sh, "resultField", String(ex.field[]))
        _set!(sh, "resultFieldLabel", field_label(res, ex.field[]))
        _set!(sh, "resultColorMode", String(ex.color_mode[]))
        scale_mode = color_scale_mode(ex)
        _set!(sh, "resultColorScale", String(scale_mode))
        if scale_mode === :absolute
            _set!(sh, "resultColorLow", _num(round(ex.color_min[]; sigdigits = 4)))
            _set!(sh, "resultColorHigh", _num(round(ex.color_max[]; sigdigits = 4)))
        else
            _set!(sh, "resultColorLow", _num(ex.color_percentiles[][1]))
            _set!(sh, "resultColorHigh", _num(ex.color_percentiles[][2]))
        end
        _set!(sh, "resultHasScale", has_scale(ex))
        _set!(sh, "resultPhysical", ex.physical_units[])
        _set!(sh, "resultIncludeFlagged", ex.include_flagged[])
        _set!(sh, "resultIsPlanar", res isa PIVResult)
        rv = revalidation_settings(ex)
        _set!(sh, "resultRevalidate", ex.revalidation[] !== nothing)
        _set!(sh, "rvUodEnable", rv.uod_enable)
        _set!(sh, "rvUodThreshold", _num(rv.uod_threshold))
        _set!(sh, "rvUodNeighborhood", rv.uod_neighborhood)
        _set!(sh, "rvMinPeakRatio", _num(rv.min_peak_ratio))
        _set!(sh, "rvReplace", rv.replace)
        _set!(sh, "resultVectors", ex.show_vectors[])
        _set!(sh, "selectionText", describe_selection(ex))
        _set!(sh, "resultsStatus", ex.status[])
        _set!(sh, "resultTool", String(ex.tool[]))
        _set!(sh, "resultToolsAvailable", res isa PIVResult)
        _set!(sh, "toolSummary", tool_summary(ex))
    end
    _set!(sh, "hasResults", ex !== nothing)
    _set!(sh, "resultsInMemory", results_in_memory(wf))
    # the frame toggles: of the representative pair, or of the result's image
    wf.step[] === :results && ex !== nothing && _set!(sh, "shown", String(ex.image_frame[]))

    _refresh_window!(sh)

    changed = false
    for (row, step) in zip(sh.step_rows, workflow_steps(wf))
        state, summary = step_status(wf, step)
        label = step_label(wf, step)
        (row.label, row.status, row.summary) == (label, String(state), summary) && continue
        row.label = label; row.status = String(state); row.summary = summary
        changed = true
    end
    changed && QML.force_model_update(sh.step_model)

    # Results step shows the results canvas; every other step the image canvas.
    show_figure!(sh.host, wf.step[] === :results && ex !== nothing ? sh.results.fig : sh.canvas.fig)
    return
end

# ---------------------------------------------------------------- QML callbacks

const _SHELL = Ref{Union{Nothing,WorkflowShell}}(nothing)
# `_TICK_HOOK[](shell)` runs on every tick (scripted smoke tests drive the
# window through it); `request_close()` closes the window from Julia.
const _TICK_HOOK = Ref{Any}(nothing)
const _CLOSE_REQUESTED = Ref(false)

"""
    request_close()

Close the open workflow window (from a callback or a tick hook).
"""
request_close() = (_CLOSE_REQUESTED[] = true; nothing)

"""
    request_grab(path)

Save an image of the open workflow window's body (step pages and viewer
canvas) to `path` (PNG) on the window's next tick, through QML
`grabToImage`. This is the way to take screenshots and render checks of a
window: it works from a tick hook or callback, also offscreen or with the
display asleep. The file appears asynchronously, shortly after the tick.
"""
request_grab(path::AbstractString) = (sh = _SHELL[]; sh === nothing || _set!(sh, "grabPath", String(path)); nothing)
hh_grab_started() = (sh = _SHELL[]; sh === nothing || _set!(sh, "grabPath", ""); nothing)

# Run a callback body on the current shell, reporting errors in the status
# line instead of letting them escape into Qt.
function _with_shell(f)
    sh = _SHELL[]
    sh === nothing && return nothing
    try
        f(sh)
    catch err
        sh.wf.status[] = "error: " * Controllers._errmsg(err)
        @error "HammerheadGUI callback failed" exception = (err, catch_backtrace())
    end
    sh.dirty[] = true
    return nothing
end

# Returns 0 (nothing to draw), 1 (redraw the canvases), or 2 (close the window).
# The Hammerhead icon (docs/make_icons.jl writes it).
const ICON_ICO = joinpath(QML_DIR, "icons", "hammerhead.ico")
const _ICON_TRIES = Ref(0)

# Windows: show the icon in the title bar and taskbar. Qt Quick windows have
# no icon property and QML.jl does not expose QGuiApplication's, so the icon
# is sent to the window (found by its title) on the first ticks after it
# opens. Elsewhere the window keeps the platform default.
function _set_window_icon(title::AbstractString)
    Sys.iswindows() || return true
    hwnd = ccall((:FindWindowW, "user32"), Ptr{Cvoid}, (Ptr{Cvoid}, Cwstring), C_NULL, title)
    hwnd == C_NULL && return false
    for (which, metric) in ((0, 49), (1, 11))     # ICON_SMALL/SM_CXSMICON, ICON_BIG/SM_CXICON
        px = ccall((:GetSystemMetrics, "user32"), Cint, (Cint,), metric)
        icon = ccall((:LoadImageW, "user32"), Ptr{Cvoid},
                     (Ptr{Cvoid}, Cwstring, Cuint, Cint, Cint, Cuint),
                     C_NULL, ICON_ICO, 1, px, px, 0x10)      # IMAGE_ICON, LR_LOADFROMFILE
        icon == C_NULL ||
            ccall((:SendMessageW, "user32"), Ptr{Cvoid}, (Ptr{Cvoid}, Cuint, Csize_t, Ptr{Cvoid}),
                  hwnd, 0x0080, which, icon)                 # WM_SETICON
    end
    return true
end

function hh_tick()
    sh = _SHELL[]
    sh === nothing && return 0
    if _ICON_TRIES[] > 0
        _ICON_TRIES[] = _set_window_icon(get(sh.shown, "title", "")) ? 0 : _ICON_TRIES[] - 1
    end
    if _TICK_HOOK[] !== nothing
        try
            _TICK_HOOK[](sh)
        catch err
            @error "tick hook failed" exception = (err, catch_backtrace())
            request_close()
        end
    end
    _CLOSE_REQUESTED[] && (_CLOSE_REQUESTED[] = false; return 2)
    while isready(sh.queue)
        f = take!(sh.queue)
        try
            f()
        catch err
            @error "HammerheadGUI background update failed" exception = (err, catch_backtrace())
        end
    end
    moving = advance!(sh.host)
    sh.dirty[] && (sh.dirty[] = false; _refresh!(sh))
    redraw = sh.canvas.dirty[] || sh.results.dirty[] || moving
    sh.canvas.dirty[] = false
    sh.results.dirty[] = false
    return redraw ? 1 : 0
end

# What the pair bar and pairing controls act on: the frame set, or both
# cameras of a stereo workflow.
_frames_target(wf::PlanarWorkflow) = wf.frames
_frames_target(wf::StereoWorkflow) = wf

_paths(urls) = sort([_url_to_path(u) for u in split(String(urls), '\n') if !isempty(u)])

hh_set_step(name) = _with_shell(sh -> set_step!(sh.wf, Symbol(String(name))))
hh_add_files(urls) = _with_shell(sh -> add_files!(sh.wf.frames, _paths(urls)))
hh_clear_files() = _with_shell(sh -> clear_files!(_frames_target(sh.wf)))
hh_set_pair_mode(mode) = _with_shell(sh -> set_pair_mode!(_frames_target(sh.wf), Symbol(String(mode))))
hh_select_pair(i) = _with_shell(sh -> go_to_pair!(sh.wf, round(Int, i)))
hh_step_pair(delta) = _with_shell(sh -> step_pair!(sh.wf, round(Int, delta)))
hh_set_contrast(on) = _with_shell(sh -> set_contrast!(sh.canvas, sh.wf, Bool(on)))
function hh_set_view_mode(mode)
    _with_shell() do sh
        m = Symbol(String(mode))
        set_view_mode!(sh.canvas.ax, m, :workflow_gesture)
        set_view_mode!(sh.results.ax, m, :results_gesture)
        sh.results.view_mode[] = m
        _set!(sh, "viewMode", String(m))
    end
end
# Show the whole image or field again (ctrl-click does the same).
function hh_reset_view()
    _with_shell() do sh
        results = sh.wf.step[] === :results && sh.wf.explorer[] !== nothing
        reset_limits!(results ? sh.results.ax : sh.canvas.ax)
        (results ? sh.results : sh.canvas).dirty[] = true
    end
end
hh_show_frame(which) = _with_shell(sh -> switch_frame!(sh.wf, Symbol(String(which))))
hh_fill_preset(level) = _with_shell(sh -> fill_preset!(sh.wf.passes, Symbol(String(level))))
function hh_set_pass(i, field, value)
    _with_shell() do sh
        k = round(Int, i)
        # a rejected edit leaves the schedule unchanged: re-send the row so the
        # control shows the kept value again
        set_pass!(sh.wf.passes, k, Symbol(String(field)), value) || (sh.pass_model[k] = sh.pass_rows[k])
    end
end
hh_add_pass() = _with_shell(sh -> add_pass!(sh.wf.passes))
hh_remove_pass(i) = _with_shell(sh -> remove_pass!(sh.wf.passes, round(Int, i)))
function hh_set_option(option, value)
    _with_shell() do sh
        o = Symbol(String(option))
        v = value isa Union{Bool,Number} ? value : Symbol(string(value))
        set_option!(sh.wf.passes, o, v)
    end
end
hh_set_mode(mode) = _with_shell(sh -> set_mode!(sh.wf.passes, Symbol(String(mode))))
function hh_particle_option(option, value)
    _with_shell() do sh
        sh.wf isa PlanarWorkflow || throw(ArgumentError("not a planar window"))
        v = value isa Bool ? value : value isa Real ? _num(value) : string(value)
        edit_particle_option!(sh.wf.particles, Symbol(String(option)), v)
    end
end
hh_set_precision(p) = _with_shell(sh -> set_image_type!(sh.wf.passes, String(p) == "Float32" ? Float32 : Float64))
hh_use_gpu(on) = _with_shell(sh -> use_gpu!(sh.wf, Bool(on)))
hh_test() = _with_shell(sh -> test_pair!(sh.wf))
hh_set_output(url) = _with_shell(sh -> (sh.wf.run.output_path[] = _url_to_path(String(url))))
hh_start_run() = _with_shell(sh -> start_run!(sh.wf))
hh_cancel_run() = _with_shell(sh -> cancel_run!(sh.wf))
# Settings of the other recording type open in a fresh session of that type.
function hh_open_settings(url)
    _with_shell() do sh
        path = _url_to_path(String(url))
        type = recording_type(path)
        if type === nothing || type === recording_type(sh.wf)
            load_settings!(sh.wf, path)
        else
            _request_switch(sh, type, wf -> load_settings!(wf, path))
        end
    end
end

# ---------------------------------------------------------------- recording type

const _PENDING_SWITCH = Ref{Any}(nothing)      # (type, then) awaiting confirmation

hh_set_modality(type) = _with_shell(sh -> _request_switch(sh, Symbol(String(type))))
function hh_confirm_switch()
    _with_shell() do sh
        p = _PENDING_SWITCH[]
        _PENDING_SWITCH[] = nothing
        _set!(sh, "switchQuestion", "")
        p === nothing || _switch_session!(sh, p...)
    end
end
hh_cancel_switch() = _with_shell(sh -> (_PENDING_SWITCH[] = nothing; _set!(sh, "switchQuestion", "")))

# Change the recording type: at once when nothing would be lost, otherwise
# after the window's confirmation (`switchQuestion`). `then(wf)` runs on the
# new session (e.g. opening the settings that asked for the switch).
function _request_switch(sh::WorkflowShell, type::Symbol, then = nothing)
    type in RECORDING_TYPES ||
        throw(ArgumentError("recording type must be planar or stereo, got $type"))
    type === recording_type(sh.wf) && then === nothing && return sh
    q = type === recording_type(sh.wf) ? nothing : switch_question(sh.wf, type)
    if q === nothing
        _switch_session!(sh, type, then)
    else
        _PENDING_SWITCH[] = (type, then)
        _set!(sh, "switchQuestion", q)
    end
    return sh
end

# Replace the window's session with a fresh one of recording type `type`.
function _switch_session!(sh::WorkflowShell, type::Symbol, then = nothing)
    old = sh.wf
    if type !== recording_type(old)
        cancel_run!(old)
        Controllers._abandon_jobs!(old)
        old.deliver[] = f -> f()
        old.spawn[] = false
        wf = new_workflow(type)
        wf.deliver[] = f -> put!(sh.queue, f)
        wf.spawn[] = true
        canvas = wf isa StereoWorkflow ? stereo_canvas(wf) : planar_canvas(wf)
        sh = WorkflowShell(wf, canvas; queue = sh.queue, base = sh)
        _SHELL[] = sh
    end
    then === nothing || then(sh.wf)
    sh.dirty[] = true
    return sh
end
hh_save_settings(url) = _with_shell(sh -> save_settings(sh.wf, _url_to_path(String(url))))
hh_open_results(url) = _with_shell(sh -> begin
    open_results!(sh.wf, _url_to_path(String(url)))
    set_step!(sh.wf, :results)
end)
function hh_save_run_results(url)
    _with_shell() do sh
        path = isempty(String(url)) ? sh.wf.run.output_path[] : _url_to_path(String(url))
        isempty(path) && throw(ArgumentError("choose a results file"))
        save_run_results!(sh.wf, path)
    end
end
hh_toggle_popout() = _with_shell(sh -> toggle_popout!(sh.host))

# Prepare step
hh_set_prepare_page(page) = _with_shell(sh -> set_prepare_page!(sh.wf, Symbol(String(page))))
hh_add_step(op) = _with_shell(sh -> add_step!(sh.wf.prepare.preview, Symbol(String(op))))
hh_remove_step(i) = _with_shell(sh -> remove_step!(sh.wf.prepare.preview, round(Int, i)))
hh_move_step(i, offset) =
    _with_shell(sh -> move_step!(sh.wf.prepare.preview, round(Int, i), round(Int, offset)))
function hh_set_step_option(i, key, value)
    _with_shell() do sh
        k = round(Int, i)
        # a rejected value leaves the step unchanged: re-send the row so the
        # field shows the kept value with the error beneath it
        edit_step_option!(sh.wf, k, String(key), string(value))
        _sync_prep_rows!(sh)
    end
end
hh_estimate_background(n) = _with_shell(sh -> estimate_background!(sh.wf; frames = round(Int, n)))
hh_show_processed(on) = _with_shell(sh -> (sh.wf.prepare.show_processed[] = Bool(on)))
hh_set_probe_window(ws) = _with_shell(sh -> set_probe_window!(sh.wf.prepare.preview, string(ws)))
hh_clear_probe() = _with_shell(sh -> clear_probe!(sh.wf.prepare.preview))

const _MASK_ACTIONS = Dict("hole" => begin_hole!, "undo" => undo_vertex!, "close" => close_active!,
                           "cancel" => cancel_active!, "delete" => delete_selected!,
                           "clear" => clear_polygons!)
function hh_mask_action(name)
    _with_shell() do sh
        me = sh.wf.prepare.mask[]
        me === nothing || _MASK_ACTIONS[String(name)](me)
    end
end
function hh_mask_morph(op, radius)
    _with_shell() do sh
        me = sh.wf.prepare.mask[]
        me === nothing && return
        n = round(Int, radius)
        String(op) == "grow" ? grow_mask!(me, n) : shrink_mask!(me, n)
    end
end
hh_load_mask(url) = _with_shell(sh -> load_mask_file!(sh.wf, _url_to_path(String(url))))
hh_save_mask(url) = _with_shell(sh -> save_mask_file(sh.wf, _url_to_path(String(url))))
hh_set_roi(r1, r2, c1, c2) = _with_shell(sh -> edit_roi!(sh.wf, string(r1), string(r2), string(c1), string(c2)))
hh_clear_roi() = _with_shell(sh -> begin
    ed = sh.wf.prepare.roi[]
    ed === nothing ? (sh.wf.roi[] = nothing) : clear_roi!(ed)
    sh.wf.prepare.roi_error[] = ""
end)
hh_set_scale(field, value) = _with_shell(sh -> edit_scale!(sh.wf, Symbol(String(field)), string(value)))
hh_clear_scale() = _with_shell(sh -> clear_scale!(sh.wf))
hh_load_ruler(url) = _with_shell(sh -> load_ruler!(sh.wf, _url_to_path(String(url))))
hh_clear_ruler() = _with_shell(sh -> clear_ruler!(sh.wf))
# Frames by folder and pattern; camera 0 is the planar window's frame set,
# 3 its per-frame mask images.
_pattern_frames(sh, k) = k == 0 ? sh.wf.frames : k == 3 ? sh.wf.frame_masks : camera_frames(sh.wf, k)
hh_add_frame_masks(urls) = _with_shell(sh -> add_files!(sh.wf.frame_masks, _paths(urls)))
hh_clear_frame_masks() = _with_shell(sh -> clear_files!(sh.wf.frame_masks))
hh_set_frame_pattern(k, dir, pattern) =
    _with_shell(sh -> set_frame_pattern!(_pattern_frames(sh, round(Int, k)), _url_to_path(String(dir)),
                                         String(pattern)))
hh_add_matching(k) = _with_shell(sh -> (n = add_matching!(_pattern_frames(sh, round(Int, k)));
                                        sh.wf.status[] = "added $n " * (k == 3 ? "mask image" : "frame") *
                                                         (n == 1 ? "" : "s")))
function hh_pattern_from_frames(k, urls)
    _with_shell() do sh
        paths = _paths(urls)
        length(paths) == 2 || throw(ArgumentError("choose two frames, e.g. the first pair"))
        dir, pattern = infer_pattern(paths...)
        set_frame_pattern!(_pattern_frames(sh, round(Int, k)), dir, pattern)
    end
end
hh_clear_scale_points() = _with_shell(sh -> begin
    st = sh.wf.prepare.scale[]
    st === nothing || clear_points!(st)
end)

function _with_explorer(f)
    _with_shell() do sh
        ex = sh.wf.explorer[]
        ex === nothing || f(ex)
    end
end
hh_result_frame(i) = _with_explorer(ex -> set_frame!(ex, round(Int, i)))
hh_result_field(key) = _with_explorer(ex -> set_field!(ex, Symbol(String(key))))
hh_result_color_mode(mode) = _with_explorer(ex -> set_color_mode!(ex, Symbol(String(mode))))
hh_result_color_scale(mode) = _with_explorer(ex -> set_color_scale_mode!(ex, Symbol(String(mode))))
function hh_result_color_bounds(lo, hi)
    _with_explorer() do ex
        if color_scale_mode(ex) === :absolute
            set_color_limits!(ex; min = string(lo), max = string(hi))
        else
            set_color_percentiles!(ex, string(lo), string(hi))
        end
    end
end
hh_result_physical(on) = _with_explorer(ex -> set_physical_units!(ex, Bool(on)))
hh_result_include_flagged(on) = _with_explorer(ex -> set_include_flagged!(ex, Bool(on)))
hh_result_revalidate(on) =
    _with_explorer(ex -> Bool(on) ? set_revalidation!(ex, revalidation_settings(ex)) :
                                    set_revalidation!(ex, nothing))
hh_result_revalidation(key, value) =
    _with_explorer(ex -> edit_revalidation!(ex, Symbol(String(key)),
                                            value isa Bool ? value : string(value)))
hh_result_vectors(on) = _with_explorer(ex -> (ex.show_vectors[] = Bool(on)))
hh_result_tool(name) = _with_explorer(ex -> set_tool!(ex, Symbol(String(name))))
hh_result_clear_tool() = _with_explorer(clear_tool!)

function _register_qml_functions()
    @qmlfunction hh_tick hh_grab_started hh_set_step hh_add_files hh_clear_files hh_set_pair_mode hh_select_pair
    @qmlfunction hh_show_frame hh_fill_preset hh_set_pass hh_add_pass hh_remove_pass hh_set_option
    @qmlfunction hh_set_mode hh_set_precision hh_test hh_set_output hh_start_run hh_cancel_run
    @qmlfunction hh_particle_option hh_step_pair hh_set_contrast hh_set_view_mode hh_reset_view
    @qmlfunction hh_open_settings hh_save_settings hh_open_results hh_save_run_results
    @qmlfunction hh_result_color_scale hh_result_color_bounds hh_result_physical
    @qmlfunction hh_result_include_flagged hh_result_revalidate hh_result_revalidation
    @qmlfunction hh_toggle_popout hh_result_frame hh_result_field hh_result_color_mode
    @qmlfunction hh_result_vectors hh_result_tool hh_result_clear_tool
    @qmlfunction hh_set_prepare_page hh_add_step hh_remove_step hh_move_step hh_set_step_option
    @qmlfunction hh_estimate_background hh_show_processed hh_set_probe_window hh_clear_probe
    @qmlfunction hh_mask_action hh_mask_morph hh_load_mask hh_save_mask hh_set_roi hh_clear_roi
    @qmlfunction hh_set_scale hh_clear_scale hh_clear_scale_points hh_load_ruler hh_clear_ruler
    @qmlfunction hh_set_frame_pattern hh_add_matching hh_pattern_from_frames
    @qmlfunction hh_add_frame_masks hh_clear_frame_masks hh_use_gpu
    @qmlfunction hh_set_modality hh_confirm_switch hh_cancel_switch
    _register_stereo_functions()
    return
end

# ---------------------------------------------------------------- launcher

# Open `qml` for `wf` and block until the window closes. `setup()` runs once
# background work goes to the window's queue (so frames added there load on a
# worker); `make_shell(wf, queue)` builds the shell.
function _run_window(setup, wf::AbstractWorkflow, qml::AbstractString, make_shell)
    _SHELL[] === nothing || throw(ArgumentError("a HammerheadGUI window is already open"))
    _check_no_glfw_screens()
    # From here on background work goes to the window's queue and nothing
    # reads image files on this (the GUI) thread: the pair loads on a worker.
    queue = Channel{Any}(Inf)
    previous_deliver, previous_spawn = wf.deliver[], wf.spawn[]
    wf.deliver[] = f -> put!(queue, f)
    wf.spawn[] = true
    try
        setup()
        _qt_init!()
        _QT_OPENED[] = true
        _warm_glyph_atlas!()
        sh = make_shell(wf, queue)
        _SHELL[] = sh
        _CLOSE_REQUESTED[] = false
        _register_qml_functions()
        _ICON_TRIES[] = 120                    # about two seconds of ticks
        loadqml(joinpath(QML_DIR, qml); app = sh.app,
                stepModel = sh.step_model, passModel = sh.pass_model,
                prepModel = sh.prep_model, (Symbol(k) => m for (k, m) in sh.models)...)
        exec()
    finally
        # the session open at the end (a change of recording type replaces it)
        final = _SHELL[] === nothing ? wf : _SHELL[].wf
        cancel_run!(final)
        _SHELL[] = nothing
        _PENDING_SWITCH[] = nothing
        if final === wf
            wf.deliver[] = previous_deliver
            wf.spawn[] = previous_spawn
        else
            final.deliver[] = f -> f()
            final.spawn[] = false
        end
        Controllers._abandon_jobs!(final)
        _release_qml_screens!()
        wf = final
    end
    return wf
end

_entry_list(x) = x isa AbstractString || x isa AbstractMatrix ? [x] : x

"""
    hammerhead(wf = nothing; type = nothing, files = nothing, files1 = nothing,
               files2 = nothing, settings = nothing, dewarpers = nothing,
               calibration = nothing) -> AbstractWorkflow

Open the Hammerhead window and return its session's workflow when the window
closes. The window analyzes one recording at a time, of the type chosen on
the Images step: one camera (planar PIV and particle analysis, a
[`PlanarWorkflow`](@ref Controllers.PlanarWorkflow)) or two cameras (stereo
PIV, a [`StereoWorkflow`](@ref Controllers.StereoWorkflow), with a
Calibration step). Its steps (Images, [Calibration,] Prepare, Passes, Test
pair, Run, Results) share one image canvas, which can be popped out into its
own window. Changing the recording type starts a fresh session, after
confirming what the current one would lose; so does opening settings or
results of the other type.

The session starts from `wf`, or a fresh one of recording `type` (`:planar`
or `:stereo`). Without either, the inputs decide: `files1`, `files2`,
`dewarpers`, `calibration`, or stereo `settings` start a stereo session,
otherwise it is planar. Planar: `files` adds frames (paths in acquisition
order). Stereo: `files1`/`files2` add each camera's frames (entry `i` of both
cameras is the same instant); `dewarpers = (dw1, dw2)` uses `ImageDewarper`s
built in a script instead of the Calibration step's fit; `calibration` opens
a saved camera rig or the calibration stored in a stereo results file.
`settings` opens a recipe or results file. Settings are saved and opened as
core `PIVRecipe` files, and a run's output file also carries its recipe.

The call blocks while the window is open (Qt runs its event loop on this
thread). Start Julia with several threads (`julia -t auto`) so tests and
batch runs leave the window responsive. Closing the window cancels a batch
in progress after its current pair; finished pairs are kept.

The window throws an `ArgumentError` when this Julia session already has a
GLMakie screen (a [`result_explorer`](@ref), [`calibration_review`](@ref), or
[`selfcal_review`](@ref) window, or an offscreen `colorbuffer`): GLMakie's
windows and the Qt window cannot share a process. Restart Julia, and browse
results in the window's Results step.
"""
function hammerhead(wf::Union{Nothing,AbstractWorkflow} = nothing; type = nothing,
                    files = nothing, files1 = nothing, files2 = nothing, settings = nothing,
                    dewarpers = nothing, calibration = nothing)
    stereo_inputs = files1 !== nothing || files2 !== nothing || dewarpers !== nothing ||
                    calibration !== nothing
    if wf === nothing
        type = something(type, stereo_inputs ? :stereo :
                               settings !== nothing && recording_type(settings) === :stereo ?
                               :stereo : :planar)
        wf = new_workflow(type)
    elseif type !== nothing && type !== recording_type(wf)
        throw(ArgumentError("type = :$type does not match the $(recording_type(wf)) workflow given"))
    end
    if wf isa PlanarWorkflow
        stereo_inputs && throw(ArgumentError("files1, files2, dewarpers and calibration " *
                                             "need a stereo session (type = :stereo)"))
    else
        files === nothing || throw(ArgumentError("a stereo session takes files1 and files2, not files"))
    end
    make_shell = (w, q) -> w isa StereoWorkflow ? StereoShell(w; queue = q) : PlanarShell(w; queue = q)
    return _run_window(wf, "HammerheadWindow.qml", make_shell) do
        if wf isa StereoWorkflow
            dewarpers === nothing || set_dewarpers!(wf, dewarpers...)
            calibration === nothing || open_calibration!(wf, calibration)
            files1 === nothing || add_files!(wf, _entry_list(files1); camera = 1)
            files2 === nothing || add_files!(wf, _entry_list(files2); camera = 2)
        else
            files === nothing || add_files!(wf.frames, _entry_list(files))
        end
        settings === nothing || load_settings!(wf, settings)
    end
end

"""
    planar_window(wf = PlanarWorkflow(); files = nothing, settings = nothing)

Open the [`hammerhead`](@ref) window on a one-camera (planar) session:
`hammerhead(wf; files, settings)`.
"""
planar_window(wf::PlanarWorkflow = PlanarWorkflow(); files = nothing, settings = nothing) =
    hammerhead(wf; files, settings)
