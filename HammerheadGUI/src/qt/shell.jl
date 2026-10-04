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

"""
State shared between a `PlanarWorkflow` and its QML window.
"""
mutable struct PlanarShell
    wf::PlanarWorkflow
    canvas::PlanarCanvas
    host::CanvasHost
    results::ResultsCanvas
    app::JuliaPropertyMap
    step_rows::Vector{StepRow}
    step_model::JuliaItemModel
    pass_rows::Vector{PassRow}
    pass_model::JuliaItemModel
    queue::Channel{Any}
    dirty::Base.RefValue{Bool}
    shown::Dict{String,Any}
end

function PlanarShell(wf::PlanarWorkflow)
    canvas = planar_canvas(wf)
    app = JuliaPropertyMap()
    host = CanvasHost(app, canvas.fig)
    step_rows = [StepRow(String(s), Controllers.STEP_LABELS[s], "todo", "") for s in WORKFLOW_STEPS]
    pass_rows = PassRow[]
    results = results_canvas()
    sh = PlanarShell(wf, canvas, host, results, app, step_rows, JuliaItemModel(step_rows),
                     pass_rows, JuliaItemModel(pass_rows), Channel{Any}(Inf), Ref(true),
                     Dict{String,Any}())
    mark = (_...) -> (sh.dirty[] = true)
    fs, pe, t, r = wf.frames, wf.passes, wf.test, wf.run
    for obs in (wf.step, wf.preprocessing, wf.mask, wf.roi, wf.scale, wf.saved, wf.settings_path,
                wf.explorer, wf.results_path, wf.status, fs.files, fs.pair_mode, fs.pair, fs.shown,
                pe.passes, pe.preset, pe.mode, pe.image_type, pe.error,
                t.result, t.running, t.status, r.output_path, r.running, r.progress, r.status,
                r.completed)
        on(mark, obs)
    end
    on(pe.passes) do _
        _sync_pass_rows!(sh)
    end
    on(wf.explorer) do ex
        set_explorer!(results, ex)
        ex === nothing && return
        for obs in (ex.frame, ex.field, ex.selection, ex.color_mode, ex.show_vectors, ex.status)
            on(mark, obs)
        end
    end
    set_explorer!(results, wf.explorer[])
    # Every key QML binds to exists from the start (bindings read them at once).
    for (k, v) in ("resultFrame" => 1, "resultFrames" => 1, "resultFieldKeys" => "",
                   "resultFieldLabels" => "", "resultField" => "", "resultFieldLabel" => "", "resultColorMode" => "robust",
                   "resultVectors" => true, "selectionText" => "", "resultsStatus" => "")
        _set!(sh, k, v)
    end
    _sync_pass_rows!(sh)
    _refresh!(sh)
    return sh
end

# Assign a QML property only when its value changed.
function _set!(sh::PlanarShell, key::String, val)
    haskey(sh.shown, key) && isequal(sh.shown[key], val) && return
    sh.shown[key] = val
    sh.app[key] = val
    return
end

function _sync_pass_rows!(sh::PlanarShell)
    rows = pass_rows(sh.wf.passes)
    if length(rows) == length(sh.pass_rows)
        for (i, r) in enumerate(rows)
            new = PassRow(i, r.window, r.search, r.overlap, r.iterations)
            old = sh.pass_rows[i]
            (old.window, old.search, old.overlap, old.iterations) ==
                (new.window, new.search, new.overlap, new.iterations) && continue
            sh.pass_model[i] = new
        end
    else
        empty!(sh.pass_rows)
        append!(sh.pass_rows, [PassRow(i, r.window, r.search, r.overlap, r.iterations)
                               for (i, r) in enumerate(rows)])
        QML.force_model_update(sh.pass_model)
    end
    return
end

function _refresh!(sh::PlanarShell)
    wf = sh.wf
    fs, pe, t, r = wf.frames, wf.passes, wf.test, wf.run
    name = isempty(wf.settings_path[]) ? "untitled settings" : basename(wf.settings_path[])
    modified = try
        settings_modified(wf)
    catch
        true
    end
    _set!(sh, "title", "Hammerhead — planar PIV — " * name * (modified ? " •" : ""))
    _set!(sh, "step", String(wf.step[]))
    _set!(sh, "status", wf.status[])

    problem = frames_problem(fs)
    _set!(sh, "framesSummary", frames_summary(fs))
    _set!(sh, "framesProblem", problem === nothing ? "" : problem)
    _set!(sh, "pairIndex", fs.pair[])
    _set!(sh, "pairCount", npairs(fs))
    _set!(sh, "pairMode", String(fs.pair_mode[]))
    _set!(sh, "shown", String(fs.shown[]))

    _set!(sh, "prepareSummary", step_status(wf, :prepare)[2])
    _set!(sh, "preset", pe.preset[] === nothing ? "custom" : String(pe.preset[]))
    _set!(sh, "passesError", pe.error[])
    _set!(sh, "passesSummary", passes_summary(pe))
    _set!(sh, "correlation", String(option_value(pe, :correlation)))
    _set!(sh, "subpixel", String(option_value(pe, :subpixel)))
    _set!(sh, "accuracy", option_value(pe, :accuracy))
    _set!(sh, "uncertainty", option_value(pe, :uncertainty))
    _set!(sh, "uodThreshold", option_value(pe, :uod_threshold))
    _set!(sh, "mode", String(pe.mode[]))
    _set!(sh, "precision", string(pe.image_type[]))

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

    ex = wf.explorer[]
    _set!(sh, "resultsLabel", step_status(wf, :results)[2])
    _set!(sh, "resultsFile", wf.results_path[] === nothing ? "" : wf.results_path[])
    if ex !== nothing
        r = current_result(ex)
        fields = available_fields(r)
        _set!(sh, "resultFrame", ex.frame[])
        _set!(sh, "resultFrames", nframes(ex))
        _set!(sh, "resultFieldKeys", join(String.(fields), "|"))
        _set!(sh, "resultFieldLabels", join([field_label(r, f) for f in fields], "|"))
        _set!(sh, "resultField", String(ex.field[]))
        _set!(sh, "resultFieldLabel", field_label(r, ex.field[]))
        _set!(sh, "resultColorMode", String(ex.color_mode[]))
        _set!(sh, "resultVectors", ex.show_vectors[])
        _set!(sh, "selectionText", describe_selection(ex))
        _set!(sh, "resultsStatus", ex.status[])
    end
    _set!(sh, "hasResults", ex !== nothing)

    changed = false
    for (row, step) in zip(sh.step_rows, WORKFLOW_STEPS)
        state, summary = step_status(wf, step)
        (row.status, row.summary) == (String(state), summary) && continue
        row.status = String(state); row.summary = summary
        changed = true
    end
    changed && QML.force_model_update(sh.step_model)

    # Results step shows the results canvas; every other step the image canvas.
    show_figure!(sh.host, wf.step[] === :results && ex !== nothing ? sh.results.fig : sh.canvas.fig)
    return
end

# ---------------------------------------------------------------- QML callbacks

const _SHELL = Ref{Union{Nothing,PlanarShell}}(nothing)
# `_TICK_HOOK[](shell)` runs on every tick (scripted smoke tests drive the
# window through it); `request_close()` closes the window from Julia.
const _TICK_HOOK = Ref{Any}(nothing)
const _CLOSE_REQUESTED = Ref(false)

"""
    request_close()

Close the open workflow window (from a callback or a tick hook).
"""
request_close() = (_CLOSE_REQUESTED[] = true; nothing)

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
function hh_tick()
    sh = _SHELL[]
    sh === nothing && return 0
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

hh_set_step(name) = _with_shell(sh -> set_step!(sh.wf, Symbol(String(name))))
hh_add_files(urls) = _with_shell(sh -> begin
    paths = [_url_to_path(u) for u in split(String(urls), '\n') if !isempty(u)]
    add_files!(sh.wf.frames, sort(paths))
end)
hh_clear_files() = _with_shell(sh -> clear_files!(sh.wf.frames))
hh_set_pair_mode(mode) = _with_shell(sh -> set_pair_mode!(sh.wf.frames, Symbol(String(mode))))
hh_select_pair(i) = _with_shell(sh -> select_pair!(sh.wf.frames, round(Int, i)))
hh_show_frame(which) = _with_shell(sh -> show_frame!(sh.wf.frames, Symbol(String(which))))
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
hh_set_precision(p) = _with_shell(sh -> set_image_type!(sh.wf.passes, String(p) == "Float32" ? Float32 : Float64))
hh_test() = _with_shell(sh -> test_pair!(sh.wf))
hh_set_output(url) = _with_shell(sh -> (sh.wf.run.output_path[] = _url_to_path(String(url))))
hh_start_run() = _with_shell(sh -> start_run!(sh.wf))
hh_cancel_run() = _with_shell(sh -> cancel_run!(sh.wf))
hh_open_settings(url) = _with_shell(sh -> load_settings!(sh.wf, _url_to_path(String(url))))
hh_save_settings(url) = _with_shell(sh -> save_settings(sh.wf, _url_to_path(String(url))))
hh_open_results(url) = _with_shell(sh -> begin
    open_results!(sh.wf, _url_to_path(String(url)))
    set_step!(sh.wf, :results)
end)
hh_use_result_settings() = _with_shell(sh -> begin
    path = sh.wf.results_path[]
    path === nothing && throw(ArgumentError("open a results file first"))
    load_settings!(sh.wf, path)
end)
hh_toggle_popout() = _with_shell(sh -> toggle_popout!(sh.host))

function _with_explorer(f)
    _with_shell() do sh
        ex = sh.wf.explorer[]
        ex === nothing || f(ex)
    end
end
hh_result_frame(i) = _with_explorer(ex -> set_frame!(ex, round(Int, i)))
hh_result_field(key) = _with_explorer(ex -> set_field!(ex, Symbol(String(key))))
hh_result_color_mode(mode) = _with_explorer(ex -> set_color_mode!(ex, Symbol(String(mode))))
hh_result_vectors(on) = _with_explorer(ex -> (ex.show_vectors[] = Bool(on)))

function _register_qml_functions()
    @qmlfunction hh_tick hh_set_step hh_add_files hh_clear_files hh_set_pair_mode hh_select_pair
    @qmlfunction hh_show_frame hh_fill_preset hh_set_pass hh_add_pass hh_remove_pass hh_set_option
    @qmlfunction hh_set_mode hh_set_precision hh_test hh_set_output hh_start_run hh_cancel_run
    @qmlfunction hh_open_settings hh_save_settings hh_open_results hh_use_result_settings
    @qmlfunction hh_toggle_popout hh_result_frame hh_result_field hh_result_color_mode
    @qmlfunction hh_result_vectors
    return
end

# ---------------------------------------------------------------- launcher

"""
    planar_window(wf = PlanarWorkflow(); files = nothing, settings = nothing) -> PlanarWorkflow

Open the planar PIV workflow window and return its workflow when the window
closes. The steps — Images, Prepare, Passes, Test pair, Run, Results — share
one image canvas, which can be popped out into its own window.

`files` adds frames (paths in acquisition order); `settings` opens a recipe
or results file. Settings are saved and opened as core `PIVRecipe` files,
and a run's output file also carries the recipe that produced it.

The call blocks while the window is open (Qt runs its event loop on this
thread). Start Julia with several threads (`julia -t auto`) so tests and
batch runs leave the window responsive. Closing the window cancels a batch
in progress after its current pair; finished pairs are kept.
"""
function planar_window(wf::PlanarWorkflow = PlanarWorkflow(); files = nothing, settings = nothing)
    _SHELL[] === nothing || throw(ArgumentError("a HammerheadGUI window is already open"))
    files === nothing || add_files!(wf.frames, files isa AbstractString ? [files] : files)
    settings === nothing || load_settings!(wf, settings)
    _qt_init!()
    sh = PlanarShell(wf)
    previous_deliver = wf.deliver[]
    wf.deliver[] = f -> put!(sh.queue, f)
    _SHELL[] = sh
    _CLOSE_REQUESTED[] = false
    try
        _register_qml_functions()
        loadqml(joinpath(QML_DIR, "PlanarWindow.qml"); app = sh.app,
                stepModel = sh.step_model, passModel = sh.pass_model)
        exec()
    finally
        cancel_run!(wf)
        _SHELL[] = nothing
        wf.deliver[] = previous_deliver
        _release_qml_screens!()
    end
    return wf
end
