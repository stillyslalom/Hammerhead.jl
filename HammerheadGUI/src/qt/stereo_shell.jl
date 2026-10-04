# The stereo window's part of the Qt shell (shell.jl holds the shared
# machinery): the two cameras' frame lists, the Calibration step's fields and
# plate lists, the dt-only scale, and the stereo callbacks.

# One calibration plate of a camera; `info` is its fit summary once fitted.
mutable struct PlateRow
    number::Int
    label::String
    z::String
    info::String
end

const StereoShell = WorkflowShell{StereoWorkflow,StereoCanvas}

StereoShell(wf::StereoWorkflow; queue::Channel{Any} = Channel{Any}(Inf)) =
    WorkflowShell(wf, stereo_canvas(wf); queue)

function _connect_window!(sh::StereoShell, mark)
    wf = sh.wf
    for fs in (wf.frames1, wf.frames2)
        for obs in (fs.files, fs.pair_mode, fs.pair, fs.shown, fs.loading, fs.loaded, fs.load_error,
                    fs.pattern_dir, fs.pattern, fs.pattern_matches, fs.pattern_error)
            on(mark, obs)
        end
    end
    on(mark, wf.camera)
    on(mark, wf.dewarped)
    cal = wf.calibration
    for f in fieldnames(StereoCalibration)
        v = getfield(cal, f)
        v isa Observable && on(mark, v)
        v isa Tuple && foreach(o -> o isa Observable && on(mark, o), v)
    end
    # a review's plane selection and refits
    function watch_review(cr)
        cr === nothing && return
        on(mark, cr.plane)
        on(mark, cr.camera)
        return
    end
    for k in 1:2
        on(watch_review, cal.reviews[k])
        watch_review(cal.reviews[k][])
        rows = PlateRow[]
        sh.rows["plates$k"] = rows
        sh.models["plates$(k)Model"] = JuliaItemModel(rows)
    end
    return sh
end

_window_kind(::StereoWorkflow) = "stereo PIV"
_no_mask_editor(::StereoWorkflow) = "build the dewarp grid (Calibration step) to draw a mask"
# No Region page; its keys stay empty for the shared page set.
function _refresh_region!(sh::StereoShell)
    for k in ("roiRowFirst", "roiRowLast", "roiColFirst", "roiColLast", "roiSummary", "roiError")
        _set!(sh, k, "")
    end
    _set!(sh, "hasRoi", false)
    return
end

function _scale_summary(wf::StereoWorkflow)
    sc = wf.scale[]
    unit = wf.calibration.length_unit[]
    sc === nothing && return "no time scale: velocities stay in $unit per frame"
    tu = sc.time_unit == "frame" && sc.dt != 1 ? "frames" : sc.time_unit
    return "$(_num(sc.dt)) $tu between exposures · lengths in $(sc.length_unit)"
end

function _refresh_frames!(sh::StereoShell)
    wf = sh.wf
    problem = frames_problem(wf)
    _set!(sh, "framesSummary", frames_summary(wf))
    _set!(sh, "framesProblem", problem === nothing ? "" : problem)
    _set!(sh, "pairMode", String(wf.frames1.pair_mode[]))
    _set!(sh, "shown", String(wf.frames1.shown[]))
    _set!(sh, "camera", wf.camera[])
    for k in 1:2
        fs = camera_frames(wf, k)
        p = frames_problem(fs)
        _set!(sh, "camera$(k)Frames", length(fs.files[]))
        _set!(sh, "camera$(k)Summary", frames_summary(fs))
        _set!(sh, "camera$(k)Problem", p === nothing ? "" : p)
        _refresh_pattern!(sh, k, fs)
    end
    return
end

_plate_name(image) = image isa AbstractString ? basename(image) : "$(size(image, 2))×$(size(image, 1)) array"

# Per-plate fit summaries of camera `k`, when its review matches the plates.
function _plate_infos(sh::StereoShell, k::Int)
    cal = sh.wf.calibration
    cr = cal.reviews[k][]
    n = length(cal.plates[k][])
    (cr === nothing || nplanes(cr) != n || fit_stale(cal)) && return fill("", n)
    key = (cr, cr.camera[])
    cached = get(sh.rows, "infos$k", nothing)
    cached !== nothing && cached[1] === key[1] && cached[2] === key[2] && return cached[3]
    infos = map(1:n) do i
        g = cr.grids[i]
        pe = plane_errors(cr, i)
        s = "$(length(g.pixels)) dots"
        if pe !== nothing && !isempty(pe.errors)
            rms = sqrt(sum(abs2, pe.errors) / length(pe.errors))
            r2(x) = Controllers.display_number(round(x; sigdigits = 2))
            s *= " · rms $(r2(rms)) · max $(r2(maximum(pe.errors))) px"
        end
        s
    end
    sh.rows["infos$k"] = (key..., infos)
    return infos
end

function _sync_plate_rows!(sh::StereoShell)
    cal = sh.wf.calibration
    for k in 1:2
        infos = _plate_infos(sh, k)
        new = [PlateRow(i, _plate_name(p.image), _num(p.z), infos[i])
               for (i, p) in enumerate(cal.plates[k][])]
        _sync_rows!(sh.rows["plates$k"], sh.models["plates$(k)Model"], new,
                    (a, b) -> (a.label, a.z, a.info) == (b.label, b.z, b.info))
    end
    return
end

function _refresh_window!(sh::StereoShell)
    wf = sh.wf
    cal = wf.calibration
    _sync_plate_rows!(sh)
    k = wf.camera[]
    cr = cal.reviews[k][]
    _set!(sh, "calPlane", cr === nothing ? 0 : cr.plane[])
    _set!(sh, "calPlates1", length(cal.plates[1][]))
    _set!(sh, "calPlates2", length(cal.plates[2][]))
    _set!(sh, "calSpacing", cal.spacing[] === nothing ? "" : _num(cal.spacing[]))
    _set!(sh, "calTwoLevel", cal.two_level[])
    _set!(sh, "calLevelSeparation", _num(cal.level_separation[]))
    off = cal.origin_offset[]
    _set!(sh, "calOriginOffset", off === nothing ? "" : _num(off[1]) * ", " * _num(off[2]))
    _set!(sh, "calInvert", cal.invert[])
    _set!(sh, "calOrientation", String(cal.orientation[]))
    _set!(sh, "calModel", String(cal.model[]))
    _set!(sh, "calLengthUnit", cal.length_unit[])
    _set!(sh, "calError", cal.error[])
    _set!(sh, "calFitting", cal.fitting[])
    _set!(sh, "calFitStatus", cal.fit_status[])
    _set!(sh, "calFitStale", fit_stale(cal))
    _set!(sh, "calCoverage", String(cal.coverage[]))
    gs = cal.grid_spacing[]
    _set!(sh, "calGridSpacing", gs isa Symbol ? "auto" : _num(gs))
    _set!(sh, "calGridZ", _num(cal.grid_z[]))
    _set!(sh, "calMargin", _num(cal.margin[]))
    _set!(sh, "calBuilding", cal.building[])
    _set!(sh, "calGridStatus", cal.grid_status[])
    _set!(sh, "hasDewarpers", cal.dewarpers[] !== nothing)
    _set!(sh, "calFitted", Controllers._fitted(cal))
    _set!(sh, "calSelfcalPairs", cal.selfcal_pairs[])
    _set!(sh, "calKeepMaps", cal.keep_disparity_maps[])
    _set!(sh, "calSelfcalRunning", cal.selfcal_running[])
    report = cal.selfcal[] === nothing ? "" : selfcal_summary(cal.selfcal[].report)
    status = cal.selfcal_status[]
    _set!(sh, "calSelfcalReport", report)
    _set!(sh, "calSelfcalStatus", status == report ? "" : status)
    _set!(sh, "hasSelfcal", cal.selfcal[] !== nothing)
    _set!(sh, "calSelfcalApplied", cal.selfcal_applied[])
    _set!(sh, "calSummary", calibration_summary(cal))
    _set!(sh, "calCanBuild", can_build_grid(cal))
    _set!(sh, "calibrationPage", String(cal.page[]))
    sc = cal.selfcal[]
    _set!(sh, "calSelfcalPasses", sc === nothing ? 0 : length(sc.report.passes))
    _set!(sh, "calHasMaps", sc !== nothing && !isempty(sc.report.disparity_maps))
    _set!(sh, "calDisparityPass", cal.disparity_pass[])
    # the Prepare step: per-camera preprocessing; the scale's lengths are world units
    _set!(sh, "separatePreprocessing", separate_preprocessing(wf))
    _set!(sh, "scaleWorldUnit", Controllers._current_scale(wf).length_unit)
    return
end

# ---------------------------------------------------------------- callbacks

function _with_stereo(f)
    _with_shell() do sh
        sh.wf isa StereoWorkflow || throw(ArgumentError("not a stereo window"))
        f(sh.wf)
    end
end

_camera(k) = round(Int, k)

hh_set_camera(k) = _with_stereo(wf -> set_camera!(wf, _camera(k)))
hh_add_camera_files(k, urls) = _with_stereo(wf -> add_files!(wf, _paths(urls); camera = _camera(k)))
hh_clear_camera_files(k) = _with_stereo(wf -> clear_files!(wf; camera = _camera(k)))

# New plates start at z = 0; the page asks for each plate's z.
hh_add_plates(k, urls) = _with_stereo(wf -> begin
    for path in _paths(urls)
        add_plate!(wf.calibration, _camera(k), path, 0.0)
    end
    set_camera!(wf, _camera(k))
end)
hh_remove_plate(k, i) = _with_stereo(wf -> remove_plate!(wf.calibration, _camera(k), round(Int, i)))
hh_clear_plates(k) = _with_stereo(wf -> clear_plates!(wf.calibration, _camera(k)))
function hh_set_plate_z(k, i, z)
    _with_stereo() do wf
        cal = wf.calibration
        try
            set_plate_z!(cal, _camera(k), round(Int, i), string(z))
            isempty(cal.error[]) || (cal.error[] = "")
        catch err
            cal.error[] = Controllers._errmsg(err)
            # re-send the rows so the field shows the kept value
            sh = _SHELL[]
            sh === nothing || QML.force_model_update(sh.models["plates$(_camera(k))Model"])
        end
    end
end
# Show a plate on the viewer: its camera, and its plane once fitted.
hh_select_plate(k, i) = _with_stereo(wf -> begin
    set_camera!(wf, _camera(k))
    cr = wf.calibration.reviews[_camera(k)][]
    cr === nothing || round(Int, i) > nplanes(cr) || set_plane!(cr, round(Int, i))
end)
hh_calibration_option(option, value) =
    _with_stereo(wf -> edit_calibration_option!(wf.calibration, Symbol(String(option)),
                                                value isa Bool ? value :
                                                value isa Real ? _num(value) : string(value)))
hh_set_calibration_page(name) = _with_stereo(wf -> set_calibration_page!(wf.calibration, String(name)))
hh_set_disparity_pass(i) = _with_stereo(wf -> set_disparity_pass!(wf.calibration, round(Int, i)))
hh_fit_calibration() = _with_stereo(wf -> fit_calibration!(wf.calibration))
hh_build_dewarpers() = _with_stereo(wf -> build_dewarpers!(wf.calibration))
hh_start_selfcal() = _with_stereo(start_selfcal!)
hh_apply_selfcal() = _with_stereo(apply_selfcal!)
hh_open_calibration(url) = _with_stereo(wf -> open_calibration!(wf, _url_to_path(String(url))))
hh_save_calibration(url) = _with_stereo(wf -> save_calibration_file(wf, _url_to_path(String(url))))
hh_set_separate_preprocessing(on) = _with_stereo(wf -> set_separate_preprocessing!(wf, Bool(on)))

function _register_stereo_functions()
    @qmlfunction hh_set_camera hh_add_camera_files hh_clear_camera_files hh_add_plates hh_remove_plate
    @qmlfunction hh_clear_plates hh_set_plate_z hh_select_plate hh_calibration_option
    @qmlfunction hh_fit_calibration hh_build_dewarpers hh_start_selfcal hh_apply_selfcal
    @qmlfunction hh_set_calibration_page hh_open_calibration hh_save_calibration
    @qmlfunction hh_set_separate_preprocessing hh_set_disparity_pass
    return
end

# ---------------------------------------------------------------- launcher

"""
    stereo_window(wf = StereoWorkflow(); files1 = nothing, files2 = nothing,
                  settings = nothing, dewarpers = nothing,
                  calibration = nothing) -> StereoWorkflow

Open the stereo PIV workflow window and return its workflow when the window
closes. The steps — Images, Calibration, Prepare, Passes, Test pair, Run,
Results — share one image canvas (camera 1 or 2, switched in the bar below
it), which can be popped out into its own window.

`files1`/`files2` add each camera's frames (paths in acquisition order;
entry `i` of both cameras is the same instant); `dewarpers = (dw1, dw2)`
uses `ImageDewarper`s built in a script instead of the Calibration step's
fit; `calibration` opens a saved camera rig (`Hammerhead.save_calibration`,
or the calibration stored in a stereo results file); `settings` opens a
recipe or results file. The settings and the calibration save separately:
the Calibration step has its own Open and Save buttons.

The call blocks while the window is open, as [`planar_window`](@ref) does;
start Julia with several threads (`julia -t auto`). Like `planar_window`, it
throws an `ArgumentError` when this Julia session already has a GLMakie
screen (e.g. a [`calibration_review`](@ref) window): review calibrations in
the Calibration step, or in a separate Julia session.
"""
function stereo_window(wf::StereoWorkflow = StereoWorkflow(); files1 = nothing, files2 = nothing,
                       settings = nothing, dewarpers = nothing, calibration = nothing)
    return _run_window(wf, "StereoWindow.qml", (w, q) -> StereoShell(w; queue = q)) do
        dewarpers === nothing || set_dewarpers!(wf, dewarpers...)
        calibration === nothing || open_calibration!(wf, calibration)
        files1 === nothing || add_files!(wf, _entry_list(files1); camera = 1)
        files2 === nothing || add_files!(wf, _entry_list(files2); camera = 2)
        settings === nothing || load_settings!(wf, settings)
    end
end
