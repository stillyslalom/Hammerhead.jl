"""
    ExperimentalQMLSession

Handle returned by [`experimental_qml_gui`](@ref). `session.directory` contains
the launch request, `session.log_path` captures child output, and `ready.toml`
records successful initialization. `session.toml` records shutdown diagnostics.
Use `isopen(session)` to check the child, `wait(session)` to await successful
exit, and `close(session)` to request shutdown and wait for owned resources.
"""
struct ExperimentalQMLSession
    process::Base.Process
    directory::String
    log_path::String
end

Base.isopen(session::ExperimentalQMLSession) = process_running(session.process)

function Base.wait(session::ExperimentalQMLSession)
    wait(session.process)
    success(session.process) || error(
        "Experimental Qt GUI exited with code $(session.process.exitcode), signal $(session.process.termsignal). See $(session.log_path)")
    return session
end

function Base.close(session::ExperimentalQMLSession; timeout::Real=120)
    timeout = Float64(timeout)
    isfinite(timeout) && timeout > 0 || throw(ArgumentError("timeout must be positive and finite"))
    if isopen(session)
        write(joinpath(session.directory, "close.request"), "close\n")
        status = timedwait(() -> !isopen(session), timeout; pollint=0.05)
        status == :timed_out && error(
            "Experimental Qt GUI is still shutting down. Retain the session handle and inspect $(session.log_path)")
    end
    wait(session)
    return nothing
end

function _qml_project_file(project)
    project === nothing && throw(ArgumentError("Activate a Julia project containing Hammerhead, HammerheadGUI, QML and QMLMakie"))
    path = abspath(String(project))
    isdir(path) && (path = joinpath(path, "Project.toml"))
    isfile(path) || throw(ArgumentError("Julia project file does not exist: $path"))
    project_data = TOML.parsefile(path)
    dependencies = get(project_data, "deps", Dict())
    required = ["Hammerhead" => string(Base.PkgId(Hammerhead).uuid),
                "HammerheadGUI" => string(Base.PkgId(@__MODULE__).uuid),
                "QML" => "2db162a6-7e43-52c3-8d84-290c1c42d82a",
                "QMLMakie" => "08f9cac3-3b11-4f1c-9d88-d0e81c500f64"]
    # Julia makes the active package itself available alongside its dependencies.
    available(name, uuid) = get(dependencies, name, nothing) == uuid ||
        (get(project_data, "name", nothing) == name && get(project_data, "uuid", nothing) == uuid)
    missing = [name for (name, uuid) in required if !available(name, uuid)]
    isempty(missing) || throw(ArgumentError(
        "Install $(join(missing, ", ")) in $path with Pkg.add before launching the experimental Qt GUI"))
    return realpath(path)
end

function _qml_input_path(value, label)
    value === nothing && return ""
    path = abspath(String(value))
    isfile(path) || throw(ArgumentError("$label file does not exist: $path"))
    return realpath(path)
end

function _qml_environment(visible::Bool)
    env = Dict{String,String}((Sys.iswindows() ? uppercase(key) : key) => value for (key, value) in ENV)
    for key in ("QT_QPA_PLATFORM", "QT_QUICK_BACKEND", "QSG_RHI_BACKEND", "QSG_RENDER_LOOP",
                "QT_QUICK_CONTROLS_STYLE", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH",
                "QMLSCENE_DEVICE", "QSG_INFO")
        pop!(env, key, nothing)
    end
    visible || (env["QT_QPA_PLATFORM"] = "offscreen")
    env["QSG_RENDER_LOOP"] = "basic"
    env["QSG_RHI_BACKEND"] = "opengl"
    env["QT_QUICK_CONTROLS_STYLE"] = "Basic"
    env["QT_QUICK_BACKEND"] = "software"
    return env
end

"""
    experimental_qml_gui(; experiment=nothing, result=nothing,
                         session_dir=nothing, project=Base.active_project(),
                         visible=true, plot=:glfw, wait=false)

Launch the experimental Qt controls and scientific plot in a fresh Julia
process. Install `Hammerhead`, `HammerheadGUI`, `QML` and `QMLMakie` in the active
project first. The child verifies that it uses the caller's Hammerhead packages.

Supply `experiment` to open a saved planar experiment, or `result` to inspect a
planar result file. With neither argument, the window opens a synthetic example.
Replay starts from the controls. `plot=:glfw` opens an interactive scientific
window; `plot=:preview` places a rendered preview in the controls window.

Returns an [`ExperimentalQMLSession`](@ref). `close(session)` requests an orderly
shutdown, including any active replay. `wait=true` waits for the application to
exit before returning. Each launch stores logs and diagnostics in a new session
directory. `session_dir` may name a directory to create; its parent must exist.
`visible=false` runs the same application with hidden windows for automation.
"""
function experimental_qml_gui(; experiment=nothing, result=nothing,
                              session_dir=nothing, project=Base.active_project(),
                              visible::Bool=true, plot::Symbol=:glfw, wait::Bool=false)
    experiment !== nothing && result !== nothing && throw(ArgumentError("Choose one initial experiment or result"))
    plot in (:glfw, :preview) || throw(ArgumentError("plot must be :glfw or :preview"))
    project_file = _qml_project_file(project)
    experiment_path = _qml_input_path(experiment, "Experiment")
    result_path = _qml_input_path(result, "Result")
    directory = if session_dir === nothing
        mktempdir(; prefix="hammerhead-qml-", cleanup=false)
    else
        path = abspath(String(session_dir))
        mkdir(path)
        path
    end
    directory = realpath(directory)
    request_path = joinpath(directory, "launch.toml")
    request = Dict{String,Any}(
        "schema_version" => 1, "session_dir" => directory,
        "project_file" => project_file,
        "core_package" => realpath(pkgdir(Hammerhead)),
        "gui_package" => realpath(pkgdir(@__MODULE__)),
        "experiment" => experiment_path, "result" => result_path,
        "plot_mode" => String(plot), "visible" => visible)
    open(request_path, "w") do io
        TOML.print(io, request; sorted=true)
    end
    script = joinpath(pkgdir(@__MODULE__), "prototypes", "qml", "run.jl")
    pools = Threads.nthreads(:interactive) == 0 ? string(Threads.nthreads(:default)) :
        "$(Threads.nthreads(:default)),$(Threads.nthreads(:interactive))"
    cmd = `$(Base.julia_cmd()) --startup-file=no --threads=$pools --project=$(dirname(project_file)) $script --launch-request=$request_path`
    cmd = Cmd(cmd; dir=directory, windows_hide=true)
    log_path = joinpath(directory, "session.log")
    child = open(log_path, "w") do io
        run(pipeline(setenv(cmd, _qml_environment(visible)); stdout=io, stderr=io); wait=false)
    end
    session = ExperimentalQMLSession(child, directory, log_path)
    wait && Base.wait(session)
    return session
end
