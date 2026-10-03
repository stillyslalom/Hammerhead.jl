module ApplicationRequest
using TOML

const KEYS = Set(["schema_version", "session_dir", "project_file", "core_package",
    "gui_package", "experiment", "result", "plot_mode", "visible"])

function read_request(path)
    isabspath(path) || throw(ArgumentError("launch request must be an absolute local path"))
    filesize(path) <= 65536 || throw(ArgumentError("launch request exceeds 64 KiB"))
    data = TOML.parsefile(path)
    Set(keys(data)) == KEYS || throw(ArgumentError("unexpected launch request fields"))
    data["schema_version"] isa Int && data["schema_version"] == 1 ||
        throw(ArgumentError("unsupported launch request version"))
    data["visible"] isa Bool || throw(ArgumentError("visible must be Boolean"))
    for key in setdiff(KEYS, Set(["schema_version", "visible"]))
        data[key] isa String || throw(ArgumentError("$key must be a string"))
    end
    data["plot_mode"] in ("glfw", "preview") || throw(ArgumentError("application plot must be glfw or preview"))
    for key in ("session_dir", "project_file", "core_package", "gui_package")
        isabspath(data[key]) || throw(ArgumentError("$key must be an absolute local path"))
    end
    for key in ("experiment", "result")
        isempty(data[key]) || isabspath(data[key]) || throw(ArgumentError("$key must be an absolute local path"))
    end
    isempty(data["experiment"]) || isempty(data["result"]) ||
        throw(ArgumentError("choose an initial experiment or result"))
    isdir(data["session_dir"]) || throw(ArgumentError("session directory must already exist"))
    data
end

function verify_packages(data, core, gui)
    Base.samefile(data["project_file"], Base.active_project()) || throw(ArgumentError("active project differs from launch request"))
    Base.samefile(data["core_package"], pkgdir(core)) || throw(ArgumentError("loaded Hammerhead differs from launch request"))
    Base.samefile(data["gui_package"], pkgdir(gui)) || throw(ArgumentError("loaded HammerheadGUI differs from launch request"))
    nothing
end

function write_report(path, data)
    temporary = path * ".partial"
    open(temporary, "w") do io
        TOML.print(io, data)
    end
    mv(temporary, path; force=true)
    nothing
end
end
