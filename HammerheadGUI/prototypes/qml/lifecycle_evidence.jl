module LifecycleEvidence
using SHA, TOML, Pkg

const QT_ENV_KEYS = ("QT_QPA_PLATFORM", "QSG_RENDER_LOOP", "QSG_RHI_BACKEND",
                     "QT_QUICK_BACKEND", "QT_QUICK_CONTROLS_STYLE", "QSG_INFO", "QMLSCENE_DEVICE")

qt_variants() = Dict(String(key) => String(value) for (key, value) in ENV
    if uppercase(String(key)) in QT_ENV_KEYS)

digest(path) = open(io -> bytes2hex(sha256(io)), path, "r")
function source_files(root)
    result = Dict{String,String}()
    for (dir, directories, files) in walkdir(root)
        filter!(name -> name != "artifacts", directories)
        for file in sort(files)
            (endswith(file, ".jl") || endswith(file, ".qml") || file in ("Project.toml", "Manifest.toml")) || continue
            path = joinpath(dir, file)
            "artifacts" in splitpath(relpath(path, root)) && continue
            result[replace(relpath(path, root), '\\' => '/')] = digest(path)
        end
    end
    result
end

function provenance(root)
    packages = Dict{String,Any}()
    for info in values(Pkg.dependencies())
        info.name in ("QML", "QMLMakie", "CxxWrap", "jlqml_jll", "Qt6Base_jll",
                      "Qt6Declarative_jll", "GLMakie", "Makie", "Hammerhead", "HammerheadGUI") || continue
        entry = Dict{String,Any}("version" => string(info.version), "source" => string(info.source))
        if info.name in ("QML", "QMLMakie", "GLMakie", "Makie", "Hammerhead", "HammerheadGUI")
            source = joinpath(info.source, "src")
            isdir(source) && (entry["source_sha256"] = source_files(source))
        end
        packages[info.name] = entry
    end
    result = Dict{String,Any}("julia" => string(VERSION), "os" => string(Sys.KERNEL),
        "architecture" => string(Sys.ARCH), "pid" => getpid(), "threads" => Threads.nthreads(),
        "active_project" => something(Base.active_project(), ""), "packages" => packages,
        "prototype_sha256" => source_files(root),
        "qt_environment" => Dict(key => get(ENV, key, "") for key in QT_ENV_KEYS),
        "qt_environment_variants" => qt_variants())
    manifest = joinpath(root, "Manifest.toml")
    isfile(manifest) && (result["manifest_sha256"] = digest(manifest))
    result
end

function write_fresh(path, data)
    (ispath(path) || islink(path)) && throw(ArgumentError("evidence destination exists: $path"))
    open(path, "w") do io
        TOML.print(io, data; sorted = true)
        flush(io)
    end
    path
end

mutable struct Journal
    directory::String
    sequence::Int
    started::UInt64
end
function Journal(directory)
    mkdir(directory)
    Journal(directory, 0, time_ns())
end
function stage!(journal::Journal, name; kwargs...)
    journal.sequence += 1
    data = Dict{String,Any}("sequence" => journal.sequence, "stage" => String(name),
        "elapsed_seconds" => (time_ns() - journal.started) / 1e9,
        "julia_thread" => Threads.threadid(), "pid" => getpid())
    for (key, value) in kwargs
        data[String(key)] = value
    end
    path = joinpath(journal.directory, lpad(string(journal.sequence), 5, '0') * ".toml")
    write_fresh(path, data)
    println("LIFECYCLE_STAGE ", journal.sequence, " ", name)
    flush(stdout)
    nothing
end
end
