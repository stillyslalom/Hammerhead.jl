using Test, HammerheadGUI, TOML

@testset "Experimental QML launch guards" begin
    mktempdir() do directory
        session_path = joinpath(directory, "session")
        optional_project = joinpath(pkgdir(HammerheadGUI), "prototypes", "qml", "Project.toml")
        missing = joinpath(directory, "missing")
        for options in ((experiment=missing, result=missing), (plot=:embedded,),
                        (project=nothing,), (project=missing,))
            @test_throws ArgumentError experimental_qml_gui(; options..., session_dir=session_path, visible=false)
            @test !ispath(session_path)
        end
        for input in (:experiment, :result)
            @test_throws ArgumentError experimental_qml_gui(;
                Dict(input=>missing)..., project=optional_project,
                session_dir=session_path, visible=false)
            @test !ispath(session_path)
        end
        partial_project = joinpath(directory, "Project.toml")
        write(partial_project, "[deps]\nHammerheadGUI = \"759a4b49-c2fc-4731-b9e2-38749bc182b9\"\n")
        error = try
            experimental_qml_gui(; project=partial_project, session_dir=session_path, visible=false)
            nothing
        catch exception
            exception
        end
        @test error isa ArgumentError
        @test all(name -> occursin(name, sprint(showerror, error)), ("Hammerhead", "QML", "QMLMakie"))
        @test !ispath(session_path)
        @test HammerheadGUI._qml_project_file(dirname(optional_project)) == realpath(optional_project)
        mkdir(session_path)
        sentinel = joinpath(session_path, "keep.txt"); write(sentinel, "existing session")
        @test_throws Exception experimental_qml_gui(; project=optional_project,
            session_dir=session_path, visible=false)
        @test read(sentinel, String) == "existing session"
        @test readdir(session_path) == ["keep.txt"]
        @test_throws Exception experimental_qml_gui(; project=optional_project,
            session_dir=partial_project, visible=false)
        @test startswith(read(partial_project, String), "[deps]")
    end
end

@testset "Experimental QML child environment" begin
    withenv("QMLSCENE_DEVICE"=>"inherited-device", "QT_QUICK_BACKEND"=>"inherited-backend",
            "QT_QPA_PLATFORM"=>"inherited-platform", "QT_PLUGIN_PATH"=>"inherited-plugins") do
        hidden = HammerheadGUI._qml_environment(false)
        @test hidden["QT_QPA_PLATFORM"] == "offscreen"
        @test hidden["QT_QUICK_BACKEND"] == "software"
        @test hidden["QSG_RENDER_LOOP"] == "basic"
        @test hidden["QSG_RHI_BACKEND"] == "opengl"
        @test hidden["QT_QUICK_CONTROLS_STYLE"] == "Basic"
        @test !any(key -> uppercase(key) in ("QMLSCENE_DEVICE", "QT_PLUGIN_PATH"), keys(hidden))
        visible = HammerheadGUI._qml_environment(true)
        @test !haskey(visible, "QT_QPA_PLATFORM")
        @test ENV["QT_QPA_PLATFORM"] == "inherited-platform"
        @test ENV["QMLSCENE_DEVICE"] == "inherited-device"
    end
end

module ExperimentalApplicationRequestTests
using Test, TOML, HammerheadGUI
include(joinpath(pkgdir(HammerheadGUI), "prototypes", "qml", "application_request.jl"))

@testset "Experimental QML request boundary" begin
    mktempdir() do directory
        request = Dict{String,Any}("schema_version"=>1, "session_dir"=>realpath(directory),
            "project_file"=>realpath(Base.active_project()),
            "core_package"=>realpath(pkgdir(HammerheadGUI.Hammerhead)),
            "gui_package"=>realpath(pkgdir(HammerheadGUI)), "experiment"=>"", "result"=>"",
            "plot_mode"=>"glfw", "visible"=>false)
        path = joinpath(directory, "launch.toml")
        write_request(data) = open(io -> TOML.print(io, data), path, "w")
        write_request(request)
        @test ApplicationRequest.read_request(path) == request
        @test ApplicationRequest.verify_packages(request, HammerheadGUI.Hammerhead, HammerheadGUI) === nothing
        for changed in (merge(request, Dict("schema_version"=>true)),
                        merge(request, Dict("schema_version"=>2)),
                        merge(request, Dict("visible"=>"false")),
                        merge(request, Dict("plot_mode"=>"embedded")),
                        merge(request, Dict("experiment"=>"relative.jld2")),
                        merge(request, Dict("experiment"=>path, "result"=>path)),
                        merge(request, Dict("extra"=>"field")))
            write_request(changed)
            @test_throws ArgumentError ApplicationRequest.read_request(path)
        end
        for key in ("project_file", "core_package", "gui_package")
            @test_throws ArgumentError ApplicationRequest.verify_packages(
                merge(request, Dict(key=>realpath(directory))), HammerheadGUI.Hammerhead, HammerheadGUI)
        end
    end
end
end
