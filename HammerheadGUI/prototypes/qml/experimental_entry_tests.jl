using Test, HammerheadGUI, TOML, SHA
using HammerheadGUI.Hammerhead
include("lifecycle_evidence.jl")
include("experiment_fixture.jl")
include(joinpath(pkgdir(HammerheadGUI), "test", "test_qml_gui.jl"))

function entry_wait_file(session, filename; timeout=240)
    path = joinpath(session.directory, filename)
    result = timedwait(() -> isfile(path) || !isopen(session), timeout; pollint=.05)
    result === :ok && isfile(path) || error(
        "Experimental entry did not publish $filename; inspect $(session.log_path)")
    TOML.parsefile(path)
end

function entry_capture(session)
    write(joinpath(session.directory, "capture.request"), "capture\n")
    data = entry_wait_file(session, "capture.toml"; timeout=90)
    @test data["status"] == "captured"
    for kind in ("controls", "scientific")
        filename = data[kind*"_file"]
        path = isabspath(filename) ? filename : joinpath(session.directory, filename)
        @test isfile(path) && filesize(path) > 1024
        @test open(io -> read(io, 8), path) == UInt8[0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a]
        @test bytes2hex(sha256(read(path))) == data[kind*"_sha256"]
    end
    data
end

function entry_case(foreign, project, source; experiment=false, plot=:glfw)
    directory = joinpath(foreign, "session $(plot) µ #%")
    session = cd(foreign) do
        options = experiment ? (experiment=relpath(source, foreign),) : (result=relpath(source, foreign),)
        experimental_qml_gui(; options..., project=relpath(project, foreign),
            session_dir=basename(directory), visible=false, plot)
    end
    try
        @test session isa ExperimentalQMLSession && isopen(session)
        @test session.directory == realpath(directory)
        request = TOML.parsefile(joinpath(directory, "launch.toml"))
        @test request[experiment ? "experiment" : "result"] == realpath(source)
        @test request["project_file"] == realpath(project)
        @test request["visible"] === false && request["plot_mode"] == String(plot)
        ready = entry_wait_file(session, "ready.toml")
        @test ready["schema_version"] == 1 && ready["status"] == "ready"
        @test ready["pid"] == Int(getpid(session.process))
        @test ready["session_dir"] == session.directory
        @test Base.samefile(ready["project_file"], project)
        @test Base.samefile(ready["core_package"], pkgdir(Hammerhead))
        @test Base.samefile(ready["gui_package"], pkgdir(HammerheadGUI))
        @test ready["visible"] === false && ready["plot_mode"] == String(plot)
        @test ready["replay_running"] === false && ready["smoke_automation"] === false
        @test ready["viewport_initialized"] === true && ready["frame"] == 1
        @test ready["nthreads_default"] == Threads.nthreads(:default)
        @test ready["nthreads_interactive"] == Threads.nthreads(:interactive)
        if experiment
            record = load_experiment(source)
            @test ready["selected_lane"] == "experiment"
            @test ready["recipe_id"] == recipe_identity(record.recipe)
            @test ready["input_id"] == record.input_id
            @test occursin("demo", lowercase(ready["displayed_identity"]))
        else
            @test ready["selected_lane"] == "result"
            @test ready["recipe_id"] == "" && ready["input_id"] == ""
            @test occursin(basename(source), ready["displayed_identity"])
        end
        capture = entry_capture(session)
        @test capture["displayed_identity"] == ready["displayed_identity"]
        @test capture["recipe_id"] == ready["recipe_id"] && capture["input_id"] == ready["input_id"]
        @test_throws ArgumentError close(session; timeout=0)
        @test isopen(session)
        @test close(session; timeout=120) === nothing
        @test !isopen(session) && wait(session) === session
        @test session.process.exitcode == 0 && session.process.termsignal == 0
        @test close(session) === nothing
        final = TOML.parsefile(joinpath(directory, "session.toml"))
        @test final["status"] == "closed" && final["error"] == ""
        @test final["cleanup_confirmed"] === true && final["replay_running"] === false
        @test final["remaining_subscriptions"] == 0 && final["file_dialog_closed"] === true
        @test final["pid"] == ready["pid"] && final["input_id"] == ready["input_id"]
        @test final["recipe_id"] == ready["recipe_id"]
        @test final["nthreads_default"] == ready["nthreads_default"]
        @test final["nthreads_interactive"] == ready["nthreads_interactive"]
        @test final["figure_generations"] >= 1
        if plot === :glfw
            owned = final["glfw_after_disposal"]
            @test owned["screen_active"] === false && owned["background_rendering"] === false
            @test owned["current_screens"] == owned["baseline_screens"]
            @test owned["generations"] == owned["releases"] >= 1
        end
        return Dict("ready"=>ready, "final"=>final, "capture"=>capture,
            "exit_code"=>Int(session.process.exitcode), "term_signal"=>Int(session.process.termsignal))
    finally
        isopen(session) && close(session; timeout=120)
    end
end

function experimental_entry_main()
    root = mktempdir(joinpath(@__DIR__, "artifacts"); prefix="experimental-entry-", cleanup=false)
    println("EXPERIMENTAL_ENTRY_EVIDENCE=", root)
    println("GUI_PATH=", pathof(HammerheadGUI))
    println("CORE_PATH=", pathof(Hammerhead))
    source_maps() = Dict("prototype"=>LifecycleEvidence.source_files(@__DIR__),
        "core"=>LifecycleEvidence.source_files(joinpath(pkgdir(Hammerhead), "src")),
        "gui"=>LifecycleEvidence.source_files(joinpath(pkgdir(HammerheadGUI), "src")))
    before = source_maps()
    foreign = joinpath(root, "foreign cwd 实验 µ #%")
    mkdir(foreign)
    fixture = experiment_fixture(joinpath(foreign, "saved inputs µ #%"))
    original_bytes = read(fixture.path)
    input_bytes = [read(f["path"]) for f in fixture.record.input_files]
    project = joinpath(@__DIR__, "Project.toml")
    evidence = Dict{String,Any}("source_sha256"=>before, "cases"=>Dict{String,Any}())
    @testset "Hidden public experimental Qt entry" begin
        evidence["cases"]["saved_preview"] = entry_case(foreign, project, fixture.path;
            experiment=true, plot=:preview)
        @test read(fixture.path) == original_bytes
        @test [read(f["path"]) for f in fixture.record.input_files] == input_bytes
        @test !ispath(fixture.output) && !ispath(fixture.history)
        @test source_maps() == before
        raw_a, raw_b = (load_image(Float64, fixture.record.input_files[k]["path"])
                       for k in fixture.record.pairs[1])
        raw = run_piv(raw_a, raw_b, PIVParameters(window_size=16, overlap=8,
            uod_enable=false); threaded=false)
        result_path = joinpath(foreign, "native vectors µ #%.jld2")
        save_results(result_path, [raw, raw])
        result_bytes = read(result_path)
        evidence["cases"]["native_glfw"] = entry_case(foreign, project, result_path; plot=:glfw)
        @test read(result_path) == result_bytes && read(fixture.path) == original_bytes
        @test [read(f["path"]) for f in fixture.record.input_files] == input_bytes
        @test !ispath(fixture.output) && !ispath(fixture.history)
        @test source_maps() == before
    end
    open(io -> TOML.print(io, evidence; sorted=true), joinpath(root, "summary.toml"), "w")
    root
end

abspath(PROGRAM_FILE) == (@__FILE__) && experimental_entry_main()
