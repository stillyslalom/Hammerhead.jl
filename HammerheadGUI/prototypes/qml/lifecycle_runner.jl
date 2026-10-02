# Offscreen child owner. No Qt import is needed in the parent harness.
using TOML, Test
include("lifecycle_evidence.jl")
using .LifecycleEvidence

const NATIVE_ERROR_PATTERNS = ("exception in render", "ContextNotAvailable", "EXCEPTION_ACCESS_VIOLATION",
    "Segmentation fault", "GLMakie can not display a scene in multiple Screens",
    "TypeError:", "ReferenceError:", "ERROR:", "LIFECYCLE_RENDER_ERROR",
    "QThreadStorage:", "QMutex: destroying locked mutex", "Failed to create RHI",
    "does not support createPlatformOpenGLContext")

function nonempty_png(path)
    isfile(path) && filesize(path) >= 1024 &&
        open(io -> read(io, 8) == UInt8[0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a], path, "r")
end

function child_environment(source = ENV; backend = "rhi")
    overrides = Dict("QT_QPA_PLATFORM" => "offscreen", "QSG_RENDER_LOOP" => "basic",
        "QSG_RHI_BACKEND" => "opengl", "QT_QUICK_CONTROLS_STYLE" => "Basic", "QSG_INFO" => "1",
        "QT_QUICK_BACKEND" => backend)
    result = Dict{String,String}()
    for key in sort!(String.(collect(keys(source))))
        uppercase(key) in LifecycleEvidence.QT_ENV_KEYS && continue
        canonical_key = Sys.iswindows() ? uppercase(key) : key
        get!(result, canonical_key, String(source[key]))
    end
    merge!(result, overrides)
    result, overrides
end

function validate_lifecycle_evidence(directory, expected)
    errors = String[]
    try
        stages = [TOML.parsefile(path) for path in sort(readdir(joinpath(directory, "stages"); join = true))]
        [stage["sequence"] for stage in stages] == collect(1:length(stages)) || push!(errors, "stage sequence is incomplete")
        provenance = TOML.parsefile(joinpath(directory, "provenance.toml"))
        provenance["qt_environment"]["QT_QPA_PLATFORM"] == "offscreen" || push!(errors, "platform is not offscreen")
        provenance["qt_environment"]["QSG_RENDER_LOOP"] == "basic" || push!(errors, "render loop is not basic")
        all(stage["pid"] == provenance["pid"] for stage in stages) || push!(errors, "stage PID differs from provenance")
        count_expected = expected["cycles"]
        select(name) = filter(stage -> stage["stage"] == name, stages)
        if expected["scenario"] == "shell"
            completed = select("software_shell_completed")
            if length(completed) != 1 || only(completed)["figure_generations"] < 4 ||
               only(completed)["application_releases"] < 4 || only(completed)["native_release_verified"] !== false
                push!(errors, "software shell did not exercise acknowledged fresh viewport ownership")
            end
            nonempty_png(joinpath(directory, "framebuffer.png")) || push!(errors, "software shell PNG missing or invalid")
            length(select("child_complete")) == 1 || push!(errors, "software child did not complete")
            return errors
        end
        if expected["scenario"] == "baseline"
            captures = select("framebuffer_captured")
            length(captures) == 1 && only(captures)["success"] === true && only(captures)["scene_screens"] == 1 ||
                push!(errors, "baseline did not capture an attached native GL screen")
            nonempty_png(joinpath(directory, "framebuffer.png")) || push!(errors, "baseline PNG missing or invalid")
            length(select("qt_cleanup_complete")) == 1 || push!(errors, "Qt cleanup did not complete")
            length(select("child_complete")) == 1 || push!(errors, "child completion missing or duplicated")
            return errors
        end
        for name in ("viewport_created", "release_requested", "release_acknowledged", "viewport_detached")
            [stage["generation"] for stage in select(name)] == collect(1:count_expected) ||
                push!(errors, "$name does not cover each generation exactly once")
        end
        if expected["scenario"] != "construction"
            [stage["generation"] for stage in select("first_frame")] == collect(1:count_expected) ||
                push!(errors, "first_frame does not cover each generation")
            length(select("framebuffer_captured")) == 1 || push!(errors, "capture stage missing or duplicated")
            png = joinpath(directory, "framebuffer.png")
            if !nonempty_png(png)
                push!(errors, "capture is not a nonempty PNG")
            end
            all(stage["scene_screens"] > 0 for stage in select("first_frame")) || push!(errors, "native first frame has no attached screen")
        end
        for stage in select("release_requested")
            stage["observer_count"] == 0 || push!(errors, "viewport observer remained subscribed")
        end
        retention = TOML.parsefile(joinpath(directory, "retention.toml"))
        retention["observer_count"] == 0 || push!(errors, "observers remained after cleanup")
        retention["figures_still_reachable"] == 0 || push!(errors, "viewport figures remained reachable after cleanup")
        if expected["release_mode"] == "context"
            length(select("native_release_complete")) == count_expected || push!(errors, "native release did not finish for each generation")
            all(stage["native_release"] === true for stage in select("release_acknowledged")) || push!(errors, "release acknowledgement was not native")
            retention["final_screens"] == retention["baseline_screens"] || push!(errors, "screen registry did not return to baseline")
        end
        if expected["scenario"] == "resize"
            length(select("resize_requested")) == 4 || push!(errors, "resize requests incomplete")
            length(select("fbo_changed")) >= 2 || push!(errors, "resize did not exercise FBO replacement")
        end
        if expected["scenario"] == "failure"
            length(select("processing_failure_handled")) == 1 || push!(errors, "processing failure was not exercised")
        end
        length(select("qt_cleanup_complete")) == 1 || push!(errors, "Qt cleanup did not complete")
        length(select("child_complete")) == 1 || push!(errors, "child completion missing or duplicated")
    catch exception
        push!(errors, sprint(showerror, exception))
    end
    errors
end

function run_lifecycle_child(command::Cmd, directory; timeout = 180.0, expected = nothing)
    timeout > 0 || throw(ArgumentError("timeout must be positive"))
    mkdir(directory)
    stdout_path, stderr_path = joinpath(directory, "stdout.log"), joinpath(directory, "stderr.log")
    started = time_ns()
    process = nothing
    timed_out = false
    owner_error = ""
    sources_before = LifecycleEvidence.source_files(@__DIR__)
    LifecycleEvidence.write_fresh(joinpath(directory, "source_before.toml"), sources_before)
    # Cmd's Windows hide flag prevents a console window; Qt remains offscreen.
    hidden = Sys.iswindows() ? Cmd(command; windows_hide = true) : command
    software_child = expected !== nothing && expected["scenario"] == "shell"
    process_environment, qt_environment = child_environment(; backend = software_child ? "software" : "rhi")
    # QMLSCENE_DEVICE is a legacy adaptation selector; no inherited variant may
    # override this explicit RHI/OpenGL native trial. Do not print other env data.
    command = setenv(hidden, process_environment)
    LifecycleEvidence.write_fresh(joinpath(directory, "invocation.toml"), Dict(
        "command" => collect(command.exec), "timeout_seconds" => timeout,
        "parent_pid" => getpid(), "environment" => qt_environment,
        "inherited_qt_variants" => LifecycleEvidence.qt_variants(),
        "unset_environment" => ["QMLSCENE_DEVICE"]))
    open(stdout_path, "w") do out
        open(stderr_path, "w") do err
            try
                process = run(pipeline(command; stdout = out, stderr = err); wait = false)
                deadline = time() + timeout
                while !process_exited(process)
                    if time() >= deadline
                        timed_out = true
                        kill(process)
                        break
                    end
                    sleep(0.05)
                end
                wait(process) # wait=false processes do not throw on nonzero exit.
            catch exception
                owner_error = sprint(showerror, exception, catch_backtrace())
            finally
                if process !== nothing && !process_exited(process)
                    kill(process)
                    wait(process)
                end
            end
        end
    end
    stdout_text, stderr_text = read(stdout_path, String), read(stderr_path, String)
    sources_after = LifecycleEvidence.source_files(@__DIR__)
    LifecycleEvidence.write_fresh(joinpath(directory, "source_after.toml"), sources_after)
    combined = stdout_text * "\n" * stderr_text
    errors = [pattern for pattern in NATIVE_ERROR_PATTERNS if occursin(pattern, combined)]
    stages_dir = joinpath(directory, "stages")
    stages = String[]
    stage_errors = String[]
    if isdir(stages_dir)
        for path in sort(readdir(stages_dir; join = true))
            try
                push!(stages, TOML.parsefile(path)["stage"])
            catch exception
                push!(stage_errors, sprint(showerror, exception))
            end
        end
    end
    child_complete = "child_complete" in stages
    evidence_errors = expected === nothing ? String[] : validate_lifecycle_evidence(directory, expected)
    sources_before == sources_after || push!(evidence_errors, "prototype source changed during the child process")
    report = Dict{String,Any}("exit_code" => process === nothing ? -1 : Int(process.exitcode),
        "timed_out" => timed_out, "owner_error" => owner_error,
        "elapsed_seconds" => (time_ns() - started) / 1e9,
        "stdout_sha256" => LifecycleEvidence.digest(stdout_path),
        "stderr_sha256" => LifecycleEvidence.digest(stderr_path),
        "error_patterns" => errors, "stages" => stages, "stage_parse_errors" => stage_errors,
        "child_complete" => child_complete, "evidence_errors" => evidence_errors)
    report["passed"] = report["exit_code"] == 0 && !timed_out && isempty(owner_error) &&
        isempty(errors) && isempty(stage_errors) && isempty(evidence_errors) && child_complete
    LifecycleEvidence.write_fresh(joinpath(directory, "process_result.toml"), report)
    report
end

function harness_tests(root)
    @testset "Hidden process evidence and timeout ownership" begin
        environment, overrides = child_environment(Dict("qt_quick_backend" => "software",
            "QT_QUICK_BACKEND" => "software", "qmlscene_DEVICE" => "software",
            "qsg_rhi_backend" => "vulkan", "QT_QPA_PLATFORM" => "windows", "PATH" => "kept"))
        @test environment["QT_QUICK_BACKEND"] == "rhi"
        @test environment["QSG_RHI_BACKEND"] == "opengl"
        @test environment["QT_QPA_PLATFORM"] == "offscreen"
        @test !any(key -> uppercase(key) == "QMLSCENE_DEVICE", keys(environment))
        @test count(key -> uppercase(key) == "QT_QUICK_BACKEND", keys(environment)) == 1
        @test environment["PATH"] == "kept"
        @test !haskey(overrides, "PATH")
        bad_png = joinpath(root, "invalid.png")
        write(bad_png, "not a framebuffer")
        @test !nonempty_png(bad_png)
        @test !nonempty_png(joinpath(root, "missing.png"))
        bad = `$(Base.julia_cmd()) --startup-file=no -e 'println("child stdout"); println(stderr,"child stderr"); exit(7)'`
        report = run_lifecycle_child(bad, joinpath(root, "exit7"); timeout = 30.)
        @test report["exit_code"] == 7
        @test !report["passed"]
        @test !report["timed_out"]
        @test occursin("child stdout", read(joinpath(root, "exit7", "stdout.log"), String))
        @test occursin("child stderr", read(joinpath(root, "exit7", "stderr.log"), String))
        hang = `$(Base.julia_cmd()) --startup-file=no -e 'println("child alive"); flush(stdout); sleep(60)'`
        report = run_lifecycle_child(hang, joinpath(root, "timeout"); timeout = 3.)
        @test report["timed_out"]
        @test !report["passed"]
        @test isempty(report["owner_error"])
        @test occursin("child alive", read(joinpath(root, "timeout", "stdout.log"), String))
    end
end

function main(args = ARGS)
    value(key, default) = begin
        matches = filter(arg -> startswith(arg, "--$key="), args)
        length(matches) <= 1 || throw(ArgumentError("duplicate option $key"))
        isempty(matches) ? default : split(only(matches), '='; limit = 2)[2]
    end
    artifacts = joinpath(@__DIR__, "artifacts")
    mkpath(artifacts)
    root = mktempdir(artifacts; prefix = "lifecycle-", cleanup = false)
    println("Evidence directory: ", root)
    "--self-test" in args && return harness_tests(root)
    cases = String.(split(value("cases", "construction,single-observe,single-context,reopen-context,separate-context,resize-context,failure-context"), ','))
    if any(name -> first(split(name, '-')) in ("reopen", "separate", "resize", "failure", "reuse"), cases) && !("single-context" in cases)
        pushfirst!(cases, "single-context")
    end
    cycles = parse(Int, value("cycles", "3"))
    timeout = parse(Float64, value("timeout", "180"))
    cycles in 1:100 || throw(ArgumentError("cycles must be 1 to 100"))
    results = Dict{String,Any}()
    minimal_native_passed = false
    for name in cases
        occursin(r"^(shell-software|baseline|construction|(single|reopen|separate|resize|failure|reuse)-(observe|context))$", name) ||
            throw(ArgumentError("unknown lifecycle case: $name"))
        scenario = first(split(name, '-'))
        if scenario in ("reopen", "separate", "resize", "failure", "reuse") && !minimal_native_passed
            results[name] = Dict("passed" => false, "skipped" => true,
                "reason" => "clean single-context render/release/exit gate has not passed")
            println(name, ": skipped; minimal native lifecycle gate has not passed")
            continue
        end
        directory = joinpath(root, name)
        script = joinpath(@__DIR__, name == "baseline" ? "lifecycle_baseline.jl" :
            name == "shell-software" ? "lifecycle_shell.jl" : "lifecycle_probe.jl")
        command = `$(Base.julia_cmd()) --startup-file=no --threads=1 --project=$(@__DIR__) $script --case=$name --cycles=$cycles --evidence=$directory`
        expected = Dict("scenario" => scenario, "release_mode" => last(split(name, '-')),
            "cycles" => scenario in ("reopen", "separate", "reuse") ? cycles : 1)
        report = run_lifecycle_child(command, directory; timeout, expected)
        results[name] = report
        name == "single-context" && (minimal_native_passed = report["passed"])
        println(name, ": passed=", report["passed"], " exit=", report["exit_code"],
            " timeout=", report["timed_out"], " final stages=", last(report["stages"], min(4, length(report["stages"]))))
    end
    LifecycleEvidence.write_fresh(joinpath(root, "summary.toml"), Dict("cases" => results,
        "all_passed" => all(report["passed"] for report in values(results)),
        "scope" => "Windows offscreen basic loop; no power-loss, desktop or cross-platform guarantee"))
    root
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    result = main()
    if !("--self-test" in ARGS)
        TOML.parsefile(joinpath(result, "summary.toml"))["all_passed"] || exit(1)
    end
end
