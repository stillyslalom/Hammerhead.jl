using Test, TOML
include("file_dialog_runner.jl")

# A protocol fixture tests evidence refusal, not rendering or native dialogs.
function dialog_fixture(directory)
    mkpath(joinpath(directory, "stages"))
    sources = Dict("fixture.jl" => "fixture-source-digest")
    packages = Dict(name => Dict("source_sha256" => sources) for name in ("Hammerhead", "HammerheadGUI"))
    provenance = Dict("pid"=>123, "packages"=>packages,
        "qt_environment"=>Dict("QT_QPA_PLATFORM"=>"offscreen", "QT_QUICK_BACKEND"=>"software", "QSG_RENDER_LOOP"=>"basic"))
    save(name, value) = open(io -> TOML.print(io, value), joinpath(directory, name), "w")
    save("provenance.toml", provenance)
    save("source_before.toml", sources); save("source_after.toml", sources)
    save("package_sources_after.toml", packages)
    names = ["child_boot"; [command == "shutdown-history" ? "dialog_shutdown" : "dialog_" * replace(command, '-'=>'_') for command in DIALOG_EXPECTED_STEPS];
        "dialog_dialog_capture"; "shell_subscriptions_disposed"; "dialog_shell_completed"; "child_complete"]
    for (i, name) in enumerate(names)
        stage = Dict{String,Any}("sequence"=>i, "pid"=>123, "julia_thread"=>1, "stage"=>name, "controller_preserved"=>true)
        name == "dialog_dialog_capture" && (stage["dialog_visible"] = true)
        if name == "shell_subscriptions_disposed"
            merge!(stage, Dict("remaining"=>0, "replay_running"=>false, "picker_active"=>false))
        end
        save(joinpath("stages", lpad(i, 4, '0') * ".toml"), stage)
    end
    captures = Dict{String,String}()
    for name in ("dialog.png", "small.png", "large.png")
        # Signature only: this fixture deliberately makes no visual claim.
        write(joinpath(directory, name), [UInt8[0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a]; zeros(UInt8, 1024)])
        captures[name] = LifecycleEvidence.digest(joinpath(directory, name))
    end
    original = joinpath(directory, "original.jld2"); write(original, "sentinel bytes")
    report = Dict{String,Any}("dialogs_forced_non_native"=>true, "processing_started"=>false,
        "cleanup_confirmed"=>true, "shutdown_preserved"=>true, "retained_frame"=>2, "qt_font_family"=>"fixture",
        "choices"=>[Dict("purpose"=>purpose, "accepted"=>true, "controller_preserved"=>true, "path"=>original) for purpose in ("record","result","output","history")],
        "captures_sha256"=>captures, "input_sha256"=>Dict(original=>LifecycleEvidence.digest(original)),
        "fresh_paths"=>[joinpath(directory, "unwritten.jld2")])
    for command in ("reject-result", "escape-result", "modal-left-output", "modal-right-output", "modal-close-output", "modal-run-output", "stale-result", "busy-output", "opening-failure-record")
        report[command] = true
    end
    for (name, width, height) in (("small",900,600),("large",1100,800))
        report["capture-"*name] = Dict("browse_controls_fit"=>true, "width"=>width, "height"=>height)
    end
    save("dialog_report.toml", report)
    save, report
end

@testset "Actual-dialog evidence contract" begin
    mktempdir() do directory
        save, report = dialog_fixture(directory)
        @test isempty(validate_dialog_evidence(directory))
        for key in ("cleanup_confirmed", "stale-result", "escape-result", "modal-left-output", "modal-close-output", "opening-failure-record")
            report[key] = false; save("dialog_report.toml", report)
            @test !isempty(validate_dialog_evidence(directory))
            report[key] = true
        end
        report["processing_started"] = true; save("dialog_report.toml", report)
        @test !isempty(validate_dialog_evidence(directory))
        report["processing_started"] = false
        report["choices"][4]["purpose"] = "output"; save("dialog_report.toml", report)
        @test !isempty(validate_dialog_evidence(directory))
        report["choices"][4]["purpose"] = "history"
        report["capture-small"]["browse_controls_fit"] = false; save("dialog_report.toml", report)
        @test !isempty(validate_dialog_evidence(directory))
        report["capture-small"]["browse_controls_fit"] = true; save("dialog_report.toml", report)
        write(only(keys(report["input_sha256"])), "changed bytes")
        @test !isempty(validate_dialog_evidence(directory))
        write(only(keys(report["input_sha256"])), "sentinel bytes")
        write(only(report["fresh_paths"]), "unintended SaveFile write")
        @test !isempty(validate_dialog_evidence(directory))
        rm(only(report["fresh_paths"]))
        save("source_after.toml", Dict("fixture.jl"=>"changed source"))
        @test !isempty(validate_dialog_evidence(directory))
        save("source_after.toml", Dict("fixture.jl"=>"fixture-source-digest"))
        stagefile = joinpath(directory,"stages","0002.toml")
        stage = TOML.parsefile(stagefile); stage["pid"] = 124
        save(joinpath("stages","0002.toml"),stage)
        @test !isempty(validate_dialog_evidence(directory))
        stage["pid"] = 123; save(joinpath("stages","0002.toml"),stage)
        write(joinpath(directory,"dialog.png"), "not a PNG")
        @test !isempty(validate_dialog_evidence(directory))
    end
end
