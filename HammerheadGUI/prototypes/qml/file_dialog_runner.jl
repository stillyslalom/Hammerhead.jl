# Reuse the established hidden process owner; apply this lane's exact contract.
include("lifecycle_runner.jl")
const DIALOG_EXPECTED_STEPS = ["open-record", "select-record", "accept-record", "open-result", "select-result", "accept-result",
    "open-output", "select-output", "accept-output", "open-history", "select-history", "accept-history",
    "reject-result", "escape-result", "modal-left-output", "modal-right-output", "modal-close-output", "modal-run-output", "stale-result", "busy-output", "opening-failure-record",
    "capture-small", "capture-large", "shutdown-history"]

function validate_dialog_evidence(directory)
    errors = String[]
    require(value, message) = value || push!(errors, message)
    try
        provenance = TOML.parsefile(joinpath(directory, "provenance.toml"))
        stages = [TOML.parsefile(path) for path in sort(readdir(joinpath(directory, "stages"); join=true))]
        report = TOML.parsefile(joinpath(directory, "dialog_report.toml"))
        require([s["sequence"] for s in stages] == collect(1:length(stages)), "incomplete stage sequence")
        require(all(s -> s["pid"] === provenance["pid"] && s["julia_thread"] === 1, stages), "stage owner identity changed")
        environment = provenance["qt_environment"]
        require(environment["QT_QPA_PLATFORM"] == "offscreen" && environment["QT_QUICK_BACKEND"] == "software" && environment["QSG_RENDER_LOOP"] == "basic", "dialog child environment was not hidden software/basic")
        select(name) = filter(s -> s["stage"] == name, stages)
        require(length(select("child_boot")) == 1 && length(select("child_complete")) == 1, "missing/duplicate lifetime endpoints")
        for command in DIALOG_EXPECTED_STEPS
            name = command == "shutdown-history" ? "dialog_shutdown" : "dialog_" * replace(command, '-' => '_')
            matched = select(name)
            require(length(matched) == 1, "missing/duplicate scenario: $command")
            if length(matched) == 1 && !startswith(command, "capture-")
                require(get(only(matched), "controller_preserved", nothing) === true, "state changed at $command")
            end
        end
        actual = select("dialog_dialog_capture")
        require(length(actual) == 1 && only(actual)["dialog_visible"] === true && report["dialogs_forced_non_native"] === true, "actually open non-native FileDialog was not captured")
        require(report["processing_started"] === false && report["cleanup_confirmed"] === true && report["shutdown_preserved"] === true, "processing or unconfirmed shutdown")
        require(report["retained_frame"] === 2, "nontrivial retained-frame fixture absent")
        require(!isempty(report["qt_font_family"]), "Qt controls font unacknowledged")
        choices = report["choices"]
        require(length(choices) == 4 && Set(c["purpose"] for c in choices) == Set(["record", "result", "output", "history"]), "accepted purpose population incomplete")
        require(all(c -> c["accepted"] === true && c["controller_preserved"] === true && c["path"] isa String && !isempty(c["path"]), choices), "invalid accepted choice ledger")
        for command in ("reject-result", "escape-result", "modal-left-output", "modal-right-output", "modal-close-output", "modal-run-output", "stale-result", "busy-output", "opening-failure-record")
            require(get(report, command, nothing) === true, "missing rejection/key-routing proof: $command")
        end
        for size in ("small", "large")
            probe = report["capture-" * size]
            require(probe["browse_controls_fit"] === true && probe["width"] == (size == "small" ? 900 : 1100) && probe["height"] == (size == "small" ? 600 : 800), "invalid compact layout")
        end
        for filename in ("dialog.png", "small.png", "large.png")
            require(nonempty_png(joinpath(directory, filename)), "invalid actual capture: $filename")
            require(get(report["captures_sha256"], filename, nothing) == LifecycleEvidence.digest(joinpath(directory, filename)), "changed capture: $filename")
        end
        for (path, digest) in report["input_sha256"]
            require(isfile(path) && LifecycleEvidence.digest(path) == digest, "chooser changed original input/destination bytes")
        end
        require(all(path -> !ispath(path) && !islink(path), report["fresh_paths"]), "SaveFile choice created a destination")
        disposal = select("shell_subscriptions_disposed")
        require(length(disposal) == 1 && only(disposal)["remaining"] === 0 && only(disposal)["replay_running"] === false && only(disposal)["picker_active"] === false, "remaining model/picker/replay lifetime")
        require(length(select("dialog_shell_completed")) == 1, "missing clean dialog-shell completion")
        before = TOML.parsefile(joinpath(directory, "source_before.toml"))
        require(before == TOML.parsefile(joinpath(directory, "source_after.toml")), "prototype source drift")
        final_packages = TOML.parsefile(joinpath(directory, "package_sources_after.toml"))
        for name in ("Hammerhead", "HammerheadGUI")
            require(provenance["packages"][name]["source_sha256"] == final_packages[name]["source_sha256"], "executed package source drift")
        end
    catch error
        push!(errors, sprint(showerror, error))
    end
    errors
end

function file_dialog_main(args=ARGS)
    options = filter(a -> startswith(a, "--timeout="), args)
    length(options) <= 1 || error("duplicate timeout")
    timeout = isempty(options) ? 240.0 : parse(Float64, split(only(options), '='; limit=2)[2])
    isfinite(timeout) && timeout > 0 || error("timeout must be finite and positive")
    artifacts = joinpath(@__DIR__, "artifacts"); mkpath(artifacts)
    root = mktempdir(artifacts; prefix="file-dialog-", cleanup=false)
    println("Evidence directory: ", root); flush(stdout)
    directory = joinpath(root, "actual-dialog")
    command = `$(Base.julia_cmd()) --startup-file=no --threads=1 --project=$(@__DIR__) $(joinpath(@__DIR__, "file_dialog_child.jl")) --evidence=$directory`
    result = run_lifecycle_child(command, directory; timeout, backend="software")
    result["dialog_evidence_errors"] = validate_dialog_evidence(directory)
    result["passed"] = result["passed"] && isempty(result["dialog_evidence_errors"])
    # No worker is requested by this lane; a failed child still cannot certify
    # that an unintended replay/descendant was cleaned up.
    result["cleanup_confirmed"] = result["passed"]
    if !result["passed"]
        LifecycleEvidence.write_fresh(joinpath(root, "incomplete_owner.toml"), Dict("subsequent_launches_allowed"=>false, "cleanup_confirmed"=>false))
    end
    LifecycleEvidence.write_fresh(joinpath(root, "summary.toml"), Dict("case"=>result, "all_passed"=>result["passed"],
        "scope"=>"actual offscreen Quick FileDialog and synthetic key-routing; no native OS dialog or replay processing", "subsequent_launches_allowed"=>result["passed"]))
    println("actual-dialog: passed=", result["passed"], " exit=", result["exit_code"], " timeout=", result["timed_out"])
    result["passed"] || exit(1)
    root
end
if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    file_dialog_main()
end
