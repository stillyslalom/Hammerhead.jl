include("lifecycle_evidence.jl")
using .LifecycleEvidence
const shell_evidence = String(split(only(filter(a -> startswith(a, "--evidence="), ARGS)), '='; limit=2)[2])
const shell_journal = LifecycleEvidence.Journal(joinpath(shell_evidence, "stages"))
LifecycleEvidence.stage!(shell_journal, "child_boot")
LifecycleEvidence.write_fresh(joinpath(shell_evidence, "provenance.toml"), LifecycleEvidence.provenance(@__DIR__))
push!(ARGS, "--software", "--plot=preview", "--file-dialog-smoke")
try
    include("run.jl")
catch error
    cleanup = all(name -> isdefined(Main, name), (:state, :viewports, :file_dialog_state)) &&
        isempty(state.subscriptions) && isempty(state.experiment.subscriptions) && !Prototype.busy(state) &&
        state.experiment.job === nothing && viewports.active === nothing && !file_dialog_state.active && file_dialog_state.closed
    LifecycleEvidence.write_fresh(joinpath(shell_evidence, "dialog_failure.toml"),
        Dict("error"=>sprint(showerror, error, catch_backtrace()), "cleanup_confirmed"=>cleanup))
    LifecycleEvidence.stage!(shell_journal, "dialog_child_failed"; error=sprint(showerror, error, catch_backtrace()), cleanup_confirmed=cleanup)
    rethrow()
end
if !isempty(smoke_error[]) || !lifecycle_done[]
    failure = isempty(smoke_error[]) ? "dialog lifecycle did not complete" : smoke_error[]
    cleanup = isempty(state.subscriptions) && isempty(state.experiment.subscriptions) &&
        !Prototype.busy(state) && state.experiment.job === nothing && viewports.active === nothing &&
        !file_dialog_state.active && file_dialog_state.closed
    LifecycleEvidence.write_fresh(joinpath(shell_evidence, "dialog_failure.toml"),
        Dict("error"=>failure, "cleanup_confirmed"=>cleanup))
    LifecycleEvidence.stage!(shell_journal, "dialog_smoke_failed"; error=failure, cleanup_confirmed=cleanup)
    error(failure)
end
const final_packages = LifecycleEvidence.provenance(@__DIR__)["packages"]
const initial_packages = TOML.parsefile(joinpath(shell_evidence, "provenance.toml"))["packages"]
for name in ("Hammerhead", "HammerheadGUI")
    initial_packages[name]["source_sha256"] == final_packages[name]["source_sha256"] || error("executed package changed during dialog evidence")
end
LifecycleEvidence.write_fresh(joinpath(shell_evidence, "package_sources_after.toml"), final_packages)
for suffix in ("dialog", "small", "large")
    cp(joinpath(@__DIR__, "artifacts", "file_dialog-" * suffix * ".png"), joinpath(shell_evidence, suffix * ".png"))
end
dialog_report["captures_sha256"] = Dict(suffix * ".png" => LifecycleEvidence.digest(joinpath(shell_evidence, suffix * ".png")) for suffix in ("dialog", "small", "large"))
dialog_report["cleanup_confirmed"] = isempty(state.subscriptions) && isempty(state.experiment.subscriptions) &&
    !Prototype.busy(state) && state.experiment.job === nothing && !file_dialog_state.active && file_dialog_state.closed && viewports.active === nothing
dialog_report["qt_font_family"] = qt_font_family[]
LifecycleEvidence.write_fresh(joinpath(shell_evidence, "dialog_report.toml"), dialog_report)
LifecycleEvidence.write_fresh(joinpath(shell_evidence, "shell_report.toml"), report)
LifecycleEvidence.stage!(shell_journal, "shell_subscriptions_disposed"; disposed=shell_subscription_count,
    remaining=length(state.subscriptions)+length(state.experiment.subscriptions), replay_running=Prototype.busy(state), picker_active=file_dialog_state.active)
dialog_report["cleanup_confirmed"] || error("dialog/model/view ownership was not released")
LifecycleEvidence.stage!(shell_journal, "dialog_shell_completed"; processing_started=false, cleanup_confirmed=true)
LifecycleEvidence.stage!(shell_journal, "child_complete")
