# Enforced startup environment comes from lifecycle_runner, not Julia ENV alone.
include("lifecycle_evidence.jl")
using .LifecycleEvidence
const shell_evidence = String(split(only(filter(arg -> startswith(arg, "--evidence="), ARGS)), '='; limit = 2)[2])
const shell_journal = LifecycleEvidence.Journal(joinpath(shell_evidence, "stages"))
LifecycleEvidence.stage!(shell_journal, "child_boot")
LifecycleEvidence.write_fresh(joinpath(shell_evidence, "provenance.toml"), LifecycleEvidence.provenance(@__DIR__))
push!(ARGS, "--software")
any(==("--case=shell-experiment-software"),ARGS) && push!(ARGS,"--experiment-smoke")
LifecycleEvidence.stage!(shell_journal, "software_smoke_begin")
try
    include("run.jl")
catch exception
    LifecycleEvidence.stage!(shell_journal, "software_shell_failed";
        error = sprint(showerror, exception, catch_backtrace()))
    rethrow()
end
const shell_report = LifecycleEvidence.TOML.parsefile(joinpath(@__DIR__, "artifacts", "software_shell.toml"))
const package_sources_after=LifecycleEvidence.provenance(@__DIR__)["packages"]
LifecycleEvidence.write_fresh(joinpath(shell_evidence,"package_sources_after.toml"),package_sources_after)
const packages_before=LifecycleEvidence.TOML.parsefile(joinpath(shell_evidence,"provenance.toml"))["packages"]
for name in ("Hammerhead","HammerheadGUI")
    packages_before[name]["source_sha256"]==package_sources_after[name]["source_sha256"] ||
        error("loaded package source changed during software evidence: $name")
end
isempty(shell_report["qt_font_family"]) && error("Qt controls font was not acknowledged")
cp(joinpath(@__DIR__, "artifacts", "software_shell.png"), joinpath(shell_evidence, "framebuffer.png"))
LifecycleEvidence.write_fresh(joinpath(shell_evidence, "shell_report.toml"), shell_report)
LifecycleEvidence.stage!(shell_journal,"shell_subscriptions_disposed";
    disposed=shell_subscription_count,remaining=length(state.subscriptions)+length(state.experiment.subscriptions),
    replay_running=Prototype.busy(state))
LifecycleEvidence.stage!(shell_journal, "software_shell_completed";
    figure_generations = shell_report["figure_generations"],
    application_releases = application_releases[],
    native_release_verified = shell_report["native_context_release_verified"])
LifecycleEvidence.stage!(shell_journal, "child_complete")
