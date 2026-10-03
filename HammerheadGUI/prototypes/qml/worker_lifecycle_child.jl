# This child owns Qt/GLFW; the replay worker owns core processing only.
include("lifecycle_evidence.jl")
using .LifecycleEvidence
const shell_evidence=String(split(only(filter(a->startswith(a,"--evidence="),ARGS)),'=';limit=2)[2])
const shell_journal=LifecycleEvidence.Journal(joinpath(shell_evidence,"stages"))
LifecycleEvidence.stage!(shell_journal,"child_boot")
LifecycleEvidence.write_fresh(joinpath(shell_evidence,"provenance.toml"),LifecycleEvidence.provenance(@__DIR__))
push!(ARGS,"--software","--plot=glfw","--worker-smoke")
try
    include("run.jl")
catch exception
    LifecycleEvidence.stage!(shell_journal,"worker_shell_failed";error=sprint(showerror,exception,catch_backtrace()))
    rethrow()
end
const final_packages=LifecycleEvidence.provenance(@__DIR__)["packages"]
LifecycleEvidence.write_fresh(joinpath(shell_evidence,"package_sources_after.toml"),final_packages)
const initial_packages=LifecycleEvidence.TOML.parsefile(joinpath(shell_evidence,"provenance.toml"))["packages"]
for name in ("Hammerhead","HammerheadGUI")
    initial_packages[name]["source_sha256"]==final_packages[name]["source_sha256"] ||
        error("loaded package source changed during worker evidence: $name")
end
const shell_report=LifecycleEvidence.TOML.parsefile(joinpath(@__DIR__,"artifacts","glfw_shell.toml"))
isempty(shell_report["qt_font_family"]) && error("Qt controls font was not acknowledged")
for (source,destination) in (("glfw_shell.png","framebuffer.png"),("glfw_scientific.png","scientific.png"))
    cp(joinpath(@__DIR__,"artifacts",source),joinpath(shell_evidence,destination))
end
for filename in ("worker_sidebar-active.png","worker_sidebar-small.png","worker_sidebar-large.png","worker_sidebar.toml")
    cp(joinpath(@__DIR__,"artifacts",filename),joinpath(shell_evidence,filename))
end
LifecycleEvidence.write_fresh(joinpath(shell_evidence,"shell_report.toml"),shell_report)
const worker_report=LifecycleEvidence.TOML.parsefile(joinpath(@__DIR__,"artifacts","worker_report.toml"))
worker_report["captures_sha256"]=Dict(filename=>LifecycleEvidence.digest(joinpath(shell_evidence,filename))
    for filename in ("framebuffer.png","scientific.png","worker_sidebar-active.png","worker_sidebar-small.png","worker_sidebar-large.png"))
LifecycleEvidence.write_fresh(joinpath(shell_evidence,"worker_report.toml"),worker_report)
LifecycleEvidence.stage!(shell_journal,"shell_subscriptions_disposed";
    disposed=shell_subscription_count,remaining=length(state.subscriptions)+length(state.experiment.subscriptions),
    replay_running=Prototype.busy(state))
LifecycleEvidence.stage!(shell_journal,"glfw_shell_completed";
    figure_generations=shell_report["figure_generations"],application_releases=application_releases[],
    native_release_verified=shell_report["native_context_release_verified"])
LifecycleEvidence.stage!(shell_journal,"child_complete")
