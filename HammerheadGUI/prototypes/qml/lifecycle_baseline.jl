# Minimal upstream-style embedding, without the local instrumented callback.
ENV["QT_QPA_PLATFORM"] = "offscreen"
ENV["QSG_RENDER_LOOP"] = "basic"
ENV["QSG_RHI_BACKEND"] = "opengl"
ENV["QSG_INFO"] = "1"
include("lifecycle_evidence.jl")
using .LifecycleEvidence
const evidence = String(split(only(filter(arg -> startswith(arg, "--evidence="), ARGS)), '='; limit = 2)[2])
const journal = LifecycleEvidence.Journal(joinpath(evidence, "stages"))
LifecycleEvidence.stage!(journal, "child_boot")
const native_provenance = LifecycleEvidence.provenance(@__DIR__)
native_provenance["imported_packages"] = ["QML", "QMLMakie", "GLMakie"]
native_provenance["source_drift_scope"] = "prototype pre/post; declared Hammerhead/GUI are not imported"
LifecycleEvidence.write_fresh(joinpath(evidence, "provenance.toml"), native_provenance)
using QML, QMLMakie, GLMakie
LifecycleEvidence.stage!(journal, "imports_complete")
const figure = Figure(size = (650, 430))
const axis = Axis(figure[1, 1]; title = "Upstream-style offscreen Qt baseline")
lines!(axis, 0:0.04:10, sin.(0:0.04:10); color = :red, linewidth = 4)
const captured = Ref(false)
function baseline_capture(success)
    captured[] = Bool(success)
    LifecycleEvidence.stage!(journal, "framebuffer_captured"; success = captured[],
        scene_screens = length(figure.scene.current_screens), all_screens = length(GLMakie.ALL_SCREENS))
    nothing
end
@qmlfunction baseline_capture
const qmlengine = loadqml(joinpath(@__DIR__, "lifecycle_baseline.qml"); plot = figure,
    capturePath = joinpath(evidence, "framebuffer.png"))
LifecycleEvidence.stage!(journal, "qml_load_complete")
QML.app_exec()
captured[] || error("baseline capture failed")
LifecycleEvidence.stage!(journal, "qt_cleanup_begin")
QML.quit(qmlengine)
QML.quit()
QML.cleanup()
QML.process_events()
LifecycleEvidence.stage!(journal, "qt_cleanup_complete")
LifecycleEvidence.stage!(journal, "child_complete")
# The normal GLMakie atexit cleanup is intentionally retained and evaluated by
# the parent process's complete logs and OS status, rather than a capture flag.
