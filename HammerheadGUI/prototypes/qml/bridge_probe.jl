# Native bridge construction only: the window remains invisible. Construction
# does not imply that Qt has rendered a framebuffer or forwarded input events.
ENV["QT_QPA_PLATFORM"] = "offscreen"
ENV["QSG_RENDER_LOOP"] = "basic"
ENV["QSG_RHI_BACKEND"] = "opengl"
const probe_started = time_ns()
using QML
using QMLMakie
using GLMakie
using TOML
const probe_import_seconds = (time_ns() - probe_started) / 1e9
const probe_events = String[]
record_probe(event) = (push!(probe_events, String(event)); nothing)
@qmlfunction record_probe
fig = Figure(size = (640, 420))
Axis(fig[1, 1])
lines!(0:10, sin.(0:10))
probe_build_started = time_ns()
loadqml(joinpath(@__DIR__, "bridge_probe.qml"); plot = fig)
probe_build_seconds = (time_ns() - probe_build_started) / 1e9
QML.exec()
mkpath(joinpath(@__DIR__, "artifacts"))
report = Dict("julia" => string(VERSION), "os" => string(Sys.KERNEL),
              "import_seconds" => probe_import_seconds,
              "qml_load_seconds" => probe_build_seconds,
              "events" => probe_events, "window_visible" => false,
              "embedded_framebuffer_rendered" => false)
open(joinpath(@__DIR__, "artifacts", "bridge_probe.toml"), "w") do io
    TOML.print(io, report)
end
@assert "makie_component_created" in probe_events
@assert "hidden_event_loop_tick" in probe_events
println(report)
