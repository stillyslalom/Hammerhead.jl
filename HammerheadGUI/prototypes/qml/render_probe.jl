ENV["QT_QPA_PLATFORM"] = "offscreen"
ENV["QSG_RENDER_LOOP"] = "basic"
ENV["QSG_RHI_BACKEND"] = "opengl"
using QML, QMLMakie, GLMakie, TOML
const capture_succeeded = Ref(false)
record_capture(success) = (capture_succeeded[] = Bool(success); nothing)
@qmlfunction record_capture
fig = Figure(size = (700, 450))
ax = Axis(fig[1, 1]; title = "Native QMLMakie framebuffer probe")
lines!(ax, 0:0.01:10, sin.(0:0.01:10); color = :red, linewidth = 4)
mkpath(joinpath(@__DIR__, "artifacts"))
path = joinpath(@__DIR__, "artifacts", "bridge_framebuffer.png")
started = time_ns()
loadqml(joinpath(@__DIR__, "render_probe.qml"); plot = fig, capturePath = path)
QML.exec()
report = Dict("qt_platform" => ENV["QT_QPA_PLATFORM"], "capture_succeeded" => capture_succeeded[],
              "elapsed_seconds" => (time_ns() - started) / 1e9)
open(joinpath(@__DIR__, "artifacts", "render_probe.toml"), "w") do io
    TOML.print(io, report)
end
println(report)
@assert capture_succeeded[] "Qt failed to capture the offscreen framebuffer"
