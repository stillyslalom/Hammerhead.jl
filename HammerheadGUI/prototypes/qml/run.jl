# Defaults to an offscreen, finite smoke run. --desktop is an explicit opt-in.
const desktop = "--desktop" in ARGS
const software = "--software" in ARGS
if !desktop
    ENV["QT_QPA_PLATFORM"] = "offscreen"
end
ENV["QSG_RENDER_LOOP"] = "basic"
ENV["QSG_RHI_BACKEND"] = "opengl"
ENV["QT_QUICK_CONTROLS_STYLE"] = "Basic"
software && (ENV["QT_QUICK_BACKEND"] = "software")
const started = time_ns()
using QML, QMLMakie, GLMakie, Observables, TOML, Pkg
include("adapter.jl")
include("viewport.jl")
const import_seconds = (time_ns() - started) / 1e9
const state = Prototype.State()
const fig, ax = viewport(state)
const lifecycle_done = Ref(false)
const captured = Ref(false)
const capture_path = joinpath(@__DIR__, "artifacts", software ? "software_shell.png" : "native_shell.png")
const preview_path = joinpath(@__DIR__, "artifacts", "scientific_preview.png")
const preview_url = Observable("")
const preview_generation = Ref(0)
const invalid_schedule_seen = Ref(false)
const open_error_seen = Ref(false)
const axis_rectangle = Ref((0., 0., 1., 1.))
mkpath(joinpath(@__DIR__, "artifacts"))

function refresh_preview()
    save(preview_path, fig; visible = false)
    rect = ax.scene.viewport[]
    width, height = size(fig.scene)
    axis_rectangle[] = (rect.origin[1] / width, 1 - (rect.origin[2] + rect.widths[2]) / height,
                        rect.widths[1] / width, rect.widths[2] / height)
    preview_generation[] += 1
    preview_url[] = "file:///" * replace(preview_path, '\\' => '/') * "?v=$(preview_generation[])"
    nothing
end
const refresh_plot = state.refresh
state.refresh = () -> begin
    refresh_plot()
    software && refresh_preview()
end
software && refresh_preview()

function change_schedule(text)
    success = Prototype.schedule(state, text)
    success || (invalid_schedule_seen[] = true)
    success
end
function open_results(path)
    success = Prototype.open_results(state, path)
    success || (open_error_seen[] = true)
    success
end
navigate_frame(i) = Prototype.navigate(state, i)
run_batch() = Prototype.run_batch(state)
cancel_batch() = Prototype.cancel_batch(state)
close_mask() = Prototype.close_mask(state)
set_mask_mode(drawing) = (state.drawing = Bool(drawing); nothing)
function pick_preview(x, y, drawing)
    left, top, width, height = axis_rectangle[]
    0 <= (x - left) / width <= 1 && 0 <= (y - top) / height <= 1 || return nothing
    limits = ax.finallimits[]
    px = limits.origin[1] + (x - left) / width * limits.widths[1]
    py = limits.origin[2] + (y - top) / height * limits.widths[2]
    Prototype.pick(state, px, py; drawing = Bool(drawing))
end
viewport_created() = (state.visualizations += 1; nothing)
lifecycle_complete() = (lifecycle_done[] = true; nothing)
record_capture(success) = (captured[] = Bool(success); nothing)
shutdown_prototype() = (state.shutdown = true; nothing)
@qmlfunction change_schedule open_results navigate_frame run_batch cancel_batch close_mask set_mask_mode pick_preview viewport_created lifecycle_complete record_capture shutdown_prototype

props = JuliaPropertyMap("scheduleError" => state.schedule_error,
                         "openError" => state.open_error, "status" => state.status,
                         "running" => state.batch.running, "frame" => state.frame,
                         "count" => state.count, "selection" => state.selection,
                         "preview" => preview_url)
const load_started = time_ns()
const qmlengine = loadqml(joinpath(@__DIR__, "main.qml"); model = props, plot = fig,
                         bridgeEnabled = !software, offscreenDisplay = true,
                         smokeMode = !desktop, capturePath = capture_path)
const load_seconds = (time_ns() - load_started) / 1e9
deadline = time() + 180
while !state.shutdown && (desktop || !(lifecycle_done[] && captured[] && !state.batch.running[]))
    !desktop && time() > deadline && error("QML shell smoke timed out")
    # exec_async starts a Julia REPL; scripts explicitly pump Qt between yields.
    QML.process_eventloop_updates()
    QML.process_events()
    state.ticks += 1
    sleep(0.01)
end
versions = Dict(info.name => string(info.version) for info in values(Pkg.dependencies())
                if info.name in ("QML", "QMLMakie", "Qt6Base_jll", "Qt6Declarative_jll",
                                 "jlqml_jll", "CxxWrap", "Makie", "GLMakie", "HammerheadGUI"))
report = Dict("julia" => string(VERSION), "os" => string(Sys.KERNEL),
              "versions" => versions, "qt_platform" => get(ENV, "QT_QPA_PLATFORM", "default"),
              "software_fallback" => software, "import_seconds" => import_seconds,
              "qml_load_seconds" => load_seconds, "total_seconds" => (time_ns() - started) / 1e9,
              "qt_event_ticks" => state.ticks, "visualization_creations" => state.visualizations,
              "capture_succeeded" => captured[], "lifecycle_complete" => lifecycle_done[],
              "status" => state.status[], "retained_state_bytes" => Base.summarysize(state),
              "maxrss_bytes" => Sys.maxrss(), "viewport_bytes" => Base.summarysize(fig),
              "invalid_schedule_seen" => invalid_schedule_seen[], "open_error_seen" => open_error_seen[])
open(joinpath(@__DIR__, "artifacts", software ? "software_shell.toml" : "native_shell.toml"), "w") do io
    TOML.print(io, report)
end
println(report)
if !desktop
    @assert captured[] && lifecycle_done[]
    @assert state.visualizations >= 4
    @assert isempty(state.schedule_error[]) && invalid_schedule_seen[] && open_error_seen[]
end
# Close GL resources while Qt still owns its contexts. Native bridge teardown is
# itself part of the feasibility probe; errors are deliberately not suppressed.
GLMakie.closeall()
QML.quit(qmlengine)
QML.quit()
QML.cleanup()
QML.process_events()
