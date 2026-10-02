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
using QML, QMLMakie, GLMakie, Observables, TOML, Pkg, SHA
include("adapter.jl")
include("viewport.jl")
include("viewport_lifecycle.jl")
const import_seconds = (time_ns() - started) / 1e9
const state = Prototype.State()
const viewports = ViewportLifecycle.Registry()
const application_releases = Ref(0)
const disposed_subscriptions = Ref(0)
const lifecycle_done = Ref(false)
const captured = Ref(false)
const smoke_error = Ref("")
const qt_font_family = Ref("")
const font_option = filter(arg -> startswith(arg, "--font="), ARGS)
length(font_option) <= 1 || error("duplicate --font option")
const font_path = isempty(font_option) ? GLMakie.Makie.assetpath("fonts", "TeXGyreHerosMakie-Regular.otf") :
    abspath(String(split(only(font_option), '='; limit = 2)[2]))
isfile(font_path) || error("Qt prototype font file does not exist: $font_path")
const font_sha256 = open(io -> bytes2hex(sha256(io)), font_path, "r")
const capture_path = joinpath(@__DIR__, "artifacts", software ? "software_shell.png" : "native_shell.png")
const preview_path = joinpath(@__DIR__, "artifacts", "scientific_preview.png")
const preview_url = Observable("")
const preview_generation = Ref(0)
const invalid_schedule_seen = Ref(false)
const open_error_seen = Ref(false)
const axis_rectangle = Ref((0., 0., 1., 1.))
mkpath(joinpath(@__DIR__, "artifacts"))

function refresh_preview()
    fig, ax = viewports.active.payload
    save(preview_path, fig; visible = false)
    rect = ax.scene.viewport[]
    width, height = size(fig.scene)
    axis_rectangle[] = (rect.origin[1] / width, 1 - (rect.origin[2] + rect.widths[2]) / height,
                        rect.widths[1] / width, rect.widths[2] / height)
    preview_generation[] += 1
    preview_url[] = "file:///" * replace(preview_path, '\\' => '/') * "?v=$(preview_generation[])"
    nothing
end
function prepare_viewport()
    lease = ViewportLifecycle.create_viewport!(viewports, () -> begin
        fig, ax, refresh, subscriptions, detach = viewport(state; managed = true)
        (fig, ax), refresh, subscriptions, () -> begin
            detach()
            state.refresh = () -> nothing
            nothing
        end
    end)
    if software
        refresh = lease.refresh
        state.refresh = () -> begin
            refresh()
            refresh_preview()
        end
        refresh_preview()
    end
    lease.generation
end
viewport_scene() = viewports.active.payload[1]
function release_viewport()
    lease = viewports.active
    lease === nothing && return nothing
    disposed_subscriptions[] += length(lease.subscriptions)
    ViewportLifecycle.request_release!(viewports)
    if software
        # These are the hidden GLFW screens created by save(preview), whose
        # context switch is implemented. This does not release Qt native screens.
        for screen in copy(lease.payload[1].scene.current_screens)
            GLMakie.destroy!(screen)
        end
    end
    ViewportLifecycle.acknowledge_release!(viewports, lease.generation)
    application_releases[] += 1
    nothing
end
function detach_viewport()
    lease = viewports.active
    lease === nothing && return nothing
    fig = lease.payload[1]
    GLMakie.Makie.current_figure() === fig && GLMakie.Makie.current_figure!(nothing)
    ViewportLifecycle.detach_viewport!(viewports, lease.generation)
    nothing
end
prepare_viewport()

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
    viewports.active === nothing && return nothing
    _, ax = viewports.active.payload
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
function smoke_failed(message)
    smoke_error[] = String(message)
    state.shutdown = true
    open(joinpath(@__DIR__, "artifacts", "software_smoke_failure.toml"), "w") do io
        TOML.print(io, Dict("stage" => "smoke_timeout", "error" => smoke_error[],
            "figure_generations" => viewports.generation,
            "application_releases" => application_releases[]))
    end
    nothing
end
record_font_family(family) = (qt_font_family[] = String(family); nothing)
@qmlfunction change_schedule open_results navigate_frame run_batch cancel_batch close_mask set_mask_mode pick_preview viewport_created lifecycle_complete record_capture shutdown_prototype prepare_viewport viewport_scene release_viewport detach_viewport smoke_failed record_font_family

props = JuliaPropertyMap("scheduleError" => state.schedule_error,
                         "openError" => state.open_error, "status" => state.status,
                         "running" => state.batch.running, "frame" => state.frame,
                         "count" => state.count, "selection" => state.selection,
                         "preview" => preview_url)
const load_started = time_ns()
const qmlengine = loadqml(joinpath(@__DIR__, "main.qml"); model = props,
                         bridgeEnabled = !software, offscreenDisplay = true,
                         smokeMode = !desktop, capturePath = capture_path,
                         fontSource = "file:///" * replace(font_path, '\\' => '/'))
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
              "maxrss_bytes" => Sys.maxrss(), "viewport_bytes" => viewports.active === nothing ? 0 : Base.summarysize(viewports.active.payload[1]),
              "figure_generations" => viewports.generation,
              "application_release_acknowledgements" => application_releases[],
              "disposed_viewport_subscriptions" => disposed_subscriptions[],
              "native_context_release_verified" => false,
              "qt_font_path" => font_path, "qt_font_sha256" => font_sha256,
              "qt_font_family" => qt_font_family[],
              "invalid_schedule_seen" => invalid_schedule_seen[], "open_error_seen" => open_error_seen[])
open(joinpath(@__DIR__, "artifacts", software ? "software_shell.toml" : "native_shell.toml"), "w") do io
    TOML.print(io, report)
end
println(report)
if !desktop
    isempty(smoke_error[]) || error(smoke_error[])
    isempty(qt_font_family[]) && error("Qt controls never acknowledged their loaded font family")
    @assert captured[] && lifecycle_done[]
    @assert state.visualizations >= 4
    @assert isempty(state.schedule_error[]) && invalid_schedule_seen[] && open_error_seen[]
end
# Dispose application callbacks first. Acknowledgement certifies application
# ownership only; Qt ownership does not prove that a GL context is current.
# Native GL/bridge failures are deliberately not suppressed or called a pass.
release_viewport()
detach_viewport()
GLMakie.closeall()
QML.quit(qmlengine)
QML.quit()
QML.cleanup()
QML.process_events()
