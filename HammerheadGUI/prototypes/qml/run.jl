# Defaults to an offscreen, finite smoke run. --desktop is an explicit opt-in.
const desktop = "--desktop" in ARGS
const software = "--software" in ARGS
const experiment_smoke = "--experiment-smoke" in ARGS
if !desktop
    ENV["QT_QPA_PLATFORM"] = "offscreen"
end
ENV["QSG_RENDER_LOOP"] = "basic"
ENV["QSG_RHI_BACKEND"] = "opengl"
ENV["QT_QUICK_CONTROLS_STYLE"] = "Basic"
software && (ENV["QT_QUICK_BACKEND"] = "software")
const started = time_ns()
using QML, QMLMakie, GLMakie, Observables, TOML, Pkg, SHA, Hammerhead, HammerheadGUI
software && GLMakie.activate!()
include("adapter.jl")
include("viewport.jl")
include("viewport_lifecycle.jl")
include("experiment_fixture.jl")
include("shell_actions.jl")
const import_seconds = (time_ns() - started) / 1e9
const state = Prototype.State()
const fixture = experiment_smoke ? experiment_fixture(mktempdir(joinpath(@__DIR__,"artifacts");prefix="experiment-data-",cleanup=false)) : nothing
const fixture_path=Observable(fixture===nothing ? "" : fixture.path)
const smoke_step=Ref(0)
const cancellation_seen=Ref(false)
const completed_replay_seen=Ref(false)
const retained_display_seen=Ref(false)
const viewport_releases_while_replaying=Ref(0)
const action_queue=ShellActions.ActionQueue(()->state.shutdown)
const pending_work=action_queue.pending
const transition_ack=Observable(0)
const transition_open=Observable(true)
const transition_separate=Observable(false)
const smoke_response=Ref(0)
function enqueue(action)
    ShellActions.enqueue!(action_queue,action)
end
function drain_actions()
    ShellActions.drain!(action_queue)
end
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
    GLMakie.Makie.update_state_before_display!(fig)
    screen=if isempty(fig.scene.current_screens)
        GLMakie.Screen(fig.scene;visible=false,start_renderloop=false)
    else
        # Makie's cached-screen apply_config! restarts the GLFW loop even for
        # save/colorbuffer(fig). Render the owned screen directly instead.
        only(fig.scene.current_screens)
    end
    GLMakie.stop_renderloop!(screen;close_after_renderloop=false)
    buffer=colorbuffer(screen)
    Hammerhead.FileIO.save(preview_path,buffer)
    for screen in fig.scene.current_screens
        GLMakie.renderloop_running(screen) && error("software preview unexpectedly started a background GLFW render loop")
    end
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
    state.experiment.controller.running[] && (viewport_releases_while_replaying[]+=1)
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
function open_results_now(path)
    success = Prototype.open_results(state, path)
    success || (open_error_seen[] = true)
    success
end
navigate_frame_now(i) = Prototype.navigate(state, i)
open_results(path)=begin captured=String(path);enqueue(()->open_results_now(captured)) end
navigate_frame(i)=begin captured=Int(i);enqueue(()->navigate_frame_now(captured)) end
run_batch() = Prototype.run_batch(state)
cancel_batch() = Prototype.cancel_batch(state)
close_mask() = enqueue(()->Prototype.close_mask(state))
set_mask_mode(drawing) = (state.drawing = Bool(drawing); nothing)
function pick_preview_now(x, y, drawing)
    viewports.active === nothing && return nothing
    _, ax = viewports.active.payload
    left, top, width, height = axis_rectangle[]
    0 <= (x - left) / width <= 1 && 0 <= (y - top) / height <= 1 || return nothing
    limits = ax.finallimits[]
    px = limits.origin[1] + (x - left) / width * limits.widths[1]
    py = limits.origin[2] + (y - top) / height * limits.widths[2]
    Prototype.pick(state, px, py; drawing = Bool(drawing))
end
pick_preview(x,y,drawing)=begin
    captured=(Float64(x),Float64(y),Bool(drawing))
    enqueue(()->pick_preview_now(captured...))
end
function queue_transition(open,separate)
    captured_open,captured_separate=Bool(open),Bool(separate)
    enqueue() do
        release_viewport();detach_viewport()
        captured_open && prepare_viewport()
        transition_open[]=captured_open;transition_separate[]=captured_separate
        transition_ack[]+=1
    end
end
viewport_created() = (state.visualizations += 1; nothing)
lifecycle_complete() = (lifecycle_done[] = true; nothing)
record_capture(success) = (captured[] = Bool(success); nothing)
shutdown_prototype() = Prototype.request_shutdown(state)
open_saved_now(path)=Prototype.open_saved_experiment(state,path)
open_saved(path)=begin captured=String(path);pending_work[] ? false : enqueue(()->open_saved_now(captured)) end
function replay_saved_now(output,history,allow)
    Prototype.configure_saved_experiment(state,output,history,allow) || return false
    Prototype.run_saved_experiment(state)
end
replay_saved(output,history,allow)=begin
    captured=(String(output),String(history),Bool(allow))
    pending_work[] ? false : enqueue(()->replay_saved_now(captured...))
end
cancel_saved()=Prototype.cancel_saved_experiment(state)
inspect_saved_now()=Prototype.inspect_saved_experiment(state)
inspect_saved()=pending_work[] ? false : enqueue(inspect_saved_now)
saved_page(delta)=Prototype.experiment_page(state,delta)
saved_section(history)=Prototype.experiment_section(state,history)
function advance_experiment_smoke()
    smoke_response[]==0 || return
    ec=state.experiment.controller
    step=smoke_step[]
    if step==0
        @assert open_saved_now(fixture.path)
        @assert Prototype.configure_saved_experiment(state,fixture.output,fixture.history,false)
        @assert Prototype.run_saved_experiment(state;progress=(i,n)->begin
            i==1 || return
            cancel_saved()
            while transition_ack[]<1
                yield() # smoke coordination only: wait for view release acknowledgement
            end
        end)
        smoke_step[]=1
        smoke_response[]=1
    elseif step==1 && !ec.running[]
        @assert ec.state[]===:cancelled && ec.progress[]==(1,3)
        @assert ec.last_run[].status===:failed
        cancellation_seen[]=true
        @assert Prototype.run_saved_experiment(state;progress=(i,n)->i==n && cancel_saved())
        smoke_step[]=2
        smoke_response[]=2
    elseif step==2 && !ec.running[]
        @assert ec.state[]===:completed && ec.progress[]==(3,3)
        completed_replay_seen[]=true
        @assert inspect_saved_now() state.experiment.error[]
        @assert navigate_frame_now(3)
        previous=state.explorer
        @assert !open_saved_now("missing-saved-experiment.jld2")
        @assert state.explorer===previous
        retained_display_seen[]=true
        @assert open_saved_now(fixture.history)
        @assert inspect_saved_now() state.experiment.error[]
        @assert navigate_frame_now(3)
        r=HammerheadGUI.current_result(state.explorer)
        Prototype.pick(state,r.x[1],r.y[1])
        @assert Prototype.configure_saved_experiment(state,fixture.output,fixture.history,false)
        for _ in 1:100
            occursin("ROI:",state.experiment.text[]) && break
            Prototype.experiment_page(state,1)
        end
        smoke_step[]=3
        smoke_response[]=3
    elseif step==3
        smoke_step[]=4
        smoke_response[]=4
    elseif step==4
        smoke_step[]=5
        smoke_response[]=5
    elseif step==5
        smoke_step[]=6
        smoke_response[]=6
    elseif step==6
        smoke_step[]=7
        smoke_response[]=7
    end
    nothing
end
experiment_smoke_step()=begin value=smoke_response[];smoke_response[]=0;value end
function smoke_failed(message)
    smoke_error[] = String(message)
    Prototype.request_shutdown(state)
    open(joinpath(@__DIR__, "artifacts", "software_smoke_failure.toml"), "w") do io
        TOML.print(io, Dict("stage" => "smoke_timeout", "error" => smoke_error[],
            "figure_generations" => viewports.generation,
            "application_releases" => application_releases[]))
    end
    nothing
end
record_font_family(family) = (qt_font_family[] = String(family); nothing)
@qmlfunction change_schedule open_results navigate_frame run_batch cancel_batch close_mask set_mask_mode pick_preview viewport_created lifecycle_complete record_capture shutdown_prototype prepare_viewport viewport_scene release_viewport detach_viewport smoke_failed record_font_family
@qmlfunction open_saved replay_saved cancel_saved inspect_saved saved_page saved_section experiment_smoke_step
@qmlfunction queue_transition

props = JuliaPropertyMap("scheduleError" => state.schedule_error,
                         "openError" => state.open_error, "status" => state.status,
                         "running" => state.batch.running, "frame" => state.frame,
                         "count" => state.count, "selection" => state.selection,
                         "preview" => preview_url,
                         "fixtureRecord"=>fixture_path,
                         "experimentRunning"=>lift((busy,pending)->busy||pending,state.experiment.controller.running,pending_work),
                         "experimentOutput"=>state.experiment.controller.output_path,
                         "experimentHistory"=>state.experiment.controller.run_record_path,
                         "experimentAllow"=>state.experiment.controller.allow_environment_change,
                         "experimentStatus"=>state.experiment.controller.status,
                         "experimentError"=>state.experiment.error,
                         "experimentWritten"=>state.experiment.written,
                         "experimentText"=>state.experiment.text,"experimentPages"=>state.experiment.pages,
                         "activeIdentity"=>state.experiment.identity,"displayedIdentity"=>state.displayed,
                         "demoDisplayed"=>lift(mode->mode===:demo,state.dataset),
                         "renderAvailable"=>state.render_available,
                         "transitionAck"=>transition_ack,"transitionOpen"=>transition_open,"transitionSeparate"=>transition_separate)
const load_started = time_ns()
const qmlengine = loadqml(joinpath(@__DIR__, "main.qml"); model = props,
                         bridgeEnabled = !software, offscreenDisplay = true,
                         smokeMode = !desktop,experimentSmoke=experiment_smoke,capturePath = capture_path,
                         fontSource = "file:///" * replace(font_path, '\\' => '/'))
const load_seconds = (time_ns() - load_started) / 1e9
deadline = time() + 180
while (!state.shutdown || Prototype.busy(state)) &&
      (desktop || !(lifecycle_done[] && captured[] && !Prototype.busy(state)))
    !desktop && time() > deadline && error("QML shell smoke timed out")
    # exec_async starts a Julia REPL; scripts explicitly pump Qt between yields.
    QML.process_eventloop_updates()
    QML.process_events()
    drain_actions()
    experiment_smoke && advance_experiment_smoke()
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
              "invalid_schedule_seen" => invalid_schedule_seen[], "open_error_seen" => open_error_seen[],
              "experiment_smoke"=>experiment_smoke,"saved_cancellation_seen"=>cancellation_seen[],
              "completed_replay_seen"=>completed_replay_seen[],
              "retained_display_seen"=>retained_display_seen[],
              "saved_state"=>String(state.experiment.controller.state[]),
              "saved_written"=>collect(state.experiment.controller.progress[]),
              "last_run_status"=>state.experiment.controller.last_run[]===nothing ? "unavailable" : String(state.experiment.controller.last_run[].status),
              "last_run_completed_pairs"=>state.experiment.controller.last_run[]===nothing ? 0 : state.experiment.controller.last_run[].completed_pairs,
              "viewport_releases_while_replaying"=>viewport_releases_while_replaying[],
              "displayed_identity"=>state.displayed[])
open(joinpath(@__DIR__, "artifacts", software ? "software_shell.toml" : "native_shell.toml"), "w") do io
    TOML.print(io, report)
end
println(report)
if !desktop
    isempty(smoke_error[]) || error(smoke_error[])
    isempty(qt_font_family[]) && error("Qt controls never acknowledged their loaded font family")
    @assert captured[] && lifecycle_done[]
    @assert state.visualizations >= (experiment_smoke ? 5 : 4)
    if experiment_smoke
        @assert cancellation_seen[] && retained_display_seen[]
    else
        @assert isempty(state.schedule_error[]) && invalid_schedule_seen[] && open_error_seen[]
    end
end
# Dispose application callbacks first. Acknowledgement certifies application
# ownership only; Qt ownership does not prove that a GL context is current.
# Native GL/bridge failures are deliberately not suppressed or called a pass.
release_viewport()
detach_viewport()
const shell_subscription_count=length(state.subscriptions)+length(state.experiment.subscriptions)
Prototype.dispose_state(state)
@assert isempty(state.subscriptions) && isempty(state.experiment.subscriptions)
GLMakie.closeall()
QML.quit(qmlengine)
QML.quit()
QML.cleanup()
QML.process_events()
