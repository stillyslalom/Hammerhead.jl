# Only lifecycle_runner.jl should launch this child. Never shows a desktop window.
ENV["QT_QPA_PLATFORM"] = "offscreen"
ENV["QSG_RENDER_LOOP"] = "basic"
ENV["QSG_RHI_BACKEND"] = "opengl"
ENV["QT_QUICK_CONTROLS_STYLE"] = "Basic"
include("lifecycle_evidence.jl")
include("viewport_lifecycle.jl")
using .LifecycleEvidence, .ViewportLifecycle
function option(key, fallback = "")
    values = filter(arg -> startswith(arg, "--$key="), ARGS)
    length(values) <= 1 || error("duplicate option $key")
    isempty(values) ? fallback : split(only(values), '='; limit = 2)[2]
end
const evidence = option("evidence")
isdir(evidence) || error("parent must create a fresh evidence directory")
const journal = LifecycleEvidence.Journal(joinpath(evidence, "stages"))
const case_name = String(option("case", "single-context"))
const scenario = String(first(split(case_name, '-')))
const release_mode = scenario == "construction" ? "observe" : String(last(split(case_name, '-')))
const cycle_count = parse(Int, option("cycles", "3"))
LifecycleEvidence.stage!(journal, "child_boot"; scenario, release_mode, cycle_count)
const native_provenance = LifecycleEvidence.provenance(@__DIR__)
native_provenance["imported_packages"] = ["QML", "QMLMakie", "GLMakie", "Observables"]
native_provenance["source_drift_scope"] = "prototype pre/post; declared Hammerhead/GUI are not imported"
LifecycleEvidence.write_fresh(joinpath(evidence, "provenance.toml"), native_provenance)
LifecycleEvidence.stage!(journal, "imports_begin")
using QML, QMLMakie, GLMakie, Observables
LifecycleEvidence.stage!(journal, "imports_complete")

const registry = ViewportLifecycle.Registry()
const generation = Observable(0)
const plot = Observable{Any}(nothing)
const released = Observable(0)
const rendered = Observable(0)
const complete = Ref(false)
const callback_errors = String[]
const signals = Observable(0)
const observer_hits = Ref(0)
const weak_figures = WeakRef[]
const figures = Dict{UInt,Int}()
const screens = Dict{Int,Any}()
const contexts = Dict{Int,UInt}()
const frame_counts = Dict{Int,Int}()
const fbo_sizes = Dict{Int,Tuple{Int,Int}}()
const released_from_render = Ref(0)
const latest_rendered = Ref(0)
const reused_figure = Ref{Any}(nothing)
const baseline_screens = length(GLMakie.ALL_SCREENS)
const failure_seen = Ref(false)

# This Windows-specific check observes the native context currently bound on the
# callback thread. Association is captured on this screen's first successful
# render callback; it is not inferred from QMLMakie's always-true valid flag.
function current_context()
    Sys.iswindows() || error("context-release trial currently requires Windows wglGetCurrentContext")
    UInt(ccall((:wglGetCurrentContext, "opengl32.dll"), Ptr{Cvoid}, ()))
end

function create_viewport()
    lease = ViewportLifecycle.create_viewport!(registry, () -> begin
        fig = if scenario == "reuse" && reused_figure[] !== nothing
            reused_figure[]
        else
            f = Figure(size = (650, 430))
            ax = Axis(f[1, 1]; title = "Qt viewport generation $(registry.generation + 1)",
                xlabel = "x", ylabel = "sin(x)")
            lines!(ax, 0:0.04:10, sin.(0:0.04:10); color = :red, linewidth = 4)
            f
        end
        scenario == "reuse" && (reused_figure[] = fig)
        subscription = on(_ -> (observer_hits[] += 1), signals)
        fig, () -> nothing, [subscription], () -> nothing
    end)
    push!(weak_figures, WeakRef(lease.payload))
    figures[objectid(lease.payload)] = lease.generation
    generation[] = lease.generation
    plot[] = lease.payload
    LifecycleEvidence.stage!(journal, "viewport_created"; generation = lease.generation,
        figure_object = string(objectid(lease.payload)), subscriptions = length(lease.subscriptions),
        all_screens = length(GLMakie.ALL_SCREENS))
    lease.generation
end

function request_release()
    gen = ViewportLifecycle.request_release!(registry)
    before = observer_hits[]
    signals[] += 1
    observer_hits[] == before || error("disposed viewport observer still fired")
    LifecycleEvidence.stage!(journal, "release_requested"; generation = gen,
        observer_count = length(signals.listeners))
    if !haskey(screens, gen)
        # Construction-only invisible viewport has never acquired a GL screen.
        scenario == "construction" || error("viewport did not render before release")
        ViewportLifecycle.acknowledge_release!(registry, gen)
        released[] = gen
        LifecycleEvidence.stage!(journal, "release_acknowledged"; generation = gen, native_release = false)
    end
    nothing
end

function detach_viewport(gen)
    gen = Int(gen)
    registry.active.generation == gen || error("stale QML viewport detach")
    figure_id = objectid(registry.active.payload)
    GLMakie.Makie.current_figure() === registry.active.payload && GLMakie.Makie.current_figure!(nothing)
    scene_screens = length(GLMakie.Makie.get_scene(registry.active.payload).current_screens)
    ViewportLifecycle.detach_viewport!(registry, gen)
    plot[] = nothing
    delete!(figures, figure_id)
    delete!(screens, gen)
    GC.gc()
    LifecycleEvidence.stage!(journal, "viewport_detached"; generation = gen,
        scene_screens_at_detach = scene_screens, all_screens = length(GLMakie.ALL_SCREENS),
        observer_count = length(signals.listeners))
    nothing
end

function record_capture(success)
    Bool(success) || error("Qt framebuffer capture failed")
    LifecycleEvidence.stage!(journal, "framebuffer_captured"; generation = generation[])
    if get(frame_counts, generation[], 0) == 0
        message = "framebuffer capture contains no acknowledged native GL frame"
        push!(callback_errors, message)
        println(stderr, "LIFECYCLE_RENDER_ERROR\n", message)
        flush(stderr)
    end
    nothing
end
function processing_failure()
    try
        error("intentional processing failure")
    catch exception
        failure_seen[] = true
        LifecycleEvidence.stage!(journal, "processing_failure_handled"; error = sprint(showerror, exception))
    end
    nothing
end
function lifecycle_complete()
    registry.active === nothing || error("completion before viewport detached")
    complete[] = true
    LifecycleEvidence.stage!(journal, "scenario_complete"; failure_seen = failure_seen[])
    nothing
end
function qml_stage(name)
    LifecycleEvidence.stage!(journal, String(name); generation = generation[])
    nothing
end
function sync_lifecycle()
    released[] = max(released_from_render[], released[])
    rendered[] = max(latest_rendered[], rendered[])
    !isempty(callback_errors)
end
@qmlfunction create_viewport request_release detach_viewport record_capture processing_failure lifecycle_complete qml_stage sync_lifecycle

# Local, opt-in instrumented callback. It reproduces QMLMakie's display path but
# retains full exceptions. Cvoid callbacks must never throw into C++; a reported
# exception makes the child fail from its ordinary Julia event-loop scope.
function instrumented_render(screen, fig)
    try
        gen = get(figures, objectid(fig), 0)
        gen > 0 || error("render callback for an unowned figure")
        lease = registry.active
        lease !== nothing && lease.generation == gen || error("render callback for a detached viewport")
        lease.phase == :released && return nothing
        if !haskey(screens, gen)
            screens[gen] = screen
            context = Sys.iswindows() ? current_context() : UInt(0)
            contexts[gen] = context
            LifecycleEvidence.stage!(journal, "screen_observed"; generation = gen,
                screen_object = string(objectid(screen)), current_context = string(context),
                window_object = string(screen.glscreen.quickwin),
                fbo_size = collect(screen.glscreen.fbo_size))
        end
        fbo_size = screen.glscreen.fbo_size
        if haskey(fbo_sizes, gen) && fbo_sizes[gen] != fbo_size
            LifecycleEvidence.stage!(journal, "fbo_changed"; generation = gen,
                before = collect(fbo_sizes[gen]), after = collect(fbo_size))
        end
        fbo_sizes[gen] = fbo_size
        if lease.phase == :release_requested
            if release_mode == "context"
                context = current_context()
                context != 0 && context == contexts[gen] || error("native context changed or missing at release")
                LifecycleEvidence.stage!(journal, "native_release_begin"; generation = gen,
                    current_context = string(context))
                GLMakie.destroy!(screen)
                screen.glscreen.context.valid = false
                LifecycleEvidence.stage!(journal, "native_release_complete"; generation = gen,
                    all_screens = length(GLMakie.ALL_SCREENS),
                    scene_screens = length(GLMakie.Makie.get_scene(fig).current_screens))
            end
            ViewportLifecycle.acknowledge_release!(registry, gen)
            released_from_render[] = gen
            LifecycleEvidence.stage!(journal, "release_acknowledged"; generation = gen,
                native_release = release_mode == "context")
            return nothing
        end
        scene = GLMakie.Makie.get_scene(fig)
        if !GLMakie.Makie.is_displayed(screen, scene)
            GLMakie.Makie.update_state_before_display!(fig)
        end
        display(screen, scene)
        frame_counts[gen] = get(frame_counts, gen, 0) + 1
        latest_rendered[] = gen
        if frame_counts[gen] == 1
            LifecycleEvidence.stage!(journal, "first_frame"; generation = gen,
                scene_screens = length(scene.current_screens), all_screens = length(GLMakie.ALL_SCREENS))
        end
    catch exception
        message = sprint(showerror, exception, catch_backtrace())
        push!(callback_errors, message)
        println(stderr, "LIFECYCLE_RENDER_ERROR\n", message)
        flush(stderr)
        LifecycleEvidence.stage!(journal, "render_failed"; error = message)
    end
    nothing
end
const rooted_render_callback = QMLMakie.CxxWrap.@safe_cfunction(instrumented_render, Cvoid, (Any, Any))
QML.set_default_makie_renderfunction(rooted_render_callback)
create_viewport()
const props = JuliaPropertyMap("plot" => plot, "generation" => generation,
    "released" => released, "rendered" => rendered)
LifecycleEvidence.stage!(journal, "qml_load_begin")
const qmlengine = loadqml(joinpath(@__DIR__, "lifecycle_probe.qml"); model = props,
    initialPlot = plot[], scenario = scenario, cycles = cycle_count, capturePath = joinpath(evidence, "framebuffer.png"))
LifecycleEvidence.stage!(journal, "qml_load_complete")
# Use the real Qt loop for this small native reproducer. app_exec propagates
# exceptions; QML.exec's convenience wrapper catches and prints them instead.
QML.app_exec()
isempty(callback_errors) || error("native render failed; full trace recorded in stderr/stages")
complete[] || error("child exited before lifecycle completion")
LifecycleEvidence.stage!(journal, "qt_cleanup_begin")
QML.quit(qmlengine)
QML.quit()
QML.cleanup()
QML.process_events()
LifecycleEvidence.stage!(journal, "qt_cleanup_complete")
GC.gc()
LifecycleEvidence.write_fresh(joinpath(evidence, "retention.toml"), Dict(
    "baseline_screens" => baseline_screens, "final_screens" => length(GLMakie.ALL_SCREENS),
    "observer_count" => length(signals.listeners),
    "figures_still_reachable" => count(ref -> ref.value !== nothing, weak_figures),
    "created_figures" => length(weak_figures), "maxrss_bytes" => Sys.maxrss(),
    "release_mode" => release_mode, "processing_failure_seen" => failure_seen[],
    "context_check" => "Windows callback-thread wgl handle; not a cross-platform context contract"))
isempty(callback_errors) || error("render errors occurred")
if release_mode == "context"
    length(GLMakie.ALL_SCREENS) == baseline_screens || error("GL screen registry did not return to baseline")
end
LifecycleEvidence.stage!(journal, "child_complete")
