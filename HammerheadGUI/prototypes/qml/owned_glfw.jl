# Prototype-only screen owner. Qt and GLFW are pumped serially by the caller.
module OwnedGLFW
using GLMakie
using Base.Threads: threadid

mutable struct WindowOwner
    screen::Any
    owner_thread::Int
    visible::Bool
    generations::Int
    releases::Int
    frames::Int
    baseline_screens::Int
    last_tick::UInt64
    max_gap_seconds::Float64
end
WindowOwner(; visible=false) = WindowOwner(nothing, threadid(), visible, 0, 0, 0,
    length(GLMakie.ALL_SCREENS), 0, 0.)

function check_owner(owner)
    threadid() == owner.owner_thread || error("GLFW screen must be serviced by its owning Julia thread")
    nothing
end

function open!(owner::WindowOwner, figure)
    check_owner(owner)
    owner.screen === nothing || error("previous GLFW screen has not been released")
    isempty(figure.scene.current_screens) || error("figure is already attached to another screen")
    screen = GLMakie.Screen(; visible=owner.visible, start_renderloop=false,
        px_per_unit=1, title="Hammerhead scientific visualization (owned GLFW)")
    try
        # Screen(scene) uses Makie's singleton and can steal an unrelated view.
        # Allocate a dedicated screen, then attach this scene explicitly.
        GLMakie.display_scene!(screen, figure.scene)
    catch
        GLMakie.destroy!(screen)
        rethrow()
    end
    owner.screen = screen
    owner.generations += 1
    owner.last_tick = 0
    GLMakie.renderloop_running(screen) && error("owned GLFW screen unexpectedly started a render task")
    screen
end

# Called only after Qt's process_events has returned, never from a QML callback.
# This mirrors GLMakie's documented manual frame loop; no async renderer exists.
function pump!(owner::WindowOwner)
    check_owner(owner)
    screen = owner.screen
    screen === nothing && return :absent
    GLMakie.renderloop_running(screen) && error("background GLFW rendering is forbidden in this mode")
    GLMakie.isopen(screen) || return :closed
    now = time_ns()
    owner.last_tick == 0 || (owner.max_gap_seconds = max(owner.max_gap_seconds, (now-owner.last_tick)/1e9))
    owner.last_tick = now
    GLMakie.pollevents(screen, GLMakie.Makie.RegularRenderTick)
    GLMakie.isopen(screen) || return :closed
    GLMakie.poll_updates(screen)
    GLMakie.render_frame(screen)
    GLMakie.GLFW.SwapBuffers(GLMakie.to_native(screen))
    owner.frames += 1
    :rendered
end

function capture(owner::WindowOwner)
    check_owner(owner)
    owner.screen === nothing && error("no scientific window to capture")
    GLMakie.renderloop_running(owner.screen) && error("background GLFW rendering is forbidden in this mode")
    copy(GLMakie.colorbuffer(owner.screen))
end

function release!(owner::WindowOwner)
    check_owner(owner)
    screen = owner.screen
    screen === nothing && return nothing
    # destroy! switches the implemented GLFW context before disposing resources.
    # Keep ownership on failure so cleanup cannot falsely claim release.
    GLMakie.destroy!(screen)
    owner.screen = nothing
    owner.releases += 1
    owner.last_tick = 0
    nothing
end

function data(owner::WindowOwner)
    (; owner_thread=owner.owner_thread, visible=owner.visible,
       generations=owner.generations, releases=owner.releases, frames=owner.frames,
       screen_active=owner.screen !== nothing,
       background_rendering=owner.screen !== nothing && GLMakie.renderloop_running(owner.screen),
       baseline_screens=owner.baseline_screens, current_screens=length(GLMakie.ALL_SCREENS),
       max_pump_gap_seconds=owner.max_gap_seconds)
end
end
