# Application ownership only. This module makes no native GL cleanup claim.
module ViewportLifecycle
using Observables

mutable struct Lease
    generation::Int
    payload::Any
    refresh::Function
    subscriptions::Vector{Any}
    detach::Function
    phase::Symbol
end

mutable struct Registry
    generation::Int
    active::Union{Nothing,Lease}
end
Registry() = Registry(0, nothing)

function create_viewport!(registry::Registry, factory::Function)
    registry.active === nothing || throw(ArgumentError("previous viewport has not detached"))
    # A failing factory leaves the previous generation and ownership unchanged.
    payload, refresh, subscriptions, detach = factory()
    registry.generation += 1
    lease = Lease(registry.generation, payload, refresh, Any[subscriptions...], detach, :active)
    registry.active = lease
    lease
end

function request_release!(registry::Registry)
    lease = registry.active
    lease === nothing && throw(ArgumentError("no active viewport"))
    lease.phase == :active || throw(ArgumentError("viewport release already requested"))
    # Stop application callbacks before GPU resources can be removed. Failed
    # detach callbacks keep the phase unchanged so their original error survives.
    foreach(Observables.off, lease.subscriptions)
    empty!(lease.subscriptions)
    lease.detach()
    lease.detach = () -> nothing
    lease.refresh = () -> nothing
    lease.phase = :release_requested
    lease.generation
end

function acknowledge_release!(registry::Registry, generation::Integer)
    lease = registry.active
    lease === nothing && throw(ArgumentError("no viewport awaiting release"))
    lease.generation == generation || throw(ArgumentError("stale release acknowledgement"))
    lease.phase == :release_requested || throw(ArgumentError("release was not requested"))
    # The caller owns the meaning of this acknowledgement. It may certify only
    # application observer disposal, or separately verified native GL release.
    lease.phase = :released
    nothing
end

function detach_viewport!(registry::Registry, generation::Integer)
    lease = registry.active
    lease === nothing && throw(ArgumentError("no viewport to detach"))
    lease.generation == generation || throw(ArgumentError("stale viewport detach"))
    lease.phase == :released || throw(ArgumentError("viewport release has not been acknowledged"))
    lease.payload = nothing
    registry.active = nothing
    nothing
end
end
