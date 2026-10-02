using Test, Observables
include("viewport_lifecycle.jl")
using .ViewportLifecycle

@testset "Per-viewport application ownership (no native GL claims)" begin
    registry = ViewportLifecycle.Registry()
    source = Observable(0)
    hits = Ref(0)
    current_refresh = Ref{Function}(() -> nothing)
    function factory()
        payload = [1, 2, 3]
        refresh = () -> (hits[] += 1; nothing)
        subscription = on(_ -> refresh(), source)
        current_refresh[] = refresh
        detach = () -> (current_refresh[] === refresh && (current_refresh[] = () -> nothing); nothing)
        payload, refresh, [subscription], detach
    end
    first_lease = ViewportLifecycle.create_viewport!(registry, factory)
    first_payload = first_lease.payload
    @test first_lease.generation == 1
    source[] = 1
    @test hits[] == 1
    @test_throws ArgumentError ViewportLifecycle.create_viewport!(registry, factory)
    @test_throws ArgumentError ViewportLifecycle.detach_viewport!(registry, 1)
    @test_throws ArgumentError ViewportLifecycle.acknowledge_release!(registry, 1)
    @test ViewportLifecycle.request_release!(registry) == 1
    source[] = 2
    current_refresh[]()
    @test hits[] == 1
    @test isempty(first_lease.subscriptions)
    @test_throws ArgumentError ViewportLifecycle.acknowledge_release!(registry, 2)
    @test_throws ArgumentError ViewportLifecycle.detach_viewport!(registry, 1)
    ViewportLifecycle.acknowledge_release!(registry, 1)
    ViewportLifecycle.detach_viewport!(registry, 1)
    @test first_lease.payload === nothing
    second_lease = ViewportLifecycle.create_viewport!(registry, factory)
    @test second_lease.payload !== first_payload
    @test second_lease.generation == 2
    @test_throws ArgumentError ViewportLifecycle.detach_viewport!(registry, 1)
    source[] = 3
    @test hits[] == 2
    ViewportLifecycle.request_release!(registry)
    ViewportLifecycle.acknowledge_release!(registry, 2)
    ViewportLifecycle.detach_viewport!(registry, 2)
    @test_throws ErrorException ViewportLifecycle.create_viewport!(registry, () -> error("factory failure"))
    @test registry.active === nothing
    @test registry.generation == 2
    function retained_lease()
        local_registry = ViewportLifecycle.Registry()
        lease = ViewportLifecycle.create_viewport!(local_registry, () -> begin
            payload = zeros(128)
            refresh = () -> sum(payload)
            # This detach closure retains the refresh callback until disposed.
            payload, refresh, Any[], () -> (refresh(); nothing)
        end)
        reference = WeakRef(lease.payload)
        ViewportLifecycle.request_release!(local_registry)
        ViewportLifecycle.acknowledge_release!(local_registry, lease.generation)
        ViewportLifecycle.detach_viewport!(local_registry, lease.generation)
        lease, reference
    end
    retained, reference = retained_lease()
    GC.gc()
    @test reference.value === nothing
    @test retained.phase == :released
    for _ in 1:20
        lease = ViewportLifecycle.create_viewport!(registry, factory)
        @test length(lease.subscriptions) == 1
        ViewportLifecycle.request_release!(registry)
        ViewportLifecycle.acknowledge_release!(registry, lease.generation)
        ViewportLifecycle.detach_viewport!(registry, lease.generation)
        @test isempty(source.listeners)
    end
end
