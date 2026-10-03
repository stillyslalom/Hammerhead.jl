using Test,Observables
include("shell_actions.jl")
@testset "Queued shell actions and cleanup" begin
    shutdown=Ref(false)
    queue=ShellActions.ActionQueue(()->shutdown[])
    events=Int[]
    @test ShellActions.enqueue!(queue,()->push!(events,1))
    @test ShellActions.enqueue!(queue,()->push!(events,2))
    @test isempty(events) && queue.pending[]
    ShellActions.drain!(queue)
    @test events==[1,2] && isempty(queue.actions) && !queue.pending[]
    primary=ErrorException("queued worker failed")
    @test ShellActions.enqueue!(queue,()->throw(primary))
    @test ShellActions.enqueue!(queue,()->push!(events,3))
    secondary=ErrorException("pending observer failed")
    listener=on(queue.pending) do pending
        pending || throw(secondary)
    end
    actual=try ShellActions.drain!(queue);nothing catch err;err end
    off(listener)
    @test actual===primary && !queue.pending[] && isempty(queue.actions)
    @test events==[1,2]
    @test ShellActions.enqueue!(queue,()->push!(events,4))
    shutdown[]=true
    ShellActions.drain!(queue)
    @test events==[1,2] && !queue.pending[] && isempty(queue.actions)
    @test !ShellActions.enqueue!(queue,()->push!(events,5))
end
