module ShellActions
using Observables
struct ActionQueue
    actions::Vector{Function}
    pending::Observable{Bool}
    stopping::Function
end
ActionQueue(stopping=()->false)=ActionQueue(Function[],Observable(false),stopping)
function enqueue!(queue,action::Function)
    queue.stopping() && return false
    push!(queue.actions,action)
    queue.pending[]=true
    true
end
function drain!(queue)
    try
        while !queue.stopping() && !isempty(queue.actions)
            popfirst!(queue.actions)()
        end
    finally
        empty!(queue.actions)
        try queue.pending[]=false catch end
    end
    nothing
end
end
