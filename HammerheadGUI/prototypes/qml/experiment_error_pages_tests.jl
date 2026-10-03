using Test,Hammerhead,HammerheadGUI,Observables
include("adapter.jl")
include("experiment_fixture.jl")

# Read what the UI exposes, rather than an internal assembled text buffer.
function inspection_page_text(state)
    saved_page=state.experiment.page[]
    Prototype.experiment_page(state,-10000)
    pages=String[]
    while true
        page=state.experiment.page[]
        push!(pages,state.experiment.text[])
        Prototype.experiment_page(state,1)
        state.experiment.page[]==page && break
    end
    Prototype.experiment_page(state,saved_page-state.experiment.page[])
    replace(join(pages,"\n"),'\n'=>"")
end
@testset "Inspection pages immediately expose and clear failed-open errors" begin
    mktempdir() do directory
        fixture=experiment_fixture(directory);state=Prototype.State()
        @test Prototype.open_saved_experiment(state,fixture.path)
        before=(state.explorer,state.displayed[],state.explorer.selection[])
        missing=joinpath(directory,repeat("long-saved-recipe-location-",7),"missing-unique-recipe.jld2")
        @test !Prototype.open_saved_experiment(state,missing)
        original=state.experiment.error[]
        @test !isempty(original) && occursin("missing-unique-recipe",original)
        @test (state.explorer,state.displayed[],state.explorer.selection[])==before
        # Navigation itself refreshes pages, so assert the already-published
        # text before calling any page action. Removing the error subscription
        # makes this assertion fail even if later navigation rebuilds correctly.
        @test startswith(replace(state.experiment.text[],'\n'=>""),"Error: "*first(original,24))
        @test occursin(replace(original,'\n'=>""),inspection_page_text(state))
        @test occursin(repr(missing),inspection_page_text(state)) # SystemError quotes/escapes its locator
        @test Prototype.open_saved_experiment(state,fixture.path)
        @test isempty(state.experiment.error[])
        @test !startswith(state.experiment.text[],"Error:")
        @test !occursin("missing-unique-recipe",inspection_page_text(state))
        Prototype.dispose_state(state)
        @test isempty(state.experiment.subscriptions)
    end
end
