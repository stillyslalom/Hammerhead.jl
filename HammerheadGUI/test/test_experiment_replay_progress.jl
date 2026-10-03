using Test, HammerheadGUI
using HammerheadGUI.Hammerhead
using HammerheadGUI.GLMakie
using FileIO: save
using ImageCore: Gray, N0f8

function gui_replay_record(dir;script=false)
    image=[mod(37i+13j+7i*j,251)/250 for i in 1:48,j in 1:48]
    files=[joinpath(dir,"gui-progress-$i.png") for i in 1:4]
    for (i,path) in enumerate(files)
        save(path,Gray{N0f8}.(circshift(image,(i-1,2(i-1)))))
    end
    reference=nothing
    if script
        path=joinpath(dir,"preprocess.jl")
        write(path,"identity(image)\n")
        reference=ScriptReference(path;entrypoint="identity(image)")
    end
    p=PIVParameters(window_size=16,overlap=8,uod_enable=false,max_iterations=1)
    ExperimentRecord([(files[i],files[i+1]) for i in 1:3],
        PIVRecipe(p;threaded=false,mask=falses(48,48),external_preprocess=reference))
end

function gui_replay_controller(record,dir,name;history=true)
    ec=ExperimentController(record)
    ec.output_path[]=joinpath(dir,"$name-results.jld2")
    ec.run_record_path[]=history ? joinpath(dir,"$name-record.jld2") : ""
    ec
end

@testset "GUI ordinary replay progress and cancellation" begin
    C=HammerheadGUI.Controllers
    mktempdir() do dir
        record=gui_replay_record(dir)
        @testset "startup observer failure releases request" begin
            for field in (:running,:error,:last_run,:progress,:state,:status)
                ec=gui_replay_controller(record,dir,"startup-$field")
                original=ErrorException("startup $field observer")
                attempted=Ref(false)
                listener=on(getfield(ec,field)) do _
                    if ec.running[] && !attempted[]
                        attempted[]=true
                        throw(original)
                    end
                end
                start!(ec;async=false)
                @test attempted[] && ec.error[]===original
                @test ec.state[]===:failed && !ec.running[]
                @test ec._task[]===nothing && ec._cancel_token[]===nothing
                @test ec.progress[]==(0,3) && ec.last_run[]===nothing
                @test !ispath(ec.output_path[]) && !ispath(ec.run_record_path[])
                off(listener)
            end
        end

        @testset "frozen request and observer reentrancy" begin
            ec=gui_replay_controller(record,dir,"frozen")
            output,history=ec.output_path[],ec.run_record_path[]
            original_id=recipe_identity(record.recipe)
            visited=Tuple{Int,Int}[]
            listener=on(ec.running) do busy
                busy || return
                @test start!(ec;async=false)===ec
                @test_throws ArgumentError open_experiment!(ec,record)
                @test_throws ArgumentError save_experiment_record!(ec,joinpath(dir,"busy.jld2"))
                ec.output_path[]=record.input_files[record.pairs[1][1]]["path"]
                ec.run_record_path[]=record.input_files[record.pairs[1][2]]["path"]
                ec.allow_environment_change[]=true
                ec.custom_preprocess[]=image->error("next request callback")
                ec.record[].recipe.mask[1]=true
            end
            start!(ec;async=false,progress=(i,n)->push!(visited,(i,n)))
            off(listener)
            @test ec.state[]===:completed && ec.error[]===nothing
            @test visited==[(1,3),(2,3),(3,3)] && ec.progress[]==(3,3)
            @test ec.last_run[].output==output && ec.last_run[].recipe_id==original_id
            @test length(load_results(output))==3 && length(load_experiment(history).runs)==1
            @test !ec.record[].recipe.mask[1]
            @test ec._task[]===nothing && ec._cancel_token[]===nothing
            cancel!(ec)
            @test ec.state[]===:completed
        end

        @testset "cancel before execution leaves files intact" begin
            ec=gui_replay_controller(record,dir,"before")
            write(ec.output_path[],"output sentinel")
            write(ec.run_record_path[],"history sentinel")
            listener=on(ec.running) do busy
                busy && cancel!(ec)
            end
            start!(ec;async=false)
            off(listener)
            @test ec.state[]===:cancelled && ec.progress[]==(0,3)
            @test ec.last_run[]===nothing && isempty(ec.record[].runs)
            @test read(ec.output_path[],String)=="output sentinel"
            @test read(ec.run_record_path[],String)=="history sentinel"
            @test !ec.running[] && ec._cancel_token[]===nothing

            scheduled=gui_replay_controller(record,dir,"scheduled-before")
            start!(scheduled)
            task=scheduled._task[]
            cancel!(scheduled)
            @test timedwait(()->!scheduled.running[],60)==:ok
            wait(task)
            @test scheduled.state[]===:cancelled && scheduled.progress[]==(0,3)
            @test scheduled.last_run[]===nothing && isempty(scheduled.record[].runs)
            @test !ispath(scheduled.output_path[]) && !ispath(scheduled.run_record_path[])
        end

        @testset "written prefix waits for prefetched cleanup" begin
            scripted=gui_replay_record(dir;script=true)
            ec=gui_replay_controller(scripted,dir,"cleanup")
            entered,release,requested=Channel{Nothing}(1),Channel{Nothing}(1),Channel{Nothing}(1)
            calls=Threads.Atomic{Int}(0)
            ec.custom_preprocess[]=image->begin
                call=Threads.atomic_add!(calls,1)+1
                if call==3
                    put!(entered,nothing)
                    take!(release)
                end
                image
            end
            start!(ec;progress=(i,n)->begin
                i==1 || return
                take!(entered)
                cancel!(ec)
                put!(requested,nothing)
            end)
            task=ec._task[]
            try
                @test timedwait(()->isready(requested),60)==:ok
                @test ec.running[] && ec.state[]===:cancel_requested
                @test ec.progress[]==(1,3) && !istaskdone(task)
            finally
                isready(release) || put!(release,nothing)
            end
            @test timedwait(()->!ec.running[],60)==:ok
            wait(task)
            @test ec.state[]===:cancelled && ec.progress[]==(1,3)
            @test occursin("1 of 3 written pairs",sprint(showerror,ec.error[]))
            @test ec.last_run[].status===:failed && ec.last_run[].completed_pairs==1
            @test length(load_results(ec.output_path[]))==1
            @test last(load_experiment(ec.run_record_path[]).runs).status===:failed
            @test ec._task[]===nothing && ec._cancel_token[]===nothing
            @test_throws ArgumentError experiment_results(ec)
            @test_throws ArgumentError experiment_quality_report(ec)
        end

        @testset "final write completion and terminal observer errors" begin
            ec=gui_replay_controller(record,dir,"final")
            secondary=ErrorException("terminal notification")
            listeners=[on(ec.state) do state
                state===:completed && throw(secondary)
            end,on(ec.running) do busy
                busy || throw(secondary)
            end]
            start!(ec;async=false,progress=(i,n)->i==n && cancel!(ec))
            foreach(off,listeners)
            @test ec.state[]===:completed && ec.error[]===nothing && !ec.running[]
            @test ec.last_run[].status===:completed && ec.progress[]==(3,3)
            @test length(load_results(ec.output_path[]))==3
        end

        @testset "processing exception remains failure" begin
            ec=gui_replay_controller(record,dir,"failure")
            primary=ErrorException("progress callback failed")
            secondary=ErrorException("terminal error observer failed")
            listener=on(ec.error) do error
                error===nothing || throw(secondary)
            end
            start!(ec;async=false,progress=(i,n)->begin
                cancel!(ec)
                throw(primary)
            end)
            off(listener)
            @test ec.state[]===:failed && ec.error[]===primary && !ec.running[]
            @test ec.last_run[].status===:failed && ec.last_run[].completed_pairs==1
            @test ec.progress[]==(1,3) && ec._task[]===nothing
            @test length(load_results(ec.output_path[]))==1

            nohistory=gui_replay_controller(record,dir,"nohistory";history=false)
            start!(nohistory;async=false,progress=(i,n)->begin
                nohistory.progress[]=(999,999)
                cancel!(nohistory)
            end)
            @test nohistory.state[]===:cancelled && nohistory.progress[]==(1,3)
            @test nohistory.last_run[]===nothing && isempty(nohistory.record[].runs)
            @test length(load_results(nohistory.output_path[]))==1

            observed=gui_replay_controller(record,dir,"observable")
            listener=on(observed.progress) do counts
                counts[1]==1 && throw(primary)
            end
            start!(observed;async=false)
            off(listener)
            @test observed.state[]===:failed && observed.error[]===primary
            @test observed.last_run[].completed_pairs==1 && observed.progress[]==(1,3)
        end

        @testset "default offscreen workflow" begin
            ec=gui_replay_controller(record,dir,"view")
            start!(ec;async=false,progress=(i,n)->cancel!(ec))
            fig=experiment_workflow(ec)
            buttons=filter(block->block isa Button,fig.content)
            @test any(b->b.label[]=="cancel after current pair",buttons)
            labels=filter(block->block isa Label,fig.content)
            @test any(l->l.text[]=="Written pairs: 1 / 3",labels)
            buffer=colorbuffer(fig;px_per_unit=1)
            @test size(buffer)==(800,1100)
            if haskey(ENV,"HAMMERHEAD_REPLAY_SCREENSHOT")
                GLMakie.save(ENV["HAMMERHEAD_REPLAY_SCREENSHOT"],fig;px_per_unit=1)
            end
            ec.running[]=true
            ec.state[]=:cancel_requested
            ec.status[]="cancellation requested; waiting for a written-pair boundary and cleanup"
            @test size(colorbuffer(fig;px_per_unit=1))==(800,1100)
            if haskey(ENV,"HAMMERHEAD_REPLAY_SCREENSHOT")
                GLMakie.save(replace(ENV["HAMMERHEAD_REPLAY_SCREENSHOT"],".png"=>"-busy.png"),fig;px_per_unit=1)
            end
            ec.running[]=false
        end
    end
end
