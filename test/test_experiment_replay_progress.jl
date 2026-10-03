using Test, Hammerhead
using FileIO: save
using ImageCore: Gray, N0f8

function replay_progress_files(dir)
    image=[mod(37i+13j+7i*j,251)/250 for i in 1:48,j in 1:48]
    files=[joinpath(dir,"progress-$i.png") for i in 1:4]
    for (i,file) in enumerate(files)
        save(file,Gray{N0f8}.(circshift(image,(i-1,2(i-1)))))
    end
    [(files[i],files[i+1]) for i in 1:3],files
end

@testset "Ordinary replay written-pair progress" begin
    mktempdir() do dir
        pairs,files=replay_progress_files(dir)
        p=PIVParameters(window_size=16,overlap=8,uod_enable=false,max_iterations=1)
        record=ExperimentRecord(pairs,PIVRecipe(p;threaded=false,mask=falses(48,48),roi=ROI(5:44,5:44)))
        recipe_id=recipe_identity(record.recipe)
        baseline=joinpath(dir,"baseline.jld2")
        @test replay_experiment(record;output=baseline).completed_pairs==3
        events=Tuple{Symbol,Int}[]
        observed=joinpath(dir,"observed.jld2")
        history=joinpath(dir,"observed-record.jld2")
        run=replay_experiment(record;output=observed,run_record=history,
            progress=(i,n)->begin
                @test n==3
                push!(events,(:progress,i))
                # The public original changes; the captured execution cannot.
                record.recipe.mask[1,1]=true
            end,
            on_measurement_history=(i,h)->push!(events,(:history,i)),
            record_measurement_history=true,record_diagnostics=true)
        @test events==[(:history,1),(:progress,1),(:history,2),(:progress,2),(:history,3),(:progress,3)]
        @test run.status===:completed && run.completed_pairs==3 && run.recipe_id==recipe_id
        @test only(load_experiment(history).runs).run_id==run.run_id
        @test all(isequal(getfield(a,k),getfield(b,k)) for (a,b) in zip(load_results(baseline),load_results(observed))
            for k in (:x,:y,:u,:v,:outliers,:mask,:uncertainty_u,:uncertainty_v))
        @test load_measurement_history(observed,3;verify_result=true)!==nothing
        record=load_experiment(history)

        primary=ErrorException("stop after first written pair")
        failedout=joinpath(dir,"failed-prefix.jld2")
        failedhistory=joinpath(dir,"failed-record.jld2")
        counts=Tuple{Int,Int}[]
        caught=try
            replay_experiment(record;output=failedout,run_record=failedhistory,
                progress=(i,n)->(push!(counts,(i,n));throw(primary)))
            nothing
        catch err
            err
        end
        @test caught===primary && counts==[(1,3)]
        @test length(load_results(failedout))==1
        failed=last(load_experiment(failedhistory).runs)
        @test failed.status===:failed && failed.completed_pairs==1
        @test failed.error==sprint(showerror,primary) && failed.output_sha256!==nothing

        # A final callback error is still failure, despite a complete numerical prefix.
        finalout=joinpath(dir,"final-prefix.jld2")
        finalhistory=joinpath(dir,"final-record.jld2")
        @test_throws ErrorException replay_experiment(record;output=finalout,run_record=finalhistory,
            progress=(i,n)->(i==n && throw(primary)))
        @test length(load_results(finalout))==3
        @test last(load_experiment(finalhistory).runs).status===:failed
        @test last(load_experiment(finalhistory).runs).completed_pairs==3

        # Preflight never calls progress or replaces either sentinel file.
        untouched=joinpath(dir,"untouched.jld2");write(untouched,"keep output")
        history_sentinel=joinpath(dir,"history-sentinel.jld2");write(history_sentinel,"keep history")
        count=Ref(0)
        invalid=deepcopy(record);invalid.creation_environment["julia_version"]="0.0.0"
        @test_throws ArgumentError replay_experiment(invalid;output=untouched,run_record=history_sentinel,
            progress=(i,n)->(count[]+=1))
        @test count[]==0 && read(untouched,String)=="keep output" && read(history_sentinel,String)=="keep history"
        if Sys.iswindows()
            for (out,log) in ((joinpath(dir,"fresh.jld2"),joinpath(dir,"FRESH.JLD2")),
                (joinpath(dir,"dot.jld2."),joinpath(dir,"dot-record.jld2")),
                (joinpath(dir,"space.jld2"),joinpath(dir,"space-record.jld2 ")))
                @test_throws ArgumentError replay_experiment(record;output=out,run_record=log,
                    progress=(i,n)->(count[]+=1))
                @test count[]==0 && !ispath(out) && !ispath(log)
            end
        end

        # A progress error remains primary if recording the failed metadata fails.
        missing_parent=joinpath(dir,"metadata-is-directory");mkdir(missing_parent)
        secondout=joinpath(dir,"metadata-failure.jld2")
        caught=try
            replay_experiment(record;output=secondout,run_record=missing_parent,
                progress=(i,n)->throw(primary))
            nothing
        catch err
            err
        end
        @test caught===primary && length(load_results(secondout))==1

        # Prefetched preprocessing can fail too; cleanup preserves the progress error.
        script=joinpath(dir,"preprocess.jl");write(script,"prepare(image)\n")
        customrecord=ExperimentRecord(pairs,PIVRecipe(p;threaded=false,
            external_preprocess=ScriptReference(script;entrypoint="prepare(image)")))
        calls=Threads.Atomic{Int}(0)
        entered=Channel{Nothing}(1);release=Channel{Nothing}(1)
        secondary=ErrorException("prefetch cleanup failure")
        custom=image->begin
            k=Threads.atomic_add!(calls,1)+1
            if k==3
                put!(entered,nothing);take!(release);throw(secondary)
            end
            image
        end
        drainedout=joinpath(dir,"drained.jld2")
        caught=try
            replay_experiment(customrecord;output=drainedout,custom_preprocess=custom,
                progress=(i,n)->begin
                    take!(entered);put!(release,nothing);throw(primary)
                end)
            nothing
        catch err
            err
        finally
            isready(release) || put!(release,nothing)
        end
        @test caught===primary && calls[]==3
        @test length(load_results(drainedout))==1
    end
end
