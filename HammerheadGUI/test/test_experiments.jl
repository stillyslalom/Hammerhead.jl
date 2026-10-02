using Test
using HammerheadGUI
using HammerheadGUI.Hammerhead
using HammerheadGUI.GLMakie
using FileIO: save
using ImageCore: Gray, N0f8

@testset "GUI experiment workflow" begin
    C=HammerheadGUI.Controllers
    mktempdir() do dir
        image=[mod(37i+13j+7i*j,251)/250 for i in 1:48,j in 1:48]
        images=(image,circshift(image,(1,2)),image,circshift(image,(1,2)))
        files=[joinpath(dir,"frame-$i.png") for i in 1:4]
        for (path,data) in zip(files,images)
            save(path,Gray{N0f8}.(data))
        end
        bc=BatchRunner(;files,window_schedule=[32,16],padding=true,
            apodization=:gauss,roi=ROI(5:44,5:44),mask=falses(48,48),
            pixel_size=0.02,dt=0.001,length_unit="mm",time_unit="s")
        bc.mask[][:,1:8].=true
        pp=PreprocessPreview(files[1];enabled=[:intensity_cap,:highpass_filter])
        C.set_background!(pp,[files[1],files[2]];method=:min)
        enable_step!(pp,:subtract_background)
        move_step!(pp,:highpass_filter,-2)
        steps=preprocess_steps(pp)
        @test first(steps).operation===:highpass_filter
        @test last(steps).operation===:intensity_cap
        @test steps[2].options["background"]==pp.background[]
        @test steps[2].options["background"]!==pp.background[]
        set_preprocess!(bc,pp)
        callback=bc.preprocess[]
        record=experiment_record(bc)
        @test record.recipe.image_type===Float64 && record.recipe.backend===:cpu
        @test record.recipe.threaded==(Threads.nthreads()>1)
        @test record.recipe.roi.rows==5:44 && record.recipe.mask==bc.mask[]
        @test record.recipe.scale.pixel_size==0.02 && record.recipe.scale.dt==0.001
        @test [s.operation for s in record.recipe.preprocessing]==[s.operation for s in steps]
        @test length(record.recipe.passes)==2
        @test all(Hammerhead._experiment_pass_data.(record.recipe.passes).==
                  Hammerhead._experiment_pass_data.(C.build_parameters(bc)))
        set_step_param!(pp,:highpass_filter,:sigma,5)
        pp.background[][1]=0.9
        @test record.recipe.preprocessing[1].options["sigma"]==3
        @test experiment_record(bc).recipe.recipe_id==record.recipe.recipe_id
        @test callback(load_image(files[1]))==Hammerhead._experiment_preprocess(record.recipe,nothing)(load_image(files[1]))
        saved=joinpath(dir,"gui-experiment.jld2")
        @test save_batch_experiment(saved,bc).recipe.recipe_id==record.recipe.recipe_id

        @testset "metadata and unsupported form refusal" begin
            bare=identity
            bc.preprocess[]=bare
            sentinel=joinpath(dir,"preserve.jld2");write(sentinel,"preserve")
            @test_throws ArgumentError save_batch_experiment(sentinel,bc)
            @test read(sentinel,String)=="preserve"
            script=joinpath(dir,"prepare.jl");write(script,"identity(image)\n")
            reference=ScriptReference(script;entrypoint="identity(image)")
            referenced=experiment_record(bc;script_reference=reference)
            @test referenced.recipe.external_preprocess.sha256==reference.sha256
            @test isempty(referenced.recipe.preprocessing)
            set_preprocess!(bc,nothing)
            @test_throws ArgumentError experiment_record(bc;script_reference=reference)
            set_preprocess!(bc,pp)
            @test_throws ArgumentError experiment_record(bc;script_reference=reference)
            bc.preprocess_snapshot[].steps[1].options["sigma"]=7
            @test_throws ArgumentError experiment_record(bc)
            set_preprocess!(bc,pp)
            bc.preprocess_snapshot[].steps[2].options["background"][1]=0.8
            @test_throws ArgumentError experiment_record(bc)
            set_preprocess!(bc,pp)
            bc.running[]=true
            @test_throws ArgumentError experiment_record(bc)
            bc.running[]=false
            memory=BatchRunner(files=[image,image],window_schedule=[16])
            @test_throws ArgumentError experiment_record(memory)
            @test_throws ArgumentError experiment_record(BatchRunner())
            # Restore original frozen built-ins for numerical replay checks.
            set_preprocess!(bc,callback)
            @test_throws ArgumentError experiment_record(bc)
        end

        @testset "exact effort expansion and dimensions" begin
            effort=BatchRunner(;files,effort=:high,window_schedule=[999])
            effort_record=experiment_record(effort)
            expected=Hammerhead.effort_schedule(:high;image_size=(48,48))
            @test Hammerhead._experiment_pass_data.(effort_record.recipe.passes)==Hammerhead._experiment_pass_data.(expected)
            set_roi!(effort,ROI(3:34,4:35))
            cropped=experiment_record(effort)
            @test Hammerhead._experiment_pass_data.(cropped.recipe.passes)==
                  Hammerhead._experiment_pass_data.(Hammerhead.effort_schedule(:high;image_size=(32,32)))
            bigger=joinpath(dir,"bigger.png");save(bigger,Gray{N0f8}.(fill(0.5,80,80)))
            varied=BatchRunner(files=[files[1],files[2],bigger,bigger],effort=:high)
            @test_throws ArgumentError experiment_record(varied)
            set_roi!(varied,ROI(1:32,1:32))
            @test length(experiment_record(varied).pairs)==2
        end

        @testset "lossless reopen, replay and lazy browsing" begin
            ec=ExperimentController(saved)
            @test ec.state[]===:ready && !ec.running[]
            @test ec.run_record_path[]==saved && !ec.allow_environment_change[]
            ec.output_path[]=joinpath(dir,"replayed.jld2")
            expected=run_piv_sequence(image_pairs(files),record.recipe.passes;
                preprocess=callback,roi=record.recipe.roi,mask=record.recipe.mask,
                scale=record.recipe.scale,threaded=record.recipe.threaded,progress=false)
            start!(ec;async=false)
            @test ec.state[]===:completed && !ec.running[] && ec.error[]===nothing
            @test ec.last_run[].status===:completed && ec.last_run[].completed_pairs==2
            @test length(ec.record[].runs)==1 && length(load_experiment(saved).runs)==1
            actual=load_results(ec.last_run[].output)
            @test all(isequal(actual[i].u,expected[i].u) && isequal(actual[i].v,expected[i].v) for i in 1:2)
            explorer=experiment_results(ec)
            @test explorer.results isa C._LazyDisplayResults
            @test nframes(explorer)==2 && current_result(explorer).scale!==nothing
            set_frame!(explorer,2)
            @test explorer.results.index==2 && length(explorer.derived_cache)<=1
            @test occursin("keep_correlation_planes",experiment_summary(ec))
            @test occursin(ec.last_run[].run_id,experiment_run_history(ec))
            @test occursin("No recorded runs",experiment_run_history(ExperimentController(record)))
            @test_throws ArgumentError experiment_results(ExperimentController())
            preserved=read(ec.last_run[].output)
            save_results(ec.last_run[].output,[expected[2]])
            @test_throws ArgumentError experiment_results(ec)
            write(ec.last_run[].output,preserved)
            @test nframes(experiment_results(ec))==2

            @testset "direct-record provenance checks" begin
                invalid=deepcopy(ec.record[])
                invalid.pairs[1][1]=4
                @test_throws ArgumentError ExperimentController(invalid)
                before=ec.record[]
                output_before=ec.output_path[]
                history_before=ec.run_record_path[]
                @test_throws ArgumentError open_experiment!(ec,invalid)
                @test ec.record[]===before && ec.output_path[]==output_before && ec.run_record_path[]==history_before
                run=ec.last_run[]
                stale=ExperimentRun(run.run_id,repeat("0",64),run.input_id,run.started_at,run.finished_at,
                    run.status,run.completed_pairs,run.output,run.output_sha256,run.environment,run.error)
                invalid=deepcopy(ec.record[]);invalid.runs[1]=stale
                @test_throws ArgumentError ExperimentController(invalid)
                @test_throws ArgumentError open_experiment!(ec,invalid)
                @test ec.record[]===before
                invalid=deepcopy(ec.record[]);invalid.creation_environment["julia_threads"]=0
                @test_throws ArgumentError ExperimentController(invalid)
                ec.last_run[]=stale
                @test_throws ArgumentError experiment_results(ec)
                stale_input=ExperimentRun(run.run_id,run.recipe_id,repeat("0",64),run.started_at,run.finished_at,
                    run.status,run.completed_pairs,run.output,run.output_sha256,run.environment,run.error)
                ec.last_run[]=stale_input
                @test_throws ArgumentError experiment_results(ec)
                ec.last_run[]=run
                @test nframes(experiment_results(ec))==2
            end

            # Non-form fields and duplicate/custom preprocessing stay untouched.
            full=PIVRecipe(PIVParameters(window_size=(16,12),search_area_size=(20,16),
                overlap=(8,6),n_peaks=2,peak_finder=:exclusion,keep_correlation_planes=true,
                validation=(:velocity_magnitude=>(min=0,max=Inf),));
                preprocessing=[PreprocessStep(:clahe;tiles=(2,3),nbins=32),
                    PreprocessStep(:invert_image),PreprocessStep(:invert_image),
                    PreprocessStep(:local_variance_normalize;epsilon=0.03)],
                backend=:cpu,image_type=Float32,predictor_smoothing=false,mask_threshold=0.3)
            complete=ExperimentRecord([(files[1],files[2])],full)
            exact=ExperimentController(complete)
            @test exact.record[]!==complete
            @test recipe_identity(exact.record[].recipe)==recipe_identity(full)
            save_experiment_record!(exact,joinpath(dir,"full.jld2"))
            @test recipe_identity(load_experiment(exact.run_record_path[]).recipe)==recipe_identity(full)
            summary=experiment_summary(exact)
            @test occursin("epsilon=0.03",summary) && occursin("nbins=32",summary)
            @test occursin("tiles=[2, 3]",summary)
            @test occursin("Float32",summary) && occursin("search_area_size=(20, 16)",summary)
            previous=exact.record[]
            @test_throws Exception open_experiment!(exact,joinpath(dir,"missing.jld2"))
            @test exact.record[]===previous
            @test_throws ArgumentError save_experiment_record!(exact,files[1])
            @test exact.record[]===previous
        end

        @testset "failure preservation and captured run settings" begin
            ec=ExperimentController(record)
            untouched=joinpath(dir,"untouched.jld2");write(untouched,"preserve")
            ec.output_path[]=files[1]
            original=read(files[1]);start!(ec;async=false)
            @test ec.state[]===:failed && ec.error[] isa ArgumentError
            @test read(files[1])==original && !ec.running[] && ec.last_run[]===nothing
            ec.output_path[]=untouched
            ec.record[].creation_environment["julia_version"]="0.0.0"
            start!(ec;async=false)
            @test ec.state[]===:failed && read(untouched,String)=="preserve"
            ec.allow_environment_change[]=true
            # Synchronous observers edit the form as soon as busy is announced;
            # execution must retain all values captured beforehand.
            seen=Ref(false)
            observer=on(ec.running) do running
                if running
                    seen[]=true
                    ec.output_path[]=files[2]
                    ec.allow_environment_change[]=false
                    ec.custom_preprocess[]=identity
                    @test_throws ArgumentError open_experiment!(ec,saved)
                    @test_throws ArgumentError save_experiment_record!(ec,saved)
                end
            end
            start!(ec;async=false)
            off(observer)
            @test seen[] && ec.state[]===:completed
            @test ec.last_run[].output==abspath(untouched)
            @test read(files[1])==original

            asynchronous=ExperimentController(record;output_path=joinpath(dir,"async-results.jld2"))
            captured_output=asynchronous.output_path[]
            start!(asynchronous)
            @test asynchronous.running[] && asynchronous.state[]===:busy
            @test_throws ArgumentError experiment_results(asynchronous)
            @test_throws ArgumentError open_experiment!(asynchronous,saved)
            asynchronous.output_path[]=files[1]
            @test timedwait(()->!asynchronous.running[],60)==:ok
            @test asynchronous.state[]===:completed && asynchronous.last_run[].output==abspath(captured_output)
            @test read(files[1])==original

            script=joinpath(dir,"failing.jl");write(script,"prepare(image)\n")
            reference=ScriptReference(script;entrypoint="prepare(image)")
            failing_record=ExperimentRecord(image_pairs(files),
                PIVRecipe(PIVParameters(window_size=16,overlap=8);external_preprocess=reference))
            failed=ExperimentController(failing_record)
            failed.output_path[]=joinpath(dir,"failed-prefix.jld2")
            failed.run_record_path[]=joinpath(dir,"failed-record.jld2")
            calls=Ref(0);primary=ErrorException("explicit callback failure")
            failed.custom_preprocess[]=img->begin
                calls[]+=1
                calls[]>=3 && throw(primary)
                img
            end
            start!(failed;async=false)
            @test failed.state[]===:failed && failed.error[]===primary && !failed.running[]
            @test failed.last_run[].status===:failed && failed.last_run[].completed_pairs==1
            @test only(failed.record[].runs).error==sprint(showerror,primary)
            @test length(load_results(failed.output_path[]))==1
            @test_throws ArgumentError experiment_results(failed)
        end

        @testset "offscreen dedicated workflow" begin
            GLMakie.activate!()
            empty=ExperimentController()
            fig=experiment_workflow(empty;batch=BatchRunner(),size=(1100,800))
            @test size(colorbuffer(fig;px_per_unit=1))==(800,1100)
            @test !any(b->b isa Button && b.label[]=="cancel",fig.content)
            snapshot=only(filter(b->b isa Button && b.label[]=="snapshot batch",fig.content))
            snapshot.clicks[]+=1
            @test empty.record[]===nothing && occursin("failed",empty.status[])
            many=ExperimentRecord([(files[1],files[2])],
                PIVRecipe(fill(PIVParameters(window_size=16,overlap=8),10)))
            loaded=ExperimentController(many)
            loadedfig=experiment_workflow(loaded;batch=bc)
            first=copy(colorbuffer(loadedfig;px_per_unit=1))
            next=only(filter(b->b isa Button && b.label[]=="next",loadedfig.content))
            next.clicks[]+=1
            @test colorbuffer(loadedfig;px_per_unit=1)!=first
            loaded.status[]="failed: changed inputs"
            @test !isempty(colorbuffer(loadedfig;px_per_unit=1))
            embedding=Figure(size=(1100,800))
            @test experiment_workflow!(embedding[1,1],loaded) isa GridLayout
            @test !isempty(colorbuffer(embedding;px_per_unit=1))
        end
    end
end
