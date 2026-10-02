using Test
using HammerheadGUI
using HammerheadGUI.Hammerhead
using HammerheadGUI.GLMakie
using FileIO: save
using ImageCore: Gray, N0f8

function gui_checkpoint_fixture(dir)
    image=[mod(37i+13j+7i*j,251)/250 for i in 1:48,j in 1:48]
    pairs=Tuple{String,String}[]
    for i in 1:3
        pair=(joinpath(dir,"a-$i.png"),joinpath(dir,"b-$i.png"))
        save(pair[1],Gray{N0f8}.(image));save(pair[2],Gray{N0f8}.(circshift(image,(i,i+1))))
        push!(pairs,pair)
    end
    mask=falses(48,48);mask[:,1:6].=true
    passes=[PIVParameters(window_size=32,overlap=16,padding=true,max_iterations=2,convergence_tol=0.1),
            PIVParameters(window_size=16,overlap=8,padding=true,uod_enable=false,uncertainty=true)]
    recipe=PIVRecipe(passes;image_type=Float32,backend=:cpu,roi=ROI(3:46,3:46),mask,
        preprocessing=[PreprocessStep(:subtract_background;background=fill(0.01,48,48)),PreprocessStep(:intensity_cap;n_sigma=2.2)],
        scale=PhysicalScale(0.02,0.001,"mm","s"),predictor_smoothing=true)
    ExperimentRecord(pairs,recipe),pairs
end

function gui_checkpoint_create(dir,name,record)
    cc=CheckpointController(record;checkpoint_dir=joinpath(dir,"$name-meta"),output_dir=joinpath(dir,"$name-results"))
    create_checkpoint!(cc)
end

@testset "GUI checkpoint workflow" begin
    C=HammerheadGUI.Controllers
    mktempdir() do dir
        record,pairs=gui_checkpoint_fixture(dir)
        @testset "exact recipes and validated replacement" begin
            cc=CheckpointController(record)
            @test cc.state[]===:recipe_ready && cc.checkpoint[]===nothing
            @test cc.record[]!==record && cc.record[].recipe.mask!==record.recipe.mask
            @test recipe_identity(cc.record[].recipe)==recipe_identity(record.recipe)
            @test cc.record[].recipe.passes[1].max_iterations==2
            @test cc.record[].recipe.passes[2].uncertainty && cc.record[].recipe.image_type===Float32
            @test cc.record[].recipe.preprocessing[1].options["background"]!==record.recipe.preprocessing[1].options["background"]
            @test_throws ArgumentError create_checkpoint!(cc)
            @test cc.record[].recipe.recipe_id==record.recipe.recipe_id && cc.checkpoint[]===nothing
            create_checkpoint!(cc;checkpoint_dir=joinpath(dir,"capture-meta"),output_dir=joinpath(dir,"capture-results"))
            original=cc.checkpoint[];original_record=cc.record[];prior=(cc.progress[],cc.state[],cc.status[])
            occupied=joinpath(dir,"occupied");mkdir(occupied);write(joinpath(occupied,"keep"),"preserve")
            @test_throws ArgumentError create_checkpoint!(cc;checkpoint_dir=joinpath(dir,"refused-meta"),output_dir=occupied)
            @test cc.checkpoint[]===original && cc.record[]===original_record
            @test (cc.progress[],cc.state[],cc.status[])==prior
            @test !ispath(joinpath(dir,"refused-meta")) && read(joinpath(occupied,"keep"),String)=="preserve"
            @test_throws Exception open_checkpoint!(cc,occupied)
            @test cc.checkpoint[]===original && cc.record[]===original_record
            edited=deepcopy(original);edited.record.pairs[1]=reverse(edited.record.pairs[1])
            @test_throws ArgumentError open_checkpoint!(cc,edited)
            @test_throws ArgumentError open_checkpoint!(cc,edited;output_dir=edited.output_dir)
            @test_throws ArgumentError CheckpointController(edited)
            @test cc.checkpoint[]===original
            cc.recover_interrupted[]=true
            open_checkpoint!(cc,original)
            @test !cc.recover_interrupted[] && cc.progress[]==(0,3)

            other=gui_checkpoint_create(dir,"other",ExperimentRecord([pairs[1]],PIVRecipe(PIVParameters(window_size=16,overlap=8))))
            handle=C.on(cc.running) do busy
                if busy
                    # Execution values were captured before this notification.
                    cc.checkpoint[]=other.checkpoint[]
                    cc.record[]=other.record[]
                    cc.resume_record[]=other.record[]
                    cc.checkpoint_dir[]=other.checkpoint_dir[]
                    cc.output_dir[]=other.output_dir[]
                    cc.recover_interrupted[]=true
                end
            end
            start!(cc;async=false)
            C.off(handle)
            @test cc.state[]===:completed && cc.error[]===nothing && cc.progress[]==(3,3)
            @test checkpoint_state(original).committed==3 && checkpoint_state(other.checkpoint[]).committed==0
            @test cc.checkpoint[].checkpoint_id==original.checkpoint_id
            @test recipe_identity(cc.record[].recipe)==recipe_identity(record.recipe)
            @test cc.checkpoint_dir[]==original.path && cc.output_dir[]==original.output_dir
            @test !cc.recover_interrupted[] && !cc.running[]

            files=reduce(vcat,collect.(pairs))
            batch=BatchRunner(;files,effort=:high,window_schedule=[999],roi=ROI(1:32,1:32))
            from_form=experiment_record(batch)
            form_cc=CheckpointController(from_form)
            @test Hammerhead._experiment_pass_data.(form_cc.record[].recipe.passes)==
                Hammerhead._experiment_pass_data.(Hammerhead.effort_schedule(:high;image_size=(32,32)))
            script=joinpath(dir,"script.jl");write(script,"prepare(image)\n")
            custom=ExperimentRecord(pairs,PIVRecipe(record.recipe.passes;
                external_preprocess=ScriptReference(script;entrypoint="prepare(image)")))
            unsupported=CheckpointController(custom)
            @test unsupported.state[]===:recipe_unsupported && unsupported.record[].recipe.external_preprocess!==nothing
            @test_throws ArgumentError create_checkpoint!(unsupported;checkpoint_dir=joinpath(dir,"script-meta"),output_dir=joinpath(dir,"script-results"))
            @test !ispath(joinpath(dir,"script-meta")) && !ispath(joinpath(dir,"script-results"))
            open_checkpoint!(unsupported,original)
            @test unsupported.checkpoint[].checkpoint_id==original.checkpoint_id
        end

        @testset "native cancellation, progress and failures" begin
            initial=gui_checkpoint_create(dir,"initial-cancel",record)
            handle=C.on(initial.running) do busy
                busy && cancel!(initial)
            end
            start!(initial)
            @test initial.running[] && initial.state[]===:cancel_requested && occursin("cancellation requested",initial.status[])
            @test timedwait(()->!initial.running[],60)===:ok
            C.off(handle)
            @test initial.state[]===:cancelled && initial.error[]===nothing && initial.progress[]==(0,3)
            @test initial.last_attempt[].status===:cancelled && !initial.data_complete[]
            @test cancel!(initial)===initial && initial.state[]===:cancelled

            cc=gui_checkpoint_create(dir,"pair-cancel",record)
            seen=Tuple{Int,Int}[]
            start!(cc;async=false,progress=(done,total)->begin
                push!(seen,(done,total));cancel!(cc)
                @test_throws ArgumentError refresh_checkpoint!(cc)
                @test_throws ArgumentError open_checkpoint!(cc,initial.checkpoint[])
                @test_throws ArgumentError checkpoint_explorer(cc)
            end)
            @test cc.state[]===:cancelled && cc.progress[]==(1,3) && cc.error[]===nothing
            @test seen==[(1,3)] && cc.checkpoint_status[]===:cancelled
            bytes=read(only(checkpoint_results(cc.checkpoint[]).paths))
            start!(cc;async=false,progress=(done,total)->push!(seen,(done,total)))
            @test seen==[(1,3),(2,3),(3,3)] && cc.last_attempt[].start_committed==1
            @test cc.state[]===:completed && cc.data_complete[] && cc.progress[]==(3,3)
            @test read(first(checkpoint_results(cc.checkpoint[]).paths))==bytes
            last=gui_checkpoint_create(dir,"final-cancel",record)
            start!(last;async=false,progress=(done,total)->(done==total && cancel!(last)))
            @test last.state[]===:completed && last.error[]===nothing && last.data_complete[]

            failed=gui_checkpoint_create(dir,"callback-failure",record)
            original=ErrorException("observer failed after committed pair")
            start!(failed;async=false,progress=(done,total)->throw(original))
            @test failed.error[]===original && failed.state[]===:failed && !failed.running[]
            @test failed.progress[]==(1,3) && failed.checkpoint_status[]===:failed && !failed.data_complete[]
            @test checkpoint_state(failed.checkpoint[]).committed==1
            start!(failed;async=false)
            @test failed.last_attempt[].start_committed==1 && failed.state[]===:completed && failed.error[]===nothing
            terminal=gui_checkpoint_create(dir,"terminal-failure",record)
            primary=ErrorException("failure after final commit")
            start!(terminal;async=false,_phase_hook=(phase,i)->(phase===:before_terminal && throw(primary)))
            @test terminal.error[]===primary && terminal.checkpoint_status[]===:failed && terminal.data_complete[]
            @test terminal.progress[]==(3,3)
            @test export_checkpoint_results!(terminal,joinpath(dir,"failed-complete-native.jld2")) isa String
            prefix_bytes=read.(checkpoint_results(terminal.checkpoint[]).paths)
            start!(terminal;async=false)
            @test terminal.state[]===:completed && terminal.last_attempt[].start_committed==3
            @test read.(checkpoint_results(terminal.checkpoint[]).paths)==prefix_bytes

            changed=gui_checkpoint_create(dir,"preflight-failure",record)
            original_bytes=read(pairs[1][1]);open(io->write(io,UInt8(0)),pairs[1][1],"a")
            start!(changed;async=false)
            @test changed.error[] isa ArgumentError && changed.state[]===:failed && changed.progress[]==(0,3)
            @test isempty(readdir(joinpath(changed.checkpoint[].path,"attempts"))) && isempty(readdir(changed.checkpoint[].output_dir))
            write(pairs[1][1],original_bytes)
        end

        @testset "explicit interrupted recovery and fixed lazy browsing" begin
            cc=gui_checkpoint_create(dir,"interrupted",record)
            cp=cc.checkpoint[]
            current,snapshot=Hammerhead._checkpoint_fresh(cp)
            Hammerhead._checkpoint_begin(current,current.record,snapshot,current.environment)
            mkdir(joinpath(cp.path,"writer-lock")) # Recognized process-stop publication gap.
            refresh_checkpoint!(cc)
            @test cc.state[]===:unfinished && cc.progress[]==(0,3) && !cc.data_complete[]
            @test_throws ArgumentError checkpoint_explorer(cc)
            start!(cc;async=false)
            @test cc.error[] isa ArgumentError && cc.checkpoint_status[]===:unfinished && cc.progress[]==(0,3)
            cc.recover_interrupted[]=true
            start!(cc;async=false,progress=(done,total)->cancel!(cc))
            @test cc.state[]===:cancelled && cc.progress[]==(1,3) && !cc.recover_interrupted[]
            @test any(n->startswith(n,"recovered-writer-lock-"),readdir(cp.path))
            explorer=checkpoint_explorer(cc)
            @test explorer.results isa C._LazyDisplayResults && explorer.results.source isa CheckpointResults
            @test nframes(explorer)==1 && explorer.results.index==1
            @test isequal(current_result(explorer).u,physical(checkpoint_results(cp)[1]).u)
            @test_throws ArgumentError C.push_result!(explorer,current_result(explorer))
            start!(cc;async=false)
            @test cc.progress[]==(3,3) && nframes(explorer)==1
            reopened=checkpoint_explorer(cc)
            @test nframes(reopened)==3 && reopened.results.index==1
            for adapter in (reopened.results,view(reopened.results,2:3))
                report=quality_report(adapter)
                @test quality_report_data(report)["provenance"]["association"]=="unassociated"
                for source in reopened.results.source.paths
                    bytes=read(source)
                    @test_throws ArgumentError save_results(source,adapter)
                    @test read(source)==bytes
                    @test_throws ArgumentError save_quality_report(source,report)
                    @test read(source)==bytes
                end
            end
            prior=current_result(reopened)
            set_frame!(reopened,2)
            @test reopened.results.index==2 && reopened.results.result!==prior
            set_frame!(reopened,1);prior=current_result(reopened)
            payload=reopened.results.source.paths[2];original=read(payload)
            open(io->write(io,UInt8(0)),payload,"a")
            @test_throws ArgumentError set_frame!(reopened,2)
            @test reopened.frame[]==1 && current_result(reopened)===prior && !isempty(reopened.status[])
            write(payload,original)
            set_frame!(reopened,3)
            @test length(reopened.derived_cache)<=1 && reopened.results.index==3

            export_path=joinpath(dir,"native.jld2")
            @test export_checkpoint_results!(cc,export_path)==export_path
            @test length(load_results(export_path;lazy=true))==3
            exported=read(export_path)
            native_explorer=ResultExplorer(export_path;lazy=true)
            for adapter in (native_explorer.results,view(native_explorer.results,2:3))
                report=quality_report(adapter)
                @test quality_report_data(report)["provenance"]["association"]=="unassociated"
                @test !haskey(quality_report_data(report)["provenance"],"source_sha256")
                @test_throws ArgumentError save_results(export_path,adapter)
                @test read(export_path)==exported
                @test_throws ArgumentError save_quality_report(export_path,report)
                @test read(export_path)==exported
            end
            @test_throws ArgumentError export_checkpoint_results!(cc,export_path)
            @test read(export_path)==exported
            @test_throws ArgumentError export_checkpoint_results!(cc,pairs[1][1])
            locked=joinpath(dir,"locked.jld2");lockdir=joinpath(dir,".locked.jld2.checkpoint-export-lock")
            mkdir(lockdir);write(joinpath(lockdir,"keep"),"preserve")
            @test_throws ArgumentError export_checkpoint_results!(cc,locked)
            @test !ispath(locked) && read(joinpath(lockdir,"keep"),String)=="preserve"
            @test cc.export_path[]==export_path && cc.data_complete[]
        end

        @testset "offscreen dedicated workflow" begin
            GLMakie.activate!()
            empty=CheckpointController()
            fig=checkpoint_workflow(empty;batch=BatchRunner())
            @test size(colorbuffer(fig;px_per_unit=1))==(800,1100)
            button(label)=only(filter(b->b isa Button && b.label[]==label,fig.content))
            button("resume committed prefix").clicks[]+=1
            button("browse fixed prefix").clicks[]+=1
            button("create checkpoint").clicks[]+=1
            @test empty.state[]===:empty && empty.checkpoint[]===nothing && empty.status[]==""
            button("snapshot batch").clicks[]+=1
            @test empty.record[]===nothing && occursin("failed",empty.status[])
            many=ExperimentRecord(pairs,PIVRecipe(fill(last(record.recipe.passes),10)))
            controller=CheckpointController()
            loaded=checkpoint_workflow(controller;record=many)
            @test length(controller.record[].recipe.passes)==10 && controller.record[]!==many
            controller.checkpoint_dir[]=joinpath(dir,"a-long-metadata-directory-that-must-remain-fully-inspectable")
            controller.output_dir[]=joinpath(dir,"a-long-output-directory-that-must-remain-fully-inspectable")
            first=copy(colorbuffer(loaded;px_per_unit=1))
            menu=only(filter(b->b isa Menu,loaded.content))
            choose=only(filter(b->b isa Button && b.label[]=="choose complete recipe…",loaded.content))
            snapshot=only(filter(b->b isa Button && b.label[]=="snapshot batch",loaded.content))
            boxes=[b.layoutobservables.computedbbox[] for b in (menu,choose,snapshot)]
            @test GLMakie.Makie.right(boxes[1])<GLMakie.Makie.left(boxes[2])
            @test GLMakie.Makie.right(boxes[2])<GLMakie.Makie.left(boxes[3])
            @test GLMakie.Makie.right(boxes[3])<=1100
            next=only(filter(b->b isa Button && b.label[]=="next",loaded.content));next.clicks[]+=1
            @test colorbuffer(loaded;px_per_unit=1)!=first
            toggle=only(filter(b->b isa Toggle,loaded.content))
            @test !toggle.active[]
            controller.recover_interrupted[]=true
            @test toggle.active[]
            open_checkpoint!(controller,joinpath(dir,"interrupted-meta"))
            @test !toggle.active[] && controller.data_complete[]
            @test !isempty(colorbuffer(loaded;px_per_unit=1))
            embedding=Figure(size=(1100,800))
            @test checkpoint_workflow!(embedding[1,1],controller) isa GridLayout
            @test !isempty(colorbuffer(embedding;px_per_unit=1))
            script=joinpath(dir,"view-script.jl");write(script,"prepare(image)\n")
            custom=ExperimentRecord(pairs,PIVRecipe(record.recipe.passes;external_preprocess=
                ScriptReference(script;entrypoint="prepare(image)")))
            unsupported=CheckpointController()
            unsupported_fig=checkpoint_workflow(unsupported;record=custom)
            @test unsupported.state[]===:recipe_unsupported && unsupported.record[].recipe.external_preprocess!==nothing
            create=only(filter(b->b isa Button && b.label[]=="create checkpoint",unsupported_fig.content))
            before=unsupported.status[];create.clicks[]+=1
            @test unsupported.status[]==before && unsupported.checkpoint[]===nothing
            @test any(b->b isa Button && b.label[]=="open checkpoint…",unsupported_fig.content)
            open_checkpoint!(unsupported,joinpath(dir,"interrupted-meta"))
            @test unsupported.data_complete[] && !isempty(colorbuffer(unsupported_fig;px_per_unit=1))
        end
    end
end
