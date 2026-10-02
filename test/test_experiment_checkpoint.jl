using Test
using Hammerhead
using JLD2
using FileIO: save
using ImageCore: Gray, N0f8

function checkpoint_test_record(dir;backend=:cpu,T=Float32)
    base=[mod(37i+13j+7i*j,251)/250 for i in 1:48,j in 1:48]
    pairs=Tuple{String,String}[]
    for i in 1:3
        a,b=joinpath(dir,"a-$i.png"),joinpath(dir,"b-$i.png")
        save(a,Gray{N0f8}.(base));save(b,Gray{N0f8}.(circshift(base,(i,i+1))))
        push!(pairs,(a,b))
    end
    mask=falses(48,48);mask[:,1:6].=true
    recipe=PIVRecipe(PIVParameters(window_size=16,overlap=8,padding=true,uod_enable=false);
        preprocessing=[PreprocessStep(:subtract_background;background=fill(0.01,48,48))],
        roi=ROI(3:46,3:46),mask,scale=PhysicalScale(0.02,0.001,"mm","s"),backend,image_type=T)
    ExperimentRecord(pairs,recipe),pairs
end

function checkpoint_test_create(dir,name,record)
    create_checkpoint(joinpath(dir,"$name-meta"),record;output_dir=joinpath(dir,"$name-results"))
end

function checkpoint_test_same(a,b)
    all(k->isequal(getfield(a,k),getfield(b,k)),
        (:x,:y,:u,:v,:peak_ratio,:correlation_moment,:uncertainty_u,:uncertainty_v,:outliers,:mask,:correlation_planes)) &&
    Hammerhead._experiment_pass_data(a.parameters)==Hammerhead._experiment_pass_data(b.parameters) &&
    all(k->getfield(a.scale,k)==getfield(b.scale,k),fieldnames(PhysicalScale))
end

function checkpoint_test_replace(path,kind,data)
    jldopen(path,"w") do f
        f["checkpoint_format_version"]=1;f["checkpoint_kind"]=kind;f["checkpoint_data"]=data
    end
end

@testset "Explicit planar checkpoints" begin
    mktempdir() do dir
        record,pairs=checkpoint_test_record(dir)
        expected=run_piv_sequence(pairs,record.recipe.passes;progress=false,image_type=Float32,
            preprocess=Hammerhead._experiment_preprocess(record.recipe,nothing),roi=record.recipe.roi,
            mask=record.recipe.mask,scale=record.recipe.scale,threaded=false)

        @testset "commit, cancellation, resume and native export" begin
            cp=checkpoint_test_create(dir,"main",record)
            @test checkpoint_state(cp).status===:ready && checkpoint_state(cp).committed==0
            @test isempty(checkpoint_results(cp))
            seen=Int[];stop=Ref(false)
            attempt=resume_checkpoint!(cp;progress=(i,n)->begin
                @test n==3
                push!(seen,i);stop[]=true
            end,cancel=()->stop[])
            @test attempt.status===:cancelled && attempt.committed==1 && attempt.start_committed==0
            @test seen==[1] && !checkpoint_state(cp).writer_lock_present
            first_index=checkpoint_results(cp)
            @test first_index isa CheckpointResults && length(first_index)==1
            @test checkpoint_test_same(first_index[1],expected[1])
            first_bytes=read(first_index.paths[1])
            @test_throws ArgumentError save_checkpoint_results(joinpath(dir,"partial-native.jld2"),cp)
            resumed=resume_checkpoint!(load_checkpoint(cp.path);progress=(i,n)->push!(seen,i))
            @test resumed.status===:completed && resumed.start_committed==1 && resumed.committed==3
            @test seen==[1,2,3] && checkpoint_state(cp).data_complete
            @test read(first_index.paths[1])==first_bytes && length(first_index)==1
            actual=checkpoint_results(cp)
            @test all(checkpoint_test_same(actual[i],expected[i]) for i in 1:3)
            @testset "generic writer aliases preserve checkpoint payloads" begin
                for index in (actual,view(actual,2:3))
                    # Views must protect every parent payload, including excluded entries.
                    report=quality_report(index)
                    @test quality_report_data(report)["provenance"]["association"]=="unassociated"
                    for payload in actual.paths
                        bytes=read(payload)
                        @test_throws ArgumentError save_results(payload,index)
                        @test read(payload)==bytes
                        @test_throws ArgumentError save_quality_report(payload,report)
                        @test read(payload)==bytes
                    end
                end
            end
            @test actual[1] isa PIVResult{Float32}
            @test isequal(physical(actual[1]).u,physical(physical(actual[1])).u)
            before=sort(readdir(joinpath(cp.path,"attempts")))
            again=resume_checkpoint!(cp)
            @test again.attempt_id==resumed.attempt_id && readdir(joinpath(cp.path,"attempts"))==before
            aggregate=joinpath(dir,"native.jld2")
            @test save_checkpoint_results(aggregate,cp)==aggregate
            native=load_results(aggregate;lazy=true)
            @test native isa ResultFile && length(native)==3
            @test all(checkpoint_test_same(native[i],expected[i]) for i in 1:3)
            @test JLD2.load(aggregate,"sources/000001")==collect(pairs[1])
            @test JLD2.load(aggregate,"checkpoint_id")==cp.checkpoint_id
            @test_throws ArgumentError save_checkpoint_results(aggregate,cp)
            @test_throws ArgumentError save_checkpoint_results(pairs[1][1],cp)
            @test_throws ArgumentError save_checkpoint_results(joinpath(cp.path,"export.jld2"),cp)
            @test_throws ArgumentError save_checkpoint_results(joinpath(cp.output_dir,"export.jld2"),cp)

            @testset "exclusive derivative publication" begin
                concurrent=joinpath(dir,"concurrent-native.jld2")
                staged=Channel{Nothing}(1);release=Channel{Nothing}(1)
                first_export=@async save_checkpoint_results(concurrent,cp;_phase_hook=(phase,i)->begin
                    if phase===:aggregate_staged
                        put!(staged,nothing);take!(release)
                    end
                end)
                take!(staged)
                try
                    @test_throws ArgumentError save_checkpoint_results(concurrent,cp)
                    @test !ispath(concurrent)
                finally
                    put!(release,nothing)
                end
                @test fetch(first_export)==concurrent
                @test length(ResultFile(concurrent))==3
                @test !ispath(joinpath(dir,".concurrent-native.jld2.checkpoint-export-lock"))

                refused=joinpath(dir,"stale-export.jld2")
                stale=joinpath(dir,".stale-export.jld2.checkpoint-export-lock")
                mkdir(stale);write(joinpath(stale,"keep"),"preserve")
                @test_throws ArgumentError save_checkpoint_results(refused,cp)
                @test !ispath(refused) && read(joinpath(stale,"keep"),String)=="preserve"

                failed=joinpath(dir,"failed-export.jld2")
                original=ErrorException("aggregate staging failure")
                @test try
                    save_checkpoint_results(failed,cp;_phase_hook=(phase,i)->throw(original));false
                catch err;err===original end
                @test !ispath(failed) && checkpoint_state(cp).data_complete
                @test save_checkpoint_results(failed,cp)==failed
            end

            untouched=checkpoint_test_create(dir,"cancel-first",record)
            @test resume_checkpoint!(untouched;cancel=()->true).status===:cancelled
            @test checkpoint_state(untouched).committed==0 && isempty(checkpoint_results(untouched))
            finish=resume_checkpoint!(untouched;cancel=()->checkpoint_state(untouched).committed==3)
            @test finish.status===:completed

            @testset "relocated locators and KA precision" begin
                relocated_dir=joinpath(dir,"relocated");mkdir(relocated_dir)
                relocated=Tuple{String,String}[]
                for pair in pairs
                    copied=Tuple(joinpath(relocated_dir,basename(p)) for p in pair)
                    for (source,target) in zip(pair,copied);Base.cp(source,target);end
                    push!(relocated,copied)
                end
                moved=ExperimentRecord(relocated,record.recipe)
                empty_cp=checkpoint_test_create(dir,"relocated",record)
                @test moved.input_id==record.input_id
                @test resume_checkpoint!(empty_cp,moved).status===:completed
                mixed=checkpoint_test_create(dir,"mixed-relocation",record)
                stop=Ref(false)
                @test resume_checkpoint!(mixed;progress=(i,n)->(stop[]=true),cancel=()->stop[]).committed==1
                @test resume_checkpoint!(mixed,moved).start_committed==1
                mixed_export=save_checkpoint_results(joinpath(dir,"mixed-native.jld2"),mixed)
                @test JLD2.load(mixed_export,Hammerhead.source_key(1))==collect(pairs[1])
                @test all(JLD2.load(mixed_export,Hammerhead.source_key(i))==collect(relocated[i]) for i in 2:3)
                @test all(checkpoint_test_same(ResultFile(mixed_export)[i],expected[i]) for i in 1:3)
                output_copy=joinpath(dir,"relocated-output");mkdir(output_copy)
                for file in readdir(empty_cp.output_dir);Base.cp(joinpath(empty_cp.output_dir,file),joinpath(output_copy,file));end
                @test all(checkpoint_test_same(checkpoint_results(load_checkpoint(empty_cp.path;output_dir=output_copy))[i],expected[i]) for i in 1:3)
                @test_throws ArgumentError load_checkpoint(empty_cp.path;output_dir=empty_cp.path)
                ka=PIVRecipe(record.recipe.passes;backend=:ka,image_type=Float64)
                ka_record=ExperimentRecord([pairs[1]],ka)
                ka_cp=checkpoint_test_create(dir,"ka",ka_record)
                @test resume_checkpoint!(ka_cp).status===:completed
                direct=only(run_piv_sequence([pairs[1]],ka.passes;backend=:ka,threaded=false,progress=false))
                @test isequal(checkpoint_results(ka_cp)[1].u,direct.u)
            end
        end

        @testset "handled failures before and after publication" begin
            for (name,original) in (("error",ErrorException("owner publication failure")),("interrupt",InterruptException()))
                cp=checkpoint_test_create(dir,"owner-$name",record)
                @test try
                    resume_checkpoint!(cp;_phase_hook=(stage,i)->(stage===:lock_owner_published && throw(original)));false
                catch err;err===original end
                state=checkpoint_state(cp)
                @test state.status===:ready && state.committed==0 && !state.writer_lock_present
                @test resume_checkpoint!(cp).status===:completed
            end
            for phase in (:during_staging,:payload_closed,:payload_published,:descriptor_staged,:descriptor_published)
                cp=checkpoint_test_create(dir,"failure-$phase",record)
                original=ErrorException("failure at $phase")
                captured=try
                    resume_checkpoint!(cp;_phase_hook=(stage,i)->(stage===phase && throw(original)))
                    nothing
                catch err;err end
                @test captured===original
                state=checkpoint_state(cp)
                @test state.status===:failed && !state.writer_lock_present
                @test state.committed==(phase===:descriptor_published ? 1 : 0)
                @test resume_checkpoint!(cp).status===:completed
                @test all(checkpoint_test_same(checkpoint_results(cp)[i],expected[i]) for i in 1:3)
            end
            cp=checkpoint_test_create(dir,"progress-failure",record)
            primary=ErrorException("progress failed after commit")
            @test try resume_checkpoint!(cp;progress=(i,n)->throw(primary));false catch err;err===primary end
            @test checkpoint_state(cp).committed==1 && checkpoint_state(cp).status===:failed
            @test resume_checkpoint!(cp).start_committed==1
            cancellation=checkpoint_test_create(dir,"publication-cancel",record)
            @test resume_checkpoint!(cancellation;_phase_hook=(stage,i)->
                (stage===:descriptor_published && throw(Hammerhead._CheckpointCancelled()))).committed==1
            @test checkpoint_state(cancellation).status===:cancelled
            @test resume_checkpoint!(cancellation).start_committed==1
            damaged=checkpoint_test_create(dir,"failure-recording",record)
            original=ErrorException("original processing failure")
            descriptor=joinpath(damaged.path,"commits",Hammerhead._checkpoint_descriptor(1))
            intact=Ref(UInt8[])
            @test try
                resume_checkpoint!(damaged;_phase_hook=(stage,i)->begin
                    if stage===:descriptor_published
                        intact[]=read(descriptor);write(descriptor,"invalid metadata");throw(original)
                    end
                end);false
            catch err;err===original end
            @test_throws Exception load_checkpoint(damaged.path)
            @test !ispath(joinpath(damaged.path,"writer-lock"))
            write(descriptor,intact[])
            @test checkpoint_state(damaged).status===:unfinished && checkpoint_state(damaged).committed==1
            @test resume_checkpoint!(damaged;recover_interrupted=true).start_committed==1
            # Live same-process recovery is refused even from a progress observer.
            live=checkpoint_test_create(dir,"live-writer",record)
            @test resume_checkpoint!(live;progress=(i,n)->begin
                @test_throws ArgumentError resume_checkpoint!(live;recover_interrupted=true)
            end).status===:completed
        end

        @testset "preflight and corruption preserve existing data" begin
            occupied=joinpath(dir,"occupied");mkdir(occupied);write(joinpath(occupied,"keep"),"keep")
            @test_throws ArgumentError create_checkpoint(joinpath(dir,"not-created"),record;output_dir=occupied)
            @test !ispath(joinpath(dir,"not-created")) && read(joinpath(occupied,"keep"),String)=="keep"
            @test_throws ArgumentError create_checkpoint(joinpath(dir,"same"),record;output_dir=joinpath(dir,"same"))
            @test_throws ArgumentError create_checkpoint(joinpath(dir,"parent"),record;output_dir=joinpath(dir,"parent","child"))
            volume_root=abspath(dir)
            while dirname(volume_root)!=volume_root;volume_root=dirname(volume_root);end
            @test_throws ArgumentError Hammerhead._checkpoint_disjoint(volume_root,dir)
            @test_throws ArgumentError Hammerhead._checkpoint_disjoint(dir,volume_root)
            @test_throws ArgumentError create_checkpoint(pairs[1][1],record;output_dir=joinpath(dir,"image-alias-out"))
            script=joinpath(dir,"custom.jl");write(script,"identity(image)")
            custom=ExperimentRecord(pairs,PIVRecipe(record.recipe.passes;
                external_preprocess=ScriptReference(script;entrypoint="identity")))
            @test_throws ArgumentError checkpoint_test_create(dir,"custom",custom)
            cp=checkpoint_test_create(dir,"corrupt",record);resume_checkpoint!(cp;cancel=()->true)
            before=readdir(joinpath(cp.path,"attempts"))
            changed=ExperimentRecord(reverse(pairs),record.recipe)
            @test_throws ArgumentError resume_checkpoint!(cp,changed)
            @test readdir(joinpath(cp.path,"attempts"))==before
            saved=read(pairs[1][1]);open(io->write(io,UInt8(0)),pairs[1][1],"a")
            @test_throws ArgumentError resume_checkpoint!(cp)
            @test readdir(joinpath(cp.path,"attempts"))==before
            write(pairs[1][1],saved)
            mismatched=deepcopy(record);mismatched.creation_environment["julia_version"]="0.0.0"
            @test_throws ArgumentError checkpoint_test_create(dir,"environment",mismatched)
            @test !ispath(joinpath(dir,"environment-meta"))
            header_path=joinpath(cp.path,"header.jld2")
            original=read(header_path);header=Hammerhead._checkpoint_read(header_path,"header")
            for (key,value) in (("total_pairs",true),("total_pairs",big(typemax(Int))+1),("recipe_id",repeat("0",64)))
                broken=deepcopy(header);broken[key]=value;checkpoint_test_replace(header_path,"header",broken)
                @test_throws ArgumentError load_checkpoint(cp.path)
            end
            jldopen(header_path,"w") do f
                f["checkpoint_format_version"]=2;f["checkpoint_kind"]="header";f["checkpoint_data"]=header
            end
            @test_throws ArgumentError load_checkpoint(cp.path)
            write(header_path,original)
            @test checkpoint_state(cp).status===:cancelled
            resume_checkpoint!(cp)
            index=checkpoint_results(cp);payload=index.paths[1];original=read(payload)
            open(io->write(io,UInt8(0)),payload,"a")
            @test_throws ArgumentError load_checkpoint(cp.path)
            @test_throws ArgumentError index[1]
            @test_throws ArgumentError resume_checkpoint!(cp)
            write(payload,original)
            commit=joinpath(cp.path,"commits",Hammerhead._checkpoint_descriptor(1))
            original_commit=read(commit);mapping=Hammerhead._checkpoint_read(commit,"commit")
            for (key,value) in (("pair_index",true),("pair_index",big(typemax(Int))+1),("result_file","../outside.jld2"),("input_id",repeat("0",64)))
                broken=deepcopy(mapping);broken[key]=value;checkpoint_test_replace(commit,"commit",broken)
                @test_throws ArgumentError load_checkpoint(cp.path)
            end
            write(commit,original_commit)
            endfile=only(filter(n->endswith(n,"end.jld2"),readdir(joinpath(cp.path,"attempts")))[1:1])
            endpath=joinpath(cp.path,"attempts",endfile);original_end=read(endpath)
            mapping=Hammerhead._checkpoint_read(endpath,"attempt_end");mapping["committed"]=true
            checkpoint_test_replace(endpath,"attempt_end",mapping)
            @test_throws ArgumentError load_checkpoint(cp.path)
            write(endpath,original_end)
            @test checkpoint_state(cp).data_complete
            beginfile=first(filter(n->endswith(n,"begin.jld2"),readdir(joinpath(cp.path,"attempts"))))
            beginpath=joinpath(cp.path,"attempts",beginfile);original_begin=read(beginpath)
            mapping=Hammerhead._checkpoint_read(beginpath,"attempt_begin");delete!(mapping,"attempt_id")
            checkpoint_test_replace(beginpath,"attempt_begin",mapping)
            @test_throws ArgumentError load_checkpoint(cp.path)
            write(beginpath,original_begin)
            actual_begin=last(filter(n->endswith(n,"begin.jld2"),readdir(joinpath(cp.path,"attempts"))))
            actual_path=joinpath(cp.path,"attempts",actual_begin);original_actual=read(actual_path)
            mapping=Hammerhead._checkpoint_read(actual_path,"attempt_begin")
            mapping["input_pairs"][1][1]=joinpath(dir,"wrong-locator.png")
            checkpoint_test_replace(actual_path,"attempt_begin",mapping)
            @test_throws ArgumentError load_checkpoint(cp.path)
            write(actual_path,original_actual)

            locked=checkpoint_test_create(dir,"unknown-lock",record)
            lockdir=joinpath(locked.path,"writer-lock");mkdir(lockdir);write(joinpath(lockdir,"keep"),"preserve")
            @test_throws ArgumentError resume_checkpoint!(locked;recover_interrupted=true)
            @test read(joinpath(lockdir,"keep"),String)=="preserve" && checkpoint_state(locked).committed==0

            staging=joinpath(dir,"rename-stage");final=joinpath(dir,"rename-final")
            write(staging,"new");write(final,"old")
            @test_throws ArgumentError Hammerhead._checkpoint_publish(staging,final)
            @test read(staging,String)=="new" && read(final,String)=="old"
            @test_throws Base.IOError Hammerhead._checkpoint_publish(joinpath(dir,"missing-stage"),joinpath(dir,"absent-final"))
            @test !ispath(joinpath(dir,"absent-final"))
            creator1=Threads.@spawn try checkpoint_test_create(dir,"competing",record) catch err;err end
            creator2=Threads.@spawn try checkpoint_test_create(dir,"competing",record) catch err;err end
            results=[fetch(creator1),fetch(creator2)]
            @test count(r->r isa ExperimentCheckpoint,results)==1
            @test count(r->r isa Exception,results)==1
        end

        @testset "hard process termination and explicit recovery" begin
            child_script=joinpath(dir,"checkpoint-child.jl")
            write(child_script,"""
            using Hammerhead
            Hammerhead.FFTW.set_num_threads(parse(Int,ARGS[5]))
            checkpoint=load_checkpoint(ARGS[1])
            target=Symbol(ARGS[3]); pair=parse(Int,ARGS[4])
            hook=(phase,index)->begin
                if phase===target && index==pair
                    write(ARGS[2],"reached")
                    while true; sleep(0.05); end
                end
            end
            resume_checkpoint!(checkpoint;_phase_hook=hook)
            """)
            for (phase,pair,committed) in ((:lock_directory_created,0,0),(:lock_owner_staging,0,0),
                                          (:during_staging,1,0),(:committed,1,1),(:before_terminal,3,3))
                cp=checkpoint_test_create(dir,"terminated-$phase",record)
                marker=joinpath(dir,"$phase.marker")
                log=joinpath(dir,"$phase.log")
                command=`$(Base.julia_cmd()) --startup-file=no --compiled-modules=yes --project=$(dirname(Base.active_project())) --threads=$(Threads.nthreads()) $child_script $(cp.path) $marker $(String(phase)) $pair $(Int(Hammerhead.FFTW.get_num_threads()))`
                command=Cmd(command;windows_hide=true,ignorestatus=true)
                process=open(log,"w") do io
                    run(pipeline(command;stdout=io,stderr=io);wait=false)
                end
                try
                    waited=timedwait(()->isfile(marker) || process_exited(process),120)
                    @test waited===:ok
                    @test isfile(marker)
                    isfile(marker) || error("checkpoint child exited before barrier: "*read(log,String))
                    kill(process,Base.SIGKILL);wait(process)
                    @test process_exited(process)
                    status=checkpoint_state(cp)
                    @test status.committed==committed && status.writer_lock_present
                    @test status.status==((phase in (:lock_directory_created,:lock_owner_staging)) ? :ready : :unfinished)
                    @test status.data_complete==(committed==3)
                    before=Dict(p=>read(p) for p in checkpoint_results(cp).paths)
                    @test_throws ArgumentError resume_checkpoint!(cp)
                    resumed=resume_checkpoint!(cp;recover_interrupted=true)
                    @test resumed.status===:completed && resumed.start_committed==committed
                    @test checkpoint_state(cp).data_complete && !checkpoint_state(cp).writer_lock_present
                    @test any(n->startswith(n,"recovered-writer-lock-"),readdir(cp.path))
                    @test all(read(p)==bytes for (p,bytes) in before)
                    @test all(checkpoint_test_same(checkpoint_results(cp)[i],expected[i]) for i in 1:3)
                    for name in readdir(joinpath(cp.path,"attempts"))
                        endswith(name,"end.jld2") || continue
                        metadata=Hammerhead._checkpoint_read(joinpath(cp.path,"attempts",name),"attempt_end")
                        metadata["status"]=="interrupted" && (@test metadata["ended_at"]===nothing)
                    end
                finally
                    if !process_exited(process);kill(process,Base.SIGKILL);wait(process);end
                end
            end
        end
    end
end
