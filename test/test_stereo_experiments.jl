using Test, Hammerhead, Random, JLD2, LinearAlgebra, StaticArrays
using FileIO: save
using ImageCore: Gray, N0f8

function sexp_fixture(dir; signed=false)
    sizes=((48,52),(52,56))
    cams=(PinholeCamera([100. 0 15. 0;0 100. 0 0;0 0 1. 100.]),
          PinholeCamera([100. 0 -15. 0;0 100. 0 0;0 0 1. 100.]))
    grid=signed ? DewarpGrid(x=24.5:-.5:1.,y=48.:-1.:1.) : DewarpGrid(x=1.:48.,y=1.:48.)
    dw=map((c,s)->ImageDewarper(c,grid,s),cams,sizes)
    rng=MersenneTwister(9021)
    paths=[joinpath(dir,"camera$role-frame$i.png") for role in 1:2,i in 1:4]
    for role in 1:2
        a=Gray{N0f8}.(rand(rng,sizes[role]...)); b=circshift(a,(1,2))
        for (i,image) in enumerate((a,b,a,b));save(paths[role,i],image);end
    end
    acquisitions=[(paths[1,i],paths[1,i+1],paths[2,i],paths[2,i+1]) for i in (1,3)]
    (;dw,grid,sizes,paths,acquisitions)
end
sexp_passes()=multipass_parameters([24,16];padding=true,uod_enable=false,validation=(),replace_outliers=false)
function sexp_same(a,b)
    all(k->isequal(getfield(a,k),getfield(b,k)),(:x,:y,:z,:u,:v,:w,:mask,:outliers,:uncertainty_u,:uncertainty_v,:uncertainty_w,:scale)) &&
        all(c->all(k->isequal(getfield(getfield(a,c),k),getfield(getfield(b,c),k)),
            (:x,:y,:u,:v,:mask,:outliers,:peak_ratio,:correlation_moment,:uncertainty_u,:uncertainty_v)),(:cam1,:cam2))
end
function sexp_run_hash(run,path)
    ExperimentRun(run.run_id,run.recipe_id,run.input_id,run.started_at,run.finished_at,run.status,
        run.completed_pairs,run.output,Hammerhead._experiment_file_digest(path),run.environment,run.error)
end
function sexp_record_rehash(record)
    provisional=StereoExperimentRecord(record.recipe,record.input_files,record.pairs,record.timing_metadata,
        record.scaling_mode,record.input_id,record.creation_environment,record.runs,record.record_paths,"")
    StereoExperimentRecord(provisional.recipe,provisional.input_files,provisional.pairs,provisional.timing_metadata,
        provisional.scaling_mode,provisional.input_id,provisional.creation_environment,provisional.runs,provisional.record_paths,
        Hammerhead._experiment_digest(Hammerhead._stereo_record_core(provisional)))
end
function sexp_rewrite(path,key,value)
    jldopen(path,"r+") do file
        haskey(file,key) && delete!(file,key)
        file[key]=value
    end
end
function sexp_record_write(path,payload;version=1)
    jldopen(path,"w") do file
        file["stereo_experiment_format_version"]=version;file["stereo_experiment"]=payload
    end
end
function sexp_payload_rehash(data)
    data["recipe_id"]=Hammerhead._experiment_digest(Hammerhead._stereo_recipe_science(data["recipe"]))
    data["recipe_sha256"]=Hammerhead._experiment_digest(data["recipe"])
    if all(p->length(p)==4,data["pairs"])
        data["input_id"]=Hammerhead._experiment_digest(Hammerhead._stereo_input_data(data["input_files"],data["pairs"],data["timing_metadata"],Symbol(data["scaling_mode"])))
    end
    core=Dict{String,Any}(k=>v for (k,v) in data if k ∉ ("runs","record_sha256"))
    data["record_sha256"]=Hammerhead._experiment_digest(core);data
end
function sexp_rebuild_result(r;parameters=r.parameters,cam1=r.cam1)
    StereoPIVResult(r.x,r.y,r.z,r.u,r.v,r.w,r.uncertainty_u,r.uncertainty_v,
        r.uncertainty_w,r.outliers,r.mask,cam1,r.cam2,parameters,r.scale)
end

@testset "Frozen fitted stereo recipe snapshots" begin
    P=[100. 0 15. 0;0 100. 0 0;.12 .27 .61 100.]
    camera=PinholeCamera(P)
    decoded=Hammerhead._stereo_camera_decode(Hammerhead._stereo_camera_data(camera))
    @test isequal(camera.P,decoded.P)
    @test isequal(PinholeCamera(P).P,camera.P) # public normalization remains unchanged
    ax=zeros(19);ay=zeros(19);ax[2]=1.;ax[4]=.15;ay[3]=1.
    soloff=SoloffCamera(SVector{19}(ax),SVector{19}(ay),SVector(0.,0.,0.),SVector(1.,1.,1.))
    transformed=TransformedCamera(soloff,Matrix{Float64}(I,3,3),[0.,0.,1.])
    for c in (soloff,transformed)
        d=Hammerhead._stereo_camera_data(c)
        @test Hammerhead._stereo_camera_data(Hammerhead._stereo_camera_decode(d))==d
    end
    nested=Hammerhead._stereo_camera_data(transformed);nested["camera"]=deepcopy(nested)
    @test_throws ArgumentError Hammerhead._stereo_camera_decode(nested)
    for change in (d->d["P"][1]=NaN,d->d["P"]=ones(2,4),d->d["P"][3,1:3].=0.)
        d=Hammerhead._stereo_camera_data(camera);change(d)
        @test_throws ArgumentError Hammerhead._stereo_camera_decode(d)
    end
    d=Hammerhead._stereo_camera_data(soloff);d["scale"][1]=0.
    @test_throws ArgumentError Hammerhead._stereo_camera_decode(d)
    mktempdir() do dir
        f=sexp_fixture(dir;signed=true);mask=falses(48,48);mask[:,1:5].=true
        steps=([PreprocessStep(:subtract_background;background=fill(.01f0,f.sizes[1]...))],PreprocessStep[])
        recipe=StereoPIVRecipe(sexp_passes(),f.dw...;mask,preprocessing=steps,roi=ROI(5:44,7:42),world_unit="mm",coordinate_frame="fitted-sheet")
        id=recipe_identity(recipe)
        mask[1,1]=false;steps[1][1].options["background"][1]=1
        @test recipe._data["mask"][1,1]
        @test recipe._data["preprocessing"][1][1]["options"]["background"][1]==.01f0
        @test recipe_identity(recipe)==id
        note1=StereoPIVRecipe(sexp_passes(),f.dw...;calibration_note="first supplied fit")
        report=SelfCalibrationReport([Hammerhead.SelfCalPass(.1,.08,4,NaN,nothing)],true,.2,SMatrix{3,3,Float64}(I),SVector(0.,0.,0.),PIVResult[])
        note2=StereoPIVRecipe(sexp_passes(),f.dw...;calibration_note="second supplied fit",self_calibration_report=report)
        @test recipe_identity(note1)==recipe_identity(note2)
        @test note1._sha256!=note2._sha256
        noted1=StereoExperimentRecord(f.acquisitions,note1)
        noted2=StereoExperimentRecord(f.acquisitions,note2)
        @test noted1.input_id==noted2.input_id && noted1._sha256!=noted2._sha256
        @test note2._data["calibration_provenance"]["self_calibration"]["disparity_maps"]=="not_embedded"
        outside=DewarpGrid(x=-10.:37.,y=-10.:37.)
        outside_dw=map(d->ImageDewarper(d.cam,outside,d.image_size),f.dw)
        @test any(outside_dw[1].mask)
        @test length(recipe_identity(StereoPIVRecipe(sexp_passes(),outside_dw...)))==64
        changed=deepcopy(f.dw[1]);changed.rows[1]+=1.
        @test_throws ArgumentError StereoPIVRecipe(sexp_passes(),changed,f.dw[2])
        for kwargs in ((;image_type=Float16),(;backend=:cuda),(;mask=ones(2,2)),(;roi=ROI(1:49,1:48)),
                       (;preprocessing=([PreprocessStep(:subtract_background;background=zeros(2,2))],PreprocessStep[])))
            @test_throws ArgumentError StereoPIVRecipe(sexp_passes(),f.dw...;kwargs...)
        end
        edited=deepcopy(recipe);edited._data["grid"]["z"]=1.
        @test_throws ArgumentError recipe_identity(edited)
        # NaN/out-of-view masks and primitive Soloff/rigid-wrapper reconstruction
        # are exercised by full replay, not just descriptor parsing.
        soloffs=(soloff,SoloffCamera(SVector{19}(ax .- [i==4 ? .3 : 0. for i in 1:19]),SVector{19}(ay),SVector(0.,0.,0.),SVector(1.,1.,1.)))
        models=(TransformedCamera(soloffs[1],Matrix{Float64}(I,3,3),[0.,0.,1.]),soloffs[2])
        grid=DewarpGrid(x=1.:48.,y=1.:48.)
        dewarpers=map((c,s)->ImageDewarper(c,grid,s),models,f.sizes)
        r=StereoPIVRecipe(sexp_passes(),dewarpers...)
        rec=StereoExperimentRecord([f.acquisitions[1]],r)
        artifact=joinpath(dir,"polynomial-rigid.jld2")
        run=replay_experiment(rec;output=artifact)
        expected=run_piv_stereo_sequence([f.acquisitions[1]],dewarpers...,sexp_passes();threaded=false,progress=false)
        @test sexp_same(only(load_results(artifact)),only(expected))
        @test verify_stereo_experiment_run(rec,run;verify_results=true)["measurement_fields_checked_acquisitions"]==1
    end
end

@testset "Stereo experiment records and strict replay" begin
    mktempdir() do dir
        f=sexp_fixture(dir);passes=sexp_passes()
        scale=PhysicalScale(pixel_size=.02,dt=9.,length_unit="mm",time_unit="ns")
        epoch=big(2)^75
        stamps=BigInt[epoch,epoch+2,epoch+5,epoch+8]
        ids=["A1","B1","A2","B2"]
        options=(;timestamps=(stamps,stamps.+1),time_units=("ns","ns"),clock_ids=("capture","capture"),source_ids=("opaque1","opaque2"),frame_ids=(ids,ids))
        recipe=StereoPIVRecipe(passes,f.dw...;scale,sync_atol=1.,world_unit="fitted-world-unit",coordinate_frame="sheet")
        pairs1=[FramePair(a[1],a[2],dt) for (a,dt) in zip(f.acquisitions,(2.,3.))]
        pairs2=[FramePair(a[3],a[4],dt) for (a,dt) in zip(f.acquisitions,(2.,3.))]
        record=StereoExperimentRecord(pairs1,pairs2,recipe;options...)
        tuple_record=StereoExperimentRecord(f.acquisitions,recipe;options...)
        @test record.scaling_mode===:pair_list && tuple_record.scaling_mode===:four_tuple
        @test record.input_id!=tuple_record.input_id
        @test length(record.input_files)==8
        @test record.timing_metadata[1]["frame_index_scope"]=="provided_ordered_file_stream"
        @test Hammerhead._stereo_record_timing(record,Hammerhead._stereo_recipe_config(recipe._data))[1]["scale"]["dt"]==2.
        @test Hammerhead._stereo_record_timing(tuple_record,Hammerhead._stereo_recipe_config(recipe._data))[1]["scale"]["dt"]==9.
        stamps[1]=0;ids[1]="edited"
        @test Hammerhead._timing_decode(record.timing_metadata[1]["timestamps"][1])==epoch
        @test record.timing_metadata[1]["frame_ids"][1]=="A1"
        stored=joinpath(dir,"stereo-experiment.jld2");save_experiment(stored,record)
        reopened=load_stereo_experiment(stored)
        @test reopened._sha256==record._sha256 && reopened.input_id==record.input_id
        @test recipe_identity(reopened.recipe)==recipe_identity(recipe)
        @test_throws ArgumentError load_experiment(stored)
        @test_throws ArgumentError load_results(stored)
        bytes=read(stored)
        @test_throws ArgumentError replay_experiment(reopened;output=stored)
        @test read(stored)==bytes
        inputbytes=read(f.paths[1,1])
        @test_throws ArgumentError save_experiment(f.paths[1,1],record)
        @test_throws ArgumentError replay_experiment(record;output=f.paths[1,1])
        @test read(f.paths[1,1])==inputbytes
        # Content identities ignore duplicate paths and opaque labels, retaining ordered roles.
        reused=StereoExperimentRecord([f.acquisitions[1],f.acquisitions[1]],recipe)
        plain=StereoExperimentRecord(f.acquisitions,recipe)
        @test reused.input_id==plain.input_id && length(reused.input_files)==4
        swapped=StereoExperimentRecord([(a[2],a[1],a[3],a[4]) for a in f.acquisitions],recipe)
        @test swapped.input_id!=plain.input_id
        relocated=joinpath(dir,"copies");mkpath(relocated)
        copies=map(f.acquisitions) do a
            map(a) do p
                dst=joinpath(relocated,basename(p));cp(p,dst);dst
            end
        end
        @test StereoExperimentRecord(copies,recipe).input_id==plain.input_id
        for kwargs in ((;timestamps=([0,2,5,5],[0,2,5,8])),(;time_units=("s","ns")),(;clock_ids=("one","two")),
                       (;timestamps=(Any[0,2,true,8],[0,2,5,8])))
            @test_throws ArgumentError StereoExperimentRecord(f.acquisitions,recipe;kwargs...)
        end
        @test_throws ArgumentError StereoExperimentRecord([FramePair(a[1],a[2],1.) for a in f.acquisitions],pairs2,recipe;
            timestamps=(BigInt[epoch,epoch+2,epoch+5,epoch+8],BigInt[epoch+1,epoch+3,epoch+6,epoch+9]),time_units=("ns","ns"))
        payload=JLD2.load(stored,"stereo_experiment")
        bad=joinpath(dir,"bad-record.jld2")
        for change in (d->d["recipe"]["cameras"][1]["P"]=zeros(2,4),
                       d->d["recipe"]["processing"]["threaded"]=0,
                       d->d["timing_metadata"][1]["camera_role"]=true,
                       d->d["timing_metadata"][2]["clock_id"]="inconsistent",
                       d->d["pairs"][1]=[1,2,3],d->d["recipe"]["grid"]["x"]["count"]=true)
            d=deepcopy(payload);change(d);sexp_payload_rehash(d);sexp_record_write(bad,d)
            @test_throws ArgumentError load_stereo_experiment(bad)
        end
        for version in (true,0,2)
            sexp_record_write(bad,payload;version)
            @test_throws ArgumentError load_stereo_experiment(bad)
        end

        @testset "CPU and KA Float32/64 numerical parity, source and settings verification" begin
            for T in (Float32,Float64),backend in (:cpu,:ka)
                signed=sexp_fixture(mktempdir(dir);signed=true)
                steps=([PreprocessStep(:subtract_background;background=fill(T(.01),signed.sizes[1]...))],
                       [PreprocessStep(:intensity_cap;n_sigma=2.)])
                mask=falses(48,48);mask[:,1:4].=true
                r=StereoPIVRecipe(passes,signed.dw...;scale,backend,image_type=T,preprocessing=steps,mask,roi=ROI(5:44,7:42),predictor_smoothing=false)
                rec=StereoExperimentRecord(signed.acquisitions,r)
                out=joinpath(dir,"$backend-$T.jld2")
                expected=run_piv_stereo_sequence(signed.acquisitions,signed.dw...,passes;
                    backend,image_type=T,scale,mask,roi=ROI(5:44,7:42),predictor_smoothing=false,threaded=false,progress=false,
                    preprocess=(img->subtract_background(img,fill(T(.01),signed.sizes[1]...)),img->intensity_cap(img;n_sigma=2.)))
                run=replay_experiment(rec;output=out)
                @test run.status===:completed && run.completed_pairs==2
                actual=load_results(out)
                @test all(sexp_same.(expected,actual))
                @test actual[1] isa StereoPIVResult{T}
                checked=verify_stereo_experiment_run(rec,run;verify_results=true,verify_inputs=true)
                @test checked["measurement_fields_checked_acquisitions"]==2 && checked["unverified_trailing_entries"]==0
                @test !checked["piv_recomputed"] && !checked["calibration_accuracy_verified"]
            end
        end

        output=joinpath(dir,"result.jld2")
        run=replay_experiment(reopened;output,run_record=stored,record_diagnostics=true,record_pair_timing=true)
        @test only(load_stereo_experiment(stored).runs).run_id==run.run_id
        verified=verify_stereo_experiment_run(reopened,run;verify_results=true)
        @test verified["measurement_fields_checked_acquisitions"]==2
        index=ResultFile(output)
        p=load_stereo_pair_timing(index,1)
        @test pair_timing_data(p)["cameras"][1]["frames"][1]["frame_id"]=="A1"
        @test execution_diagnostics_data(load_stereo_execution_diagnostics(index,1))["pair_index"]==1
        original=read(output)
        raw=index[1];badnative=joinpath(dir,"bad-native.jld2")
        write(badnative,original)
        labels=JLD2.load(badnative,Hammerhead.source_key(1));reverse!(labels)
        sexp_rewrite(badnative,Hammerhead.source_key(1),labels)
        @test verify_stereo_experiment_run(reopened,sexp_run_hash(run,badnative);output=badnative)["output_integrity_checked"]
        @test_throws ArgumentError verify_stereo_experiment_run(reopened,sexp_run_hash(run,badnative);output=badnative,verify_results=true)
        write(badnative,original)
        wrong=sexp_rebuild_result(raw;parameters=PIVParameters(window_size=16,overlap=8,padding=false))
        sexp_rewrite(badnative,Hammerhead.result_key(1),wrong)
        @test_throws ArgumentError verify_stereo_experiment_run(reopened,sexp_run_hash(run,badnative);output=badnative,verify_results=true)
        # A deliberately recomputed measurement hash does not bypass recipe grid checks.
        write(badnative,original);wrong=deepcopy(raw);wrong.cam1.x[1]+=1;wrong.cam2.x[1]+=1
        sexp_rewrite(badnative,Hammerhead.result_key(1),wrong)
        association=JLD2.load(badnative,"stereo_experiment_run")
        association["association"]["entry_measurement_sha256"][1]=Hammerhead._stereo_execution_measurement_digest(wrong)
        association["association_sha256"]=Hammerhead._experiment_digest(association["association"])
        sexp_rewrite(badnative,"stereo_experiment_run",association)
        resealed=sexp_run_hash(run,badnative)
        @test verify_stereo_experiment_run(reopened,resealed;output=badnative)["run_association_checked"]
        @test_throws ArgumentError verify_stereo_experiment_run(reopened,resealed;output=badnative,verify_results=true)
        for mutate in (d->d["diagnostics_recorded"]=1,d->d["completed_pairs"]=true,
                       d->d["actual_map_sha256"]=String["bad","bad"],d->d["binding_basis"]="all_serialized_content")
            write(badnative,original);a=JLD2.load(badnative,"stereo_experiment_run")
            mutate(a["association"]);a["association_sha256"]=Hammerhead._experiment_digest(a["association"])
            sexp_rewrite(badnative,"stereo_experiment_run",a)
            @test_throws ArgumentError verify_stereo_experiment_run(reopened,sexp_run_hash(run,badnative);output=badnative)
        end
        write(badnative,original);sexp_rewrite(badnative,"pair_timing_format_version",1)
        @test_throws ArgumentError verify_stereo_experiment_run(reopened,sexp_run_hash(run,badnative);output=badnative)
        write(badnative,original)
        sexp_rewrite(badnative,Hammerhead.result_key(3),raw)
        @test_throws ArgumentError verify_stereo_experiment_run(reopened,sexp_run_hash(run,badnative);output=badnative)
        write(badnative,original);sexp_rewrite(badnative,"stereo_pair_timing_format_version",true)
        @test_throws ArgumentError verify_stereo_experiment_run(reopened,sexp_run_hash(run,badnative);output=badnative)
        bytes=read(output)
        @test_throws ArgumentError save_experiment(output,load_stereo_experiment(stored))
        @test read(output)==bytes
        modified=deepcopy(reopened);modified.creation_environment["julia_threads"]+=1
        modified=sexp_record_rehash(modified)
        sentinel=joinpath(dir,"sentinel.jld2");write(sentinel,"preserve")
        @test_throws ArgumentError replay_experiment(modified;output=sentinel)
        @test read(sentinel,String)=="preserve"
        override=replay_experiment(modified;output=sentinel,allow_environment_change=true)
        @test override.environment["julia_threads"]==Threads.nthreads()
        @test verify_stereo_experiment_run(modified,override;verify_results=true)["measurement_fields_checked"]

        @testset "Failed runs retain only verified completed prefixes" begin
            failpath=joinpath(dir,"failed.jld2");failrecord=joinpath(dir,"failed-record.jld2")
            @test_throws ErrorException replay_experiment(plain;output=failpath,run_record=failrecord,progress=(i,n)->error("stop after publication"))
            failed=only(load_stereo_experiment(failrecord).runs)
            @test failed.status===:failed && failed.completed_pairs==1 && occursin("stop after publication",failed.error)
            status=verify_stereo_experiment_run(plain,failed;verify_results=true)
            @test status["measurement_fields_checked_acquisitions"]==1 && status["unverified_trailing_entries"]==0
            plainbytes=read(failpath)
            swapped=JLD2.load(failpath,Hammerhead.source_key(1));swapped[1],swapped[2]=swapped[2],swapped[1]
            sexp_rewrite(failpath,Hammerhead.source_key(1),swapped)
            @test_throws ArgumentError verify_stereo_experiment_run(plain,sexp_run_hash(failed,failpath);verify_results=true)
            write(failpath,plainbytes)
            sexp_rewrite(failpath,Hammerhead.result_key(2),raw)
            withtail=sexp_run_hash(failed,failpath)
            status=verify_stereo_experiment_run(plain,withtail;verify_results=true)
            @test status["unverified_trailing_entries"]==1 && status["measurement_fields_checked_acquisitions"]==1
            sexp_rewrite(failpath,Hammerhead.result_key(3),raw)
            @test_throws ArgumentError verify_stereo_experiment_run(plain,sexp_run_hash(failed,failpath))
            callbackpath=joinpath(dir,"callback-failed.jld2");callbackrecord=joinpath(dir,"callback-record.jld2")
            @test_throws ErrorException replay_experiment(plain;output=callbackpath,run_record=callbackrecord,on_pair_timing=(i,p)->error("callback failed before write"))
            callbackrun=only(load_stereo_experiment(callbackrecord).runs)
            @test callbackrun.completed_pairs==0
            @test verify_stereo_experiment_run(plain,callbackrun;verify_results=true)["measurement_fields_checked_acquisitions"]==0
            sexp_rewrite(callbackpath,"stereo_execution_diagnostics_format_version",2)
            @test_throws ArgumentError verify_stereo_experiment_run(plain,sexp_run_hash(callbackrun,callbackpath))
        end
        @testset "Preflight source integrity and replay snapshots" begin
            recorded=StereoExperimentRecord([f.acquisitions[1]],recipe)
            sentinel=joinpath(dir,"protected.jld2");write(sentinel,"not opened")
            original_input=read(f.paths[1,1]);save(f.paths[1,1],Gray{N0f8}.(zeros(48,52)))
            @test_throws ArgumentError replay_experiment(recorded;output=sentinel)
            @test read(sentinel,String)=="not opened"
            write(f.paths[1,1],original_input)
            external=deepcopy(plain);snapshot_output=joinpath(dir,"snapshot.jld2")
            snaprun=replay_experiment(external;output=snapshot_output,on_pair_timing=(i,p)->begin
                external.input_files[1]["path"]="caller edited descriptor"
                external.recipe._data["grid"]["z"]=10.
            end)
            @test snaprun.status===:completed
            @test verify_stereo_experiment_run(plain,snaprun;verify_results=true)["measurement_fields_checked_acquisitions"]==2
            @test_throws ArgumentError replay_experiment(external;output=sentinel)
            @test read(sentinel,String)=="not opened"
        end
    end
end
