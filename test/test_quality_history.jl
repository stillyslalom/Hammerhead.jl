using Test, Hammerhead, JLD2, TOML

function report_history_fixture(;restored=false,unknown=false)
    p=PIVParameters(window_size=16,overlap=8,n_peaks=3,uod_enable=false,
        replace_outliers=true,validation=(VelocityMagnitudeValidator(0,1),),uncertainty=true)
    u=zeros(3,3);u[2,2]=20;u[1,2]=20;u[3,3]=NaN
    v=zeros(3,3);v[3,3]=NaN
    mask=falses(3,3);mask[3,3]=true
    ratio=fill(2.,3,3);ratio[3,3]=NaN
    r=PIVResult([10.,20.,30.],[10.,20.,30.],u,v,ratio,zeros(3,3),fill(.2,3,3),fill(.3,3,3),
        falses(3,3),mask,p)
    observation=Hammerhead._HistoryObservation(Float64,(3,3),p)
    Hammerhead._history_begin!(observation,r.u,r.v,r.u,r.v,1)
    au=fill(NaN,3,3,2);av=copy(au);au[2,2,1]=0;av[2,2,1]=0
    Hammerhead.validate_and_replace!(r,p,true;history=observation,alternatives=(au,av))
    if restored
        r.u[r.outliers]=observation.primary_u[r.outliers];r.v[r.outliers]=observation.primary_v[r.outliers]
        observation.primary_restored .= r.outliers
    end
    unknown && (observation.stages[3]["name"]="unrecognized stored label")
    h=Hammerhead._history_finish(observation,r,:cpu,(64,64),1)
    r,h
end
function write_report_history(path,entries,histories)
    jldopen(path,"w") do f
        f["format_version"]=1
        for i in eachindex(entries)
            key="results/"*lpad(string(i*7),6,'0')
            f[key]=entries[i]
            histories[i]===nothing || Hammerhead._write_measurement_history(f,key,histories[i])
        end
    end
end

@testset "Verified recorded-history quality reports" begin
    r,h=report_history_fixture()
    @test verify_measurement_history(h,r)===true
    node=measurement_history_at(h,CartesianIndex(2,2))
    @test node.x==20 && node.y==20 && node.primary_u==20 && node.accepted_peak_rank==2
    @test node.final_origin=="alternative" && node.uncertainty_u_status=="finite_nonnegative"
    @test node.rejection_name=="validation[1]:VelocityMagnitudeValidator"
    @test !any(v->v isa AbstractArray,values(node))
    @test_throws BoundsError measurement_history_at(h,CartesianIndex(4,1))
    scaled=with_scale(r,PhysicalScale(pixel_size=.2,dt=.5))
    @test_throws ArgumentError verify_measurement_history(h,scaled)
    @test_throws ArgumentError verify_measurement_history(h,physical(scaled))
    original=deepcopy(r);original.u[1]=99
    @test_throws ArgumentError verify_measurement_history(h,original)
    small=PIVResult([1.],[1.],zeros(1,1),zeros(1,1),ones(1,1),zeros(1,1),fill(NaN,1,1),fill(NaN,1,1),falses(1,1),falses(1,1),r.parameters)
    stereo=StereoPIVResult(small.x,small.y,0.,small.u,small.v,small.u,small.uncertainty_u,small.uncertainty_v,small.uncertainty_u,small.outliers,small.mask,small,small,small.parameters)
    particles=Particles([1.],[1.],[1.],[1.])
    ptv=PTVResult([1.],[1.],[0.],[0.],[0.],falses(1),[1],[1],particles,particles,PTVParameters())
    tracking=TrackingResult(Trajectory{Float64}[],1,PTVParameters())
    mktempdir() do dir
        path=joinpath(dir,"mixed.jld2")
        write_report_history(path,[r,small,stereo,ptv,tracking],[h,nothing,nothing,nothing,nothing])
        report=quality_report(ResultFile(path);include_measurement_history=true)
        data=quality_report_data(report);c=data["measurement_history"]["counts"]
        @test data["quality_report_format_version"]==2
        @test data["entry_kinds"]==Dict("planar"=>2,"stereo"=>1,"ptv"=>1,"tracking"=>1)
        @test c["recorded_entries"]==1 && c["missing_entries"]==1 && c["planar_entries"]==2
        @test c["nodes"]==9 && c["masked"]==1 && c["unmasked"]==8
        @test c["pre_substitution_flagged"]==2 && c["first_rejected"]==2 && c["rejection_configured_builtin"]==2
        @test c["accepted_alternative"]==1 && c["fill_attempted"]==1 && c["fill_assigned"]==1 && c["primary_restored"]==0
        @test c["origin_primary"]==6 && c["origin_alternative"]==1 && c["origin_fill"]==1
        @test c["unsupported_stereo_entries"]==c["unsupported_ptv_entries"]==c["unsupported_tracking_entries"]==1
        f=data["measurement_history"]["fractions"]
        @test f["recorded_entry_fraction"]["value"]==1/2
        @test f["fill_assigned_fraction"]["numerator"]==1 && f["fill_assigned_fraction"]["denominator"]==8
        @test f["fill_assigned_fraction"]["value"]==1/8 # missing plain node is not in this denominator
        @test data["groups"]["planar"]["counts"]["nodes"]==10
        @test Set(keys(data["groups"]))==Set(["planar","stereo"])
        @test data["unavailable"]["uncertainty_measurement_association"]["reason_code"]=="applicability_not_established"
        @test data["unavailable"]["earlier_pass_and_sweep_history"]["available"]===false
        @test !haskey(data["unavailable"],"replacement_history")
        @test data["provenance"]["source_sha256"]==Hammerhead._experiment_file_digest(path)
        @test_throws ArgumentError quality_report(ResultFile(path)) # legacy rejects PTV/tracking
        @test_throws ArgumentError quality_report([r];include_measurement_history=true)
        @test_throws ArgumentError quality_report(view(ResultFile(path),1:2);include_measurement_history=true)
        @test_throws ArgumentError save_quality_report(path,report)
        dest=joinpath(dir,"quality.toml");save_quality_report(dest,report)
        @test quality_report_data(load_quality_report(dest))==data
        @test occursin("Coverage: 1 / 2 planar entries; 1 missing",sprint(show,MIME"text/plain"(),report))
        @test occursin("history-covered unmasked nodes",sprint(show,MIME"text/plain"(),report))
        @test !any(v->v isa AbstractMatrix,values(data))

        rr,hh=report_history_fixture(;restored=true,unknown=true)
        write_report_history(path,[rr],[hh])
        restored_data=quality_report_data(quality_report(ResultFile(path);include_measurement_history=true))
        counts=restored_data["measurement_history"]["counts"]
        @test counts["rejection_unclassified"]==2 && counts["rejection_configured_builtin"]==0
        @test counts["fill_assigned"]==1 && counts["primary_restored"]==1 && counts["origin_fill"]==0 && counts["origin_primary"]==7
        integer_fraction=deepcopy(restored_data)
        integer_fraction["measurement_history"]["fractions"]["recorded_entry_fraction"]["value"]=1
        @test_throws ArgumentError Hammerhead._quality_wrap(integer_fraction)
        # Unknown labels stay unclassified even with apparently suggestive substrings.
        @test Hammerhead._quality_rejection_bucket(Dict("name"=>"almost_implicit_uod","kind"=>"built_in"))=="unclassified"
        changed=deepcopy(rr);changed.v[1]=10
        write_report_history(path,[changed],[hh])
        @test_throws ArgumentError quality_report(ResultFile(path);include_measurement_history=true)
        write_report_history(path,[ptv],[h])
        @test_throws ArgumentError quality_report(ResultFile(path);include_measurement_history=true) # refuse existing history on unsupported kind
        write_report_history(path,[small,ptv],[nothing,nothing])
        absent=quality_report_data(quality_report(ResultFile(path);include_measurement_history=true))
        @test absent["measurement_history"]["counts"]["missing_entries"]==1
        @test !absent["measurement_history"]["fractions"]["fill_assigned_fraction"]["available"]
        @test !haskey(absent["measurement_history"]["fractions"]["fill_assigned_fraction"],"value")
        save_results(path,PIVResult[])
        empty=quality_report_data(quality_report(ResultFile(path);include_measurement_history=true))
        @test isempty(empty["groups"]) && all(==(0),values(empty["entry_kinds"]))
        @test !empty["measurement_history"]["fractions"]["recorded_entry_fraction"]["available"]
        jldopen(path,"a+") do f;f["measurement_history_format_version"]=true;end
        @test_throws ArgumentError quality_report(ResultFile(path);include_measurement_history=true)

        plain=joinpath(dir,"plain.jld2");save_results(plain,small)
        v1=quality_report_data(quality_report(ResultFile(plain)))
        @test v1["quality_report_format_version"]==1 && !haskey(v1,"measurement_history")
        @test v1["unavailable"]["replacement_history"]["reason_code"]=="not_persisted"
        save_quality_report(dest,Hammerhead._quality_wrap(v1))
        @test quality_report_data(load_quality_report(dest))==v1
        for change in (d->d["quality_report_format_version"]=true,
                       d->begin
                           d["entry_kinds"]=Dict{String,Any}(d["entry_kinds"])
                           d["entry_kinds"]["ptv"]=true
                       end,
                       d->d["measurement_history"]["counts"]["recorded_entries"]=99,
                       d->d["measurement_history"]["counts"]["fill_assigned"]=9,
                       d->d["measurement_history"]["counts"]["rejection_unclassified"]=1,
                       d->d["measurement_history"]["counts"]["origin_fill"]=typemax(Int),
                       d->d["measurement_history"]["fractions"]["fill_assigned_fraction"]["denominator"]=9)
            candidate=deepcopy(data);change(candidate)
            @test_throws ArgumentError Hammerhead._quality_wrap(candidate)
        end
        # Associated mode verifies the same native bytes/recipe/input IDs and
        # refuses later changes; flags alone never manufacture history events.
        fixtures=joinpath(pkgdir(Hammerhead),"test","reference_images","A")
        files=sort(filter(f->endswith(lowercase(f),".tif"),readdir(fixtures;join=true)))
        recipe=PIVRecipe(PIVParameters(window_size=16,overlap=8);roi=ROI(1:48,1:48),image_type=Float32)
        record=ExperimentRecord([(files[1],files[2])],recipe)
        output=joinpath(dir,"replay.jld2")
        run=replay_experiment(record;output,record_measurement_history=true)
        associated=quality_report_data(quality_report(record,run;include_measurement_history=true))
        @test associated["provenance"]["association"]=="recorded_output_verified" && associated["measurement_history"]["counts"]["recorded_entries"]==1
        @test associated["provenance"]["recipe_id"]==recipe_identity(recipe)
        good_result=ResultFile(output)[1]
        good_history=measurement_history_data(load_measurement_history(output))
        for change in (d->d["association"]=nothing,
                       d->d["association"]["recipe_id"]="0"^64,
                       d->d["association"]["input_id"]="0"^64,
                       d->d["pair_index"]=2)
            candidate=deepcopy(good_history);change(candidate)
            modified=PIVMeasurementHistory(candidate,Hammerhead._history_digest(candidate))
            write_report_history(output,[good_result],[modified])
            run_data=Hammerhead._experiment_run_data(run)
            run_data["output_sha256"]=Hammerhead._experiment_file_digest(output)
            current_run=Hammerhead._experiment_run(run_data,record)
            @test_throws ArgumentError quality_report(record,current_run;include_measurement_history=true)
            generic=quality_report_data(quality_report(ResultFile(output);include_measurement_history=true))
            @test generic["provenance"]["association"]=="unassociated"
        end
        save_results(output,small)
        @test_throws ArgumentError quality_report(record,run;include_measurement_history=true)
    end
end
