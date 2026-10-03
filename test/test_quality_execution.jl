using Test, Hammerhead, Random, JLD2, TOML

function quality_execution_planar(;T=Float64,flat=false,history=false,kwargs...)
    a=flat ? zeros(T,48,48) : rand(MersenneTwister(370),T,48,48)
    p=PIVParameters(window_size=16,overlap=8,padding=true,uod_enable=false,
        validation=(),replace_outliers=false;kwargs...)
    d=Ref{Any}(nothing);h=Ref{Any}(nothing)
    result=run_piv(a,circshift(a,(1,2)),p;threaded=false,on_diagnostics=x->(d[]=x),
        on_measurement_history=history ? x->(h[]=x) : nothing)
    result,d[],h[]
end

function quality_execution_stereo(;T=Float64,flat=false,grid=DewarpGrid(x=1.:48.,y=1.:48.),roi=nothing,scale=nothing)
    cameras=(PinholeCamera([100. 0 15. 0;0 100. 0 0;0 0 1. 100.]),
        PinholeCamera([100. 0 -15. 0;0 100. 0 0;0 0 1. 100.]))
    dw1,dw2=map(c->ImageDewarper(c,grid,(48,48)),cameras)
    a=flat ? zeros(T,48,48) : rand(MersenneTwister(371),T,48,48)
    b=circshift(a,(1,2))
    schedule=multipass_parameters([24,16];padding=true,uod_enable=false,validation=(),replace_outliers=false,
        final=(;max_iterations=4,convergence_tol=1e6))
    d=Ref{Any}(nothing)
    result=run_piv_stereo(a,b,a,circshift(a,(-1,1)),dw1,dw2,schedule;threaded=false,roi,scale,on_diagnostics=x->(d[]=x))
    result,d[]
end
function quality_execution_native(path,results;planar=fill(nothing,length(results)),stereo=fill(nothing,length(results)),history=fill(nothing,length(results)))
    jldopen(path,"w") do file
        file["format_version"]=1
        for i in eachindex(results)
            key="results/"*lpad(string(i*7),6,'0')
            file[key]=results[i]
            planar[i]===nothing || Hammerhead._write_execution_diagnostics(file,key,planar[i])
            stereo[i]===nothing || Hammerhead._write_stereo_execution_diagnostics(file,key,stereo[i],results[i])
            history[i]===nothing || Hammerhead._write_measurement_history(file,key,history[i])
        end
    end
    path
end
quality_execution_data(path;kwargs...)=quality_report_data(quality_report(ResultFile(path);include_execution_diagnostics=true,kwargs...))
function quality_execution_only_metadata(x)
    x isa Union{PIVResult,StereoPIVResult,PTVResult,TrackingResult,PIVRecipe} && return false
    x isa AbstractDict && return all(quality_execution_only_metadata,values(x))
    x isa AbstractArray && return ndims(x)==1 && all(quality_execution_only_metadata,x)
    x isa Tuple && return all(quality_execution_only_metadata,x)
    x isa Union{String,Real,Bool,Nothing}
end
@noinline function quality_execution_supplied_lifetime(packet,raw)
    supplied=deepcopy(raw)
    weak=WeakRef(supplied.u)
    data=execution_diagnostics_data(packet;result=supplied)
    weak,data
end
function quality_execution_rewrite_entry(path,key,mutate;stereo=false)
    prefix=stereo ? "stereo_execution_diagnostics/" : "execution_diagnostics/"
    entrykey=prefix*key
    jldopen(path,"r+") do file
        entry=file[entrykey];mutate(entry)
        stereo && (entry["diagnostics_sha256"]=Hammerhead._history_digest(entry["diagnostics"]))
        delete!(file,entrykey);file[entrykey]=entry
    end
end

@testset "Execution report strict schema and arithmetic" begin
    planar,pdiag,_=quality_execution_planar(max_iterations=2)
    mktempdir() do dir
        path=quality_execution_native(joinpath(dir,"source.jld2"),[planar];planar=[pdiag])
        data=quality_execution_data(path)
        function bad_report(change)
            # Match parsed TOML containers so assigning Bool cannot silently
            # convert it to Int in a typed in-memory counter dictionary.
            candidate=TOML.parse(sprint(io->TOML.print(io,data)));change(candidate)
            for group in values(candidate["execution_diagnostics"]["groups"])
                group["fractions"]=Hammerhead._quality_execution_fractions(group["counts"])
            end
            candidate
        end
        pc(d)=d["execution_diagnostics"]["groups"]["planar"]["counts"]
        mutations=(d->d["quality_report_format_version"]=true,
            d->d["execution_diagnostics"]["verification_time"]="fresh_load_verified",
            d->d["execution_diagnostics"]["groups"]["cam1"]["residual_unit"]="mm",
            d->d["execution_diagnostics"]["groups"]["cam2"]["binding"]="full_result_verified",
            d->d["execution_diagnostics"]["unsupported_entries"]["ptv"]=false,
            d->pc(d)["requested_sweeps"]=true,
            d->pc(d)["missing_entries"]=-1,
            d->begin pc(d)["recorded_entries"]=typemax(Int);pc(d)["missing_entries"]=1 end,
            d->begin
                c=pc(d);c["stop_single_sweep"]=1;c["stop_iteration_budget"]=0
            end,
            d->begin
                c=pc(d);c["requested_sweeps"]=c["executed_sweeps"]=4;c["tolerance_checks"]=2
                c["last_checks_present"]=1;c["last_checks_absent"]=0;c["last_check_infinite"]=1
            end,
            d->begin
                c=pc(d);c["requested_sweeps"]=c["executed_sweeps"]=4;c["tolerance_checks"]=2
                c["last_checks_present"]=1;c["last_checks_absent"]=0;c["last_check_nan"]=1
            end,
            d->begin
                c=pc(d);c["requested_sweeps"]=c["executed_sweeps"]=4;c["tolerance_checks"]=2
                c["last_checks_present"]=1;c["last_checks_absent"]=0;c["last_check_finite"]=c["last_check_empty"]=1
                c["last_check_included"]=c["last_check_finite_support"]=1
            end,
            d->begin
                c=pc(d);c["stop_iteration_budget"]=0;c["stop_tolerance_condition_met"]=1
                c["tolerance_checks"]=c["last_checks_present"]=c["last_check_finite"]=1;c["last_checks_absent"]=0
                c["last_check_included"]=c["last_check_finite_support"]=1
            end,
            d->begin
                c=pc(d);c["last_check_excluded"]=1
            end,
            d->begin
                c=pc(d);c["final_primary_unmasked"]+=1;c["final_primary_nodes"]+=1
            end,
            d->begin
                c=pc(d);c["passes"]=c["requested_sweeps"]=c["executed_sweeps"]=typemax(Int)
                c["stop_single_sweep"]=typemax(Int)
            end)
        for change in mutations
            malformed=bad_report(change)
            @test_throws ArgumentError Hammerhead._quality_wrap(malformed)
        end
        for value in (0,1,"false")
            malformed=deepcopy(data)
            malformed["execution_diagnostics"]["groups"]["cam1"]["fractions"]["recorded_entry_fraction"]["available"]=value
            @test_throws ArgumentError Hammerhead._quality_wrap(malformed)
        end
        malformed=deepcopy(data)
        malformed["execution_diagnostics"]["groups"]["planar"]["fractions"]["recorded_entry_fraction"]["value"]=1
        @test_throws ArgumentError Hammerhead._quality_wrap(malformed)
        badtoml=joinpath(dir,"malformed.toml")
        open(badtoml,"w") do io
            TOML.print(io,bad_report(d->d["execution_diagnostics"]["groups"]["cam1"]["residual_unit"]="world"))
        end
        @test_throws ArgumentError load_quality_report(badtoml)
        # Valid individual counts can overflow only when multiple observations are summed.
        huge=execution_diagnostics_data(pdiag)
        huge["passes"][1]["requested_iterations"]=huge["passes"][1]["executed_iterations"]=typemax(Int)
        huge["passes"][1]["requested_tolerance"]=0.
        hugepacket=Hammerhead._execution_decode(huge)
        overflowing=quality_execution_native(joinpath(dir,"overflow.jld2"),[planar,planar];planar=[hugepacket,hugepacket])
        @test_throws OverflowError quality_execution_data(overflowing)
    end
end

@testset "Native execution markers, kinds, linkage and binding" begin
    planar,pdiag,_=quality_execution_planar()
    stereo,sdiag=quality_execution_stereo()
    mktempdir() do dir
        empty=quality_execution_native(joinpath(dir,"empty.jld2"),PIVResult[])
        data=quality_execution_data(empty)
        @test isempty(data["groups"]) && all(==(0),values(data["entry_kinds"]))
        @test all(group->all(==(0),values(group["counts"])),values(data["execution_diagnostics"]["groups"]))
        for marker in ("execution_diagnostics_format_version","stereo_execution_diagnostics_format_version"),value in (true,2)
            candidate=quality_execution_native(joinpath(dir,"$marker-$value.jld2"),PIVResult[])
            jldopen(candidate,"a+") do file;file[marker]=value;end
            @test_throws ArgumentError quality_execution_data(candidate)
            @test quality_report_data(quality_report(ResultFile(candidate)))["quality_report_format_version"]==1
        end
        for group in ("execution_diagnostics","stereo_execution_diagnostics")
            candidate=quality_execution_native(joinpath(dir,"$group-missing-marker.jld2"),PIVResult[])
            jldopen(candidate,"a+") do file;file[group*"/000007"]=Dict("unread"=>true);end
            @test_throws ArgumentError quality_execution_data(candidate)
        end
        path=quality_execution_native(joinpath(dir,"planar.jld2"),[planar];planar=[pdiag])
        quality_execution_rewrite_entry(path,"000007",e->e["result_key"]="results/000008")
        @test_throws ArgumentError quality_execution_data(path)
        path=quality_execution_native(joinpath(dir,"stereo.jld2"),[stereo];stereo=[sdiag])
        quality_execution_rewrite_entry(path,"000007",e->e["result_key"]="results/000008";stereo=true)
        @test_throws ArgumentError quality_execution_data(path)
        wrong=quality_execution_native(joinpath(dir,"wrong-planar.jld2"),[stereo];planar=[pdiag])
        @test_throws ArgumentError quality_execution_data(wrong)
        good=quality_execution_native(joinpath(dir,"stereo-good.jld2"),[stereo];stereo=[sdiag])
        entry=jldopen(f->f["stereo_execution_diagnostics/000007"],good)
        wrong=quality_execution_native(joinpath(dir,"wrong-stereo.jld2"),[planar])
        jldopen(wrong,"a+") do file
            file["stereo_execution_diagnostics_format_version"]=1
            file["stereo_execution_diagnostics/000007"]=entry
        end
        @test_throws ArgumentError quality_execution_data(wrong)
        particles=Particles([1.],[1.],[1.],[1.])
        ptv=PTVResult([1.],[1.],[0.],[0.],[0.],falses(1),[1],[1],particles,particles,PTVParameters())
        wrong=quality_execution_native(joinpath(dir,"wrong-ptv.jld2"),[ptv];planar=[pdiag])
        @test_throws ArgumentError quality_execution_data(wrong)
        changed=deepcopy(stereo);changed.cam2.v[1]=99.
        jldopen(good,"r+") do file
            delete!(file,"results/000007");file["results/000007"]=changed
        end
        @test_throws ArgumentError quality_execution_data(good)
        @test quality_report_data(quality_report(ResultFile(good)))["quality_report_format_version"]==1
        # Planar v1 execution has entry linkage only: no numerical binding is invented.
        changed=deepcopy(planar);changed.u[1]=99.
        unbound=quality_execution_native(joinpath(dir,"unbound-planar.jld2"),[changed];planar=[pdiag])
        @test quality_execution_data(unbound)["execution_diagnostics"]["groups"]["planar"]["binding"]=="entry_key_only"
        stale=ResultFile(unbound)
        open(unbound,"a") do io;write(io,UInt8(0));end
        @test_throws ArgumentError quality_report(stale;include_execution_diagnostics=true)
    end
end

@testset "Execution reports preserve verified experiment association" begin
    fixture=joinpath(pkgdir(Hammerhead),"test","reference_images","A")
    files=sort(filter(p->endswith(lowercase(p),".tif"),readdir(fixture;join=true)))
    recipe=PIVRecipe(PIVParameters(window_size=16,overlap=8,max_iterations=2);roi=ROI(1:48,1:48),image_type=Float32,threaded=false)
    record=ExperimentRecord([(files[1],files[2])],recipe)
    mktempdir() do dir
        output=joinpath(dir,"run.jld2")
        run=replay_experiment(record;output,record_diagnostics=true,record_measurement_history=true)
        report=quality_report(record,run;include_execution_diagnostics=true)
        data=quality_report_data(report)
        @test data["quality_report_format_version"]==3 && data["provenance"]["association"]=="recorded_output_verified"
        @test data["provenance"]["recipe_id"]==recipe_identity(recipe)
        @test data["execution_diagnostics"]["groups"]["planar"]["counts"]["recorded_entries"]==1
        both=quality_report_data(quality_report(record,run;include_execution_diagnostics=true,include_measurement_history=true))
        @test both["measurement_history"]["counts"]["recorded_entries"]==1
        @test both["execution_diagnostics"]==data["execution_diagnostics"]
        bytes=read(files[1])
        @test_throws ArgumentError save_quality_report(files[1],report)
        @test read(files[1])==bytes
        original=read(output)
        for change in (d->d["association"]=nothing,d->d["association"]["recipe_id"]="0"^64,
                d->d["association"]["input_id"]="0"^64,d->d["pair_index"]=2)
            write(output,original)
            quality_execution_rewrite_entry(output,"000001",e->change(e["diagnostics"]))
            modified=Hammerhead._experiment_run_data(run)
            modified["output_sha256"]=Hammerhead._experiment_file_digest(output)
            current=Hammerhead._experiment_run(modified,record)
            @test_throws ArgumentError quality_report(record,current;include_execution_diagnostics=true)
            @test quality_execution_data(output)["provenance"]["association"]=="unassociated"
        end
        write(output,original)
        jldopen(output,"r+") do file;delete!(file,"execution_diagnostics/000001");end
        modified=Hammerhead._experiment_run_data(run);modified["output_sha256"]=Hammerhead._experiment_file_digest(output)
        current=Hammerhead._experiment_run(modified,record)
        missing=quality_report_data(quality_report(record,current;include_execution_diagnostics=true))
        @test missing["execution_diagnostics"]["groups"]["planar"]["counts"]["missing_entries"]==1
        @test_throws ArgumentError quality_report(record,run;include_execution_diagnostics=true)
    end
end
@testset "Execution-aware native quality reports" begin
    planar,pdiag,history=quality_execution_planar(history=true,max_iterations=2)
    stereo,sdiag=quality_execution_stereo(T=Float32,grid=DewarpGrid(x=1.:.5:24.5,y=48.:-1.:1.),
        roi=ROI(5:44,7:42),scale=PhysicalScale(dt=.001,length_unit="mm",time_unit="s"))
    flat,fdiag=quality_execution_stereo(flat=true)
    particles=Particles([1.],[1.],[1.],[1.])
    ptv=PTVResult([1.],[1.],[0.],[0.],[0.],falses(1),[1],[1],particles,particles,PTVParameters())
    tracking=TrackingResult(Trajectory{Float64}[],1,PTVParameters())
    mktempdir() do dir
        path=joinpath(dir,"mixed.jld2")
        entries=[planar,planar,stereo,stereo,flat,ptv,tracking]
        quality_execution_native(path,entries;planar=[pdiag,nothing,nothing,nothing,nothing,nothing,nothing],
            stereo=[nothing,nothing,sdiag,nothing,fdiag,nothing,nothing],history=[history,nothing,nothing,nothing,nothing,nothing,nothing])
        index=ResultFile(path)
        report=quality_report(index;include_execution_diagnostics=true)
        data=quality_report_data(report);exec=data["execution_diagnostics"];groups=exec["groups"]
        @test data["quality_report_format_version"]==3 && !haskey(data,"measurement_history")
        @test data["entry_kinds"]==Dict("planar"=>2,"stereo"=>3,"ptv"=>1,"tracking"=>1)
        @test Set(keys(data["groups"]))==Set(["planar","stereo"])
        @test data["groups"]==quality_report_data(quality_report(entries[1:5]))["groups"]
        @test exec["unsupported_entries"]==Dict("ptv"=>1,"tracking"=>1)
        @test exec["verification_time"]=="report_generation"
        @test exec["last_check_scope"]=="last_recorded_check_per_pass"
        @test data["generator"]["weighting"]=="field_nodes_and_execution_observations"
        @test groups["planar"]["binding"]=="entry_key_only"
        pc=groups["planar"]["counts"]
        @test (pc["eligible_entries"],pc["recorded_entries"],pc["missing_entries"])==(2,1,1)
        @test (pc["passes"],pc["requested_sweeps"],pc["executed_sweeps"],pc["tolerance_checks"])==(1,2,2,0)
        @test pc["stop_iteration_budget"]==pc["last_checks_absent"]==1
        @test pc["final_primary_nodes"]==length(planar.u)
        for (role,child) in (("cam1",sdiag.cam1),("cam2",sdiag.cam2))
            c=groups[role]["counts"]
            @test (c["eligible_entries"],c["recorded_entries"],c["missing_entries"])==(3,2,1)
            @test (c["passes"],c["requested_sweeps"],c["executed_sweeps"],c["tolerance_checks"])==(4,10,6,2)
            @test c["stop_single_sweep"]==c["stop_tolerance_condition_met"]==2
            @test c["last_checks_present"]==c["last_checks_absent"]==2
            @test c["last_check_finite"]==2 && c["last_check_empty"]>=1
            @test c["final_primary_nodes"]==length(stereo.u)+length(flat.u)
            @test c["final_primary_finite"]==last(child.passes).residual.finite_count
            @test c["final_primary_nonfinite"]>=length(flat.u)
            @test groups[role]["residual_unit"]=="px" && groups[role]["coordinate_basis"]=="dewarped_pixels_x_columns_y_rows"
            @test groups[role]["binding"]=="raw_measurement_fields_checked_at_report_generation"
            @test !haskey(groups[role],"mean_magnitude")
            fraction=groups[role]["fractions"]["recorded_entry_fraction"]
            @test fraction["numerator"]==2 && fraction["denominator"]==3 && fraction["value"]==2/3
        end
        @test quality_execution_only_metadata(data)
        @test data["unavailable"]["pooled_execution_residual_amplitudes"]["reason_code"]=="not_aggregated"
        @test data["unavailable"]["replacement_history"]["available"]===false
        combined=quality_execution_data(path;include_measurement_history=true)
        @test combined["measurement_history"]["counts"]["recorded_entries"]==1
        @test combined["execution_diagnostics"]==exec
        @test combined["measurement_history"]["counts"]["unsupported_stereo_entries"]==3
        @test combined["unavailable"]["uncertainty_measurement_association"]["reason_code"]=="applicability_not_established"
        @test !haskey(combined["unavailable"],"replacement_history")
        @test quality_report_data(quality_report(index;include_measurement_history=true))["quality_report_format_version"]==2
        bare=joinpath(dir,"bare.jld2");save_results(bare,[planar,planar])
        @test quality_report_data(quality_report(ResultFile(bare)))["quality_report_format_version"]==1
        absent=quality_execution_data(bare)
        @test absent["execution_diagnostics"]["groups"]["planar"]["counts"]["missing_entries"]==2
        @test !absent["execution_diagnostics"]["groups"]["cam1"]["fractions"]["recorded_entry_fraction"]["available"]
        @test !haskey(absent["execution_diagnostics"]["groups"]["cam1"]["fractions"]["recorded_entry_fraction"],"value")
        @test_throws ArgumentError quality_report([planar];include_execution_diagnostics=true)
        @test_throws ArgumentError quality_report(view(index,1:2);include_execution_diagnostics=true)
        @test_throws ArgumentError quality_report(physical.(entries[1:2]);include_execution_diagnostics=true)
        output=joinpath(dir,"quality-v3.toml");save_quality_report(output,report)
        @test quality_report_data(load_quality_report(output))==data
        original_bytes=read(path)
        @test_throws ArgumentError save_quality_report(path,report)
        @test read(path)==original_bytes
        shown=sprint(show,MIME"text/plain"(),load_quality_report(output))
        @test occursin("not freshly verify",shown) && occursin("not pooled",shown)
        @test occursin("entry_key_only",shown) && occursin("dewarped_pixels",shown)
        copied=quality_report_data(report);copied["execution_diagnostics"]["groups"]["cam1"]["counts"]["passes"]=99
        @test quality_report_data(report)==data

        # Supplied raw verification is independent of file access and retains no payload.
        packet=load_stereo_execution_diagnostics(index,3)
        raw=index[3]
        @test execution_diagnostics_data(packet)["verification"]["inspection_state"]=="metadata_only"
        verified=execution_diagnostics_data(packet;result=raw)
        @test verified["verification"]["inspection_state"]=="supplied_measurement_fields_verified"
        @test verified["verification"]["measurement_field_binding_checked"]===true
        @test verified["verification"]["calibration"]===verified["verification"]["source_inputs"]===false
        @test execution_diagnostics_data(packet)["verification"]["inspection_state"]=="metadata_only"
        @test_throws ArgumentError execution_diagnostics_data(packet;result=physical(raw))
        edited=deepcopy(raw);edited.cam1.u[1]=99.
        @test_throws ArgumentError execution_diagnostics_data(packet;result=edited)
        @test_throws ArgumentError execution_diagnostics_data(packet;result=planar)
        weak,detached=quality_execution_supplied_lifetime(packet,raw)
        GC.gc();GC.gc()
        @test weak.value===nothing && quality_execution_only_metadata(detached)
        rm(path)
        @test execution_diagnostics_data(packet;result=raw)["verification"]["measurement_field_binding_checked"]
        @test quality_report_data(load_quality_report(output))==data # Past verification; no source reopen.
    end
end
