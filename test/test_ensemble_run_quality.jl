using Test, Hammerhead, Random, JLD2, TOML

erq_parameters(;kwargs...) = PIVParameters(window_size=16,overlap=8,padding=true,
    uod_enable=false,validation=(),replace_outliers=false;kwargs...)
function erq_pool(;T=Float64,backend=:cpu,flat=false,masked=false,uq=true)
    a = flat ? zeros(T,32,32) : rand(MersenneTwister(1807),T,32,32)
    packet=Ref{Any}()
    schedule=[erq_parameters(max_iterations=5,convergence_tol=Inf,uncertainty=uq),
              erq_parameters(max_iterations=3,convergence_tol=2.,uncertainty=uq)]
    result=run_piv_ensemble([(a,circshift(a,(1,2))),(a,circshift(a,(1,2)))],schedule;
        backend,threaded=false,progress=false,mask=masked ? trues(32,32) : nothing,
        scale=PhysicalScale(pixel_size=.02,dt=.001,length_unit="mm",time_unit="s"),
        on_diagnostics=d->(packet[]=d))
    result,packet[]
end
function erq_pair()
    a=rand(MersenneTwister(1808),32,32);d=Ref{Any}();h=Ref{Any}()
    r=run_piv(a,circshift(a,(1,1)),erq_parameters();threaded=false,
        on_diagnostics=x->(d[]=x),on_measurement_history=x->(h[]=x))
    r,d[],h[]
end
function erq_stereo()
    cameras=(PinholeCamera([100. 0 15. 0;0 100. 0 0;0 0 1. 100.]),
        PinholeCamera([100. 0 -15. 0;0 100. 0 0;0 0 1. 100.]))
    grid=DewarpGrid(x=1.:32.,y=32.:-1.:1.)
    dw=map(c->ImageDewarper(c,grid,(32,32)),cameras)
    a=rand(MersenneTwister(1809),32,32);d=Ref{Any}()
    r=run_piv_stereo(a,circshift(a,(1,1)),a,circshift(a,(1,-1)),dw...,erq_parameters();
        threaded=false,on_diagnostics=x->(d[]=x))
    r,d[]
end
function erq_native(path,entries;ensemble=fill(nothing,length(entries)),
        planar=fill(nothing,length(entries)),history=fill(nothing,length(entries)),stereo=fill(nothing,length(entries)))
    jldopen(path,"w") do file
        file["format_version"]=1
        for i in eachindex(entries)
            key="results/"*lpad(string(i*3),6,'0')
            file[key]=entries[i]
            ensemble[i]===nothing || Hammerhead._write_ensemble_execution_diagnostics(file,key,ensemble[i],entries[i])
            planar[i]===nothing || Hammerhead._write_execution_diagnostics(file,key,planar[i])
            history[i]===nothing || Hammerhead._write_measurement_history(file,key,history[i])
            stereo[i]===nothing || Hammerhead._write_stereo_execution_diagnostics(file,key,stereo[i],entries[i])
        end
    end
    path
end
function erq_replace(path,key,value)
    jldopen(path,"r+") do file
        haskey(file,key) && delete!(file,key)
        file[key]=value
    end
end
erq_report(path;kwargs...)=quality_report(ResultFile(path);include_ensemble_execution_diagnostics=true,kwargs...)
erq_data(path;kwargs...)=quality_report_data(erq_report(path;kwargs...))
function erq_metadata_only(x)
    x isa Union{PIVResult,StereoPIVResult,PTVResult,TrackingResult,PIVParameters} && return false
    x isa AbstractDict && return all(erq_metadata_only,values(x))
    x isa AbstractArray && return ndims(x)==1 && all(erq_metadata_only,x)
    x isa Union{Tuple,NamedTuple} && return all(erq_metadata_only,x)
    x===nothing || x isa Union{Number,AbstractString,Symbol}
end

@testset "Recorded ensemble quality reports" begin
    mktempdir() do dir
        @testset "CPU/KA precision and separate pooled populations" begin
            for T in (Float32,Float64),backend in (:cpu,:ka)
                raw,packet=erq_pool(;T,backend)
                path=erq_native(joinpath(dir,"$T-$backend.jld2"),[raw];ensemble=[packet])
                report=erq_report(path);data=quality_report_data(report)
                section=data["ensemble_execution_diagnostics"];c=section["counts"]
                @test data["quality_report_format_version"]===4
                @test section["classification"]==Dict("planar_entries_examined"=>1,"recorded_ensemble_entries"=>1,
                    "recorded_planar_iteration_entries"=>0,"entries_without_execution_metadata"=>0)
                @test c["pair_observations"]==2 && c["all_pass_pair_observations"]==4
                @test c["passes"]==c["executed_pooling_sweeps"]==2
                @test c["ignored_requested_iterations"]==8 && c["tolerance_checks"]==0
                @test c["passes_with_positive_ignored_tolerance"]==2
                @test c["all_pass_window_opportunities"]==sum(p.contributions.window_opportunities for p in packet.passes)
                @test c["final_nodes"]==length(raw.u) && c["final_unmasked"]==count(!,raw.mask)
                @test c["final_uq_admitted_window_pair_updates"]==last(packet.passes).uncertainty.admitted_window_pair_updates
                @test c["final_uq_u_finite_nonnegative_count"]==last(packet.passes).uncertainty.u.finite_nonnegative_count
                @test section["residual_unit"]==section["uncertainty_unit"]=="px"
                @test data["groups"]==quality_report_data(quality_report([raw]))["groups"]
                @test !haskey(data,"execution_diagnostics") && !haskey(data,"measurement_history")
                @test erq_metadata_only(data)
                @test !occursin("mean_magnitude",string(data)) && !occursin("rms_magnitude",string(data))
                saved=joinpath(dir,"$T-$backend.toml");save_quality_report(saved,report)
                @test quality_report_data(load_quality_report(saved))==data
                @test occursin("requested iterations ignored",sprint(show,MIME"text/plain"(),report))
                @test occursin("entries without execution metadata",sprint(show,MIME"text/plain"(),report))
                data["ensemble_execution_diagnostics"]["counts"]["passes"]=999
                @test quality_report_data(report)["ensemble_execution_diagnostics"]["counts"]["passes"]==2
                original=read(path);@test_throws ArgumentError save_quality_report(path,report)
                @test read(path)==original
            end
        end
        pooled,packet=erq_pool();pair,pdiag,history=erq_pair();stereo,sdiag=erq_stereo()
        particles=Particles([1.],[1.],[1.],[1.])
        ptv=PTVResult([1.],[1.],[0.],[0.],[0.],falses(1),[1],[1],particles,particles,PTVParameters())
        tracking=TrackingResult(Trajectory{Float64}[],1,PTVParameters())
        @testset "Mixed kinds, unknown workflow and unchanged defaults" begin
            entries=[pooled,pair,pair,stereo,ptv,tracking]
            path=erq_native(joinpath(dir,"mixed.jld2"),entries;
                ensemble=[packet,nothing,nothing,nothing,nothing,nothing],
                planar=[nothing,pdiag,nothing,nothing,nothing,nothing],
                history=[nothing,history,nothing,nothing,nothing,nothing],
                stereo=[nothing,nothing,nothing,sdiag,nothing,nothing])
            data=erq_data(path;include_execution_diagnostics=true,include_measurement_history=true)
            section=data["ensemble_execution_diagnostics"]
            @test section["classification"]==Dict("planar_entries_examined"=>3,"recorded_ensemble_entries"=>1,
                "recorded_planar_iteration_entries"=>1,"entries_without_execution_metadata"=>1)
            @test section["unsupported_entries"]==Dict("stereo"=>1,"ptv"=>1,"tracking"=>1)
            @test section["fractions"]["recorded_ensemble_fraction_of_planar_entries"]["value"]==1/3
            @test data["execution_diagnostics"]["groups"]["planar"]["counts"]["missing_entries"]==2
            @test data["execution_diagnostics"]["groups"]["cam1"]["counts"]["recorded_entries"]==1
            @test data["measurement_history"]["counts"]["missing_entries"]==2
            @test data["groups"]==quality_report_data(quality_report(entries[1:4]))["groups"]
            @test_throws ArgumentError quality_report(ResultFile(path)) # default refuses PTV/tracking
            @test quality_report_data(quality_report(ResultFile(path);include_measurement_history=true))["quality_report_format_version"]===2
            @test_throws ArgumentError quality_report(ResultFile(path);include_execution_diagnostics=true)
            bare=erq_native(joinpath(dir,"bare.jld2"),[pair,pair];planar=[pdiag,nothing],history=[history,nothing])
            @test quality_report_data(quality_report(ResultFile(bare)))["quality_report_format_version"]===1
            @test quality_report_data(quality_report(ResultFile(bare);include_execution_diagnostics=true))["quality_report_format_version"]===3
            newdata=erq_data(bare;include_execution_diagnostics=true,include_measurement_history=true)
            @test newdata["ensemble_execution_diagnostics"]["classification"]["recorded_ensemble_entries"]==0
            @test all(iszero,values(newdata["ensemble_execution_diagnostics"]["counts"]))
            @test_throws ArgumentError quality_report([pooled];include_ensemble_execution_diagnostics=true)
            @test_throws ArgumentError quality_report(view(ResultFile(path),1:2);include_ensemble_execution_diagnostics=true)
            @test_throws ArgumentError quality_report(physical.([pooled]);include_ensemble_execution_diagnostics=true)
        end
        @testset "Degenerate, masked, disabled UQ and empty files" begin
            for options in ((;flat=true),(;masked=true),(;uq=false))
                raw,d=erq_pool(;options...)
                path=erq_native(joinpath(dir,"special-$(first(keys(options))).jld2"),[raw];ensemble=[d])
                data=erq_data(path);c=data["ensemble_execution_diagnostics"]["counts"]
                @test c["final_primary_finite"]==last(d.passes).residual.finite_count
                @test c["all_pass_source_gated_window_pairs"]==sum(p.contributions.source_gated_window_pairs for p in d.passes)
                @test c["final_masked"]==count(raw.mask)
                @test c["final_uq_evaluated_entries"]==Int(last(d.passes).uncertainty.evaluated)
                if !last(d.passes).uncertainty.evaluated
                    @test c["final_uq_evaluated_unmasked_nodes"]==0
                    @test !data["ensemble_execution_diagnostics"]["fractions"]["final_uq_u_numeric_availability"]["available"]
                end
            end
            empty=erq_native(joinpath(dir,"empty.jld2"),Any[])
            data=erq_data(empty)
            @test all(iszero,values(data["ensemble_execution_diagnostics"]["counts"]))
            @test all(m->m["available"]===false,values(data["ensemble_execution_diagnostics"]["fractions"]))
            for (key,value) in (("ensemble_execution_diagnostics_format_version",true),
                    ("ensemble_execution_diagnostics_format_version",2),
                    ("ensemble_execution_diagnostics/000001",Dict("result_key"=>"results/000001")),
                    ("stereo_execution_diagnostics_format_version",2),("execution_diagnostics_format_version",true))
                bad=erq_native(joinpath(dir,"emptybad.jld2"),Any[]);erq_replace(bad,key,value)
                @test_throws ArgumentError erq_report(bad)
            end
            for group in ("execution_diagnostics","stereo_execution_diagnostics","measurement_history","ensemble_execution_diagnostics")
                for kind in (:malformed_root,:orphan_empty,:orphan_nonempty)
                    bad=erq_native(joinpath(dir,"root-$group-$kind.jld2"),kind===:orphan_nonempty ? [pair] : Any[])
                    erq_replace(bad,group*"_format_version",1)
                    erq_replace(bad,kind===:malformed_root ? group : group*"/000999",Dict("result_key"=>"results/000999"))
                    @test_throws ArgumentError erq_report(bad)
                end
            end
            history_only=erq_native(joinpath(dir,"history-only.jld2"),[pair];history=[history])
            data=erq_data(history_only;include_measurement_history=true)
            @test data["ensemble_execution_diagnostics"]["classification"]["entries_without_execution_metadata"]==1
            @test data["measurement_history"]["counts"]["recorded_entries"]==1
        end
        @testset "Wrong-kind, crossed packets, raw shape and resealed metadata" begin
            good=erq_native(joinpath(dir,"good.jld2"),[pooled];ensemble=[packet]);bytes=read(good)
            for raw in (ptv,tracking,stereo)
                bad=joinpath(dir,"wrongkind.jld2");write(bad,bytes);erq_replace(bad,"results/000003",raw)
                @test_throws ArgumentError erq_report(bad)
            end
            for (key,value) in (("execution_diagnostics/000003",nothing),("measurement_history/000003",nothing))
                bad=joinpath(dir,"crossed.jld2");write(bad,bytes)
                jldopen(bad,"r+") do f
                    key=="execution_diagnostics/000003" ? Hammerhead._write_execution_diagnostics(f,"results/000003",pdiag) :
                        Hammerhead._write_measurement_history(f,"results/000003",history)
                end
                @test_throws ArgumentError erq_report(bad)
            end
            bad=joinpath(dir,"rawbad.jld2");write(bad,bytes)
            changed=deepcopy(pooled);changed.u[1]+=1;erq_replace(bad,"results/000003",changed)
            @test load_ensemble_execution_diagnostics(bad)!==nothing
            @test_throws ArgumentError erq_report(bad)
            write(bad,bytes);changed=deepcopy(pooled);resize!(changed.x,length(changed.x)-1)
            erq_replace(bad,"results/000003",changed)
            entry=jldopen(f->f["ensemble_execution_diagnostics/000003"],bad,"r")
            entry["diagnostics"]["measurement_sha256"]=Hammerhead._history_result_digest(changed)
            entry["diagnostics_sha256"]=Hammerhead._experiment_digest(entry["diagnostics"])
            erq_replace(bad,"ensemble_execution_diagnostics/000003",entry)
            @test load_ensemble_execution_diagnostics(bad)!==nothing
            @test_throws ArgumentError erq_report(bad)
            write(bad,bytes);entry=jldopen(f->f["ensemble_execution_diagnostics/000003"],bad,"r")
            entry["result_key"]="results/000099";erq_replace(bad,"ensemble_execution_diagnostics/000003",entry)
            @test_throws ArgumentError erq_report(bad)
            write(bad,bytes);erq_replace(bad,"ensemble_execution_diagnostics/000099",entry)
            @test_throws ArgumentError erq_report(bad)
        end
        @testset "Strict v4 saved schema and generation-time verification" begin
            path=erq_native(joinpath(dir,"schema.jld2"),[pooled];ensemble=[packet])
            report=erq_report(path);data=quality_report_data(report)
            saved=joinpath(dir,"report.toml");save_quality_report(saved,report)
            good_toml=sprint(io->TOML.print(io,data))
            for (case,mutate) in enumerate((
                d->d["ensemble_execution_diagnostics"]["classification"]["recorded_ensemble_entries"]=true,
                d->d["ensemble_execution_diagnostics"]["counts"]["tolerance_checks"]=1,
                d->d["ensemble_execution_diagnostics"]["counts"]["all_pass_window_opportunities"]=typemax(Int),
                d->d["ensemble_execution_diagnostics"]["counts"]["final_primary_finite"]+=1,
                d->d["ensemble_execution_diagnostics"]["counts"]["final_uq_u_nonfinite_count"]+=1,
                d->d["ensemble_execution_diagnostics"]["counts"]["final_uq_evaluated_entries"]=0,
                d->d["ensemble_execution_diagnostics"]["fractions"]["recorded_ensemble_fraction_of_planar_entries"]["available"]=1,
                d->d["ensemble_execution_diagnostics"]["verification_time"]="fresh_read_verification",
                d->d["ensemble_execution_diagnostics"]["counts"]["mean_residual"]=1.,
                d->d["ensemble_execution_diagnostics"]["counts"]["final_masked"]+=1))
                @testset "Malformed v4 case $case" begin
                    # Native generated counter Dicts convert Bool to Int on assignment.
                    # Parsed TOML preserves the deliberately malformed scalar type.
                    malformed=TOML.parse(good_toml);mutate(malformed)
                    case==1 && @test malformed["ensemble_execution_diagnostics"]["classification"]["recorded_ensemble_entries"] isa Bool
                    open(saved,"w") do io;TOML.print(io,malformed);end
                    @test_throws ArgumentError load_quality_report(saved)
                end
            end
            save_quality_report(saved,report);rm(path)
            @test quality_report_data(load_quality_report(saved))==data # no fresh source read
            @test occursin("no fresh result verification",sprint(show,MIME"text/plain"(),load_quality_report(saved)))
        end
        @testset "One result entry per pool and bounded retained metadata" begin
            path=erq_native(joinpath(dir,"many.jld2"),fill(pooled,20);ensemble=fill(packet,20))
            report=erq_report(path);data=quality_report_data(report)
            @test data["groups"]["planar"]["counts"]["entries"]==20 # not 40 pair results
            @test data["ensemble_execution_diagnostics"]["counts"]["pair_observations"]==40
            @test data["ensemble_execution_diagnostics"]["counts"]["passes"]==40
            @test erq_metadata_only(data)
            classification=Dict(k=>0 for k in Hammerhead._QUALITY_ENSEMBLE_CLASSIFICATION)
            counts=Dict(k=>0 for k in Hammerhead._quality_ensemble_counter_names())
            function observe_ephemeral()
                raw=deepcopy(pooled);weak=WeakRef(raw)
                Hammerhead._quality_ensemble_observe!(classification,counts,ResultFile(path),1,raw)
                weak
            end
            weak=observe_ephemeral();GC.gc(true);GC.gc(true)
            @test weak.value===nothing
            @test classification["recorded_ensemble_entries"]==1
        end
        @testset "Exact whole-file mapping and detached request" begin
            path=erq_native(joinpath(dir,"whole-index.jld2"),[pooled,pair];ensemble=[packet,nothing])
            original=read(path)
            for (name,mutate) in ((:omitted,index->pop!(index.entry_keys)),
                    (:duplicated,index->push!(index.entry_keys,last(index.entry_keys))),
                    (:reordered,index->reverse!(index.entry_keys)))
                @testset "$name index" begin
                    index=ResultFile(path);mutate(index)
                    @test_throws ArgumentError quality_report(index;include_ensemble_execution_diagnostics=true)
                    @test read(path)==original
                    destination=joinpath(dir,"$name-destination.toml");write(destination,"sentinel")
                    @test_throws ArgumentError begin
                        report=quality_report(index;include_ensemble_execution_diagnostics=true)
                        save_quality_report(destination,report)
                    end
                    @test read(destination,String)=="sentinel"
                end
            end
            index=ResultFile(path)
            snapshot=Hammerhead._quality_whole_file_index(index)
            @test snapshot!==index && snapshot.entry_keys!==index.entry_keys
            @test snapshot.entry_keys==["000003","000006"] # gapped keys preserve native ordering
            pop!(index.entry_keys)
            data=quality_report_data(quality_report(snapshot;include_ensemble_execution_diagnostics=true))
            @test data["provenance"]["source_selection"]=="whole_file"
            @test data["provenance"]["source_index_entries"]==2
            @test data["ensemble_execution_diagnostics"]["classification"]["entries_without_execution_metadata"]==1
            @test data["ensemble_execution_diagnostics"]["classification"]["planar_entries_examined"]==2
            empty=erq_native(joinpath(dir,"empty-index.jld2"),Any[])
            @test isempty(Hammerhead._quality_whole_file_index(ResultFile(empty)))
            @test erq_data(empty)["provenance"]["source_index_entries"]==0
            malformed=erq_native(joinpath(dir,"malformed-results-root.jld2"),Any[])
            erq_replace(malformed,"results",Dict("000003"=>pair))
            index=ResultFile(malformed) # legacy constructor sees dictionary keys
            @test_throws ArgumentError Hammerhead._quality_whole_file_index(index)
            @test_throws ArgumentError quality_report(index;include_ensemble_execution_diagnostics=true)
            metadata_only=erq_native(joinpath(dir,"unreadable-payload.jld2"),[123])
            index=ResultFile(metadata_only)
            @test length(Hammerhead._quality_whole_file_index(index))==1 # mapping guard loads no payload
            @test_throws ArgumentError quality_report(index;include_ensemble_execution_diagnostics=true)
        end
    end
end
