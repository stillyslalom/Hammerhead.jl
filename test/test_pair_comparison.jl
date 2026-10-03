using Test, Hammerhead, TOML
using FileIO: save
using ImageCore: Gray, N0f8

function pair_test_result(x,y;u=ones(length(y),length(x)),v=ones(length(y),length(x)),
                          mask=falses(size(u)),outliers=falses(size(u)),
                          uncertainty_u=fill(0.1,size(u)),uncertainty_v=fill(0.2,size(u)))
    p=PIVParameters(window_size=4,overlap=2,uncertainty=true)
    PIVResult(Float64.(x),Float64.(y),u,v,ones(size(u)),ones(size(u)),
        uncertainty_u,uncertainty_v,outliers,mask,p)
end

function pair_test_files(directory)
    texture=[mod(17i+31j+7i*j,251)/250 for i in 1:48,j in 1:48]
    a,b,c=(joinpath(directory,name*".png") for name in ("a","b","c"))
    save(a,Gray{N0f8}.(texture));save(b,Gray{N0f8}.(circshift(texture,(1,2))))
    save(c,Gray{N0f8}.(circshift(texture,(2,3))))
    a,b,c
end

function pair_test_no_payloads(value)
    if value isa AbstractDict
        return all(pair_test_no_payloads,values(value))
    elseif value isa AbstractArray
        return ndims(value)==1 && all(pair_test_no_payloads,value)
    else
        return value isa Union{AbstractString,Number}
    end
end

Base.@noinline function pair_test_transient_payloads()
    a,b=pair_test_result(1:2,1:2),pair_test_result(1:2,1:2)
    refs=(WeakRef(a.u),WeakRef(b.u),WeakRef(a.mask))
    data=Dict("common"=>Hammerhead._pair_measure(a,b,1.0),
        "before"=>Hammerhead._pair_native(a),"after"=>Hammerhead._pair_native(b))
    data,refs
end

@testset "Representative-pair recipe comparisons" begin
    @testset "Exact coordinates and common population golden moments" begin
        a=pair_test_result(1:4,1:3)
        b=pair_test_result(2:5,1:3)
        a.mask[1,2:3].=true;b.mask[1,[1,3]].=true
        b.outliers[2,2]=true
        a.u[2,4]=NaN
        a.outliers[3,2]=true;b.outliers[3,1:2].=true;a.v[3,3]=Inf
        a.u[2,2]=10;a.v[2,2]=20;b.u[2,1]=13;b.v[2,1]=24
        a.u[3,4]=5;a.v[3,4]=7;b.u[3,3]=4;b.v[3,3]=9
        a.uncertainty_u[2,2]=0.2;a.uncertainty_v[2,2]=0.4
        b.uncertainty_u[2,1]=0.3;b.uncertainty_v[2,1]=0.6
        a.uncertainty_v[3,4]=-0.1
        report=Hammerhead._pair_measure(a,b,1.0)
        @test report["common_x_nodes"]==3 && report["common_y_nodes"]==3
        @test report["counts"]==Dict("nodes"=>9,"masked_both"=>1,"masked_before_only"=>1,"masked_after_only"=>1,
            "unmasked_both"=>6,"valid_both"=>2,"valid_before_only"=>1,"valid_after_only"=>1,"invalid_both"=>2)
        @test report["uq_counts_on_joint_valid"]==Dict("available_both"=>1,"available_before_only"=>0,"available_after_only"=>1,"unavailable_both"=>0)
        metric=report["velocity_difference"]
        @test metric["count"]==2 && metric["available"]
        @test metric["components"]["u"]["mean"]==1
        @test metric["components"]["v"]["mean"]==3
        @test metric["components"]["u"]["rms"]≈sqrt(5)
        @test metric["components"]["v"]["rms"]≈sqrt(10)
        @test metric["vector_rms_difference"]≈sqrt(15)
        uq=report["stored_uncertainty"]
        @test uq["count"]==1 && uq["components"]["difference_u"]["mean"]≈0.1
        @test uq["components"]["difference_v"]["rms"]≈0.2
        @test uq["components"]["before_u"]["mean"]==0.2
        @test uq["components"]["after_v"]["mean"]==0.6

        none=Hammerhead._pair_measure(pair_test_result([1.5],[1.5]),pair_test_result([1.0],[1.0]),1.0)
        @test none["velocity_difference"]==Dict("available"=>false,"count"=>0,"reason_code"=>"no_common_nodes")
        @test none["stored_uncertainty"]["reason_code"]=="no_common_nodes"
        masked=pair_test_result([1.0],[1.0];mask=trues(1,1))
        @test Hammerhead._pair_measure(masked,masked,1.0)["velocity_difference"]["reason_code"]=="no_joint_valid_nodes"
        zero=pair_test_result([1.0],[1.0];uncertainty_u=zeros(1,1),uncertainty_v=zeros(1,1))
        @test Hammerhead._pair_measure(zero,zero,1.0)["stored_uncertainty"]["available"]
        baduq=pair_test_result([1.0],[1.0];uncertainty_u=fill(NaN,1,1))
        @test Hammerhead._pair_measure(baduq,baduq,1.0)["stored_uncertainty"]["reason_code"]=="no_joint_uq_nodes"
        for axis in ([1.,1.],[2.,1.],[1.,NaN],[1.,Inf])
            @test_throws ArgumentError Hammerhead._pair_measure(pair_test_result(axis,[1.]),pair_test_result(axis,[1.]),1.0)
        end
        wrong=pair_test_result([1.,2.],[1.];u=zeros(2,2),v=zeros(2,2))
        @test_throws DimensionMismatch Hammerhead._pair_measure(wrong,wrong,1.0)
        @test Hammerhead._pair_intersect(Float32[8.5,16.5,24.5],Float64[16.5,32.5])==([2],[1])
        @test Hammerhead._pair_measure(a,b,20.0)["velocity_difference"]["components"]["u"]["mean"]==20
        detached,refs=pair_test_transient_payloads()
        GC.gc();GC.gc()
        @test all(ref->ref.value===nothing,refs)
        @test pair_test_no_payloads(detached)
    end

    @testset "Stable large values and explicit arithmetic overflow" begin
        a=pair_test_result(1:2,[1.];u=zeros(1,2),v=zeros(1,2))
        b=pair_test_result(1:2,[1.];u=reshape([1e200,-1e200],1,2),v=reshape([1e200,-1e200],1,2))
        metric=Hammerhead._pair_measure(a,b,1.0)["velocity_difference"]
        @test metric["available"] && metric["components"]["u"]["rms"]≈1e200
        @test metric["components"]["u"]["mean"]==0
        @test metric["vector_rms_difference"]≈hypot(1e200,1e200)
        offset=1e150
        oa=pair_test_result(1:2,[1.];u=fill(offset,1,2),v=zeros(1,2))
        ob=pair_test_result(1:2,[1.];u=fill(nextfloat(offset),1,2),v=zeros(1,2))
        delta=nextfloat(offset)-offset
        @test Hammerhead._pair_measure(oa,ob,1.0)["velocity_difference"]["components"]["u"]["mean"]==delta
        extreme=pair_test_result([1.],[1.];u=fill(floatmax(Float64),1,1),v=fill(floatmax(Float64),1,1))
        negative=pair_test_result([1.],[1.];u=fill(-floatmax(Float64),1,1),v=zeros(1,1))
        for metric in (Hammerhead._pair_measure(extreme,negative,1.0)["velocity_difference"],
                       Hammerhead._pair_measure(extreme,extreme,2.0)["velocity_difference"],
                       Hammerhead._pair_measure(pair_test_result([1.],[1.];u=zeros(1,1),v=zeros(1,1)),extreme,1.0)["velocity_difference"])
            @test metric==Dict("available"=>false,"count"=>1,"reason_code"=>"nonfinite_arithmetic")
        end
        uqoverflow=Hammerhead._pair_measure(a,a,1e308)["stored_uncertainty"]
        @test uqoverflow["available"] # 0.1/0.2 times 1e308 remains finite.
        hugeuq=pair_test_result([1.],[1.];uncertainty_u=fill(2.,1,1))
        @test Hammerhead._pair_measure(hugeuq,hugeuq,1e308)["stored_uncertainty"]["reason_code"]=="nonfinite_arithmetic"
    end

    @testset "Selected file verification, complete processing and portable snapshots" begin
        mktempdir() do directory
            a,b,c=pair_test_files(directory)
            p=PIVParameters(window_size=16,overlap=8,padding=true,uod_enable=false)
            recipe=PIVRecipe(p;image_type=Float32)
            before=ExperimentRecord([(a,b),(a,c)],recipe)
            after=ExperimentRecord([(a,b)],recipe)
            record_path=save_experiment(joinpath(directory,"before.jld2"),before)
            known_result=joinpath(directory,"old-failed-result.jld2");write(known_result,"historic failed output")
            now=time()
            push!(before.runs,ExperimentRun(string(Hammerhead.UUIDs.uuid4()),recipe_identity(before.recipe),before.input_id,
                now,now,:failed,0,known_result,Hammerhead._experiment_file_digest(known_result),
                deepcopy(before.creation_environment),"fixture failure"))
            # An unrelated source can disappear; selected verification is bounded.
            saved_c=read(c);rm(c)
            report=compare_recipe_pair(before,after;pair_indices=(1,1))
            data=pair_comparison_data(report)
            @test before.input_id!=after.input_id
            @test data["provenance"]["verification"]=="selected_ordered_pair_bytes_and_dimensions"
            @test data["common"]["velocity_difference"]["components"]["u"]["rms"]==0
            @test data["common"]["stored_uncertainty"]["reason_code"]=="no_joint_uq_nodes"
            @test isempty(data["settings_changes"])
            @test data["native"]["before"]["counts"]==data["native"]["after"]["counts"]
            @test occursin("RMS difference",sprint(show,MIME"text/plain"(),report))
            @test occursin("accuracy and uncertainty coverage not evaluated",sprint(show,MIME"text/plain"(),report))
            @test occursin("RecipePairComparison",sprint(show,report))
            data["common"]["counts"]["nodes"]=-1
            @test pair_comparison_data(report)["common"]["counts"]["nodes"]>0
            @test pair_test_no_payloads(pair_comparison_data(report))
            @test !pair_test_no_payloads(Dict("hidden"=>ones(2,2)))
            write(c,saved_c)

            moved_a,moved_b=joinpath(directory,"moved-a.png"),joinpath(directory,"moved-b.png")
            cp(a,moved_a);cp(b,moved_b)
            moved=ExperimentRecord([(moved_a,moved_b)],recipe)
            relocated=compare_recipe_pair(before,moved;pair_indices=[1,1])
            @test pair_comparison_data(relocated)["provenance"]["ordered_pair_id"]==pair_comparison_data(report)["provenance"]["ordered_pair_id"]
            @test pair_comparison_data(relocated)["provenance"]["after"]["files"][1]["path"]==realpath(moved_a)
            for indices in ((true,1),(0,1),(1,2),(1,),"1,1")
                @test_throws ArgumentError compare_recipe_pair(before,after;pair_indices=indices)
            end
            reversed=ExperimentRecord([(b,a)],recipe)
            changed=ExperimentRecord([(a,c)],recipe)
            @test_throws ArgumentError compare_recipe_pair(before,reversed;pair_indices=(1,1))
            @test_throws ArgumentError compare_recipe_pair(before,changed;pair_indices=(1,1))
            saved_a=read(a);open(io->write(io,UInt8(0)),a,"a")
            @test_throws ArgumentError compare_recipe_pair(before,after;pair_indices=(1,1))
            write(a,saved_a)
            corrupted=deepcopy(after);corrupted.recipe.passes[1]=PIVParameters(window_size=8,overlap=4)
            @test_throws ArgumentError compare_recipe_pair(before,corrupted;pair_indices=(1,1))
            script=joinpath(directory,"custom.jl");write(script,"identity(image)")
            custom=ExperimentRecord([(a,b)],PIVRecipe(p;external_preprocess=ScriptReference(script;entrypoint="identity")))
            @test_throws ArgumentError compare_recipe_pair(before,custom;pair_indices=(1,1))
            oldenv=deepcopy(after);oldenv.creation_environment["julia_version"]="0.0.0"
            @test_throws ArgumentError compare_recipe_pair(before,oldenv;pair_indices=(1,1))
            @test pair_comparison_data(compare_recipe_pair(before,oldenv;pair_indices=(1,1),allow_environment_change=true))["provenance"]["allow_environment_change"]

            # New processing uses the complete precision/preprocess/mask/ROI recipe.
            mask=falses(48,48);mask[5:12,7:14].=true
            revision=PIVRecipe(PIVParameters(window_size=8,overlap=4,padding=true,uod_enable=false,uncertainty=true);
                image_type=Float64,preprocessing=[PreprocessStep(:highpass_filter;sigma=2)],mask,roi=ROI(5:44,5:44))
            revised=ExperimentRecord([(a,b)],revision)
            comparison=compare_recipe_pair(before,revised;pair_indices=(1,1))
            comparison_data=pair_comparison_data(comparison)
            @test !isempty(comparison_data["settings_changes"])
            @test comparison_data["common"]["counts"]["nodes"]>0
            direct=run_piv(highpass_filter(load_image(Float64,a);sigma=2),highpass_filter(load_image(Float64,b);sigma=2),revision.passes;
                mask,roi=revision.roi,threaded=false)
            original=run_piv(load_image(Float32,a),load_image(Float32,b),recipe.passes;threaded=false)
            @test comparison_data["common"]==Hammerhead._pair_measure(original,direct,1.0)
            @test comparison_data["native"]["after"]==Hammerhead._pair_native(direct)
            shifted=ExperimentRecord([(a,b)],PIVRecipe(p;roi=ROI(2:47,2:47)))
            @test pair_comparison_data(compare_recipe_pair(before,shifted;pair_indices=(1,1)))["common"]["velocity_difference"]["reason_code"]=="no_common_nodes"

            scale=PhysicalScale(0.02,0.001,"mm","s")
            scaled=ExperimentRecord([(a,b)],PIVRecipe(p;image_type=Float32,scale))
            physical_report=compare_recipe_pair(scaled,scaled;pair_indices=(1,1),basis=:physical)
            @test pair_comparison_data(physical_report)["basis"]["unit"]=="mm/s"
            @test pair_comparison_data(physical_report)["basis"]["factor"]==20
            @test pair_comparison_data(compare_recipe_pair(before,scaled;pair_indices=(1,1)))["basis"]["quantity"]=="displacement"
            calibrated_revision=PIVRecipe(revision.passes;image_type=revision.image_type,
                preprocessing=revision.preprocessing,mask=revision.mask,roi=revision.roi,scale)
            physical_changed=pair_comparison_data(compare_recipe_pair(scaled,
                ExperimentRecord([(a,b)],calibrated_revision);pair_indices=(1,1),basis=:physical))
            @test physical_changed["common"]==Hammerhead._pair_measure(original,direct,20.0)
            missing=joinpath(directory,"temporarily-missing.png");mv(a,missing)
            try
                @test_throws ArgumentError compare_recipe_pair(before,scaled;pair_indices=(1,1),basis=:physical)
                for mismatch in (PhysicalScale(0.03,0.001,"mm","s"),PhysicalScale(0.02,0.002,"mm","s"),PhysicalScale(0.02,0.001,"m","s"))
                    other=deepcopy(scaled)
                    other_recipe=PIVRecipe(p;image_type=Float32,scale=mismatch)
                    other=ExperimentRecord(other_recipe,other.input_files,other.pairs,other.input_id,other.creation_environment,other.runs,other.record_paths)
                    err=try compare_recipe_pair(scaled,other;pair_indices=(1,1),basis=:physical);nothing catch e;e end
                    @test err isa ArgumentError && occursin("identical scale",sprint(showerror,err))
                end
            finally
                mv(missing,a)
            end
            @test_throws ArgumentError compare_recipe_pair(before,after;pair_indices=(1,1),basis=:unknown)

            destination=joinpath(directory,"comparison.toml")
            @test save_pair_comparison(destination,comparison)==destination
            reopened=load_pair_comparison(destination)
            @test pair_comparison_data(reopened)==comparison_data
            for protected in (a,b,c,record_path,known_result)
                bytes=read(protected)
                @test_throws ArgumentError save_pair_comparison(protected,load_pair_comparison(destination))
                @test read(protected)==bytes
            end
            extra=joinpath(directory,"result.jld2");write(extra,"protected output")
            @test_throws ArgumentError save_pair_comparison(extra,reopened;protected_paths=[extra])
            @test read(extra,String)=="protected output"
            link=joinpath(directory,"hardlink.png")
            try
                hardlink(a,link)
                @test_throws ArgumentError save_pair_comparison(link,reopened)
                @test read(link)==read(a)
            catch error
                error isa Base.IOError || rethrow()
            end
            malformed=joinpath(directory,"malformed.toml")
            for mutation in (d->(d["pair_comparison_format_version"]=2),d->(d["common"]["counts"]["valid_both"]+=1),
                             d->(d["common"]["velocity_difference"]["components"]["u"]["rms"]=Inf),
                             d->(d["basis"]["factor"]=2.0),d->empty!(d["protected_locators"]),
                             d->(d["provenance"]["ordered_pair_id"]=repeat("0",64)),
                             d->(d["unavailable"]["accuracy"]["available"]=0),
                             d->(d["unavailable"]["accuracy"]["available"]=true),
                             d->(d["unavailable"]["accuracy"]["reason_code"]=1),
                             d->(d["unavailable"]["accuracy"]["extra"]="unknown"))
                altered=deepcopy(comparison_data);mutation(altered)
                open(io->TOML.print(io,altered),malformed,"w")
                @test_throws ArgumentError load_pair_comparison(malformed)
            end
            # Preserve the common-grid partition and metric counts while inventing
            # UQ availability absent from each native result in turn.
            for side in ("before","after")
                altered=pair_comparison_data(report)
                uq=altered["common"]["uq_counts_on_joint_valid"]
                uq["unavailable_both"]-=1
                uq["available_$(side)_only"]+=1
                open(io->TOML.print(io,altered),malformed,"w")
                @test_throws ArgumentError load_pair_comparison(malformed)
            end
            altered_report=deepcopy(report);altered_report._data["common"]["counts"]["nodes"]+=1
            @test_throws ArgumentError pair_comparison_data(altered_report)
        end
    end
end
