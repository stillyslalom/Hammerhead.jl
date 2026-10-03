using Test, Hammerhead, TOML

function calibrated_ptv_fixture(::Type{T}=Float64) where T
    pa=Particles(T[10,30],T[20,40],ones(T,2),fill(T(3),2))
    pb=Particles(T[11,32],T[17,44],ones(T,2),fill(T(3),2))
    PTVResult(copy(pa.x),copy(pa.y),T[1,2],T[-3,4],T[0.25,0.5],BitVector([false,true]),
        [1,2],[1,2],pa,pb,PTVParameters())
end
function calibrated_timed_fixture(t,times;unit=nothing)
    pre=Hammerhead._tracking_preflight([zeros(2,2) for _ in times],times,unit,nothing,nothing,nothing)
    Hammerhead._tracking_bind(TrackingResult([t],length(times),PTVParameters()),pre.data)
end
function calibrated_csv_rows(path)
    chars=collect(read(path,String)); records=Vector{String}[]; record=String[]
    field=IOBuffer();quoted=false;i=1
    while i<=length(chars)
        c=chars[i]
        if c=='"'
            if quoted && i<length(chars) && chars[i+1]=='"'
                print(field,'"');i+=1
            else
                quoted=!quoted
            end
        elseif !quoted && c==','
            push!(record,String(take!(field)))
        elseif !quoted && c=='\n'
            push!(record,String(take!(field)));push!(records,record);record=String[]
        elseif !quoted && c=='\r'
        else
            print(field,c)
        end
        i+=1
    end
    @test !quoted && isempty(record) && position(field)==0
    header=first(records)
    @test all(row->length(row)==length(header),records[2:end])
    [Dict(zip(header,row)) for row in records[2:end]]
end
function calibrated_test_metadata(path;mutate=identity,csv_hash=nothing)
    wrapper=TOML.parsefile(path)
    data=wrapper["metadata"]
    mutate(data)
    csv_hash===nothing || (data["csv_sha256"]=csv_hash)
    data["geometry_sha256"]=Hammerhead._cal_digest(data["geometry"])
    wrapper["metadata_sha256"]=Hammerhead._cal_digest(data)
    open(path,"w") do io;TOML.print(io,wrapper;sorted=true);end
end

@testset "Calibrated PTV vector basis and metadata" begin
    r=calibrated_ptv_fixture()
    original=deepcopy(r)
    mktempdir() do dir
        transforms=[PlanarTransform([1.0 0;0 1],[-10.,-20.]),
            PlanarTransform([0.0 2;-3 0],[7.,9.]),PlanarTransform([2.0 1;-1 3],[5.,-7.])]
        for (id,transform) in enumerate(transforms),dt in (nothing,0.25)
            path=joinpath(dir,"ptv-$id-$(dt).csv")
            options=dt===nothing ? (;) : (;dt,time_unit="s")
            files=export_calibrated_table(path,r;transform,length_unit="mm",coordinate_frame="lab_xy",options...)
            @test files.csv_path==path && files.metadata_path==path*".metadata.toml"
            report=load_calibrated_table_metadata(files.metadata_path)
            data=calibrated_table_data(report)
            @test data["row_count"]==2 && data["result_kind"]=="ptv"
            @test data["verification"]["csv_structure_and_hash_at_load"]===true
            @test data["verification"]["numerical_result"]===false && data["verification"]["source_bytes"]===false
            @test data["geometry"]["output_coordinate_frame"]=="lab_xy"
            @test data["geometry"]["coordinate_frame_provenance"]=="opaque_user_label"
            @test data["geometry"]["transform"]["matrix"]==[collect(transform.matrix[1,:]),collect(transform.matrix[2,:])]
            @test data["geometry"]["transform"]["offset"]==collect(transform.offset)
            rows=calibrated_csv_rows(path)
            for (k,row) in enumerate(rows)
                @test parse.(Float64,[row["x"],row["y"]])≈collect(transform_point(transform,(r.x[k],r.y[k])))
                @test parse.(Float64,[row["u"],row["v"]])≈collect(transform_vector(transform,(r.u[k],r.v[k])))./(dt===nothing ? 1 : dt)
                @test row["outlier"]==string(r.outliers[k])
                @test row["match_residual"]=="" && parse(Float64,row["match_residual_pixel"])==r.match_residual[k]
                @test row["match_residual_pixel_unit"]=="px"
                @test row["vector_quantity"]==(dt===nothing ? "pair_displacement" : "pair_mean_velocity")
                @test row["vector_unit"]==(dt===nothing ? "mm" : "mm/s")
                @test row["velocity_unit"]==(dt===nothing ? "" : "mm/s")
            end
            @test data["diagnostics"]["match_residual"]=="unavailable_direction_not_retained"
            data["geometry"]["output_coordinate_frame"]="edited"
            @test calibrated_table_data(report)["geometry"]["output_coordinate_frame"]=="lab_xy"
            @test occursin("CSV verified at load",sprint(show,report))
        end
        @test r.x==original.x && r.u==original.u && r.outliers==original.outliers && r.match_residual==original.match_residual
        single=calibrated_ptv_fixture(Float32)
        path=joinpath(dir,"single.csv")
        transform=PlanarTransform(Float32[0.3 0;0 -0.7],Float32[1,2])
        files=export_calibrated_table(path,single;transform,length_unit="mm",coordinate_frame="float32")
        meta=calibrated_table_data(load_calibrated_table_metadata(files.metadata_path))
        @test meta["geometry"]["transform"]["original_numeric_type"]=="Float32"
        @test meta["geometry"]["transform"]["original_precision_bits"]==fill(24,6)
        @test meta["geometry"]["transform"]["applied_numeric_type"]=="Float64"
        @test parse(Float64,first(calibrated_csv_rows(path))["u"])==Float64(Float32(0.3))
        # An exact nonsingular determinant can underflow floating det tozero.
        tiny=PlanarTransform([1e-200 0.0;0.0 2e-200],zeros(2))
        files=export_calibrated_table(joinpath(dir,"tiny.csv"),r;transform=tiny,length_unit="m",coordinate_frame="tiny")
        @test calibrated_table_data(load_calibrated_table_metadata(files.metadata_path))["row_count"]==2
        # Large affine origin never enters vector differencing.
        translated=PlanarTransform([2.0 1;-1 3],[1e300,-1e300])
        files=export_calibrated_table(joinpath(dir,"origin.csv"),r;transform=translated,length_unit="mm",coordinate_frame="offset")
        @test [parse(Float64,row["u"]) for row in calibrated_csv_rows(files.csv_path)]==[-1,8]
        legacy=joinpath(dir,"legacy.csv")
        export_table(legacy,r)
        before=read(legacy)
        @test_throws ArgumentError export_table(legacy,r;transform=translated,length_unit="mm")
        @test read(legacy)==before
    end
end

@testset "Calibrated ordinal and actual-time secants" begin
    t=Trajectory{Float64}(2,[10.,14.,26.],[20.,18.,12.],[2,3,6])
    ordinal=TrackingResult([t],6,PTVParameters())
    transform=PlanarTransform([0.0 2;-3 0],[7.,9.])
    mktempdir() do dir
        for dt in (nothing,0.5)
            opts=dt===nothing ? (;) : (;dt,time_unit="s")
            files=export_calibrated_table(joinpath(dir,"ordinal-$dt.csv"),ordinal;transform,length_unit="mm",coordinate_frame="lab",opts...)
            rows=calibrated_csv_rows(files.csv_path)
            @test parse.(Float64,getindex.(rows,"u"))==fill(dt===nothing ? -4 : -8,3)
            @test parse.(Float64,getindex.(rows,"v"))==fill(dt===nothing ? -12 : -24,3)
            @test parse.(Float64,getindex.(rows,"elapsed_time"))==[1,2,5].*(dt===nothing ? 1 : dt)
            @test getindex.(rows,"gap_before")==["0","0","2"]
            @test all(row->row["sample_time_numerator"]=="",rows)
            data=calibrated_table_data(load_calibrated_table_metadata(files.metadata_path))
            @test data["diagnostics"]["match_residual"]=="not_recorded"
            @test data["time"]["basis"]==(dt===nothing ? "ordinal_frames" : "explicit_uniform_interval")
        end
        epoch=BigInt(typemax(Int64))+9
        timed=calibrated_timed_fixture(Trajectory{Float64}(1,[0.,8.,10.],[0.,4.,5.],[1,3,4]),epoch.+[0,1,4,5];unit="ns")
        large=PlanarTransform([2.0 1;-1 3],[1e300,-1e300])
        files=export_calibrated_table(joinpath(dir,"actual.csv"),timed;transform=large,length_unit="mm",coordinate_frame="lab")
        rows=calibrated_csv_rows(files.csv_path)
        @test parse.(Float64,getindex.(rows,"u"))==fill(5,3)
        @test parse.(Float64,getindex.(rows,"v"))==fill(1,3)
        @test getindex.(rows,"sample_time_numerator")==string.(epoch.+[0,4,5])
        @test getindex.(rows,"velocity_start_frame")==["1","1","3"]
        @test getindex.(rows,"velocity_end_frame")==["3","4","4"]
        data=calibrated_table_data(load_calibrated_table_metadata(files.metadata_path))
        @test data["time"]["basis"]=="actual_sample_times"
        @test data["time"]["unit"]["value"]=="ns"
        restored=Hammerhead._cal_unpack(data["timing_snapshot"]["value"])
        @test restored==tracking_timing_data(timed)
        @test data["verification"]["numerical_result"]===false
        unknown=calibrated_timed_fixture(Trajectory(1,[0.,1.,16.],[0.,2.,32.]),[0,1,4])
        for label in (nothing,"s")
            files=export_calibrated_table(joinpath(dir,"unknown-$label.csv"),unknown;transform=PlanarTransform([1.0 0;0 1],zeros(2)),
                length_unit="mm",coordinate_frame="lab",time_unit=label)
            data=calibrated_table_data(load_calibrated_table_metadata(files.metadata_path))
            @test data["time"]["unit_provenance"]==(label===nothing ? "unknown" : "explicit_export_time_unit_assumption")
            @test Hammerhead._cal_unpack(data["timing_snapshot"]["value"])["time_unit"]===nothing
            rows=calibrated_csv_rows(files.csv_path)
            @test parse.(Float64,getindex.(rows,"u"))==[1,4,5] # Outer secant, not derivative at middle observation.
            @test all(row->row["velocity_unit"]==(label===nothing ? "" : "mm/s"),rows)
        end
        source=FrameSource(3,i->zeros(32,32);timestamps=epoch.+[0,1,4],source_id="camera",frame_ids=["A","B","C"],
            labels=["raw\n\"$i\"" for i in 1:3],time_unit="ns",clock_id="clock")
        refs=[FrameRef(source,i) for i in 1:3]
        pre=Hammerhead._tracking_preflight(refs,:source,nothing,nothing,nothing,nothing)
        linked=Hammerhead._tracking_bind(TrackingResult([Trajectory(1,[0.,1.,4.],[0.,0.,0.])],3,PTVParameters()),pre.data)
        files=export_calibrated_table(joinpath(dir,"source.csv"),linked;transform,length_unit="mm",coordinate_frame="lab")
        @test calibrated_table_data(load_calibrated_table_metadata(files.metadata_path))["row_count"]==3
        source_rows=calibrated_csv_rows(files.csv_path)
        @test getindex.(source_rows,"source_frame_id")==["A","B","C"]
        @test getindex.(source_rows,"source_label")==source.labels
        @test all(row->row["clock_id"]==row["acquisition_clock_id"]=="clock",source_rows)
    end
end

@testset "Calibrated empty tables and numerical validity" begin
    empty_p=Particles(Float64[],Float64[],Float64[],Float64[])
    empty_ptv=PTVResult(Float64[],Float64[],Float64[],Float64[],Float64[],falses(0),Int[],Int[],empty_p,empty_p,PTVParameters())
    ordinal=TrackingResult(Trajectory{Float64}[],0,PTVParameters())
    pre=Hammerhead._tracking_preflight([zeros(2,2),zeros(2,2)],[0,1],nothing,nothing,nothing,nothing)
    timed=Hammerhead._tracking_bind(TrackingResult(Trajectory{Float64}[],2,PTVParameters()),pre.data)
    transform=PlanarTransform([2.0 1;0 -3],[7.,8.])
    mktempdir() do dir
        for (id,result) in enumerate((empty_ptv,ordinal,timed))
            files=export_calibrated_table(joinpath(dir,"empty-$id.csv"),result;transform,length_unit="mm",coordinate_frame="empty_lab")
            @test isempty(calibrated_csv_rows(files.csv_path))
            data=calibrated_table_data(load_calibrated_table_metadata(files.metadata_path))
            @test data["row_count"]==0
            @test data["geometry"]["transform"]["offset"]==[7.,8.]
            @test data["geometry"]["output_coordinate_frame"]=="empty_lab"
        end
        singleton=TrackingResult([Trajectory{Float64}(2,[2.],[3.],[2])],2,PTVParameters())
        files=export_calibrated_table(joinpath(dir,"single.csv"),singleton;transform,length_unit="mm",coordinate_frame="lab")
        row=only(calibrated_csv_rows(files.csv_path))
        @test row["u"]==row["v"]=="" && row["velocity_valid"]=="false"
        invalid=TrackingResult([Trajectory(1,[0.,NaN,2.],[0.,1.,2.])],3,PTVParameters())
        files=export_calibrated_table(joinpath(dir,"invalid.csv"),invalid;transform,length_unit="mm",coordinate_frame="lab")
        rows=calibrated_csv_rows(files.csv_path)
        @test getindex.(rows,"position_valid")==["true","false","true"]
        @test all(row->row["velocity_valid"]=="false",rows)
        # Exact finite input subtraction would overflow Float64 individually,
        # but after transform/division its final result is representable.
        huge=big(10)^310
        safe=calibrated_timed_fixture(Trajectory(1,[-floatmax(Float64),floatmax(Float64)],[0.,0.]),[0,huge])
        files=export_calibrated_table(joinpath(dir,"large.csv"),safe;transform=PlanarTransform([1.0 0;0 1],zeros(2)),length_unit="mm",coordinate_frame="lab")
        @test all(row->isfinite(parse(Float64,row["u"])),calibrated_csv_rows(files.csv_path))
    end
end

@testset "Calibrated rejection before output and protected aliases" begin
    r=calibrated_ptv_fixture();transform=PlanarTransform([2.0 1;-1 3],zeros(2))
    base=(;transform,length_unit="mm",coordinate_frame="lab")
    mktempdir() do dir
        csv,meta=joinpath(dir,"preserved.csv"),joinpath(dir,"preserved.toml")
        write(csv,"CSV preserved");write(meta,"metadata preserved")
        bad=[(;length_unit=""),(;coordinate_frame=""),(;coordinate_frame=3),(;time_unit="s"),(;dt=0.,time_unit="s"),
            (;dt=NaN,time_unit="s"),(;dt=true,time_unit="s"),(;dt=0.5),(;transform=PlanarTransform(zeros(2,2),zeros(2))),
            (;transform=PlanarTransform([Inf 0.;0. 1.],zeros(2))),
            (;transform=PlanarTransform(BigFloat[big"1e-1000" 0;0 1],BigFloat[0,0])),
            (;transform=PlanarTransform(BigFloat[big"1e1000" 0;0 1],BigFloat[0,0])),
            (;transform=PlanarTransform([floatmax(Float64) 0;0 1.],zeros(2))),
            (;transform=PlanarTransform([1e-300 0.;0 1e-300],zeros(2)),dt=big(10)^1000,time_unit="s")]
        for options in bad
            @test_throws ArgumentError export_calibrated_table(csv,r;merge(base,options)...,metadata_path=meta,overwrite=true)
            @test read(csv,String)=="CSV preserved" && read(meta,String)=="metadata preserved"
        end
        @test_throws ArgumentError export_calibrated_table(csv,r;base...,metadata_path=meta)
        @test_throws ArgumentError export_calibrated_table(csv,r;base...,metadata_path=csv,overwrite=true)
        @test_throws ArgumentError export_calibrated_table(csv,r;base...,metadata_path=meta,overwrite=true,protected_paths=[csv])
        @test_throws ArgumentError export_calibrated_table(csv,r;base...,metadata_path=meta,overwrite=true,protected_paths=[meta])
        if Sys.iswindows()
            fresh=joinpath(dir,"fresh.csv");case_alias=joinpath(dir,"FRESH.CSV")
            for overwrite in (false,true)
                @test_throws ArgumentError export_calibrated_table(fresh,r;base...,metadata_path=case_alias,overwrite)
                @test !ispath(fresh) && !ispath(case_alias)
            end
            @test_throws ArgumentError export_calibrated_table(fresh,r;base...,overwrite=true,protected_paths=[case_alias])
            @test !ispath(fresh) && !ispath(fresh*".metadata.toml")
            for suffix in ("."," ")
                ambiguous=fresh*suffix
                @test_throws ArgumentError export_calibrated_table(fresh,r;base...,metadata_path=ambiguous,overwrite=true)
                @test !ispath(fresh) && !ispath(ambiguous)
                @test_throws ArgumentError export_calibrated_table(ambiguous,r;base...,metadata_path=joinpath(dir,"fresh.toml"))
                @test !ispath(fresh) && !ispath(joinpath(dir,"fresh.toml"))
                ambiguous_parent=joinpath(dir,"uncreated"*suffix,"table.csv")
                @test_throws ArgumentError export_calibrated_table(ambiguous_parent,r;base...)
                @test !ispath(joinpath(dir,"uncreated"))
            end
        end
        parent=joinpath(dir,"parent");linked=joinpath(dir,"linked-parent");mkdir(parent)
        linked_parent=try
            symlink(parent,linked;dir_target=true)
            true
        catch err
            @info "Directory alias regression unavailable on this filesystem" exception=err
            false
        end
        if linked_parent
            # Resolve the existing parent even when several trailing components
            # have not been created yet, before either output is published.
            fresh=joinpath(parent,"new-subdirectory","fresh.csv")
            alias=joinpath(linked,"new-subdirectory","fresh.csv")
            @test_throws ArgumentError export_calibrated_table(fresh,r;base...,metadata_path=alias,overwrite=true)
            @test !ispath(dirname(fresh)) && !ispath(fresh) && !ispath(alias)
            @test_throws ArgumentError export_calibrated_table(fresh,r;base...,overwrite=true,protected_paths=[alias])
            @test !ispath(dirname(fresh))
        end
        @test_throws ArgumentError export_calibrated_table(csv,with_scale(r,PhysicalScale());base...,metadata_path=meta,overwrite=true)
        malformed=calibrated_ptv_fixture();pop!(malformed.v)
        @test_throws ArgumentError export_calibrated_table(csv,malformed;base...,metadata_path=meta,overwrite=true)
        invalid_track=TrackingResult([Trajectory{Float64}(1,[1.,2.],[1.,2.],[1,1])],2,PTVParameters())
        @test_throws ArgumentError export_calibrated_table(csv,invalid_track;base...,metadata_path=meta,overwrite=true)
        timed=calibrated_timed_fixture(Trajectory(1,[0.,1.],[0.,1.]),[0,1];unit="s")
        for options in ((;dt=1.,time_unit="s"),(;time_unit="ms"))
            @test_throws ArgumentError export_calibrated_table(csv,timed;merge(base,options)...,metadata_path=meta,overwrite=true)
        end
        converted=physical(with_scale(timed,PhysicalScale(pixel_size=2.,time_unit="s")))
        @test_throws ArgumentError export_calibrated_table(csv,converted;base...,metadata_path=meta,overwrite=true)
        timed.result.trajectories[1].x[1]=9
        @test_throws ArgumentError export_calibrated_table(csv,timed;base...,metadata_path=meta,overwrite=true)
        source=joinpath(dir,"source.jld2");save_results(source,r)
        index=ResultFile(source)
        @test_throws ArgumentError export_calibrated_table(source,index,1;base...,metadata_path=meta,overwrite=true)
        @test_throws ArgumentError export_calibrated_table(csv,index,1;base...,metadata_path=source,overwrite=true)
        @test only(load_results(source)) isa PTVResult
        alias=joinpath(dir,"alias.jld2");hardlink(source,alias)
        @test_throws ArgumentError export_calibrated_table(alias,index,1;base...,metadata_path=meta,overwrite=true)
        @test read(csv,String)=="CSV preserved" && read(meta,String)=="metadata preserved"
        files=export_calibrated_table(csv,r;base...,metadata_path=meta,overwrite=true)
        @test calibrated_table_data(load_calibrated_table_metadata(files.metadata_path))["row_count"]==2
    end
end

@testset "Calibrated metadata and CSV verification boundaries" begin
    r=calibrated_ptv_fixture();base=(;transform=PlanarTransform([2.0 1;-1 3],zeros(2)),length_unit="mm",coordinate_frame="lab")
    mktempdir() do dir
        files=export_calibrated_table(joinpath(dir,"table.csv"),r;base...,frame_id="α,\"run\"\nβ")
        goodcsv=read(files.csv_path);goodmeta=read(files.metadata_path)
        @test calibrated_table_data(load_calibrated_table_metadata(files.metadata_path))["row_count"]==2
        metadata_only=load_calibrated_table_metadata(files.metadata_path;verify_csv=false)
        @test calibrated_table_data(metadata_only)["verification"]["csv_structure_and_hash_at_load"]===false
        @test calibrated_table_data(metadata_only)["verification"]["numerical_result"]===false
        rm(files.csv_path)
        @test_throws ArgumentError load_calibrated_table_metadata(files.metadata_path)
        @test load_calibrated_table_metadata(files.metadata_path;verify_csv=false) isa CalibratedTableMetadata
        write(files.csv_path,goodcsv)
        for mutate in (d->d["calibrated_table_format_version"]=true,d->d["calibrated_table_format_version"]=2,
            d->d["row_count"]=-1,d->d["row_count"]=true,d->d["result_kind"]="stereo",
            d->d["geometry"]["coordinate_frame_provenance"]="verified_world_frame",
            d->d["geometry"]["transform"]["matrix"][1][1]=Inf,
            d->d["geometry"]["transform"]["original_precision_bits"][1]=24,
            d->d["time"]["unit"]["available"]=0,
            d->d["time"]["quantity"]="actual_time_secant",
            d->d["diagnostics"]["raw_pixel_residual"]=1,
            d->d["diagnostics"]["match_residual"]="exact_transformed_norm")
            write(files.metadata_path,goodmeta)
            calibrated_test_metadata(files.metadata_path;mutate)
            @test_throws ArgumentError load_calibrated_table_metadata(files.metadata_path;verify_csv=false)
        end
        write(files.metadata_path,goodmeta)
        calibrated_test_metadata(files.metadata_path;mutate=d->d["row_count"]=3)
        @test_throws ArgumentError load_calibrated_table_metadata(files.metadata_path)
        @test load_calibrated_table_metadata(files.metadata_path;verify_csv=false) isa CalibratedTableMetadata
        write(files.metadata_path,goodmeta)
        # A forged metadata hash never makes malformed CSV structure acceptable.
        header=first(split(String(copy(goodcsv)),'\n'))*"\n"
        for contents in ("wrong header\n",header*"1,2\n",header*"\"unterminated\n",header*"bad\"quote\n",header*"\"closed\"x\n")
            write(files.csv_path,contents);write(files.metadata_path,goodmeta)
            calibrated_test_metadata(files.metadata_path;csv_hash=Hammerhead._experiment_file_digest(files.csv_path))
            @test_throws ArgumentError load_calibrated_table_metadata(files.metadata_path)
        end
        write(files.csv_path,goodcsv);write(files.metadata_path,goodmeta)
        report=load_calibrated_table_metadata(files.metadata_path)
        report._data["row_count"]=99
        @test_throws ArgumentError calibrated_table_data(report)
        other=export_calibrated_table(joinpath(dir,"other.csv"),r;base...,dt=0.5,time_unit="s")
        @test_throws ArgumentError load_calibrated_table_metadata(files.metadata_path;csv_path=other.csv_path)
        moved=joinpath(dir,"relocated.csv");write(moved,goodcsv)
        restored=load_calibrated_table_metadata(files.metadata_path;csv_path=moved)
        @test moved in calibrated_table_data(restored)["local_protected_paths"]
        @test_throws ArgumentError export_calibrated_table(moved,r;base...,overwrite=true,protected_paths=calibrated_table_data(restored)["local_protected_paths"])
    end
end
