using Test, Hammerhead, JLD2, TOML

function artifact_portability_fixture(locators=String[])
    raw=TrackingResult([Trajectory(1,[0.0,2.0],[1.0,1.0])],2,PTVParameters())
    pre=Hammerhead._tracking_preflight([zeros(2,2),zeros(2,2)],[0,1],"s",nothing,nothing,nothing)
    pre.data["protected_paths"]=sort!(unique(String.(locators)))
    Hammerhead._tracking_bind(raw,pre.data)
end
function artifact_rewrite_metadata(path,mutate)
    wrapper=TOML.parsefile(path);mutate(wrapper["metadata"])
    wrapper["metadata_sha256"]=Hammerhead._cal_digest(wrapper["metadata"])
    open(path,"w") do io;TOML.print(io,wrapper;sorted=true);end
end

@testset "Portable lexical locators and explicit local paths" begin
    H=Hammerhead
    for path in ("/", "/capture/α/a.tif",raw"C:\capture\a.tif","D:/capture/a.tif",
                 raw"\\server\share\a.tif","//server/share/a.tif",raw"\\?\C:\capture\a.tif",raw"\\?\UNC\server\share\a.tif")
        @test H._artifact_absolute_locator(path)
    end
    for path in ("", "a.tif",raw"C:relative.tif",raw"\current_drive\a.tif",raw"\\server",
                 raw"\\\server",raw"\\server\\share",raw"\\?\UNC\server",raw"\\.\pipe\x","/a\0b")
        @test !H._artifact_absolute_locator(path)
    end
    # V1 has no path-style tag: leading // is conservatively UNC, not guessed POSIX.
    @test H._artifact_locator_style("//server/share")==:windows
    @test H._artifact_local_locator("/capture/a.tif")==!Sys.iswindows()
    @test H._artifact_local_locator(raw"C:\capture\a.tif")==Sys.iswindows()
    @test H._artifact_local_locator(raw"\\server\share\a.tif")==Sys.iswindows()
    foreign=Sys.iswindows() ? "/capture/a.tif" : raw"C:\capture\a.tif"
    @test_throws ArgumentError H._artifact_local_path(foreign)
    @test_throws ArgumentError H._artifact_local_path(raw"C:relative.tif")
    @test_throws ArgumentError H._artifact_local_path(raw"\current_drive\a.tif")
    @test_throws ArgumentError H._artifact_local_path("")
    @test !H._artifact_relative_locator(foreign)
    @test !H._artifact_relative_locator(raw"C:relative.csv")
    @test H._artifact_relative_locator(raw"subdir\table.csv")
    mktempdir() do dir
        @test H._artifact_local_path(joinpath(dir,"child","..","a"))==joinpath(dir,"a")
        @test H._artifact_alias(joinpath(dir,"child","..","a"),joinpath(dir,"a"))
        @test H._artifact_resolve_relative(dir,"table.csv")==joinpath(dir,"table.csv")
        if !Sys.iswindows()
            @test_throws ArgumentError H._artifact_resolve_relative(dir,raw"subdir\table.csv")
        end
        source=joinpath(dir,"source");write(source,"source bytes")
        alias=joinpath(dir,"alias");hardlink(source,alias)
        @test H._artifact_alias(source,alias)
        actual=joinpath(dir,"actual");mkdir(actual)
        linked=joinpath(dir,"linked")
        try
            symlink(actual,linked;dir_target=true)
            @test H._artifact_alias(joinpath(actual,"fresh"),joinpath(linked,"fresh"))
        catch error
            error isa Base.IOError || rethrow()
        end
        if Sys.iswindows()
            @test H._artifact_alias(joinpath(dir,"Fresh.csv"),joinpath(dir,"fresh.CSV"))
            ordinary=joinpath(dir,"namespace.csv");extended="\\\\?\\"*ordinary
            @test H._artifact_alias(ordinary,extended)
            base=(transform=PlanarTransform([1.0 0.0;0.0 1.0],[0.0,0.0]),length_unit="mm",coordinate_frame="provided")
            @test_throws ArgumentError export_calibrated_table(ordinary,artifact_portability_fixture();base...,metadata_path=extended,overwrite=true)
            @test !ispath(ordinary)
            for spelling in ("a.","a ",joinpath("folder.","a"),joinpath("folder ","a"))
                @test_throws ArgumentError H._artifact_prospective_path(joinpath(dir,spelling))
            end
        end
    end
end

@testset "Foreign timed provenance and local protection" begin
    H=Hammerhead
    foreign=Sys.iswindows() ? ["/capture/scene/a.tif","/capture/scene/b.tif"] :
        [raw"C:\capture\scene\a.tif",raw"\\server\share\b.tif"]
    timed=artifact_portability_fixture(foreign)
    original=tracking_timing_data(timed)
    @test original["protected_paths"]==sort(foreign)
    @test H._artifact_local_protected_paths(original["protected_paths"])==String[]
    @test trajectory_velocities(timed,1)==([2.0,2.0],[0.0,0.0])
    mktempdir() do dir
        path=joinpath(dir,"foreign.jld2")
        save_timed_tracking(path,timed)
        jldopen(path,"r") do file
            @test file["timed_tracking_format_version"]==1
            @test isequal(file["tracking_timing"],original)
            @test file["tracking_timing_sha256"]==timed.timing._sha256
        end
        loaded=load_timed_tracking(path)
        data=tracking_timing_data(loaded)
        @test all(p->p in data["protected_paths"],foreign)
        @test path in data["protected_paths"]
        @test trajectory_velocities(loaded,1)==trajectory_velocities(timed,1)
        @test tracking_speed_summary(loaded).speeds==[2.0]
        scaled=with_scale(loaded,PhysicalScale(pixel_size=0.25,dt=99,length_unit="mm",time_unit="s"))
        converted=physical(scaled)
        @test trajectory_velocities(scaled,1)==trajectory_velocities(converted,1)==([0.5,0.5],[0.0,0.0])
        @test tracking_speed_summary(converted).speeds==[0.5]
        @test tracking_speed_summary(converted).length_unit=="mm"
        @test physical(converted)===converted
        @test all(p->p in tracking_timing_data(converted)["protected_paths"],foreign)
        bytes=read(path)
        @test_throws ArgumentError save_timed_tracking(path,loaded)
        @test_throws ArgumentError export_table(path,loaded)
        @test read(path)==bytes
        alias=joinpath(dir,"hardlink.jld2");hardlink(path,alias)
        @test_throws ArgumentError save_timed_tracking(alias,loaded)
        @test_throws ArgumentError export_table(alias,loaded)
        @test read(path)==bytes
        relocated_input=joinpath(dir,"relocated-input.tif");write(relocated_input,"local input")
        inputbytes=read(relocated_input)
        @test_throws ArgumentError save_timed_tracking(relocated_input,loaded;protected_paths=[relocated_input])
        @test_throws ArgumentError export_table(relocated_input,loaded;protected_paths=[relocated_input])
        @test read(relocated_input)==inputbytes
        saved=joinpath(dir,"relocated.jld2")
        save_timed_tracking(saved,loaded;protected_paths=[relocated_input])
        @test relocated_input in tracking_timing_data(load_timed_tracking(saved))["protected_paths"]
        @test tracking_timing_data(loaded)==data # save adds context only to new artifact
        csv=joinpath(dir,"tracks.csv");export_table(csv,loaded;protected_paths=[relocated_input])
        @test startswith(read(csv,String),"schema_version,result_type,")
        @test tracking_timing_data(timed)==original
        @test_throws ArgumentError save_timed_tracking(joinpath(dir,"bad.jld2"),loaded;protected_paths=foreign)
        @test_throws ArgumentError export_table(joinpath(dir,"bad.csv"),loaded;protected_paths=foreign)
        @test_throws ArgumentError save_timed_tracking(first(foreign),loaded)
        @test_throws ArgumentError load_timed_tracking(first(foreign))
        @test !ispath(joinpath(dir,"bad.jld2")) && !ispath(joinpath(dir,"bad.csv"))
        # A foreign locator must not be abspath'd into a native protection claim.
        target=joinpath(dir,"unrelated.jld2")
        mimic=Sys.iswindows() ? replace(target[3:end],'\\'=>'/') : "C:"*replace(target,'/'=>'\\')
        @test !H._artifact_local_locator(mimic)
        unlocated=artifact_portability_fixture([mimic])
        save_timed_tracking(target,unlocated)
        @test isfile(target)
        if Sys.iswindows()
            @test_throws ArgumentError save_timed_tracking(joinpath(dir,"FOREIGN.JLD2"),loaded)
            for tail in ("destination.","destination ")
                @test_throws ArgumentError save_timed_tracking(joinpath(dir,tail),loaded)
                @test_throws ArgumentError export_table(joinpath(dir,tail),loaded)
            end
        end
    end
    for locator in ("relative.tif",raw"C:relative.tif",raw"\\server", "")
        @test_throws ArgumentError artifact_portability_fixture([locator])
    end
end

@testset "Foreign calibrated metadata, relocation and native protection" begin
    H=Hammerhead
    foreign=Sys.iswindows() ? ["/capture/scene/a.tif","/capture/scene/b.tif"] :
        [raw"C:\capture\scene\a.tif",raw"\\server\share\b.tif"]
    timed=artifact_portability_fixture(foreign)
    base=(transform=PlanarTransform([0.1 0.0;0.0 -0.2],[1.0,2.0]),length_unit="mm",coordinate_frame="provided_frame")
    mktempdir() do dir
        local_input=joinpath(dir,"relocated-input.tif");write(local_input,"source input")
        files=export_calibrated_table(joinpath(dir,"tracks.csv"),timed;base...,protected_paths=[local_input])
        wrapper=TOML.parsefile(files.metadata_path)
        raw=deepcopy(wrapper["metadata"])
        @test raw["protected_locators"]==sort([foreign;local_input])
        @test H._cal_unpack(raw["timing_snapshot"]["value"])["protected_paths"]==sort(foreign)
        report=load_calibrated_table_metadata(files.metadata_path)
        data=calibrated_table_data(report)
        @test report._sha256==wrapper["metadata_sha256"]
        @test data["protected_locators"]==raw["protected_locators"]
        @test data["local_protected_paths"]==sort([local_input;files.csv_path;files.metadata_path])
        @test data["verification"]["csv_structure_and_hash_at_load"]===true
        csvbytes,metabytes=read(files.csv_path),read(files.metadata_path)
        moved=joinpath(dir,"moved.csv");write(moved,csvbytes)
        restored=load_calibrated_table_metadata(files.metadata_path;csv_path=moved)
        copydata=calibrated_table_data(restored)
        @test copydata["protected_locators"]==raw["protected_locators"]
        @test moved in copydata["local_protected_paths"]
        @test !(files.csv_path in copydata["local_protected_paths"])
        for target in (moved,files.metadata_path,local_input)
            before=read(target)
            @test_throws ArgumentError export_calibrated_table(target,timed;base...,overwrite=true,protected_paths=copydata["local_protected_paths"])
            @test read(target)==before
        end
        metadata_only=load_calibrated_table_metadata(files.metadata_path;csv_path=joinpath(dir,"missing.csv"),verify_csv=false)
        @test calibrated_table_data(metadata_only)["verification"]["csv_structure_and_hash_at_load"]===false
        @test calibrated_table_data(metadata_only)["protected_locators"]==raw["protected_locators"]
        for relative in ("/absolute/table.csv",raw"C:\absolute\table.csv",raw"\\server\share\table.csv",raw"C:table.csv")
            write(files.metadata_path,metabytes)
            artifact_rewrite_metadata(files.metadata_path,d->(d["csv_relative_path"]=relative))
            @test_throws ArgumentError load_calibrated_table_metadata(files.metadata_path;csv_path=moved,verify_csv=false)
        end
        write(files.metadata_path,metabytes)
        artifact_rewrite_metadata(files.metadata_path,d->(d["csv_relative_path"]=raw"foreign\table.csv"))
        @test load_calibrated_table_metadata(files.metadata_path;csv_path=moved) isa CalibratedTableMetadata
        if !Sys.iswindows()
            @test_throws ArgumentError load_calibrated_table_metadata(files.metadata_path;verify_csv=false)
        end
        write(files.metadata_path,metabytes)
        @test_throws ArgumentError export_calibrated_table(joinpath(dir,"bad.csv"),timed;base...,protected_paths=foreign)
        @test_throws ArgumentError export_calibrated_table(first(foreign),timed;base...)
        @test_throws ArgumentError load_calibrated_table_metadata(files.metadata_path;csv_path=first(foreign))
        @test_throws ArgumentError load_calibrated_table_metadata(first(foreign);csv_path=moved)
        @test read(files.csv_path)==csvbytes
        @test read(files.metadata_path)==metabytes
        @test !ispath(joinpath(dir,"bad.csv"))
    end
end
