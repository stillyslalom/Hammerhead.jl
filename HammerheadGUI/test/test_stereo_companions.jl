using Test,HammerheadGUI
using HammerheadGUI.Controllers
using HammerheadGUI.Hammerhead
using HammerheadGUI.GLMakie

function gui_workflow_report_pages(fig)
    pager=only(filter(b->b isa Label && startswith(b.text[],"page "),fig.content))
    previous=only(filter(b->b isa Button && b.label[]=="previous",fig.content))
    next=only(filter(b->b isa Button && b.label[]=="next",fig.content))
    body=only(filter(b->b isa Label && startswith(b.text[],"selected output:"),fig.content))
    count=parse(Int,last(split(pager.text[]," / ")))
    for _ in 1:count
        previous.clicks[]+=1
    end
    pages=String[]
    for _ in 1:count
        push!(pages,body.text[])
        next.clicks[]+=1
    end
    for _ in 1:count
        previous.clicks[]+=1
    end
    pages
end

function gui_stereo_companion_fixture()
    grid=DewarpGrid(x=1.:48.,y=48.:-1.:1.)
    cameras=(PinholeCamera([100. 0 15. 0;0 100. 0 0;0 0 1. 100.]),
             PinholeCamera([100. 0 -15. 0;0 100. 0 0;0 0 1. 100.]))
    dw1,dw2=map(c->ImageDewarper(c,grid,(48,48)),cameras)
    a=reshape(sin.(Float64.(1:2304)),48,48)
    b=reshape(cos.(Float64.(1:2304).*1.3),48,48)
    packet=Ref{Any}(nothing)
    raw=run_piv_stereo(a,circshift(a,(1,2)),b,circshift(b,(-1,1)),dw1,dw2,
        PIVParameters(window_size=16,overlap=8,max_iterations=3,convergence_tol=1e6,uncertainty=true);
        roi=ROI(5:44,7:46),threaded=false,
        scale=PhysicalScale(pixel_size=.2,dt=.5,length_unit="mm",time_unit="s"),
        on_diagnostics=d->(packet[]=d))
    raw,packet[]
end

@testset "Saved experiment execution-aware reports" begin
    mktempdir() do dir
        a=reshape(Float64.(sin.(1:1024)),32,32)
        files=[joinpath(dir,"a.png"),joinpath(dir,"b.png")]
        for (path,image) in zip(files,(a,circshift(a,(1,2))))
            Hammerhead.FileIO.save(path,Hammerhead.Gray.((image.+1)./2))
        end
        recipe=PIVRecipe(PIVParameters(window_size=16,overlap=8,max_iterations=2);
            threaded=false,scale=PhysicalScale(pixel_size=.02,dt=.001,length_unit="mm",time_unit="s"))
        record=ExperimentRecord([(files[1],files[2])],recipe)
        history=joinpath(dir,"history.jld2");output=joinpath(dir,"vectors.jld2")
        save_experiment(history,record)
        run=replay_experiment(record;output,run_record=history,record_diagnostics=true,record_measurement_history=true)
        ec=ExperimentController(history)
        ec.output_path[]=output
        @test ec.last_run[].run_id==run.run_id
        @test quality_report_data(experiment_quality_report(ec))["quality_report_format_version"]==1
        @test quality_report_data(experiment_quality_report(ec;include_measurement_history=true))["quality_report_format_version"]==2
        report=experiment_quality_report(ec;include_execution_diagnostics=true)
        data=quality_report_data(report)
        @test data["quality_report_format_version"]==3 && haskey(data,"execution_diagnostics")
        core=quality_report_data(quality_report(ec.record[],ec.last_run[];include_execution_diagnostics=true))
        @test all(data[k]==core[k] for k in ("groups","provenance","execution_diagnostics"))
        @test data["provenance"]["run_id"]==run.run_id
        both=experiment_quality_report(ec;include_measurement_history=true,include_execution_diagnostics=true)
        @test quality_report_data(both)["quality_report_format_version"]==3
        target=joinpath(dir,"execution-quality.toml")
        save_experiment_quality_report(target,ec;include_execution_diagnostics=true)
        @test quality_report_data(load_quality_report(target))["execution_diagnostics"]==data["execution_diagnostics"]
        for protected in (history,output,files...)
            bytes=read(protected)
            @test_throws ArgumentError save_experiment_quality_report(protected,ec;include_execution_diagnostics=true)
            @test read(protected)==bytes
        end
        destination=Ref(joinpath(dir,"view-quality.toml"))
        toggles=Ref{Any}(nothing)
        calls=Ref(0)
        picker=()->begin
            calls[]+=1
            # Changed next choices must not alter the captured report request.
            toggles[][2].active[]=false;toggles[][3].active[]=false
            destination[]
        end
        fig=experiment_workflow(ec;report_path_picker=picker,size=(1100,800))
        toggles[]=filter(b->b isa Toggle,fig.content)
        @test length(toggles[])==3
        toggles[][2].active[]=true;toggles[][3].active[]=true
        save_button=only(filter(b->b isa Button && startswith(b.label[],"save quality report"),fig.content))
        save_button.clicks[]+=1
        @test calls[]==1 && ec.status[]=="quality report saved"
        saved=quality_report_data(load_quality_report(destination[]))
        @test saved["quality_report_format_version"]==3 && haskey(saved,"measurement_history")
        @test saved["provenance"]["run_id"]==run.run_id
        report_pages=gui_workflow_report_pages(fig)
        @test occursin("Reported run: "*run.run_id,replace(join(report_pages),"\n"=>""))
        @test size(colorbuffer(fig;px_per_unit=1,visible=false))==(800,1100)
        if haskey(ENV,"HAMMERHEAD_EXECUTION_REPORT_SCREENSHOT")
            GLMakie.save(ENV["HAMMERHEAD_EXECUTION_REPORT_SCREENSHOT"],fig;visible=false)
        end
        # Refused scans/saves keep the previous report and its own provenance.
        destination[]=output;before=read(output)
        save_button.clicks[]+=1
        @test occursin("failed",ec.status[]) && read(output)==before
        retained_pages=gui_workflow_report_pages(fig)
        @test occursin("Reported run: "*run.run_id,replace(join(retained_pages),"\n"=>""))
        @test quality_report_data(load_quality_report(joinpath(dir,"view-quality.toml")))["provenance"]["run_id"]==run.run_id
        ec.running[]=true;save_button.clicks[]+=1
        @test calls[]==2
        @test_throws ArgumentError experiment_quality_report(ec;include_execution_diagnostics=true)
        ec.running[]=false
        stale=read(output)
        changed=deepcopy(ResultFile(output)[1]);changed.u[1]=123.
        save_results(output,[changed])
        @test_throws ArgumentError experiment_quality_report(ec;include_execution_diagnostics=true)
        write(output,stale)
    end
end
function gui_stereo_companion_file(path,entries,packets;planar=nothing)
    Hammerhead.jldopen(path,"w") do file
        file["format_version"]=1
        for i in eachindex(entries)
            key=Hammerhead.result_key(i)
            file[key]=entries[i]
            packets[i]===nothing || Hammerhead._write_stereo_execution_diagnostics(file,key,packets[i],entries[i])
            planar===nothing || Hammerhead._write_execution_diagnostics(file,key,planar)
        end
    end
    path
end
@noinline function gui_stereo_old_payload(path)
    ex=ResultExplorer(path;lazy=true)
    set_companion_inspection!(ex)
    refs=(WeakRef(current_result(ex).u),WeakRef(current_result(ex).cam1.u))
    set_frame!(ex,2)
    ex,refs
end

@testset "Stereo recorded processing details" begin
    C=HammerheadGUI.Controllers
    raw,packet=gui_stereo_companion_fixture()
    mktempdir() do dir
        good=gui_stereo_companion_file(joinpath(dir,"good.jld2"),[raw,raw,raw],[packet,packet,nothing])
        ex=ResultExplorer(good;lazy=true)
        select_nearest!(ex,current_result(ex).x[2],current_result(ex).y[2])
        selection=ex.selection[]
        set_companion_inspection!(ex)
        @test ex.results.companions.stereo_diagnostics isa StereoPIVExecutionDiagnostics
        @test ex.results.companions.history===nothing && ex.results.companions.diagnostics===nothing
        @test ex.results.companions.state===:stereo_verified && ex.selection[]==selection
        @test current_result(ex).x≈raw.x.*.2 && current_result(ex).y≈raw.y.*.2
        @test current_result(ex).u≈raw.u.*.4 nans=true
        @test isequal(current_result(ex).cam1.u,raw.cam1.u) # retained camera measurements remain dewarped pixels
        @test field_label(current_result(ex),:magnitude)=="speed (mm/s)"
        @test field_label(raw.cam1,:magnitude)=="|displacement| (px)"
        @test selection_point(current_result(ex),selection)==(raw.x[2]*.2,raw.y[2]*.2)
        summary=companion_summary(ex)
        @test occursin("raw reconstructed and camera measurement binding verified",summary)
        @test occursin("Camera 1",summary) && occursin("Camera 2",summary)
        @test occursin("dewarped px",summary) && occursin("2/3 sweeps",summary)
        @test occursin("Calibration, source inputs",summary)
        @test occursin("not recorded",describe_companion_selection(ex))
        @test occursin("mm/s",describe_selection(ex))
        @test !any(v->v===raw,values(ex.results.companions))
        @test ex.results.companions.stereo_diagnostics._verification===:metadata_only # supplied-result verification does not mutate packet
        current_result(ex).w[1]=123.
        @test_throws ArgumentError companion_summary(ex)
        @test occursin("physical display fields changed",ex.status[])
        set_companion_inspection!(ex,false);set_companion_inspection!(ex,true)
        current_result(ex).cam2.u[1]=123.
        @test_throws ArgumentError describe_companion_selection(ex)
        set_companion_inspection!(ex,false);set_companion_inspection!(ex,true)
        set_frame!(ex,3)
        @test ex.results.companions.stereo_diagnostics===nothing && occursin("not recorded",companion_summary(ex))
        set_frame!(ex,1)
        # Binding corruption is introduced after normal, validated writing.
        bad=gui_stereo_companion_file(joinpath(dir,"bad.jld2"),[raw,raw,raw],[packet,packet,nothing])
        Hammerhead.jldopen(bad,"r+") do file
            changed=deepcopy(file[Hammerhead.result_key(2)]);changed.w[1]=123.
            delete!(file,Hammerhead.result_key(2));file[Hammerhead.result_key(2)]=changed
        end
        bad_ex=ResultExplorer(bad;lazy=true);set_companion_inspection!(bad_ex)
        bad_ex.selection[]=CartesianIndex(2,2)
        old=current_result(bad_ex);bundle=bad_ex.results.companions
        @test_throws ArgumentError set_frame!(bad_ex,2)
        @test bad_ex.frame[]==1 && bad_ex.selection[]==CartesianIndex(2,2)
        @test current_result(bad_ex)===old && bad_ex.results.companions===bundle && bad_ex.companion_enabled[]
        @test_throws ArgumentError (bad_ex.frame[]=2)
        @test bad_ex.frame[]==1 && current_result(bad_ex)===old
        set_frame!(bad_ex,3)
        @test isempty(bad_ex.status[]) && bad_ex.results.companions.stereo_diagnostics===nothing
        set_companion_inspection!(bad_ex,false);set_frame!(bad_ex,2)
        before=current_result(bad_ex);before_bundle=bad_ex.results.companions
        @test_throws ArgumentError set_companion_inspection!(bad_ex)
        @test !bad_ex.companion_enabled[] && current_result(bad_ex)===before && bad_ex.results.companions===before_bundle
        # Wrong-kind metadata is refused even when a result would be unsupported.
        wrong=gui_stereo_companion_file(joinpath(dir,"wrong.jld2"),[raw],[packet];planar=packet.cam1)
        @test_throws ArgumentError set_companion_inspection!(ResultExplorer(wrong;lazy=true))
        malformed=gui_stereo_companion_file(joinpath(dir,"malformed.jld2"),[raw],[packet])
        Hammerhead.jldopen(malformed,"r+") do file
            key=Hammerhead._stereo_execution_key(Hammerhead.result_key(1))
            entry=file[key];entry["diagnostics"]["geometry"]["z"]=99.
            delete!(file,key);file[key]=entry
        end
        @test_throws ArgumentError set_companion_inspection!(ResultExplorer(malformed;lazy=true))
        retained,refs=gui_stereo_old_payload(good);GC.gc(true)
        @test all(r->r.value===nothing,refs) && retained.results.index==2

        @testset "Offscreen stereo details and scaled labels" begin
            fig=result_explorer(ex;size=(1000,700))
            ex.selection[]=CartesianIndex(2,2)
            image=colorbuffer(fig;px_per_unit=1,visible=false)
            @test size(image)==(700,1000)
            @test any(b->b isa Label && occursin("Camera 1",b.text[]),fig.content)
            @test any(b->b isa Menu && ("speed",:magnitude) in b.options[],fig.content)
            next=only(filter(b->b isa Button && b.label[]=="next details",fig.content))
            for i in 1:8
                any(b->b isa Label && occursin("Camera 2",b.text[]),fig.content) && break
                next.clicks[]+=1
            end
            @test any(b->b isa Label && occursin("Camera 2",b.text[]),fig.content)
            if haskey(ENV,"HAMMERHEAD_STEREO_COMPANION_SCREENSHOT")
                GLMakie.save(ENV["HAMMERHEAD_STEREO_COMPANION_SCREENSHOT"],fig;visible=false)
            end
            set_companion_inspection!(ex,false)
            @test ex.results.companions.stereo_diagnostics===nothing
            @test !isempty(colorbuffer(fig;px_per_unit=1,visible=false))
        end
    end
end
