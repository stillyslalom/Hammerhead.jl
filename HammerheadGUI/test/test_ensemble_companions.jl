using Test,HammerheadGUI
using HammerheadGUI.Controllers,HammerheadGUI.Hammerhead,HammerheadGUI.GLMakie

function gui_ensemble_fixture()
    a=[mod(37i+13j+7i*j,251)/250 for i in 1:48,j in 1:48]
    p=PIVParameters(window_size=16,overlap=8,max_iterations=5,convergence_tol=Inf,
        uod_enable=false,validation=(),replace_outliers=false,uncertainty=true)
    scale=PhysicalScale(pixel_size=.02,dt=.01,length_unit="mm",time_unit="s")
    packet=Ref{Any}(nothing)
    raw=run_piv_ensemble([(a,circshift(a,(1,2))),(a,circshift(a,(1,2)))],[p,p];
        threaded=false,progress=false,scale,on_diagnostics=d->(packet[]=d))
    ordinary=Ref{Any}(nothing)
    single=run_piv(a,circshift(a,(1,2)),p;threaded=false,scale,on_diagnostics=d->(ordinary[]=d))
    raw,packet[],single,ordinary[]
end
function gui_ensemble_file(path,entries,pools=fill(nothing,length(entries)),ordinary=fill(nothing,length(entries)))
    Hammerhead.jldopen(path,"w") do f
        f["format_version"]=1
        for i in eachindex(entries)
            key="results/"*lpad(string(i),6,'0')
            f[key]=entries[i]
            pools[i]===nothing || Hammerhead._write_ensemble_execution_diagnostics(f,key,pools[i],entries[i])
            ordinary[i]===nothing || Hammerhead._write_execution_diagnostics(f,key,ordinary[i])
        end
    end
    path
end
@noinline function gui_ensemble_previous_view(path)
    ex=ResultExplorer(path;lazy=true);set_companion_inspection!(ex)
    fig=result_explorer(ex)
    colorbuffer(fig;px_per_unit=1)
    old=WeakRef(current_result(ex).u)
    set_frame!(ex,3)
    colorbuffer(fig;px_per_unit=1)
    ex,fig,old
end
function gui_quality_pages(fig)
    pager=only(filter(b->b isa Label && startswith(b.text[],"report page "),fig.content))
    previous=only(filter(b->b isa Button && b.label[]=="previous report page",fig.content))
    next=only(filter(b->b isa Button && b.label[]=="next report page",fig.content))
    body=only(filter(b->b isa Label && startswith(b.text[],"Status: "),fig.content))
    count=parse(Int,last(split(pager.text[]," / ")))
    for _ in 1:count;previous.clicks[]+=1;end
    pages=String[]
    for _ in 1:count;push!(pages,body.text[]);next.clicks[]+=1;end
    for _ in 1:count;previous.clicks[]+=1;end
    join(pages,"\n")
end
function gui_ensemble_mouse!(fig,block)
    box=block.layoutobservables.computedbbox[]
    events(fig).mouseposition[]=Tuple(Float64.(box.origin+box.widths/2))
    events(fig).mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.left,Mouse.press)
    events(fig).mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.left,Mouse.release)
end

@testset "Ensemble companion explorer and whole-file reports" begin
    C=HammerheadGUI.Controllers
    raw,pool,single,ordinary=gui_ensemble_fixture()
    mktempdir() do dir
        path=gui_ensemble_file(joinpath(dir,"mixed.jld2"),[raw,single,single],[pool,nothing,nothing],[nothing,ordinary,nothing])
        ex=ResultExplorer(path;lazy=true)
        @test !ex.companion_enabled[] && ex.results.companions.ensemble_diagnostics===nothing
        set_companion_inspection!(ex)
        @test ex.results.companions.state===:ensemble_verified
        @test ex.results.companions.ensemble_diagnostics isa EnsemblePIVExecutionDiagnostics
        @test ex.results.companions.history===nothing && ex.results.companions.diagnostics===nothing
        @test current_result(ex).x≈raw.x.*.02
        @test current_result(ex).u≈raw.u.*2 nans=true
        @test occursin("speed (mm/s)",field_label(current_result(ex),:magnitude))
        @test execution_diagnostics_data(ex.results.companions.ensemble_diagnostics;result=raw)["verification"]["measurement_field_binding_checked"]
        @test_throws ArgumentError execution_diagnostics_data(pool;result=current_result(ex))
        summary=companion_summary(ex)
        for text in ("2 input pairs","one pooled sweep","iterations and tolerance ignored","Window/pair opportunities",
                "finite flat nonzero","Source-support pairs","processing px","before validation","Per-node ensemble history is unavailable")
            @test occursin(text,summary)
        end
        select_nearest!(ex,current_result(ex).x[2],current_result(ex).y[2])
        @test occursin("not this selected node",describe_companion_selection(ex))
        @test occursin("mm/s",describe_selection(ex))
        @test !any(x->x===raw,values(ex.results.companions))
        set_frame!(ex,2)
        @test ex.results.companions.ensemble_diagnostics===nothing
        @test occursin("not verified against the displayed vector values",companion_summary(ex))
        set_frame!(ex,3)
        @test occursin("not recorded",companion_summary(ex))
        @test !occursin("ensemble",companion_summary(ex))
        set_frame!(ex,1)
        current_result(ex).u[1]+=123
        @test_throws ArgumentError companion_summary(ex)
        @test occursin("physical display fields changed",ex.status[])
        set_companion_inspection!(ex,false);set_companion_inspection!(ex)
        @test current_result(ex).u[1]!=raw.u[1]*2+123

        # Stored metadata can decode without a raw read, but enabling inspection
        # must reject changed measurement fields before replacing the display.
        bad=joinpath(dir,"bad.jld2");cp(path,bad)
        Hammerhead.jldopen(bad,"r+") do f
            changed=deepcopy(raw);changed.v[1]+=1
            delete!(f,"results/000001");f["results/000001"]=changed
        end
        @test load_ensemble_execution_diagnostics(ResultFile(bad),1) isa EnsemblePIVExecutionDiagnostics
        badex=ResultExplorer(bad;lazy=true)
        before=current_result(badex);bundle=badex.results.companions
        @test_throws ArgumentError set_companion_inspection!(badex)
        @test !badex.companion_enabled[] && current_result(badex)===before && badex.results.companions===bundle
        set_companion_inspection!(badex,false)
        # Corruption on the second frame also preserves selection and enabled mode.
        broken=gui_ensemble_file(joinpath(dir,"broken.jld2"),[raw,raw,single],[pool,pool,nothing])
        Hammerhead.jldopen(broken,"r+") do f
            changed=deepcopy(raw);changed.u[1]+=1
            delete!(f,"results/000002");f["results/000002"]=changed
        end
        bx=ResultExplorer(broken;lazy=true);set_companion_inspection!(bx)
        select_nearest!(bx,current_result(bx).x[2],current_result(bx).y[2])
        previous=current_result(bx);selection=bx.selection[];packet=bx.results.companions
        @test_throws ArgumentError set_frame!(bx,2)
        @test bx.frame[]==1 && bx.selection[]==selection && current_result(bx)===previous && bx.results.companions===packet
        @test_throws ArgumentError (bx.frame[]=2)
        @test bx.frame[]==1 && bx.companion_enabled[]
        set_frame!(bx,3);@test isempty(bx.status[])

        conflicting=gui_ensemble_file(joinpath(dir,"conflict.jld2"),[raw],[pool],[ordinary])
        @test_throws ArgumentError set_companion_inspection!(ResultExplorer(conflicting;lazy=true))
        stereo=StereoPIVResult(raw.x,raw.y,0.,raw.u,raw.v,raw.u,raw.uncertainty_u,raw.uncertainty_v,
            raw.uncertainty_u,raw.outliers,raw.mask,raw,raw,raw.parameters)
        wrongkind=gui_ensemble_file(joinpath(dir,"wrong-kind.jld2"),[raw,stereo],[pool,nothing])
        Hammerhead.jldopen(wrongkind,"r+") do f
            entry=f["ensemble_execution_diagnostics/000001"]
            entry["result_key"]="results/000002"
            f["ensemble_execution_diagnostics/000002"]=entry
        end
        sx=ResultExplorer(wrongkind;lazy=true);set_companion_inspection!(sx)
        @test_throws ArgumentError set_frame!(sx,2)
        @test sx.frame[]==1
        malformed=gui_ensemble_file(joinpath(dir,"unknown-version.jld2"),[single])
        Hammerhead.jldopen(malformed,"r+") do f;f["ensemble_execution_diagnostics_format_version"]=99;end
        @test_throws ArgumentError set_companion_inspection!(ResultExplorer(malformed;lazy=true))

        @testset "Report provenance, captured requests and protected destinations" begin
            @test quality_report_data(explorer_quality_report(ex))["quality_report_format_version"]==1
            v4=explorer_quality_report(ex;include_execution_diagnostics=true,include_ensemble_execution_diagnostics=true)
            data=quality_report_data(v4)
            @test data["quality_report_format_version"]==4
            @test data["provenance"]["association"]=="unassociated"
            @test data["provenance"]["source_selection"]=="whole_file"
            classification=data["ensemble_execution_diagnostics"]["classification"]
            @test classification["planar_entries_examined"]==3
            @test classification["recorded_ensemble_entries"]==1
            @test classification["recorded_planar_iteration_entries"]==1
            @test classification["entries_without_execution_metadata"]==1
            @test_throws ArgumentError explorer_quality_report(ResultExplorer(raw;path))
            bytes=read(path)
            @test_throws ArgumentError save_explorer_quality_report(path,ex;include_ensemble_execution_diagnostics=true)
            @test read(path)==bytes
            destination=joinpath(dir,"quality.toml")
            saved=save_explorer_quality_report(destination,ex;include_ensemble_execution_diagnostics=true)
            @test quality_report_data(load_quality_report(destination))==quality_report_data(saved)
            for changed_keys in (keys->pop!(keys),keys->push!(keys,last(keys)),keys->reverse!(keys))
                tampered=ResultFile(path);changed_keys(tampered.entry_keys)
                tx=ResultExplorer(tampered)
                @test_throws ArgumentError explorer_quality_report(tx)
                @test_throws ArgumentError explorer_quality_report(tx;include_ensemble_execution_diagnostics=true)
                @test_throws ArgumentError save_explorer_quality_report(joinpath(dir,"invalid-index.toml"),tx)
                @test_throws ArgumentError C._ResultQualityController(tx)
                @test !isfile(joinpath(dir,"invalid-index.toml"))
            end
            # A report window owns its independent opening snapshot.
            opening=ResultFile(path);opening_ex=ResultExplorer(opening)
            detached=C._ResultQualityController(opening_ex)
            @test detached.index!==opening && detached.index.entry_keys!==opening.entry_keys
            pop!(opening.entry_keys)
            @test quality_report_data(C._generate_result_quality!(detached))["provenance"]["source_index_entries"]==3
            # Mutation after the request starts cannot change its population.
            for trigger in (:running,:picker)
                request=C._ResultQualityController(ex);request.ensemble[]=true
                original=request.index
                called=Ref(false)
                mutate=()->pop!(original.entry_keys)
                listener=trigger===:running ? on(v->v && mutate(),request.running) : nothing
                captured=C._generate_result_quality!(request;path_picker=trigger===:picker ?
                    ()->(mutate();destination) : nothing)
                listener===nothing || off(listener)
                @test quality_report_data(captured)["provenance"]["source_index_entries"]==3
                @test quality_report_data(captured)["ensemble_execution_diagnostics"]["classification"]["entries_without_execution_metadata"]==1
                previous_saved=read(destination)
                @test_throws ArgumentError C._generate_result_quality!(request;path_picker=()->(called[]=true;destination))
                @test !called[] && !request.running[] && request.report[]===captured
                @test occursin("Report failed; previous report retained",request.status[])
                @test read(destination)==previous_saved
                @test read(path)==bytes
            end
            controller=C._ResultQualityController(ex);controller.ensemble[]=true
            busy=Ref(false)
            listener=on(controller.running) do running
                running || return
                controller.ensemble[]=false;controller.history[]=true
                controller.index=ResultFile(bad) # next source choice cannot alter the captured index
                busy[]=try C._generate_result_quality!(controller);false catch err;err isa ArgumentError end
            end
            picked=Ref(false)
            report=C._generate_result_quality!(controller;path_picker=()->begin
                picked[]=controller.running[];controller.execution[]=true;destination
            end)
            off(listener)
            controller.index=ex.results.source
            @test busy[] && picked[] && !controller.running[]
            captured=quality_report_data(report)
            @test captured["quality_report_format_version"]==4 && !haskey(captured,"measurement_history") && !haskey(captured,"execution_diagnostics")
            @test occursin(captured["provenance"]["source_sha256"],C._result_quality_text(controller))
            @test C._generate_result_quality!(controller;path_picker=()->nothing)===nothing
            @test controller.report[]===report && !controller.running[]
            @test_throws ErrorException C._generate_result_quality!(controller;path_picker=()->error("dialog failed"))
            @test controller.report[]===report && occursin("dialog failed",controller.status[])
            @test_throws ArgumentError C._generate_result_quality!(controller;path_picker=()->path)
            @test controller.report[]===report && read(path)==bytes
            throwing=on(controller.running) do value;value && error("busy observer failure");end
            @test_throws ErrorException C._generate_result_quality!(controller)
            off(throwing)
            @test !controller.running[] && controller.report[]===report
            controller.ensemble[]=true;controller.history[]=false;controller.execution[]=false
            publication=on(controller.report) do value;value===report || error("report observer failure");end
            @test_throws ErrorException C._generate_result_quality!(controller)
            off(publication)
            @test !controller.running[] && controller.report[]===report && occursin("report observer failure",controller.status[])
            changing=gui_ensemble_file(joinpath(dir,"changing.jld2"),[raw],[pool])
            cc=C._ResultQualityController(ResultExplorer(changing;lazy=true));cc.ensemble[]=true
            @test_throws ArgumentError C._generate_result_quality!(cc;path_picker=()->begin
                gui_ensemble_file(changing,[single]);joinpath(dir,"must-not-exist.toml")
            end)
            @test !isfile(joinpath(dir,"must-not-exist.toml")) && cc.report[]===nothing && !cc.running[]
        end

        @testset "Hidden rendering, launch, paging and release" begin
            released,releasefig,old=gui_ensemble_previous_view(path)
            GC.gc(true)
            @test old.value===nothing && released.frame[]==3
            viewex=ResultExplorer(path;lazy=true);set_companion_inspection!(viewex)
            viewex.selection[]=CartesianIndex(2,2)
            launched=Ref{Any}(nothing)
            fig=result_explorer(viewex;size=(1000,700),report_view_launcher=f->(launched[]=f))
            @test size(colorbuffer(fig;px_per_unit=1))==(700,1000)
            launch=only(filter(b->b isa Button && b.label[]=="whole-file quality report",fig.content))
            gui_ensemble_mouse!(fig,launch)
            @test launched[] isa Figure
            reportfig=launched[]
            @test size(colorbuffer(reportfig;px_per_unit=1))==(650,900)
            toggles=filter(b->b isa Toggle,reportfig.content);last(toggles).active[]=true
            generate=only(filter(b->b isa Button && b.label[]=="generate report",reportfig.content))
            gui_ensemble_mouse!(reportfig,generate)
            colorbuffer(reportfig;px_per_unit=1)
            visible_body=only(filter(b->b isa Label && startswith(b.text[],"Status: "),reportfig.content))
            @test length(split(visible_body.text[],'\n'))>=8
            @test maximum(length,split(visible_body.text[],'\n'))>=60
            text=gui_quality_pages(reportfig)
            @test occursin("Format: 4",text) && occursin("unassociated",text)
            @test occursin(Hammerhead._experiment_file_digest(path),replace(text,"\n"=>""))
            set_frame!(viewex,2)
            @test gui_quality_pages(reportfig)==text
            large=result_quality_report(viewex;size=(1100,800))
            @test size(colorbuffer(large;px_per_unit=1))==(800,1100)
            last(filter(b->b isa Toggle,large.content)).active[]=true
            gui_ensemble_mouse!(large,only(filter(b->b isa Button && b.label[]=="generate report",large.content)))
            colorbuffer(large;px_per_unit=1)
            @test occursin("Format: 4",gui_quality_pages(large))
            for (window,width,height) in ((reportfig,900,650),(large,1100,800)), b in window.content
                b isa Union{Button,Toggle,Label} || continue
                box=b.layoutobservables.computedbbox[]
                @test all(isfinite,box.origin) && all(isfinite,box.widths)
                @test box.origin[1]>=-1 && box.origin[2]>=-1 &&
                    box.origin[1]+box.widths[1]<=width+1 && box.origin[2]+box.widths[2]<=height+1
            end
            if haskey(ENV,"HAMMERHEAD_ENSEMBLE_SCREENSHOT")
                set_frame!(viewex,1);viewex.selection[]=CartesianIndex(2,2)
                GLMakie.save(ENV["HAMMERHEAD_ENSEMBLE_SCREENSHOT"],fig)
                GLMakie.save(replace(ENV["HAMMERHEAD_ENSEMBLE_SCREENSHOT"],".png"=>"-report.png"),reportfig)
                GLMakie.save(replace(ENV["HAMMERHEAD_ENSEMBLE_SCREENSHOT"],".png"=>"-report-large.png"),large)
            end
        end
    end
end
