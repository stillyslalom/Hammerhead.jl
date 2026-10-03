using Test, HammerheadGUI
using HammerheadGUI.Hammerhead
using HammerheadGUI.GLMakie

function companion_native_file(path,entries,histories,diagnostics=fill(nothing,length(entries)))
    Hammerhead.jldopen(path,"w") do file
        file["format_version"]=1
        for i in eachindex(entries)
            key="results/"*lpad(string(i),6,'0')
            file[key]=entries[i]
            histories[i]===nothing || Hammerhead._write_measurement_history(file,key,histories[i])
            diagnostics[i]===nothing || Hammerhead._write_execution_diagnostics(file,key,diagnostics[i])
        end
    end
end

# Keep the old payload outside test-macro temporaries in the calling scope.
@noinline function companion_previous_payload(path)
    explorer=ResultExplorer(path;lazy=true)
    set_companion_inspection!(explorer)
    old=WeakRef(explorer.results.companions.history._data["primary_u"])
    set_frame!(explorer,2)
    old,explorer
end

@testset "Recorded companion inspection" begin
    C=HammerheadGUI.Controllers
    a=reshape(Float32.(sin.(1:4096)),64,64)
    b=circshift(a,(1,2))
    scale=PhysicalScale(pixel_size=.2,dt=.5,length_unit="mm",time_unit="s")
    packet=Ref{Any}();execution=Ref{Any}()
    raw=run_piv(a,b,PIVParameters(window_size=16,overlap=8,max_iterations=2,uncertainty=true);
        roi=ROI(9:56,13:60),scale,threaded=false,on_measurement_history=h->(packet[]=h),on_diagnostics=d->(execution[]=d))
    mktempdir() do dir
        good=joinpath(dir,"good.jld2")
        companion_native_file(good,[raw,raw],[packet[],packet[]],[execution[],execution[]])
        ex=ResultExplorer(good;lazy=true)
        @test !ex.companion_enabled[] && ex.results.companions.history===nothing
        @test occursin("off",companion_summary(ex))
        select_nearest!(ex,current_result(ex).x[2],current_result(ex).y[2])
        selected=ex.selection[]
        set_companion_inspection!(ex)
        @test ex.companion_enabled[] && ex.results.inspection && ex.selection[]==selected
        @test ex.results.companions.history isa PIVMeasurementHistory && ex.results.companions.diagnostics isa PIVExecutionDiagnostics
        @test verify_measurement_history(ex.results.companions.history,raw)
        @test current_result(ex).x≈raw.x.*.2
        @test current_result(ex).u≈raw.u.*.4 nans=true
        @test current_result(ex).scale.length_unit=="mm"
        @test occursin("raw measurement binding verified",companion_summary(ex))
        @test occursin("not verified against the displayed vector values",companion_summary(ex))
        @test occursin("2/2 sweeps",companion_summary(ex))
        @test occursin("not evaluated",companion_summary(ex))
        @test occursin("Primary residual mean/RMS/max",companion_summary(ex))
        text=describe_companion_selection(ex)
        @test occursin("Raw x/y:",text) && occursin("px",text) && occursin("not establish applicability",text)
        @test occursin("mm/s",describe_selection(ex))
        node=measurement_history_at(ex.results.companions.history,selected)
        @test node.x==raw.x[2] && node.y==raw.y[2] # selection maps by grid index, not converted coordinate equality
        @test !any(v->v===raw,values(ex.results.companions))
        @test current_result(ex)===current_result(ex)
        old,release_ex=companion_previous_payload(good)
        GC.gc(true)
        @test old.value===nothing
        @test release_ex.results.index==2
        set_frame!(ex,2)
        @test isempty(ex.derived_cache)
        @test ex.results.index==2 && ex.results.companions.history!==nothing
        # A stale mutation of displayed physical arrays never masquerades as
        # a checked raw measurement; reenable deliberately reloads the source.
        current_result(ex).u[1]=123
        @test_throws ArgumentError companion_summary(ex)
        @test occursin("physical display fields changed",ex.status[])
        @test_throws ArgumentError describe_companion_selection(ex)
        set_companion_inspection!(ex,false)
        @test ex.results.companions.history===nothing && occursin("off",companion_summary(ex))
        set_companion_inspection!(ex,true)
        @test current_result(ex).u[1]!=123 && isempty(ex.status[])
        ex.results.companions.history._data["primary_u"][1]=123
        @test_throws ArgumentError companion_summary(ex)
        @test occursin("mutated after capture",ex.status[])
        set_companion_inspection!(ex,false)
        ex.companion_enabled[]=true
        @test ex.companion_enabled[] && occursin("verified",companion_summary(ex))

        changed=deepcopy(raw);changed.v[1]+=1
        broken=joinpath(dir,"broken.jld2")
        companion_native_file(broken,[raw,changed,raw],[packet[],packet[],nothing],[execution[],nothing,nothing])
        broken_ex=ResultExplorer(broken;lazy=true)
        set_companion_inspection!(broken_ex)
        select_nearest!(broken_ex,current_result(broken_ex).x[2],current_result(broken_ex).y[2])
        set_field!(broken_ex,:vorticity);C.current_field_values(broken_ex)
        set_tool!(broken_ex,:profile);C.click!(broken_ex,current_result(broken_ex).x[1],current_result(broken_ex).y[1])
        saved_selection=broken_ex.selection[];saved=current_result(broken_ex);saved_packet=broken_ex.results.companions
        @test_throws ArgumentError set_frame!(broken_ex,2)
        @test broken_ex.frame[]==1 && broken_ex.selection[]==saved_selection
        @test current_result(broken_ex)===saved && broken_ex.results.companions===saved_packet && broken_ex.companion_enabled[]
        @test broken_ex.tool[]==:profile && length(broken_ex.derived_cache)==1
        @test_throws ArgumentError (broken_ex.frame[]=2)
        @test broken_ex.frame[]==1 && current_result(broken_ex)===saved && broken_ex.results.companions===saved_packet
        set_frame!(broken_ex,3)
        @test occursin("not recorded",companion_summary(broken_ex)) && isempty(describe_companion_selection(broken_ex))
        @test isempty(broken_ex.status[]) && broken_ex.results.companions.history===nothing
        badfirst=joinpath(dir,"bad-first.jld2")
        companion_native_file(badfirst,[changed],[packet[]])
        mode_ex=ResultExplorer(badfirst;lazy=true)
        select_nearest!(mode_ex,current_result(mode_ex).x[1],current_result(mode_ex).y[1])
        before=current_result(mode_ex);before_selection=mode_ex.selection[];before_bundle=mode_ex.results.companions
        @test_throws ArgumentError set_companion_inspection!(mode_ex,true)
        @test !mode_ex.companion_enabled[] && !mode_ex.results.inspection && current_result(mode_ex)===before
        @test mode_ex.selection[]==before_selection && mode_ex.results.companions===before_bundle
        @test_throws ArgumentError (mode_ex.companion_enabled[]=true)
        @test !mode_ex.companion_enabled[] && mode_ex.results.companions===before_bundle
        eager=ResultExplorer(raw;path=good)
        @test_throws ArgumentError set_companion_inspection!(eager,true)
        @test occursin("no indexed native association",companion_summary(eager))
        @test !eager.companion_enabled[]

        stereo=StereoPIVResult(raw.x,raw.y,0.,raw.u,raw.v,raw.u,raw.uncertainty_u,raw.uncertainty_v,raw.uncertainty_u,raw.outliers,raw.mask,raw,raw,raw.parameters)
        mixed=joinpath(dir,"mixed.jld2");companion_native_file(mixed,[raw,stereo],[packet[],nothing])
        mixed_ex=ResultExplorer(mixed;lazy=true);set_companion_inspection!(mixed_ex);set_frame!(mixed_ex,2)
        @test occursin("not recorded",companion_summary(mixed_ex)) && occursin("not recorded",describe_companion_selection(mixed_ex))
        unsupported=joinpath(dir,"unsupported-packet.jld2");companion_native_file(unsupported,[raw,stereo],[packet[],packet[]])
        unsupported_ex=ResultExplorer(unsupported;lazy=true);set_companion_inspection!(unsupported_ex)
        @test_throws ArgumentError set_frame!(unsupported_ex,2)
        @test unsupported_ex.frame[]==1
        # A checkpoint payload is not a claimed history/diagnostics index.
        checkpoint_path=joinpath(dir,"checkpoint-payload.jld2")
        save_results(checkpoint_path,raw)
        checkpoint_index=CheckpointResults([checkpoint_path],[Hammerhead._experiment_file_digest(checkpoint_path)])
        checkpoint_ex=ResultExplorer(checkpoint_index)
        @test occursin("checkpoint",companion_summary(checkpoint_ex))
        @test_throws ArgumentError set_companion_inspection!(checkpoint_ex)

        @testset "Offscreen panel, widget recovery and retained payloads" begin
            view_ex=ResultExplorer(broken;lazy=true)
            fig=result_explorer(view_ex;size=(1100,850))
            before=copy(colorbuffer(fig;px_per_unit=1))
            toggle=last(filter(block->block isa Toggle,fig.content))
            plot_axis=first(filter(block->block isa Axis,fig.content))
            toggle.active[]=true
            view_ex.selection[]=CartesianIndex(2,2)
            @test view_ex.companion_enabled[]
            image=copy(colorbuffer(fig;px_per_unit=1))
            enabled_height=plot_axis.scene.viewport[].widths[2]
            @test size(image)==(850,1100) && image!=before
            @test any(block->block isa Label && occursin("Raw x/y",block.text[]),fig.content)
            next_details=only(filter(block->block isa Button && block.label[]=="next details",fig.content))
            previous_details=only(filter(block->block isa Button && block.label[]=="previous details",fig.content))
            next_details.clicks[]+=1
            @test any(block->block isa Label && occursin("Numerical availability",block.text[]),fig.content)
            previous_details.clicks[]+=1
            @test any(block->block isa Label && occursin("Raw x/y",block.text[]),fig.content)
            slider=only(filter(block->block isa Slider,fig.content))
            set_close_to!(slider,2)
            @test view_ex.frame[]==1 && slider.value[]==1 && view_ex.selection[]==CartesianIndex(2,2)
            @noinline function old_companion_view(ex)
                refs=(WeakRef(current_result(ex).u),WeakRef(ex.results.companions.history._data["primary_u"]))
                set_frame!(ex,3)
                refs
            end
            refs=old_companion_view(view_ex);colorbuffer(fig;px_per_unit=1);GC.gc(true)
            @test all(ref->ref.value===nothing,refs)
            toggle.active[]=false
            @test !view_ex.companion_enabled[] && view_ex.results.companions.history===nothing
            @test !isempty(colorbuffer(fig;px_per_unit=1))
            @test plot_axis.scene.viewport[].widths[2]>enabled_height
            set_frame!(view_ex,1);toggle.active[]=true;view_ex.selection[]=CartesianIndex(2,2)
            set_tool!(view_ex,:profile)
            C.click!(view_ex,current_result(view_ex).x[1],current_result(view_ex).y[1])
            C.click!(view_ex,current_result(view_ex).x[end],current_result(view_ex).y[end])
            colorbuffer(fig;px_per_unit=1)
            @test view_ex.profile_data[]!==nothing && count(block->block isa Axis,fig.content)==1
            @test any(block->block isa Label && occursin("Close recorded details",block.text[]),fig.content)
            toggle.active[]=false;colorbuffer(fig;px_per_unit=1)
            @test view_ex.profile_data[]!==nothing && count(block->block isa Axis,fig.content)==2
            set_tool!(view_ex,:inspect);toggle.active[]=true;view_ex.selection[]=CartesianIndex(2,2)
            # Save evidence only when the focused harness supplies its ignored path.
            if haskey(ENV,"HAMMERHEAD_COMPANION_SCREENSHOT")
                GLMakie.save(ENV["HAMMERHEAD_COMPANION_SCREENSHOT"],fig)
            end
            toggle.active[]=false
            @test !any(block->block isa Label && block.visible[] && occursin("Raw x/y",block.text[]),fig.content)
            compact=result_explorer(view_ex;size=(1000,700))
            set_companion_inspection!(view_ex,true);view_ex.selection[]=CartesianIndex(2,2)
            @test size(colorbuffer(compact;px_per_unit=1))==(700,1000)
            if haskey(ENV,"HAMMERHEAD_COMPANION_SCREENSHOT")
                GLMakie.save(replace(ENV["HAMMERHEAD_COMPANION_SCREENSHOT"],".png"=>"-compact.png"),compact)
            end
            set_tool!(view_ex,:profile)
            C.click!(view_ex,current_result(view_ex).x[1],current_result(view_ex).y[1])
            C.click!(view_ex,current_result(view_ex).x[end],current_result(view_ex).y[end])
            @test size(colorbuffer(compact;px_per_unit=1))==(700,1000) && view_ex.profile_data[]!==nothing
            if haskey(ENV,"HAMMERHEAD_COMPANION_SCREENSHOT")
                GLMakie.save(replace(ENV["HAMMERHEAD_COMPANION_SCREENSHOT"],".png"=>"-compact-profile.png"),compact)
            end
            set_companion_inspection!(view_ex,false)
            @test count(block->block isa Axis,compact.content)==2
            if haskey(ENV,"HAMMERHEAD_COMPANION_SCREENSHOT")
                GLMakie.save(replace(ENV["HAMMERHEAD_COMPANION_SCREENSHOT"],".png"=>"-compact-profile-off.png"),compact)
            end
        end
    end
end
