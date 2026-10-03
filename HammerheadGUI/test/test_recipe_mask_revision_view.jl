using Test,HammerheadGUI,HammerheadGUI.Controllers
using HammerheadGUI.Hammerhead,HammerheadGUI.GLMakie

function mask_revision_view_fixture(directory;absent=false)
    mkpath(directory)
    image=Float32[.5+.2*sin(i/3)+.15*cos(j/5) for i in 1:48,j in 1:64]
    files=[joinpath(directory,"frame-$i.png") for i in 1:3]
    for (i,path) in enumerate(files)
        Hammerhead.FileIO.save(path,Hammerhead.Gray.(circshift(image,(i-1,2(i-1)))))
    end
    mask=falses(48,64);mask[10:12,15:18].=true
    script=joinpath(directory,"never-execute.jl")
    write(script,"error(\"saved mask reference must not execute scripts\")")
    recipe=PIVRecipe([PIVParameters(window_size=16,overlap=8)];
        preprocessing=[PreprocessStep(:invert_image),PreprocessStep(:invert_image)],
        external_preprocess=ScriptReference(script;entrypoint="manual callback"),
        roi=ROI(5:44,9:56),mask=absent ? nothing : mask,
        scale=PhysicalScale(pixel_size=.02,dt=.001),image_type=Float32,threaded=false,
        predictor_smoothing=false,mask_threshold=.4)
    record=ExperimentRecord([(files[1],files[2]),(files[2],files[3])],recipe)
    source=save_experiment(joinpath(directory,"source.jld2"),record)
    load_experiment(source),source,files
end
mask_revision_button(fig,label)=only(filter(b->b isa Button && b.label[]==label,fig.content))
function mask_revision_render!(fig,screen)
    # Supply elapsed desktop frames for all independent Toggle listeners.
    for _ in 1:4
        tick=events(fig).tick[]
        events(fig).tick[]=GLMakie.Makie.Tick(GLMakie.Makie.UnknownTickState,tick.count+1,tick.time+1.,1.)
    end
    colorbuffer(screen)
end
function mask_revision_mouse!(fig,block,screen)
    sleep(.3);mask_revision_render!(fig,screen)
    rect=block.layoutobservables.computedbbox[];clicks=block isa Button ? block.clicks[] : nothing
    events(fig).mouseposition[]=Tuple(Float64.(rect.origin+rect.widths/2))
    events(fig).mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.left,Mouse.press)
    events(fig).mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.left,Mouse.release)
    block isa Button && @test block.clicks[]==clicks+1
end
function mask_revision_point!(fig,ax,screen,x,y;button=Mouse.left)
    mask_revision_render!(fig,screen)
    rect=ax.scene.viewport[]
    # Explicit original-image mapping, independent of the gesture controller.
    events(fig).mouseposition[]=(rect.origin[1]+(x-.5)/64*rect.widths[1],
        rect.origin[2]+(48.5-y)/48*rect.widths[2])
    events(fig).mousebutton[]=GLMakie.Makie.MouseButtonEvent(button,Mouse.press)
    events(fig).mousebutton[]=GLMakie.Makie.MouseButtonEvent(button,Mouse.release)
end
function mask_revision_wait(rc,mc)
    @test timedwait(()->!rc.running[] && !mc.running[],90.;pollint=.01)==:ok
    @test rc.task[]===nothing && mc.task[]===nothing
end
mask_revision_within(rect,w,h)=all(isfinite,rect.origin) && all(isfinite,rect.widths) &&
    rect.origin[1]>=-1 && rect.origin[2]>=-1 &&
    rect.origin[1]+rect.widths[1]<=w+1 && rect.origin[2]+rect.widths[2]<=h+1
function mask_revision_pages(fig)
    previous=mask_revision_button(fig,"previous text page");next=mask_revision_button(fig,"next text page")
    pager=only(filter(b->b isa Label && startswith(b.text[],"text page "),fig.content))
    body=only(filter(b->b isa Label && startswith(b.text[],"Status: "),fig.content))
    count=parse(Int,last(split(pager.text[]," / ")))
    for _ in 1:count;previous.clicks[]+=1;end
    pages=String[]
    for _ in 1:count;push!(pages,body.text[]);next.clicks[]+=1;end
    replace(join(pages,"\n"),'\n'=>"")
end
function mask_revision_other_fields(recipe)
    data=Hammerhead._experiment_recipe_data(recipe);delete!(data,"mask");data
end

@testset "Saved mask drawing, explicit application, import and guarded save" begin
    mktempdir() do directory
        record,source,files=mask_revision_view_fixture(joinpath(directory,"Unicode λ reference"))
        source_bytes=read(source);input_bytes=read.(files)
        rc=RecipeRevisionController(record);mc=RecipeMaskReferenceController()
        save_choice=Ref{Any}(joinpath(directory,"revised.jld2"));save_calls=Ref(0)
        mask_choice=Ref{Any}(nothing);mask_calls=Ref(0);opened=Ref{Any}(nothing)
        fig=recipe_mask_revision(rc;reference_controller=mc,size=(900,600),
            save_path_picker=()->(save_calls[]+=1;save_choice[]),
            mask_path_picker=()->(mask_calls[]+=1;mask_choice[]),open_revision=r->(opened[]=r))
        screen=GLMakie.Screen(fig.scene;visible=false,start_renderloop=false)
        try
            @test mc.bundle[]===nothing
            boxes=filter(b->b isa Textbox,fig.content);toggles=filter(b->b isa Toggle,fig.content)
            @test length(boxes)==2 && length(toggles)==2
            ax=only(filter(b->b isa Axis,fig.content))
            mask_revision_mouse!(fig,mask_revision_button(fig,"load raw reference"),screen)
            @test mc.running[]
            toggles[1].active[]=false;@test toggles[1].active[] && rc.mask_draft[].enabled
            mask_revision_wait(rc,mc)
            @test mc.state[]===:completed && size(mc.bundle[].raw_image)==(48,64)
            @test mc.bundle[].raw_image==Hammerhead._experiment_load_image(record.input_files[1],Float32)
            @test mc.bundle[].roi==record.recipe.roi && mc.bundle[].recipe_id==record.recipe.recipe_id
            @test polygon_mask(mc.bundle[].editor)==record.recipe.mask
            me=mc.bundle[].editor;reference_id=mc.bundle[].recipe_id
            raw_plot=first(filter(p->p isa GLMakie.Makie.Heatmap,ax.scene.plots))
            initial=copy(rc.mask_draft[].raster)
            mask_revision_point!(fig,ax,screen,25.,20.)
            @test length(me.active[])==1 && isapprox(me.active[][1][1],25.;atol=.01)
            @test isapprox(me.active[][1][2],20.;atol=.01)
            mask_revision_mouse!(fig,mask_revision_button(fig,"Apply raster (enable mask)"),screen)
            @test rc.state[]===:failed && rc.mask_draft[].raster==initial && !isempty(me.active[])
            mask_revision_mouse!(fig,mask_revision_button(fig,"grow 1 px"),screen)
            @test !isempty(me.active[]) && me.raster[]==initial
            mask_revision_mouse!(fig,mask_revision_button(fig,"undo vertex"),screen)
            @test isempty(me.active[])
            mask_revision_point!(fig,ax,screen,25.,20.)
            for point in ((38.,20.),(38.,30.),(25.,30.));mask_revision_point!(fig,ax,screen,point...);end
            mask_revision_point!(fig,ax,screen,25.,30.;button=Mouse.right)
            @test isempty(me.active[]) && length(me.polygons[])==1
            @test rc.mask_draft[].raster==initial && all(polygon_mask(me)[10:12,15:18])
            @test first(filter(p->p isa GLMakie.Makie.Heatmap,ax.scene.plots))===raw_plot
            mask_revision_mouse!(fig,mask_revision_button(fig,"Apply raster (enable mask)"),screen)
            mask_revision_wait(rc,mc)
            @test rc.state[]===:completed && rc.mask_draft[].enabled && all(rc.mask_draft[].raster[22:28,27:36])
            @test mc.bundle[].recipe_id==reference_id && mc.bundle[].roi==record.recipe.roi
            @test mask_revision_other_fields(rc.candidate[])==mask_revision_other_fields(record.recipe)
            # A hole can intentionally unmask the imported raster as well.
            mask_revision_mouse!(fig,mask_revision_button(fig,"draw hole"),screen)
            for point in ((14.,9.),(19.,9.),(19.,13.),(14.,13.));mask_revision_point!(fig,ax,screen,point...);end
            mask_revision_mouse!(fig,mask_revision_button(fig,"close polygon"),screen)
            @test !any(polygon_mask(me)[10:12,15:18])
            mask_revision_mouse!(fig,mask_revision_button(fig,"Apply raster (enable mask)"),screen)
            mask_revision_wait(rc,mc);@test !any(rc.mask_draft[].raster[10:12,15:18])
            applied=copy(rc.mask_draft[].raster)
            mask_revision_point!(fig,ax,screen,30.,25.)
            @test me.selected[]===1
            mask_revision_mouse!(fig,mask_revision_button(fig,"delete selected"),screen)
            @test length(me.polygons[])==1 && me.holes[]==[true] && rc.mask_draft[].raster==applied
            mask_revision_mouse!(fig,toggles[1],screen)
            @test !rc.mask_draft[].enabled && rc.mask_draft[].raster==applied
            mask_revision_mouse!(fig,mask_revision_button(fig,"Apply raster (enable mask)"),screen)
            @test rc.state[]===:failed && occursin("reload",sprint(showerror,rc.error[])) && !rc.mask_draft[].enabled
            # Restoring identical content is allowed; the guard is content-based.
            mask_revision_mouse!(fig,toggles[1],screen)
            mask_revision_mouse!(fig,mask_revision_button(fig,"clear all editor pixels"),screen)
            @test !any(polygon_mask(me)) && rc.mask_draft[].raster==applied
            mask_revision_mouse!(fig,mask_revision_button(fig,"Apply raster (enable mask)"),screen)
            mask_revision_wait(rc,mc);@test rc.mask_draft[].enabled && !any(rc.mask_draft[].raster)
            mask_revision_mouse!(fig,mask_revision_button(fig,"reset to imported mask"),screen)
            @test rc.mask_draft[].raster==initial && !any(polygon_mask(me))
            mask_revision_mouse!(fig,mask_revision_button(fig,"Apply raster (enable mask)"),screen)
            @test rc.state[]===:failed && rc.mask_draft[].raster==initial

            # Invalid visible import text never reuses a previous threshold.
            boxes[2].displayed_string[]="not a threshold"
            mask_revision_mouse!(fig,mask_revision_button(fig,"import mask image..."),screen)
            @test mask_calls[]==0 && rc.mask_draft[].raster==initial
            boxes[2].displayed_string[]="0.5"
            mask_revision_mouse!(fig,mask_revision_button(fig,"import mask image..."),screen)
            mask_revision_wait(rc,mc);@test rc.state[]===:cancelled && mask_calls[]==1
            mask_path=joinpath(directory,"import mask # λ.png")
            imported=falses(48,64);imported[25:32,34:44].=true
            Hammerhead.FileIO.save(mask_path,Hammerhead.Gray.(imported));mask_choice[]=mask_path
            mask_revision_mouse!(fig,toggles[2],screen)
            mask_revision_mouse!(fig,mask_revision_button(fig,"import mask image..."),screen)
            mask_revision_wait(rc,mc)
            @test rc.mask_draft[].raster==.!imported && rc.mask_draft[].enabled
            @test abspath(mask_path) in rc.protected_paths[]
            toggles[2].active[]=false
            mask_revision_mouse!(fig,mask_revision_button(fig,"reset to imported mask"),screen)
            mask_revision_mouse!(fig,mask_revision_button(fig,"load raw reference"),screen)
            mask_revision_wait(rc,mc)
            @test mc.bundle[].editor!==me && polygon_mask(mc.bundle[].editor)==initial
            me=nothing
            mask_revision_mouse!(fig,mask_revision_button(fig,"grow 1 px"),screen)
            @test polygon_mask(mc.bundle[].editor)==Hammerhead.grow_mask(initial,1)
            mask_revision_mouse!(fig,mask_revision_button(fig,"shrink 1 px"),screen)
            @test polygon_mask(mc.bundle[].editor)==Hammerhead.shrink_mask(Hammerhead.grow_mask(initial,1),1)
            mask_revision_mouse!(fig,mask_revision_button(fig,"Apply raster (enable mask)"),screen)
            mask_revision_wait(rc,mc)
            mask_revision_mouse!(fig,mask_revision_button(fig,"save distinct revision..."),screen)
            @test rc.running[];mask_revision_wait(rc,mc)
            saved=rc.saved_record[];bytes=read(rc.saved_path[])
            @test rc.state[]===:completed && save_calls[]==1 && saved.recipe.mask==rc.mask_draft[].raster
            @test saved.input_id==record.input_id && isempty(saved.runs)
            @test mask_revision_other_fields(saved.recipe)==mask_revision_other_fields(record.recipe)
            @test read(source)==source_bytes && read.(files)==input_bytes
            save_choice[]=""
            mask_revision_mouse!(fig,mask_revision_button(fig,"save distinct revision..."),screen)
            mask_revision_wait(rc,mc);@test rc.state[]===:cancelled && rc.saved_record[]===saved && read(rc.saved_path[])==bytes
            mask_bytes=read(mask_path);save_choice[]=mask_path
            mask_revision_mouse!(fig,mask_revision_button(fig,"save distinct revision..."),screen)
            mask_revision_wait(rc,mc);@test rc.state[]===:failed && read(mask_path)==mask_bytes && rc.saved_record[]===saved
            mask_revision_mouse!(fig,mask_revision_button(fig,"open saved revision..."),screen)
            @test timedwait(()->opened[]!==nothing,30.;pollint=.01)==:ok
            @test opened[]!==saved && opened[].recipe.recipe_id==saved.recipe.recipe_id
            @test rc.original.recipe.recipe_id==record.recipe.recipe_id && rc.original.runs==record.runs

            # Clean drawing state for reader-facing captures. Original ROI is
            # drawn at half-pixel edges without cropping or physical conversion.
            mask_revision_mouse!(fig,mask_revision_button(fig,"validate / metadata diff"),screen)
            mask_revision_wait(rc,mc)
            tabs=only(filter(b->b isa Menu && length(b.options[])==5,fig.content))
            tabs.i_selected[]=3;mask_revision_render!(fig,screen)
            details=mask_revision_pages(fig)
            @test occursin(reference_id,details) && occursin(record.input_id,details)
            @test occursin("Raw full-frame pixels; no preprocessing or scripts",details)
            tabs.i_selected[]=4;mask_revision_render!(fig,screen)
            imported_settings=mask_revision_pages(fig)
            @test occursin("invert_image",imported_settings) && occursin("manual callback",imported_settings)
            @test occursin("predictor_smoothing=false",imported_settings) && occursin("mask_threshold=0.4",imported_settings)
            tabs.i_selected[]=1
            # Morphology rasterized the polygons, leaving only the ROI line.
            roi_line=only(filter(p->p isa GLMakie.Makie.Lines,ax.scene.plots))
            @test roi_line[1][]==Point2f[(8.5,4.5),(56.5,4.5),(56.5,44.5),(8.5,44.5),(8.5,4.5)]
            @test ax.yreversed[]
            for (w,h) in ((900,600),(1100,800))
                resize!(fig.scene,w,h);buffer=copy(mask_revision_render!(fig,screen))
                @test size(buffer)==(h,w)
                widgets=filter(b->b isa Union{Button,Toggle,Menu,Textbox,Label} && b.blockscene.visible[],fig.content)
                @test all(b->mask_revision_within(b.layoutobservables.computedbbox[],w,h),widgets)
                @test all(b->mask_revision_within(GLMakie.Makie.boundingbox(b.blockscene),w,h),filter(b->b isa Label,widgets))
                for toggle in toggles
                    point=only(only(filter(p->p isa GLMakie.Makie.Scatter,toggle.blockscene.plots))[1][])
                    rect=toggle.layoutobservables.computedbbox[]
                    @test isapprox(point[1],rect.origin[1]+(toggle.active[] ? rect.widths[1]-toggle.markersize[]/2 : toggle.markersize[]/2);atol=1f-4)
                    @test isapprox(point[2],rect.origin[2]+rect.widths[2]/2;atol=1f-4)
                end
                if haskey(ENV,"HAMMERHEAD_MASK_REVISION_SCREENSHOT")
                    Hammerhead.FileIO.save(replace(ENV["HAMMERHEAD_MASK_REVISION_SCREENSHOT"],".png"=>"-$(w)x$(h).png"),buffer)
                end
            end
            boxes[2].displayed_string[]=repeat("invalid",100)
            @test length(boxes[2].displayed_string[])==700
            resize!(fig.scene,900,600);mask_revision_render!(fig,screen)
            @test all(b->mask_revision_within(b.layoutobservables.computedbbox[],900,600),boxes)
        finally
            GLMakie.destroy!(screen)
        end
    end
end

Base.@noinline function mask_revision_old_references(mc,rc)
    bundle=mc.bundle[]
    refs=WeakRef.((bundle.editor,bundle.raw_image,bundle.editor.image))
    load_recipe_mask_reference!(mc,rc;frame=:b,async=false)
    refs
end
@testset "Absent mask, replaced reference release and protected workflow launch" begin
    mktempdir() do directory
        record,source,files=mask_revision_view_fixture(directory;absent=true)
        rc=RecipeRevisionController(record);mc=RecipeMaskReferenceController()
        fig=recipe_mask_revision(rc;reference_controller=mc,size=(900,600))
        screen=GLMakie.Screen(fig.scene;visible=false,start_renderloop=false)
        try
            @test !rc.mask_draft[].enabled && rc.mask_draft[].raster===nothing
            mask_revision_mouse!(fig,mask_revision_button(fig,"load raw reference"),screen)
            mask_revision_wait(rc,mc)
            @test mc.bundle[].editor.raster[]===nothing && rc.mask_draft[].raster===nothing
            refs=mask_revision_old_references(mc,rc)
            mask_revision_render!(fig,screen);GC.gc(true);GC.gc(true)
            @test all(ref->ref.value===nothing,refs)
            @test mc.bundle[].frame===:b && !rc.mask_draft[].enabled
            mask_revision_mouse!(fig,mask_revision_button(fig,"Apply raster (enable mask)"),screen)
            mask_revision_wait(rc,mc)
            @test rc.mask_draft[].enabled && rc.mask_draft[].raster!==nothing && !any(rc.mask_draft[].raster)
            mask_revision_mouse!(fig,mask_revision_button(fig,"reset to imported mask"),screen)
            @test !rc.mask_draft[].enabled && rc.mask_draft[].raster===nothing
        finally
            GLMakie.destroy!(screen)
        end
        ec=ExperimentController(record);ec.output_path[]="relative-output.jld2";ec.run_record_path[]="relative-history.jld2"
        original=ec.record[];protected=abspath.([ec.output_path[],ec.run_record_path[]]);launched=Ref{Any}(nothing)
        workflow=experiment_workflow(ec;size=(900,600),mask_revision_launcher=r->(launched[]=r))
        screen=GLMakie.Screen(workflow.scene;visible=false,start_renderloop=false)
        try
            mask_revision_mouse!(workflow,mask_revision_button(workflow,"revise mask..."),screen)
            ec.output_path[]=joinpath(directory,"later.jld2")
            @test timedwait(()->launched[]!==nothing,30.;pollint=.01)==:ok
            @test launched[].original!==original && launched[].original.recipe.recipe_id==record.recipe.recipe_id
            @test launched[].protected_paths[]==protected
            @test ec.record[]===original && ec.last_run[]===nothing
            for (w,h) in ((900,600),(1100,800))
                resize!(workflow.scene,w,h);mask_revision_render!(workflow,screen)
                widgets=filter(b->b isa Union{Button,Menu,Textbox,Label} && b.blockscene.visible[],workflow.content)
                @test all(b->mask_revision_within(b.layoutobservables.computedbbox[],w,h),widgets)
            end
        finally
            GLMakie.destroy!(screen)
        end
    end
end
