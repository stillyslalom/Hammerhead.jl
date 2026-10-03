using Test,HammerheadGUI,HammerheadGUI.Controllers
using HammerheadGUI.Hammerhead,HammerheadGUI.GLMakie

function geometry_view_fixture(directory;absent=false)
    mkpath(directory)
    image=Float32[mod(13i+31j+i*j,251)/250 for i in 1:48,j in 1:64]
    files=[joinpath(directory,"frame-$i.png") for i in 1:3]
    for (i,path) in enumerate(files)
        Hammerhead.FileIO.save(path,Hammerhead.Gray.(circshift(image,(i-1,2(i-1)))))
    end
    mask=falses(48,64);mask[10:12,15:18].=true
    recipe=PIVRecipe([PIVParameters(window_size=16,overlap=8,validation=(VelocityMagnitudeValidator(0,20),))];
        preprocessing=[PreprocessStep(:subtract_background;background=fill(.012345678901234,48,64)),
            PreprocessStep(:invert_image),PreprocessStep(:invert_image)],
        roi=absent ? nothing : ROI(5:44,9:56),mask,
        scale=absent ? nothing : PhysicalScale(pixel_size=nextfloat(.02),dt=nextfloat(.001),length_unit="mm λ",time_unit="s"),
        image_type=Float32,threaded=false,predictor_smoothing=false,mask_threshold=.4)
    record=ExperimentRecord([(files[1],files[2]),(files[2],files[3])],recipe)
    source=save_experiment(joinpath(directory,"source.jld2"),record)
    load_experiment(source),source,files
end
geometry_view_button(fig,label)=only(filter(b->b isa Button && b.label[]==label,fig.content))
function geometry_view_render!(fig,screen)
    # Hidden start_renderloop=false screens do not emit elapsed animation ticks.
    # Supply successive frames as the desktop loop would: completing an earlier
    # animation removes its listener, so another frame services remaining knobs.
    for _ in 1:3
        tick=events(fig).tick[]
        events(fig).tick[]=GLMakie.Makie.Tick(GLMakie.Makie.UnknownTickState,tick.count+1,tick.time+1.,1.)
    end
    colorbuffer(screen)
end
function geometry_view_mouse!(fig,block,screen)
    sleep(.3);geometry_view_render!(fig,screen)
    rect=block.layoutobservables.computedbbox[]
    clicks=block isa Button ? block.clicks[] : nothing
    events(fig).mouseposition[]=Tuple(Float64.(rect.origin+rect.widths/2))
    events(fig).mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.left,Mouse.press)
    events(fig).mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.left,Mouse.release)
    block isa Button && @test block.clicks[]==clicks+1
end
function geometry_view_wait(rc)
    @test timedwait(()->!rc.running[],90.;pollint=.01)==:ok
    @test rc.task[]===nothing
end
function geometry_view_within(rect,width,height)
    all(isfinite,rect.origin) && all(isfinite,rect.widths) &&
        rect.origin[1]>=-1 && rect.origin[2]>=-1 &&
        rect.origin[1]+rect.widths[1]<=width+1 && rect.origin[2]+rect.widths[2]<=height+1
end
function geometry_view_pages(fig)
    previous=geometry_view_button(fig,"previous text page");next=geometry_view_button(fig,"next text page")
    pager=only(filter(b->b isa Label && startswith(b.text[],"text page "),fig.content))
    body=only(filter(b->b isa Label && startswith(b.text[],"Status: "),fig.content))
    count=parse(Int,last(split(pager.text[]," / ")))
    for _ in 1:count;previous.clicks[]+=1;end
    pages=String[]
    for _ in 1:count;push!(pages,body.text[]);next.clicks[]+=1;end
    replace(join(pages,"\n"),'\n'=>"")
end
function geometry_view_other_fields(recipe)
    data=Hammerhead._experiment_recipe_data(recipe)
    for key in ("roi","scale");delete!(data,key);end
    data
end

@testset "ROI / scale actual editing, raw invalid drafts and distinct saves" begin
    mktempdir() do directory
        record,source,files=geometry_view_fixture(joinpath(directory,"Unicode λ geometry"))
        source_bytes=read(source);input_bytes=read.(files)
        destination=Ref{Any}(joinpath(directory,"revision.jld2"));calls=Ref(0);opened=Ref{Any}(nothing)
        picker=()->(calls[]+=1;destination[])
        rc=RecipeRevisionController(record)
        fig=recipe_geometry_revision(rc;size=(900,600),save_path_picker=picker,open_revision=r->(opened[]=r))
        screen=GLMakie.Screen(fig.scene;visible=false,start_renderloop=false)
        try
            @test size(colorbuffer(screen))==(600,900)
            boxes=filter(b->b isa Textbox,fig.content);toggles=filter(b->b isa Toggle,fig.content)
            @test length(boxes)==8 && length(toggles)==2
            @test parse(Float64,boxes[5].displayed_string[])===record.recipe.scale.pixel_size
            @test parse(Float64,boxes[6].displayed_string[])===record.recipe.scale.dt
            @test boxes[7].displayed_string[]=="mm λ" && boxes[8].displayed_string[]=="s"
            @test [b.displayed_string[] for b in boxes[1:4]]==["5","44","9","56"]
            geometry_view_mouse!(fig,geometry_view_button(fig,"validate / metadata diff"),screen)
            @test rc.running[]
            original_row=boxes[1].displayed_string[]
            boxes[1].displayed_string[]="busy edit";toggles[1].active[]=false
            @test boxes[1].displayed_string[]==original_row && toggles[1].active[]
            geometry_view_wait(rc)
            @test rc.state[]===:completed && rc.candidate[].recipe_id==record.recipe.recipe_id
            candidate=rc.candidate[];diff=rc.diff[]

            # Type in a real focused control; current raw text, not stored_string,
            # is used on Save and cannot fall back to a previously valid ROI.
            boxes[1].displayed_string[]=""
            geometry_view_mouse!(fig,boxes[1],screen);events(fig).unicode_input[]='z'
            @test occursin("z",boxes[1].displayed_string[]) && rc.dirty[]
            @test rc.roi_draft[].values[:row_first]==boxes[1].displayed_string[]
            geometry_view_mouse!(fig,geometry_view_button(fig,"save distinct revision..."),screen)
            geometry_view_wait(rc)
            @test rc.state[]===:failed && calls[]==0 && !ispath(destination[])
            @test rc.candidate[]===candidate && rc.diff[]===diff

            # Reset/disable means nothing, retaining the invalid text for a later
            # re-enable. It never fabricates a full-frame crop or scale factors.
            geometry_view_mouse!(fig,geometry_view_button(fig,"reset ROI to nothing"),screen)
            @test !rc.roi_draft[].enabled && occursin("z",boxes[1].displayed_string[])
            geometry_view_mouse!(fig,geometry_view_button(fig,"validate / metadata diff"),screen)
            geometry_view_wait(rc);@test rc.candidate[].roi===nothing
            geometry_view_mouse!(fig,toggles[1],screen)
            @test rc.roi_draft[].enabled
            geometry_view_mouse!(fig,geometry_view_button(fig,"validate / metadata diff"),screen)
            geometry_view_wait(rc);@test rc.state[]===:failed && rc.candidate[].roi===nothing
            set_revision_roi!(rc,Dict(:row_first=>"6",:row_last=>"43",:col_first=>"10",:col_last=>"55"))
            @test [b.displayed_string[] for b in boxes[1:4]]==["6","43","10","55"]
            boxes[5].displayed_string[]="0";boxes[6].displayed_string[]="Inf"
            geometry_view_mouse!(fig,geometry_view_button(fig,"validate / metadata diff"),screen)
            geometry_view_wait(rc);@test rc.state[]===:failed
            geometry_view_mouse!(fig,geometry_view_button(fig,"reset scale to nothing"),screen)
            @test !toggles[2].active[] && boxes[5].displayed_string[]=="0" && boxes[6].displayed_string[]=="Inf"
            geometry_view_mouse!(fig,geometry_view_button(fig,"validate / metadata diff"),screen)
            geometry_view_wait(rc);@test rc.candidate[].scale===nothing
            geometry_view_mouse!(fig,toggles[2],screen)
            set_revision_scale!(rc,Dict(:pixel_size=>"0.03125000000000001",:dt=>"0.002",:length_unit=>"µm / chosen",:time_unit=>"ms"))
            @test boxes[7].displayed_string[]=="µm / chosen" && rc.scale_draft[].enabled
            geometry_view_mouse!(fig,geometry_view_button(fig,"save distinct revision..."),screen)
            geometry_view_wait(rc)
            saved=rc.saved_record[];path=rc.saved_path[];bytes=read(path)
            @test rc.state[]===:completed && calls[]==1 && saved.recipe.roi==ROI(6:43,10:55)
            @test saved.recipe.scale.pixel_size==parse(Float64,"0.03125000000000001") && saved.recipe.scale.dt==.002
            @test saved.recipe.scale.length_unit=="µm / chosen" && saved.recipe.scale.time_unit=="ms"
            @test saved.input_id==record.input_id && isempty(saved.runs)
            @test geometry_view_other_fields(saved.recipe)==geometry_view_other_fields(record.recipe)
            @test read(source)==source_bytes && read.(files)==input_bytes && rc.original.recipe.recipe_id==record.recipe.recipe_id
            destination[]=""
            geometry_view_mouse!(fig,geometry_view_button(fig,"save distinct revision..."),screen)
            geometry_view_wait(rc)
            @test rc.state[]===:cancelled && rc.saved_record[]===saved && read(path)==bytes
            destination[]=source
            geometry_view_mouse!(fig,geometry_view_button(fig,"save distinct revision..."),screen)
            geometry_view_wait(rc)
            @test rc.state[]===:failed && rc.saved_record[]===saved && read(source)==source_bytes
            geometry_view_mouse!(fig,geometry_view_button(fig,"open saved revision..."),screen)
            @test timedwait(()->opened[]!==nothing,30.;pollint=.01)==:ok
            @test opened[]!==saved && opened[].recipe.recipe_id==saved.recipe.recipe_id
            @test rc.original.runs==record.runs && rc.original.recipe.recipe_id==record.recipe.recipe_id

            section=only(filter(b->b isa Menu,fig.content))
            section.i_selected[]=2;colorbuffer(screen)
            imported=geometry_view_pages(fig)
            @test occursin("subtract_background",imported) && occursin("VelocityMagnitudeValidator",imported)
            @test occursin("predictor_smoothing=false",imported) && occursin("precision=Float32",imported)
            @test occursin("mm λ",imported) && occursin("ROI",imported)
            # Clean current draft/capture, then long invalid raw text layout;
            # clipped textbox content remains editable without resizing the form.
            geometry_view_mouse!(fig,geometry_view_button(fig,"validate / metadata diff"),screen)
            geometry_view_wait(rc);section.i_selected[]=1
            for (width,height) in ((900,600),(1100,800))
                resize!(fig.scene,width,height);buffer=copy(geometry_view_render!(fig,screen))
                @test size(buffer)==(height,width)
                widgets=filter(b->b isa Union{Button,Toggle,Menu,Textbox,Label} && b.blockscene.visible[],fig.content)
                @test all(b->geometry_view_within(b.layoutobservables.computedbbox[],width,height),widgets)
                @test all(b->geometry_view_within(GLMakie.Makie.boundingbox(b.blockscene),width,height),filter(b->b isa Label,widgets))
                @test all(b->b.width[]>150,boxes)
                for toggle in toggles
                    knob=only(filter(p->p isa GLMakie.Makie.Scatter,toggle.blockscene.plots))
                    point=only(knob[1][]);rect=toggle.layoutobservables.computedbbox[]
                    expected_x=rect.origin[1]+(toggle.active[] ? rect.widths[1]-toggle.markersize[]/2 : toggle.markersize[]/2)
                    @test isapprox(point[1],expected_x;atol=1f-4,rtol=0)
                    @test isapprox(point[2],rect.origin[2]+rect.widths[2]/2;atol=1f-4,rtol=0)
                end
                if haskey(ENV,"HAMMERHEAD_GEOMETRY_SCREENSHOT")
                    Hammerhead.FileIO.save(replace(ENV["HAMMERHEAD_GEOMETRY_SCREENSHOT"],".png"=>"-$(width)x$(height).png"),buffer)
                end
            end
            boxes[1].displayed_string[]=repeat("invalid",100)
            boxes[7].displayed_string[]=repeat("long unit ",100)
            resize!(fig.scene,900,600);colorbuffer(screen)
            @test rc.dirty[] && length(rc.roi_draft[].values[:row_first])==700
            @test all(b->geometry_view_within(b.layoutobservables.computedbbox[],900,600),boxes)
            @test rc.saved_record[]===saved
        finally
            GLMakie.destroy!(screen)
        end
    end
end

@testset "Absent metadata and separate protected workflow launch" begin
    mktempdir() do directory
        record,source,files=geometry_view_fixture(directory;absent=true)
        rc=RecipeRevisionController(record);fig=recipe_geometry_revision(rc;size=(900,600))
        screen=GLMakie.Screen(fig.scene;visible=false,start_renderloop=false)
        try
            boxes=filter(b->b isa Textbox,fig.content)
            @test !rc.roi_draft[].enabled && !rc.scale_draft[].enabled
            @test all(b->isempty(b.displayed_string[]),boxes[5:8])
            geometry_view_mouse!(fig,geometry_view_button(fig,"validate / metadata diff"),screen)
            geometry_view_wait(rc)
            @test rc.candidate[].roi===nothing && rc.candidate[].scale===nothing && rc.candidate[].recipe_id==record.recipe.recipe_id
            # A shared controller may already contain changes from the other
            # editors. This form preserves those current drafts, not just the
            # immutable imported settings shown in its read-only tab.
            set_revision_pass!(rc,1,Dict(:max_iterations=>"2"))
            insert_revision_preprocess!(rc,4,:intensity_cap)
            set_revision_preprocess!(rc,4,Dict(:n_sigma=>"2.75"))
            expected_other=geometry_view_other_fields(revision_recipe(rc))
            geometry_view_mouse!(fig,geometry_view_button(fig,"validate / metadata diff"),screen)
            geometry_view_wait(rc)
            @test geometry_view_other_fields(rc.candidate[])==expected_other
            @test rc.original.recipe.recipe_id==record.recipe.recipe_id
        finally
            GLMakie.destroy!(screen)
        end
        ec=ExperimentController(record);ec.output_path[]="relative-output.jld2";ec.run_record_path[]="relative-history.jld2"
        source_record=ec.record[];launched=Ref{Any}(nothing)
        protected=abspath.([ec.output_path[],ec.run_record_path[]])
        workflow=experiment_workflow(ec;size=(900,600),geometry_revision_launcher=r->(launched[]=r))
        screen=GLMakie.Screen(workflow.scene;visible=false,start_renderloop=false)
        try
            geometry_view_mouse!(workflow,geometry_view_button(workflow,"revise ROI / scale..."),screen)
            ec.output_path[]=joinpath(directory,"later-output.jld2")
            @test timedwait(()->launched[]!==nothing,30.;pollint=.01)==:ok
            @test launched[].original!==source_record && launched[].original.recipe.recipe_id==record.recipe.recipe_id
            @test launched[].protected_paths[]==protected
            @test ec.record[]===source_record && ec.last_run[]===nothing
            for (width,height) in ((900,600),(1100,800))
                resize!(workflow.scene,width,height);colorbuffer(screen)
                widgets=filter(b->b isa Union{Button,Menu,Textbox,Label} && b.blockscene.visible[],workflow.content)
                @test all(b->geometry_view_within(b.layoutobservables.computedbbox[],width,height),widgets)
            end
        finally
            GLMakie.destroy!(screen)
        end
    end
end
