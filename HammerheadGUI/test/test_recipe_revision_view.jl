using Test,HammerheadGUI,HammerheadGUI.Controllers
using HammerheadGUI.Hammerhead,HammerheadGUI.GLMakie

function revision_view_fixture(directory)
    mkpath(directory)
    image=[mod(37i+13j+7i*j,251)/250 for i in 1:48,j in 1:48]
    files=[joinpath(directory,"frame-$i.png") for i in 1:3]
    for (i,path) in enumerate(files)
        Hammerhead.FileIO.save(path,Hammerhead.Gray.(circshift(image,(i-1,2(i-1)))))
    end
    a=PIVParameters(window_size=32,overlap=16,max_iterations=3,padding=true,
        apodization=:gauss,uncertainty=true,validation=(VelocityMagnitudeValidator(0,20),))
    b=PIVParameters(window_size=16,overlap=8,n_peaks=1,replace_outliers=false,
        validation=(VelocityMagnitudeValidator(1,30),))
    recipe=PIVRecipe([a,b,a,b,a];image_type=Float32,threaded=false,
        roi=ROI(5:44,5:44),mask=falses(48,48),predictor_smoothing=false,mask_threshold=.4,
        preprocessing=[PreprocessStep(:highpass_filter;sigma=2)],
        scale=PhysicalScale(pixel_size=.02,dt=.001,length_unit="mm",time_unit="s"))
    record=ExperimentRecord([(files[1],files[2]),(files[2],files[3])],recipe)
    source=save_experiment(joinpath(directory,"source-experiment.jld2"),record)
    load_experiment(source),source,files
end
revision_view_button(fig,label)=only(filter(b->b isa Button && b.label[]==label,fig.content))
function revision_view_mouse!(fig,block;screen)
    # Real users see a rendered allocation between actions. Also separate
    # clicks beyond Makie's double-click interval: Button handles single clicks.
    sleep(.35)
    colorbuffer(screen)
    rect=block.layoutobservables.computedbbox[]
    events(fig).mouseposition[]=Tuple(Float64.(rect.origin+rect.widths/2))
    events(fig).mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.left,Mouse.press)
    events(fig).mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.left,Mouse.release)
end
function revision_view_wait(rc)
    @test timedwait(()->!rc.running[],60.;pollint=.01)==:ok
    @test rc.task[]===nothing
end
function revision_view_menus(fig)
    menus=filter(b->b isa Menu,fig.content)
    passes=only(filter(b->any(o->first(o)=="pass 1",b.options[]),menus))
    groups=only(filter(b->any(o->first(o)=="Geometry",b.options[]),menus))
    sections=only(filter(b->any(o->last(o)===:original,b.options[]),menus))
    passes,groups,sections
end
function revision_view_pages(fig)
    previous=revision_view_button(fig,"previous text page")
    next=revision_view_button(fig,"next text page")
    pager=only(filter(b->b isa Label && startswith(b.text[],"text page "),fig.content))
    body=only(filter(b->b isa Label && startswith(b.text[],"Status: "),fig.content))
    count=parse(Int,last(split(pager.text[]," / ")))
    for _ in 1:count;previous.clicks[]+=1;end
    result=String[]
    for _ in 1:count;push!(result,body.text[]);next.clicks[]+=1;end
    for _ in 1:count;previous.clicks[]+=1;end
    replace(join(result,"\n"),'\n'=>"")
end
function revision_view_within(rect,width,height)
    all(isfinite,rect.origin) && all(isfinite,rect.widths) &&
        rect.origin[1]>=-1 && rect.origin[2]>=-1 &&
        rect.origin[1]+rect.widths[1]<=width+1 && rect.origin[2]+rect.widths[2]<=height+1
end

@testset "Lossless pass editor actual controls, invalid text and distinct saves" begin
    mktempdir() do directory
        record,source,files=revision_view_fixture(joinpath(directory,"Unicode λ recording"))
        source_bytes=read(source);input_bytes=read.(files)
        destination=Ref{Any}(joinpath(directory,"revision-one.jld2"));picker_calls=Ref(0)
        opened=Ref{Any}(nothing)
        rc=RecipeRevisionController(record)
        picker=()->begin
            picker_calls[]+=1
            destination[]
        end
        fig=recipe_revision(rc;size=(900,600),save_path_picker=picker,open_revision=r->(opened[]=r))
        screen=GLMakie.Screen(fig.scene;visible=false,start_renderloop=false)
        try
            @test size(colorbuffer(screen))==(600,900)
            boxes=filter(b->b isa Textbox,fig.content)
            @test length(boxes)==6
            pass_menu,group_menu,section_menu=revision_view_menus(fig)
            revision_view_mouse!(fig,revision_view_button(fig,"validate / preview changes");screen)
            @test rc.running[]
            pass_menu.i_selected[]=2;group_menu.i_selected[]=2
            @test rc.selected[]==1 && pass_menu.selection[]==1 && group_menu.selection[]==1
            revision_view_wait(rc)
            @test rc.state[]===:completed && !rc.dirty[]
            retained_candidate=rc.candidate[];retained_diff=rc.diff[]

            # Type into an actual focused Makie editor, not its last submitted
            # string. The raw draft becomes dirty before Apply/Save/navigation.
            boxes[1].displayed_string[]=""
            revision_view_mouse!(fig,boxes[1];screen)
            events(fig).unicode_input[]='z'
            @test occursin("z",boxes[1].displayed_string[])
            @test rc.dirty[] && rc.drafts[][1][:window_size]==boxes[1].displayed_string[]
            revision_view_mouse!(fig,revision_view_button(fig,"save distinct revision...");screen)
            revision_view_wait(rc)
            @test rc.state[]===:failed && picker_calls[]==0 && !ispath(destination[])
            @test rc.candidate[]===retained_candidate && rc.diff[]===retained_diff
            pass_menu.i_selected[]=2
            @test rc.selected[]==2 && rc.drafts[][1][:window_size]!="32, 32"
            revision_view_mouse!(fig,revision_view_button(fig,"validate / preview changes");screen)
            revision_view_wait(rc)
            @test rc.state[]===:failed && rc.candidate[]===retained_candidate

            # External controller changes refresh without mixing the old row's
            # visible fields into the newly selected row.
            set_revision_pass!(rc,1,Dict(:window_size=>"32, 32"))
            rc.selected[]=1
            @test boxes[1].displayed_string[]=="32, 32" && pass_menu.selection[]==1
            first_validation=rc.templates[][1].validation
            revision_view_mouse!(fig,revision_view_button(fig,"move down");screen)
            @test rc.selected[]==2 && rc.templates[][2].validation==first_validation
            revision_view_mouse!(fig,revision_view_button(fig,"duplicate pass");screen)
            @test length(rc.drafts[])==6 && rc.selected[]==3 && rc.templates[][3].validation==first_validation
            revision_view_mouse!(fig,revision_view_button(fig,"delete pass");screen)
            @test length(rc.drafts[])==5
            section_menu.i_selected[]=findfirst(o->last(o)===:original,section_menu.options[])
            colorbuffer(screen)
            imported=revision_view_pages(fig)
            @test occursin("VelocityMagnitudeValidator",imported) && occursin("pass 5",imported)
            @test occursin("predictor_smoothing=false",imported) && occursin("precision=Float32",imported)
            @test occursin("highpass_filter",imported) && occursin("mm",imported) && occursin(files[1],imported)

            # The complete current draft is captured before the picker; the
            # busy view restores attempted editor/menu changes in the picker.
            changed=rc.selected[]
            set_revision_pass!(rc,changed,Dict(:max_iterations=>"4"))
            destination[]=joinpath(directory,"revision-one.jld2")
            revision_view_mouse!(fig,revision_view_button(fig,"save distinct revision...");screen)
            revision_view_wait(rc)
            @test rc.state[]===:completed && rc.saved_record[]!==nothing && isfile(destination[])
            saved=rc.saved_record[];saved_path=rc.saved_path[];saved_bytes=read(saved_path)
            @test saved.recipe.passes[changed].max_iterations==4 && saved.input_id==record.input_id && isempty(saved.runs)
            @test [(p.operation,p.options) for p in saved.recipe.preprocessing]==[(p.operation,p.options) for p in record.recipe.preprocessing] && saved.recipe.mask==record.recipe.mask
            @test saved.recipe.roi==record.recipe.roi && saved.recipe.scale==record.recipe.scale
            @test read(source)==source_bytes && read.(files)==input_bytes && rc.original.recipe.recipe_id==record.recipe.recipe_id
            destination[]=""
            revision_view_mouse!(fig,revision_view_button(fig,"save distinct revision...");screen)
            revision_view_wait(rc)
            @test rc.state[]===:cancelled && rc.saved_record[]===saved && rc.saved_path[]==saved_path && read(saved_path)==saved_bytes
            destination[]=source
            previous_picker_calls=picker_calls[]
            save_button=revision_view_button(fig,"save distinct revision...")
            previous_clicks=save_button.clicks[]
            revision_view_mouse!(fig,save_button;screen)
            @test save_button.clicks[]==previous_clicks+1 && rc.running[]
            revision_view_wait(rc)
            @test picker_calls[]==previous_picker_calls+1
            @test rc.state[]===:failed
            @test read(source)==source_bytes
            @test rc.saved_record[]===saved
            revision_view_mouse!(fig,revision_view_button(fig,"open saved revision...");screen)
            @test timedwait(()->opened[]!==nothing,30.;pollint=.01)==:ok
            @test opened[]!==saved && opened[].recipe.recipe_id==saved.recipe.recipe_id
            @test rc.original.recipe.recipe_id==record.recipe.recipe_id && rc.original.runs==record.runs

            # Largest editor group and a long invalid raw value with >=5 passes
            # must not push persistent actions outside either supported size.
            set_revision_pass!(rc,rc.selected[],Dict(:window_size=>repeat("invalid",40)))
            group_menu.i_selected[]=findfirst(o->first(o)=="Correlation",group_menu.options[])
            for (width,height) in ((900,600),(1100,800))
                resize!(fig.scene,width,height)
                buffer=copy(colorbuffer(screen))
                @test size(buffer)==(height,width)
                widgets=filter(b->b isa Union{Button,Menu,Textbox,Label} && b.blockscene.visible[],fig.content)
                @test all(b->revision_view_within(b.layoutobservables.computedbbox[],width,height),widgets)
                @test all(b->revision_view_within(GLMakie.Makie.boundingbox(b.blockscene),width,height),filter(b->b isa Label,widgets))
                @test length(filter(b->b isa Textbox && b.blockscene.visible[],widgets))==6
                if haskey(ENV,"HAMMERHEAD_REVISION_SCREENSHOT")
                    path=replace(ENV["HAMMERHEAD_REVISION_SCREENSHOT"],".png"=>"-$(width)x$(height).png")
                    Hammerhead.FileIO.save(path,buffer)
                end
            end
        finally
            GLMakie.destroy!(screen)
        end
    end
end

@testset "Workflow launches a detached revision without replacing source state" begin
    mktempdir() do directory
        record,source,files=revision_view_fixture(directory)
        ec=ExperimentController(record);ec.output_path[]="relative-native.jld2";ec.run_record_path[]="relative-history.jld2"
        identity=ec.record[].recipe.recipe_id;source_state=(ec.record[],ec.last_run[],ec.output_path[],ec.run_record_path[])
        launched=Ref{Any}(nothing)
        expected=abspath.([ec.output_path[],ec.run_record_path[]])
        fig=experiment_workflow(ec;size=(900,600),revision_launcher=rc->(launched[]=rc))
        screen=GLMakie.Screen(fig.scene;visible=false,start_renderloop=false)
        try
            colorbuffer(screen)
            revision_view_mouse!(fig,revision_view_button(fig,"revise pass schedule...");screen)
            ec.output_path[]=joinpath(directory,"next-output.jld2")
            @test timedwait(()->launched[]!==nothing,30.;pollint=.01)==:ok
            @test launched[].original.recipe.recipe_id==identity && launched[].original!==source_state[1]
            @test launched[].protected_paths[]==expected
            @test ec.record[]===source_state[1] && ec.last_run[]===source_state[2]
        finally
            GLMakie.destroy!(screen)
        end
    end
end
