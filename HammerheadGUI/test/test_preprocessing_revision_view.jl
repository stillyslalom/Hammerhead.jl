using Test,HammerheadGUI,HammerheadGUI.Controllers
using HammerheadGUI.Hammerhead,HammerheadGUI.GLMakie

function preprocessing_view_fixture(directory;external=false)
    mkpath(directory)
    image=Float32[.3+.1*sin(i/3)+.15*cos(j/5)+mod(i*j,17)/50 for i in 1:48,j in 1:64]
    files=[joinpath(directory,"frame-$i.png") for i in 1:3]
    for (i,path) in enumerate(files)
        Hammerhead.FileIO.save(path,Hammerhead.Gray.(circshift(image,(i-1,2(i-1)))))
    end
    steps=[PreprocessStep(:subtract_background;background=fill(.01,48,64)),
        PreprocessStep(:intensity_cap;n_sigma=2.5),PreprocessStep(:highpass_filter;sigma=1.5),
        PreprocessStep(:clahe;tiles=(2,3),nbins=32,clip_limit=2.5),
        PreprocessStep(:percentile_stretch;low=5,high=95),
        PreprocessStep(:local_variance_normalize;sigma=1.2,epsilon=.02),
        PreprocessStep(:invert_image),PreprocessStep(:invert_image),
        PreprocessStep(:highpass_filter;sigma=.8),PreprocessStep(:intensity_cap;n_sigma=3)]
    script=nothing
    if external
        path=joinpath(directory,"never-execute.jl")
        write(path,"error(\"This referenced script must never execute\")")
        script=ScriptReference(path;entrypoint="manual callback")
    end
    mask=falses(48,64);mask[10:12,14:17].=true
    pass=PIVParameters(window_size=16,overlap=8,validation=(VelocityMagnitudeValidator(0,20),))
    recipe=PIVRecipe([pass];preprocessing=steps,external_preprocess=script,mask,roi=ROI(5:44,9:56),
        scale=PhysicalScale(pixel_size=.02,dt=.001,length_unit="mm",time_unit="s"),
        image_type=Float32,threaded=false,predictor_smoothing=false,mask_threshold=.4)
    record=ExperimentRecord([(files[1],files[2]),(files[2],files[3])],recipe)
    source=save_experiment(joinpath(directory,"source.jld2"),record)
    load_experiment(source),source,files
end
preprocessing_view_button(fig,label)=only(filter(b->b isa Button && b.label[]==label,fig.content))
function preprocessing_view_mouse!(fig,block,screen)
    sleep(.3);colorbuffer(screen)
    rect=block.layoutobservables.computedbbox[]
    clicks=block isa Button ? block.clicks[] : nothing
    events(fig).mouseposition[]=Tuple(Float64.(rect.origin+rect.widths/2))
    events(fig).mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.left,Mouse.press)
    events(fig).mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.left,Mouse.release)
    block isa Button && @test block.clicks[]==clicks+1
end
function preprocessing_view_wait(controller)
    @test timedwait(()->!controller.running[],90.;pollint=.01)==:ok
    @test controller.task[]===nothing
end
function preprocessing_view_menus(fig)
    menus=filter(b->b isa Menu,fig.content)
    step=only(filter(b->any(o->last(o) isa Int,b.options[]),menus))
    operation=only(filter(b->any(o->last(o)===:clahe,b.options[]),menus))
    pane=only(filter(b->any(o->last(o)===:pixel_details,b.options[]),menus))
    step,operation,pane
end
function preprocessing_view_within(rect,width,height)
    all(isfinite,rect.origin) && all(isfinite,rect.widths) &&
        rect.origin[1]>=-1 && rect.origin[2]>=-1 &&
        rect.origin[1]+rect.widths[1]<=width+1 && rect.origin[2]+rect.widths[2]<=height+1
end
function preprocessing_view_pages(fig)
    previous=preprocessing_view_button(fig,"previous text page")
    next=preprocessing_view_button(fig,"next text page")
    pager=only(filter(b->b isa Label && startswith(b.text[],"text page "),fig.content))
    body=only(filter(b->b isa Label && (startswith(b.text[],"Status: ") || startswith(b.text[],"Pixel status: ")),fig.content))
    count=parse(Int,last(split(pager.text[]," / ")))
    for _ in 1:count;previous.clicks[]+=1;end
    pages=String[]
    for _ in 1:count;push!(pages,body.text[]);next.clicks[]+=1;end
    replace(join(pages,"\n"),'\n'=>"")
end

@testset "Ordered preprocessing view actual edits and captured pixels" begin
    mktempdir() do directory
        record,source,files=preprocessing_view_fixture(joinpath(directory,"Unicode λ images"))
        source_bytes=read(source);input_bytes=read.(files)
        rc=RecipeRevisionController(record);pc=RecipeImagePreviewController()
        destination=Ref{Any}(joinpath(directory,"revision.jld2"));picker_calls=Ref(0)
        background_choice=Ref{Any}(nothing);background_calls=Ref(0);opened=Ref{Any}(nothing)
        picker=()->begin;picker_calls[]+=1;destination[];end
        background_picker=()->begin;background_calls[]+=1;background_choice[];end
        fig=preprocessing_revision(rc;preview_controller=pc,size=(900,600),save_path_picker=picker,
            background_path_picker=background_picker,open_revision=r->(opened[]=r))
        screen=GLMakie.Screen(fig.scene;visible=false,start_renderloop=false)
        try
            @test size(colorbuffer(screen))==(600,900)
            step,operation,pane=preprocessing_view_menus(fig)
            boxes=filter(b->b isa Textbox,fig.content);options=boxes[1:3];pair=boxes[4]
            @test length(rc.preprocessing_drafts[])==10
            @test eltype(rc.preprocessing_drafts[][1].background)===Float64
            preprocessing_view_mouse!(fig,preprocessing_view_button(fig,"validate / metadata diff"),screen)
            @test rc.running[]
            step.i_selected[]=4
            @test rc.preprocessing_selected[]==1 && step.selection[]==1
            preprocessing_view_wait(rc)
            @test rc.state[]===:completed && !rc.dirty[]
            @test rc.candidate[].recipe_id==record.recipe.recipe_id
            candidate=rc.candidate[]
            preprocessing_view_mouse!(fig,preprocessing_view_button(fig,"preview verified pair"),screen)
            @test pc.running[]
            pair.displayed_string[]="2"
            preprocessing_view_wait(pc)
            bundle=pc.bundle[]
            @test bundle!==nothing && bundle.pair_index==1 && bundle.recipe_id==record.recipe.recipe_id
            @test bundle.image_type===Float32 && eltype(bundle.processed_a)===Float32
            @test size(bundle.raw_a)==(48,64) && size(bundle.processed_a)==(48,64)
            @test bundle.roi==record.recipe.roi && bundle.mask==record.recipe.mask
            @test read(source)==source_bytes && read.(files)==input_bytes
            if haskey(ENV,"HAMMERHEAD_PREPROCESSING_SCREENSHOT")
                rc.preprocessing_selected[]=4;pair.displayed_string[]="1"
                pane.i_selected[]=findfirst(o->last(o)===:pixels,pane.options[])
                for (width,height) in ((900,600),(1100,800))
                    resize!(fig.scene,width,height)
                    path=replace(ENV["HAMMERHEAD_PREPROCESSING_SCREENSHOT"],".png"=>"-clean-pixels-$(width)x$(height).png")
                    Hammerhead.FileIO.save(path,copy(colorbuffer(screen)))
                end
                resize!(fig.scene,900,600);pair.displayed_string[]="2";rc.preprocessing_selected[]=1
                pane.i_selected[]=findfirst(o->last(o)===:changes,pane.options[])
            end
            image_axes=filter(b->b isa Axis,fig.content)
            @test length(image_axes)==2
            shared_range=(min(minimum(bundle.raw_a),minimum(bundle.processed_a)),
                max(maximum(bundle.raw_a),maximum(bundle.processed_a)))
            for axis in image_axes
                heatmaps=filter(plot->plot isa Heatmap,axis.scene.plots)
                @test Tuple(first(heatmaps).colorrange[])==shared_range
                @test axis.yreversed[]
                rectangles=filter(plot->plot isa Lines,axis.scene.plots)
                @test only(rectangles)[1][]==Point2f[(8.5,4.5),(56.5,4.5),(56.5,44.5),(8.5,44.5),(8.5,4.5)]
            end
            pair.displayed_string[]="invalid pair"
            preprocessing_view_mouse!(fig,preprocessing_view_button(fig,"preview verified pair"),screen)
            @test pc.state[]===:failed && pc.bundle[]===bundle && !pc.running[]
            @test occursin("positive integer",pc.status[])
            pair.displayed_string[]="2"

            step.i_selected[]=4
            @test rc.preprocessing_selected[]==4 && options[1].displayed_string[]=="2, 3"
            options[1].displayed_string[]=""
            preprocessing_view_mouse!(fig,options[1],screen);events(fig).unicode_input[]='x'
            @test occursin("x",rc.preprocessing_drafts[][4].options[:tiles]) && rc.dirty[]
            preprocessing_view_mouse!(fig,preprocessing_view_button(fig,"save distinct revision..."),screen)
            preprocessing_view_wait(rc)
            @test rc.state[]===:failed && picker_calls[]==0 && !ispath(destination[])
            @test rc.candidate[]===candidate && pc.bundle[]===bundle
            step.i_selected[]=5
            @test occursin("x",rc.preprocessing_drafts[][4].options[:tiles])
            set_revision_preprocess!(rc,4,Dict(:tiles=>"2, 3"));rc.preprocessing_selected[]=4
            @test options[1].displayed_string[]=="2, 3"
            original_operations=[row.operation for row in rc.preprocessing_drafts[]]
            preprocessing_view_mouse!(fig,preprocessing_view_button(fig,"duplicate step"),screen)
            @test length(rc.preprocessing_drafts[])==11 && rc.preprocessing_selected[]==5
            @test rc.preprocessing_drafts[][5].options==rc.preprocessing_drafts[][4].options
            preprocessing_view_mouse!(fig,preprocessing_view_button(fig,"move down"),screen)
            @test rc.preprocessing_selected[]==6
            preprocessing_view_mouse!(fig,preprocessing_view_button(fig,"delete step"),screen)
            @test [row.operation for row in rc.preprocessing_drafts[]]==original_operations

            rc.preprocessing_selected[]=1
            preprocessing_view_mouse!(fig,preprocessing_view_button(fig,"load background as Float32..."),screen)
            preprocessing_view_wait(rc)
            @test background_calls[]==1 && eltype(rc.preprocessing_drafts[][1].background)===Float64
            bg_path=joinpath(directory,"replacement background.png")
            Hammerhead.FileIO.save(bg_path,Hammerhead.Gray.(fill(.02,48,64)))
            bg_bytes=read(bg_path);background_choice[]=bg_path
            preprocessing_view_mouse!(fig,preprocessing_view_button(fig,"load background as Float32..."),screen)
            preprocessing_view_wait(rc)
            @test eltype(rc.preprocessing_drafts[][1].background)===Float32
            @test abspath(bg_path) in rc.protected_paths[]
            @test read(bg_path)==bg_bytes && pc.bundle[]===bundle
            set_revision_preprocess!(rc,2,Dict(:n_sigma=>"3.5"))
            preprocessing_view_mouse!(fig,preprocessing_view_button(fig,"save distinct revision..."),screen)
            preprocessing_view_wait(rc)
            @test rc.state[]===:completed && isfile(destination[])
            saved=rc.saved_record[];saved_path=rc.saved_path[];saved_bytes=read(saved_path)
            @test saved.input_id==record.input_id && isempty(saved.runs)
            before=Hammerhead._experiment_recipe_data(record.recipe);after=Hammerhead._experiment_recipe_data(saved.recipe)
            delete!(before,"preprocessing");delete!(after,"preprocessing")
            @test isequal(before,after)
            @test saved.recipe.preprocessing[2].options["n_sigma"]==3.5
            destination[]=nothing
            preprocessing_view_mouse!(fig,preprocessing_view_button(fig,"save distinct revision..."),screen)
            preprocessing_view_wait(rc)
            @test rc.state[]===:cancelled && rc.saved_record[]===saved && read(saved_path)==saved_bytes
            destination[]=source
            preprocessing_view_mouse!(fig,preprocessing_view_button(fig,"save distinct revision..."),screen)
            preprocessing_view_wait(rc)
            @test rc.state[]===:failed && read(source)==source_bytes && rc.saved_record[]===saved
            preprocessing_view_mouse!(fig,preprocessing_view_button(fig,"open saved revision..."),screen)
            @test timedwait(()->opened[]!==nothing,30.;pollint=.01)==:ok
            @test opened[]!==saved && opened[].recipe.recipe_id==saved.recipe.recipe_id
            @test pc.bundle[]===bundle && rc.original.recipe.recipe_id==record.recipe.recipe_id
            @test read.(files)==input_bytes

            # No automatic pixel work after a long invalid edit. Every option,
            # including nondefault CLAHE settings, remains reachable in small UI.
            rc.preprocessing_selected[]=4
            set_revision_preprocess!(rc,4,Dict(:tiles=>repeat("invalid",40)))
            @test pc.bundle[]===bundle && !pc.running[]
            pane.i_selected[]=findfirst(o->last(o)===:original,pane.options[])
            colorbuffer(screen);text=preprocessing_view_pages(fig)
            @test occursin("predictor_smoothing=false",text) && occursin("precision=Float32",text)
            @test occursin("local_variance_normalize",text) && occursin("mm",text)
            for section in (:changes,:pixels), (width,height) in ((900,600),(1100,800))
                pane.i_selected[]=findfirst(o->last(o)===section,pane.options[])
                resize!(fig.scene,width,height);buffer=copy(colorbuffer(screen))
                @test size(buffer)==(height,width)
                widgets=filter(b->b isa Union{Button,Menu,Textbox,Label,Axis} && b.blockscene.visible[],fig.content)
                for b in widgets
                    rect=b.layoutobservables.computedbbox[]
                    if !preprocessing_view_within(rect,width,height)
                        println("OUTSIDE_WIDGET ",section," ",width,"x",height," ",typeof(b)," ",rect,
                            b isa Label ? " TEXT="*repr(b.text[]) : "")
                    end
                    if b isa Label && !preprocessing_view_within(GLMakie.Makie.boundingbox(b.blockscene),width,height)
                        println("OUTSIDE_GLYPHS ",section," ",width,"x",height," ",GLMakie.Makie.boundingbox(b.blockscene)," TEXT=",repr(b.text[]))
                    end
                end
                @test all(b->preprocessing_view_within(b.layoutobservables.computedbbox[],width,height),widgets)
                @test all(b->preprocessing_view_within(GLMakie.Makie.boundingbox(b.blockscene),width,height),filter(b->b isa Label,widgets))
                @test length(filter(b->b isa Textbox && b!==pair,widgets))==3
                if haskey(ENV,"HAMMERHEAD_PREPROCESSING_SCREENSHOT")
                    path=replace(ENV["HAMMERHEAD_PREPROCESSING_SCREENSHOT"],".png"=>"-$section-$(width)x$(height).png")
                    Hammerhead.FileIO.save(path,buffer)
                end
            end
        finally
            GLMakie.destroy!(screen)
        end
    end
end

@testset "Shared display range stays finite at constant intensity extremes" begin
    for value in (0.,1.,-1.,floatmax(Float64),-floatmax(Float64),floatmax(Float32),-floatmax(Float32))
        range=HammerheadGUI._preprocessing_image_range((fill(value,2,2),fill(value,2,2)))
        @test all(isfinite,range) && range[1]<range[2]
        @test range[1]<=value<=range[2]
    end
end

@testset "Empty chain, external script refusal and protected launch" begin
    mktempdir() do directory
        record,source,files=preprocessing_view_fixture(directory;external=true)
        rc=RecipeRevisionController(record);pc=RecipeImagePreviewController()
        fig=preprocessing_revision(rc;preview_controller=pc,size=(900,600))
        screen=GLMakie.Screen(fig.scene;visible=false,start_renderloop=false)
        try
            preprocessing_view_mouse!(fig,preprocessing_view_button(fig,"preview verified pair"),screen)
            preprocessing_view_wait(pc)
            @test pc.state[]===:failed && pc.bundle[]===nothing
            @test rc.original.recipe.external_preprocess.sha256==record.recipe.external_preprocess.sha256
            for _ in 1:10;delete_revision_preprocess!(rc,1);end
            @test isempty(rc.preprocessing_drafts[]) && rc.preprocessing_selected[]==0
            step,operation,pane=preprocessing_view_menus(fig)
            @test step.selection[]==0
            for op in (:intensity_cap,:highpass_filter,:clahe,:percentile_stretch,:invert_image,:local_variance_normalize,:subtract_background)
                operation.i_selected[]=findfirst(o->last(o)===op,operation.options[])
                preprocessing_view_mouse!(fig,preprocessing_view_button(fig,"add"),screen)
                @test rc.preprocessing_drafts[][rc.preprocessing_selected[]].operation===op
            end
            @test length(rc.preprocessing_drafts[])==7
            @test revision_recipe(RecipeRevisionController(record)).external_preprocess.sha256==record.recipe.external_preprocess.sha256
        finally
            GLMakie.destroy!(screen)
        end
        ec=ExperimentController(record);ec.output_path[]="relative-result.jld2";ec.run_record_path[]="relative-history.jld2"
        source_record=ec.record[];launched=Ref{Any}(nothing)
        protected=abspath.([ec.output_path[],ec.run_record_path[]])
        workflow=experiment_workflow(ec;size=(900,600),preprocessing_revision_launcher=r->(launched[]=r))
        screen=GLMakie.Screen(workflow.scene;visible=false,start_renderloop=false)
        try
            preprocessing_view_mouse!(workflow,preprocessing_view_button(workflow,"revise preprocessing..."),screen)
            ec.output_path[]=joinpath(directory,"later-output.jld2")
            @test timedwait(()->launched[]!==nothing,30.;pollint=.01)==:ok
            @test launched[].original!==source_record && launched[].original.recipe.recipe_id==record.recipe.recipe_id
            @test launched[].protected_paths[]==protected
            @test ec.record[]===source_record && ec.last_run[]===nothing
            for (width,height) in ((900,600),(1100,800))
                resize!(workflow.scene,width,height);colorbuffer(screen)
                widgets=filter(b->b isa Union{Button,Menu,Textbox,Label} && b.blockscene.visible[],workflow.content)
                @test all(b->preprocessing_view_within(b.layoutobservables.computedbbox[],width,height),widgets)
            end
        finally
            GLMakie.destroy!(screen)
        end
    end
end
