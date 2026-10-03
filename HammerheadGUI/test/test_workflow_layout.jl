using Test,HammerheadGUI,HammerheadGUI.Controllers
using HammerheadGUI.Hammerhead,HammerheadGUI.GLMakie

function workflow_layout_fixture(directory)
    mkpath(directory)
    a=[mod(37i+13j+7i*j,251)/250 for i in 1:48,j in 1:48]
    files=[joinpath(directory,"frame-$i.png") for i in 1:3]
    for (i,path) in enumerate(files)
        Hammerhead.FileIO.save(path,Hammerhead.Gray.(circshift(a,(i-1,2(i-1)))))
    end
    recipe=PIVRecipe(fill(PIVParameters(window_size=16,overlap=8,max_iterations=1),5);
        threaded=false,roi=ROI(5:44,5:44),mask=falses(48,48),
        preprocessing=[PreprocessStep(:highpass_filter;sigma=2)],
        scale=PhysicalScale(pixel_size=.02,dt=.001,length_unit="mm",time_unit="s"))
    record=ExperimentRecord([(files[1],files[2]),(files[2],files[3])],recipe)
    record,files
end
workflow_controls(fig)=only(filter(b->b isa Menu && ("Files",:files) in b.options[],fig.content))
workflow_body(fig)=only(filter(b->b isa Label && startswith(b.text[],"selected output:"),fig.content))
workflow_button(fig,label)=only(filter(b->b isa Button && startswith(b.label[],label),fig.content))
function workflow_choose!(menu,kind)
    menu.i_selected[]=findfirst(option->last(option)===kind,menu.options[])
end
function workflow_within(rect,width,height;tolerance=1.)
    all(isfinite,rect.origin) && all(isfinite,rect.widths) &&
        rect.origin[1]>=-tolerance && rect.origin[2]>=-tolerance &&
        rect.origin[1]+rect.widths[1]<=width+tolerance && rect.origin[2]+rect.widths[2]<=height+tolerance
end
function workflow_page_contents(fig)
    body=workflow_body(fig)
    next=workflow_button(fig,"next")
    previous=workflow_button(fig,"previous")
    pager=only(filter(b->b isa Label && startswith(b.text[],"page "),fig.content))
    count=parse(Int,last(split(pager.text[]," / ")))
    for i in 1:count
        previous.clicks[]+=1
    end
    text=String[]
    for i in 1:count
        push!(text,body.text[])
        next.clicks[]+=1
    end
    text
end
function workflow_mouse_click!(fig,point)
    ev=events(fig)
    ev.mouseposition[]=Tuple(Float64.(point))
    ev.mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.left,Mouse.press)
    ev.mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.left,Mouse.release)
end

@testset "Compact saved-workflow sections and responsive pages" begin
    mktempdir() do directory
        record,files=workflow_layout_fixture(joinpath(directory,
            "recording_with_long_saved_experiment_result_and_run_history_paths",
            "representative_sequence_with_complete_processing_settings"))
        ec=ExperimentController(record)
        ec.output_path[]=joinpath(dirname(files[1]),"native-results.jld2")
        ec.run_record_path[]=joinpath(dirname(files[1]),"run-history.jld2")
        identity=recipe_identity(ec.record[].recipe)
        longerror="failed: output verification refused; "*repeat("0123456789abcdef",12)*"; retain the previous displayed report"
        ec.status[]=longerror
        fig=experiment_workflow(ec;size=(900,600),batch=BatchRunner(),report_path_picker=()->"")
        screen=GLMakie.Screen(fig.scene;visible=false,start_renderloop=false)
        controls=workflow_controls(fig)
        body=workflow_body(fig)
        toggles=filter(b->b isa Toggle,fig.content)
        @test length(toggles)==3
        toggles[2].active[]=true;toggles[3].active[]=true
        initial_blocks=length(fig.content)
        listeners=(length(ec.running.listeners),length(ec.status.listeners),length(ec.allow_environment_change.listeners))
        for (width,height) in ((900,600),(1100,800))
            resize!(fig.scene,width,height)
            for kind in (:files,:replay,:reports)
                workflow_choose!(controls,kind)
                @test size(colorbuffer(screen))==(height,width)
                visible=filter(b->b isa Union{Button,Label,Menu,Toggle} && b.blockscene.visible[],fig.content)
                @test all(b->workflow_within(b.layoutobservables.computedbbox[],width,height),visible)
                @test all(b->workflow_within(GLMakie.Makie.boundingbox(b.blockscene),width,height),filter(b->b isa Label,visible))
                @test workflow_button(fig,"cancel after current pair").blockscene.visible[]
                @test any(b->b isa Label && startswith(b.text[],"Written pairs:") && b.blockscene.visible[],fig.content)
                @test all(b->b.layoutobservables.computedbbox[].origin[1]<-5000,
                    filter(b->b isa Union{Button,Label,Menu,Toggle} && !b.blockscene.visible[],fig.content))
                if haskey(ENV,"HAMMERHEAD_WORKFLOW_LAYOUT_SCREENSHOT")
                    path=replace(ENV["HAMMERHEAD_WORKFLOW_LAYOUT_SCREENSHOT"],".png"=>"-$(width)x$(height)-$kind.png")
                    Hammerhead.FileIO.save(path,copy(colorbuffer(screen)))
                end
            end
        end
        @test toggles[2].active[] && toggles[3].active[] && recipe_identity(ec.record[].recipe)==identity
        pages=workflow_page_contents(fig)
        unwrapped=replace(join(pages,"\n"),"\n"=>"")
        @test occursin(ec.output_path[],unwrapped) && occursin(ec.run_record_path[],unwrapped)
        @test occursin(replace(longerror,"\n"=>""),unwrapped)
        @test occursin(identity,unwrapped) && occursin("pass 5",lowercase(unwrapped))
        for i in 1:15
            workflow_choose!(controls,(:files,:replay,:reports)[mod1(i,3)])
            resize!(fig.scene,i%2==0 ? 900 : 1100,i%2==0 ? 600 : 800)
        end
        colorbuffer(screen)
        @test length(fig.content)==initial_blocks
        @test listeners==(length(ec.running.listeners),length(ec.status.listeners),length(ec.allow_environment_change.listeners))
        # Invisible controls must not respond at their former on-screen spot.
        workflow_choose!(controls,:files);colorbuffer(screen)
        snapshot=workflow_button(fig,"snapshot batch")
        point=snapshot.layoutobservables.computedbbox[].origin.+snapshot.layoutobservables.computedbbox[].widths./2
        clicks=snapshot.clicks[]
        workflow_choose!(controls,:reports);colorbuffer(screen)
        workflow_mouse_click!(fig,point)
        @test snapshot.clicks[]==clicks
        toggles[2].active[]=true;toggles[3].active[]=true # an actual visible toggle at that location may change
        workflow_choose!(controls,:files);colorbuffer(screen)
        status_calls=Ref(0)
        listener=on(ec.status) do _
            status_calls[]+=1
        end
        point=snapshot.layoutobservables.computedbbox[].origin.+snapshot.layoutobservables.computedbbox[].widths./2
        workflow_mouse_click!(fig,point)
        @test snapshot.clicks[]==clicks+1 && status_calls[]==1
        @test occursin("failed",ec.status[]) && recipe_identity(ec.record[].recipe)==identity
        off(listener)
        # Busy/cancelled/completed status retains the same persistent widgets.
        for (running,state) in ((true,:busy),(true,:cancel_requested),(false,:cancelled),(false,:completed))
            ec.running[]=running;ec.state[]=state;ec.status[]=longerror;ec.progress[]=(1,2)
            for kind in (:files,:replay,:reports)
                workflow_choose!(controls,kind);colorbuffer(screen)
                @test workflow_button(fig,"cancel after current pair").blockscene.visible[]
                @test any(b->b isa Label && b.text[]=="Written pairs: 1 / 2" && b.blockscene.visible[],fig.content)
            end
        end
        @test toggles[2].active[] && toggles[3].active[]
        GLMakie.destroy!(screen)
        GLMakie.Makie.current_figure()===fig && GLMakie.Makie.current_figure!(nothing)
    end
end
