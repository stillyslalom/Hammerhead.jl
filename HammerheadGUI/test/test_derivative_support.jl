using Test,HammerheadGUI,HammerheadGUI.Controllers
using HammerheadGUI.Hammerhead,HammerheadGUI.GLMakie
const GDS=HammerheadGUI.Controllers

function gui_derivative_result(;x=collect(100.:104.),y=collect(50.:54.),scale=nothing)
    u=[2xx+3yy for yy in y,xx in x];v=[5xx-2yy for yy in y,xx in x]
    dims=size(u)
    PIVResult(x,y,u,v,ones(dims),ones(dims),fill(NaN,dims),fill(NaN,dims),
        falses(dims),falses(dims),PIVParameters(window_size=16,overlap=8),nothing,scale)
end
@noinline function gui_derivative_released(path)
    ex=ResultExplorer(path;lazy=true)
    set_tool!(ex,:derivative_support)
    d=GDS._derived(ex)
    refs=(WeakRef(d.support.x.kind),WeakRef(current_result(ex).u))
    set_frame!(ex,2)
    ex,refs
end
@testset "Explorer derivative policy and honest support maps" begin
    r=gui_derivative_result()
    ex=ResultExplorer(r)
    original=available_fields(r)
    for field in GDS.DERIVED_FIELDS
        set_field!(ex,field)
        @test isequal(current_field_values(ex),field_values(r,field))
    end
    @test derivative_support_summary(ex).state===:off
    @test !hasproperty(GDS._derived(ex),:support)
    set_tool!(ex,:derivative_support)
    @test available_fields(r)==original
    @test all(f->f in available_fields(ex),GDS.DERIVATIVE_SUPPORT_FIELDS)
    s=derivative_support_summary(ex)
    @test s.nodes==25 && s.eligible==25 && s.x_supported==25 && s.y_supported==25
    @test s.finite_dudx==s.finite_dudy==s.finite_dvdx==s.finite_dvdy==25
    @test all(v->v isa Union{Symbol,Int},values(s)) # detached scalar API
    set_field!(ex,:derivative_x_stencil)
    @test current_field_values(ex)[3,:]==[2,1,1,1,3]
    set_color_limits!(ex;min=100,max=200)
    @test current_color_limits(ex)==(-.5,3.5)
    @test ex.color_min[]==100 && ex.color_max[]==200
    set_field!(ex,:derivative_finite_count)
    @test all(==(4),current_field_values(ex))
    set_derivative_stencil!(ex,:centered)
    set_field!(ex,:vorticity)
    @test count(isfinite,current_field_values(ex))==9
    @test occursin("both neighbors",GDS._current_field_label(ex))
    before=copy(current_field_values(ex))
    set_tool!(ex,:profile)
    @test ex.derivative_stencil[]===:centered && isequal(current_field_values(ex),before)
    @test !hasproperty(GDS._derived(ex),:support)
    click!(ex,101.,51.);click!(ex,103.,53.)
    @test ex.profile_data[]!==nothing
    profile=ex.profile_data[]
    set_derivative_stencil!(ex,:available)
    @test ex.profile_data[]===profile # u/v sampling is not a gradient computation
    set_tool!(ex,:circulation)
    for p in ((100.,50.),(104.,50.),(104.,54.),(100.,54.))
        click!(ex,p...)
    end
    alt_click!(ex)
    @test ex.circulation_result[].valid_area==16
    line=ex.circulation_result[].line
    set_derivative_stencil!(ex,:centered)
    @test ex.circulation_result[].valid_area≈4
    @test ex.circulation_result[].coverage_fraction≈.25
    @test ex.circulation_result[].line==line
    @test occursin("both immediate neighbors",tool_summary(ex))
    @test_throws ArgumentError set_derivative_stencil!(ex,:bad)
    @test ex.derivative_stencil[]===:centered
    @test_throws ArgumentError (ex.derivative_stencil[]=:bad)
    @test ex.derivative_stencil[]===:centered
    set_tool!(ex,:inspect)
    @test ex.derivative_stencil[]===:centered && count(isfinite,current_field_values(ex))==9
    set_field!(ex,:u)
    @test current_color_limits(ex)==(100.,200.) # scalar choices survived categorical maps

    # Current flags alone never establish replacement history.
    flagged=gui_derivative_result();flagged.outliers[3,3]=true
    fex=ResultExplorer(flagged);set_tool!(fex,:derivative_support)
    select_nearest!(fex,102.,52.)
    @test occursin("current outlier flag",describe_selection(fex))
    @test !occursin("replaced",describe_selection(fex))
    @test occursin("current outlier flag",describe_derivative_selection(fex))
    set_field!(fex,:derivative_eligibility)
    @test !current_field_values(fex)[3,3] && count(current_field_values(fex))==24
    set_field!(fex,:derivative_x_stencil)
    @test current_field_values(fex)[3,3]==0 && current_field_values(fex)[3,2]==3 && current_field_values(fex)[3,4]==2
    select_nearest!(fex,101.,52.)
    explanation=describe_derivative_selection(fex)
    @test occursin("first (3, 1)",explanation) && occursin("second (3, 2)",explanation)
    @test occursin("next immediate node: current outlier flag",explanation)
    set_derivative_stencil!(fex,:centered)
    @test current_field_values(fex)[3,2]==0
    flagged.u[1,1]=Inf
    @test_throws ArgumentError derivative_support_summary(fex)
    @test_throws ArgumentError describe_derivative_selection(fex)
    set_tool!(fex,:derivative_support) # explicit rebuild; no raw reload
    @test derivative_support_summary(fex).eligible==23
    @test !GDS._derived(fex).support.center_eligible[1,1]

    scaled=gui_derivative_result(;x=collect(104.:-1.:100.),scale=PhysicalScale(.02,.1,"mm","s"))
    sex=ResultExplorer(scaled);set_tool!(sex,:derivative_support)
    select_nearest!(sex,2.04,1.04)
    @test sex.selection[]==CartesianIndex(3,3)
    text=describe_derivative_selection(sex)
    @test occursin("signed span: -0.04 mm",text) && occursin("(1/mm)",text) && occursin("(1/s)",text)
    @test GDS._derived(sex).dudx≈fill(20.,5,5)
    twice=ResultExplorer(physical(scaled));set_tool!(twice,:derivative_support)
    @test isequal(GDS._derived(sex).dudx,GDS._derived(twice).dudx)
    # Unrepresentable weights do not erase a finite direct quotient.
    tiny=nextfloat(0.);sub=gui_derivative_result(;x=[0.,tiny,2tiny],y=[0.,1.,2.])
    sub.u .= repeat(reshape(sub.x,1,:),3,1);sub.v .= 0
    tex=ResultExplorer(sub);set_tool!(tex,:derivative_support)
    select_nearest!(tex,tiny,1.)
    @test all(==(1),GDS._derived(tex).dudx)
    @test occursin("weights unrepresentable",describe_derivative_selection(tex))
    malformed=gui_derivative_result(;x=[0.,1.,1.])
    mex=ResultExplorer(malformed)
    @test_throws ArgumentError set_tool!(mex,:derivative_support)
    @test mex.tool[]===:inspect && isempty(mex.derived_cache)
    singleton=ResultExplorer(gui_derivative_result(;x=[0.]))
    @test derivative_support_summary(singleton).state===:singleton_grid
    @test_throws ArgumentError set_tool!(singleton,:derivative_support)

    # A controller-only listener must never receive a support-map fallback
    # that is unavailable on the newly published tracking frame.
    track=TrackingResult([Trajectory{Float64}(1,[1.,2.],[1.,2.],[1,2])],2,PTVParameters())
    transition=ResultExplorer([gui_derivative_result(),track])
    set_tool!(transition,:derivative_support)
    set_field!(transition,:derivative_x_stencil)
    notified=Symbol[]
    on(transition.field) do field
        push!(notified,field)
        @test field in available_fields(transition)
    end
    set_frame!(transition,2)
    @test !isempty(notified) && all(==(:speed),notified)
    @test transition.field[]===:speed && transition.tool[]===:inspect

    mktempdir() do directory
        path=joinpath(directory,"grid.jld2")
        save_results(path,[gui_derivative_result(),gui_derivative_result(;x=collect(10.:15.))])
        lazy,refs=gui_derivative_released(path)
        GC.gc(true);GC.gc(true)
        @test all(ref->ref.value===nothing,refs)
        @test length(lazy.derived_cache)<=1 && lazy.derivative_stencil[]===:available
        set_derivative_stencil!(lazy,:centered)
        set_frame!(lazy,1)
        @test lazy.derivative_stencil[]===:centered
        @test derivative_support_summary(lazy).x_supported==15
        # Unreadable next result preserves the current rich cache/frame/selection.
        good=current_result(lazy);select_nearest!(lazy,102.,52.)
        cached=GDS._derived(lazy)
        Hammerhead.jldopen(path,"a+") do file
            delete!(file,Hammerhead.result_key(2))
            file[Hammerhead.result_key(2)]="malformed result"
        end
        @test_throws ArgumentError set_frame!(lazy,2)
        @test lazy.frame[]==1 && current_result(lazy)===good && GDS._derived(lazy)===cached
        @test lazy.selection[]==CartesianIndex(3,3)
    end
end

@testset "Derivative discrete legends and paged offscreen inspection" begin
    r=gui_derivative_result(;scale=PhysicalScale(.02,.1,"mm","s"))
    r.mask[2,3]=true;r.outliers[3,3]=true;r.u[4,2]=Inf
    ex=ResultExplorer(r);set_tool!(ex,:derivative_support)
    select_nearest!(ex,2.02,1.04)
    fig=result_explorer(ex;size=(1000,700))
    screen=GLMakie.Screen(fig.scene;visible=false,start_renderloop=false)
    tools=only(filter(b->b isa Menu && ("derivative support",:derivative_support) in b.options[],fig.content))
    @test tools.selection[]===:derivative_support
    for (field,suffix) in ((:derivative_x_stencil,"stencils"),(:derivative_finite_count,"finite"),(:vorticity,"vorticity"))
        set_field!(ex,field)
        @test size(colorbuffer(screen))==(700,1000)
        bars=filter(b->b isa Colorbar,fig.content)
        @test length(bars)==1
        if field in GDS.DERIVATIVE_SUPPORT_FIELDS
            @test bars[1].ticks[]==GDS._derivative_map_legend(field)
            @test bars[1].limits[]==GDS._derivative_map_limits(field)
        end
        @test any(b->b isa Label && occursin("eligible centers",b.text[]),fig.content)
        if haskey(ENV,"HAMMERHEAD_DERIVATIVE_SCREENSHOT")
            Hammerhead.FileIO.save(replace(ENV["HAMMERHEAD_DERIVATIVE_SCREENSHOT"],".png"=>"-$suffix.png"),copy(colorbuffer(screen)))
        end
    end
    set_derivative_stencil!(ex,:centered)
    @test occursin("both neighbors",only(filter(b->b isa Colorbar,fig.content)).label[])
    policy=only(filter(b->b isa Menu && ("require both neighbors",:centered) in b.options[],fig.content))
    # Real mouse activation of the policy menu, rather than a direct observable write.
    function mouse!(point)
        ev=events(fig)
        ev.mouseposition[]=Tuple(Float64.(point))
        ev.mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.left,Mouse.press)
        ev.mousebutton[]=GLMakie.Makie.MouseButtonEvent(Mouse.left,Mouse.release)
    end
    box=policy.layoutobservables.computedbbox[]
    former=box.origin.+box.widths./2
    policy.direction[]=:down
    mouse!(former);colorbuffer(screen)
    @test policy.is_open[]
    # The first option is immediately below the selector, in the live menu's
    # own popup region; this uses the selector's measured row height.
    mouse!((former[1],box.origin[2]-box.widths[2]/2));colorbuffer(screen)
    @test ex.derivative_stencil[]===:available
    set_derivative_stencil!(ex,:centered)
    set_tool!(ex,:profile)
    @test occursin("both neighbors",only(filter(b->b isa Colorbar,fig.content)).label[])
    click!(ex,2.00,1.00);click!(ex,2.08,1.08)
    @test ex.profile_data[]!==nothing && length(filter(b->b isa Axis,fig.content))==2
    @test policy.layoutobservables.computedbbox[].origin[1]<-5000
    mouse!(former)
    @test ex.derivative_stencil[]===:centered
    GLMakie.destroy!(screen)
    # Model notifications must publish a compatible kind/tool/selection bundle
    # before any view callback attempts to draw the new frame.
    base=gui_derivative_result()
    stereo=StereoPIVResult(base.x,base.y,0.,base.u,base.v,copy(base.u),
        base.uncertainty_u,base.uncertainty_v,copy(base.uncertainty_u),
        base.outliers,base.mask,base,base,base.parameters)
    particles=Particles([1.,2.],[1.,2.],[1.,1.],[3.,3.])
    ptv=PTVResult([1.,2.],[1.,2.],[1.,1.],[0.,0.],[.1,.1],falses(2),
        [1,2],[1,2],particles,particles,PTVParameters())
    track=TrackingResult([Trajectory{Float64}(1,[1.,2.],[1.,2.],[1,2])],2,PTVParameters())
    mixed=ResultExplorer([base,gui_derivative_result(;x=[0.]),stereo,ptv,track,base])
    set_derivative_stencil!(mixed,:centered);set_tool!(mixed,:derivative_support)
    set_field!(mixed,:derivative_x_stencil);select_nearest!(mixed,102.,52.)
    other=result_explorer(mixed;size=(1000,700))
    other_screen=GLMakie.Screen(other.scene;visible=false,start_renderloop=false)
    for i in 2:6
        set_frame!(mixed,i)
        @test size(colorbuffer(other_screen))==(700,1000)
        @test mixed.tool[]===:inspect && mixed.derivative_stencil[]===:centered
        @test mixed.field[]===:magnitude || mixed.field[]===:speed
        @test !hasproperty(get(mixed.derived_cache,i,(;)),:support)
        if i<6
            @test_throws ArgumentError set_tool!(mixed,:derivative_support)
            @test derivative_support_summary(mixed).state in (:singleton_grid,:unsupported_kind)
        end
    end
    set_tool!(mixed,:derivative_support);set_field!(mixed,:derivative_x_stencil)
    @test size(colorbuffer(other_screen))==(700,1000)
    @test count(!iszero,current_field_values(mixed))==15
    GLMakie.destroy!(other_screen)
    GLMakie.Makie.current_figure!(nothing)
end
