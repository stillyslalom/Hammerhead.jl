# Result-explorer view: the GLMakie shell around a Controllers.ResultExplorer.
# All state lives in the controller; this file only renders it and forwards
# widget/mouse input into the controller API.

"""
    result_explorer(source; size = (1000, 700)) -> Figure
    result_explorer(path::AbstractString; lazy = false, format = :native, size = (1000, 700)) -> Figure

Open results from a `PIVResult`, `StereoPIVResult`, `PTVResult`, or
`TrackingResult`, a sequence of results, a saved-results path, or a
[`ResultExplorer`](@ref). Pass a controller to control the open view through
its observables.
Use `lazy = true` for a completed file, or supply a [`ResultFile`](@ref),
to retain only the selected display result while navigating. Read failures
appear below the inspection panel and leave the previous frame displayed.

For a gridded (`PIVResult` / `StereoPIVResult`) result the view shows a
scalar field (magnitude, components, diagnostics, or available uncertainty)
as a heatmap in image orientation (y down) with vector arrows. Flagged
vectors can be highlighted. A `PTVResult` is drawn as a
colored particle scatter with optional displacement arrows, and a
`TrackingResult` as trajectory polylines colored by mean speed (breaks at
frame gaps). A frame slider scrubs a sequence, and a click-to-inspect panel
summarizes the selected item in physical units when a scale is attached.
A `TimedTrackingResult` or an explicit `format=:timed_tracking` artifact keeps
actual timestamps through spatial conversion. The entire trajectory bundle is
one result (no lazy per-sample navigation). Colors show arithmetic observation
means of actual-time secant magnitudes; unavailable speeds use neutral gray,
including singleton tracks. Breaks mark omitted selected frames, not long
elapsed intervals alone. Selection describes the whole track and exact time
range, not a picked observation's instantaneous velocity. Unknown/assumed time
units remain labeled. Timing integrity checks scan the whole current bundle.
The unchecked "recorded processing details" toggle opts into verified raw
history/execution inspection for lazy native files. The paged details panel
labels raw primary/residual measurements in pixels beside displayed units and
returns its vertical space when disabled. It makes no uncertainty applicability
or accuracy claim and does not infer absent history from current flags.
While details are open, profile lines remain overlaid but their separate graph
is hidden; closing details reveals a profile placed on the current display.

The planar "derivative support" tool uses that drawer for scalar support counts
and selected immediate contributors. Its four discrete maps keep excluded and
unavailable nodes visible with fixed legends. The policy selector can require
both immediate neighbors; this explorer-wide policy persists after changing
tools and remains labelled on derived scalar fields. Nonfinite scalar output
is gray while inspecting support. Recorded details return after leaving the
tool if their toggle remains enabled. No measurement origin or uncertainty
applicability is inferred from derivative support or current flags.
"""
result_explorer(source; kwargs...) = result_explorer(ResultExplorer(source); kwargs...)
result_explorer(path::AbstractString; lazy::Bool = false,format::Symbol=:native, kwargs...) =
    result_explorer(ResultExplorer(path; lazy,format); kwargs...)

function result_explorer(ex::ResultExplorer; size = (1000, 700))
    fig = Figure(; size)
    result_explorer!(fig[1, 1], ex)
    return fig
end

# Preserve every digit of exact timestamps/opaque clock labels, including long
# unbroken tokens, without allowing the selected-track panel to grow unbounded.
function _timed_selection_pages(text)
    lines=String[]
    for line in split(text,'\n')
        chars=collect(line)
        while length(chars)>28
            boundary=findlast(isspace,view(chars,1:28))
            width=boundary===nothing || boundary<=1 ? 28 : boundary-1
            push!(lines,rstrip(String(chars[1:width])))
            consumed=boundary===nothing || boundary<=1 ? width : boundary
            chars=chars[consumed+1:end]
        end
        push!(lines,String(chars))
    end
    [join(lines[i:min(i+6,length(lines))],"\n") for i in 1:7:length(lines)]
end

"""
    result_explorer!(target, ex::ResultExplorer) -> GridLayout

Build the result-explorer view into `target` (a `GridPosition`, e.g.
`fig[1, 2]`), for embedding in a larger layout.
"""
function result_explorer!(target, ex::ResultExplorer)
    gl = GridLayout(target)
    n = nframes(ex)
    timed_selection=current_result(ex) isa TimedTrackingResult

    ax = Axis(gl[1, 1]; xlabel = "x", ylabel = "y",
              yreversed = true, aspect = DataAspect(),
              title = ex.path === nothing ? "" : basename(ex.path))

    # Standalone colorbar driven by observables so the heatmap can be
    # recreated per refresh (grid sizes may change across a mixed sequence).
    crange = Observable((0.0, 1.0))
    clabel = Observable("")
    cmap=Observable{Any}(:viridis)
    cticks=Observable{Any}(GLMakie.Makie.automatic)
    colorbar=Ref{Any}(Colorbar(gl[1,2];colormap=:viridis,limits=crange,label=clabel))
    colorbar_kind=Ref(:scalar)
    function refresh_colorbar!()
        kind=ex.field[] in Controllers.DERIVATIVE_SUPPORT_FIELDS ? ex.field[] : :scalar
        if kind!==colorbar_kind[]
            # Makie's categorical mapping/type observers are not updated
            # atomically when changing a continuous colormap into bands.
            # Recreate only on legend-type changes, with constant mapping.
            delete!(colorbar[])
            colorbar[]=Colorbar(gl[1,2];colormap=cmap[],limits=crange,label=clabel,ticks=cticks[])
            colorbar_kind[]=kind
        end
    end

    controls = GridLayout(gl[1:4, 3]; tellheight = false, valign = :top)
    rowgap!(controls,4)
    Label(controls[1, 1], "field"; halign = :left, font = :bold)
    menu = Menu(controls[2, 1]; options = [("|displacement|", :magnitude)])
    toggles = GridLayout(controls[3, 1]; halign = :left)
    vec_toggle = Toggle(toggles[1, 1]; active = ex.show_vectors[])
    Label(toggles[1, 2], "vectors"; halign = :left)
    out_toggle = Toggle(toggles[2, 1]; active = ex.highlight_outliers[])
    Label(toggles[2, 2], "flag outliers"; halign = :left)
    Label(controls[4, 1], "color range"; halign = :left, font = :bold)
    cgrid = GridLayout(controls[5, 1]; halign = :left)
    cmode_menu = Menu(cgrid[1, 1:2]; tellwidth = false,
                      options = [("robust (2–98%)", :robust), ("full range", :full)])
    cmin_box = Textbox(cgrid[2, 1]; placeholder = "min: auto", width = 100)
    cmax_box = Textbox(cgrid[2, 2]; placeholder = "max: auto", width = 100)
    Label(controls[6, 1], "tool"; halign = :left, font = :bold)
    tool_menu = Menu(controls[7, 1]; tellwidth = false,
                     options = [("inspect", :inspect), ("profile", :profile),
                                ("circulation", :circulation),("derivative support",:derivative_support)])
    tool_info = Label(controls[8, 1], ""; halign = :left, justification = :left,
                      word_wrap = true, width = 210, tellwidth = false)
    inspection_hint=Label(controls[9, 1], "click a vector to inspect"; halign = :left, font = :bold)
    selection_panel=GridLayout(controls[10,1])
    info = Label(selection_panel[1,1:3], ""; halign = :left,valign=:top, justification = :left,fontsize=14,
        word_wrap=true,width=220,tellwidth=false,height=timed_selection ? 155 : Auto())
    selection_previous=Button(selection_panel[2,1];label="previous",fontsize=12,tellwidth=false)
    selection_page_label=Label(selection_panel[2,2],"";fontsize=12)
    selection_next=Button(selection_panel[2,3];label="next",fontsize=12,tellwidth=false)
    for widget in (selection_previous,selection_next,selection_page_label)
        widget.blockscene.visible[]=timed_selection
    end
    rowsize!(selection_panel,2,Fixed(timed_selection ? 28 : 0))
    Label(controls[11, 1], ex.status; halign = :left, justification = :left,
          word_wrap = true, width = 210, tellwidth = false)
    companion_mode=GridLayout(controls[12,1])
    colgap!(companion_mode,6)
    companion_toggle=Toggle(companion_mode[1,1];active=ex.companion_enabled[],halign=:left)
    Label(companion_mode[1,2],"recorded processing details";halign=:left,word_wrap=true,width=170,fontsize=13,tellwidth=false)
    companion_panel=GridLayout(gl[4,1:2])
    # Reserve profile space only while a profile is displayed. An empty Auto
    # row would otherwise share the plot's height when details add a fourth row.
    rowsize!(gl,3,Fixed(0))
    companion_pager=GridLayout(companion_panel[1,1:2])
    companion_previous=Button(companion_pager[1,1];label="previous details",tellwidth=false)
    companion_page_label=Label(companion_pager[1,2],"")
    companion_next=Button(companion_pager[1,3];label="next details",tellwidth=false)
    stencil_menu=Menu(companion_pager[1,4];options=[("available neighbors",:available),("require both neighbors",:centered)],tellwidth=false,fontsize=12)
    on(stencil_menu.layoutobservables.suggestedbbox;priority=typemax(Int)) do box
        if ex.tool[]!==:derivative_support && box.origin[1]>-5000
            stencil_menu.layoutobservables.suggestedbbox[]=Rect2f(-10000,-10000,box.widths...)
            return Consume(true)
        end
        Consume(false)
    end
    companion_info=Label(companion_panel[2,1],"";halign=:left,valign=:top,justification=:left,
        word_wrap=true,width=300,tellwidth=false,fontsize=13)
    companion_node=Label(companion_panel[2,2],"";halign=:left,valign=:top,justification=:left,
        word_wrap=true,width=300,tellwidth=false,fontsize=13)
    colsize!(gl, 3, Fixed(230))

    Label(gl[2, 1:2][1, 1],current_result(ex) isa TimedTrackingResult ? "bundle" : "frame")
    slider = Slider(gl[2, 1:2][1, 2]; range = 1:max(n, 1), startvalue = ex.frame[])
    Label(gl[2, 1:2][1, 3], lift((i, m) -> "$i / $m", ex.frame, ex.count))
    # Grow the slider range as a live batch appends results (push_result!).
    on(ex.count) do m
        rng = 1:max(m, 1)
        rng == slider.range[] || (slider.range[] = rng)
    end

    # Widget -> controller (equality guards break the notification cycles;
    # Observables notify even when the value is unchanged).
    _sync_toggle!(vec_toggle, ex.show_vectors)
    _sync_toggle!(out_toggle, ex.highlight_outliers)
    on(companion_toggle.active) do enabled
        enabled==ex.companion_enabled[] && return
        try
            set_companion_inspection!(ex,enabled)
        catch err
            isempty(ex.status[]) && (ex.status[]=Controllers._errmsg(err))
            companion_toggle.active[]=ex.companion_enabled[]
        end
    end
    on(ex.companion_enabled) do enabled
        enabled==companion_toggle.active[] || (companion_toggle.active[]=enabled)
    end
    companion_page=Observable(1)
    companion_pages=Ref([("","")])
    function show_companion_page!()
        page=clamp(companion_page[],1,length(companion_pages[]))
        left,right=companion_pages[][page]
        companion_info.text[]=left
        companion_node.text[]=right
        companion_page_label.text[]="details $page / $(length(companion_pages[]))"
    end
    on(_->show_companion_page!(),companion_page)
    on(_->(companion_page[]=max(1,companion_page[]-1)),companion_previous.clicks)
    on(_->(companion_page[]=min(length(companion_pages[]),companion_page[]+1)),companion_next.clicks)
    function companion_chunks(text)
        lines=String[]
        for line in split(text,'\n')
            current=""
            for word in split(line)
                if !isempty(current) && length(current)+length(word)+1>46
                    push!(lines,current)
                    current=word
                else
                    current=isempty(current) ? word : current*" "*word
                end
            end
            push!(lines,current)
        end
        [join(lines[i:min(i+10,length(lines))],"\n") for i in 1:11:length(lines)]
    end
    function refresh_companions!()
        derivatives=ex.tool[]===:derivative_support
        details=ex.companion_enabled[] || derivatives
        rowsize!(gl,4,Fixed(details ? 220 : 0))
        companion_info.visible[]=details
        companion_node.visible[]=details
        for block in (companion_previous,companion_next)
            block.blockscene.visible[]=details
        end
        companion_page_label.visible[]=details
        stencil_menu.blockscene.visible[]=derivatives
        colsize!(companion_pager,4,Fixed(derivatives ? 220 : 0))
        try
            summary,node=derivatives ? Controllers._derivative_text(ex) : Controllers._companion_text(ex)
            left,right=companion_chunks(summary),companion_chunks(node)
            companion_pages[]=[(i<=length(left) ? left[i] : "",i<=length(right) ? right[i] : "") for i in 1:max(length(left),length(right))]
        catch err
            companion_pages[]=[("Processing details unavailable: $(Controllers._errmsg(err))","")]
        end
        companion_page[]=1
    end
    onany((args...)->refresh_companions!(),ex.frame,ex.selection,ex.companion_enabled,ex.tool,ex.derivative_stencil,ex.field)
    on(stencil_menu.selection) do policy
        (policy===nothing || policy===ex.derivative_stencil[]) && return
        try
            set_derivative_stencil!(ex,policy)
        catch err
            ex.status[]=Controllers._errmsg(err)
            stencil_menu.i_selected[]=findfirst(o->last(o)===ex.derivative_stencil[],stencil_menu.options[])
        end
    end
    _sync_menu!(stencil_menu,ex.derivative_stencil)
    on(slider.value) do i
        i == ex.frame[] && return
        try
            set_frame!(ex, i)
        catch err
            # The controller reports the error and keeps the prior result.
            isempty(ex.status[]) && (ex.status[] = Controllers._errmsg(err))
            set_close_to!(slider, ex.frame[])
        end
    end
    on(ex.frame) do i
        i == slider.value[] || set_close_to!(slider, i)
    end
    on(menu.selection) do f
        f === nothing || f == ex.field[] || set_field!(ex, f)
    end
    _sync_menu!(cmode_menu, ex.color_mode)
    # Tool menu: set_tool! rejects analysis tools on non-planar results —
    # revert the menu selection instead of leaving it out of sync.
    on(tool_menu.selection) do t
        (t === nothing || t == ex.tool[]) && return
        try
            set_tool!(ex, t)
        catch err
            ex.status[]=Controllers._errmsg(err)
            i = findfirst(==(ex.tool[]), Controllers.EXPLORER_TOOLS)
            tool_menu.i_selected[] = something(i, 1)
        end
    end
    on(ex.tool) do t
        i = findfirst(==(t), Controllers.EXPLORER_TOOLS)
        (i === nothing || i == tool_menu.i_selected[]) || (tool_menu.i_selected[] = i)
    end
    tool_menu.i_selected[]=something(findfirst(==(ex.tool[]),Controllers.EXPLORER_TOOLS),1)
    # Manual colorbar bounds: one-way widget -> controller (the box's
    # placeholder documents the cleared state); junk entries are ignored.
    on(cmin_box.stored_string) do s
        s === nothing && return
        try
            set_color_limits!(ex; min = s)
        catch
        end
    end
    on(cmax_box.stored_string) do s
        s === nothing && return
        try
            set_color_limits!(ex; max = s)
        catch
        end
    end

    # Clicks route through the active tool (left = place/select, right =
    # close a circulation contour).
    on(events(ax.scene).mousebutton) do mb
        if mb.action == Mouse.press && GLMakie.Makie.is_mouseinside(ax.scene)
            if mb.button == Mouse.left
                pos = GLMakie.Makie.mouseposition(ax.scene)
                Controllers.click!(ex, pos[1], pos[2])
            elseif mb.button == Mouse.right
                Controllers.alt_click!(ex)
            end
        end
        return Consume(false)
    end

    # Selection marker and info panel.
    sel_points = Observable(Point2f[])
    sel_plot = scatter!(ax, sel_points; color = :transparent,
                        strokecolor = :cyan, strokewidth = 2.5, markersize = 16)
    translate!(sel_plot, 0, 0, 2)
    selection_pages=Ref(String[])
    selection_page=Ref(1)
    function show_selection_page!()
        pages=selection_pages[]
        isempty(pages) && return
        selection_page[]=clamp(selection_page[],1,length(pages))
        info.text[]=pages[selection_page[]]
        selection_page_label.text[]="$(selection_page[]) / $(length(pages))"
    end
    on(selection_previous.clicks) do _
        selection_page[]-=1
        show_selection_page!()
    end
    on(selection_next.clicks) do _
        selection_page[]+=1
        show_selection_page!()
    end
    function refresh_selection!()
        sel,tool=ex.selection[],ex.tool[]
        pt = selection_point(current_result(ex), sel)
        sel_points[] = pt === nothing ? Point2f[] : [Point2f(pt[1], pt[2])]
        text=tool===:inspect ? describe_selection(ex) : ""
        if timed_selection
            selection_pages[]=isempty(text) ? ["Select a trajectory to inspect its actual-time range and speed."] : _timed_selection_pages(text)
            selection_page[]=1
            show_selection_page!()
        else
            info.text[]=text
        end
        inspection_hint.text[]=tool===:inspect ? (current_result(ex) isa TimedTrackingResult ? "click a trajectory to inspect" : "click a vector to inspect") : ""
    end
    onany((args...)->refresh_selection!(),ex.selection,ex.frame,ex.tool)

    # Tool overlay (profile line / circulation contour) and the profile
    # side panel, which appears as a third row while a profile is set.
    tool_line = Observable(Point2f[])
    tool_plot = lines!(ax, tool_line; color = :cyan, linewidth = 2)
    tool_marks = scatter!(ax, tool_line; color = :cyan, markersize = 8)
    translate!(tool_plot, 0, 0, 2)
    translate!(tool_marks, 0, 0, 2)
    profile_ax = Ref{Any}(nothing)
    function refresh_tools!()
        pts = [Point2f(p[1], p[2]) for p in ex.tool_points[]]
        # a closed circulation contour draws with its closing segment
        res = ex.circulation_result[]
        ex.tool[] === :circulation && res !== nothing && !isempty(pts) &&
            push!(pts, pts[1])
        tool_line[] = pts
        tool_info.text[] = tool_summary(ex)
        pd = ex.profile_data[]
        if pd === nothing || ex.companion_enabled[] || ex.tool[]===:derivative_support
            if profile_ax[] !== nothing
                delete!(profile_ax[])
                profile_ax[] = nothing
            end
            rowsize!(gl,3,Fixed(0))
            pd===nothing || (tool_info.text[]*="\nClose recorded details to show the profile graph.")
        else
            rowsize!(gl,3,Fixed(150))
            if profile_ax[] === nothing
                # span only the plot + colorbar columns: the controls column
                # holds the (tellheight = false) info text, which would
                # otherwise overlap the panel
                profile_ax[] = Axis(gl[3, 1:2]; height = 150,
                                    xlabel = "s along line",
                                    ylabel = "sampled value")
            end
            ax2 = profile_ax[]
            empty!(ax2)
            lines!(ax2, pd.s, pd.u; color = :steelblue)
            lines!(ax2, pd.s, pd.v; color = :darkorange)
            lines!(ax2, pd.s, hypot.(pd.u, pd.v); color = :black)
            autolimits!(ax2)
        end
        return
    end
    onany((args...) -> refresh_tools!(),
          ex.tool, ex.tool_points, ex.profile_data, ex.circulation_result,ex.companion_enabled)

    function refresh_menu!()
        r=current_result(ex)
        fields = available_fields(ex)
        opts = [(f in Controllers.DERIVATIVE_SUPPORT_FIELDS ? Controllers._DERIVATIVE_MAP_NAMES[findfirst(==(f),Controllers.DERIVATIVE_SUPPORT_FIELDS)] : f===:magnitude && r.scale!==nothing ? "speed" : field_name(f), f) for f in fields]
        opts == menu.options[] || (menu.options[] = opts)
        i = something(findfirst(==(ex.field[]), fields), 1)
        i == menu.i_selected[] || (menu.i_selected[] = i)
    end
    on(_ -> refresh_menu!(), ex.frame)
    on(_ -> refresh_menu!(), ex.field) # sync the menu on programmatic set_field!
    on(_ -> refresh_menu!(),ex.tool)

    # The plots are recreated per refresh rather than driven by per-argument
    # observables: grid sizes (and the plot type itself, across a mixed
    # sequence) can change between frames, and sequential x/y/data updates
    # would render transiently mismatched args.
    plots = Any[]
    has_drawn = Ref(false)

    # Quiver-style fast path: linesegments shafts + one rotated triangle
    # marker per arrow head. arrows2d is a poly/mesh recipe whose pixel-space
    # tip sizing recomputes per pan/zoom frame — interaction crawls with
    # thousands of arrows — while these two primitive plots are static in
    # data space and cheap to render.
    function _draw_arrows!(r)
        ex.show_vectors[] || return
        d = vector_data(r)
        isempty(d.x) && return
        ls = auto_lengthscale(r, d)
        n = length(d.x)
        segs = Vector{Point2f}(undef, 2n)
        tips = Vector{Point2f}(undef, n)
        rots = Vector{Float32}(undef, n)
        for k in 1:n
            tip = Point2f(d.x[k] + ls * d.u[k], d.y[k] + ls * d.v[k])
            segs[2k - 1] = Point2f(d.x[k], d.y[k])
            segs[2k] = tip
            tips[k] = tip
            # marker rotation is screen-space CCW and the axis is yreversed,
            # so (du, dv) points along (du, -dv) on screen; :utriangle points
            # up (screen +y), hence the -π/2 offset.
            rots[k] = Float32(atan(-d.v[k], d.u[k]) - π / 2)
        end
        if ex.highlight_outliers[] && any(d.outlier)
            cols = [o ? :red : :black for o in d.outlier]
            shafts = linesegments!(ax, segs; color = repeat(cols, inner = 2),
                                   linewidth = 1.5)
            heads = scatter!(ax, tips; marker = :utriangle, rotation = rots,
                             markersize = 9, color = cols)
        else
            shafts = linesegments!(ax, segs; color = :black, linewidth = 1.5)
            heads = scatter!(ax, tips; marker = :utriangle, rotation = rots,
                             markersize = 9, color = :black)
        end
        translate!(shafts, 0, 0, 1)
        translate!(heads, 0, 0, 1)
        push!(plots, shafts, heads)
        return
    end

    function _draw!(r::Union{PIVResult,StereoPIVResult})
        data = Controllers.current_field_values(ex)   # cached for derived fields
        lo, hi = current_color_limits(ex)
        crange[] = (lo, hi)
        clabel[] = Controllers._current_field_label(ex)
        categorical=ex.field[] in Controllers.DERIVATIVE_SUPPORT_FIELDS
        if categorical
            ticks,labels=Controllers._derivative_map_legend(ex.field[])
            cmap[]=GLMakie.Makie.cgrad([:gray75,:steelblue,:darkorange,:purple,:darkcyan][1:length(ticks)],length(ticks);categorical=true)
            cticks[]=(ticks,labels)
        else
            cmap[]=:viridis;cticks[]=GLMakie.Makie.automatic
        end
        h = heatmap!(ax, collect(r.x), collect(r.y), permutedims(data);
                     colormap = cmap[], colorrange = (lo, hi),
                     nan_color = ex.tool[]===:derivative_support ? :gray85 : :transparent)
        translate!(h, 0, 0, -1)
        push!(plots, h)
        _draw_arrows!(r)
        return
    end

    function _draw!(r::PTVResult)
        data = field_values(r, ex.field[])
        lo, hi = current_color_limits(ex)
        crange[] = (lo, hi)
        clabel[] = field_label(r, ex.field[])
        if !isempty(r.x)
            sc = scatter!(ax, Point2f.(r.x, r.y); color = Float64.(data),
                          colormap = :viridis, colorrange = (lo, hi), markersize = 9)
            translate!(sc, 0, 0, -1)
            push!(plots, sc)
        end
        _draw_arrows!(r)
        return
    end

    function _draw!(r::TrackingResult)
        speeds = field_values(r, :speed)
        lo, hi = current_color_limits(ex)
        crange[] = (lo, hi)
        clabel[] = field_label(r, :speed)
        for (k, t) in pairs(r.trajectories)
            xs, ys = trajectory_points(t)
            length(xs) >= 2 || continue
            col = isfinite(speeds[k]) ? speeds[k] : lo
            ln = lines!(ax, Point2f.(xs, ys); color = Float64(col),
                        colormap = :viridis, colorrange = (lo, hi))
            push!(plots, ln)
        end
        return
    end

    function _draw!(r::TimedTrackingResult)
        summary=Hammerhead.tracking_speed_summary(r) # one whole validation, never per track
        lo,hi=Controllers._current_color_limits(ex,r,summary.speeds)
        crange[]=(lo,hi)
        clabel[]=string("observation-mean secant speed (",summary.length_unit,"/",
            something(summary.time_unit,"unknown sample-time unit"),")")
        unavailable=count(!,summary.available)
        ax.title[]=string(ex.path===nothing ? "Actual-time trajectories" : basename(ex.path),
            "\n",unavailable," / ",length(summary.available)," speeds unavailable (gray)")
        for (k,t) in pairs(r.result.trajectories)
            xs,ys=trajectory_points(t)
            isempty(xs) && continue
            positions=Point2f.(xs,ys)
            options=summary.available[k] ? (;color=summary.speeds[k],colormap=:viridis,colorrange=(lo,hi)) : (;color=:gray55)
            push!(plots,lines!(ax,positions;options...))
            # A singleton (or isolated observations on either side of a gap)
            # must remain visible/selectable even when no segment can be drawn.
            push!(plots,scatter!(ax,positions;markersize=5,options...))
        end
        return
    end

    function refresh_plots!()
        r = current_result(ex)
        cmap[]=:viridis;cticks[]=GLMakie.Makie.automatic
        scale=r isa TimedTrackingResult ? r.result.scale : r.scale
        ax.xlabel[], ax.ylabel[] = Hammerhead.plot_axis_labels(scale)

        # capture/restore targetlimits (not limits!): the pre-reversal rect,
        # so the image-orientation yreversed flip survives the restore
        limits = ax.targetlimits[]
        for p in plots
            delete!(ax, p)
        end
        empty!(plots)
        _draw!(r)
        refresh_colorbar!()
        has_drawn[] && (ax.targetlimits[] = limits) # keep the user's zoom across refreshes
        has_drawn[] = true
        return
    end
    onany((args...) -> refresh_plots!(),
          ex.frame, ex.field, ex.show_vectors, ex.highlight_outliers,
          ex.color_mode, ex.color_min, ex.color_max,ex.tool)

    refresh_menu!()
    refresh_companions!()
    refresh_plots!()
    refresh_selection!()
    refresh_tools!()
    return gl
end
