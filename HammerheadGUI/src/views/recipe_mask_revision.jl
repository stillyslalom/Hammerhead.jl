"""
    recipe_mask_revision(controller; reference_controller=RecipeMaskReferenceController(),
        size=(1100,800), save_path_picker, mask_path_picker, open_revision)

Revise a saved planar recipe's full-image exclusion mask. Explicitly load a
verified raw reference, draw exclusions or holes, and apply the combined raster.
Reference loading does not execute preprocessing or scripts. Red pixels are
excluded; the cyan outline is the captured original-image ROI. Disabling retains
the draft raster, clearing the editor differs from resetting the imported mask,
and holes can remove exclusions from the imported raster too.

Metadata validation and saving are separate actions. Unfinished polygons and a
mask draft replaced since reference loading refuse Apply; reload the reference to
begin a new session. Saving preserves other current recipe drafts. The separate
Open saved revision action opens a new workflow. Queued verification/I/O may
pause rendering.
"""
function recipe_mask_revision(rc::RecipeRevisionController;
        reference_controller=RecipeMaskReferenceController(),size=(1100,800),
        save_path_picker::Function=()->save_file(;filterlist="jld2"),
        mask_path_picker::Function=()->pick_file(),
        open_revision::Function=record->display(GLMakie.Screen(),experiment_workflow(ExperimentController(record))))
    fig=Figure(;size)
    recipe_mask_revision!(fig[1,1],rc;reference_controller,save_path_picker,mask_path_picker,open_revision)
    fig
end

"""
    recipe_mask_revision!(target, controller; reference_controller,
        save_path_picker, mask_path_picker, open_revision)

Embed the saved-mask editor. Pointer gestures edit only the current reference
session; Apply explicitly enables its raster. Import decoding options and all
recipe drafts are captured by the controller before pickers/observers. The raw
reference keeps its own identity when drafts change. Details and complete
imported settings remain available in paged read-only tabs.
"""
function recipe_mask_revision!(target,rc::RecipeRevisionController;
        reference_controller=RecipeMaskReferenceController(),
        save_path_picker::Function=()->save_file(;filterlist="jld2"),
        mask_path_picker::Function=()->pick_file(),
        open_revision::Function=record->display(GLMakie.Screen(),experiment_workflow(ExperimentController(record))))
    mc=reference_controller;gl=GridLayout(target);colsize!(gl,1,Fixed(240));colgap!(gl,16)
    controls=GridLayout(gl[1,1];valign=:top,tellheight=false);colsize!(controls,1,Fixed(240))
    Label(controls[1,1],"Revise exclusions";font=:bold,halign=:left,fontsize=15)
    source=GridLayout(controls[2,1])
    Label(source[1,1],"Pair";fontsize=12)
    pair=Textbox(source[1,2];stored_string="1",width=50,height=25,fontsize=12)
    frame=Menu(source[1,3];options=[("frame A",:a),("frame B",:b)],width=112,fontsize=12)
    load=Button(controls[3,1];label="load raw reference",height=26,fontsize=12,tellwidth=false)
    enabled_row=GridLayout(controls[4,1]);enabled=Toggle(enabled_row[1,1];active=rc.mask_draft[].enabled)
    Label(enabled_row[1,2],"Mask enabled";fontsize=12,halign=:left,tellwidth=false)
    decoding=GridLayout(controls[5,1]);colgap!(decoding,5)
    Label(decoding[1,1],"Threshold";fontsize=11)
    threshold=Textbox(decoding[1,2];stored_string="0.5",width=55,height=25,fontsize=12)
    invert=Toggle(decoding[1,3];active=false,width=35,height=20,markersize=20)
    Label(decoding[1,4],"invert";fontsize=11,tellwidth=false)
    import_mask=Button(controls[6,1];label="import mask image...",height=26,fontsize=12,tellwidth=false)
    reset=Button(controls[7,1];label="reset to imported mask",height=26,fontsize=12,tellwidth=false)
    function button_pair(row,left,right)
        layout=GridLayout(controls[row,1]);colgap!(layout,8)
        a=Button(layout[1,1];label=left,width=116,height=26,fontsize=12,tellwidth=false)
        b=Button(layout[1,2];label=right,width=116,height=26,fontsize=12,tellwidth=false)
        colsize!(layout,1,Fixed(116));colsize!(layout,2,Fixed(116));a,b
    end
    close,hole=button_pair(8,"close polygon","draw hole")
    undo,delete=button_pair(9,"undo vertex","delete selected")
    grow,shrink=button_pair(10,"grow 1 px","shrink 1 px")
    clear=Button(controls[11,1];label="clear all editor pixels",height=26,fontsize=12,tellwidth=false)
    apply=Button(controls[12,1];label="Apply raster (enable mask)",height=28,fontsize=12,tellwidth=false)
    validate=Button(controls[13,1];label="validate / metadata diff",height=26,fontsize=12,tellwidth=false)
    save=Button(controls[14,1];label="save distinct revision...",height=26,fontsize=12,tellwidth=false)
    open=Button(controls[15,1];label="open saved revision...",height=26,fontsize=12,tellwidth=false)
    brief_status=lift(rc.status,mc.status,rc.error,mc.error) do revision,reference,revision_error,reference_error
        a=revision_error===nothing ? first(split(revision,r"[.;]";limit=2)) : "Action failed — see details"
        b=reference_error===nothing ? first(split(reference,r"[.;]";limit=2)) : "Reference needs attention — see details"
        a*".\n"*b*"."
    end
    Label(controls[16,1],brief_status;
        fontsize=11,halign=:left,justification=:left,tellwidth=false,word_wrap=true,height=60)
    Label(controls[17,1],"Red: excluded. Cyan: captured ROI.\nHoles also erase imported pixels.\nEdits take effect only on Apply.";
        fontsize=11,halign=:left,justification=:left,tellwidth=false)
    rowgap!(controls,2)
    content=GridLayout(gl[1,2]);section=Observable(:canvas)
    tabs=Menu(content[1,1:3];options=[("draw on raw reference",:canvas),("metadata changes",:changes),
        ("reference / draft details",:details),("all imported settings",:original),("last saved revision",:saved)],fontsize=12,tellwidth=false)
    text_panel=GridLayout(content[2,1:3]);previous=Button(text_panel[1,1];label="previous text page",height=26,fontsize=12,tellwidth=false)
    next=Button(text_panel[1,3];label="next text page",height=26,fontsize=12,tellwidth=false)
    page=Observable(1);room=Observable((60,15))
    original_text=experiment_summary(ExperimentController(rc.original))
    full=lift(rc.preview_text,rc.status,rc.error,rc.dirty,rc.saved_record,rc.saved_path,mc.bundle,mc.status,mc.error,rc.mask_draft,section) do diff,status,error,dirty,saved,path,bundle,reference_status,reference_error,draft,pane
        prefix="Status: $status\n"*(error===nothing ? "" : "Error: $(Controllers._errmsg(error))\n")
        pane===:original && return prefix*original_text
        pane===:saved && return prefix*(saved===nothing ? "No revision saved." :
            "Saved revision: $path\nRecipe: $(saved.recipe.recipe_id)\nInputs: $(saved.input_id)\nHistory: $(length(saved.runs)) runs\nOpen creates a separate workflow.")
        if pane===:details
            draft_text="Mask enabled: $(draft.enabled)\nRetained raster: $(draft.raster===nothing ? "nothing" : "$(size(draft.raster)), $(count(draft.raster)) excluded pixels")\n"
            return prefix*draft_text*"Reference status: $reference_status\n"*(reference_error===nothing ? "" : "Reference error: $(Controllers._errmsg(reference_error))\n")*
                (bundle===nothing ? "No reference loaded. Choose a pair and load it explicitly." :
                "Captured recipe: $(bundle.recipe_id)\nSource recipe: $(bundle.source_recipe_id)\nInputs: $(bundle.input_id)\nPair: $(bundle.pair_index), frame $(bundle.frame)\nInput: $(repr(bundle.input_descriptor))\nROI: $(repr(bundle.roi))\nRaw full-frame pixels; no preprocessing or scripts.\nReference identity stays fixed when current drafts change.")
        end
        prefix*(dirty ? "Current drafts differ from the last validated preview.\n" : "Draft matches validated preview.\n")*diff
    end
    chunks=lift((text,capacity)->_workflow_pages(text,capacity...),full,room)
    page_label=Label(text_panel[1,2],lift((n,c)->"text page $(clamp(n,1,length(c))) / $(length(c))",page,chunks);fontsize=12)
    body=Label(text_panel[2,1:3],lift((n,c)->c[clamp(n,1,length(c))],page,chunks);
        halign=:left,justification=:left,valign=:top,fontsize=12,tellwidth=false,tellheight=false,
        font=GLMakie.Makie.assetpath("fonts","DejaVuSansMono.ttf"))
    on(body.layoutobservables.suggestedbbox) do rect
        capacity=(max(1,floor(Int,(rect.widths[1]-8)/7.5)),max(1,floor(Int,(rect.widths[2]-8)/15.5)))
        room[]==capacity || (room[]=capacity)
    end
    canvas=GridLayout(content[2,1:3]);canvas_note=Label(canvas[1,1],"Choose a pair, then load its raw reference.";
        fontsize=12,halign=:left,justification=:left,tellwidth=false)
    ax=Axis(canvas[2,1];xlabel="original x (px)",ylabel="original y (px)",aspect=DataAspect(),yreversed=true)
    text_mode=Observable(false)
    text_setter=_workflow_panel!(text_panel,(previous,next,page_label,body),text_mode,true)
    canvas_setter=_workflow_panel!(canvas,(canvas_note,ax),section,:canvas)
    syncing=Ref(false);plots=Any[];reference_plots=Any[];listeners=Any[]
    busy()=rc.running[] || mc.running[]
    function report_error(err)
        rc.error[]=err;rc.state[]=:failed;rc.status[]="failed: $(Controllers._errmsg(err))"
    end
    function guarded(action)
        busy() && return
        try action() catch err;report_error(err);end
    end
    current_editor()=mc.bundle[]===nothing ? throw(ArgumentError("load a raw reference first")) : mc.bundle[].editor
    function edit(action;finished=false)
        guarded() do
            me=current_editor()
            finished && !isempty(me.active[]) && throw(ArgumentError("close or undo the unfinished polygon first"))
            action(me)
        end
    end
    function refresh_plots!()
        for plot in plots;delete!(ax,plot);end
        empty!(plots);bundle=mc.bundle[]
        bundle===nothing && return
        me=bundle.editor;nr,nc=size(me.image)
        overlay=heatmap!(ax,1:nc,1:nr,permutedims(ifelse.(polygon_mask(me),1f0,NaN32));
            colormap=[RGBAf(1,.1,.1,.35),RGBAf(1,.1,.1,.35)],colorrange=(0,1),nan_color=:transparent)
        translate!(overlay,0,0,1);push!(plots,overlay)
        for (i,(polygon,is_hole)) in enumerate(zip(me.polygons[],me.holes[]))
            outline=lines!(ax,[Point2f(v...) for v in [polygon;[polygon[1]]]];
                color=me.selected[]===i ? :yellow : is_hole ? :dodgerblue : :red,linewidth=2)
            translate!(outline,0,0,2);push!(plots,outline)
        end
        if !isempty(me.active[])
            points=[Point2f(v...) for v in me.active[]]
            line=lines!(ax,points;color=:orange,linewidth=2);vertices=scatter!(ax,points;color=:orange,markersize=7)
            translate!(line,0,0,3);translate!(vertices,0,0,3);append!(plots,[line,vertices])
        end
        canvas_note.text[]="Pair $(bundle.pair_index), frame $(uppercase(String(bundle.frame))) — raw reference\nLeft click: vertices / select. Right click: close. $(status_text(me))"
    end
    function bind_reference!()
        foreach(off,listeners);empty!(listeners)
        for plot in reference_plots;delete!(ax,plot);end
        empty!(reference_plots)
        if mc.bundle[]!==nothing
            bundle=mc.bundle[];me=bundle.editor;nr,nc=size(bundle.raw_image)
            # Full-frame image copies and the captured ROI belong to the
            # reference, not every drawing gesture. Only mutable mask plots
            # are recreated by the editor's observable listeners below.
            image=heatmap!(ax,1:nc,1:nr,permutedims(bundle.raw_image);colormap=:grays,
                colorrange=_preprocessing_image_range((bundle.raw_image,bundle.raw_image)))
            translate!(image,0,0,-2);push!(reference_plots,image)
            if bundle.roi!==nothing
                roi=bundle.roi;l=first(roi.cols)-.5;r=last(roi.cols)+.5;t=first(roi.rows)-.5;b=last(roi.rows)+.5
                outline=lines!(ax,Point2f[(l,t),(r,t),(r,b),(l,b),(l,t)];color=:cyan,linewidth=2)
                translate!(outline,0,0,4);push!(reference_plots,outline)
            end
            limits!(ax,.5,nc+.5,nr+.5,.5)
            for observable in (me.polygons,me.holes,me.raster,me.active,me.selected,me.hole_mode)
                push!(listeners,on(_->refresh_plots!(),observable))
            end
        else
            canvas_note.text[]="Choose a pair, then load its raw reference."
        end
        refresh_plots!()
    end
    on(_->bind_reference!(),mc.bundle)
    on(events(ax.scene).mousebutton) do event
        if section[]===:canvas && !busy() && event.action==Mouse.press && GLMakie.Makie.is_mouseinside(ax.scene)
            edit() do me
                if event.button==Mouse.left
                    position=GLMakie.Makie.mouseposition(ax.scene);click!(me,position[1],position[2])
                elseif event.button==Mouse.right;alt_click!(me);end
            end
        end
        Consume(false)
    end
    on(events(ax.scene).keyboardbutton) do event
        if section[]===:canvas && event.action==Keyboard.press && GLMakie.Makie.is_mouseinside(ax.scene)
            event.key==Keyboard.backspace && edit(undo_vertex!)
            event.key==Keyboard.delete && edit(delete_selected!)
        end
        Consume(false)
    end
    on(load.clicks) do _
        guarded() do
            index=parse(Int,pair.displayed_string[]);which=frame.selection[]
            load_recipe_mask_reference!(mc,rc;pair_index=index,frame=which)
            section[]=:canvas
        end
    end
    on(enabled.active) do value
        syncing[] && return
        if busy();syncing[]=true;enabled.active[]=rc.mask_draft[].enabled;syncing[]=false;return;end
        guarded(()->set_revision_mask!(rc;enabled=Bool(value)))
    end
    on(rc.mask_draft) do draft
        syncing[]=true;enabled.active[]=draft.enabled;syncing[]=false
    end
    on(import_mask.clicks) do _
        guarded() do
            level=parse(Float64,threshold.displayed_string[])
            load_revision_mask!(rc,mask_path_picker;threshold=level,invert=Bool(invert.active[]))
        end
    end
    on(_->guarded(()->reset_revision_mask!(rc)),reset.clicks)
    on(_->edit(close_active!),close.clicks);on(_->edit(begin_hole!),hole.clicks)
    on(_->edit(undo_vertex!),undo.clicks);on(_->edit(delete_selected!),delete.clicks)
    on(_->edit(grow_mask!;finished=true),grow.clicks);on(_->edit(shrink_mask!;finished=true),shrink.clicks)
    on(_->edit(clear_polygons!),clear.clicks)
    on(_->guarded(()->apply_revision_mask!(rc,mc)),apply.clicks)
    on(_->guarded(()->apply_recipe_revision!(rc)),validate.clicks)
    on(_->guarded(()->save_recipe_revision!(rc,save_path_picker)),save.clicks)
    on(open.clicks) do _
        guarded() do
            rc.saved_record[]===nothing && throw(ArgumentError("save a revision first"))
            record=deepcopy(rc.saved_record[])
            @async begin;yield();try open_revision(record) catch err;report_error(err);end;end
        end
    end
    _sync_menu!(tabs,section)
    on(section) do pane;text_mode[]=pane!==:canvas;text_setter();canvas_setter();end
    on(_->(page[]=1),full)
    on(_->(page[]=max(1,page[]-1)),previous.clicks);on(_->(page[]=min(length(chunks[]),page[]+1)),next.clicks)
    rowgap!(content,5);bind_reference!();text_setter();canvas_setter()
    gl
end
