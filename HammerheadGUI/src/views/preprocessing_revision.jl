const _PREPROCESSING_VIEW_OPERATIONS=[:subtract_background,:intensity_cap,:highpass_filter,
    :clahe,:percentile_stretch,:invert_image,:local_variance_normalize]

function _preprocessing_image_range(images)
    lower=min(minimum(images[1]),minimum(images[2]));upper=max(maximum(images[1]),maximum(images[2]))
    if lower==upper
        gap=max(1.,abs(Float64(lower))*eps(Float64))
        # At either finite extreme, pad only toward the representable interior.
        lower=isfinite(lower-gap) ? lower-gap : Float64(lower)
        upper=isfinite(upper+gap) ? upper+gap : Float64(upper)
    end
    (lower,upper)
end

"""
    preprocessing_revision(controller; preview_controller=RecipeImagePreviewController(),
        size=(1100,800), save_path_picker, background_path_picker, open_revision)

Edit complete ordered built-in preprocessing of a saved planar recipe. Duplicate
operations and embedded background precision are retained. Every other recipe
setting remains inspectable and unchanged. Explicit metadata validation, verified
pair-image preview and saving are separate actions; editing never loads pixels.
Pixel previews show full original/processed frames with read-only original-image
ROI/mask overlays. Their captured identity remains distinct from current drafts.
Referenced scripts are retained but numerical preview refuses to execute them.
Queued decoding/hashing/I/O can pause rendering; no cancellation or responsiveness
guarantee is provided. Saving/opening uses a separate guarded revision workflow.
"""
function preprocessing_revision(rc::RecipeRevisionController;
        preview_controller=RecipeImagePreviewController(),size=(1100,800),
        save_path_picker::Function=()->save_file(;filterlist="jld2"),
        background_path_picker::Function=()->pick_file(),
        open_revision::Function=record->display(GLMakie.Screen(),experiment_workflow(ExperimentController(record))))
    fig=Figure(;size)
    preprocessing_revision!(fig[1,1],rc;preview_controller,save_path_picker,background_path_picker,open_revision)
    fig
end

"""
    preprocessing_revision!(target, controller; preview_controller,
        save_path_picker, background_path_picker, open_revision) -> GridLayout

Embed the ordered preprocessing editor. All raw option drafts are parsed by the
controller, including invisible rows. Typed background replacement is explicit.
Picker requests and numerical preview snapshots are captured before callbacks
or task yields. Last valid metadata, saved records and pixel bundles keep their
own identities after invalid edits or failed/cancelled requests.
"""
function preprocessing_revision!(target,rc::RecipeRevisionController;
        preview_controller=RecipeImagePreviewController(),
        save_path_picker::Function=()->save_file(;filterlist="jld2"),
        background_path_picker::Function=()->pick_file(),
        open_revision::Function=record->display(GLMakie.Screen(),experiment_workflow(ExperimentController(record))))
    pc=preview_controller;gl=GridLayout(target);colsize!(gl,1,Fixed(240));colgap!(gl,18)
    controls=GridLayout(gl[1,1];valign=:top,tellheight=false);colsize!(controls,1,Fixed(240))
    Label(controls[1,1],"preprocessing revision";font=:bold,halign=:left)
    step_menu=Menu(controls[2,1];options=[("empty chain",0)],width=240,tellwidth=false)
    step_list=Label(controls[3,1],"";halign=:left,justification=:left,fontsize=12,tellwidth=false)
    order=GridLayout(controls[4,1]);colgap!(order,10)
    up=Button(order[1,1];label="move up",width=115,height=26,fontsize=12,tellwidth=false)
    down=Button(order[1,2];label="move down",width=115,height=26,fontsize=12,tellwidth=false)
    colsize!(order,1,Fixed(115));colsize!(order,2,Fixed(115))
    edit=GridLayout(controls[5,1]);colgap!(edit,10)
    duplicate=Button(edit[1,1];label="duplicate step",width=115,height=26,fontsize=12,tellwidth=false)
    remove=Button(edit[1,2];label="delete step",width=115,height=26,fontsize=12,tellwidth=false)
    colsize!(edit,1,Fixed(115));colsize!(edit,2,Fixed(115))
    catalogue=GridLayout(controls[6,1]);colgap!(catalogue,8)
    operation_menu=Menu(catalogue[1,1];options=[(String(op),op) for op in _PREPROCESSING_VIEW_OPERATIONS],
        width=162,fontsize=11,tellwidth=false)
    add=Button(catalogue[1,2];label="add",width=70,height=28,fontsize=12,tellwidth=false)
    colsize!(catalogue,1,Fixed(162));colsize!(catalogue,2,Fixed(70))
    background=Button(controls[7,1];label="load background as $(rc.original.recipe.image_type)...",
        width=240,height=28,fontsize=12,tellwidth=false)
    validate=Button(controls[8,1];label="validate / metadata diff",width=240,height=28,fontsize=12,tellwidth=false)
    save=Button(controls[9,1];label="save distinct revision...",width=240,height=28,fontsize=12,tellwidth=false)
    open=Button(controls[10,1];label="open saved revision...",width=240,height=28,fontsize=12,tellwidth=false)
    Label(controls[11,1],lift(d->d ? "Current raw draft changed.\nMetadata is last validated." : "Draft matches metadata preview.",rc.dirty);
        halign=:left,justification=:left,fontsize=12,tellwidth=false)
    Label(controls[12,1],lift(s->_experiment_wrap_text(s;columns=29,max_lines=2),rc.status);
        halign=:left,justification=:left,fontsize=12,tellwidth=false)
    Label(controls[13,1],"Other recipe settings remain intact.\nExplicit preview/save may pause UI.";
        halign=:left,justification=:left,fontsize=11,tellwidth=false)
    # Configure gaps after the rows exist: newly inserted rows otherwise retain
    # GridLayout's defaults, exceeding the compact window's sidebar height.
    rowgap!(controls,2)

    content=GridLayout(gl[1,2]);rowgap!(content,5)
    selected_label=Label(content[1,1:3],"";halign=:left,fontsize=14,font=:bold,tellwidth=false)
    editors=GridLayout(content[2,1:3]);rowgap!(editors,4)
    labels=Label[];boxes=Textbox[];rowlayouts=GridLayout[];row_active=Observable{Bool}[];setters=Function[]
    for index in 1:3
        row=GridLayout(editors[index,1]);colsize!(row,1,Fixed(120));colgap!(row,8)
        label=Label(row[1,1],"";halign=:left,fontsize=12,tellwidth=false)
        box=Textbox(row[1,2];stored_string="",height=28,width=300,fontsize=12,tellwidth=false)
        on(row.layoutobservables.suggestedbbox) do rect
            w=max(1.,rect.widths[1]-128.);box.width[]==w || (box.width[]=w)
        end
        active=Observable(true);push!(row_active,active)
        push!(labels,label);push!(boxes,box);push!(rowlayouts,row)
        push!(setters,_workflow_panel!(row,(label,box),active,true))
    end
    background_text=Label(content[3,1:3],"";halign=:left,justification=:left,fontsize=11,tellwidth=false)
    preview_tools=GridLayout(content[4,1:3]);colgap!(preview_tools,8)
    pair_label=Label(preview_tools[1,1],"ordered pair";fontsize=12)
    pair_box=Textbox(preview_tools[1,2];stored_string="1",width=54,height=28,fontsize=12)
    frame_menu=Menu(preview_tools[1,3];options=[("frame A",:a),("frame B",:b)],width=98,fontsize=12)
    pixels=Button(preview_tools[1,4];label="preview verified pair",height=28,fontsize=12,tellwidth=false)
    pixel_status=Label(preview_tools[2,1:4],lift(s->_experiment_wrap_text(s;columns=76,max_lines=1),pc.status);
        halign=:left,justification=:left,fontsize=11,tellwidth=false)
    panes=Observable(:changes)
    tabs=Menu(content[5,1:3];options=[("metadata changes",:changes),("all imported settings",:original),
        ("last saved revision",:saved),("captured original / processed images",:pixels),
        ("pixel preview identity / status",:pixel_details)],tellwidth=false,fontsize=12)
    text_panel=GridLayout(content[6,1:3]);rowgap!(text_panel,5)
    previous=Button(text_panel[1,1];label="previous text page",height=26,fontsize=12,tellwidth=false)
    next=Button(text_panel[1,3];label="next text page",height=26,fontsize=12,tellwidth=false)
    page=Observable(1);room=Observable((60,10))
    original_text=experiment_summary(ExperimentController(rc.original))
    full=lift(rc.preview_text,rc.status,rc.dirty,rc.saved_record,rc.saved_path,pc.bundle,pc.status,panes) do diff,status,dirty,saved,path,bundle,pixel_status,pane
        prefix="Status: $status\n"
        pane===:original && return prefix*"Only ordered preprocessing is edited here.\n"*original_text
        pane===:saved && return prefix*(saved===nothing ? "No revision saved." :
            "Saved revision: $path\nRecipe: $(saved.recipe.recipe_id)\nInputs: $(saved.input_id)\nHistory: $(length(saved.runs)) runs")
        pane===:pixel_details && return "Pixel status: $pixel_status\n"*(bundle===nothing ? "No pixel preview." :
            "Captured recipe: $(bundle.recipe_id)\nCaptured inputs: $(bundle.input_id)\nOrdered pair: $(bundle.pair_index)\nPaths: $(join(bundle.input_paths,"\n"))\nPrecision: $(bundle.image_type)\nFull-frame preprocessing before ROI; overlays are read-only.\nCurrent raw drafts may differ. This is not a PIV accuracy or validity measurement.")
        prefix*(dirty ? "Raw drafts differ from last metadata preview.\n" : "Draft matches metadata preview.\n")*diff
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
    image_panel=GridLayout(content[6,1:3]);colgap!(image_panel,12)
    image_identity=Label(image_panel[1,1:2],"No pixel preview yet.";halign=:left,justification=:left,fontsize=11,tellwidth=false)
    raw_axis=Axis(image_panel[2,1];title="original",aspect=DataAspect(),yreversed=true,xlabel="original x (px)",ylabel="original y (px)")
    processed_axis=Axis(image_panel[2,2];title="processed",aspect=DataAspect(),yreversed=true,xlabel="original x (px)",ylabel="original y (px)")
    # Text is shared by all non-image tabs. Use its own boolean visibility gate.
    text_mode=Observable(true)
    text_setter=_workflow_panel!(text_panel,(previous,next,page_label,body),text_mode,true)
    image_setter=_workflow_panel!(image_panel,(image_identity,raw_axis,processed_axis),panes,:pixels)
    plots=Any[]
    function refresh_images!()
        for (axis,plot) in plots;delete!(axis,plot);end
        empty!(plots);bundle=pc.bundle[]
        if bundle===nothing
            image_identity.text[]="No pixel preview yet; choose an ordered pair and preview explicitly."
            return
        end
        frame=frame_menu.selection[]
        images=frame===:a ? (bundle.raw_a,bundle.processed_a) : (bundle.raw_b,bundle.processed_b)
        shared_range=_preprocessing_image_range(images);lower,upper=shared_range
        image_identity.text[]="Captured pair $(bundle.pair_index), $(bundle.image_type), recipe $(first(bundle.recipe_id,16))...\nShared intensity range: $(round(lower;sigdigits=5)) to $(round(upper;sigdigits=5)); cyan ROI / red excluded mask."
        for (axis,image) in zip((raw_axis,processed_axis),images)
            nr,nc=Base.size(image)
            push!(plots,(axis,heatmap!(axis,1:nc,1:nr,permutedims(image);colormap=:grays,colorrange=shared_range)))
            if bundle.mask!==nothing
                overlay=heatmap!(axis,1:nc,1:nr,permutedims(ifelse.(bundle.mask,1f0,NaN32));
                    colormap=[RGBAf(1,.1,.1,.3),RGBAf(1,.1,.1,.3)],colorrange=(0,1),nan_color=:transparent)
                translate!(overlay,0,0,1);push!(plots,(axis,overlay))
            end
            if bundle.roi!==nothing
                rr=bundle.roi;left=first(rr.cols)-.5;right=last(rr.cols)+.5;top=first(rr.rows)-.5;bottom=last(rr.rows)+.5
                roi_plot=lines!(axis,Point2f[(left,top),(right,top),(right,bottom),(left,bottom),(left,top)];color=:cyan,linewidth=2)
                translate!(roi_plot,0,0,2);push!(plots,(axis,roi_plot))
            end
            limits!(axis,.5,nc+.5,nr+.5,.5)
        end
    end
    syncing=Ref(false);shown=Ref(rc.preprocessing_selected[])
    function report_error(err)
        rc.error[]=err;rc.status[]="failed: $(Controllers._errmsg(err))"
    end
    function guarded(action)
        rc.running[] && return
        try action() catch err;report_error(err);end
    end
    function refresh!()
        syncing[]=true
        try
            drafts=rc.preprocessing_drafts[];selected=isempty(drafts) ? 0 : clamp(rc.preprocessing_selected[],1,length(drafts))
            options=isempty(drafts) ? [("empty chain",0)] : [("$i: $(row.operation)",i) for (i,row) in enumerate(drafts)]
            step_menu.options[]==options || (step_menu.options[]=options)
            step_menu.i_selected[]=max(1,selected);shown[]=selected
            lines=isempty(drafts) ? ["Empty chain: identity preprocessing."] :
                ["$(i==selected ? ">" : " ") $i: $(drafts[i].operation)" for i in max(1,selected-1):min(length(drafts),selected+1)]
            step_list.text[]=join([length(line)>29 ? first(line,26)*"..." : line for line in lines],"\n")
            fields=selected==0 ? NamedTuple[] : preprocessing_fields(drafts[selected].operation)
            selected_label.text[]=selected==0 ? "No preprocessing steps" : "Step $selected: $(drafts[selected].operation)"
            for i in 1:3
                row_active[i][]=i<=length(fields);setters[i]();rowsize!(editors,i,i<=length(fields) ? Fixed(28) : Fixed(0))
                if i<=length(fields)
                    field=fields[i];labels[i].text[]=field.label;value=drafts[selected].options[field.key]
                    boxes[i].stored_string[]=value;boxes[i].displayed_string[]=value
                end
            end
            bg=selected==0 ? nothing : drafts[selected].background
            background_text.text[]=bg===nothing ? "No array payload for this step. Options are raw text; no code is evaluated." :
                "Embedded background: $(Base.size(bg)), $(eltype(bg)); retained exactly until explicit replacement."
        finally
            syncing[]=false
        end
    end
    on(step_menu.selection) do index
        syncing[] && return
        rc.running[] && return refresh!()
        guarded() do;rc.preprocessing_selected[]=Int(index);end
    end
    for (i,box) in enumerate(boxes)
        on(box.displayed_string) do text
            syncing[] && return
            rc.running[] && return refresh!()
            shown[]>0 || return
            fields=preprocessing_fields(rc.preprocessing_drafts[][shown[]].operation)
            i<=length(fields) || return
            guarded() do;set_revision_preprocess!(rc,shown[],Dict(fields[i].key=>String(text)));end
        end
    end
    on(_->(syncing[] || refresh!()),rc.preprocessing_drafts)
    on(_->(syncing[] || refresh!()),rc.preprocessing_selected)
    for (button,action) in ((up,()->move_revision_preprocess!(rc,shown[],max(1,shown[]-1))),
            (down,()->move_revision_preprocess!(rc,shown[],min(length(rc.preprocessing_drafts[]),shown[]+1))),
            (duplicate,()->insert_revision_preprocess!(rc,shown[]+1;source=shown[])),
            (remove,()->delete_revision_preprocess!(rc,shown[])))
        on(button.clicks) do _
            guarded() do;shown[]>0 || throw(ArgumentError("select a preprocessing step"));action();end
        end
    end
    on(add.clicks) do _
        guarded() do;insert_revision_preprocess!(rc,shown[]+1,operation_menu.selection[]);end
    end
    on(background.clicks) do _
        guarded() do
            shown[]>0 && rc.preprocessing_drafts[][shown[]].operation===:subtract_background || throw(ArgumentError("select a background-subtraction step"))
            load_revision_background!(rc,shown[],background_path_picker)
        end
    end
    on(_->guarded(()->apply_recipe_revision!(rc)),validate.clicks)
    on(_->guarded(()->save_recipe_revision!(rc,save_path_picker)),save.clicks)
    on(open.clicks) do _
        guarded() do
            rc.saved_record[]===nothing && throw(ArgumentError("save a revision first"))
            record=deepcopy(rc.saved_record[])
            @async begin;yield();try open_revision(record) catch err;report_error(err);end;end
        end
    end
    on(pixels.clicks) do _
        (rc.running[] || pc.running[]) && return
        try
            index=tryparse(Int,strip(pair_box.displayed_string[]))
            index===nothing && throw(ArgumentError("ordered pair must be a positive integer"))
            preview_recipe_images!(pc,rc;pair_index=index)
        catch err
            pc.error[]=err;pc.state[]=:failed
            pc.status[]="Image preview request failed: $(Controllers._errmsg(err)); retained images keep their captured identity."
        end
    end
    _sync_menu!(tabs,panes)
    on(panes) do pane
        text_mode[]=pane!==:pixels;text_setter();image_setter()
    end
    on(_->refresh_images!(),pc.bundle);on(_->refresh_images!(),frame_menu.selection)
    on(_->(page[]=1),full)
    on(_->(page[]=max(1,page[]-1)),previous.clicks)
    on(_->(page[]=min(length(chunks[]),page[]+1)),next.clicks)
    refresh!();refresh_images!();text_setter();image_setter()
    gl
end
