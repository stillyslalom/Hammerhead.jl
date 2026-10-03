"""
    recipe_geometry_revision(controller; size=(1100,800), save_path_picker, open_revision)

Edit original-image ROI bounds and isotropic `PhysicalScale` metadata of a saved
planar recipe. Raw invalid text is retained. Enabling or disabling either setting
preserves its text; explicit reset disables it, composing `nothing`. Numeric
scale factors must match the supplied unit labels; no calibration is inferred.
Other current pass/preprocessing drafts are preserved. Original imported recipe
settings remain inspectable in their own read-only tab.

Validation and saving are separate queued controller actions. This form performs
no pixel or PIV preview. File verification/I/O may pause rendering. A saved
revision opens in a separate workflow without replacing source history.
"""
function recipe_geometry_revision(rc::RecipeRevisionController;size=(1100,800),
        save_path_picker::Function=()->save_file(;filterlist="jld2"),
        open_revision::Function=record->display(GLMakie.Screen(),experiment_workflow(ExperimentController(record))))
    fig=Figure(;size)
    recipe_geometry_revision!(fig[1,1],rc;save_path_picker,open_revision)
    fig
end

"""
    recipe_geometry_revision!(target, controller; save_path_picker, open_revision)

Embed the ROI/scale revision form. Every field updates raw controller drafts,
including invalid or disabled text. Apply/save capture the complete recipe before
observers or pickers. Last validated and saved identities remain separate from
current drafts. ROI coordinates are inclusive rows/columns of the original image;
pixel size is one isotropic length per pixel and `dt` is time per image pair.
"""
function recipe_geometry_revision!(target,rc::RecipeRevisionController;
        save_path_picker::Function=()->save_file(;filterlist="jld2"),
        open_revision::Function=record->display(GLMakie.Screen(),experiment_workflow(ExperimentController(record))))
    gl=GridLayout(target);colsize!(gl,1,Fixed(230));colgap!(gl,18)
    controls=GridLayout(gl[1,1];valign=:top,tellheight=false);colsize!(controls,1,Fixed(230))
    Label(controls[1,1],"ROI / scale revision";font=:bold,halign=:left,fontsize=15)
    validate=Button(controls[2,1];label="validate / metadata diff",height=30,width=230,fontsize=13,tellwidth=false)
    save=Button(controls[3,1];label="save distinct revision...",height=30,width=230,fontsize=13,tellwidth=false)
    open=Button(controls[4,1];label="open saved revision...",height=30,width=230,fontsize=13,tellwidth=false)
    Label(controls[5,1],lift(d->d ? "Raw drafts changed.\nPreview is last validated." : "Draft matches validated preview.",rc.dirty);
        halign=:left,justification=:left,fontsize=12,tellwidth=false)
    Label(controls[6,1],lift(s->_experiment_wrap_text(s;columns=28,max_lines=3),rc.status);
        halign=:left,justification=:left,fontsize=12,tellwidth=false)
    Label(controls[7,1],"ROI uses original-image pixels.\nBounds include both endpoints.\nDisabled values remain as drafts.\nReset means no ROI / no scale.";
        halign=:left,justification=:left,fontsize=11,tellwidth=false)
    Label(controls[8,1],"Scale is isotropic. Numeric factors\nmust match your chosen units.\nLabels do not convert factors.\nNo calibration line is inferred.";
        halign=:left,justification=:left,fontsize=11,tellwidth=false)
    Label(controls[9,1],"Other settings and inputs retained.\nNo automatic images or PIV.\nSave verifies inputs; I/O may pause UI.\nSource history stays separate.";
        halign=:left,justification=:left,fontsize=11,tellwidth=false)
    rowgap!(controls,7)

    content=GridLayout(gl[1,2]);form=GridLayout(content[1,1:3]);colgap!(form,18)
    syncing=Ref(false);boxes=Vector{Textbox}[];toggles=Toggle[]
    descriptors=((draft=rc.roi_draft,fields=revision_roi_fields(),setter=set_revision_roi!,title="Original-image ROI",enabled="ROI enabled",reset="reset ROI to nothing"),
        (draft=rc.scale_draft,fields=revision_scale_fields(),setter=set_revision_scale!,title="Physical scale",enabled="Physical scale enabled",reset="reset scale to nothing"))
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
            for (n,desc) in enumerate(descriptors)
                toggles[n].active[]=desc.draft[].enabled
                for (box,field) in zip(boxes[n],desc.fields)
                    value=desc.draft[].values[field.key]
                    if box.displayed_string[]!=value
                        box.stored_string[]=value;box.displayed_string[]=value
                    end
                end
            end
        finally
            syncing[]=false
        end
    end
    for (n,desc) in enumerate(descriptors)
        panel=GridLayout(form[1,n])
        Label(panel[1,1],desc.title;halign=:left,font=:bold,fontsize=14,tellwidth=false)
        enabled=GridLayout(panel[2,1]);colgap!(enabled,8)
        toggle=Toggle(enabled[1,1];active=desc.draft[].enabled,halign=:left)
        Label(enabled[1,2],desc.enabled;halign=:left,fontsize=12,tellwidth=false)
        push!(toggles,toggle);editors=Textbox[];push!(boxes,editors)
        for (i,field) in enumerate(desc.fields)
            row=GridLayout(panel[i+2,1])
            Label(row[1,1],field.label;halign=:left,fontsize=12,tellwidth=false)
            box=Textbox(row[2,1];stored_string=desc.draft[].values[field.key],height=28,width=270,fontsize=12,tellwidth=false)
            on(row.layoutobservables.suggestedbbox) do rect
                width=max(1.,rect.widths[1]);box.width[]==width || (box.width[]=width)
            end
            rowgap!(row,2)
            push!(editors,box)
            on(box.displayed_string) do text
                syncing[] && return
                rc.running[] && return refresh!()
                guarded() do;desc.setter(rc,Dict(field.key=>String(text)));end
            end
        end
        reset=Button(panel[7,1];label=desc.reset,height=28,fontsize=12,tellwidth=false)
        on(toggle.active) do enabled
            syncing[] && return
            rc.running[] && return refresh!()
            guarded() do;desc.setter(rc,Dict{Symbol,String}();enabled=Bool(enabled));end
        end
        on(reset.clicks) do _
            guarded() do;desc.setter(rc,Dict{Symbol,String}();enabled=false);end
        end
        on(_->(syncing[] || refresh!()),desc.draft)
        rowgap!(panel,4)
    end

    section=Observable(:changes)
    tabs=Menu(content[2,1:3];options=[("metadata changes",:changes),("all imported settings (read-only)",:original),
        ("last saved revision",:saved)],tellwidth=false,fontsize=12)
    previous=Button(content[3,1];label="previous text page",height=26,fontsize=12,tellwidth=false)
    next=Button(content[3,3];label="next text page",height=26,fontsize=12,tellwidth=false)
    page=Observable(1);room=Observable((65,10))
    original_text=experiment_summary(ExperimentController(rc.original))
    full=lift(rc.preview_text,rc.status,rc.error,rc.dirty,rc.saved_record,rc.saved_path,section) do diff,status,error,dirty,saved,path,pane
        prefix="Status: $status\n"*(error===nothing ? "" : "Error: $(Controllers._errmsg(error))\n")
        pane===:original && return prefix*"ROI / scale drafts are editable; imported settings below are read-only.\n"*original_text
        pane===:saved && return prefix*(saved===nothing ? "No revision saved." :
            "Saved revision: $path\nRecipe: $(saved.recipe.recipe_id)\nInputs: $(saved.input_id)\nHistory: $(length(saved.runs)) runs\nOpen creates a separate saved workflow.")
        prefix*(dirty ? "Current drafts differ from last validated metadata.\n" : "Draft matches metadata preview.\n")*diff
    end
    chunks=lift((text,capacity)->_workflow_pages(text,capacity...),full,room)
    Label(content[3,2],lift((n,c)->"text page $(clamp(n,1,length(c))) / $(length(c))",page,chunks);fontsize=12)
    body=Label(content[4,1:3],lift((n,c)->c[clamp(n,1,length(c))],page,chunks);
        halign=:left,justification=:left,valign=:top,fontsize=12,tellwidth=false,tellheight=false,
        font=GLMakie.Makie.assetpath("fonts","DejaVuSansMono.ttf"))
    on(body.layoutobservables.suggestedbbox) do rect
        capacity=(max(1,floor(Int,(rect.widths[1]-8)/7.5)),max(1,floor(Int,(rect.widths[2]-8)/15.5)))
        room[]==capacity || (room[]=capacity)
    end
    on(validate.clicks) do _;guarded(()->apply_recipe_revision!(rc));end
    on(save.clicks) do _;guarded(()->save_recipe_revision!(rc,save_path_picker));end
    on(open.clicks) do _
        guarded() do
            rc.saved_record[]===nothing && throw(ArgumentError("save a revision first"))
            record=deepcopy(rc.saved_record[])
            @async begin
                yield()
                try open_revision(record) catch err;report_error(err);end
            end
        end
    end
    _sync_menu!(tabs,section)
    on(_->(page[]=1),full)
    on(_->(page[]=max(1,page[]-1)),previous.clicks)
    on(_->(page[]=min(length(chunks[]),page[]+1)),next.clicks)
    rowgap!(content,6);colgap!(gl,18);refresh!()
    gl
end
