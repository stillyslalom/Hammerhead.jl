function _revision_field_pages(schema)
    pages=NamedTuple[]
    for group in unique(field.group for field in schema)
        fields=filter(field->field.group===group,schema)
        for first in 1:6:length(fields)
            index=length(pages)+1
            title=uppercasefirst(replace(String(group),'_'=>' '))
            length(fields)>6 && (title*=" $(cld(first,6))")
            push!(pages,(id=index,label=title,fields=fields[first:min(first+5,end)]))
        end
    end
    pages
end

"""
    recipe_revision(controller; size=(1100,800), save_path_picker, open_revision) -> Figure

Edit the ordered passes of a saved planar recipe. Every other imported setting
and each pass's validation tuple remain intact and inspectable. Drafts retain
invalid text; preview and save parse the complete current schedule instead of
reusing previously valid values. Saving writes a distinct, protected experiment
record with fresh history. Opening it creates a separate saved-experiment lane.

Preview/save are queued on the GUI owner task after callbacks return. Input
verification and file I/O may still pause rendering; no background responsiveness
or cancellation is promised. Referenced scripts are retained, never executed.
"""
function recipe_revision(rc::RecipeRevisionController;size=(1100,800),
                         save_path_picker::Function=()->save_file(;filterlist="jld2"),
                         open_revision::Function=record->display(GLMakie.Screen(),experiment_workflow(ExperimentController(record))))
    fig=Figure(;size)
    recipe_revision!(fig[1,1],rc;save_path_picker,open_revision)
    fig
end

"""
    recipe_revision!(target, controller; save_path_picker, open_revision) -> GridLayout

Embed the complete pass-revision editor. Text currently visible in the editor is
captured before selecting another pass/field group, previewing or saving. The
last valid preview and saved revision retain their own identities after failures.
"""
function recipe_revision!(target,rc::RecipeRevisionController;
                          save_path_picker::Function=()->save_file(;filterlist="jld2"),
                          open_revision::Function=record->display(GLMakie.Screen(),experiment_workflow(ExperimentController(record))))
    gl=GridLayout(target)
    colsize!(gl,1,Fixed(240));colgap!(gl,20)
    controls=GridLayout(gl[1,1];valign=:top,tellheight=false)
    colsize!(controls,1,Fixed(240))
    rowgap!(controls,5)
    Label(controls[1,1],"saved recipe revision";font=:bold,halign=:left)
    Label(controls[2,1],"Ordered passes";halign=:left,fontsize=13)
    pass_menu=Menu(controls[3,1];options=[("pass $i",i) for i in eachindex(rc.drafts[])],width=240,tellwidth=false)
    pass_list=Label(controls[4,1],"";halign=:left,justification=:left,fontsize=12,tellwidth=false)
    ordering=GridLayout(controls[5,1]);colgap!(ordering,10)
    up=Button(ordering[1,1];label="move up",width=115,height=28,fontsize=13,tellwidth=false)
    down=Button(ordering[1,2];label="move down",width=115,height=28,fontsize=13,tellwidth=false)
    colsize!(ordering,1,Fixed(115));colsize!(ordering,2,Fixed(115))
    rows=GridLayout(controls[6,1]);colgap!(rows,10)
    duplicate=Button(rows[1,1];label="duplicate pass",width=115,height=28,fontsize=13,tellwidth=false)
    remove=Button(rows[1,2];label="delete pass",width=115,height=28,fontsize=13,tellwidth=false)
    colsize!(rows,1,Fixed(115));colsize!(rows,2,Fixed(115))
    preview=Button(controls[7,1];label="validate / preview changes",width=240,height=30,fontsize=13,tellwidth=false)
    save=Button(controls[8,1];label="save distinct revision...",width=240,height=30,fontsize=13,tellwidth=false)
    open=Button(controls[9,1];label="open saved revision...",width=240,height=30,fontsize=13,tellwidth=false)
    Label(controls[10,1],lift(rc.dirty) do dirty
        dirty ? "Draft changed; preview is last valid." : "Draft matches validated preview."
    end;halign=:left,justification=:left,fontsize=12,width=240,tellwidth=false,word_wrap=true)
    Label(controls[11,1],lift(text->_experiment_wrap_text(text;columns=29,max_lines=3),rc.status);
        halign=:left,justification=:left,fontsize=12,width=240,tellwidth=false)
    Label(controls[12,1],"Revision preserves inputs, preprocessing,\nROI, mask, scale and execution settings.\nSave verifies inputs; I/O may pause UI.\nThe source record/history stays separate.";
        halign=:left,justification=:left,fontsize=11,tellwidth=false)

    content=GridLayout(gl[1,2];valign=:top,tellheight=false)
    rowgap!(content,6)
    fields=_revision_field_pages(revision_fields())
    field_page=Observable(1)
    groups=Menu(content[1,1:3];options=[(p.label,p.id) for p in fields],tellwidth=false)
    editors=GridLayout(content[2,1:3]);rowgap!(editors,4)
    labels=Label[];boxes=Textbox[];visible=Observable{Bool}[];rowlayouts=GridLayout[]
    setters=Function[]
    for i in 1:6
        row=GridLayout(editors[i,1]);push!(rowlayouts,row)
        colsize!(row,1,Fixed(175));colgap!(row,10)
        label=Label(row[1,1],"";halign=:left,fontsize=13,tellwidth=false)
        box=Textbox(row[1,2];stored_string="",width=300,height=30,fontsize=13,tellwidth=false)
        on(row.layoutobservables.suggestedbbox) do rect
            width=max(1.,rect.widths[1]-185.)
            box.width[]==width || (box.width[]=width)
        end
        push!(labels,label);push!(boxes,box)
        active=Observable(true);push!(visible,active)
        push!(setters,_workflow_panel!(row,(label,box),active,true))
    end
    hint=Label(content[3,1:3],"Tuple entries are (row, column); booleans are true or false.\nValidation tuple is preserved read-only in imported settings.";
        halign=:left,justification=:left,fontsize=11,tellwidth=false)
    sections=Observable(:changes)
    tabs=Menu(content[4,1:3];options=[("metadata changes / preview",:changes),
        ("all imported settings (read-only)",:original),("last saved revision",:saved)],tellwidth=false)
    previous=Button(content[5,1];label="previous text page",height=28,fontsize=13,tellwidth=false)
    next=Button(content[5,3];label="next text page",height=28,fontsize=13,tellwidth=false)
    page=Observable(1);capacity=Observable((64,8))
    original_text=experiment_summary(ExperimentController(rc.original))*"\n\nOrdered input files (unchanged):\n"*
        join(["pair $i: $(rc.original.input_files[pair[1]]["path"]) -> $(rc.original.input_files[pair[2]]["path"])" for (i,pair) in enumerate(rc.original.pairs)],"\n")
    full=lift(rc.preview_text,rc.dirty,rc.saved_record,rc.saved_path,rc.status,sections,rc.templates,rc.selected) do preview_text,dirty,saved,path,status,section,templates,selected
        prefix="Status: $status\n"
        if section===:original
            return prefix*"Every imported setting is retained; only pass drafts are editable.\nSelected pass validation (read-only): $(templates[clamp(selected,1,length(templates))].validation)\n\n"*original_text
        elseif section===:saved
            return prefix*(saved===nothing ? "No revision has been saved." :
                "Saved revision: $path\nRecipe: $(saved.recipe.recipe_id)\nOrdered input: $(saved.input_id)\nHistory: $(length(saved.runs)) runs\nOpen creates a separate replay/comparison workflow.")
        end
        prefix*(dirty ? "Current draft differs from the last validated preview.\n" : "Current draft has a validated preview.\n")*preview_text
    end
    chunks=lift((text,room)->_workflow_pages(text,room...),full,capacity)
    Label(content[5,2],lift((index,pages)->"text page $(clamp(index,1,length(pages))) / $(length(pages))",page,chunks);fontsize=12)
    body=Label(content[6,1:3],lift((index,pages)->pages[clamp(index,1,length(pages))],page,chunks);
        halign=:left,justification=:left,valign=:top,font=GLMakie.Makie.assetpath("fonts","DejaVuSansMono.ttf"),
        fontsize=12,tellwidth=false,tellheight=false)
    function reflow!(rect)
        room=(max(1,floor(Int,(rect.widths[1]-8)/7.5)),max(1,floor(Int,(rect.widths[2]-8)/15.5)))
        capacity[]==room || (capacity[]=room)
    end
    on(reflow!,body.layoutobservables.suggestedbbox)
    syncing=Ref(false)
    shown_pass=Ref(rc.selected[]);shown_page=Ref(field_page[])
    function flush!()
        syncing[] && return
        draft=copy(rc.drafts[][shown_pass[]])
        for (i,field) in enumerate(fields[shown_page[]].fields)
            draft[field.key]=boxes[i].displayed_string[]
        end
        set_revision_pass!(rc,shown_pass[],draft)
    end
    function refresh!()
        syncing[]=true
        try
            selected=clamp(rc.selected[],1,length(rc.drafts[]))
            options=[("pass $i",i) for i in eachindex(rc.drafts[])]
            pass_menu.options[]==options || (pass_menu.options[]=options)
            pass_menu.i_selected[]=selected
            groups.i_selected[]=field_page[]
            shown_pass[]=selected;shown_page[]=field_page[]
            start=max(1,selected-1);last=min(length(rc.drafts[]),start+2)
            schedule=["$(i==selected ? ">" : " ") $i: window $(get(rc.drafts[][i],:window_size,"?"))" for i in start:last]
            pass_list.text[]=join([length(line)>29 ? String(collect(line)[1:26])*"..." : line for line in schedule],"\n")
            current=fields[field_page[]].fields
            for i in 1:6
                visible[i][]=i<=length(current)
                setters[i]()
                rowsize!(editors,i,i<=length(current) ? Fixed(30) : Fixed(0))
                if i<=length(current)
                    labels[i].text[]=current[i].label
                    value=rc.drafts[][selected][current[i].key]
                    if boxes[i].displayed_string[]!=value
                        boxes[i].stored_string[]=value
                        boxes[i].displayed_string[]=value
                    end
                end
            end
        finally
            syncing[]=false
        end
    end
    function guarded(action)
        rc.running[] && return
        try action()
        catch error
            rc.status[]="failed: $(Controllers._errmsg(error))"
        end
    end
    on(pass_menu.selection) do index
        syncing[] && return
        rc.running[] && return refresh!()
        guarded() do
            flush!();rc.selected[]=Int(index);refresh!()
        end
    end
    on(groups.selection) do index
        syncing[] && return
        rc.running[] && return refresh!()
        guarded() do
            flush!();field_page[]=Int(index);refresh!()
        end
    end
    # Raw text is live draft state, including invalid edits. Programmatic
    # refresh is guarded; no parser/file inspection runs in these callbacks.
    for (i,box) in enumerate(boxes)
        on(box.displayed_string) do text
            syncing[] && return
            if rc.running[]
                refresh!()
                return
            end
            current=fields[shown_page[]].fields
            i<=length(current) || return
            guarded() do
                draft=copy(rc.drafts[][shown_pass[]]);draft[current[i].key]=String(text)
                set_revision_pass!(rc,shown_pass[],draft)
            end
        end
    end
    on(_->(syncing[] || refresh!()),rc.drafts)
    on(_->(syncing[] || refresh!()),rc.selected)
    for (button,action) in ((up,()->move_revision_pass!(rc,rc.selected[],max(1,rc.selected[]-1))),
                            (down,()->move_revision_pass!(rc,rc.selected[],min(length(rc.drafts[]),rc.selected[]+1))),
                            (duplicate,()->insert_revision_pass!(rc,rc.selected[]+1;source=rc.selected[])),
                            (remove,()->delete_revision_pass!(rc,rc.selected[])))
        on(button.clicks) do _
            guarded() do
                flush!();action();refresh!()
            end
        end
    end
    on(preview.clicks) do _
        guarded() do
            flush!();apply_recipe_revision!(rc)
        end
    end
    on(save.clicks) do _
        guarded() do
            flush!();save_recipe_revision!(rc,save_path_picker)
        end
    end
    on(open.clicks) do _
        guarded() do
            rc.saved_record[]===nothing && throw(ArgumentError("save a revision first"))
            record=deepcopy(rc.saved_record[])
            @async begin
                yield()
                try open_revision(record)
                catch error
                    rc.status[]="failed opening saved revision: $(Controllers._errmsg(error))"
                end
            end
        end
    end
    _sync_menu!(tabs,sections)
    on(_->(page[]=1),full)
    on(_->(page[]=max(1,page[]-1)),previous.clicks)
    on(_->(page[]=min(length(chunks[]),page[]+1)),next.clicks)
    refresh!();reflow!(body.layoutobservables.suggestedbbox[])
    gl
end
