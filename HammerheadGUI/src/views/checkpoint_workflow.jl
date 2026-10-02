"""
    checkpoint_workflow(controller=CheckpointController(); record=nothing,
                        batch=nothing, size=(1100,800)) -> Figure

Open a dedicated planar checkpoint workflow. A supplied complete experiment
`record` is copied as a creation candidate; it is never projected onto the batch
form. Unsupported script recipes remain inspectable with a clear creation
refusal, and opening an existing checkpoint remains available. With `batch`,
snapshot the current supported form using the existing exact experiment bridge.

Choose separate new/empty metadata and per-pair result directories, create/open,
resume, request cancellation at a committed pair boundary, explicitly refresh,
browse a fixed lazy prefix and export complete data to a fresh native file.
Recovery requires the unchecked stopped-writer assertion on each invocation.
Software identity is strict; there is no environment override. Initial
verification/current-pair computation can pause rendering. No in-pass progress,
immediate cancellation, live reader or diagnostics are implied.
"""
function checkpoint_workflow(cc::CheckpointController=CheckpointController();
                             record::Union{Nothing,ExperimentRecord}=nothing,
                             batch::Union{Nothing,BatchRunner}=nothing,size=(1100,800))
    record===nothing || Controllers._checkpoint_recipe!(cc,record)
    fig=Figure(;size)
    checkpoint_workflow!(fig[1,1],cc;batch)
    fig
end

function _checkpoint_button!(target,label,enabled)
    neutral=RGBf(0.94,0.94,0.94)
    color=lift(v->v ? neutral : RGBf(0.97,0.97,0.97),enabled)
    text=lift(v->v ? RGBf(0.12,0.12,0.12) : RGBf(0.65,0.65,0.65),enabled)
    button=Button(target;label,fontsize=14,height=28,padding=(6,6,4,4),tellwidth=false,buttoncolor=color,
        buttoncolor_hover=color,buttoncolor_active=color,labelcolor=text,
        labelcolor_hover=text,labelcolor_active=text)
    button
end

"""
    checkpoint_workflow!(target, controller; batch=nothing) -> GridLayout

Embed the checkpoint view. All execution logic stays in the controller; disabled
actions are visually muted and their callbacks refuse to run. Long paths and
exact recipe text wrap explicitly and remain fully reachable in paginated text.
"""
function checkpoint_workflow!(target,cc::CheckpointController;batch::Union{Nothing,BatchRunner}=nothing)
    gl=GridLayout(target)
    controls=GridLayout(gl[1,1];valign=:top,tellheight=false)
    rowgap!(controls,5)
    mono=GLMakie.Makie.assetpath("fonts","DejaVuSansMono.ttf")
    idle=lift(!,cc.running)
    has_store=lift(cp->cp!==nothing,cc.checkpoint)
    loaded_idle=lift((a,b)->a && b,idle,has_store)
    create_allowed=lift(cc.running,cc.record,cc.checkpoint) do busy,record,cp
        !busy && cp===nothing && record!==nothing && record.recipe.external_preprocess===nothing
    end
    prefix_allowed=lift((ready,p)->ready && p[1]>0,loaded_idle,cc.progress)
    export_allowed=lift((ready,complete)->ready && complete,loaded_idle,cc.data_complete)
    snapshot_allowed=lift(busy->!busy && batch!==nothing,cc.running)
    Label(controls[1,1],"experiment checkpoint";font=:bold,halign=:left)
    open_btn=_checkpoint_button!(controls[2,1],"open checkpoint…",idle)
    metadata_btn=_checkpoint_button!(controls[3,1],"metadata directory…",create_allowed)
    output_btn=_checkpoint_button!(controls[4,1],"per-pair result directory…",create_allowed)
    paths=lift(cc.checkpoint_dir,cc.output_dir) do metadata,output
        "Metadata: "*(isempty(metadata) ? "not selected" : metadata)*"\nResults: "*(isempty(output) ? "not selected" : output)
    end
    Label(controls[5,1],lift(p->_experiment_wrap_text(p;columns=29,max_lines=5),paths);
        halign=:left,justification=:left,font=mono,fontsize=11,width=250,tellwidth=false)
    create_btn=_checkpoint_button!(controls[6,1],"create checkpoint",create_allowed)
    recovery_row=GridLayout(controls[7,1])
    recovery=Toggle(recovery_row[1,1];active=false,halign=:left)
    Label(recovery_row[1,2],"former writer has stopped";halign=:left,fontsize=12)
    _sync_toggle!(recovery,cc.recover_interrupted)
    Label(controls[8,1],"Explicit recovery assertion; reset\nafter every open/create/start.\nNever recover an active writer.";
        halign=:left,justification=:left,fontsize=11)
    resume_btn=_checkpoint_button!(controls[9,1],"resume committed prefix",loaded_idle)
    cancel_btn=_checkpoint_button!(controls[10,1],"cancel after current pair",cc.running)
    refresh_btn=_checkpoint_button!(controls[11,1],"refresh checkpoint state",loaded_idle)
    browse_btn=_checkpoint_button!(controls[12,1],"browse fixed prefix",prefix_allowed)
    export_btn=_checkpoint_button!(controls[13,1],"export complete native file…",export_allowed)
    counts=lift(cc.progress,cc.checkpoint_status,cc.data_complete) do p,status,complete
        "$(p[1]) / $(p[2]) committed\nAttempt: $status\nData complete: $complete"
    end
    Label(controls[14,1],counts;halign=:left,justification=:left,font=mono,fontsize=12)
    Label(controls[15,1],lift(s->_experiment_wrap_text(s;columns=29,max_lines=3),cc.status);
        halign=:left,justification=:left,font=mono,fontsize=11,width=250,tellwidth=false)

    content=GridLayout(gl[1,2];valign=:top,tellheight=false)
    section=Observable(:recipe);page=Observable(1)
    toolbar=GridLayout(content[1,1:3];tellwidth=false,halign=:left)
    tabs=Menu(toolbar[1,1];width=230,options=[("complete recipe",:recipe),("checkpoint state",:state)])
    recipe_btn=_checkpoint_button!(toolbar[1,2],"choose complete recipe…",idle)
    batch_btn=_checkpoint_button!(toolbar[1,3],"snapshot batch",snapshot_allowed)
    colsize!(toolbar,1,Fixed(230));colsize!(toolbar,2,Fixed(240));colsize!(toolbar,3,Fixed(170))
    colgap!(toolbar,12)
    previous=Button(content[2,1];label="previous",tellwidth=false)
    next=Button(content[2,3];label="next",tellwidth=false)
    recipe_text=lift(cc.record) do record
        record===nothing ? "Choose a complete recipe or open an existing checkpoint." :
            experiment_summary(ExperimentController(record))
    end
    fulltext=lift(recipe_text,cc.checkpoint,cc.progress,cc.checkpoint_status,cc.data_complete,
                  cc.last_attempt,cc.status,cc.checkpoint_dir,cc.output_dir,cc.export_path,section) do recipe,cp,p,status,complete,attempt,message,metadata,output,export_path,which
        paths="Metadata directory: $metadata\nPer-pair output: $output\nNative export: $export_path\nStatus: $message\n\n"
        if which===:recipe
            return paths*recipe
        end
        store=cp===nothing ? "No checkpoint is open." : "Checkpoint: $(cp.checkpoint_id)\nRecipe: $(recipe_identity(cp.record.recipe))\nOrdered input: $(cp.record.input_id)"
        last=attempt===nothing ? "No attempt returned in this session." :
            "Last returned attempt: $(attempt.attempt_id)\nStatus: $(attempt.status)\nStarted after: $(attempt.start_committed)\nCommitted: $(attempt.committed) / $(attempt.total_pairs)\nStarted at: $(attempt.started_at)\nFinished at: $(attempt.finished_at)"
        paths*store*"\nCommitted: $(p[1]) / $(p[2])\nNative attempt status: $status\nData complete: $complete\n\n$last\n\n"*
            "Progress counts published descriptors. Cancellation waits for the pair in flight.\nA complete prefix can have an unfinished/failed attempt. Recovery is never automatic.\nBrowsing opens a fixed verified prefix, not a live reader. Refresh only while idle.\nSoftware must match exactly; no environment override or custom-script restart.\nVerification/current-pair processing may pause rendering; no in-pass progress.\nNative export is optional and requires a fresh destination.\nDetailed saved attempt history remains in the core checkpoint store."
    end
    chunks=lift(fulltext) do text
        lines=split(_experiment_wrap_text(text;columns=86),'\n')
        [join(lines[i:min(i+24,length(lines))],"\n") for i in 1:25:length(lines)]
    end
    Label(content[2,2],lift((i,pages)->"page $(clamp(i,1,length(pages))) / $(length(pages))",page,chunks))
    Label(content[3,1:3],lift((i,pages)->pages[clamp(i,1,length(pages))],page,chunks);
        halign=:left,justification=:left,valign=:top,font=mono,fontsize=12,tellwidth=false,tellheight=false)
    colsize!(gl,1,Fixed(270));colgap!(gl,20)
    _sync_menu!(tabs,section)
    on(_->(page[]=1),section)
    on(_->(page[]=min(page[],length(chunks[]))),chunks)
    on(_->(page[]=max(1,page[]-1)),previous.clicks)
    on(_->(page[]=min(length(chunks[]),page[]+1)),next.clicks)

    guarded=(allowed,action)->begin
        allowed[] || return
        try action()
        catch err
            cc.status[]="failed: $(Controllers._errmsg(err))"
        end
    end
    on(open_btn.clicks) do _
        guarded(idle,()->begin
            path=pick_folder()
            isempty(path) || open_checkpoint!(cc,path)
        end)
    end
    on(recipe_btn.clicks) do _
        guarded(idle,()->begin
            path=pick_file(;filterlist="jld2")
            isempty(path) || Controllers._checkpoint_recipe!(cc,load_experiment(path))
        end)
    end
    on(batch_btn.clicks) do _
        guarded(snapshot_allowed,()->Controllers._checkpoint_recipe!(cc,experiment_record(batch)))
    end
    on(metadata_btn.clicks) do _
        guarded(create_allowed,()->begin
            path=pick_folder();isempty(path) || (cc.checkpoint_dir[]=path)
        end)
    end
    on(output_btn.clicks) do _
        guarded(create_allowed,()->begin
            path=pick_folder();isempty(path) || (cc.output_dir[]=path)
        end)
    end
    on(_->guarded(create_allowed,()->create_checkpoint!(cc)),create_btn.clicks)
    on(_->guarded(loaded_idle,()->start!(cc)),resume_btn.clicks)
    on(_->guarded(cc.running,()->cancel!(cc)),cancel_btn.clicks)
    on(_->guarded(loaded_idle,()->refresh_checkpoint!(cc)),refresh_btn.clicks)
    on(browse_btn.clicks) do _
        guarded(prefix_allowed,()->display(GLMakie.Screen(),result_explorer(checkpoint_explorer(cc))))
    end
    on(export_btn.clicks) do _
        guarded(export_allowed,()->begin
            path=save_file(;filterlist="jld2")
            isempty(path) || export_checkpoint_results!(cc,path)
        end)
    end
    gl
end
