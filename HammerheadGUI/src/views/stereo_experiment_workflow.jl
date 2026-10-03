"""
    stereo_experiment_workflow(controller=StereoExperimentController(); batch=nothing,
        size=(1100,800), report_path_picker=...) -> Figure

Open the separate saved fitted-stereo lane. Snapshot a file-based fitted batch
or open complete immutable-by-contract settings, select historical run IDs,
replay/cancel after acquisition writes, verify lazy completed output and save
associated reports. Camera fitting/scripts/checkpoints are not executed.
Files/Replay/Reports controls and allocation-sized text pages support 900x600
and larger windows; cancellation/progress/status remain visible. Prior reports
keep their own identities after failed actions or selection changes. Execution
and input-verification report options are captured before the path dialog.
Verification is at opening/generation time, not calibration accuracy or ongoing
validity. Cooperative tasks/offscreen checks do not prove desktop responsiveness.
"""
function stereo_experiment_workflow(ec::StereoExperimentController=StereoExperimentController();
                             batch::Union{Nothing,StereoBatchRunner}=nothing,size=(1100,800),
                             report_path_picker::Function=()->save_file(;filterlist="toml"))
    fig=Figure(;size)
    stereo_experiment_workflow!(fig[1,1],ec;batch,report_path_picker)
    fig
end

"""
    stereo_experiment_workflow!(target, controller::StereoExperimentController;
        batch=nothing, report_path_picker=...) -> GridLayout

Embed the saved-stereo view without reducing imported settings to the batch form.
The optional report picker returns a destination string; it is called only after
capturing the report request. Native result displays retain their captured run ID.
"""
function stereo_experiment_workflow!(target,ec::StereoExperimentController;
                              batch::Union{Nothing,StereoBatchRunner}=nothing,
                              report_path_picker::Function=()->save_file(;filterlist="toml"))
    gl=GridLayout(target)
    controls=GridLayout(gl[1,1];valign=:top,tellheight=false)
    rowgap!(controls,3)
    Label(controls[1,1],"saved stereo experiment";font=:bold,halign=:left)
    control_section=Observable(ec.record[]===nothing ? :files : :replay)
    control_menu=Menu(controls[2,1];options=[("Files",:files),("Replay",:replay),("Reports",:reports)],tellwidth=false)
    files_panel=GridLayout(controls[3,1];valign=:top)
    replay_panel=GridLayout(controls[4,1];valign=:top)
    reports_panel=GridLayout(controls[5,1];valign=:top)
    for panel in (files_panel,replay_panel,reports_panel)
        rowgap!(panel,4)
    end
    open_btn=Button(files_panel[1,1];label="open experiment…",tellwidth=false,height=28,fontsize=14)
    save_btn=Button(files_panel[2,1];label="save experiment…",tellwidth=false,height=28,fontsize=14)
    snapshot_btn=Button(files_panel[3,1];label="snapshot stereo batch",tellwidth=false,height=28,fontsize=14)
    output_btn=Button(files_panel[4,1];label="choose result output…",tellwidth=false,height=28,fontsize=14)
    mono=GLMakie.Makie.assetpath("fonts","DejaVuSansMono.ttf")
    output_label=lift(p->_experiment_wrap_text(isempty(p) ? "No result output selected." : p;max_lines=1),ec.output_path)
    output_info=Label(files_panel[5,1],output_label;halign=:left,justification=:left,
          font=mono,fontsize=12,width=240,tellwidth=false)
    history_btn=Button(files_panel[6,1];label="choose run record…",tellwidth=false,height=28,fontsize=14)
    history_label=lift(p->_experiment_wrap_text(isempty(p) ? "Run history is not persisted." : "Run record: $p";max_lines=1),ec.run_record_path)
    history_info=Label(files_panel[7,1],history_label;halign=:left,justification=:left,
          font=mono,fontsize=12,width=240,tellwidth=false)
    environment=GridLayout(replay_panel[1,1])
    override_toggle=Toggle(environment[1,1];active=ec.allow_environment_change[],halign=:left)
    environment_info=Label(environment[1,2],"allow environment changes";halign=:left,fontsize=13)
    recording=GridLayout(replay_panel[2:3,1])
    diagnostics_toggle=Toggle(recording[1,1];active=ec.record_diagnostics[],halign=:left)
    diagnostics_info=Label(recording[1,2],"record camera execution";halign=:left,fontsize=13)
    timing_toggle=Toggle(recording[2,1];active=ec.record_pair_timing[],halign=:left)
    timing_info=Label(recording[2,2],"record acquisition timing";halign=:left,fontsize=13)
    run_menu=Menu(replay_panel[6,1];options=[("no recorded runs","")],tellwidth=false,height=28)
    run_btn=Button(replay_panel[4,1];label="replay exact recipe",tellwidth=false,height=28,fontsize=14)
    explore_btn=Button(replay_panel[5,1];label="view completed results",tellwidth=false,height=28,fontsize=14)
    quality_btn=Button(reports_panel[1,1];label="save quality report…",tellwidth=false,height=28,fontsize=14)
    quality_mode=GridLayout(reports_panel[2,1])
    quality_inputs_toggle=Toggle(quality_mode[1,1];active=false,halign=:left)
    inputs_mode_info=Label(quality_mode[1,2],"check current input bytes";halign=:left,fontsize=13,word_wrap=true,width=195,tellwidth=false)
    quality_execution_toggle=Toggle(quality_mode[2,1];active=false,halign=:left)
    execution_mode_info=Label(quality_mode[2,2],"include recorded execution in report";halign=:left,fontsize=13,word_wrap=true,width=195,tellwidth=false)
    persistent=GridLayout(controls[6,1];valign=:top)
    rowgap!(persistent,4)
    cancel_color=lift(busy->busy ? RGBf(.12,.12,.12) : RGBf(.65,.65,.65),ec.running)
    cancel_btn=Button(persistent[1,1];label="cancel after acquisition",tellwidth=false,height=28,fontsize=14,
        labelcolor=cancel_color,labelcolor_hover=cancel_color,labelcolor_active=cancel_color)
    counts=lift((p)->"Written acquisitions: $(p[1]) / $(p[2])",ec.progress)
    Label(persistent[2,1],counts;halign=:left,fontsize=13)
    status_label=lift(s->_experiment_wrap_text(s;max_lines=2),ec.status)
    Label(persistent[3,1],status_label;halign=:left,justification=:left,
          font=mono,fontsize=12,width=240,tellwidth=false)
    kinds=(:files,:replay,:reports)
    visible=(_workflow_panel!(files_panel,(open_btn,save_btn,snapshot_btn,output_btn,output_info,history_btn,history_info),control_section,:files),
        _workflow_panel!(replay_panel,(override_toggle,environment_info,diagnostics_toggle,diagnostics_info,timing_toggle,timing_info,run_menu,run_btn,explore_btn),control_section,:replay),
        _workflow_panel!(reports_panel,(quality_btn,quality_inputs_toggle,inputs_mode_info,quality_execution_toggle,execution_mode_info),control_section,:reports))
    function show_controls!()
        foreach(f->f(),visible)
        for (i,kind) in enumerate(kinds)
            rowsize!(controls,i+2,control_section[]===kind ? Auto() : Fixed(0))
        end
    end
    on(_->show_controls!(),control_section)
    _sync_menu!(control_menu,control_section)
    show_controls!()

    content=GridLayout(gl[1,2];valign=:top,tellheight=false)
    section=Observable(:recipe)
    page=Observable(1)
    report_text=Observable("Save a quality report to inspect its summary here.")
    tabs=Menu(content[1,1:3];options=[("complete recipe",:recipe),("run history",:history),("quality report",:quality)])
    previous=Button(content[2,1];label="previous",tellwidth=false)
    next=Button(content[2,3];label="next",tellwidth=false)
    recipe_text=lift(_->experiment_summary(ec),ec.record)
    history_text=lift(_->experiment_run_history(ec),ec.record)
    fulltext=lift(recipe_text,history_text,section,ec.output_path,ec.run_record_path,ec.status,report_text,ec.selected_run_id,ec.active_request,ec.last_run) do recipe,recorded,which,output,history,status,report,selected,active,latest
        details=which===:recipe ? recipe : which===:history ? recorded : report
        active_text=active===nothing ? "Active replay: none\n" :
            "Active recipe: $(active.recipe_id)\nActive input: $(active.input_id)\nActive output: $(active.output)\nActive run record: $(active.run_record)\n"
        "Next replay output: $output\nNext run record: $history\n"*active_text*
        "Latest recorded attempt: $(latest===nothing ? "unavailable" : latest.run_id)\nSelected historical run: $selected\nStatus: $status\n"*
        "Replay starts at acquisition 1; cancel waits for writes/cleanup. Stereo prefixes are not resumable.\n\n"*details
    end
    capacity=Observable((88,22))
    chunks=lift((text,size)->_workflow_pages(text,size...),fulltext,capacity)
    page_label=lift(page,chunks) do i,pages
        "page $(clamp(i,1,length(pages))) / $(length(pages))"
    end
    Label(content[2,2],page_label)
    text=lift(page,chunks) do i,pages
        pages[clamp(i,1,length(pages))]
    end
    body=Label(content[3,1:3],text;halign=:left,justification=:left,valign=:top,
          font=mono,fontsize=13,tellwidth=false,tellheight=false)
    function reflow!(box)
        # DejaVu Sans Mono at 13px; leave space beyond its measured advance and
        # line height. Allocation, not intrinsic glyph size, drives pagination.
        next_capacity=(max(1,floor(Int,(box.widths[1]-8)/8.1)),max(1,floor(Int,(box.widths[2]-8)/16.5)))
        next_capacity==capacity[] || (capacity[]=next_capacity)
    end
    on(reflow!,body.layoutobservables.suggestedbbox)
    colsize!(gl,1,Fixed(260))
    colgap!(gl,24)
    _sync_menu!(tabs,section)
    _sync_toggle!(override_toggle,ec.allow_environment_change)
    _sync_toggle!(diagnostics_toggle,ec.record_diagnostics)
    _sync_toggle!(timing_toggle,ec.record_pair_timing)
    updating_runs=Ref(false)
    function refresh_runs!()
        updating_runs[]=true
        try
            record=ec.record[]
            options=record===nothing || isempty(record.runs) ? [("no recorded runs","")] :
                [("$(first(r.run_id,8)) | $(r.status) | $(r.completed_pairs)/$(length(record.pairs))",r.run_id) for r in record.runs]
            run_menu.options[]=options
            chosen=findfirst(o->last(o)==ec.selected_run_id[],options)
            run_menu.i_selected[]=chosen===nothing ? 1 : chosen
        finally
            updating_runs[]=false
        end
    end
    on(_->refresh_runs!(),ec.record)
    on(_->refresh_runs!(),ec.selected_run_id)
    refresh_runs!()
    on(_->(page[]=1),fulltext)
    on(chunks) do pages
        index=clamp(page[],1,length(pages))
        index==page[] || (page[]=index)
    end
    on(_->(page[]=max(1,page[]-1)),previous.clicks)
    on(_->(page[]=min(length(chunks[]),page[]+1)),next.clicks)
    reflow!(body.layoutobservables.suggestedbbox[])
    guarded = action -> begin
        if ec.running[]
            return
        end
        try
            action()
        catch err
            ec.status[]="failed: $(Controllers._errmsg(err))"
        end
    end
    on(run_menu.selection) do id
        updating_runs[] && return
        if ec.running[]
            refresh_runs!()
            return
        end
        guarded() do
            isempty(id) || select_experiment_run!(ec,id)
        end
    end
    on(open_btn.clicks) do _
        guarded() do
            path=pick_file(;filterlist="jld2")
            isempty(path) || open_experiment!(ec,path)
        end
    end
    on(save_btn.clicks) do _
        guarded() do
            path=save_file(;filterlist="jld2")
            isempty(path) || save_experiment_record!(ec,path)
        end
    end
    on(snapshot_btn.clicks) do _
        guarded() do
            batch===nothing && throw(ArgumentError("open this workflow with a StereoBatchRunner to snapshot its form"))
            open_experiment!(ec,experiment_record(batch))
        end
    end
    on(output_btn.clicks) do _
        guarded() do
            path=save_file(;filterlist="jld2")
            isempty(path) || (ec.output_path[]=path)
        end
    end
    on(history_btn.clicks) do _
        guarded() do
            path=save_file(;filterlist="jld2")
            isempty(path) || (ec.run_record_path[]=path)
        end
    end
    on(_->start!(ec),run_btn.clicks)
    on(_->cancel!(ec),cancel_btn.clicks)
    on(quality_btn.clicks) do _
        verify_inputs=quality_inputs_toggle.active[]
        include_execution_diagnostics=quality_execution_toggle.active[]
        guarded() do
            # Freeze the selected record/run and all protected destinations before
            # a dialog or observer can change next choices.
            request=StereoExperimentController(deepcopy(ec.record[]);
                output_path=ec.output_path[],run_record_path=ec.run_record_path[])
            request.selected_run_id.val=ec.selected_run_id[]
            path=report_path_picker()
            isempty(path) && return
            report=save_experiment_quality_report(path,request;verify_inputs,include_execution_diagnostics)
            provenance=quality_report_data(report)["provenance"]
            report_text[]="Saved report: $path\nReported run: $(provenance["run_id"])\nRecipe: $(provenance["recipe_id"])\nInput: $(provenance["input_id"])\nVerification describes report generation time.\n\n"*sprint(show,MIME"text/plain"(),report)
            section[]=:quality
            ec.status[]="quality report saved"
        end
    end
    on(explore_btn.clicks) do _
        guarded() do
            record,run=Controllers._stereo_selected_request(ec)
            explorer=experiment_results(ec;run_id=run.run_id)
            figure=result_explorer(explorer)
            Label(figure[0,1],"Displayed stereo run: $(run.run_id)";fontsize=12)
            display(GLMakie.Screen(),figure)
        end
    end
    gl
end
