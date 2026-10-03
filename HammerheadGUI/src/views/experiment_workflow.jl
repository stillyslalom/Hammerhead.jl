"""
    experiment_workflow(controller=ExperimentController(); batch=nothing, size=(1100,800)) -> Figure

Open a dedicated saved planar-experiment workflow. Recipes and run history are
read-only and retain all core fields. Open/save records, choose native result
output, explicitly allow environment changes, replay, and explore completed
output lazily, and save a shared run-quality report. With `batch=BatchRunner(...)`,
snapshot supported current form settings; unsupported callbacks produce an error
without saving a partial recipe.

Replay reports written-pair progress and accepts cancellation after the current
pair. It remains busy through prefetched loading/output/history cleanup. A GUI
cancelled attempt can have a failed v1 run record; native prefixes are not
resumable. Preflight/current-pair work can pause rendering: cooperative tasks and
offscreen tests do not establish responsive interaction. Open the checkpoint
workflow for resumable built-in recipes; its candidate preserves the complete recipe.
Referenced custom scripts require a caller-supplied controller function and are
never loaded by this view. Files, Replay and Reports group the controls while
cancel, written-pair progress and a status preview remain visible. Switching
groups preserves settings and report choices. Text pages adapt to their allocated
space, including full destinations, status, identities and recipe/report details;
the standalone layout supports 900×600 and larger windows.
Report history/execution toggles are independent, unchecked by default and
captured before selecting a path. Execution opts into core format 3; report
verification is at generation time, and failed scans/saves retain the previous
summary with its own run/recipe/input identity. `report_path_picker` optionally
returns a destination string instead of opening the default TOML save dialog.
"""
function experiment_workflow(ec::ExperimentController=ExperimentController();
                             batch::Union{Nothing,BatchRunner}=nothing,size=(1100,800),
                             report_path_picker::Function=()->save_file(;filterlist="toml"))
    fig=Figure(;size)
    experiment_workflow!(fig[1,1],ec;batch,report_path_picker)
    fig
end

function _experiment_wrap_text(text; columns=28,max_lines=nothing)
    lines=String[]
    for line in split(text,'\n')
        chars=collect(line)
        isempty(chars) && push!(lines,"")
        for i in 1:columns:length(chars)
            push!(lines,String(chars[i:min(i+columns-1,length(chars))]))
        end
    end
    if max_lines!==nothing && length(lines)>max_lines
        return join(lines[1:max_lines],"\n")*"\n(full value in text pages)"
    end
    join(lines,"\n")
end

function _workflow_panel!(panel,blocks,mode,kind)
    # Makie's widget hit boxes remain active when only drawing is hidden.
    # Redirect inactive allocations before child layout callbacks run, so their
    # mouse regions cannot overlap the visible panel. Widgets stay alive.
    on(panel.layoutobservables.suggestedbbox;priority=typemax(Int)) do box
        if mode[]!==kind && box.origin[1]>-5000
            panel.layoutobservables.suggestedbbox[]=Rect2f(-10000,-10000,box.widths...)
            return Consume(true)
        end
        Consume(false)
    end
    ()->foreach(b->(b.blockscene.visible[]=mode[]===kind),blocks)
end

function _workflow_pages(text,columns,lines_per_page)
    lines=split(_experiment_wrap_text(text;columns), '\n';keepempty=true)
    [join(lines[i:min(i+lines_per_page-1,length(lines))],"\n") for i in 1:lines_per_page:length(lines)]
end

"""
    experiment_workflow!(target, controller; batch=nothing) -> GridLayout

Embed the saved-experiment workflow. The batch bridge creates a new intact
record; reopened recipes are never projected onto the narrow batch form.
"""
function experiment_workflow!(target,ec::ExperimentController;
                              batch::Union{Nothing,BatchRunner}=nothing,
                              report_path_picker::Function=()->save_file(;filterlist="toml"))
    gl=GridLayout(target)
    controls=GridLayout(gl[1,1];valign=:top,tellheight=false)
    rowgap!(controls,3)
    Label(controls[1,1],"saved experiment";font=:bold,halign=:left)
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
    snapshot_btn=Button(files_panel[3,1];label="snapshot batch",tellwidth=false,height=28,fontsize=14)
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
    run_btn=Button(replay_panel[2,1];label="replay exact recipe",tellwidth=false,height=28,fontsize=14)
    explore_btn=Button(replay_panel[3,1];label="view completed results",tellwidth=false,height=28,fontsize=14)
    quality_btn=Button(reports_panel[1,1];label="save quality report…",tellwidth=false,height=28,fontsize=14)
    quality_mode=GridLayout(reports_panel[2,1])
    quality_history_toggle=Toggle(quality_mode[1,1];active=false,halign=:left)
    history_mode_info=Label(quality_mode[1,2],"include recorded history in report";halign=:left,fontsize=13,word_wrap=true,width=195,tellwidth=false)
    quality_execution_toggle=Toggle(quality_mode[2,1];active=false,halign=:left)
    execution_mode_info=Label(quality_mode[2,2],"include recorded execution in report";halign=:left,fontsize=13,word_wrap=true,width=195,tellwidth=false)
    persistent=GridLayout(controls[6,1];valign=:top)
    rowgap!(persistent,4)
    cancel_color=lift(busy->busy ? RGBf(.12,.12,.12) : RGBf(.65,.65,.65),ec.running)
    cancel_btn=Button(persistent[1,1];label="cancel after current pair",tellwidth=false,height=28,fontsize=14,
        labelcolor=cancel_color,labelcolor_hover=cancel_color,labelcolor_active=cancel_color)
    counts=lift((p)->"Written pairs: $(p[1]) / $(p[2])",ec.progress)
    Label(persistent[2,1],counts;halign=:left,fontsize=13)
    status_label=lift(s->_experiment_wrap_text(s;max_lines=2),ec.status)
    Label(persistent[3,1],status_label;halign=:left,justification=:left,
          font=mono,fontsize=12,width=240,tellwidth=false)
    kinds=(:files,:replay,:reports)
    visible=(_workflow_panel!(files_panel,(open_btn,save_btn,snapshot_btn,output_btn,output_info,history_btn,history_info),control_section,:files),
        _workflow_panel!(replay_panel,(override_toggle,environment_info,run_btn,explore_btn),control_section,:replay),
        _workflow_panel!(reports_panel,(quality_btn,quality_history_toggle,history_mode_info,quality_execution_toggle,execution_mode_info),control_section,:reports))
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
    tabs=Menu(content[1,1:2];options=[("complete recipe",:recipe),("run history",:history),("quality report",:quality)])
    checkpoint_btn=Button(content[1,3];label="checkpoint / resume…",tellwidth=false)
    comparison_btn=Button(content[4,1:3];label="compare a representative pair…",tellwidth=false)
    previous=Button(content[2,1];label="previous",tellwidth=false)
    next=Button(content[2,3];label="next",tellwidth=false)
    fulltext=lift(ec.record,section,ec.output_path,ec.run_record_path,ec.status,report_text) do _,which,output,history,status,report
        details=which===:recipe ? experiment_summary(ec) : which===:history ? experiment_run_history(ec) : report
        "selected output: $output\nrun record: $history\nstatus: $status\n"*
        "Replay starts at pair 1; cancel waits for writes/cleanup; use checkpoints to resume.\n\n"*details
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
            batch===nothing && throw(ArgumentError("open this workflow with a BatchRunner to snapshot its form"))
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
    on(ec.record) do _
        report_text[]="Experiment changed. Save a new quality report to inspect its summary."
    end
    on(ec.last_run) do _
        report_text[]="Run changed. Save a new quality report to inspect its summary."
    end
    on(quality_btn.clicks) do _
        include_measurement_history=quality_history_toggle.active[]
        include_execution_diagnostics=quality_execution_toggle.active[]
        guarded() do
            path=report_path_picker()
            isempty(path) && return
            report=save_experiment_quality_report(path,ec;include_measurement_history,include_execution_diagnostics)
            provenance=quality_report_data(report)["provenance"]
            report_text[]="Saved report: $path\nReported run: $(provenance["run_id"])\nRecipe: $(provenance["recipe_id"])\nInput: $(provenance["input_id"])\nVerification describes report generation time.\n\n"*sprint(show,MIME"text/plain"(),report)
            section[]=:quality
            ec.status[]="quality report saved"
        end
    end
    on(explore_btn.clicks) do _
        guarded() do
            display(GLMakie.Screen(),result_explorer(experiment_results(ec)))
        end
    end
    on(checkpoint_btn.clicks) do _
        guarded() do
            record=deepcopy(ec.record[])
            display(GLMakie.Screen(),checkpoint_workflow(CheckpointController();record))
        end
    end
    on(comparison_btn.clicks) do _
        guarded() do
            record=deepcopy(ec.record[])
            record===nothing && throw(ArgumentError("open or snapshot an experiment first"))
            protected=filter(!isempty,[ec.output_path[],ec.run_record_path[]])
            controller=RecipeComparisonController(record;protected_paths=protected)
            display(GLMakie.Screen(),recipe_comparison(controller))
        end
    end
    gl
end
