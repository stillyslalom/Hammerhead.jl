"""
    experiment_workflow(controller=ExperimentController(); batch=nothing, size=(1100,800)) -> Figure

Open a dedicated saved planar-experiment workflow. Recipes and run history are
read-only and retain all core fields. Open/save records, choose native result
output, explicitly allow environment changes, replay, and explore completed
output lazily, and save a shared run-quality report. With `batch=BatchRunner(...)`,
snapshot supported current form settings; unsupported callbacks produce an error
without saving a partial recipe.

Replay has a busy/completed/failed status, without live progress or cancellation.
Open the checkpoint workflow for resumable built-in recipes with pair-boundary
progress and cancellation; its creation candidate preserves the complete recipe.
Referenced custom scripts require a caller-supplied controller function and are
never loaded by this view. Large recipe/history text can be inspected in pages.
"""
function experiment_workflow(ec::ExperimentController=ExperimentController();
                             batch::Union{Nothing,BatchRunner}=nothing,size=(1100,800))
    fig=Figure(;size)
    experiment_workflow!(fig[1,1],ec;batch)
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
        return join(lines[1:max_lines],"\n")*"\n(full value in recipe pages)"
    end
    join(lines,"\n")
end

"""
    experiment_workflow!(target, controller; batch=nothing) -> GridLayout

Embed the saved-experiment workflow. The batch bridge creates a new intact
record; reopened recipes are never projected onto the narrow batch form.
"""
function experiment_workflow!(target,ec::ExperimentController;
                              batch::Union{Nothing,BatchRunner}=nothing)
    gl=GridLayout(target)
    controls=GridLayout(gl[1,1];valign=:top,tellheight=false)
    Label(controls[1,1],"saved experiment";font=:bold,halign=:left)
    open_btn=Button(controls[2,1];label="open experiment…",tellwidth=false)
    save_btn=Button(controls[3,1];label="save experiment…",tellwidth=false)
    snapshot_btn=Button(controls[4,1];label="snapshot batch",tellwidth=false)
    output_btn=Button(controls[5,1];label="choose result output…",tellwidth=false)
    mono=GLMakie.Makie.assetpath("fonts","DejaVuSansMono.ttf")
    output_label=lift(p->_experiment_wrap_text(isempty(p) ? "No result output selected." : p;max_lines=3),ec.output_path)
    Label(controls[6,1],output_label;halign=:left,justification=:left,
          font=mono,fontsize=12,width=240,tellwidth=false)
    history_btn=Button(controls[7,1];label="choose run record…",tellwidth=false)
    history_label=lift(p->_experiment_wrap_text(isempty(p) ? "Run history is not persisted." : "Run record: $p";max_lines=3),ec.run_record_path)
    Label(controls[8,1],history_label;halign=:left,justification=:left,
          font=mono,fontsize=12,width=240,tellwidth=false)
    override_toggle=Toggle(controls[9,1];active=ec.allow_environment_change[],halign=:left)
    Label(controls[10,1],"allow environment changes";halign=:left)
    run_btn=Button(controls[11,1];label="replay exact recipe",tellwidth=false)
    explore_btn=Button(controls[12,1];label="view completed results",tellwidth=false)
    quality_btn=Button(controls[13,1];label="save quality report…",tellwidth=false)
    quality_history_toggle=Toggle(controls[14,1];active=false,halign=:left)
    Label(controls[15,1],"include recorded history in report";halign=:left)
    status_label=lift(s->_experiment_wrap_text(s;max_lines=3),ec.status)
    Label(controls[16,1],status_label;halign=:left,justification=:left,
          font=mono,fontsize=12,width=240,tellwidth=false)
    Label(controls[17,1],"Replay starts from pair 1.\nUse checkpoints for progress,\ncancellation and resume.";
          halign=:left,justification=:left,word_wrap=true,width=240,tellwidth=false)

    content=GridLayout(gl[1,2];valign=:top,tellheight=false)
    section=Observable(:recipe)
    page=Observable(1)
    report_text=Observable("Save a quality report to inspect its summary here.")
    tabs=Menu(content[1,1:2];options=[("complete recipe",:recipe),("run history",:history),("quality report",:quality)])
    checkpoint_btn=Button(content[1,3];label="checkpoint / resume…",tellwidth=false)
    previous=Button(content[2,1];label="previous",tellwidth=false)
    next=Button(content[2,3];label="next",tellwidth=false)
    fulltext=lift(ec.record,section,ec.output_path,ec.run_record_path,ec.status,report_text) do _,which,output,history,status,report
        details=which===:recipe ? experiment_summary(ec) : which===:history ? experiment_run_history(ec) : report
        "selected output: $output\nrun record: $history\nstatus: $status\n\n"*details
    end
    # Wrap long full-pass descriptions explicitly so every field is reachable,
    # rather than allowing an enormous Label to extend outside the figure.
    chunks=lift(fulltext) do text
        lines=String[]
        for line in split(text,'\n')
            chars=collect(line)
            isempty(chars) && push!(lines,"")
            for first in 1:88:length(chars)
                push!(lines,String(chars[first:min(first+87,length(chars))]))
            end
        end
        [join(lines[i:min(i+21,length(lines))],"\n") for i in 1:22:length(lines)]
    end
    page_label=lift(page,chunks) do i,pages
        "page $(clamp(i,1,length(pages))) / $(length(pages))"
    end
    Label(content[2,2],page_label)
    text=lift(page,chunks) do i,pages
        pages[clamp(i,1,length(pages))]
    end
    Label(content[3,1:3],text;halign=:left,justification=:left,valign=:top,
          font=mono,fontsize=13,tellwidth=false,tellheight=false)
    colsize!(gl,1,Fixed(260))
    colgap!(gl,24)
    _sync_menu!(tabs,section)
    _sync_toggle!(override_toggle,ec.allow_environment_change)
    on(_->(page[]=1),fulltext)
    on(_->(page[]=max(1,page[]-1)),previous.clicks)
    on(_->(page[]=min(length(chunks[]),page[]+1)),next.clicks)
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
    on(ec.record) do _
        report_text[]="Experiment changed. Save a new quality report to inspect its summary."
    end
    on(ec.last_run) do _
        report_text[]="Run changed. Save a new quality report to inspect its summary."
    end
    on(quality_btn.clicks) do _
        include_measurement_history=quality_history_toggle.active[]
        guarded() do
            path=save_file(;filterlist="toml")
            isempty(path) && return
            report=save_experiment_quality_report(path,ec;include_measurement_history)
            report_text[]="Saved report: $path\n\n"*sprint(show,MIME"text/plain"(),report)
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
    gl
end
