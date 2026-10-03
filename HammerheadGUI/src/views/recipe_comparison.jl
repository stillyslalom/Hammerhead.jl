"""
    recipe_comparison(controller=RecipeComparisonController(); size=(1100,800)) -> Figure

Inspect or run a saved-recipe representative-pair comparison in a separate
window. Choose explicit before/after records and pair indices, pixel or physical
basis, and a separately unchecked environment override. The complete core recipes
are never projected into editable settings forms. Paged request/report sections
keep current choices separate from the last report's recorded provenance.
Loading a report is read-only and does not reverify inputs or rerun recipes.
The run is one pair per recipe, with no live progress/cancellation guarantee;
CPU preflight/computation can block despite asynchronous scheduling.
"""
function recipe_comparison(cc::RecipeComparisonController=RecipeComparisonController();size=(1100,800))
    fig=Figure(;size)
    recipe_comparison!(fig[1,1],cc)
    fig
end

function _comparison_pages(text;columns=82,lines_per_page=25)
    lines=String[]
    for line in split(text,'\n')
        chars=collect(line)
        while length(chars)>columns
            boundary=findlast(isspace,view(chars,1:columns))
            width=boundary===nothing || boundary<=1 ? columns : boundary-1
            push!(lines,rstrip(String(chars[1:width])))
            consumed=boundary===nothing || boundary<=1 ? width : boundary
            chars=chars[consumed+1:end]
        end
        push!(lines,String(chars))
    end
    [join(lines[i:min(i+lines_per_page-1,length(lines))],"\n") for i in 1:lines_per_page:length(lines)]
end

"""
    recipe_comparison!(target, controller) -> GridLayout

Embed the comparison view. Record/report actions, pair edits and computation
are refused while busy. Basis/environment controls may describe the next
attempt; the running request is already frozen. Save exports the last report without silently
running current choices. Every settings/population/provenance line is reachable
through pages; raw numerical result arrays are not retained by this view.
"""
function recipe_comparison!(target,cc::RecipeComparisonController)
    gl=GridLayout(target)
    controls=GridLayout(gl[1,1];valign=:top,tellheight=false)
    rowgap!(controls,8)
    colsize!(gl,1,Fixed(290));colgap!(gl,24)
    Label(controls[1,1],"representative pair comparison";halign=:left,font=:bold)
    before_btn=Button(controls[2,1];label="open before experiment…",tellwidth=false)
    after_btn=Button(controls[3,1];label="open after experiment…",tellwidth=false)
    indices=GridLayout(controls[4,1])
    Label(indices[1,1],"before pair";halign=:left)
    Label(indices[1,2],"after pair";halign=:left)
    before_box=Textbox(indices[2,1];stored_string=string(cc.pair_indices[][1]),width=130)
    after_box=Textbox(indices[2,2];stored_string=string(cc.pair_indices[][2]),width=130)
    Label(controls[5,1],"difference basis";halign=:left)
    basis=Menu(controls[6,1];options=[("raw displacement (px)",:pixels),("physical velocity",:physical)],tellwidth=false)
    _sync_menu!(basis,cc.basis)
    environment=GridLayout(controls[7,1])
    override=Toggle(environment[1,1];active=cc.allow_environment_change[])
    Label(environment[1,2],"allow environment changes";halign=:left,fontsize=13)
    _sync_toggle!(override,cc.allow_environment_change)
    run_btn=Button(controls[8,1];label="compare selected pair",tellwidth=false)
    report_actions=GridLayout(controls[9,1])
    load_btn=Button(report_actions[1,1];label="open report…",tellwidth=false)
    save_btn=Button(report_actions[1,2];label="save last report…",tellwidth=false)
    Label(controls[10,1],cc.status;halign=:left,justification=:left,
        word_wrap=true,width=285,tellwidth=false,fontsize=13)
    Label(controls[11,1],"Differences show sensitivity, not accuracy.\nOnly exact common grid nodes are paired.\nPhysical basis requires identical scales.\nNo progress or cancellation guarantee.";
        halign=:left,justification=:left,word_wrap=true,width=285,tellwidth=false,fontsize=13)
    content=GridLayout(gl[1,2];valign=:top,tellheight=false)
    section=Observable(:request)
    sections=Menu(content[1,1:3];options=[("current request",:request),("last report summary",:summary),
        ("recorded settings changes",:settings),("recorded populations and differences",:populations),
        ("recorded provenance",:provenance)],tellwidth=false)
    _sync_menu!(sections,section)
    previous=Button(content[2,1];label="previous",tellwidth=false)
    page=Observable(1)
    next=Button(content[2,3];label="next",tellwidth=false)
    function report_text()
        try
            comparison_summary(cc;section=section[])
        catch err
            "Report inspection failed: $(Controllers._errmsg(err))"
        end
    end
    pages=Observable(_comparison_pages(report_text()))
    page_label=lift(page,pages) do i,p
        "page $(clamp(i,1,length(p))) / $(length(p))"
    end
    Label(content[2,2],page_label)
    text=lift(page,pages) do i,p
        p[clamp(i,1,length(p))]
    end
    Label(content[3,1:3],text;halign=:left,valign=:top,justification=:left,
        fontsize=13,font=GLMakie.Makie.assetpath("fonts","DejaVuSansMono.ttf"),tellwidth=false,tellheight=false)
    function refresh!()
        pages[]=_comparison_pages(report_text())
        page[]=1
    end
    onany((args...)->refresh!(),cc.before,cc.after,cc.pair_indices,cc.basis,
        cc.allow_environment_change,cc.report,cc.report_origin,section)
    on(_->(page[]=max(1,page[]-1)),previous.clicks)
    on(_->(page[]=min(length(pages[]),page[]+1)),next.clicks)
    guarded=action->begin
        try
            Controllers._comparison_idle(cc)
            action()
        catch err
            cc.status[]="Action failed: $(Controllers._errmsg(err))"
        end
    end
    function apply_pairs!()
        a=tryparse(Int,something(before_box.stored_string[],""))
        b=tryparse(Int,something(after_box.stored_string[],""))
        a===nothing || b===nothing ? throw(ArgumentError("pair indices must be integers")) : set_comparison_pairs!(cc,a,b)
    end
    update_pairs!()=guarded(apply_pairs!)
    on(_->update_pairs!(),before_box.stored_string)
    on(_->update_pairs!(),after_box.stored_string)
    on(cc.pair_indices) do pair
        string(pair[1])==before_box.stored_string[] || (before_box.stored_string[]=string(pair[1]))
        string(pair[2])==after_box.stored_string[] || (after_box.stored_string[]=string(pair[2]))
    end
    for (button,side) in ((before_btn,:before),(after_btn,:after))
        on(button.clicks) do _
            guarded() do
                path=pick_file(;filterlist="jld2")
                isempty(path) || open_comparison_record!(cc,side,path)
            end
        end
    end
    on(run_btn.clicks) do _
        guarded() do
            apply_pairs!() # invalid visible input must never rerun the old indices
            compare!(cc)
            section[]=:summary
        end
    end
    on(load_btn.clicks) do _
        guarded() do
            path=pick_file(;filterlist="toml")
            isempty(path) && return
            open_comparison_report!(cc,path)
            section[]=:summary
        end
    end
    on(save_btn.clicks) do _
        guarded() do
            path=save_file(;filterlist="toml")
            isempty(path) || save_comparison_report!(cc,path)
        end
    end
    gl
end
