"""
    result_quality_report(ex::ResultExplorer; size=(900,650), report_path_picker=...) -> Figure

Open a separate whole-file report window for a lazy native explorer. All options
start unchecked. Generate a detached scalar report or generate/save TOML; ensemble
observations opt into core v4 without coercing a saved planar recipe association.
The complete sorted source index is checked and detached when this window opens
and for each request. Options are captured before
notifications or a save dialog. Failure retains the previous report with its own
file digest and provenance. Text is paged; changing frames in the explorer does
not relabel the report. Scanning is synchronous and may pause the event loop.
`report_path_picker` is an optional function returning a path or `nothing`.
"""
function result_quality_report(ex::ResultExplorer;size=(900,650),
        report_path_picker::Function=()->save_file(;filterlist="toml"))
    controller=Controllers._ResultQualityController(ex)
    fig=Figure(;size)
    Label(fig[1,1:3],"Whole-file quality report (native file; no experiment association)";halign=:left,fontsize=18)
    options=GridLayout(fig[2,1:3])
    for (row,label,setting) in ((1,"recorded final-sweep history",controller.history),
            (2,"recorded planar/stereo iteration observations",controller.execution),
            (3,"recorded ensemble pooling observations (format 4)",controller.ensemble))
        toggle=Toggle(options[row,1];active=setting[],halign=:left)
        Label(options[row,2],label;halign=:left,tellwidth=false)
        on(toggle.active) do value
            setting[]=value
        end
    end
    generate=Button(fig[3,1];label="generate report")
    save=Button(fig[3,2];label="generate / save TOML")
    status_preview=lift(controller.status) do text
        chars=collect(first(split(text,'\n')))
        length(chars)>90 ? String(chars[1:90])*"... (full status in report pages)" : String(chars)
    end
    Label(fig[4,1:3],status_preview;halign=:left,word_wrap=true,tellwidth=false)
    mono=GLMakie.Makie.assetpath("fonts","DejaVuSansMono.ttf")
    body=Label(fig[5,1:3],"";halign=:left,valign=:top,justification=:left,fontsize=13,font=mono,
        tellwidth=false,tellheight=false)
    previous=Button(fig[6,1];label="previous report page")
    page_label=Label(fig[6,2],"")
    next=Button(fig[6,3];label="next report page")
    Label(fig[7,1:3],"Counts describe recorded observations; no stationarity, independent-sample, accuracy or uncertainty-coverage claim.";
        halign=:left,word_wrap=true,tellwidth=false,fontsize=12)
    for column in 1:3
        colsize!(fig.layout,column,Relative(1/3))
    end
    pages=Ref(String[]);page=Observable(1);refreshing=Ref(false)
    function show_page!()
        isempty(pages[]) && return
        page.val=clamp(page[],1,length(pages[]))
        text=pages[][page[]]
        body.text[]==text || (body.text[]=text)
        label="report page $(page[]) / $(length(pages[]))"
        page_label.text[]==label || (page_label.text[]=label)
    end
    function refresh_pages!()
        refreshing[] && return
        refreshing[]=true
        try
            # Use allocated space, not the current page's intrinsic glyph box.
            # Otherwise a short initial page feeds a permanently tiny capacity.
            box=body.layoutobservables.suggestedbbox[]
            columns=max(1,floor(Int,(box.widths[1]-8)/8.1))
            lines=max(1,floor(Int,(box.widths[2]-8)/16.5))
            updated=_workflow_pages(Controllers._result_quality_text(controller),columns,lines)
            if updated!=pages[]
                pages[]=updated
                page.val=1
                show_page!()
            end
        finally
            refreshing[]=false
        end
    end
    on(_->show_page!(),page)
    on(_->(page[]=max(1,page[]-1)),previous.clicks)
    on(_->(page[]=min(length(pages[]),page[]+1)),next.clicks)
    on(_->refresh_pages!(),body.layoutobservables.suggestedbbox)
    onany((args...)->refresh_pages!(),controller.report,controller.status)
    on(generate.clicks) do _
        try Controllers._generate_result_quality!(controller) catch end
    end
    on(save.clicks) do _
        try Controllers._generate_result_quality!(controller;path_picker=report_path_picker) catch end
    end
    refresh_pages!()
    fig
end
