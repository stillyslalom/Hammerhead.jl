using Test, Hammerhead, TOML
using FileIO: save
using ImageCore: Gray, N0f16
include(joinpath(@__DIR__,"..","bench","validation_ptv_tracking.jl"))
const PTVC=ValidationPTVTracking

# Enumerate ALL partial injections, independently of potentials/dummy penalties.
function ptvc_objective(cost,allowed)
    n,m=size(cost);best=(-1,Inf)
    function visit(i,used,nmatches,total)
        if i>n
            if nmatches>best[1] || nmatches==best[1] && total<best[2];best=(nmatches,total);end
            return
        end
        visit(i+1,used,nmatches,total)
        for j in 1:m
            if allowed[i,j] && !(j in used)
                push!(used,j);visit(i+1,used,nmatches+1,total+cost[i,j]);delete!(used,j)
            end
        end
    end
    visit(1,Set{Int}(),0,0.);best
end
function ptvc_clip(rows,n;shape=(64,64))
    images=[zeros(shape) for _ in 1:n]
    metadata=Dict{String,Any}("frames"=>[Dict("ordinal"=>k,"decoded_sha256"=>PTVC.V.pixel_digest(im)) for (k,im) in enumerate(images)],
        "truth_sha256"=>Hammerhead._experiment_digest(rows),"annotation_kind"=>"controlled_metric_fixture")
    PTVC.AnnotatedClip("metric fixture",images,rows,metadata,String[])
end
function ptvc_particles(c)
    [begin
        rows=PTVC.visible_truth(c,k)
        Particles([r["x"] for r in rows],[r["y"] for r in rows],ones(length(rows)),fill(3.,length(rows)))
    end for k in eachindex(c.images)]
end
function ptvc_pair(a,b,ia,ib;flags=falses(length(ia)))
    PTVResult(a.x[ia],a.y[ia],b.x[ib].-a.x[ia],b.y[ib].-a.y[ia],zeros(length(ia)),BitVector(flags),ia,ib,a,b,PTVParameters())
end
function ptvc_track(c,paths)
    map=Dict((r["id"],r["frame"])=>r for r in c.rows)
    ts=Trajectory{Float64}[]
    for path in paths
        rs=[map[(id,f)] for (id,f) in path]
        push!(ts,Trajectory{Float64}(first(rs)["frame"],[r["x"] for r in rs],[r["y"] for r in rs],[r["frame"] for r in rs]))
    end
    TrackingResult(ts,length(c.images),PTVParameters())
end
# Independent character-state CSV reader for round-trip tests; quoted newline
# and escaped quotes are parsed, rather than splitting scientific IDs on commas.
function ptvc_csv(text)
    rows=Vector{String}[];row=String[];cell=IOBuffer();quoted=false;chars=collect(text);i=1
    while i<=length(chars)
        c=chars[i]
        if c=='"'
            if quoted && i<length(chars) && chars[i+1]=='"';write(cell,'"');i+=1
            else;quoted=!quoted;end
        elseif !quoted && c in (',','\n')
            push!(row,String(take!(cell)))
            if c=='\n';push!(rows,row);row=String[];end
        else;write(cell,c);end
        i+=1
    end
    quoted && error("unterminated quote")
    !isempty(row) && (push!(row,String(take!(cell)));push!(rows,row))
    rows
end

@testset "Independent exhaustive cardinality/cost assignment" begin
    for (n,m) in ((0,0),(0,3),(3,0),(1,3),(3,1),(2,2),(2,3),(3,2),(3,3))
        cost=[Float64((i-j)^2+(i+j)/32) for i in 1:n,j in 1:m]
        for mask in 0:(2^(n*m)-1)
            allowed=BitMatrix(reshape([!iszero(mask & (1<<(k-1))) for k in 1:n*m],n,m))
            expected=ptvc_objective(cost,allowed);actual=PTVC.assignment(cost,allowed)
            @test count(!=(0),actual)==expected[1]
            @test sum(cost[i,actual[i]] for i in 1:n if actual[i]!=0;init=0.)≈expected[2] atol=1e-12
            @test length(unique(filter(!=(0),actual)))==expected[1]
        end
    end
    # The production greedy cost order would take (1,1), preventing cardinality2.
    @test PTVC.assignment([0. .1;.2 0.],BitMatrix([1 1;1 0]))==[2,1]
    @test_throws ArgumentError PTVC.assignment([Inf;;],trues(1,1))
    @test_throws ArgumentError PTVC.assignment([-1.;;],trues(1,1))
    @test_throws ArgumentError PTVC.assignment([floatmax(Float64);;],trues(1,1))
    @test_throws ArgumentError PTVC.assignment(zeros(2,2),falses(1,1))
end

@testset "All detection/identity populations and nuisance ambiguity" begin
    truth=[PTVC.truth_row("a",1,10.,10.),PTVC.truth_row("b",1,20.,20.)]
    a=PTVC.associate([10.,10.2,20.,40.,NaN],[10.,10.,20.,40.,0.],truth)
    @test a.matched==2
    @test [l["status"] for l in a.labels]==["ambiguous","ambiguous","unique","unmapped","unmapped"]
    @test a.truth_ambiguous==[true,false]
    @test a.labels[3]["id"]=="b"
    close=PTVC.associate([10.3],[10.],[truth[1],PTVC.truth_row("b",1,10.6,10.)])
    @test close.labels[1]["status"]=="ambiguous"
    @test length(close.labels[1]["candidate_ids"])==2
    nuisance=PTVC.associate([10.1],[10.],[truth[1]];nuisance=[PTVC.truth_row("clutter",1,10.2,10.;role="nuisance")])
    @test nuisance.labels[1]["status"]=="ambiguous"
    @test nuisance.truth_ambiguous==[true]
    @test nuisance.matched==1
    @test nuisance.labels[1]["candidate_nuisance_ids"]==["clutter"]
    @test_throws ArgumentError PTVC.associate([1.],Float64[],truth)
    for g in (0.,NaN,Inf);@test_throws ArgumentError PTVC.associate([1.],[1.],truth;gate=g);end
    @test PTVC.edge_class(nuisance.labels[1],Dict("status"=>"unique","id"=>"a"))=="ambiguous"
    @test PTVC.edge_class(nuisance.labels[1],PTVC.empty_label())=="unmapped"
    @test PTVC.bounds(2,3,10)["strict_lower"]["fraction"]==.2
    @test PTVC.bounds(2,3,10)["conservative_upper"]["fraction"]==.5
    @test !PTVC.bounds(0,0,0)["strict_lower"]["available"]
    rows=[PTVC.truth_row(id,k,x,10.) for k in 1:2 for (id,x) in (("a",10.),("b",20.))]
    for k in 1:2
        push!(rows,PTVC.truth_row("hidden",k,30.,10.;visible=false,reason="scheduled_absence"))
        push!(rows,PTVC.truth_row("clutter",k,10.2,10.;role="nuisance"))
    end
    c=ptvc_clip(rows,2);p=ptvc_particles(c)
    d=PTVC.detection_metrics(c,p)
    @test d.metrics["counts"]["visible_target_observations"]==4
    @test d.metrics["counts"]["localized_tp"]==4
    @test d.metrics["counts"]["ambiguous_predictions"]==2
    @test d.metrics["counts"]["scheduled_absence_observations"]==2
    @test d.metrics["counts"]["rendered_nuisance_observations"]==2
    pairs=[ptvc_pair(p[1],p[2],[1,2],[1,2];flags=BitVector([0,1]))]
    score=PTVC.correspondence_metrics(c,pairs,p,d.assoc)
    @test score["raw"]["counts"]["confirmed_correct"]==1
    @test score["raw"]["counts"]["ambiguous"]==1
    @test score["accepted"]["counts"]["predictions"]==1
    @test score["accepted"]["precision_bounds"]["strict_lower"]["fraction"]==0.
    @test score["accepted"]["precision_bounds"]["conservative_upper"]["fraction"]==1.
    @test score["raw"]["counts"]["full_truth_pairs"]==2
    @test score["raw"]["counts"]["both_localized_truth_pairs"]==2
    duplicate=ptvc_pair(p[1],p[2],[1,1],[1,2])
    @test_throws ArgumentError PTVC.correspondence_metrics(c,[duplicate],p,d.assoc)
    malformed=ptvc_pair(p[1],p[2],[1],[1]);empty!(malformed.v)
    @test_throws ArgumentError PTVC.correspondence_metrics(c,[malformed],p,d.assoc)
    bad=ptvc_pair(p[1],p[2],[1],[1]);bad.index_b[1]=20
    @test_throws ArgumentError PTVC.correspondence_metrics(c,[bad],p,d.assoc)
    wrong=PTVC.correspondence_metrics(c,[ptvc_pair(p[1],p[2],[2],[1])],p,d.assoc)
    @test wrong["raw"]["counts"]["ambiguous"]==1 # not known wrong, because b endpoint could be ambiguous a
end

@testset "Exact returned-ID switches, fragmentation and scheduled gap events" begin
    rows=[PTVC.truth_row(id,k,x+.2k,10.) for k in 1:6 for (id,x) in (("a",10.),("b",30.))]
    c=ptvc_clip(rows,6);p=ptvc_particles(c);d=PTVC.detection_metrics(c,p)
    # One truth ID splits into two output IDs, with an intervening visible miss.
    split=ptvc_track(c,[[ ("a",1),("a",2) ],[ ("a",4),("a",5),("a",6) ],[("b",k) for k in 1:6]])
    s=PTVC.tracking_metrics(c,split,p,d.assoc).metrics
    @test s["counts"]["extra_confirmed_track_ids_fragmentation"]==1
    @test s["counts"]["confirmed_target_track_id_changes"]==1
    @test s["counts"]["tracked_untracked_tracked_episodes"]==1
    @test s["counts"]["observed_same_track_identity_changes"]==0
    @test s["counts"]["detections_not_in_returned_tracks"]==1
    @test s["counts"]["confirmed_retained_truth_observations"]==11
    switched=ptvc_track(c,[[("a",1),("a",2),("b",3),("b",4)],[("b",1),("b",2),("a",3),("a",4)]])
    s=PTVC.tracking_metrics(c,switched,p,d.assoc).metrics
    @test s["counts"]["observed_same_track_identity_changes"]==2
    @test s["counts"]["confirmed_wrong"]==2
    @test s["counts"]["confirmed_target_track_id_changes"]==2
    duplicate=ptvc_track(c,[[ ("a",1),("a",2) ],[ ("a",1),("a",3) ]])
    @test_throws ArgumentError PTVC.tracking_metrics(c,duplicate,p,d.assoc)
    for r in rows
        if r["id"]=="a" && r["frame"] in (3,4);r["visible"]=false;r["visibility_reason"]="scheduled_absence";end
    end
    g=ptvc_clip(rows,6);p=ptvc_particles(g);d=PTVC.detection_metrics(g,p)
    bridged=ptvc_track(g,[[ ("a",1),("a",2),("a",5),("a",6) ],[("b",k) for k in 1:6]])
    s=PTVC.tracking_metrics(g,bridged,p,d.assoc).metrics;event=only(s["gap_events"])
    @test event["missed_frames"]==2
    @test event["confirmed_recovered"]===true
    @test event["two_prior_visible_samples"]===true
    @test event["two_prior_confirmed_same_track_samples"]===true
    @test s["bridges"]["counts"]["correct_scheduled_absence"]==1
    @test s["gap_recovery"][2]["full_recovery_bounds"]["strict_lower"]["fraction"]==1.
    ended=ptvc_track(g,[[ ("a",1),("a",2) ],[ ("a",5),("a",6) ]])
    se=PTVC.tracking_metrics(g,ended,p,d.assoc).metrics
    @test !only(se["gap_events"])["confirmed_recovered"]
    @test se["counts"]["extra_confirmed_track_ids_fragmentation"]==1
    # Provided real/independent annotations can assert absence without claiming
    # that the benchmark deliberately removed the particle from its renderer.
    annotated=deepcopy(g)
    for r in annotated.rows;r["visibility_reason"]=="scheduled_absence" && (r["visibility_reason"]="annotated_absence");end
    sa=PTVC.tracking_metrics(annotated,bridged,p,d.assoc).metrics
    @test only(sa["gap_events"])["absence_kind"]=="annotated"
    @test sa["gap_recovery"][2]["counts"]["annotated_truth_events"]==1
    @test sa["gap_recovery"][2]["counts"]["scheduled_truth_events"]==0
    missing=deepcopy(p);missing[5]=Particles(Float64[],Float64[],Float64[],Float64[])
    dm=PTVC.detection_metrics(g,missing)
    sm=PTVC.tracking_metrics(g,ended,missing,dm.assoc).metrics
    @test sm["gap_recovery"][2]["counts"]["truth_events"]==1
    @test sm["gap_recovery"][2]["counts"]["both_endpoints_localized"]==0
    @test !sm["gap_recovery"][2]["both_localized_recovery_bounds"]["strict_lower"]["available"]
    # The same returned jump across VISIBLE truth is a missed-detection/retention
    # bridge, not a deliberately invisible gap success; full recall stays lower.
    visible=ptvc_clip([PTVC.truth_row("a",k,10.0 + .2k,10.) for k in 1:6],6)
    pp=ptvc_particles(visible);dd=PTVC.detection_metrics(visible,pp)
    jumped=ptvc_track(visible,[[ ("a",1),("a",2),("a",5),("a",6) ]])
    sj=PTVC.tracking_metrics(visible,jumped,pp,dd.assoc).metrics
    @test isempty(sj["gap_events"])
    @test sj["bridges"]["counts"]["correct_visible_detection_or_retention_miss"]==1
    @test sj["counts"]["confirmed_correct"]==3
    @test sj["counts"]["confirmed_truth_edges_recovered"]==2
    @test sj["full_edge_recall_bounds"]["strict_lower"]["fraction"]==.4
    # A missed visible detection is still a truth FN; it is not an absent row.
    pp[3]=Particles(Float64[],Float64[],Float64[],Float64[])
    dd=PTVC.detection_metrics(visible,pp)
    @test dd.metrics["counts"]["unlocalized_fn"]==1
    @test dd.metrics["counts"]["invisible_target_observations"]==0
    # Ambiguity interrupts target-centric memory; it cannot create a claimed ID
    # switch across an unresolved image or imply a correctly recovered gap.
    ambiguous=ptvc_clip([PTVC.truth_row(id,k,k==3 ? 10.3 : x+.2k,10.) for k in 1:6 for (id,x) in (("a",10.),("b",30.))],6)
    ap=ptvc_particles(ambiguous);ad=PTVC.detection_metrics(ambiguous,ap)
    at=ptvc_track(ambiguous,[[ ("a",1),("a",2) ],[ ("a",4),("a",5),("a",6) ]])
    am=PTVC.tracking_metrics(ambiguous,at,ap,ad.assoc).metrics
    @test am["counts"]["confirmed_target_track_id_changes"]==0
    @test am["counts"]["ambiguous_visible_samples_resetting_identity_memory"]==2
    @test am["counts"]["extra_confirmed_track_ids_fragmentation"]==1 # distinct CONFIRMED IDs still visible; not a standard MOT score
end

@testset "Annotation/schema/image identity refusals and explicit relocation" begin
    c=PTVC.synthetic_clip(7321,"one_frame_gap";size=64,nframes=5)
    @test c.metadata["truth_sha256"]==Hammerhead._experiment_digest(c.rows)
    @test PTVC.checked_clip(c)===c
    @test c.metadata["frames"]==PTVC.synthetic_clip(7321,"one_frame_gap";size=64,nframes=5).metadata["frames"]
    @test_throws ArgumentError PTVC.synthetic_clip(true,"clean")
    @test_throws ArgumentError PTVC.validate_truth([1],5,(64,64))
    for mutate in (r->pop!(r),r->push!(r,deepcopy(first(r))),r->(r[1]["visible"]=1),r->(r[1]["x"]=NaN),r->(r[1]["frame"]=true),r->(r[1]["visibility_reason"]="maybe"))
        rows=deepcopy(c.rows);mutate(rows)
        @test_throws ArgumentError PTVC.validate_truth(rows,5,(64,64))
    end
    cc=deepcopy(c);cc.images[1][1]=1.
    @test_throws ArgumentError PTVC.checked_clip(cc)
    cc=deepcopy(c);cc.rows[1]["x"]+=.1
    @test_throws ArgumentError PTVC.checked_clip(cc)
    mktempdir() do dir
        frames=Dict{String,Any}[]
        for k in 1:2
            path=joinpath(dir,"frame$k.png");save(path,Gray{N0f16}.(zeros(8,8)))
            image=Hammerhead.load_frame(path,Float64)
            push!(frames,Dict("ordinal"=>k,"locator"=>basename(path),"bytes"=>filesize(path),
                "sha256"=>PTVC.V.file_digest(path),"decoded_sha256"=>PTVC.V.pixel_digest(image),"decoded_processing_type"=>"Float64"))
        end
        data=Dict{String,Any}("schema_version"=>PTVC.ANNOTATION_SCHEMA,"clip_id"=>"import fixture", "annotation_kind"=>"independent_synthetic",
            "source_uri"=>"https://example.invalid/fixture","citation"=>"schema test only","license"=>"test fixture",
            "coordinate_convention"=>"one_based_pixel_centers_x_columns_y_rows","length_unit"=>"px","image_shape"=>[8,8],
            "frames"=>frames,"truth"=>[PTVC.truth_row("quoted,\"id\"\nline",k,4.,4.) for k in 1:2])
        manifest=joinpath(dir,"clip.toml")
        write_data(d)=open(io->TOML.print(io,d;sorted=true),manifest,"w")
        write_data(data);loaded=PTVC.load_clip(manifest)
        @test loaded.metadata["annotation_kind"]=="independent_synthetic"
        @test loaded.rows==data["truth"]
        @test length(loaded.protected_paths)==3
        @test PTVC.checked_clip(loaded)===loaded
        unknown=deepcopy(data);delete!(unknown["truth"][1],"intensity")
        write_data(unknown);missing_intensity=PTVC.load_clip(manifest)
        @test missing_intensity.rows[1]["intensity"]=="unknown"
        @test !haskey(missing_intensity.metadata["imported_manifest"]["truth"][1],"intensity")
        exported=PTVC.evaluate_clip(missing_intensity;predictor=nothing,max_gaps=(0,))
        @test occursin("\"unknown\"",exported.artifacts["clip-001-truth.csv"])
        csv=ptvc_csv(exported.artifacts["clip-001-truth.csv"])
        @test csv[2][1]==missing_intensity.rows[1]["id"]
        @test csv[2][end]=="unknown"
        @test parse(Int,csv[2][2])==missing_intensity.rows[1]["frame"]
        @test parse(Float64,csv[2][3])==missing_intensity.rows[1]["x"]
        @test parse(Bool,csv[2][5])==missing_intensity.rows[1]["visible"]
        unknown["truth"]=deepcopy(missing_intensity.rows);write_data(unknown)
        @test PTVC.load_clip(manifest).rows==missing_intensity.rows
        foreign=deepcopy(data);foreign["frames"][1]["locator"]=Sys.iswindows() ? "/foreign/input.png" : "C:\\foreign\\input.png"
        write_data(foreign)
        @test_throws ArgumentError PTVC.load_clip(manifest)
        relocated=PTVC.load_clip(manifest;relocation=Dict(1=>joinpath(dir,"frame1.png")))
        @test relocated.metadata["frames"][1]["locator"]==foreign["frames"][1]["locator"]
        @test relocated.images==loaded.images
        for mutate in (d->(d["schema_version"]="unknown"),d->(d["frames"][1]["ordinal"]=true),d->(d["image_shape"]=[9,8]),d->(d["frames"][1]["bytes"]+=1),d->(d["frames"][1]["locator"]="../outside.png"),d->(d["frames"][1]["decoded_sha256"]=repeat("0",64)),d->pop!(d["truth"]),d->(d["frames"][1]["decoded_processing_type"]="Float32"))
            bad=deepcopy(data);mutate(bad);write_data(bad)
            @test_throws ArgumentError PTVC.load_clip(manifest)
        end
        write_data(data);loaded=PTVC.load_clip(manifest)
        write(joinpath(dir,"frame1.png"),"changed")
        @test_throws ArgumentError PTVC.checked_clip(loaded)
        @test_throws ArgumentError PTVC.load_clip(manifest)
        sentinel=joinpath(dir,"preserve.txt");write(sentinel,"preserve")
        @test_throws ArgumentError PTVC.main(["--manifest=$manifest","--output=$dir"])
        @test read(sentinel,String)=="preserve"
    end
    @test occursin("\"quoted,\"\"id\"\"\nline\"",PTVC.csv_rows(("id",),[("quoted,\"id\"\nline",)]))
    ids=["with;semicolon","with,comma","quote\"and\nline","α"]
    cs=ptvc_csv(PTVC.csv_rows(("candidate_ids",),[(PTVC.csv_ids(ids),)]))
    @test TOML.parse(cs[2][1])["ids"]==ids
end

@testset "Cheap production API image clips, exports and conserved pooling" begin
    clean=PTVC.synthetic_clip(7321,"clean";size=64,nframes=5)
    gap=PTVC.synthetic_clip(7321,"one_frame_gap";size=64,nframes=5)
    params=PTVParameters(uod_enable=false)
    bundle=PTVC.run_study(;clips=[clean,gap],params,predictor=nothing,max_gaps=(0,1))
    report=bundle.report
    @test report["environment"]["processing_threaded"]===true
    @test report["calls"]["ptv_pairs"]==8
    @test report["calls"]["tracking"]==4
    @test report["calls"]["matcher_transitions"]==24
    @test report["calls"]["piv_predictor_upper_bound"]==0
    @test report["calls"]["detector_invocations"]==46
    for row in report["groups"]
        dc=row["detection"]["counts"]
        @test dc["predictions"]==dc["localized_tp"]+dc["unassigned_fp"]
        @test dc["visible_target_observations"]==dc["localized_tp"]+dc["unlocalized_fn"]
        for name in ("raw","accepted")
            ct=row["correspondence"][name]["counts"]
            @test ct["predictions"]==sum(ct[k] for k in PTVC.CLASSES)
            @test ct["confirmed_correct"]<=ct["full_truth_pairs"]
        end
        for t in row["tracking"]
            ct=t["metrics"]["counts"]
            @test ct["predictions"]==sum(ct[k] for k in PTVC.CLASSES)
            @test ct["observations"]==ct["unique_observations"]+ct["ambiguous_observations"]+ct["unmapped_observations"]
            @test ct["detections"]==ct["retained_detection_observations"]+ct["detections_not_in_returned_tracks"]
        end
    end
    forced=report["groups"][2]
    no_bridge=forced["tracking"][1]["metrics"]["gap_recovery"][1]
    bridge=forced["tracking"][2]["metrics"]["gap_recovery"][1]
    @test no_bridge["counts"]["confirmed_recovered"]==0
    @test bridge["counts"]["confirmed_recovered"]==bridge["counts"]["truth_events"]==6
    @test report["pooled"]["detection"]["counts"]["predictions"]==sum(r["detection"]["counts"]["predictions"] for r in report["groups"])
    native=bundle.artifacts["clip-002-tracks-gap-1.csv"]
    @test startswith(native,"schema_version,result_type,")
    @test occursin("trajectory_id",first(split(native,'\n')))
    @test startswith(bundle.artifacts["clip-002-track-associations-gap-1.csv"],"trajectory_id,observation_index,")
    @test occursin("Full edge recall bounds",PTVC.markdown_report(report))
    @test_throws ArgumentError PTVC.evaluate_clip(clean;predictor=(x=[1.,64.],y=[1.,64.],u=ones(2,2),v=zeros(2,2)))
    @test_throws ArgumentError PTVC.evaluate_clip(clean;max_gaps=(0,0))
    @test_throws ArgumentError PTVC.evaluate_clip(clean;prefix="../unsafe")
    mktempdir() do dir
        destination=joinpath(dir,"score")
        # Source stability can legitimately be false during another agent's edits;
        # never relax writer's guard. Test the public publication protocol using a
        # stable snapshot flag independently of numerical/scientific evidence.
        frozen=deepcopy(bundle);frozen.report["environment"]=PTVC.environment_record();frozen.report["source_and_environment_stable"]=true
        paths=PTVC.write_report(destination,frozen)
        @test all(isfile,paths)
        reread=TOML.parsefile(first(paths))
        @test reread["pooled"]==report["pooled"]
        saved=read(first(paths),String)
        @test_throws ArgumentError PTVC.write_report(destination,frozen)
        @test read(first(paths),String)==saved
        @test_throws ArgumentError PTVC.check_output(joinpath(dir,"input");protected_paths=[joinpath(dir,"input")])
        if Sys.iswindows()
            @test_throws ArgumentError PTVC.check_output(joinpath(dir,"new-input");protected_paths=[joinpath(dir,"NEW-INPUT")])
            aliased=deepcopy(frozen);aliased.artifacts["PTV_TRACKING.TOML"]="alias"
            @test_throws ArgumentError PTVC.write_report(joinpath(dir,"case-alias"),aliased)
            @test !ispath(joinpath(dir,"case-alias"))
        end
        @test_throws ArgumentError PTVC.check_output(joinpath(PTVC.ROOT,"test","unsafe"))
        unstable=deepcopy(bundle);unstable.report["source_and_environment_stable"]=false
        @test_throws ArgumentError PTVC.write_report(joinpath(dir,"unstable"),unstable)
        @test !ispath(joinpath(dir,"unstable"))
        @test_throws ArgumentError PTVC.main(["--unknown"])
        blank=ptvc_clip(Dict{String,Any}[],2)
        empty=PTVC.evaluate_clip(blank;params,predictor=nothing,max_gaps=(0,))
        @test !empty.row["detection"]["recall"]["available"]
        @test empty.row["correspondence"]["raw"]["counts"]["predictions"]==0
        @test !empty.row["tracking"][1]["metrics"]["edge_precision_bounds"]["strict_lower"]["available"]
    end
end
