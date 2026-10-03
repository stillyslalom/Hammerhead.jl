# Bench-only annotated particle evaluation; no production/default changes.
module ValidationPTVTracking
using Hammerhead, Statistics, TOML, Dates
using Hammerhead.SyntheticData: generate_gaussian_particle!
include("validation_uncertainty.jl")
const U=ValidationUncertainty
const V=U.V
const ROOT=V.ROOT
const SCHEMA="hammerhead-annotated-ptv-tracking-1"
const ANNOTATION_SCHEMA="hammerhead-annotated-particle-clip-1"
const MARKER="Hammerhead annotated PTV/tracking scorecard"
const GATE=.75
const CONDITIONS=("clean","one_frame_gap","two_frame_gap","encounter_noise_clutter")
const SEEDS=(7321,7322)
const CLASSES=("confirmed_correct","confirmed_wrong","unmapped","ambiguous")
err(s)=throw(ArgumentError(s))
integer(x)=x isa Int && !(x isa Bool)
finite(x)=x isa Real && !(x isa Bool) && isfinite(x) && isfinite(Float64(x))
keys_exact(d,ks)=d isa AbstractDict && Set(keys(d))==Set(ks) || err("malformed annotation fields")
ratio(n,d)=Dict{String,Any}("numerator"=>n,"denominator"=>d,"available"=>d>0,
    d>0 ? ("fraction"=>n/d) : ("reason"=>"empty_denominator"))
function bounds(correct,ambiguous,denominator)
    Dict("strict_lower"=>ratio(correct,denominator),
        "conservative_upper"=>ratio(min(denominator,correct+ambiguous),denominator))
end

struct AnnotatedClip
    id::String
    images::Vector{Matrix{Float64}}
    rows::Vector{Dict{String,Any}}
    metadata::Dict{String,Any}
    protected_paths::Vector{String}
end

function truth_row(id,frame,x,y;visible=true,reason="visible",role="target",intensity=1.)
    Dict{String,Any}("id"=>string(id),"frame"=>frame,"x"=>Float64(x),"y"=>Float64(y),
        "visible"=>visible,"visibility_reason"=>reason,"role"=>role,"intensity"=>intensity===nothing ? "unknown" : Float64(intensity))
end
function validate_truth(rows,nframes,shape)
    integer(nframes) && nframes>=2 || err("at least two ordered frames required")
    length(shape)==2 && all(n->integer(n) && n>=4,shape) || err("invalid image shape")
    rows isa AbstractVector || err("annotation rows must be a vector")
    seen=Set{Tuple{String,Int}}();roles=Dict{String,String}()
    for r in rows
        r isa AbstractDict || err("truth row must be a mapping")
        mandatory=["id","frame","x","y","visible","visibility_reason","role"]
        keys_exact(r,haskey(r,"intensity") ? [mandatory;"intensity"] : mandatory)
        r["id"] isa String && !isempty(r["id"]) && !occursin('\0',r["id"]) || err("invalid truth ID")
        integer(r["frame"]) && 1<=r["frame"]<=nframes || err("invalid truth frame")
        all(k->finite(r[k]),("x","y")) || err("nonfinite truth coordinates")
        intensity=get(r,"intensity","unknown")
        intensity=="unknown" || finite(intensity) && intensity>=0 || err("invalid known intensity")
        r["visible"] isa Bool || err("visibility must be Boolean")
        r["role"] in ("target","nuisance") || err("invalid annotation role")
        r["visibility_reason"] in ("visible","scheduled_absence","outside_image","annotated_absence") || err("invalid visibility reason")
        r["visible"] == (r["visibility_reason"]=="visible") || err("inconsistent visibility reason")
        inside=1<=r["x"]<=shape[2] && 1<=r["y"]<=shape[1]
        r["visible"] && !inside && err("visible truth must lie inside image")
        r["visibility_reason"]=="outside_image" && inside && err("outside-image truth lies inside")
        pair=(r["id"],r["frame"]);pair in seen && err("duplicate ID/frame annotation");push!(seen,pair)
        get!(roles,r["id"],r["role"])==r["role"] || err("role changed within identity")
    end
    length(seen)==length(roles)*nframes || err("every ID needs an explicit row/status for every selected frame")
    nothing
end
function checked_clip(c::AnnotatedClip)
    isempty(c.id) && err("empty clip ID")
    length(c.images)>=2 && all(i->size(i)==size(first(c.images)) && all(isfinite,i),c.images) || err("invalid images")
    validate_truth(c.rows,length(c.images),size(first(c.images)))
    length(c.metadata["frames"])==length(c.images) || err("image identity count mismatch")
    for (k,img) in enumerate(c.images)
        f=c.metadata["frames"][k]
        f["ordinal"]===k && f["decoded_sha256"]==V.pixel_digest(img) || err("changed image pixels/ordering")
    end
    c.metadata["truth_sha256"]==Hammerhead._experiment_digest(c.rows) || err("changed truth annotations")
    for f in get(c.metadata,"file_identities",Any[])
        isfile(f["path"]) && filesize(f["path"])==f["bytes"] && V.file_digest(f["path"])==f["sha256"] || err("changed input/manifest file")
    end
    c
end

# Rectangular Hungarian shortest-augmenting-path assignment. Dummy columns cost
# more than every possible real-edge cost combined: cardinality precedes cost.
# This implementation does not call the production greedy matcher/cell list.
function assignment(cost::Matrix{Float64},allowed::BitMatrix)
    size(cost)==size(allowed) || err("assignment shape mismatch")
    n,m=size(cost);n==0 && return Int[]
    all(i->!allowed[i] || isfinite(cost[i]) && cost[i]>=0,eachindex(cost)) || err("invalid assignment cost")
    largest=isempty(cost) ? 0. : maximum(cost[allowed];init=0.)
    penalty=(n+1)*(largest+1.);forbidden=(n+1)*penalty
    isfinite(forbidden) || err("assignment penalty overflow")
    nc=m+n; a=fill(penalty,n,nc)
    for j in 1:m,i in 1:n;a[i,j]=allowed[i,j] ? cost[i,j] : forbidden;end
    u=zeros(n);v=zeros(nc+1);p=zeros(Int,nc+1);way=zeros(Int,nc+1)
    for i in 1:n
        p[1]=i;j0=1;minv=fill(Inf,nc+1);used=falses(nc+1)
        while true
            used[j0]=true;i0=p[j0];delta=Inf;j1=0
            for j in 2:nc+1
                used[j] && continue
                cur=a[i0,j-1]-u[i0]-v[j]
                if cur<minv[j];minv[j]=cur;way[j]=j0;end
                if minv[j]<delta;delta=minv[j];j1=j;end
            end
            isfinite(delta) && j1!=0 || err("assignment arithmetic unavailable")
            for j in 1:nc+1
                if used[j];u[p[j]]+=delta;v[j]-=delta;else;minv[j]-=delta;end
            end
            j0=j1;p[j0]==0 && break
        end
        while true
            j1=way[j0];p[j0]=p[j1];j0=j1;j0==1 && break
        end
    end
    out=zeros(Int,n)
    for j in 2:m+1;p[j]==0 || (allowed[p[j],j-1] && (out[p[j]]=j-1));end
    out
end

function associate(x,y,truth;gate=GATE,nuisance=Dict{String,Any}[])
    length(x)==length(y) || err("coordinate length mismatch")
    finite(gate) && gate>0 && isfinite(gate^2) || err("invalid localization gate")
    n,m=length(x),length(truth);cost=zeros(n,m);allowed=falses(n,m)
    for j in 1:m,i in 1:n
        if finite(x[i]) && finite(y[i])
            d=hypot(x[i]-truth[j]["x"],y[i]-truth[j]["y"])
            if isfinite(d) && d<=gate;allowed[i,j]=true;cost[i,j]=d^2;end
        end
    end
    map=assignment(cost,allowed);nd=vec(sum(allowed;dims=2));nt=vec(sum(allowed;dims=1))
    # Conservatively ambiguous anywhere with competing in-gate detections/IDs;
    # optimization uniqueness does not identify contributors to merged intensity.
    nuisance_ids=[[r["id"] for r in nuisance if finite(x[i]) && finite(y[i]) && hypot(x[i]-r["x"],y[i]-r["y"])<=gate] for i in 1:n]
    ambiguous=[nd[i]>0 && (nd[i]>1 || !isempty(nuisance_ids[i]) || any(j->nt[j]>1,findall(view(allowed,i,:)))) for i in 1:n]
    labels=[Dict{String,Any}("status"=>ambiguous[i] ? "ambiguous" : map[i]==0 ? "unmapped" : "unique",
        "id"=>map[i]==0 ? "" : truth[map[i]]["id"],"truth_index"=>map[i],
        "candidate_ids"=>[truth[j]["id"] for j in findall(view(allowed,i,:))],
        "candidate_nuisance_ids"=>nuisance_ids[i]) for i in 1:n]
    truth_ambiguous=[nt[j]>1 || any(i->nd[i]>1 || !isempty(nuisance_ids[i]),findall(view(allowed,:,j))) for j in 1:m]
    (;labels,map,truth_ambiguous,matched=count(!=(0),map),cost=sum(cost[i,map[i]] for i in 1:n if map[i]!=0;init=0.))
end
visible_truth(c,k)=sort!([r for r in c.rows if r["frame"]==k && r["role"]=="target" && r["visible"]];by=r->r["id"])
function edge_class(a,b)
    (a["status"]=="unmapped" || b["status"]=="unmapped") && return "unmapped"
    (a["status"]=="ambiguous" || b["status"]=="ambiguous") && return "ambiguous"
    a["id"]==b["id"] ? "confirmed_correct" : "confirmed_wrong"
end
empty_label()=Dict{String,Any}("status"=>"unmapped","id"=>"","truth_index"=>0,"candidate_ids"=>String[],"candidate_nuisance_ids"=>String[])
counts()=Dict(k=>0 for k in CLASSES)
function detection_metrics(c,particles;gate=GATE)
    length(particles)==length(c.images) || err("detection frame count mismatch")
    assoc=[associate(p.x,p.y,visible_truth(c,k);gate,nuisance=[r for r in c.rows if r["frame"]==k && r["role"]=="nuisance" && r["visible"]]) for (k,p) in enumerate(particles)]
    ntruth=sum(length(visible_truth(c,k)) for k in eachindex(c.images))
    n=sum(length(p) for p in particles);tp=sum(a.matched for a in assoc)
    ct=Dict("visible_target_observations"=>ntruth,"predictions"=>n,"localized_tp"=>tp,
        "unassigned_fp"=>n-tp,"unlocalized_fn"=>ntruth-tp,
        "ambiguous_predictions"=>sum(count(l->l["status"]=="ambiguous",a.labels) for a in assoc),
        "invisible_target_observations"=>count(r->r["role"]=="target" && !r["visible"],c.rows),
        "scheduled_absence_observations"=>count(r->r["role"]=="target" && r["visibility_reason"]=="scheduled_absence",c.rows),
        "outside_image_observations"=>count(r->r["role"]=="target" && r["visibility_reason"]=="outside_image",c.rows),
        "rendered_nuisance_observations"=>count(r->r["role"]=="nuisance" && r["visible"],c.rows))
    (;assoc,metrics=Dict("counts"=>ct,"precision"=>ratio(tp,n),"recall"=>ratio(tp,ntruth)))
end

function correspondence_metrics(c,results,particles,assoc)
    length(results)==length(c.images)-1 || err("pair result count mismatch")
    raw=counts();accepted=counts();full=0;conditional=0
    errors=(u=U.Moment(),v=U.Moment());accepted_errors=(u=U.Moment(),v=U.Moment())
    for (k,r) in enumerate(results)
        n=length(r.index_a)
        all(a->length(a)==n,(r.x,r.y,r.u,r.v,r.index_b,r.match_residual,r.outliers)) || err("malformed correspondence arrays")
        length(unique(r.index_a))==n && length(unique(r.index_b))==n || err("duplicate correspondence endpoint indices")
        all(i->1<=i<=length(r.particles_a),r.index_a) && all(i->1<=i<=length(r.particles_b),r.index_b) || err("correspondence endpoint out of bounds")
        for (p,q) in ((r.particles_a,particles[k]),(r.particles_b,particles[k+1]))
            all(f->isequal(getfield(p,f),getfield(q,f)),fieldnames(typeof(p))) || err("repeated detector output differs from cached observations")
        end
        a=Dict(r["id"]=>r for r in visible_truth(c,k));b=Dict(r["id"]=>r for r in visible_truth(c,k+1))
        full+=length(intersect(keys(a),keys(b)))
        mapped(a)=Set(l["id"] for l in a.labels if l["truth_index"]!=0)
        conditional+=length(intersect(mapped(assoc[k]),mapped(assoc[k+1])))
        for m in eachindex(r.index_a)
            ia,ib=r.index_a[m],r.index_b[m]
            checkbounds(assoc[k].labels,ia);checkbounds(assoc[k+1].labels,ib)
            la,lb=assoc[k].labels[ia],assoc[k+1].labels[ib];cl=edge_class(la,lb)
            raw[cl]+=1;r.outliers[m] || (accepted[cl]+=1)
            if cl=="confirmed_correct"
                ra,rb=a[la["id"]],b[lb["id"]]
                du=r.u[m]-(rb["x"]-ra["x"]);dv=r.v[m]-(rb["y"]-ra["y"])
                U.add!(errors.u,du);U.add!(errors.v,dv)
                if !r.outliers[m];U.add!(accepted_errors.u,du);U.add!(accepted_errors.v,dv);end
            end
        end
    end
    populations=Dict{String,Any}()
    for (name,ct,e) in (("raw",raw,errors),("accepted",accepted,accepted_errors))
        n=sum(values(ct))
        populations[name]=Dict("counts"=>merge(ct,Dict("predictions"=>n,"full_truth_pairs"=>full,"both_localized_truth_pairs"=>conditional)),
            "precision_bounds"=>bounds(ct["confirmed_correct"],ct["ambiguous"],n),
            "full_recall_bounds"=>bounds(ct["confirmed_correct"],ct["ambiguous"],full),
            "both_localized_recall_bounds"=>bounds(ct["confirmed_correct"],ct["ambiguous"],conditional),
            "correct_pair_displacement_error"=>Dict("u"=>U.summary(e.u),"v"=>U.summary(e.v)))
    end
    populations
end

function tracking_metrics(c,result::TrackingResult,particles,assoc)
    result.n_frames==length(c.images) || err("tracking frame count mismatch")
    labels=Dict{Tuple{Int,Int},Dict{String,Any}}();observations=Dict{String,Any}[]
    used=Set{Tuple{Int,Int}}();confirmed=Dict{Tuple{String,Int},Int}();edges=NamedTuple[]
    ct=merge(counts(),Dict("predictions"=>0,"observations"=>0,"unique_observations"=>0,
        "ambiguous_observations"=>0,"unmapped_observations"=>0,"observed_same_track_identity_changes"=>0))
    for (tid,t) in enumerate(result.trajectories)
        length(t.x)==length(t.y)==length(t.frames) && length(t)>=2 || err("malformed/minimum-length tracking output")
        t.start_frame==first(t.frames) && all(k->1<=k<=result.n_frames,t.frames) && all(>(0),diff(t.frames)) || err("malformed trajectory frames")
        for k in eachindex(t.frames)
            f=t.frames[k];p=particles[f]
            j=findfirst(i->isequal(p.x[i],t.x[k]) && isequal(p.y[i],t.y[k]),eachindex(p.x))
            if j!==nothing
                (f,j) in used && err("duplicate returned detection observation")
                push!(used,(f,j))
            end
            l=j===nothing ? empty_label() : assoc[f].labels[j];labels[(tid,k)]=l
            ct["observations"]+=1;ct[l["status"]*"_observations"]+=1
            l["status"]=="unique" && (confirmed[(l["id"],f)]=tid)
            push!(observations,Dict("trajectory_id"=>tid,"observation_index"=>k,"frame"=>f,
                "x"=>Float64(t.x[k]),"y"=>Float64(t.y[k]),"detection_index"=>something(j,0),
                "identity_status"=>l["status"],"confirmed_truth_id"=>l["status"]=="unique" ? l["id"] : "",
                "candidate_truth_ids"=>copy(l["candidate_ids"]),"candidate_nuisance_ids"=>copy(l["candidate_nuisance_ids"])))
            if k>1
                a=labels[(tid,k-1)];cl=edge_class(a,l)
                ct["predictions"]+=1;ct[cl]+=1
                cl=="confirmed_wrong" && (ct["observed_same_track_identity_changes"]+=1)
                push!(edges,(trajectory_id=tid,frame_a=t.frames[k-1],frame_b=f,class=cl,
                    id=cl=="confirmed_correct" ? l["id"] : ""))
            end
        end
    end
    ct["detections"]=sum(length(p) for p in particles)
    ct["retained_detection_observations"]=length(used)
    ct["detections_not_in_returned_tracks"]=ct["detections"]-length(used)
    ct["visible_truth_observations"]=sum(length(visible_truth(c,k)) for k in eachindex(c.images))
    ct["confirmed_retained_truth_observations"]=length(confirmed)
    ct["visible_truth_not_confirmed_retained"]=ct["visible_truth_observations"]-length(confirmed)
    rowmap=Dict((r["id"],r["frame"])=>r for r in c.rows if r["role"]=="target")
    ids=sort!(unique(r["id"] for r in c.rows if r["role"]=="target"))
    localized=[Set(l["id"] for l in a.labels if l["truth_index"]!=0) for a in assoc]
    ambiguous=[Set(truth[j]["id"] for j in eachindex(truth) if assoc[k].truth_ambiguous[j])
        for (k,truth) in enumerate([visible_truth(c,k) for k in eachindex(c.images)])]
    edge_set=Set((e.id,e.frame_a,e.frame_b) for e in edges if e.class=="confirmed_correct")
    identity_rows=Dict{String,Any}[];gap_rows=Dict{String,Any}[]
    # All adjacent visible samples of an ID define truth edges, including forced
    # gaps. No frames/positions are synthesized into a returned trajectory.
    edge_denominator=0;localized_edges=0;switches=0;skipped=0;fragments=0;episodes=0
    for id in ids
        visible=[k for k in eachindex(c.images) if rowmap[(id,k)]["visible"]]
        track_ids=Set{Int}();previous=0;seen=false;broken=false;ns=0;ne=0;na=0
        for k in visible
            if id in ambiguous[k]
                na+=1;previous=0;seen=false;broken=false
                continue
            end
            tid=get(confirmed,(id,k),0)
            if tid==0
                broken=seen
            else
                push!(track_ids,tid)
                previous!=0 && previous!=tid && (ns+=1)
                seen && broken && (ne+=1)
                previous=tid;seen=true;broken=false
            end
        end
        nf=max(0,length(track_ids)-1);switches+=ns;episodes+=ne;skipped+=na;fragments+=nf
        push!(identity_rows,Dict("truth_id"=>id,"visible_samples"=>length(visible),
            "confirmed_returned_track_ids"=>sort!(collect(track_ids)),"extra_confirmed_track_ids"=>nf,
            "confirmed_target_track_id_changes"=>ns,"tracked_untracked_tracked_episodes"=>ne,
            "ambiguous_visible_samples_resetting_identity_memory"=>na))
        for j in 2:length(visible)
            a,b=visible[j-1],visible[j];edge_denominator+=1
            id in localized[a] && id in localized[b] && (localized_edges+=1)
            b==a+1 && continue
            middle=[rowmap[(id,k)]["visibility_reason"] for k in a+1:b-1]
            all(r->r in ("scheduled_absence","annotated_absence"),middle) || continue
            amb=id in ambiguous[a] || id in ambiguous[b]
            detected=id in localized[a] && id in localized[b]
            recovered=(id,a,b) in edge_set
            lead=get(confirmed,(id,a),0)
            established=lead!=0 && count(k->k<=a && get(confirmed,(id,k),0)==lead,visible)>=2
            push!(gap_rows,Dict("truth_id"=>id,"frame_a"=>a,"frame_b"=>b,"missed_frames"=>b-a-1,
                "absence_kind"=>all(==("scheduled_absence"),middle) ? "scheduled" : all(==("annotated_absence"),middle) ? "annotated" : "mixed",
                "both_endpoints_localized"=>detected,"identity_ambiguous"=>amb,
                "confirmed_recovered"=>recovered,"two_prior_visible_samples"=>j-1>=2,
                "two_prior_confirmed_same_track_samples"=>established))
        end
    end
    ct["truth_consecutive_visible_edges"]=edge_denominator
    ct["both_localized_truth_edges"]=localized_edges
    ct["confirmed_target_track_id_changes"]=switches
    ct["extra_confirmed_track_ids_fragmentation"]=fragments
    ct["tracked_untracked_tracked_episodes"]=episodes
    ct["ambiguous_visible_samples_resetting_identity_memory"]=skipped
    # Nonconsecutive same-ID predictions can span visible missed detections as
    # well as true absences. They remain correct identity bridges, but are not
    # automatically true consecutive-visible edges (recall numerator).
    true_edge_set=Set((id,a,b) for id in ids for (a,b) in zip(
        [k for k in eachindex(c.images) if rowmap[(id,k)]["visible"]][1:end-1],
        [k for k in eachindex(c.images) if rowmap[(id,k)]["visible"]][2:end]))
    correct_truth_edges=length(intersect(edge_set,true_edge_set))
    ct["confirmed_truth_edges_recovered"]=correct_truth_edges
    bridges=counts();bridges["predictions"]=0
    for name in ("correct_scheduled_absence","correct_visible_detection_or_retention_miss","correct_other_absence");bridges[name]=0;end
    for e in edges
        e.frame_b>e.frame_a+1 || continue
        bridges["predictions"]+=1;bridges[e.class]+=1
        if e.class=="confirmed_correct"
            middle=[rowmap[(e.id,k)] for k in e.frame_a+1:e.frame_b-1]
            kind=any(r->r["visible"],middle) ? "correct_visible_detection_or_retention_miss" :
                all(r->r["visibility_reason"]=="scheduled_absence",middle) ? "correct_scheduled_absence" : "correct_other_absence"
            bridges[kind]+=1
        end
    end
    gap_metrics=Dict{String,Any}[]
    for g in 1:length(c.images)-2
        rows=filter(r->r["missed_frames"]==g,gap_rows);n=length(rows)
        nr=count(r->r["confirmed_recovered"],rows);na=count(r->r["identity_ambiguous"],rows)
        nd=count(r->r["both_endpoints_localized"],rows);ne=count(r->r["two_prior_visible_samples"],rows)
        nre=count(r->r["two_prior_visible_samples"] && r["confirmed_recovered"],rows)
        nrea=count(r->r["two_prior_visible_samples"] && r["identity_ambiguous"],rows)
        push!(gap_metrics,Dict("missed_frames"=>g,"counts"=>Dict("truth_events"=>n,"confirmed_recovered"=>nr,
            "scheduled_truth_events"=>count(r->r["absence_kind"]=="scheduled",rows),
            "annotated_truth_events"=>count(r->r["absence_kind"]=="annotated",rows),
            "mixed_truth_events"=>count(r->r["absence_kind"]=="mixed",rows),
            "ambiguous_events"=>na,"both_endpoints_localized"=>nd,"two_prior_visible_events"=>ne,
            "two_prior_visible_recovered"=>nre,"two_prior_visible_ambiguous"=>nrea,
            "retained_established_events"=>count(r->r["two_prior_confirmed_same_track_samples"],rows)),
            "full_recovery_bounds"=>bounds(nr,na,n),"both_localized_recovery_bounds"=>bounds(nr,na,nd),
            "two_prior_visible_recovery_bounds"=>bounds(nre,nrea,ne)))
    end
    metrics=Dict("counts"=>ct,"edge_precision_bounds"=>bounds(ct["confirmed_correct"],ct["ambiguous"],ct["predictions"]),
        "full_edge_recall_bounds"=>bounds(correct_truth_edges,ct["ambiguous"],edge_denominator),
        "both_localized_edge_recall_bounds"=>bounds(correct_truth_edges,ct["ambiguous"],localized_edges),
        "confirmed_observation_recall"=>ratio(length(confirmed),ct["visible_truth_observations"]),
        "bridges"=>Dict("counts"=>bridges,"precision_bounds"=>bounds(bridges["confirmed_correct"],bridges["ambiguous"],bridges["predictions"])),
        "gap_recovery"=>gap_metrics,"per_identity"=>identity_rows,"gap_events"=>gap_rows)
    (;metrics,observations,edges)
end

function synthetic_clip(seed,condition;size=128,nframes=8)
    integer(seed) && seed>=0 && condition in CONDITIONS || err("unknown synthetic seed/condition")
    integer(size) && size>=64 && integer(nframes) && nframes>=5 || err("invalid synthetic clip dimensions")
    state=Ref(UInt64(seed));images=[zeros(size,size) for _ in 1:nframes];rows=Dict{String,Any}[]
    positions=[(x+2V.uniform!(state)-1,y+2V.uniform!(state)-1) for y in range(14,size-24;length=5) for x in range(14,size-24;length=5)]
    for (i,(x,y)) in enumerate(positions),k in 1:nframes
        xx=x+.8*(k-1);yy=y+.4*(k-1)
        missing=condition=="one_frame_gap" && i%4==0 && k==3 || condition=="two_frame_gap" && i%4==0 && k in (3,4)
        intensity=condition=="encounter_noise_clutter" && i%5==0 ? .06 : 1.
        push!(rows,truth_row("target-$i",k,xx,yy;visible=!missing,reason=missing ? "scheduled_absence" : "visible",intensity))
    end
    if condition=="encounter_noise_clutter"
        # Distinct known IDs can have unresolved images. Such associations are
        # explicitly ambiguous, not arbitrarily counted as proven switches.
        for (id,sgn) in (("encounter-A",1.),("encounter-B",-1.)),k in 1:nframes
            push!(rows,truth_row(id,k,size*.5+sgn*1.2*(k-(nframes+1)/2),size*.5+sgn*.3))
        end
        for k in 1:nframes
            xx=size-2+1.2*(k-1);inside=xx<=size
            push!(rows,truth_row("boundary",k,xx,18.;visible=inside,reason=inside ? "visible" : "outside_image"))
            push!(rows,truth_row("nuisance",k,size*.75,9.0 + .3k;role="nuisance"))
        end
    end
    for r in rows
        r["visible"] || continue
        generate_gaussian_particle!(images[r["frame"]],(r["x"],r["y"]),3.,r["intensity"])
    end
    if condition=="encounter_noise_clutter"
        for img in images,i in eachindex(img);img[i]+=.01*(2V.uniform!(state)-1);end
    end
    # Canonical ordering is independent of renderer insertion order.
    sort!(rows;by=r->(r["frame"],r["id"]))
    metadata=Dict{String,Any}("annotation_kind"=>"controlled_synthetic","generator_version"=>"annotated-particles-splitmix64-1",
        "seed"=>seed,"condition"=>condition,"image_shape"=>[size,size],"n_frames"=>nframes,
        "renderer"=>"SyntheticData.generate_gaussian_particle!; diameter4sigma3px; exact rounded/clipped square bbox; point sampling; no normalization",
        "gaussian_sigma_px"=>.75,"bbox_radius_px"=>3,"bbox_rounding"=>"round(Int,center±ceil(3sigma)); bounds clipped to1:size; ties nearest-even",
        "noise"=>condition=="encounter_noise_clutter" ? "additive centered uniform[-.01,.01],unclipped" : "none",
        "coordinates"=>"one-based pixel centers;x columns;y rows;ordinal frame time;no acquisition timestamps",
        "frames"=>[Dict("ordinal"=>k,"decoded_sha256"=>V.pixel_digest(img),"decoded_processing_type"=>"Float64") for (k,img) in enumerate(images)],
        "truth_sha256"=>Hammerhead._experiment_digest(rows))
    checked_clip(AnnotatedClip("seed$(seed)-$condition",images,rows,metadata,String[]))
end

# File input is an explicit receiving-host binding. Relocation is ordinal=>local
# path and must match the original bytes and processing-pixel hash exactly.
function load_clip(path;relocation=Dict{Int,String}())
    localpath=Hammerhead._artifact_local_path(path)
    isfile(localpath) || err("annotation manifest missing")
    initial=V.file_digest(localpath);data=TOML.parsefile(localpath)
    keys_exact(data,["schema_version","clip_id","annotation_kind","source_uri","citation","license",
        "coordinate_convention","length_unit","image_shape","frames","truth"])
    data["schema_version"]==ANNOTATION_SCHEMA || err("unsupported annotation manifest version")
    data["annotation_kind"] in ("independent_synthetic","manual_real") || err("unsupported imported annotation kind")
    all(k->data[k] isa String && !isempty(data[k]),("clip_id","source_uri","citation","license")) || err("missing annotation provenance")
    data["coordinate_convention"]=="one_based_pixel_centers_x_columns_y_rows" && data["length_unit"]=="px" || err("unsupported annotation coordinate convention")
    frames=data["frames"];frames isa AbstractVector && length(frames)>=2 || err("invalid manifest frames")
    relocation isa AbstractDict && all(k->integer(k) && 1<=k<=length(frames),keys(relocation)) || err("invalid relocation ordinals")
    validate_truth(data["truth"],length(frames),data["image_shape"])
    consumed=String[localpath];images=Matrix{Float64}[];identities=Dict{String,Any}[]
    for (k,f) in enumerate(frames)
        keys_exact(f,["ordinal","locator","bytes","sha256","decoded_sha256","decoded_processing_type"])
        f["ordinal"]===k && integer(f["bytes"]) && f["bytes"]>=0 || err("invalid frame ordinal/size")
        f["locator"] isa String && !isempty(f["locator"]) && Hammerhead._experiment_hash(f["sha256"]) && Hammerhead._experiment_hash(f["decoded_sha256"]) || err("invalid frame locator/hash")
        f["decoded_processing_type"]=="Float64" || err("unsupported processing precision")
        input=if haskey(relocation,k)
            relocation[k] isa AbstractString || err("relocation must be a local path")
            Hammerhead._artifact_local_path(relocation[k])
        else
            Hammerhead._artifact_relative_locator(f["locator"]) || err("foreign/absolute manifest locator requires explicit local relocation")
            p=Hammerhead._artifact_resolve_relative(dirname(localpath),f["locator"])
            U.within(U.resolved_path(p),realpath(dirname(localpath))) || err("manifest image escapes its directory; use explicit relocation")
            p
        end
        isfile(input) && filesize(input)==f["bytes"] && V.file_digest(input)==f["sha256"] || err("changed/missing input image")
        image=Matrix{Float64}(Hammerhead.load_frame(input,Float64))
        collect(size(image))==data["image_shape"] && all(isfinite,image) && V.pixel_digest(image)==f["decoded_sha256"] || err("decoded image shape/pixel identity mismatch")
        push!(images,image);push!(consumed,input);push!(identities,Dict("path"=>input,"bytes"=>f["bytes"],"sha256"=>f["sha256"]))
    end
    initial==V.file_digest(localpath) || err("manifest changed while loading")
    push!(identities,Dict("path"=>localpath,"bytes"=>filesize(localpath),"sha256"=>initial))
    rows=Dict{String,Any}[merge(Dict{String,Any}("intensity"=>"unknown"),Dict{String,Any}(r)) for r in data["truth"]]
    metadata=Dict{String,Any}("annotation_kind"=>data["annotation_kind"],"imported_manifest"=>deepcopy(data),
        "manifest_path"=>localpath,"frames"=>deepcopy(frames),"file_identities"=>identities,
        "truth_sha256"=>Hammerhead._experiment_digest(rows))
    checked_clip(AnnotatedClip(data["clip_id"],images,rows,metadata,unique(consumed)))
end

csv_cell(v)=v isa AbstractString ? "\""*replace(v,'"'=>"\"\"")*"\"" : string(v)
function csv_rows(columns,rows)
    io=IOBuffer();println(io,join(columns,','))
    for r in rows;println(io,join(csv_cell.(r),','));end
    String(take!(io))
end
function csv_ids(ids)
    # Reversible structured cell even for IDs containing semicolons/newlines.
    io=IOBuffer();TOML.print(io,Dict("ids"=>ids));strip(String(take!(io)))
end
function native_table(result;frame_id="")
    mktemp() do path,io
        close(io);export_table(path,result;frame_id);read(path,String)
    end
end
function scientific_recipe(params,predictor,piv_passes,max_gaps,min_track_length)
    predictor in (:piv,nothing) || err("benchmark supports :piv or nothing, never truth/custom predictors")
    min_track_length isa Int && min_track_length>=2 || err("invalid minimum track length")
    all(g->integer(g) && g>=0,max_gaps) && length(unique(max_gaps))==length(max_gaps) && !isempty(max_gaps) || err("invalid gap policies")
    Dict("ptv_parameters"=>Dict(String(k)=>V.serial(getfield(params,k)) for k in fieldnames(PTVParameters)),
        "predictor"=>predictor===nothing ? "none_zero_displacement" : "production_piv",
        "piv_passes"=>[V.pass_recipe(p) for p in piv_passes],"backend"=>"cpu","processing_precision"=>"Float64",
        "threads"=>Threads.nthreads(),"piv_driver_threaded_default"=>true,"mask"=>"none","roi"=>"none",
        "preprocessing"=>"none","scale"=>"none","timestamps"=>"none_ordinal_frames",
        "max_gap_values"=>collect(max_gaps),"min_track_length"=>min_track_length,
        "later_tracking_field_predictor"=>"production bin/smooth previous accepted links; 32px window/16px overlap/min_count3",
        "track_output"=>"returned tracks only; singleton/internal rejected candidate history unavailable")
end
function evaluate_clip(c::AnnotatedClip;params=PTVParameters(),predictor=:piv,
        piv_passes=multipass_parameters([64,32]),max_gaps=(0,1,2),min_track_length=2,gate=GATE,prefix="clip-001")
    recipe=scientific_recipe(params,predictor,piv_passes,max_gaps,min_track_length)
    occursin(r"^clip-[0-9]+$",prefix) || err("invalid artifact prefix")
    c=checked_clip(deepcopy(c))
    particles=[detect_particles(img,params) for img in c.images]
    detection=detection_metrics(c,particles;gate)
    pairs=[run_ptv(c.images[k],c.images[k+1],params;predictor,piv_passes) for k in 1:length(c.images)-1]
    correspondence=correspondence_metrics(c,pairs,particles,detection.assoc)
    artifacts=Dict{String,String}();tracking=Dict{String,Any}[]
    artifacts["$prefix-truth.csv"]=csv_rows(("truth_id","frame","x_px","y_px","visible","visibility_reason","role","intensity"),
        ((r["id"],r["frame"],r["x"],r["y"],r["visible"],r["visibility_reason"],r["role"],r["intensity"]) for r in c.rows))
    artifacts["$prefix-detection-associations.csv"]=csv_rows(("frame","detection_index","x_px","y_px","identity_status","confirmed_truth_id","operational_assigned_truth_id","candidate_truth_ids","candidate_nuisance_ids"),
        ((k,j,p.x[j],p.y[j],l["status"],l["status"]=="unique" ? l["id"] : "",l["id"],csv_ids(l["candidate_ids"]),csv_ids(l["candidate_nuisance_ids"]))
         for (k,p) in enumerate(particles) for (j,l) in enumerate(detection.assoc[k].labels)))
    for (k,r) in enumerate(pairs);artifacts["$prefix-ptv-pair-$k.csv"]=native_table(r;frame_id=string(k));end
    for gap in max_gaps
        result=track_particles(c.images,params;predictor,piv_passes,min_track_length,max_gap=gap,progress=false)
        score=tracking_metrics(c,result,particles,detection.assoc)
        push!(tracking,Dict("max_gap"=>gap,"metrics"=>score.metrics,
            "native_table"=>"$prefix-tracks-gap-$gap.csv","association_table"=>"$prefix-track-associations-gap-$gap.csv"))
        artifacts["$prefix-tracks-gap-$gap.csv"]=native_table(result;frame_id=c.id)
        artifacts["$prefix-track-associations-gap-$gap.csv"]=csv_rows(("trajectory_id","observation_index","frame","x_px","y_px","detection_index","identity_status","confirmed_truth_id","candidate_truth_ids","candidate_nuisance_ids"),
            ((r["trajectory_id"],r["observation_index"],r["frame"],r["x"],r["y"],r["detection_index"],r["identity_status"],r["confirmed_truth_id"],csv_ids(r["candidate_truth_ids"]),csv_ids(r["candidate_nuisance_ids"])) for r in score.observations))
        artifacts["$prefix-track-edges-gap-$gap.csv"]=csv_rows(("trajectory_id","frame_a","frame_b","classification","confirmed_same_truth_id"),
            ((e.trajectory_id,e.frame_a,e.frame_b,e.class,e.id) for e in score.edges))
    end
    checked_clip(c)
    row=Dict{String,Any}("clip_id"=>c.id,"input"=>c.metadata,"truth_rows"=>c.rows,"recipe"=>recipe,
        "detection"=>detection.metrics,"correspondence"=>correspondence,"tracking"=>tracking,
        "artifact_prefix"=>prefix,"calls"=>Dict("standalone_detection"=>length(c.images),
            "ptv_pairs"=>length(pairs),"tracking"=>length(max_gaps),
            "matcher_transitions"=>(1+length(max_gaps))*(length(c.images)-1),
            "piv_predictor_upper_bound"=>predictor===:piv ? length(pairs)+length(max_gaps) : 0,
            "detector_invocations"=>length(c.images)+2length(pairs)+length(max_gaps)*length(c.images)))
    (;row,artifacts)
end

sum_counts(ds)=Dict(k=>sum(get(d,k,0) for d in ds) for k in union((Set(keys(d)) for d in ds)...))
function pooled(rows,max_gaps)
    d=sum_counts([r["detection"]["counts"] for r in rows])
    detection=Dict("counts"=>d,"precision"=>ratio(d["localized_tp"],d["predictions"]),"recall"=>ratio(d["localized_tp"],d["visible_target_observations"]))
    correspondence=Dict{String,Any}()
    for pop in ("raw","accepted")
        ct=sum_counts([r["correspondence"][pop]["counts"] for r in rows])
        correspondence[pop]=Dict("counts"=>ct,"precision_bounds"=>bounds(ct["confirmed_correct"],ct["ambiguous"],ct["predictions"]),
            "full_recall_bounds"=>bounds(ct["confirmed_correct"],ct["ambiguous"],ct["full_truth_pairs"]),
            "both_localized_recall_bounds"=>bounds(ct["confirmed_correct"],ct["ambiguous"],ct["both_localized_truth_pairs"]))
    end
    tracking=Dict{String,Any}[]
    for gap in max_gaps
        ms=[only(filter(t->t["max_gap"]==gap,r["tracking"]))["metrics"] for r in rows]
        ct=sum_counts([m["counts"] for m in ms]);bridges=sum_counts([m["bridges"]["counts"] for m in ms])
        gs=Dict{String,Any}[]
        for g in sort!(unique(r["missed_frames"] for m in ms for r in m["gap_recovery"]))
            gc=sum_counts([r["counts"] for m in ms for r in m["gap_recovery"] if r["missed_frames"]==g])
            push!(gs,Dict("missed_frames"=>g,"counts"=>gc,
                "full_recovery_bounds"=>bounds(gc["confirmed_recovered"],gc["ambiguous_events"],gc["truth_events"]),
                "both_localized_recovery_bounds"=>bounds(gc["confirmed_recovered"],gc["ambiguous_events"],gc["both_endpoints_localized"]),
                "two_prior_visible_recovery_bounds"=>bounds(gc["two_prior_visible_recovered"],gc["two_prior_visible_ambiguous"],gc["two_prior_visible_events"])))
        end
        push!(tracking,Dict("max_gap"=>gap,"counts"=>ct,"edge_precision_bounds"=>bounds(ct["confirmed_correct"],ct["ambiguous"],ct["predictions"]),
            "full_edge_recall_bounds"=>bounds(ct["confirmed_truth_edges_recovered"],ct["ambiguous"],ct["truth_consecutive_visible_edges"]),
            "confirmed_observation_recall"=>ratio(ct["confirmed_retained_truth_observations"],ct["visible_truth_observations"]),
            "bridges"=>Dict("counts"=>bridges,"precision_bounds"=>bounds(bridges["confirmed_correct"],bridges["ambiguous"],bridges["predictions"])),"gap_recovery"=>gs))
    end
    Dict("detection"=>detection,"correspondence"=>correspondence,"tracking"=>tracking)
end

function environment_record()
    env=U.environment_record();append!(env["source_files"],V.fixture_identity([@__FILE__]))
    # PTV's internal run_piv keeps its threaded default; the older PIV scorecard
    # source of this environment mapping uses explicit threaded=false instead.
    env["processing_threaded"]=true
    env
end
function run_study(;clips=nothing,params=PTVParameters(),predictor=:piv,piv_passes=multipass_parameters([64,32]),max_gaps=(0,1,2),min_track_length=2,gate=GATE)
    env=environment_record();rows=Dict{String,Any}[];artifacts=Dict{String,String}()
    input=clips===nothing ? [(s,k) for s in SEEDS for k in CONDITIONS] : clips
    isempty(input) && err("at least one annotated clip required")
    for (i,item) in enumerate(input)
        c=clips===nothing ? synthetic_clip(item...) : item
        result=evaluate_clip(c;params,predictor,piv_passes,max_gaps,min_track_length,gate,prefix="clip-"*lpad(i,3,'0'))
        push!(rows,result.row);merge!(artifacts,result.artifacts)
    end
    stable=U.stable_environment(env)
    annotation_kinds=[r["input"]["annotation_kind"] for r in rows]
    supplied=any(k->k in ("independent_synthetic","manual_real"),annotation_kinds)
    report=Dict{String,Any}("schema_version"=>SCHEMA,"generated_utc"=>string(now(UTC)),"environment"=>env,
        "source_and_environment_stable"=>stable,"provenance_status"=>stable ? "stable on-disk source/environment; fresh process required" : "source/environment drift; regenerate",
        "groups"=>rows,"pooled"=>pooled(rows,max_gaps),"calls"=>sum_counts([r["calls"] for r in rows]),
        "scoring"=>Dict("localization_gate_px"=>gate,"matching"=>"maximum cardinality then minimum total squared distance; independent rectangular Hungarian; row/column order breaks exact ties",
            "ambiguity"=>"multiple in-gate truth IDs or competing detections; optimization assignment never proves merged-image identity",
            "detection"=>"operational localized TP count; FP=all detections-TP,FN=all eligible visible targets-TP; invisible/nuisance ledger separate",
            "precision"=>"all predictions conserved into confirmed_correct/confirmed_wrong/unmapped/ambiguous; strict lower=correct/N; conservative upper=(correct+ambiguous)/N",
            "recall"=>"full visible truth pair/edge/event denominator; both-localized conditional denominator from operational one-to-one association, ambiguity retained; conservative upper capped at1",
            "switches"=>"same-track consecutive observed truth ID changes and target-centric output-ID changes separately; target memory persists across misses/absence but resets on ambiguous visible samples",
            "fragmentation"=>"extra distinct confirmed returned track IDs per truth ID; tracked/untracked/tracked episodes across visible samples separately; ambiguous samples reset episode memory",
            "gaps"=>"declared scheduled/annotated invisible intervals with visible bracketing endpoints; absence kinds separate; recovery requires one exact same-ID consecutive returned edge; no fabricated gap samples",
            "retention"=>"min_track_length output filtering includes singleton losses; internal candidate/UOD rejection history unavailable",
            "error"=>"correct pairs only,measured-minus-reference displacement; bias/RMS separate from all-match precision/recall",
            "pooling"=>"sum counts across clips then divide; no averaging per-clip ratios; clips/nodes are not independent calibration samples"),
        "external_evidence"=>Dict("annotation_kinds"=>annotation_kinds,
            "independent_recordings"=>supplied ? "explicitly supplied annotation provenance,not automatically independently verified" : "unavailable: controlled in-house synthetic only",
            "real_recordings"=>"manual_real" in annotation_kinds ? "supplied manual_real annotation; accuracy/provenance not independently certified" : "unavailable:no manual_real recording supplied"),
        "limitations"=>["Controlled annotation and shared production Gaussian rendering do not establish external/experimental performance.",
            "Localization association is operational,not observation of which particle generated a merged intensity peak; ambiguity bounds are conservative,not probabilistic.",
            "Returned-track filtering hides singleton/internal candidate histories; visible-detection misses remain different from scheduled invisibility.",
            "Combined encounter/noise/clutter condition describes failure exposure,not causal isolation of its factors.",
            "No estimator/default changes,truth predictors,timing/peak-memory measurements,GPU evidence or universal accuracy thresholds.",
            "Native export_table IDs index returned trajectories,not physical truth IDs; scoring/annotation CSVs are separately labelled.",
            "Source hashes attest on-disk files,not already loaded code; use a fresh process with frozen source."])
    (;report,artifacts)
end

function check_output(directory;protected_paths=String[])
    target=Hammerhead._artifact_local_path(directory);canonical=U.resolved_path(target)
    ispath(target) || islink(target) ? err("scorecard requires a fresh output directory") : nothing
    U.within(canonical,realpath(ROOT)) && !U.within(canonical,U.resolved_path(joinpath(ROOT,"bench","profile-output"))) && err("repository output must stay under bench/profile-output")
    for p in protected_paths
        Hammerhead._artifact_alias(target,p) && err("output aliases consumed input")
        U.within(canonical,U.resolved_path(p)) && err("output descends from consumed input")
    end
    target
end
function markdown_report(report)
    io=IOBuffer();println(io,"<!-- $MARKER -->\n# Annotated PTV/tracking scorecard\n\n$(report["provenance_status"]).\n")
    println(io,"All-prediction precision is an interval: confirmed-correct/N to (confirmed-correct+ambiguous)/N. Unknown identity is not proved wrong. All counts/recipes/truth/input hashes are in `ptv_tracking.toml`.\n")
    fmt(r)=r["available"] ? "$(round(r["fraction"];sigdigits=5)) ($(r["numerator"])/$(r["denominator"]))" : "unavailable (0)"
    interval(b)=fmt(b["strict_lower"])*" — "*fmt(b["conservative_upper"])
    println(io,"| Clip | Detection TP / predictions / visible truth | Raw pair C/W/U/A | Accepted pair C/W/U/A | Accepted precision bounds | Full recall bounds | Correct-pair u/v RMS px |")
    println(io,"|---|---:|---:|---:|---|---|---|")
    for row in report["groups"]
        d=row["detection"]["counts"];raw=row["correspondence"]["raw"];a=row["correspondence"]["accepted"]
        parts(c)=join([c["counts"][k] for k in CLASSES],'/')
        rms(k)=a["correct_pair_displacement_error"][k]["available"] ? string(round(a["correct_pair_displacement_error"][k]["rms"];sigdigits=5)) : "unavailable"
        println(io,"| $(row["clip_id"]) | $(d["localized_tp"]) / $(d["predictions"]) / $(d["visible_target_observations"]) | $(parts(raw)) | $(parts(a)) | $(interval(a["precision_bounds"])) | $(interval(a["full_recall_bounds"])) | $(rms("u")) / $(rms("v")) |")
    end
    println(io,"\nC/W/U/A = confirmed correct / confirmed wrong / unmapped / ambiguous. Detection FP/FN, invisible/nuisance and conditional-recall populations remain in TOML.\n")
    println(io,"| Clip | max_gap | Edges C/W/U/A | Edge precision bounds | Full edge recall bounds | Confirmed observations / visible truth | Same-track ID changes / target output-ID changes | Extra track IDs / restart episodes | Detections not retained | Ambiguous visible samples resetting memory |")
    println(io,"|---|---:|---:|---|---|---:|---:|---:|---:|---:|")
    for row in report["groups"],t in row["tracking"]
        m=t["metrics"];c=m["counts"]
        println(io,"| $(row["clip_id"]) | $(t["max_gap"]) | $(join([c[k] for k in CLASSES],'/')) | $(interval(m["edge_precision_bounds"])) | $(interval(m["full_edge_recall_bounds"])) | $(c["confirmed_retained_truth_observations"])/$(c["visible_truth_observations"]) | $(c["observed_same_track_identity_changes"]) / $(c["confirmed_target_track_id_changes"]) | $(c["extra_confirmed_track_ids_fragmentation"]) / $(c["tracked_untracked_tracked_episodes"]) | $(c["detections_not_in_returned_tracks"]) | $(c["ambiguous_visible_samples_resetting_identity_memory"]) |")
    end
    println(io,"\n| Clip | max_gap | Declared missing frames | Recovered / ambiguous / truth events | Full recovery bounds | Both-localized recovery bounds | Two-prior-visible recovery bounds |")
    println(io,"|---|---:|---:|---:|---|---|---|")
    for row in report["groups"],t in row["tracking"],g in t["metrics"]["gap_recovery"]
        c=g["counts"];c["truth_events"]==0 && continue
        println(io,"| $(row["clip_id"]) | $(t["max_gap"]) | $(g["missed_frames"]) | $(c["confirmed_recovered"]) / $(c["ambiguous_events"]) / $(c["truth_events"]) | $(interval(g["full_recovery_bounds"])) | $(interval(g["both_localized_recovery_bounds"])) | $(interval(g["two_prior_visible_recovery_bounds"])) |")
    end
    println(io,"\nBridge precision counts every returned frame jump,including wrong/unmapped/ambiguous endpoints; same-ID bridges across missed visible detections are separated from scheduled-absence recovery in TOML.\n")
    foreach(s->println(io,"- ",s),report["limitations"]);String(take!(io))
end
function write_report(directory,bundle;protected_paths=String[])
    report,artifacts=bundle.report,bundle.artifacts
    report["schema_version"]==SCHEMA && report["source_and_environment_stable"]===true && U.stable_environment(report["environment"]) || err("unsupported/unstable evidence")
    for r in report["groups"],f in get(r["input"],"file_identities",Any[])
        isfile(f["path"]) && filesize(f["path"])==f["bytes"] && V.file_digest(f["path"])==f["sha256"] || err("input changed before publication")
        push!(protected_paths,f["path"])
    end
    target=check_output(directory;protected_paths)
    names=vcat(["ptv_tracking.toml","ptv_tracking.md"],sort!(collect(keys(artifacts))))
    all(n->basename(n)==n && !occursin(':',n) && !isempty(n),names) || err("unsafe artifact filename")
    paths=joinpath.(target,names)
    for i in eachindex(paths)
        any(p->Hammerhead._artifact_alias(paths[i],p),protected_paths) && err("artifact aliases input")
        any(j->Hammerhead._artifact_alias(paths[i],paths[j]),1:i-1) && err("artifact outputs alias")
    end
    io=IOBuffer();println(io,"# $MARKER");TOML.print(io,report;sorted=true);text=String(take!(io));md=markdown_report(report)
    # Exclusive directory acquisition; failed publication leaves evidence/partial
    # output for inspection. No existing files/directories are removed/replaced.
    mkpath(dirname(target));mkdir(target)
    for (path,content) in zip(paths,[text,md,[artifacts[n] for n in names[3:end]]...])
        ispath(path) || islink(path) ? err("artifact destination already exists") : nothing
        write(path,content)
    end
    paths
end
function main(args=ARGS)
    output=joinpath(ROOT,"bench","profile-output","validation-ptv-tracking");manifest=nothing
    for arg in args
        if arg=="--help"
            println("julia --project=. --threads=1 bench/validation_ptv_tracking.jl [--output=fresh-directory] [--manifest=annotated-clip.toml]");return nothing
        elseif startswith(arg,"--output=");output=arg[10:end]
        elseif startswith(arg,"--manifest=");manifest=arg[12:end]
        else;err("unknown option $arg");end
    end
    # Entire input manifest is validated/read before output directory acquisition.
    clips=manifest===nothing ? nothing : [load_clip(manifest)]
    protected=clips===nothing ? String[] : reduce(vcat,[c.protected_paths for c in clips];init=String[])
    check_output(output;protected_paths=protected)
    bundle=run_study(;clips);foreach(println,write_report(output,bundle;protected_paths=protected));bundle
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    ValidationPTVTracking.main()
end
