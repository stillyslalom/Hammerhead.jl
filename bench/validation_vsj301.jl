#!/usr/bin/env julia
# Bench-only independent synthetic evaluation; production algorithms unchanged.
module ValidationVSJ301
using Hammerhead, TOML, SHA, Dates
include(joinpath(@__DIR__, "validation_ptv_tracking.jl"))
const P = ValidationPTVTracking
const V = P.V
const U = P.U
const ROOT = P.ROOT
const SCHEMA = "hammerhead-vsj301-evaluation-1"
const ANNOTATION_SCHEMA = "hammerhead-sparse-particle-clip-2"
const CACHE_SCHEMA = "hammerhead-vsj301-cache-1"
const OFFSETS = (0.0, 0.5, 1.0)
const GAPS = (0, 1, 2)
const CLASSES = ("associated_same_listed_id", "associated_different_listed_ids", "ambiguous", "unscorable_unannotated")
const ARCHIVES = Dict("301_raw.zip"=>6292551, "301_ptc.zip"=>11100328)
const BASE = "https://www.vsj.jp/~pivstd/"
err(s) = throw(ArgumentError(s))
integer(x) = x isa Integer && !(x isa Bool)
finite(x) = x isa Real && !(x isa Bool) && isfinite(x)
keys_exact(x, ks) = x isa AbstractDict && Set(keys(x))==Set(ks) ? nothing : err("malformed mapping/schema")

struct SparseClip
    images::Vector{Matrix{Float64}}
    rows::Vector{Dict{String,Any}}
    metadata::Dict{String,Any}
    protected_paths::Vector{String}
end

function parse_particles(text, frame)
    integer(frame) && frame>=1 || err("invalid annotation frame")
    rows=Dict{String,Any}[]; seen=Set{String}()
    for line in split(text, '\n')
        isempty(strip(line)) && continue
        fields=split(line);length(fields)==7 || err("particle file needs seven columns")
        id=tryparse(Int,fields[1]);id!==nothing && id>0 || err("invalid particle ID")
        ident=string(id);ident in seen && err("duplicate particle ID/frame");push!(seen,ident)
        values=tryparse.(Float64,fields[2:end]);all(x->x!==nothing && isfinite(x),values) || err("nonfinite/malformed particle row")
        values[6]>=0 || err("negative particle intensity")
        push!(rows,Dict("id"=>ident,"frame"=>Int(frame),"position_status"=>"provided",
            "visibility_status"=>"unknown","world_x_cm"=>values[1],"world_y_cm"=>values[2],
            "world_z_cm"=>values[3],"source_x_px"=>values[4],"source_y_px"=>values[5],
            "source_peak_intensity"=>values[6]))
    end
    rows
end

function sparse_ledger(provided, nframes)
    integer(nframes) && nframes>=2 || err("at least two frames required")
    seen=Set{Tuple{String,Int}}()
    for r in provided
        keys_exact(r,["id","frame","position_status","visibility_status","world_x_cm","world_y_cm","world_z_cm","source_x_px","source_y_px","source_peak_intensity"])
        r["id"] isa String && !isempty(r["id"]) || err("invalid identity")
        integer(r["frame"]) && 1<=r["frame"]<=nframes || err("invalid frame")
        r["position_status"]=="provided" && r["visibility_status"]=="unknown" || err("unsupported sparse status")
        all(k->finite(r[k]),("world_x_cm","world_y_cm","world_z_cm","source_x_px","source_y_px","source_peak_intensity")) && r["source_peak_intensity"]>=0 || err("invalid particle values")
        key=(r["id"],r["frame"]);key in seen && err("duplicate annotation");push!(seen,key)
    end
    map=Dict((r["id"],r["frame"])=>r for r in provided)
    ids=sort!(unique(r["id"] for r in provided))
    [haskey(map,(id,k)) ? deepcopy(map[(id,k)]) : Dict{String,Any}("id"=>id,"frame"=>k,
        "position_status"=>"unknown","visibility_status"=>"unknown",
        "world_x_cm"=>"unknown","world_y_cm"=>"unknown","world_z_cm"=>"unknown",
        "source_x_px"=>"unknown","source_y_px"=>"unknown","source_peak_intensity"=>"unknown")
        for k in 1:nframes for id in ids]
end

function decode_raw(bytes, shape=(256,256))
    length(shape)==2 && all(n->integer(n) && n>0,shape) || err("invalid RAW shape")
    length(bytes)==prod(shape) || err("RAW byte count disagrees with shape")
    # Source bytes: x fastest, first row at top. No contrast normalization.
    permutedims(reshape(Float64.(bytes)./255,shape[2],shape[1]))
end

function checked_clip(c::SparseClip)
    length(c.images)>=2 && all(im->size(im)==size(first(c.images)) && all(isfinite,im),c.images) || err("invalid processing images")
    supplied=[r for r in c.rows if r["position_status"]=="provided"]
    isequal(c.rows,sparse_ledger(supplied,length(c.images))) || err("malformed/incomplete sparse ledger")
    c.metadata["schema_version"]==ANNOTATION_SCHEMA || err("unsupported sparse annotation version")
    c.metadata["truth_sha256"]==Hammerhead._experiment_digest(c.rows) || err("changed annotations")
    length(c.metadata["frames"])==length(c.images) || err("image identity count mismatch")
    for (k,im) in enumerate(c.images)
        f=c.metadata["frames"][k]
        f["ordinal"]===k && f["decoded_sha256"]==V.pixel_digest(im) || err("changed processing pixels/order")
    end
    for f in get(c.metadata,"file_identities",Any[])
        isfile(f["path"]) && !islink(f["path"]) && filesize(f["path"])==f["bytes"] && V.file_digest(f["path"])==f["sha256"] || err("changed cache input")
    end
    c
end

function load_cache(directory;python=nothing)
    root=Hammerhead._artifact_local_path(directory)
    isdir(root) && !islink(root) || err("cache must be a regular directory")
    path=joinpath(root,"cache.toml");isfile(path) && !islink(path) || err("completed cache manifest missing")
    filesize(path)<=128_000 || err("cache manifest exceeds size limit")
    initial=V.file_digest(path);data=TOML.parsefile(path)
    keys_exact(data,["schema_version","source_frames","citation","permission_statement","redistribution_license","hash_authenticity","pages","archives","members","acquired_utc","preparer_sha256","python_version"])
    data["schema_version"]==CACHE_SCHEMA && data["source_frames"]==collect(0:7) && all(integer,data["source_frames"]) || err("unsupported cache/frame selection")
    all(k->data[k] isa String && !isempty(data[k]),("citation","permission_statement","redistribution_license","hash_authenticity")) || err("missing cache provenance")
    all(k->data[k] isa String && !isempty(data[k]),("acquired_utc","python_version")) && Hammerhead._experiment_hash(data["preparer_sha256"]) || err("invalid acquisition provenance")
    data["permission_statement"]=="Anybody can download and analyse the Standard images for any purpose." || err("publisher permission requires review")
    executable=python===nothing ? something(Sys.which("python"),Sys.which("python3"),"") : Hammerhead._artifact_local_path(python)
    isempty(executable) && err("Python3.11+ required for fresh offline archive/member binding audit")
    helper=joinpath(@__DIR__,"prepare_vsj301.py");helper_hash=V.file_digest(helper)
    audit_stdout=read(Cmd([executable,helper,"--offline","--cache="*root]),String)
    V.file_digest(helper)==helper_hash || err("offline audit helper changed during execution")
    expectedpages=Dict("usage.html"=>BASE,"dataset.html"=>BASE*"image3d/image301.html","particle-format.html"=>BASE*"image3d/ptc.html")
    identities=Dict{String,Any}[];names=Set(["cache.toml"])
    for section in ("pages","archives","members")
        data[section] isa AbstractVector || err("cache records must be arrays")
        for r in data[section]
            keys_exact(r,section=="members" ? ["path","archive","bytes","sha256","crc32"] : ["path","url","final_url","bytes","sha256","etag","last_modified"])
            name=r["path"]
            name isa String && occursin(r"^[A-Za-z0-9_.-]+$",name) && basename(name)==name && !(name in names) || err("unsafe/duplicate cache path")
            push!(names,name)
            bound=section=="pages" ? 128_000 : section=="members" ? 300_000 : 12_000_000
            integer(r["bytes"]) && 0<=r["bytes"]<=bound && Hammerhead._experiment_hash(r["sha256"]) || err("invalid cache identity")
            f=joinpath(root,name);isfile(f) && !islink(f) && filesize(f)==r["bytes"] && V.file_digest(f)==r["sha256"] || err("changed/missing cache file")
            push!(identities,Dict("path"=>f,"bytes"=>r["bytes"],"sha256"=>r["sha256"]))
            if section!="members"
                expected=section=="pages" ? get(expectedpages,name,"") : haskey(ARCHIVES,name) ? BASE*"image3d/"*name : ""
                expected!="" && r["url"]==expected && r["final_url"]==expected || err("publisher URL changed")
            else
                integer(r["crc32"]) && 0<=r["crc32"]<=typemax(UInt32) || err("invalid member CRC")
            end
        end
    end
    length(data["pages"])==3 && length(data["archives"])==2 && length(data["members"])==16 || err("incomplete cache")
    Dict(r["path"]=>r["bytes"] for r in data["archives"])==ARCHIVES || err("archive sizes changed")
    Set(readdir(root))==names || err("unknown cache files refused")
    members=Dict(r["path"]=>r for r in data["members"])
    images=Matrix{Float64}[];provided=Dict{String,Any}[]
    for k in 0:7
        for (prefix,ext,archive) in (("img","raw","301_raw.zip"),("ptc","dat","301_ptc.zip"))
            name=prefix*lpad(k,3,'0')*"."*ext
            haskey(members,name) && members[name]["archive"]==archive || err("member/archive binding changed")
        end
        push!(images,decode_raw(read(joinpath(root,"img"*lpad(k,3,'0')*".raw"))))
        append!(provided,parse_particles(read(joinpath(root,"ptc"*lpad(k,3,'0')*".dat"),String),k+1))
    end
    V.file_digest(path)==initial || err("manifest changed while reading")
    push!(identities,Dict("path"=>path,"bytes"=>filesize(path),"sha256"=>initial))
    rows=sparse_ledger(provided,8)
    metadata=Dict{String,Any}("schema_version"=>ANNOTATION_SCHEMA,"annotation_kind"=>"independent_synthetic",
        "publisher_cache"=>data,"source_frame_numbers"=>collect(0:7),"publisher_frame_interval_s"=>.005,
        "timing"=>"ordinal processing frames; publisher nominal interval, no absolute timestamps",
        "raw_encoding"=>"UInt8,x-fastest,top-row-first,256x256;Float64 divided by255;no contrast normalization",
        "registration"=>"unverified origin; fixed offsets0,.5,1 evaluated from identical production outputs",
        "frames"=>[Dict("ordinal"=>k,"source_frame"=>k-1,"decoded_sha256"=>V.pixel_digest(im)) for (k,im) in enumerate(images)],
        "truth_sha256"=>Hammerhead._experiment_digest(rows),"file_identities"=>identities,
        "offline_zip_audit"=>Dict("boundary"=>"Python stdlib independently rereads ZIP,verifies CRC and selected member/archive bytes;Julia verifies pinned files/decoded pixels",
            "helper_sha256"=>helper_hash,"interpreter_path"=>executable,"stdout"=>audit_stdout),"protected_cache_directory"=>root)
    checked_clip(SparseClip(images,rows,metadata,[root;[r["path"] for r in identities]]))
end

# Independent sparse in-gate graph. Components preserve the global objective;
# no production greedy matcher/cell list and no annotation/detection deletion.
function graph_assignment(edges, m;max_edges=200_000,max_component=512,max_cells=1_000_000)
    integer(m) && m>=0 && all(x->integer(x) && x>0,(max_edges,max_component,max_cells)) || err("invalid resource limits")
    n=length(edges);sum(length,edges;init=0)<=max_edges || err("association edge resource limit")
    parent=collect(1:n+m)
    function root(a)
        while parent[a]!=a;parent[a]=parent[parent[a]];a=parent[a];end
        a
    end
    for i in 1:n
        js=Set{Int}()
        for (j,cost) in edges[i]
            integer(j) && 1<=j<=m && finite(cost) && cost>=0 && !(j in js) || err("invalid/duplicate graph edge")
            push!(js,j);a=root(i);b=root(n+j);parent[b]=a
        end
    end
    groups=Dict{Int,Tuple{Vector{Int},Vector{Int}}}()
    for i in 1:n
        isempty(edges[i]) && continue
        push!(get!(groups,root(i), (Int[],Int[]))[1],i)
    end
    linked=Set(j for es in edges for (j,_) in es)
    for j in sort!(collect(linked));push!(get!(groups,root(n+j),(Int[],Int[]))[2],j);end
    out=zeros(Int,n);sizes=Int[]
    for key in sort!(collect(keys(groups)))
        ii,jj=groups[key];nr,nc=length(ii),length(jj)
        nr+nc<=max_component && big(nr)*(nc+nr)<=max_cells || err("association component resource limit")
        push!(sizes,nr+nc);cost=zeros(nr,nc);allowed=falses(nr,nc);jmap=Dict(j=>q for (q,j) in enumerate(jj))
        for (p,i) in enumerate(ii),(j,c) in edges[i];cost[p,jmap[j]]=c;allowed[p,jmap[j]]=true;end
        map=P.assignment(cost,allowed)
        for (p,i) in enumerate(ii);map[p]==0 || (out[i]=jj[map[p]]);end
    end
    (;map=out,counts=Dict("edges"=>sum(length,edges;init=0),"components"=>length(sizes),
        "largest_component_vertices"=>maximum(sizes;init=0),"isolated_predictions"=>count(isempty,edges),
        "isolated_annotations"=>m-length(linked)))
end

function associate(x,y,truth;gate=.75,max_edges=200_000,max_component=512,max_cells=1_000_000)
    length(x)==length(y) && finite(gate) && gate>0 && isfinite(gate^2) || err("invalid association input")
    length(truth)<=10_000 && length(x)<=20_000 || err("association node resource limit")
    bins=Dict{Tuple{Int,Int},Vector{Int}}()
    function cell(a,b)
        aa,bb=a/gate,b/gate
        all(v->isfinite(v) && abs(v)<div(typemax(Int),4),(aa,bb)) || err("association cell arithmetic unavailable")
        (floor(Int,aa),floor(Int,bb))
    end
    for (j,r) in enumerate(truth)
        finite(r["x"]) && finite(r["y"]) || err("invalid annotation coordinate")
        push!(get!(bins,cell(r["x"],r["y"]),Int[]),j)
    end
    edges=[Tuple{Int,Float64}[] for _ in eachindex(x)];nt=zeros(Int,length(truth));total=0
    for i in eachindex(x)
        finite(x[i]) && finite(y[i]) || continue
        bx,by=cell(x[i],y[i])
        for dx in -1:1,dy in -1:1,j in get(bins,(bx+dx,by+dy),Int[])
            distance=hypot(x[i]-truth[j]["x"],y[i]-truth[j]["y"])
            if distance<=gate
                push!(edges[i],(j,distance^2));nt[j]+=1;total+=1;total<=max_edges || err("association edge resource limit")
            end
        end
        sort!(edges[i];by=first)
    end
    assigned=graph_assignment(edges,length(truth);max_edges,max_component,max_cells)
    labels=[begin
        js=[j for (j,_) in edges[i]];amb=length(js)>1 || any(j->nt[j]>1,js)
        j=assigned.map[i]
        Dict("status"=>amb ? "ambiguous" : j==0 ? "unscorable_unannotated" : "unique",
            "id"=>j==0 ? "" : truth[j]["id"],"index"=>j,"candidate_ids"=>[truth[q]["id"] for q in js])
    end for i in eachindex(x)]
    truth_ambiguous=nt.>1
    for es in edges
        length(es)>1 || continue
        for (j,_) in es;truth_ambiguous[j]=true;end
    end
    (;labels,map=assigned.map,truth_ambiguous,
        resources=assigned.counts,matched=count(!=(0),assigned.map))
end

provided(c,k,offset)=sort!([Dict("id"=>r["id"],"x"=>r["source_x_px"]+offset,"y"=>r["source_y_px"]+offset) for r in c.rows if r["frame"]==k && r["position_status"]=="provided"];by=r->r["id"])
function edge_class(a,b)
    (a["status"]=="unscorable_unannotated" || b["status"]=="unscorable_unannotated") && return "unscorable_unannotated"
    (a["status"]=="ambiguous" || b["status"]=="ambiguous") && return "ambiguous"
    a["id"]==b["id"] ? "associated_same_listed_id" : "associated_different_listed_ids"
end
counts()=Dict(k=>0 for k in CLASSES)
function metrics(c;denominator=sum(values(c)))
    n=sum(values(c));correct=c["associated_same_listed_id"];unknown=c["ambiguous"]+c["unscorable_unannotated"]
    Dict("counts"=>merge(copy(c),Dict("predictions"=>n)),"all_prediction_precision_bounds"=>P.bounds(correct,unknown,n),
        "listed_endpoint_recall_bounds"=>P.bounds(correct,c["ambiguous"],denominator))
end

function score(c,particles,pairs,tracks,offset;gate=.75,limits=(max_edges=200_000,max_component=512,max_cells=1_000_000))
    finite(offset) && offset in OFFSETS || err("unapproved coordinate hypothesis")
    n=length(c.images);length(particles)==n && length(pairs)==n-1 || err("output frame count mismatch")
    truth=[provided(c,k,offset) for k in 1:n]
    truth_indices=[Dict(r["id"]=>j for (j,r) in enumerate(t)) for t in truth]
    assoc=[associate(p.x,p.y,truth[k];gate,limits...) for (k,p) in enumerate(particles)]
    localized_ids=[Set(l["id"] for l in a.labels if l["index"]!=0) for a in assoc]
    artifacts=Dict{String,String}();tag=offset==0 ? "offset-0" : offset==.5 ? "offset-0p5" : "offset-1"
    artifacts["$tag-detection-associations.csv"]=P.csv_rows(("frame","detection_index","x_px","y_px","identity_status","unique_associated_id","operational_assigned_id","candidate_ids"),
        ((k,i,p.x[i],p.y[i],l["status"],l["status"]=="unique" ? l["id"] : "",l["id"],P.csv_ids(l["candidate_ids"])) for (k,p) in enumerate(particles) for (i,l) in enumerate(assoc[k].labels)))
    raw=counts();accepted=counts();full=0;both=0;moments=Dict(k=>U.Moment() for k in ("u","v"));acceptedmom=Dict(k=>U.Moment() for k in ("u","v"))
    pairedrows=Any[]
    for (k,r) in enumerate(pairs)
        np=length(r.index_a);all(a->length(a)==np,(r.index_b,r.x,r.y,r.u,r.v,r.outliers,r.match_residual)) || err("malformed pair output")
        length(unique(r.index_a))==np && length(unique(r.index_b))==np || err("duplicate pair endpoints")
        for (p,q) in ((r.particles_a,particles[k]),(r.particles_b,particles[k+1]))
            all(f->isequal(getfield(p,f),getfield(q,f)),fieldnames(typeof(p))) || err("repeated detector differs")
        end
        ta=Dict(r["id"]=>r for r in truth[k]);tb=Dict(r["id"]=>r for r in truth[k+1]);ids=intersect(Set(keys(ta)),Set(keys(tb)))
        full+=length(ids)
        both+=length(intersect(ids,localized_ids[k],localized_ids[k+1]))
        for i in 1:np
            a,b=r.index_a[i],r.index_b[i]
            1<=a<=length(particles[k]) && 1<=b<=length(particles[k+1]) || err("pair endpoint out of bounds")
            la,lb=assoc[k].labels[a],assoc[k+1].labels[b];cl=edge_class(la,lb);raw[cl]+=1;r.outliers[i] || (accepted[cl]+=1)
            id=cl=="associated_same_listed_id" ? la["id"] : ""
            if !isempty(id)
                for (key,val) in (("u",r.u[i]-(tb[id]["x"]-ta[id]["x"])),("v",r.v[i]-(tb[id]["y"]-ta[id]["y"])))
                    U.add!(moments[key],val);r.outliers[i] || U.add!(acceptedmom[key],val)
                end
            end
            push!(pairedrows,(k,i,a,b,r.outliers[i],cl,id))
        end
    end
    artifacts["$tag-pair-associations.csv"]=P.csv_rows(("frame_a","match_index","detection_a","detection_b","outlier","classification","unique_associated_id"),pairedrows)
    correspondence=Dict("raw"=>metrics(raw;denominator=full),"accepted"=>metrics(accepted;denominator=full),
        "listed_adjacent_id_pairs"=>full,"both_operationally_localized_listed_pairs"=>both,
        "raw_same_listed_id_pair_error_px"=>Dict(k=>U.summary(v) for (k,v) in moments),
        "accepted_same_listed_id_pair_error_px"=>Dict(k=>U.summary(v) for (k,v) in acceptedmom))
    tracking=Dict{String,Any}[]
    for (gap,result) in tracks
        result.n_frames==n || err("tracking frame count differs")
        ct=counts();observations=Any[];edges=Any[];used=Set{Tuple{Int,Int}}();confirmed=Dict{Tuple{String,Int},Int}();edge_set=Set{Tuple{String,Int,Int}}()
        uniqueobs=0;ambobs=0;unknownobs=0
        for (tid,t) in enumerate(result.trajectories)
            length(t.x)==length(t.y)==length(t.frames) && length(t)>=2 && t.start_frame==first(t.frames) && all(k->1<=k<=n,t.frames) && all(>(0),diff(t.frames)) || err("malformed track")
            labels=Any[]
            for (q,k) in enumerate(t.frames)
                p=particles[k];matches=findall(j->isequal(p.x[j],t.x[q]) && isequal(p.y[j],t.y[q]),eachindex(p.x))
                length(matches)==1 || err("track observation needs one exact cached detection producer; missing/duplicate coordinates refused")
                i=only(matches)
                (k,i) in used && err("duplicate returned detection");push!(used,(k,i));l=assoc[k].labels[i];push!(labels,l)
                if l["status"]=="unique";uniqueobs+=1;confirmed[(l["id"],k)]=tid
                elseif l["status"]=="ambiguous";ambobs+=1
                else;unknownobs+=1;end
                push!(observations,(tid,q,k,t.x[q],t.y[q],i,l["status"],l["status"]=="unique" ? l["id"] : "",P.csv_ids(l["candidate_ids"])))
                if q>1
                    cl=edge_class(labels[q-1],l);ct[cl]+=1;id=cl=="associated_same_listed_id" ? l["id"] : ""
                    isempty(id) || push!(edge_set,(id,t.frames[q-1],k))
                    push!(edges,(tid,t.frames[q-1],k,cl,id))
                end
            end
        end
        ids=sort!(unique(r["id"] for r in c.rows));identityrows=Any[];relinkrows=Any[]
        rowmap=Dict((r["id"],r["frame"])=>r for r in c.rows)
        adjacent=0;recovered_adjacent=0;switches=0;fragments=0;episodes=0;unknownsamples=0;ambsamples=0
        for id in ids
            previous=0;seen=false;broken=false;trackids=Set{Int}();iswitch=0;irestart=0;iunknown=0;iamb=0
            ks=[k for k in 1:n if rowmap[(id,k)]["position_status"]=="provided"]
            for k in 1:n
                if rowmap[(id,k)]["position_status"]=="unknown"
                    iunknown+=1;previous=0;seen=false;broken=false;continue
                end
                q=truth_indices[k][id]
                if assoc[k].truth_ambiguous[q];iamb+=1;previous=0;seen=false;broken=false;continue;end
                tid=get(confirmed,(id,k),0)
                if tid==0;broken=seen
                else
                    push!(trackids,tid);previous!=0 && previous!=tid && (iswitch+=1)
                    seen && broken && (irestart+=1);previous=tid;seen=true;broken=false
                end
            end
            switches+=iswitch;fragments+=max(0,length(trackids)-1);episodes+=irestart;unknownsamples+=iunknown;ambsamples+=iamb
            push!(identityrows,(id,length(ks),iunknown,iamb,P.csv_ids(string.(sort!(collect(trackids)))),max(0,length(trackids)-1),iswitch,irestart))
            for j in 2:length(ks)
                a,b=ks[j-1],ks[j]
                if b==a+1;adjacent+=1;(id,a,b) in edge_set && (recovered_adjacent+=1)
                else
                    qa=truth_indices[a][id];qb=truth_indices[b][id]
                    ambiguous=assoc[a].truth_ambiguous[qa] || assoc[b].truth_ambiguous[qb]
                    push!(relinkrows,(id,a,b,b-a-1,id in localized_ids[a] && id in localized_ids[b],ambiguous,(id,a,b) in edge_set))
                end
            end
        end
        tm=metrics(ct;denominator=adjacent)
        # Only exact adjacent listed edges credit this recall; a long same-ID
        # edge across a missed listed row is not an extra adjacent true pair.
        delete!(tm,"listed_endpoint_recall_bounds")
        tm["listed_adjacent_edge_recall"]=P.ratio(recovered_adjacent,adjacent)
        tm["counts"]=merge(tm["counts"],Dict("returned_observations"=>length(used),"unique_observations"=>uniqueobs,
            "ambiguous_observations"=>ambobs,"unscorable_observations"=>unknownobs,"detections_not_retained"=>sum(length,particles)-length(used),
            "listed_adjacent_id_edges"=>adjacent,"associated_adjacent_edges_recovered"=>recovered_adjacent,
            "associated_same_track_identity_changes"=>ct["associated_different_listed_ids"],"target_output_id_changes_within_annotated_runs"=>switches,
            "extra_associated_track_ids_per_listed_identity"=>fragments,"restart_episodes_within_annotated_runs"=>episodes,
            "unknown_samples_resetting_identity_memory"=>unknownsamples,"ambiguous_samples_resetting_identity_memory"=>ambsamples))
        relink_counts=Dict("annotation_reappearance_events"=>length(relinkrows),"associated_exact_endpoint_links"=>count(r->r[7],relinkrows),
            "ambiguous_endpoint_events"=>count(r->r[6],relinkrows),"both_localized_events"=>count(r->r[5],relinkrows))
        tm["annotation_endpoint_relinking"]=Dict("counts"=>relink_counts,"observed_link_fraction"=>P.ratio(relink_counts["associated_exact_endpoint_links"],length(relinkrows)),
            "interpretation"=>"endpoint association diagnostic only; unknown middle positions/visibility; no true-absence or gap-recovery recall")
        push!(tracking,Dict("max_gap"=>gap,"metrics"=>tm))
        artifacts["$tag-track-$gap-associations.csv"]=P.csv_rows(("trajectory_id","observation_index","frame","x_px","y_px","detection_index","status","unique_associated_id","candidate_ids"),observations)
        artifacts["$tag-track-$gap-edges.csv"]=P.csv_rows(("trajectory_id","frame_a","frame_b","classification","unique_associated_id"),edges)
        artifacts["$tag-track-$gap-identities.csv"]=P.csv_rows(("truth_id","provided_samples","unknown_samples","ambiguous_samples","unique_associated_track_ids","extra_track_ids","changes_in_annotated_runs","restart_episodes_in_annotated_runs"),identityrows)
        artifacts["$tag-track-$gap-endpoint-relinking.csv"]=P.csv_rows(("truth_id","frame_a","frame_b","unknown_middle_frames","both_endpoints_localized","ambiguous_endpoints","associated_exact_endpoint_link"),relinkrows)
    end
    listed=sum(length,truth);predictions=sum(length,particles);localized=sum(a.matched for a in assoc)
    row=Dict("coordinate_offset_px"=>offset,"detection"=>Dict("counts"=>Dict("listed_annotation_rows"=>listed,"predictions"=>predictions,
        "operational_localized_listed_rows"=>localized,"unlocalized_listed_rows"=>listed-localized,"unassigned_predictions"=>predictions-localized,
        "ambiguous_predictions"=>sum(count(l->l["status"]=="ambiguous",a.labels) for a in assoc)),
        "listed_localization_recall"=>P.ratio(localized,listed)),"correspondence"=>correspondence,"tracking"=>tracking,
        "association_resources"=>[a.resources for a in assoc])
    (;row,artifacts)
end

function environment_record()
    env=P.environment_record()
    # Unlike the complete-v1 builder's historical hardcoded label, current
    # internal run_piv receives no override and follows its thread-count default.
    env["processing_threaded"]=Threads.nthreads()>1
    append!(env["source_files"],V.fixture_identity([@__FILE__,joinpath(@__DIR__,"prepare_vsj301.py")]))
    env
end

function run_study(clip;params=PTVParameters(),predictor=:piv,piv_passes=multipass_parameters([64,32]),
        max_gaps=GAPS,min_track_length=2,gate=.75,limits=(max_edges=200_000,max_component=512,max_cells=1_000_000))
    env=environment_record();c=checked_clip(deepcopy(clip))
    recipe=P.scientific_recipe(params,predictor,piv_passes,max_gaps,min_track_length)
    recipe["piv_driver_threaded_default"]=Threads.nthreads()>1
    finite(gate) && gate>0 || err("invalid localization gate")
    # All three registration hypotheses score these SAME production objects.
    # Known positions/IDs never enter PIV or tracking predictors.
    particles=[detect_particles(im,params) for im in c.images]
    pairs=[run_ptv(c.images[k],c.images[k+1],params;predictor,piv_passes) for k in 1:length(c.images)-1]
    tracks=[gap=>track_particles(c.images,params;predictor,piv_passes,min_track_length,max_gap=gap,progress=false) for gap in max_gaps]
    rows=Dict{String,Any}[];artifacts=Dict{String,String}()
    for offset in OFFSETS
        result=score(c,particles,pairs,tracks,offset;gate,limits)
        push!(rows,result.row);merge!(artifacts,result.artifacts)
    end
    artifacts["sparse-annotation-ledger.csv"]=P.csv_rows(("truth_id","frame","source_frame","position_status","visibility_status","source_x_px","source_y_px","world_x_cm","world_y_cm","world_z_cm","source_peak_intensity"),
        ((r["id"],r["frame"],r["frame"]-1,r["position_status"],r["visibility_status"],r["source_x_px"],r["source_y_px"],r["world_x_cm"],r["world_y_cm"],r["world_z_cm"],r["source_peak_intensity"]) for r in c.rows))
    for (k,r) in enumerate(pairs);artifacts["native-ptv-pair-$k.csv"]=P.native_table(r;frame_id=string(k-1));end
    for (gap,r) in tracks;artifacts["native-tracks-gap-$gap.csv"]=P.native_table(r;frame_id="VSJ301000-007");end
    checked_clip(c);stable=U.stable_environment(env);n=length(c.images)
    report=Dict{String,Any}("schema_version"=>SCHEMA,"generated_utc"=>string(now(UTC)),"environment"=>env,
        "source_and_environment_stable"=>stable,"input"=>c.metadata,"recipe"=>recipe,"registration_hypotheses"=>rows,
        "processing_outputs_shared_across_hypotheses"=>true,
        "registration_policy"=>Dict("coordinate_offsets_px"=>collect(OFFSETS),
            "offset0"=>"source coordinates interpreted directly in one-based processing coordinates; no shift control",
            "offset0p5"=>"hypothesis source coordinates are measured from upper/left pixel-area boundary; first center at0.5",
            "offset1"=>"hypothesis source coordinates are zero-based pixel-center indices; first center at0",
            "verification"=>"none of these interpretations verified by examined primary documentation; no winner selected"),
        "scoring"=>Dict("gate_px"=>gate,"limits"=>Dict(string(k)=>v for (k,v) in Base.pairs(limits)),
            "assignment"=>"independent in-gate graph connected components; maximum cardinality then minimum sum squared distance; deterministic sorted source IDs/detection indices",
            "precision"=>"conditional operational identification interval associated_same/N to(associated_same+ambiguous+unscorable)/N; all predictions retained; not confidence intervals",
            "detection"=>"operational localization against listed centroids; unassigned predictions not proved false; unlocalized listed rows not proved visibly detectable",
            "recall"=>"pair/track adjacent-frame ID denominator requires provided coordinates in both frames; never visibility or unannotated-interval recall",
            "fragmentation"=>"extra uniquely associated output track IDs per listed identity; unknown intervals may cause legitimate interruptions; changes/restarts additionally restricted to consecutive annotated runs",
            "relinking"=>"exact returned same-ID edge between successive provided endpoints separated by unknown rows; endpoint diagnostic only; not true gap-recovery recall"),
        "calls"=>Dict("standalone_detection"=>n,"ptv_pairs"=>n-1,"tracking"=>length(max_gaps),
            "detector_invocations"=>n+2(n-1)+length(max_gaps)*n,"matcher_transitions"=>(1+length(max_gaps))*(n-1),
            "full_piv_predictor_upper_bound"=>predictor===:piv ? n-1+length(max_gaps) : 0),
        "annotation_counts"=>Dict("identities"=>length(unique(r["id"] for r in c.rows)),"provided_positions"=>count(r->r["position_status"]=="provided",c.rows),
            "unknown_positions"=>count(r->r["position_status"]=="unknown",c.rows),"unknown_visibility"=>length(c.rows)),
        "limitations"=>["Independent synthetic source, not an independent real recording or visibility-complete ground truth.",
            "Origin unresolved; all three fixed hypotheses use identical production outputs; no best registration selected.",
            "Annotation membership changes do not establish laser-sheet exit, boundary crossing or missed visible detections.",
            "Gate-defined ambiguity cannot identify all contributors to merged intensity; unlisted faint contributors may remain.",
            "Source centroids rounded to0.01px; world coordinates to0.001cm. Same-listed-ID pair errors compare provided endpoints, not the Eulerian vector file.",
            "Intensity retained as original source peak units, not guaranteed observed image intensity or detectability.",
            "min_track_length filters singleton/internal candidates; reported loss includes linking and retention.",
            "First-use cache hashes are local identities, not publisher-signed checksums; cache/raw truth not redistributed.",
            "No true-absence gap recall, universal benchmark threshold, calibration claim or estimator/default change."])
    (;report,artifacts)
end

function markdown_report(r)
    io=IOBuffer();println(io,"# VSJ301 annotated-only evaluation\n\nIndependent synthetic frames000-007. Origin and visibility remain unverified. All hypotheses score identical processing outputs. No winning offset is selected. All unique IDs/classes and precision bounds are conditional on the gated annotation association, not proof of the physical intensity contributor.\n")
    println(io,"Precision intervals are conditional operational identification bounds, including unscorable/unannotated endpoints. They are not confidence intervals. Localization and recall denominators contain listed coordinates, not known visible particles.\n")
    fmt(q)=q["available"] ? "$(round(q["fraction"];sigdigits=5)) ($(q["numerator"])/$(q["denominator"]))" : "unavailable (0)"
    interval(q)=fmt(q["strict_lower"])*" to "*fmt(q["conservative_upper"])
    println(io,"| Offset px | Localized / predictions / listed rows | Raw pair S/D/A/U | Accepted pair S/D/A/U | Accepted precision bounds | Listed adjacent-pair recall bounds | Same-listed-ID accepted u/v RMS px |\n|---:|---:|---:|---:|---|---|---|")
    for h in r["registration_hypotheses"]
        d=h["detection"]["counts"];c=h["correspondence"];a=c["accepted"]
        parts(m)=join([m["counts"][k] for k in CLASSES],'/')
        rms(k)=c["accepted_same_listed_id_pair_error_px"][k]["available"] ? string(round(c["accepted_same_listed_id_pair_error_px"][k]["rms"];sigdigits=5)) : "unavailable"
        println(io,"| $(h["coordinate_offset_px"]) | $(d["operational_localized_listed_rows"])/$(d["predictions"])/$(d["listed_annotation_rows"]) | $(parts(c["raw"])) | $(parts(a)) | $(interval(a["all_prediction_precision_bounds"])) | $(interval(a["listed_endpoint_recall_bounds"])) | $(rms("u")) / $(rms("v")) |")
    end
    println(io,"\nS/D/A/U = associated same listed ID / associated different listed IDs / ambiguous / unscorable-unannotated.\n\n| Offset | max_gap | Edges S/D/A/U | Precision bounds | Exact adjacent listed-edge recall | Unique / ambiguous / unscorable observations | Detections not retained | Same-track changes / annotated-run output changes | Extra track IDs / annotated-run restarts | Endpoint links / reappearance events |\n|---:|---:|---:|---|---|---:|---:|---:|---:|---:|")
    for h in r["registration_hypotheses"],t in h["tracking"]
        m=t["metrics"];c=m["counts"];g=m["annotation_endpoint_relinking"]["counts"]
        println(io,"| $(h["coordinate_offset_px"]) | $(t["max_gap"]) | $(join([c[k] for k in CLASSES],'/')) | $(interval(m["all_prediction_precision_bounds"])) | $(fmt(m["listed_adjacent_edge_recall"])) | $(c["unique_observations"])/$(c["ambiguous_observations"])/$(c["unscorable_observations"]) | $(c["detections_not_retained"]) | $(c["associated_same_track_identity_changes"])/$(c["target_output_id_changes_within_annotated_runs"]) | $(c["extra_associated_track_ids_per_listed_identity"])/$(c["restart_episodes_within_annotated_runs"]) | $(g["associated_exact_endpoint_links"])/$(g["annotation_reappearance_events"]) |")
    end
    println(io,"\nReappearance rows do not establish genuine invisibility or gap recovery. Unknown/ambiguous memory resets, all input/source hashes, complete recipes, resource counts and ledgers are in TOML/CSV.\n")
    foreach(s->println(io,"- ",s),r["limitations"]);String(take!(io))
end

function write_report(directory,bundle)
    r=bundle.report;r["schema_version"]==SCHEMA && r["source_and_environment_stable"]===true && U.stable_environment(r["environment"]) || err("unstable/unsupported report")
    protected=String[]
    haskey(r["input"],"protected_cache_directory") && push!(protected,r["input"]["protected_cache_directory"])
    for f in get(r["input"],"file_identities",Any[])
        isfile(f["path"]) && !islink(f["path"]) && filesize(f["path"])==f["bytes"] && V.file_digest(f["path"])==f["sha256"] || err("changed consumed input")
        push!(protected,f["path"])
    end
    target=P.check_output(directory;protected_paths=protected)
    names=["vsj301.toml","vsj301.md",sort!(collect(keys(bundle.artifacts)))...]
    all(s->basename(s)==s && occursin(r"^[A-Za-z0-9_.-]+$",s),names) || err("unsafe output name")
    paths=joinpath.(target,names)
    for i in eachindex(paths)
        any(p->Hammerhead._artifact_alias(paths[i],p),protected) && err("output aliases input")
        any(j->Hammerhead._artifact_alias(paths[i],paths[j]),1:i-1) && err("output aliases output")
    end
    io=IOBuffer();TOML.print(io,r;sorted=true);contents=[String(take!(io)),markdown_report(r),[bundle.artifacts[s] for s in names[3:end]]...]
    mkpath(dirname(target));mkdir(target) # exclusive fresh directory; preserve partial failures
    for (p,text) in zip(paths,contents)
        ispath(p) || islink(p) ? err("occupied output destination") : nothing
        write(p,text)
    end
    paths
end

function main(args=ARGS)
    cache=joinpath(ROOT,"bench","profile-output","vsj301-cache");output=joinpath(ROOT,"bench","profile-output","validation-vsj301");python=nothing
    for arg in args
        if arg=="--help"
            println("julia --project=. --threads=1 bench/validation_vsj301.jl --cache=audited-cache --output=fresh-directory");return nothing
        elseif startswith(arg,"--cache=");cache=arg[9:end]
        elseif startswith(arg,"--output=");output=arg[10:end]
        elseif startswith(arg,"--python=");python=arg[10:end]
        else;err("unknown option $arg");end
    end
    clip=load_cache(cache;python);P.check_output(output;protected_paths=clip.protected_paths)
    bundle=run_study(clip);foreach(println,write_report(output,bundle));bundle
end
end
if abspath(PROGRAM_FILE)==@__FILE__
    ValidationVSJ301.main()
end
