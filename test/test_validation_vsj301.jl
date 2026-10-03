using Test, Hammerhead, TOML, Random
include(joinpath(@__DIR__,"..","bench","validation_vsj301.jl"))
const VSJC=ValidationVSJ301

function vsjc_clip(texts;shape=(32,32),images=[zeros(shape) for _ in texts])
    supplied=reduce(vcat,[VSJC.parse_particles(t,k) for (k,t) in enumerate(texts)];init=Dict{String,Any}[])
    rows=VSJC.sparse_ledger(supplied,length(texts))
    metadata=Dict{String,Any}("schema_version"=>VSJC.ANNOTATION_SCHEMA,"truth_sha256"=>Hammerhead._experiment_digest(rows),
        "frames"=>[Dict("ordinal"=>k,"decoded_sha256"=>VSJC.V.pixel_digest(im)) for (k,im) in enumerate(images)],"file_identities"=>Any[])
    VSJC.checked_clip(VSJC.SparseClip(images,rows,metadata,String[]))
end
function vsjc_particles(c)
    [begin
        t=VSJC.provided(c,k,0.);Particles([r["x"] for r in t],[r["y"] for r in t],ones(length(t)),fill(3.,length(t)))
    end for k in eachindex(c.images)]
end
function vsjc_pair(a,b,ia,ib;flags=falses(length(ia)))
    PTVResult(a.x[ia],a.y[ia],b.x[ib].-a.x[ia],b.y[ib].-a.y[ia],zeros(length(ia)),BitVector(flags),ia,ib,a,b,PTVParameters())
end
function vsjc_objective(edges,m)
    best=(-1,Inf)
    function visit(i,used,n,cost)
        if i>length(edges)
            if n>best[1] || n==best[1] && cost<best[2];best=(n,cost);end
            return
        end
        visit(i+1,used,n,cost)
        for (j,c) in edges[i]
            j in used && continue
            push!(used,j);visit(i+1,used,n+1,cost+c);delete!(used,j)
        end
    end
    visit(1,Set{Int}(),0,0.);best
end

@testset "Sparse graph objective: exhaustive independent partial injections" begin
    for n in 0:3,m in 0:3,mask in 0:(2^(n*m)-1)
        edges=[Tuple{Int,Float64}[] for _ in 1:n]
        for i in 1:n,j in 1:m
            mask & (1<<((i-1)*m+j-1))!=0 && push!(edges[i],(j,Float64(mod(i+2j,3))))
        end
        result=VSJC.graph_assignment(edges,m);map=result.map
        cost=sum(only(c for (q,c) in edges[i] if q==j) for (i,j) in enumerate(map) if j!=0;init=0.)
        @test (count(!=(0),map),cost)==vsjc_objective(edges,m)
        @test length(unique(filter(!=(0),map)))==count(!=(0),map)
    end
    edges=[[(1,0.)],[(2,0.)],Tuple{Int,Float64}[]]
    r=VSJC.graph_assignment(edges,3)
    @test r.map==[1,2,0]
    @test r.counts==Dict("edges"=>2,"components"=>2,"largest_component_vertices"=>2,"isolated_predictions"=>1,"isolated_annotations"=>1)
    @test_throws ArgumentError VSJC.graph_assignment(edges,3;max_edges=1)
    @test_throws ArgumentError VSJC.graph_assignment([[(1,0.),(2,1.)]],2;max_component=2)
    @test_throws ArgumentError VSJC.graph_assignment([[(1,0.),(2,1.)]],2;max_cells=1)
    @test_throws ArgumentError VSJC.graph_assignment([[(1,0.),(1,1.)]],1)
    @test_throws ArgumentError VSJC.graph_assignment([[(2,0.)]],1)
    @test_throws ArgumentError VSJC.graph_assignment([[(1,Inf)]],1)
end

@testset "Sparse unknown schema, RAW convention and contributor association" begin
    @test VSJC.decode_raw(UInt8[0,64,128,255,1,2],(2,3))==[0 64 128;255 1 2]./255
    @test_throws ArgumentError VSJC.decode_raw(UInt8[0],(2,3))
    for t in ("1 0 0 0 8 8", "0 0 0 0 8 8 4", "1 0 0 0 NaN 8 4", "1 0 0 0 8 8 -1", "1 0 0 0 8 8 4\n1 0 0 0 8 8 4")
        @test_throws ArgumentError VSJC.parse_particles(t,1)
    end
    c=vsjc_clip(["1 0 0 0 8 8 4", "", "1 0 0 0 10 8 4\n2 0 0 0 11 8 1"])
    @test length(c.rows)==6
    @test count(r->r["position_status"]=="unknown",c.rows)==3
    @test all(r->r["visibility_status"]=="unknown",c.rows)
    @test c.rows[3]["source_x_px"]=="unknown"
    io=IOBuffer();TOML.print(io,Dict("schema_version"=>VSJC.ANNOTATION_SCHEMA,"truth"=>c.rows));roundtrip=TOML.parse(String(take!(io)))
    @test roundtrip["truth"]==c.rows
    @test VSJC.provided(c,1,.5)[1]["x"]==8.5
    changed=deepcopy(c);changed.rows[3]["source_x_px"]=9.
    @test_throws ArgumentError VSJC.checked_clip(changed)
    truth=[Dict("id"=>"target","x"=>8.,"y"=>8.),Dict("id"=>"faint-contributor","x"=>8.4,"y"=>8.)]
    r=VSJC.associate([8.,20.],[8.,20.],truth)
    @test r.matched==1
    @test r.labels[1]["status"]=="ambiguous"
    @test r.labels[1]["candidate_ids"]==["target","faint-contributor"]
    @test r.labels[2]["status"]=="unscorable_unannotated"
    @test all(r.truth_ambiguous)
    @test_throws ArgumentError VSJC.associate([1e300],[1.],truth)
    @test_throws ArgumentError VSJC.associate([8.],[8.],truth;max_edges=1)
    a=Dict("status"=>"unique","id"=>"a");b=Dict("status"=>"unique","id"=>"b")
    @test VSJC.edge_class(a,a)=="associated_same_listed_id"
    @test VSJC.edge_class(a,b)=="associated_different_listed_ids"
    @test VSJC.edge_class(a,r.labels[2])=="unscorable_unannotated"
    m=VSJC.metrics(Dict("associated_same_listed_id"=>2,"associated_different_listed_ids"=>1,"ambiguous"=>1,"unscorable_unannotated"=>2);denominator=10)
    @test m["all_prediction_precision_bounds"]["strict_lower"]["fraction"]==2/6
    @test m["all_prediction_precision_bounds"]["conservative_upper"]["fraction"]==5/6
    @test m["listed_endpoint_recall_bounds"]["conservative_upper"]["fraction"]==3/10
    @test !VSJC.metrics(VSJC.counts())["all_prediction_precision_bounds"]["strict_lower"]["available"]
end

@testset "Unknown intervals reset switches; endpoint relinking is not absence truth" begin
    c=vsjc_clip(["1 0 0 0 8 8 4\n2 0 0 0 20 20 2", "1 0 0 0 9 8 4", "2 0 0 0 22 20 2", "1 0 0 0 11 8 4\n2 0 0 0 23 20 2"])
    particles=vsjc_particles(c)
    pairs=[vsjc_pair(particles[1],particles[2],[1],[1]),vsjc_pair(particles[2],particles[3],Int[],Int[]),vsjc_pair(particles[3],particles[4],[1],[2])]
    paths=[Trajectory{Float64}(1,[8.,9.,11.],[8.,8.,8.],[1,2,4]),Trajectory{Float64}(1,[20.,22.,23.],[20.,20.,20.],[1,3,4])]
    track=TrackingResult(paths,4,PTVParameters())
    r=VSJC.score(c,particles,pairs,[2=>track],0.)
    @test r.row["correspondence"]["listed_adjacent_id_pairs"]==2
    tm=only(r.row["tracking"])["metrics"];ct=tm["counts"]
    @test ct["associated_same_listed_id"]==4
    @test ct["associated_adjacent_edges_recovered"]==2
    @test ct["unknown_samples_resetting_identity_memory"]==2
    @test ct["target_output_id_changes_within_annotated_runs"]==0
    @test tm["listed_adjacent_edge_recall"]["fraction"]==1
    @test tm["annotation_endpoint_relinking"]["counts"]["annotation_reappearance_events"]==2
    @test tm["annotation_endpoint_relinking"]["counts"]["associated_exact_endpoint_links"]==2
    @test !haskey(tm,"gap_recovery")
    @test !haskey(tm,"full_edge_recall_bounds")
    splittrack=TrackingResult([Trajectory{Float64}(1,[8.,9.],[8.,8.],[1,2]),Trajectory{Float64}(3,[22.,11.],[20.,8.],[3,4])],4,PTVParameters())
    st=VSJC.score(c,particles,pairs,[0=>splittrack],0.).row["tracking"][1]["metrics"]["counts"]
    @test st["associated_different_listed_ids"]==1
    @test st["target_output_id_changes_within_annotated_runs"]==0
    @test st["extra_associated_track_ids_per_listed_identity"]==1
    bad=copy(pairs);bad[1]=vsjc_pair(particles[1],particles[2],[1,1],[1,1])
    @test_throws ArgumentError VSJC.score(c,particles,bad,[2=>track],0.)
    @test_throws ArgumentError VSJC.score(c,particles,pairs,[2=>track],true)
    @test_throws ArgumentError VSJC.score(c,particles,pairs,[2=>TrackingResult([paths[1],paths[1]],4,PTVParameters())],0.)
    dupeclip=vsjc_clip(["1 0 0 0 8 8 4","1 0 0 0 9 8 4"])
    dupeparticles=[Particles([8.,8.],[8.,8.],ones(2),fill(3.,2)),Particles([9.,9.],[8.,8.],ones(2),fill(3.,2))]
    dupepairs=[vsjc_pair(dupeparticles[1],dupeparticles[2],Int[],Int[])]
    dupetrack=TrackingResult([Trajectory{Float64}(1,[8.,9.],[8.,8.],[1,2])],2,PTVParameters())
    @test_throws ArgumentError VSJC.score(dupeclip,dupeparticles,dupepairs,[0=>dupetrack],0.)
end

@testset "Cheap actual production outputs shared by all registration hypotheses" begin
    images=[zeros(64,64) for _ in 1:3];texts=String[]
    for k in 1:3
        lines=String[]
        for (i,(x,y)) in enumerate(((20.,20.),(40.,20.),(20.,40.),(40.,40.)))
            xx=x+.5(k-1);Hammerhead.SyntheticData.generate_gaussian_particle!(images[k],(xx,y),3.,1.)
            push!(lines,"$i 0 0 0 $xx $y 240")
        end
        push!(texts,join(lines,'\n'))
    end
    c=vsjc_clip(texts;shape=(64,64),images)
    params=PTVParameters(uod_enable=false)
    bundle=VSJC.run_study(c;params,predictor=nothing,max_gaps=(0,1))
    @test bundle.report["processing_outputs_shared_across_hypotheses"]===true
    @test [r["coordinate_offset_px"] for r in bundle.report["registration_hypotheses"]]==[0.,.5,1.]
    @test bundle.report["calls"]==Dict("standalone_detection"=>3,"ptv_pairs"=>2,"tracking"=>2,"detector_invocations"=>13,"matcher_transitions"=>6,"full_piv_predictor_upper_bound"=>0)
    @test bundle.report["annotation_counts"]["unknown_visibility"]==12
    @test bundle.report["recipe"]["piv_driver_threaded_default"]==(Threads.nthreads()>1)
    @test bundle.report["environment"]["processing_threaded"]==(Threads.nthreads()>1)
    @test count(startswith("native-"),keys(bundle.artifacts))==4
    @test haskey(bundle.artifacts,"sparse-annotation-ledger.csv")
    @test occursin("conditional operational",VSJC.markdown_report(bundle.report))
    # Unchanged complete-v1 lane still uses its original dense assignment and
    # complete visibility contract. No sparse interpretation is injected.
    v1=VSJC.P.synthetic_clip(7321,"clean";size=64,nframes=5)
    original=deepcopy(v1.rows);VSJC.P.checked_clip(v1)
    @test v1.rows==original
    @test VSJC.P.ANNOTATION_SCHEMA=="hammerhead-annotated-particle-clip-1"
    @test_throws ArgumentError VSJC.P.validate_truth(c.rows,3,(64,64))
    mktempdir() do directory
        output=joinpath(directory,"report")
        @test VSJC.P.check_output(output)==output
        # No independent evidence claims from mutable cross-agent identity:
        # publication uses a fresh checked environment only for this protocol fixture.
        bundle.report["environment"]=VSJC.environment_record()
        bundle.report["source_and_environment_stable"]=true
        paths=VSJC.write_report(output,bundle);before=read(first(paths))
        @test TOML.parsefile(first(paths))["schema_version"]==VSJC.SCHEMA
        @test_throws ArgumentError VSJC.write_report(output,bundle)
        @test read(first(paths))==before
        @test_throws ArgumentError VSJC.P.check_output(output;protected_paths=[first(paths)])
        @test_throws ArgumentError VSJC.P.check_output(joinpath(first(paths),"descendant");protected_paths=[first(paths)])
        bad=deepcopy(bundle);bad.artifacts["../escape.csv"]="bad"
        @test_throws ArgumentError VSJC.write_report(joinpath(directory,"bad"),bad)
        @test !ispath(joinpath(directory,"bad"))
    end
end

@testset "Cache/schema preflight refuses before interpreter/output access" begin
    mktempdir() do directory
        @test_throws ArgumentError VSJC.load_cache(directory;python="missing-interpreter")
        path=joinpath(directory,"cache.toml")
        write(path,"schema_version = \"unknown\"\n")
        @test_throws ArgumentError VSJC.load_cache(directory;python="missing-interpreter")
        write(path,repeat("x",128001))
        @test_throws ArgumentError VSJC.load_cache(directory;python="missing-interpreter")
        @test filesize(path)==128001
    end
end

function vsjc_python_interpreter()
    candidates=unique(filter(!isnothing,[Sys.which("python"),Sys.which("python3")]))
    for executable in candidates
        # Windows app-execution aliases can launch the Store instead of Python.
        # The optional core fixture must not open an interactive installer.
        Sys.iswindows() && occursin(r"(?i)[\\/]WindowsApps[\\/]",executable) && continue
        probe=Cmd(Cmd([executable,"-c","import sys, tomllib; assert sys.version_info >= (3, 11)"]);windows_hide=true)
        process=try
            run(pipeline(probe,stdout=devnull,stderr=devnull);wait=false)
        catch
            continue
        end
        status=timedwait(()->process_exited(process),5.0;pollint=.05)
        if status!=:ok
            try;kill(process,Base.SIGKILL);catch;end
            wait(process)
            continue
        end
        success(process) && return executable
    end
    ""
end

@testset "Python cache acquisition/audit guards (stdlib fixture; no network)" begin
    python=vsjc_python_interpreter()
    if isempty(python)
        @test_skip "Python3.11+ with tomllib unavailable; optional bench-only cache fixture skipped"
    else
        script=raw"""
import importlib.util,sys,tempfile,io,zipfile,pathlib,unittest.mock as mock
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('preparer',sys.argv[1]);p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
checks=0
def check(value):
 global checks
 assert value;checks+=1
def refused(f):
 global checks
 try:f()
 except (ValueError,FileNotFoundError):checks+=1;return
 raise AssertionError('expected refusal')
def archive(prefix,ext,body):
 stream=io.BytesIO()
 with zipfile.ZipFile(stream,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for i in range(145):z.writestr(f'{prefix}{i:03d}.{ext}',body)
 return stream.getvalue()
raw=archive('img','raw',bytes(65536));ptc=archive('ptc','dat',b'1 0 0 0 8 8 240\n')
check(len(p.archive_members(raw,'301_raw.zip'))==8)
check(len(p.archive_members(ptc,'301_ptc.zip'))==8)
bad=io.BytesIO()
with zipfile.ZipFile(bad,'w') as z:z.writestr('../escape',b'x')
refused(lambda:p.archive_members(bad.getvalue(),'301_raw.zip'))
refused(lambda:p.check_terms(b'<html>permission missing</html>'))
payload={p.BASE:b'<p>'+p.PERMISSION.encode()+b'</p>',p.TERMS['dataset.html']:b'dataset',p.TERMS['particle-format.html']:b'format',p.BASE+'image3d/301_raw.zip':raw,p.BASE+'image3d/301_ptc.zip':ptc}
p.ARCHIVES={'301_raw.zip':len(raw),'301_ptc.zip':len(ptc)}
calls=[]
def fetch(url,bound):
 calls.append(url);check(len(payload[url])<=bound)
 return payload[url],url,{'ETag':'fixture','Last-Modified':'fixture','Content-Type':'fixture'}
p.fetch=fetch
with tempfile.TemporaryDirectory() as td:
 cache=pathlib.Path(td)/'cache'
 data=p.acquire(cache);check(len(data['members'])==16);check(len(calls)==5)
 before=(cache/'img000.raw').read_bytes()
 refused(lambda:p.acquire(cache));check((cache/'img000.raw').read_bytes()==before)
 p.fetch=lambda *a:(_ for _ in ()).throw(AssertionError('offline touched network'))
 check(p.audit(cache)['schema_version']==p.SCHEMA)
 (cache/'unknown.txt').write_text('unknown');refused(lambda:p.audit(cache));(cache/'unknown.txt').unlink()
 source=cache/'img000.raw';source.write_bytes(b'changed');refused(lambda:p.audit(cache));source.write_bytes(before)
 huge=pathlib.Path(td)/'huge'
 with huge.open('wb') as s:s.truncate(12000001)
 with mock.patch('pathlib.Path.open',side_effect=AssertionError('oversized file must not be opened')):
  refused(lambda:p.file_identity(huge))
 manifest=cache/'cache.toml';saved=manifest.read_text();manifest.write_text(saved.replace('ordinal = 0','ordinal = 0'))
 manifest.write_text(saved.replace('source_frames = [0,','source_frames = [false,'));refused(lambda:p.audit(cache));manifest.write_text(saved)
 with mock.patch('os.path.commonpath',side_effect=ValueError('distinct volumes')):
  check(p.check_new_cache(pathlib.Path(td)/'different-volume')==pathlib.Path(td)/'different-volume')
 refused(lambda:p.check_new_cache(p.ROOT/'bench'/'unignored-cache'))
print('Python cache guard assertions:',checks)
"""
        helper=joinpath(@__DIR__,"..","bench","prepare_vsj301.py")
        fixture=Cmd(Cmd([python,"-",helper]);windows_hide=true)
        text=read(pipeline(fixture,stdin=IOBuffer(script)),String)
        @test occursin("Python cache guard assertions:",text)
        @test parse(Int,last(split(strip(text))))>=20
    end
end
