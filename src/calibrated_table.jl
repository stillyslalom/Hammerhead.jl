const CALIBRATED_TABLE_FORMAT_VERSION = 1
const _CALIBRATED_TABLE_SCHEMA = "hammerhead-calibrated-scattered-table-1"
const _CALIBRATED_TABLE_COLUMNS = (TIMED_TRACKING_TABLE_COLUMNS..., "vector_quantity", "vector_unit",
    "match_residual_pixel", "match_residual_pixel_unit")
_cal_error(message) = throw(ArgumentError(message))

# TOML can change vector element types (notably empty arrays). Integrity covers
# primitive values/shape, while the schema independently enforces scalar types.
function _cal_canonical(value)
    value isa AbstractDict && return Dict{String,Any}(k=>_cal_canonical(v) for (k,v) in value)
    value isa AbstractVector && return Any[_cal_canonical(v) for v in value]
    value
end
_cal_digest(value) = _history_digest(_cal_canonical(value))

"""
    CalibratedTableMetadata

Detached metadata for one calibrated CSV/TOML pair. No result, image, trajectory
or CSV row payload is retained. Use [`calibrated_table_data`](@ref) for copied
settings and past verification status. Integrity and CSV verification do not
authenticate inputs, verify numerical results, or prove a common physical frame.
"""
struct CalibratedTableMetadata
    _data::Dict{String,Any}
    _sha256::String
    _metadata_path::String
    _csv_path::String
    _csv_verified::Bool
end

_cal_optional(value) = value === nothing ? Dict{String,Any}("available"=>false) :
    Dict{String,Any}("available"=>true,"value"=>value)
function _cal_optional_value(data, validate)
    data isa AbstractDict && get(data,"available",nothing) isa Bool || _cal_error("invalid optional metadata")
    _experiment_keys(data,data["available"] ? ["available","value"] : ["available"],"optional calibrated metadata")
    data["available"] || return nothing
    validate(data["value"])
    data["value"]
end
_cal_label(value) = value isa String && !isempty(value) ? value : _cal_error("metadata labels must be nonempty strings")
_cal_input_label(value) = value isa AbstractString ? _cal_label(String(value)) : _cal_error("labels must be strings")

# TOML has no null value. Use an unambiguous private marker inside the existing
# tracking-timing schema; its own validator checks the decoded snapshot.
function _cal_pack(value)
    value === nothing && return Dict("__calibrated_null__"=>"nothing")
    value isa AbstractDict && return Dict{String,Any}(k=>_cal_pack(v) for (k,v) in value)
    value isa AbstractVector && return [_cal_pack(v) for v in value]
    value
end
function _cal_unpack(value)
    value isa AbstractDict && Set(keys(value))==Set(["__calibrated_null__"]) &&
        value["__calibrated_null__"]=="nothing" && return nothing
    value isa AbstractDict && return Dict{String,Any}(k=>_cal_unpack(v) for (k,v) in value)
    value isa AbstractVector && return [_cal_unpack(v) for v in value]
    value
end

function _cal_transform(transform)
    transform isa PlanarTransform || _cal_error("transform must be a PlanarTransform")
    originals = [collect(vec(transform.matrix));collect(transform.offset)]
    all(v->v isa Union{Float16,Float32,Float64,BigFloat} && isfinite(v),originals) ||
        _cal_error("transform coefficients must be finite supported floating values")
    applied = Float64.(originals)
    all(i->isfinite(applied[i]) && (originals[i]==0 || applied[i]!=0),eachindex(applied)) ||
        _cal_error("transform normalization overflows or underflows Float64")
    matrix = [applied[1] applied[3];applied[2] applied[4]]
    _cal_nonsingular(matrix)
    data = Dict{String,Any}("matrix"=>[collect(matrix[1,:]),collect(matrix[2,:])],"offset"=>applied[5:6],
        "original_numeric_type"=>string(eltype(transform.matrix)),
        "original_precision_bits"=>Int[precision(v) for v in originals],"applied_numeric_type"=>"Float64")
    matrix,applied[5:6],data
end
function _cal_nonsingular(matrix)
    # Exact determinant of the applied floating coefficients: a valid tiny
    # determinant must not be rejected just because floating det underflows.
    a,b,c,d = Rational{BigInt}.((matrix[1,1],matrix[1,2],matrix[2,1],matrix[2,2]))
    a*d-b*c != 0 || _cal_error("applied transform is singular")
    nothing
end
function _cal_float(exact)
    value = Float64(exact)
    isfinite(value) && (exact==0 || value!=0) || _cal_error("calibrated output overflows or underflows Float64")
    value
end
function _cal_affine(matrix,offset,x,y)
    if !isfinite(x) || !isfinite(y)
        return Tuple(matrix*[Float64(x),Float64(y)]+offset)
    end
    ex,ey = Rational{BigInt}(x),Rational{BigInt}(y)
    ntuple(i->_cal_float(Rational{BigInt}(matrix[i,1])*ex+Rational{BigInt}(matrix[i,2])*ey+Rational{BigInt}(offset[i])),2)
end
function _cal_vector(matrix,dx,dy,interval)
    if !isfinite(dx) || !isfinite(dy)
        return Tuple((matrix*[Float64(dx),Float64(dy)])./Float64(interval))
    end
    ex,ey = Rational{BigInt}(dx),Rational{BigInt}(dy)
    ntuple(i->_cal_float((Rational{BigInt}(matrix[i,1])*ex+Rational{BigInt}(matrix[i,2])*ey)/interval),2)
end
function _cal_secant(matrix,t,a,b,interval)
    if all(isfinite,(t.x[a],t.x[b],t.y[a],t.y[b]))
        dx=Rational{BigInt}(t.x[b])-Rational{BigInt}(t.x[a])
        dy=Rational{BigInt}(t.y[b])-Rational{BigInt}(t.y[a])
    else
        dx=Float64(t.x[b])-Float64(t.x[a]);dy=Float64(t.y[b])-Float64(t.y[a])
    end
    _cal_vector(matrix,dx,dy,interval)
end

function _cal_check_ptv(r)
    n=length(r.x)
    all(a->length(a)==n,(r.y,r.u,r.v,r.match_residual,r.outliers,r.index_a,r.index_b)) || _cal_error("PTV match arrays have inconsistent lengths")
    for p in (r.particles_a,r.particles_b)
        length(p.x)==length(p.y)==length(p.intensity)==length(p.diameter) || _cal_error("PTV detection arrays have inconsistent lengths")
    end
    all(i->1<=i<=length(r.particles_a),r.index_a) && all(i->1<=i<=length(r.particles_b),r.index_b) ||
        _cal_error("PTV match indices are outside their detection arrays")
    nothing
end
function _cal_result_digest(r::PTVResult)
    _history_digest(Dict{String,Any}(String(k)=>getfield(r,k) for k in
        (:x,:y,:u,:v,:match_residual,:outliers,:index_a,:index_b)))
end
_cal_result_digest(r::TrackingResult) = _tracking_result_digest(r)
_cal_result_digest(r::TimedTrackingResult) = (_tracking_check(r); _history_digest(r.timing._data))

function _cal_prepare(result,transform,length_unit,coordinate_frame,dt,time_unit,protected_paths,labels)
    result isa Union{PTVResult,TrackingResult,TimedTrackingResult} || _cal_error("calibrated table supports PTV and tracking only")
    lu=_cal_input_label(length_unit); frame=_cal_input_label(coordinate_frame)
    matrix,offset,transform_data=_cal_transform(transform)
    timing = result isa TimedTrackingResult ? _tracking_check(result) : nothing
    raw = result isa TimedTrackingResult ? result.result : result
    raw.scale===nothing || _cal_error("calibrated export requires raw pixels with no attached scale")
    timing===nothing || timing["position_basis"]=="pixels" || _cal_error("timed positions are already converted")
    raw isa PTVResult ? _cal_check_ptv(raw) : _check_tracking_table(raw)
    label = time_unit===nothing ? nothing : _cal_input_label(time_unit)
    encoded_interval = dt===nothing ? nothing : _timing_number(dt)
    interval = encoded_interval===nothing ? big(1)//big(1) : _timing_positive(encoded_interval)
    times = timing===nothing ? nothing : [_timing_decode(v) for v in timing["sample_times"]]
    if timing!==nothing
        dt===nothing || _cal_error("timed tracking uses actual times; dt is not accepted")
        known=timing["effective_time_unit"]
        label===nothing || known===nothing || label==known || _cal_error("export time unit conflicts with chosen timeline")
        unit=label===nothing ? known : label
        unit_source=known===nothing ? (label===nothing ? "unknown" : "explicit_export_time_unit_assumption") : "chosen_timeline"
        basis="actual_sample_times";quantity="actual_time_secant"
    elseif encoded_interval!==nothing
        label===nothing && _cal_error("dt requires a nonempty time_unit")
        unit=label;unit_source="explicit_interval_label"
        basis=raw isa PTVResult ? "explicit_pair_delay" : "explicit_uniform_interval"
        quantity=raw isa PTVResult ? "pair_mean_velocity" : "uniform_interval_secant"
    else
        label===nothing || _cal_error("time_unit requires dt for ordinal/PTV data")
        unit=raw isa PTVResult ? nothing : "frame"
        unit_source=raw isa PTVResult ? "unknown" : "selected_frame_coordinate"
        basis=raw isa PTVResult ? "pair_displacement" : "ordinal_frames"
        quantity=raw isa PTVResult ? "pair_displacement" : "ordinal_frame_secant"
    end
    vector_unit = quantity=="pair_displacement" ? lu : unit===nothing ? nothing : lu*"/"*unit
    all(p->p isa AbstractString,protected_paths) || _cal_error("protected_paths must contain paths")
    protected=_artifact_local_path.(String.(collect(protected_paths)))
    timing===nothing || append!(protected,timing["protected_paths"])
    frozen_labels=Dict{String,String}(k=>String(getproperty(labels,Symbol(k))) for k in ("frame_id","source_a","source_b"))
    policy=Dict{String,Any}("basis"=>basis,"quantity"=>quantity,"unit"=>_cal_optional(unit),
        "unit_provenance"=>unit_source,"interval"=>_cal_optional(encoded_interval))
    (result=result,raw=raw,timing=timing,times=times,matrix=matrix,offset=offset,transform=transform_data,
        length_unit=lu,coordinate_frame=frame,interval=interval,unit=unit,vector_unit=vector_unit,policy=policy,
        protected=sort!(unique(protected)),labels=frozen_labels,signature=_cal_result_digest(result))
end

function _cal_row(context,point_id)
    row=Dict{String,Any}(k=>missing for k in _CALIBRATED_TABLE_COLUMNS)
    merge!(row,context.labels)
    row["schema_version"]=_CALIBRATED_TABLE_SCHEMA
    row["result_type"]=context.raw isa PTVResult ? "ptv" : "tracking"
    row["point_id"]=point_id;row["length_unit"]=context.length_unit
    row["time_unit"]=something(context.unit,missing)
    row["velocity_unit"]=context.policy["quantity"]=="pair_displacement" ? missing : something(context.vector_unit,missing)
    row["vector_quantity"]=context.policy["quantity"];row["vector_unit"]=something(context.vector_unit,missing)
    row
end
function _cal_emit(io,row)
    io===nothing || println(io,join((_csv(row[k]) for k in _CALIBRATED_TABLE_COLUMNS),','))
    nothing
end
function _cal_rows(io,c)
    io===nothing || println(io,join(_CALIBRATED_TABLE_COLUMNS,','))
    n=0
    if c.raw isa PTVResult
        r=c.raw
        for k in eachindex(r.x)
            n+=1;row=_cal_row(c,n)
            row["x"],row["y"]=_cal_affine(c.matrix,c.offset,r.x[k],r.y[k])
            row["u"],row["v"]=_cal_vector(c.matrix,r.u[k],r.v[k],c.interval)
            row["masked"]=false;row["outlier"]=r.outliers[k]
            row["index_a"]=r.index_a[k];row["index_b"]=r.index_b[k]
            row["match_residual_pixel"]=r.match_residual[k];row["match_residual_pixel_unit"]="px"
            row["position_valid"]=isfinite(row["x"]) && isfinite(row["y"])
            row["velocity_valid"]=row["position_valid"] && isfinite(row["u"]) && isfinite(row["v"])
            row["time_provenance"]=c.policy["basis"]
            _cal_emit(io,row)
        end
    else
        for (id,t) in enumerate(c.raw.trajectories), k in eachindex(t.x)
            n+=1;row=_cal_row(c,n);f=t.frames[k]
            row["x"],row["y"]=_cal_affine(c.matrix,c.offset,t.x[k],t.y[k])
            row["trajectory_id"]=id;row["observation_id"]=k;row["frame_index"]=f
            row["gap_before"]=k==1 ? 0 : f-t.frames[k-1]-1
            elapsed=c.times===nothing ? (f-1)*c.interval : c.times[f]-c.times[1]
            ef=Float64(elapsed)
            row["elapsed_time"]=isfinite(ef) && (elapsed==0 || ef!=0) ? ef : missing
            row["elapsed_time_numerator"]=string(numerator(elapsed));row["elapsed_time_denominator"]=string(denominator(elapsed))
            row["time_provenance"]=c.policy["basis"]
            row["position_valid"]=isfinite(row["x"]) && isfinite(row["y"])
            row["velocity_valid"]=false
            if length(t)>=2
                a,b=k==1 ? (1,2) : k==length(t) ? (k-1,k) : (k-1,k+1)
                duration=c.times===nothing ? (t.frames[b]-t.frames[a])*c.interval : c.times[t.frames[b]]-c.times[t.frames[a]]
                row["u"],row["v"]=_cal_secant(c.matrix,t,a,b,duration)
                row["velocity_start_frame"]=t.frames[a];row["velocity_end_frame"]=t.frames[b]
                row["velocity_valid"]=row["position_valid"] && isfinite(row["u"]) && isfinite(row["v"])
            end
            if c.timing!==nothing
                sample=c.timing["sample_times"][f];source=c.timing["frames"][f]
                for (column,key) in (("sample_time_numerator","numerator"),("sample_time_denominator","denominator"),("sample_time_numeric_type","numeric_type"))
                    row[column]=sample[key]
                end
                row["clock_id"]=something(c.timing["clock_id"],missing)
                for (column,key) in (("source_id","source_id"),("source_frame_id","frame_id"),("source_frame_index","frame_index"),
                    ("source_label","label"),("acquisition_time_unit","time_unit"),("acquisition_clock_id","clock_id"))
                    row[column]=something(source[key],missing)
                end
                row["effective_time_unit_provenance"]=c.policy["unit_provenance"]
            end
            _cal_emit(io,row)
        end
    end
    n
end

_cal_prospective_path(path) = _artifact_prospective_path(path)
_cal_alias(a,b) = _artifact_alias(a,b)

function _cal_guard(csv,metadata,protected,overwrite)
    _cal_alias(csv,metadata) && _cal_error("CSV and metadata destinations alias each other")
    for path in (csv,metadata)
        islink(path) && !ispath(path) && _cal_error("dangling output symlinks are not supported")
        ispath(path) && !isfile(path) && _cal_error("calibrated output destination must be a file")
        any(source->_artifact_alias(path,source;allow_unavailable_other=true),_artifact_local_protected_paths(protected)) &&
            _cal_error("calibrated destination aliases a protected local source")
        !overwrite && (ispath(path)||islink(path)) && _cal_error("destination exists; use overwrite=true for unrelated outputs")
    end
    nothing
end

"""
    export_calibrated_table(csv_path, result; transform, length_unit, coordinate_frame,
        metadata_path=csv_path*".metadata.toml", dt=nothing, time_unit=nothing,
        overwrite=false, protected_paths=[], frame_id="", source_a="", source_b="")

Write a raw PTV/ordinal-tracking/timed-tracking result to a calibrated CSV plus
version-1 TOML metadata, returning `(csv_path, metadata_path)`. Ordinary
`export_table` behavior is unchanged. Positions use `A*p+b`; vectors use `A*d`.
PTV without dt exports displacement, with dt pair mean velocity. Ordinal tracking
dt denotes a uniform selected-frame interval; timed tracking rejects dt and uses
exact actual secant durations. Unknown time units remain unknown unless explicitly
labeled as an export assumption. Attached scales and converted timed positions
are refused. Original validator decisions and raw-pixel residuals are preserved;
transformed scalar residuals/UQ are unavailable. `coordinate_frame` is an opaque
provided label, not proof of a shared physical frame.

Validate all arithmetic and known aliases before output. Both files are prepared
and closed before sequential publication; publication of the pair is not atomic.
Default overwrite=false refuses existing destinations. An I/O failure can leave
an old or incomplete pair; the metadata reader detects missing/mismatched CSVs.
Concurrent writers/externally changing inputs are unsupported. Literal input paths
not carried by a bare result must be supplied as receiving-host paths in
protected_paths. Foreign recorded timed locators remain verbatim provenance;
they are never interpreted as local sources or relocated implicitly.
"""
function export_calibrated_table(csv_path::AbstractString,result; transform,length_unit,coordinate_frame,
    metadata_path::AbstractString=csv_path*".metadata.toml",dt=nothing,time_unit=nothing,
    overwrite::Bool=false,protected_paths=String[],frame_id="",source_a="",source_b="")
    csv,metadata=_artifact_local_path(csv_path),_artifact_local_path(metadata_path)
    c=_cal_prepare(result,transform,length_unit,coordinate_frame,dt,time_unit,protected_paths,(;frame_id,source_a,source_b))
    _cal_guard(csv,metadata,c.protected,overwrite)
    n=_cal_rows(nothing,c)
    _cal_result_digest(result)==c.signature || _cal_error("source result changed during preflight")
    mkpath(dirname(csv));mkpath(dirname(metadata))
    csvtemp,handle=mktemp(dirname(csv));close(handle)
    metatemp=""
    try
        open(csvtemp,"w") do io
            _cal_rows(io,c)==n || _cal_error("row count changed while exporting")
        end
        _cal_result_digest(result)==c.signature || _cal_error("source result changed during export")
        kind=result isa TimedTrackingResult ? "timed_tracking" : result isa TrackingResult ? "tracking" : "ptv"
        geometry=Dict{String,Any}("source_basis"=>"original_image_pixels_x_columns_y_rows", "output_coordinate_frame"=>c.coordinate_frame,
            "coordinate_frame_provenance"=>"opaque_user_label","length_unit"=>c.length_unit,"transform"=>c.transform)
        data=Dict{String,Any}("calibrated_table_format_version"=>CALIBRATED_TABLE_FORMAT_VERSION,
            "csv_schema"=>_CALIBRATED_TABLE_SCHEMA,"csv_relative_path"=>relpath(csv,dirname(metadata)),
            "csv_sha256"=>_experiment_file_digest(csvtemp),"row_count"=>n,"result_kind"=>kind,
            "geometry"=>geometry,"geometry_sha256"=>_cal_digest(geometry),"time"=>c.policy,
            "timing_snapshot"=>_cal_optional(c.timing===nothing ? nothing : _cal_pack(c.timing)),
            "diagnostics"=>Dict{String,Any}("outlier_basis"=>kind=="ptv" ? "original_pixel_validation" : "not_recorded",
                "match_residual"=>kind=="ptv" ? "unavailable_direction_not_retained" : "not_recorded","raw_pixel_residual"=>kind=="ptv", "uncertainty"=>"not_retained"),
            "protected_locators"=>c.protected)
        _cal_validate(data)
        wrapper=Dict("metadata"=>data,"metadata_sha256"=>_cal_digest(data))
        metatemp,handle=mktemp(dirname(metadata));close(handle)
        open(metatemp,"w") do io;TOML.print(io,wrapper;sorted=true);end
        _cal_guard(csv,metadata,c.protected,overwrite)
        _cal_result_digest(result)==c.signature || _cal_error("source result changed before publication")
        mv(csvtemp,csv;force=overwrite)
        _cal_alias(csv,metadata) && _cal_error("CSV and metadata alias after CSV publication")
        mv(metatemp,metadata;force=overwrite)
    finally
        ispath(csvtemp) && rm(csvtemp;force=true)
        isempty(metatemp) || !ispath(metatemp) || rm(metatemp;force=true)
    end
    (csv_path=csv,metadata_path=metadata)
end

function _cal_validate(data)
    _experiment_keys(data,["calibrated_table_format_version","csv_schema","csv_relative_path","csv_sha256","row_count","result_kind",
        "geometry","geometry_sha256","time","timing_snapshot","diagnostics","protected_locators"],"calibrated table metadata")
    data["calibrated_table_format_version"] isa Int && data["calibrated_table_format_version"]==CALIBRATED_TABLE_FORMAT_VERSION || _cal_error("unsupported calibrated table version")
    data["csv_schema"]==_CALIBRATED_TABLE_SCHEMA || _cal_error("unsupported calibrated CSV schema")
    data["csv_relative_path"] isa String && _artifact_relative_locator(data["csv_relative_path"]) || _cal_error("CSV location must be relative to its metadata")
    _experiment_hash(data["csv_sha256"]) || _cal_error("invalid CSV digest")
    data["row_count"] isa Int && data["row_count"]>=0 || _cal_error("invalid CSV row count")
    kind=data["result_kind"]
    kind in ("ptv","tracking","timed_tracking") || _cal_error("unsupported calibrated result kind")
    g=data["geometry"]
    _experiment_keys(g,["source_basis","output_coordinate_frame","coordinate_frame_provenance","length_unit","transform"],"calibrated geometry")
    g["source_basis"]=="original_image_pixels_x_columns_y_rows" && g["coordinate_frame_provenance"]=="opaque_user_label" || _cal_error("unsupported coordinate convention")
    _cal_label(g["output_coordinate_frame"]);_cal_label(g["length_unit"])
    t=g["transform"]
    _experiment_keys(t,["matrix","offset","original_numeric_type","original_precision_bits","applied_numeric_type"],"calibrated transform")
    m=t["matrix"];o=t["offset"]
    m isa AbstractVector && length(m)==2 && all(r->r isa AbstractVector && length(r)==2 && all(v->v isa Float64 && isfinite(v),r),m) || _cal_error("invalid affine matrix")
    o isa AbstractVector && length(o)==2 && all(v->v isa Float64 && isfinite(v),o) || _cal_error("invalid affine offset")
    _cal_nonsingular([m[1][1] m[1][2];m[2][1] m[2][2]])
    t["applied_numeric_type"]=="Float64" && t["original_numeric_type"] in ("Float16","Float32","Float64","BigFloat") || _cal_error("unsupported coefficient precision")
    bits=t["original_precision_bits"]
    bits isa AbstractVector && length(bits)==6 && all(b->b isa Int && b>=2,bits) || _cal_error("invalid original coefficient precision")
    expected=get(Dict("Float16"=>11,"Float32"=>24,"Float64"=>53),t["original_numeric_type"],nothing)
    expected===nothing || all(==(expected),bits) || _cal_error("inconsistent original precision")
    _experiment_hash(data["geometry_sha256"]) && _cal_digest(g)==data["geometry_sha256"] || _cal_error("geometry digest mismatch")
    time=data["time"]
    _experiment_keys(time,["basis","quantity","unit","unit_provenance","interval"],"calibrated time policy")
    unit=_cal_optional_value(time["unit"],_cal_label)
    encoded=_cal_optional_value(time["interval"],v->(_timing_positive(v)===nothing && _cal_error("missing interval")))
    timing=_cal_optional_value(data["timing_snapshot"],v->_tracking_validate(_cal_unpack(v)))
    if kind=="timed_tracking"
        timing===nothing && _cal_error("timed export needs its original timing snapshot")
        snapshot=_cal_unpack(timing)
        snapshot["position_basis"]=="pixels" && snapshot["scale"]===nothing || _cal_error("timed source is not raw pixels")
        time["basis"]=="actual_sample_times" && time["quantity"]=="actual_time_secant" && encoded===nothing || _cal_error("invalid actual-time export policy")
        known=snapshot["effective_time_unit"]
        if known===nothing
            time["unit_provenance"]==(unit===nothing ? "unknown" : "explicit_export_time_unit_assumption") || _cal_error("unknown time unit must be labeled as an assumption")
        else
            unit==known && time["unit_provenance"]=="chosen_timeline" || _cal_error("export unit differs from known sample unit")
        end
    else
        timing===nothing || _cal_error("ordinal/PTV export cannot claim a timing snapshot")
        if encoded===nothing
            time["basis"]==(kind=="ptv" ? "pair_displacement" : "ordinal_frames") &&
                time["quantity"]==(kind=="ptv" ? "pair_displacement" : "ordinal_frame_secant") &&
                isequal(unit,kind=="ptv" ? nothing : "frame") &&
                time["unit_provenance"]==(kind=="ptv" ? "unknown" : "selected_frame_coordinate") || _cal_error("inconsistent displacement/frame policy")
        else
            unit!==nothing && time["basis"]==(kind=="ptv" ? "explicit_pair_delay" : "explicit_uniform_interval") &&
                time["quantity"]==(kind=="ptv" ? "pair_mean_velocity" : "uniform_interval_secant") &&
                time["unit_provenance"]=="explicit_interval_label" || _cal_error("inconsistent declared interval policy")
        end
    end
    d=data["diagnostics"]
    _experiment_keys(d,["outlier_basis","match_residual","raw_pixel_residual","uncertainty"],"calibrated diagnostics")
    d["outlier_basis"]==(kind=="ptv" ? "original_pixel_validation" : "not_recorded") &&
        d["match_residual"]==(kind=="ptv" ? "unavailable_direction_not_retained" : "not_recorded") &&
        d["raw_pixel_residual"] isa Bool && d["raw_pixel_residual"]===(kind=="ptv") &&
        d["uncertainty"]=="not_retained" || _cal_error("unsupported diagnostic availability claims")
    paths=data["protected_locators"]
    paths isa AbstractVector && all(p->p isa String && _artifact_absolute_locator(p),paths) && paths==sort(unique(paths)) || _cal_error("invalid protected locators")
    if timing!==nothing
        all(p->p in paths,_cal_unpack(timing)["protected_paths"]) || _cal_error("timed input locators were dropped")
    end
    nothing
end

# Streaming structural CSV parser. It retains only the bounded header plus
# counters/quote state; observation fields and arbitrary multiline labels are
# not buffered. The separate SHA pass likewise streams the file.
function _cal_csv_structure(path)
    expected=join(_CALIBRATED_TABLE_COLUMNS,',')*"\n"
    open(path,"r") do io
        header=read(io,ncodeunits(expected))
        header==collect(codeunits(expected)) || _cal_error("calibrated CSV header mismatch")
        state=:start;fields=1;rows=0;active=false
        while !eof(io)
            bytes=read(io,8192)
            for b in bytes
                active=true
                if state===:quoted
                    b==0x22 && (state=:closed)
                    continue
                elseif state===:closed
                    if b==0x22
                        state=:quoted;continue
                    end
                    b in (0x2c,0x0a) || _cal_error("invalid bytes after a quoted CSV field")
                elseif b==0x22
                    state===:start || _cal_error("quote inside an unquoted CSV field")
                    state=:quoted;continue
                end
                if b==0x2c
                    fields+=1;state=:start
                elseif b==0x0a
                    fields==length(_CALIBRATED_TABLE_COLUMNS) || _cal_error("CSV column count mismatch")
                    rows+=1;fields=1;state=:start;active=false
                else
                    b==0x0d && _cal_error("unquoted carriage return in calibrated CSV")
                    state=:unquoted
                end
            end
        end
        !active && state===:start || _cal_error("CSV has an incomplete trailing record")
        rows
    end
end

"""
    load_calibrated_table_metadata(metadata_path; csv_path=nothing, verify_csv=true)

Load a completed version-1 calibrated metadata artifact. Validate schema and
integrity; default verify_csv=true also streams the associated CSV to check its
SHA-256, exact header, quoting/column structure and row count. No row/result payload
is retained. Missing/mismatched pairs are refused. csv_path can relocate the CSV
explicitly; otherwise its recorded relative path is resolved beside metadata.
Foreign absolute source locators remain verbatim provenance and are never read.
A foreign backslash separator dialect requires explicit receiving-host csv_path
on POSIX; separators are not translated or guessed. Explicit path arguments
must refer to the receiving host.
verify_csv=false reads metadata only and does not certify the CSV, numerical
results, acquisition sources or coordinate-frame labels. File size/mtime checks
detect some changes; concurrent writers and atomic snapshots are unsupported.
"""
function load_calibrated_table_metadata(metadata_path::AbstractString;csv_path=nothing,verify_csv::Bool=true)
    full=_artifact_local_path(metadata_path);stamp=stat(full)
    wrapper=try TOML.parsefile(full) catch;_cal_error("cannot parse calibrated metadata");end
    _experiment_keys(wrapper,["metadata","metadata_sha256"],"calibrated metadata artifact")
    data=wrapper["metadata"]
    _cal_validate(data)
    digest=wrapper["metadata_sha256"]
    _experiment_hash(digest) && _cal_digest(data)==digest || _cal_error("calibrated metadata integrity mismatch")
    table=csv_path===nothing ? _artifact_resolve_relative(dirname(full),data["csv_relative_path"]) :
        csv_path isa AbstractString ? _artifact_local_path(csv_path) : _cal_error("csv_path must be a local path")
    _cal_alias(full,table) && _cal_error("CSV aliases its metadata")
    if verify_csv
        isfile(table) || _cal_error("calibrated CSV is missing; pair is incomplete")
        before=stat(table)
        _experiment_file_digest(table)==data["csv_sha256"] || _cal_error("calibrated CSV digest mismatch; pair is incomplete or changed")
        _cal_csv_structure(table)==data["row_count"] || _cal_error("calibrated CSV row count mismatch")
        after=stat(table)
        before.size==after.size && before.mtime==after.mtime || _cal_error("CSV changed during verification")
    end
    after=stat(full)
    after.size==stamp.size && after.mtime==stamp.mtime || _cal_error("metadata changed during verification")
    CalibratedTableMetadata(data,digest,full,table,verify_csv)
end

"""
    calibrated_table_data(report::CalibratedTableMetadata) -> Dict{String,Any}

Return copied validated settings, exact timing context, diagnostic availability
and CSV association. `verification` records checks made at load, not continuing
verification; numerical results/source bytes are never verified by this reader.
`protected_locators` retains recorded provenance verbatim. Computed
`local_protected_paths` includes locally interpretable recorded sources and the
actual local metadata/associated CSV paths after relocation; pass that list to
later writers. Foreign locators are not resolved and no input/result/CSV files
are reopened by this getter.
"""
function calibrated_table_data(report::CalibratedTableMetadata)
    _cal_digest(report._data)==report._sha256 || _cal_error("calibrated metadata object changed")
    _cal_validate(report._data)
    data=deepcopy(report._data)
    data["local_protected_paths"]=sort!(unique([_artifact_local_protected_paths(data["protected_locators"]);report._metadata_path;report._csv_path]))
    data["verification"]=Dict("metadata_schema_and_integrity"=>true,"csv_structure_and_hash_at_load"=>report._csv_verified,
        "numerical_result"=>false,"source_bytes"=>false)
    data
end

function Base.show(io::IO,report::CalibratedTableMetadata)
    data=calibrated_table_data(report)
    print(io,"CalibratedTableMetadata(",data["result_kind"],", ",data["row_count"]," rows, CSV ",
        report._csv_verified ? "verified at load" : "not checked",
        ", frame=",data["geometry"]["output_coordinate_frame"],")")
end

function export_calibrated_table(csv_path::AbstractString,index::ResultFile,i::Integer;protected_paths=String[],kwargs...)
    export_calibrated_table(csv_path,index[i];protected_paths=[_result_protected_paths(index);protected_paths],kwargs...)
end
