const MEASUREMENT_HISTORY_FORMAT_VERSION = 1
const _HISTORY_ORIGINS = ["unavailable", "primary", "alternative", "fill", "masked", "custom_unclassified"]
const _HISTORY_UQ_STATES = ["not_requested", "finite_nonnegative", "negative", "nonfinite", "masked"]
const _HISTORY_BINDING_FIELDS = ("x", "y", "u", "v", "peak_ratio", "correlation_moment",
    "uncertainty_u", "uncertainty_v", "outliers", "mask", "scale")

function _reject_measurement_history(kwargs,workflow)
    any(k->haskey(kwargs,k),(:on_measurement_history,:record_measurement_history,:_history_association)) &&
        throw(ArgumentError("measurement history currently supports planar PIV only; $workflow history is not implemented"))
    nothing
end

"""
    PIVMeasurementHistory

Detached companion observations for the final pass's final executed sweep of
planar PIV. Access copied, validated primitive data with
[`measurement_history_data`](@ref). This object owns mutable array storage;
it is not an immutable snapshot. Editing its private storage invalidates its
integrity digest. No result, image, correlation plane or earlier-sweep trace is
retained. Default PIV processing allocates no measurement-history observations.

Rejection reasons are the first observed transition to an outlier at an actual
validation stage, not all failed criteria. Alternative acceptance and median
assignment are branch events, not inferred from numerical differences.
Uncertainty numerical availability does not establish accuracy, coverage or
applicability to a selected/fill vector.
"""
struct PIVMeasurementHistory
    _data::Dict{String,Any}
    _sha256::String
end

# Bounded canonical hashing: never build an array-sized encoded byte string.
struct _HistoryDigestIO <: IO
    context::SHA.SHA2_256_CTX
    buffer::IOBuffer
end
function Base.flush(io::_HistoryDigestIO)
    position(io.buffer) == 0 || SHA.update!(io.context, take!(io.buffer))
    nothing
end
function Base.write(io::_HistoryDigestIO, b::UInt8)
    n = write(io.buffer, b)
    position(io.buffer) >= 8192 && flush(io)
    n
end
function Base.unsafe_write(io::_HistoryDigestIO, p::Ptr{UInt8}, n::UInt)
    written = Base.unsafe_write(io.buffer, p, n)
    position(io.buffer) >= 8192 && flush(io)
    written
end
function _history_digest(data)
    io = _HistoryDigestIO(SHA.SHA2_256_CTX(), IOBuffer(sizehint=8192))
    _experiment_canonical(io, data)
    flush(io)
    bytes2hex(SHA.digest!(io.context))
end
function _history_result_digest(r::PIVResult)
    data = Dict{String,Any}(k => getfield(r, Symbol(k)) for k in _HISTORY_BINDING_FIELDS if k != "scale")
    s = r.scale
    data["scale"] = s === nothing ? nothing : Dict{String,Any}(String(k)=>getfield(s,k) for k in fieldnames(PhysicalScale))
    _history_digest(data)
end

mutable struct _HistoryObservation{T}
    primary_u::Matrix{T}
    primary_v::Matrix{T}
    residual_u::Matrix{T}
    residual_v::Matrix{T}
    first_rejection::Matrix{Int}
    before_flags::BitMatrix
    pre_substitution::BitMatrix
    accepted_rank::Matrix{Int}
    fill_attempted::BitMatrix
    fill_assigned::BitMatrix
    primary_restored::BitMatrix
    stages::Vector{Dict{String,Any}}
    custom::Bool
    sweep::Int
end
function _history_stages(params)
    stages = [Dict{String,Any}("name"=>"nonfinite_primary", "kind"=>"built_in")]
    params.uod_enable && push!(stages, Dict{String,Any}("name"=>"implicit_uod", "kind"=>"built_in"))
    push!(stages, Dict{String,Any}("name"=>"implicit_peak_ratio", "kind"=>"built_in"))
    custom = false
    for (i,spec) in enumerate(params.validation)
        v = parse_validator(spec)
        builtin = v isa Union{PeakRatioValidator,CorrelationMomentValidator,VelocityMagnitudeValidator,UniversalOutlierValidator}
        custom |= !builtin
        push!(stages, Dict{String,Any}("name"=>"validation[$i]:$(nameof(typeof(v)))", "kind"=>builtin ? "built_in" : "custom"))
    end
    stages, custom
end
function _HistoryObservation(::Type{T}, shape, params) where T
    stages, custom = _history_stages(params)
    _HistoryObservation(zeros(T,shape),zeros(T,shape),zeros(T,shape),zeros(T,shape),zeros(Int,shape),
        falses(shape),falses(shape),zeros(Int,shape),falses(shape),falses(shape),falses(shape),stages,custom,0)
end
function _history_begin!(h, u, v, ru, rv, sweep)
    copyto!(h.primary_u,u); copyto!(h.primary_v,v)
    copyto!(h.residual_u,ru); copyto!(h.residual_v,rv)
    for array in (h.first_rejection,h.before_flags,h.pre_substitution,h.accepted_rank,h.fill_attempted,h.fill_assigned,h.primary_restored)
        fill!(array,zero(eltype(array)))
    end
    h.sweep = sweep
    nothing
end
function _history_before!(h, result)
    copyto!(h.before_flags,result.outliers)
    nothing
end
function _history_after!(h,result,stage)
    for i in eachindex(result.outliers)
        !result.mask[i] && result.outliers[i] && !h.before_flags[i] && h.first_rejection[i] == 0 && (h.first_rejection[i] = stage)
    end
    nothing
end
function _history_validator!(h,result,validator,stage)
    _history_before!(h,result)
    apply_validator!(result,validator)
    _history_after!(h,result,stage)
    nothing
end
function _history_uq_status(values, mask, requested)
    out = zeros(UInt8,size(values))
    for i in eachindex(values,mask)
        out[i] = mask[i] ? 4 : !requested ? 0 : !isfinite(values[i]) ? 3 : values[i] < 0 ? 2 : 1
    end
    out
end
function _history_finish(h,result,backend,processing_size,pass_index)
    # Masked residual cells were never correlated; do not expose their scratch
    # zero initialization as a measured displacement.
    h.residual_u[result.mask] .= eltype(h.residual_u)(NaN)
    h.residual_v[result.mask] .= eltype(h.residual_v)(NaN)
    origin = zeros(UInt8,size(result.u))
    for i in eachindex(origin)
        origin[i] = result.mask[i] ? 4 : h.primary_restored[i] ?
            (isfinite(h.primary_u[i]) && isfinite(h.primary_v[i]) ? 1 : 0) :
            h.fill_assigned[i] ? 3 : h.accepted_rank[i] > 0 ? 2 : h.custom ? 5 :
            isfinite(h.primary_u[i]) && isfinite(h.primary_v[i]) ? 1 : 0
    end
    data = Dict{String,Any}(
        "history_format_version"=>MEASUREMENT_HISTORY_FORMAT_VERSION,
        "history_id"=>string(UUIDs.uuid4()), "backend"=>String(backend),
        "image_type"=>string(eltype(result.u)), "processing_size"=>collect(processing_size),
        "core_source_sha256"=>_experiment_software()["core_source_sha256"],
        "pair_index"=>nothing,"association"=>nothing,"pass_index"=>pass_index,"sweep_index"=>h.sweep,
        "n_peaks"=>result.parameters.n_peaks,"coordinate_basis"=>"original_image_pixels","displacement_unit"=>"px",
        "x"=>copy(result.x),"y"=>copy(result.y),"mask"=>copy(result.mask),
        "primary_u"=>h.primary_u,"primary_v"=>h.primary_v,"primary_residual_u"=>h.residual_u,"primary_residual_v"=>h.residual_v,
        "first_rejection_stage"=>h.first_rejection,"rejection_stages"=>h.stages,
        "pre_substitution_outliers"=>h.pre_substitution,"accepted_peak_rank"=>h.accepted_rank,
        "fill_attempted"=>h.fill_attempted,"fill_assigned"=>h.fill_assigned,"primary_restored"=>h.primary_restored,
        "final_origin"=>origin,"origin_codes"=>copy(_HISTORY_ORIGINS),"final_outliers"=>copy(result.outliers),
        "custom_validators_present"=>h.custom,
        "uncertainty_u_status"=>_history_uq_status(result.uncertainty_u,result.mask,result.parameters.uncertainty),
        "uncertainty_v_status"=>_history_uq_status(result.uncertainty_v,result.mask,result.parameters.uncertainty),
        "uncertainty_status_codes"=>copy(_HISTORY_UQ_STATES),
        "uncertainty_basis"=>"final_deformed_windows_zero_shift_statistics; alternatives_not_reestimated; fills_not_propagated; applicability_and_coverage_not_established",
        "measurement_sha256"=>_history_result_digest(result))
    _history_validate(data)
    PIVMeasurementHistory(data,_history_digest(data))
end
function _history_context(h,i,association)
    data = copy(_history_checked_data(h))
    data["pair_index"] = i
    data["association"] = association === nothing ? nothing : Dict{String,Any}("recipe_id"=>association.recipe_id,"input_id"=>association.input_id)
    PIVMeasurementHistory(data,_history_digest(data))
end

const _HISTORY_FIELDS = ["history_format_version","history_id","backend","image_type","processing_size","core_source_sha256","pair_index","association",
    "pass_index","sweep_index","n_peaks","coordinate_basis","displacement_unit","x","y","mask","primary_u","primary_v","primary_residual_u","primary_residual_v",
    "first_rejection_stage","rejection_stages","pre_substitution_outliers","accepted_peak_rank","fill_attempted","fill_assigned","primary_restored","final_origin","origin_codes",
    "final_outliers","custom_validators_present","uncertainty_u_status","uncertainty_v_status","uncertainty_status_codes","uncertainty_basis","measurement_sha256"]
function _history_validate(data)
    _experiment_keys(data,_HISTORY_FIELDS,"measurement history")
    data["history_format_version"] === MEASUREMENT_HISTORY_FORMAT_VERSION || _experiment_error("unsupported measurement history version")
    data["history_id"] isa String && try UUIDs.UUID(data["history_id"]); true catch; false end || _experiment_error("invalid history ID")
    data["backend"] in ("cpu","ka","cuda","amdgpu") || _experiment_error("invalid history backend")
    data["image_type"] in ("Float32","Float64") || _experiment_error("invalid history precision")
    all(k->_experiment_hash(data[k]),("core_source_sha256","measurement_sha256")) || _experiment_error("invalid history identity")
    shape = data["processing_size"]
    shape isa Vector{Int} && length(shape)==2 && all(>(0),shape) || _experiment_error("invalid history processing size")
    pixel_count = try Base.Checked.checked_mul(shape...) catch; _experiment_error("history processing size overflows") end
    for k in ("pass_index","sweep_index","n_peaks")
        data[k] isa Int && data[k]>0 || _experiment_error("invalid history $k")
    end
    i=data["pair_index"]
    i===nothing || (i isa Int && i>0) || _experiment_error("invalid history pair index")
    a=data["association"]
    if a!==nothing
        _experiment_keys(a,["recipe_id","input_id"],"history association")
        all(_experiment_hash,values(a)) && i!==nothing || _experiment_error("invalid history association")
    end
    data["coordinate_basis"]=="original_image_pixels" && data["displacement_unit"]=="px" || _experiment_error("unsupported history coordinates")
    T=data["image_type"]=="Float32" ? Float32 : Float64
    for k in ("x","y")
        data[k] isa Vector{T} && !isempty(data[k]) && all(isfinite,data[k]) && issorted(data[k]) && allunique(data[k]) || _experiment_error("invalid history $k axis")
    end
    gridshape=(length(data["y"]),length(data["x"]))
    Base.Checked.checked_mul(gridshape...) <= pixel_count || _experiment_error("history grid exceeds processing size")
    for k in ("primary_u","primary_v","primary_residual_u","primary_residual_v")
        data[k] isa Matrix{T} && size(data[k])==gridshape || _experiment_error("invalid history $k array")
    end
    for k in ("mask","pre_substitution_outliers","fill_attempted","fill_assigned","primary_restored","final_outliers")
        data[k] isa AbstractMatrix{Bool} && size(data[k])==gridshape || _experiment_error("invalid history $k flags")
    end
    stages=data["rejection_stages"]
    stages isa AbstractVector && !isempty(stages) || _experiment_error("missing rejection stages")
    for stage in stages
        _experiment_keys(stage,["name","kind"],"rejection stage")
        stage["name"] isa String && !isempty(stage["name"]) && stage["kind"] in ("built_in","custom") || _experiment_error("invalid rejection stage")
    end
    stages[1]==Dict("name"=>"nonfinite_primary","kind"=>"built_in") || _experiment_error("missing initial nonfinite-primary stage")
    data["custom_validators_present"] isa Bool && data["custom_validators_present"]==any(s->s["kind"]=="custom",stages) || _experiment_error("invalid custom validator declaration")
    for (key,limit) in (("first_rejection_stage",length(stages)),("accepted_peak_rank",data["n_peaks"]))
        data[key] isa Matrix{Int} && size(data[key])==gridshape && all(v->0<=v<=limit,data[key]) || _experiment_error("invalid history $key")
    end
    any(==(1),data["accepted_peak_rank"]) && _experiment_error("accepted alternative rank must be at least two")
    data["origin_codes"]==_HISTORY_ORIGINS && data["uncertainty_status_codes"]==_HISTORY_UQ_STATES || _experiment_error("unknown history codes")
    for (key,limit) in (("final_origin",5),("uncertainty_u_status",4),("uncertainty_v_status",4))
        data[key] isa Matrix{UInt8} && size(data[key])==gridshape && all(v->v<=limit,data[key]) || _experiment_error("invalid history $key codes")
    end
    data["uncertainty_basis"]=="final_deformed_windows_zero_shift_statistics; alternatives_not_reestimated; fills_not_propagated; applicability_and_coverage_not_established" || _experiment_error("unknown uncertainty basis")
    for j in eachindex(data["mask"])
        masked=data["mask"][j]
        masked && (data["first_rejection_stage"][j]!=0 || data["accepted_peak_rank"][j]!=0 || data["pre_substitution_outliers"][j] || data["fill_attempted"][j] || data["fill_assigned"][j] || data["primary_restored"][j] || data["final_outliers"][j]) && _experiment_error("masked node has measurement events")
        (data["final_origin"][j]==4)==masked || _experiment_error("mask/origin mismatch")
        for key in ("uncertainty_u_status","uncertainty_v_status")
            (data[key][j]==4)==masked || _experiment_error("mask/uncertainty status mismatch")
        end
        data["fill_assigned"][j] && !data["fill_attempted"][j] && _experiment_error("fill assignment without attempt")
        data["accepted_peak_rank"][j]>0 && !data["pre_substitution_outliers"][j] && _experiment_error("alternative without rejection")
        data["primary_restored"][j] && !data["final_outliers"][j] && _experiment_error("primary restoration without final flag")
        accepted=data["accepted_peak_rank"][j]>0
        attempted=data["fill_attempted"][j]
        assigned=data["fill_assigned"][j]
        restored=data["primary_restored"][j]
        flagged=data["final_outliers"][j]
        accepted && (attempted || assigned || restored || flagged) && _experiment_error("accepted alternative has incompatible fill/flag events")
        attempted && (!data["pre_substitution_outliers"][j] || !flagged) && _experiment_error("fill attempt without rejected output")
        restored && !attempted && _experiment_error("primary restoration without internal fill attempt")
        !masked && flagged != (data["pre_substitution_outliers"][j] && !accepted) && _experiment_error("inconsistent final outlier state")
        !data["custom_validators_present"] && !masked && (data["first_rejection_stage"][j]>0)!=data["pre_substitution_outliers"][j] && _experiment_error("inconsistent first rejection stage")
        finite_primary=isfinite(data["primary_u"][j]) && isfinite(data["primary_v"][j])
        expected=masked ? 4 : restored ? (finite_primary ? 1 : 0) : assigned ? 3 : accepted ? 2 :
            data["custom_validators_present"] ? 5 : finite_primary ? 1 : 0
        data["final_origin"][j]==expected || _experiment_error("origin disagrees with measured events")
    end
    data
end
function _history_checked_data(h::PIVMeasurementHistory)
    _history_validate(h._data)
    _experiment_hash(h._sha256) && _history_digest(h._data)==h._sha256 || _experiment_error("measurement history was mutated after capture")
    h._data
end

"""
    measurement_history_data(history::PIVMeasurementHistory) -> Dict{String,Any}

Return detached, validated version-1 primitive data with column-major matrices
indexed `[row, column]` on the stored `y`/`x` axes. Origin and uncertainty status
codes are zero-based indices into their code tables. `accepted_peak_rank=0`
means no alternative was accepted; ranks >=2 name the accepted ordered peak.
`first_rejection_stage=0` means no observed rejection; other entries are
one-based indices into `rejection_stages`. A fill assignment may be nonfinite
or numerically unchanged, and may be undone by primary restoration.
Custom validators run once; their arbitrary field effects remain unclassified.
"""
measurement_history_data(h::PIVMeasurementHistory)=deepcopy(_history_checked_data(h))

function _history_check_result(h,result)
    data=_history_checked_data(h)
    result isa PIVResult || _experiment_error("measurement history requires a planar PIV result payload")
    _history_result_digest(result)==data["measurement_sha256"] || _experiment_error("result measurement fields changed after history capture")
    nothing
end

"""
    verify_measurement_history(history::PIVMeasurementHistory, result::PIVResult) -> true

Validate packet integrity and its numerical binding against a raw, pixel-native
result already loaded by the caller. Throws `ArgumentError` on mutation or
mismatch. This reads no file and retains no result. Verify before `physical`
conversion: converted coordinates/components do not have the original binding.
Verification does not authenticate the source or establish uncertainty validity.
"""
function verify_measurement_history(h::PIVMeasurementHistory,result)
    _history_check_result(h,result)
    true
end

"""
    measurement_history_at(history::PIVMeasurementHistory, index::CartesianIndex{2})

Return detached scalar observations for one `[row, column]` node: raw pixel
coordinates, primary/residual components, first rejection stage/name/kind,
accepted alternative rank, fill/restoration events, final origin/flag/mask and
per-component uncertainty numerical status. No full packet copy or file/result
read occurs. Integrity verification scans/hashes the packet on each call; this
is O(grid nodes), while the returned named tuple has constant size. Stage zero
returns `nothing` for its name/kind. Numerical availability and primary origin
do not establish uncertainty applicability, accuracy or coverage.
"""
function measurement_history_at(h::PIVMeasurementHistory,index::CartesianIndex{2})
    d=_history_checked_data(h)
    _history_node(d,index)
end
function _history_node(d,index)
    checkbounds(d["mask"],index)
    row,column=Tuple(index)
    stage=d["first_rejection_stage"][index]
    reason=stage==0 ? nothing : d["rejection_stages"][stage]
    (index=index,x=d["x"][column],y=d["y"][row],
     primary_u=d["primary_u"][index],primary_v=d["primary_v"][index],
     primary_residual_u=d["primary_residual_u"][index],primary_residual_v=d["primary_residual_v"][index],
     first_rejection_stage=stage,rejection_name=reason===nothing ? nothing : reason["name"],
     rejection_kind=reason===nothing ? nothing : reason["kind"],
     pre_substitution_outlier=d["pre_substitution_outliers"][index],accepted_peak_rank=d["accepted_peak_rank"][index],
     fill_attempted=d["fill_attempted"][index],fill_assigned=d["fill_assigned"][index],primary_restored=d["primary_restored"][index],
     final_origin=d["origin_codes"][Int(d["final_origin"][index])+1],final_outlier=d["final_outliers"][index],masked=d["mask"][index],
     uncertainty_u_status=d["uncertainty_status_codes"][Int(d["uncertainty_u_status"][index])+1],
     uncertainty_v_status=d["uncertainty_status_codes"][Int(d["uncertainty_v_status"][index])+1])
end
_history_key(key)="measurement_history/"*last(split(key,'/'))
function _check_measurement_history_format(file)
    if !haskey(file,"measurement_history_format_version")
        haskey(file,"measurement_history") && _experiment_error("measurement history lacks format version")
        return false
    end
    file["measurement_history_format_version"]===MEASUREMENT_HISTORY_FORMAT_VERSION || _experiment_error("unsupported measurement history version")
    true
end
function _write_measurement_history(file,key,h)
    data=_history_checked_data(h)
    haskey(file,"measurement_history_format_version") || (file["measurement_history_format_version"]=MEASUREMENT_HISTORY_FORMAT_VERSION)
    file[_history_key(key)]=Dict{String,Any}("result_key"=>key,"history"=>data,"history_sha256"=>h._sha256)
    nothing
end

"""
    load_measurement_history(index::ResultFile, i; verify_result=false)
    load_measurement_history(path::AbstractString, i=1; verify_result=false)

Load one optional planar measurement-history companion, opening/closing the
completed native file per access. Missing companions return `nothing`; unknown
versions, malformed data, wrong result keys or changed packet digests throw.
By default this does not deserialize any result. `verify_result=true` also
reads the selected result and checks its numerical measurement binding (axes,
u/v, peak metrics, stored UQ, flags, mask and scale; not correlation planes or
parameter objects). No concurrent-writer guarantee is made. Pair indices are
absolute within the supplied sequence, including one-result-per-file output.
Replay association identifies verified recipe/input snapshots; the whole-file
run hash additionally covers the companion. Saving results alone does not copy
history. Current on-disk source identity requires an unchanged checkout/fresh
process to describe the loaded implementation; it is not a module attestation.
"""
function load_measurement_history(index::ResultFile,i::Integer;verify_result::Bool=false)
    checkbounds(index,i); _check_result_file(index)
    key="results/"*index.entry_keys[i]
    h=jldopen(index.path,"r") do file
        _check_results_format(file,index.path)
        _check_measurement_history_format(file) || return nothing
        haskey(file,_history_key(key)) || return nothing
        entry=file[_history_key(key)]
        _experiment_keys(entry,["result_key","history","history_sha256"],"measurement history entry")
        entry["result_key"]==key || _experiment_error("measurement history/result key mismatch")
        history=PIVMeasurementHistory(entry["history"],entry["history_sha256"])
        _history_checked_data(history)
        verify_result && _history_check_result(history,file[key])
        history
    end
    _check_result_file(index)
    h
end
load_measurement_history(path::AbstractString,i::Integer=1;kwargs...)=load_measurement_history(ResultFile(path),i;kwargs...)

Base.show(io::IO,h::PIVMeasurementHistory)=print(io,"PIVMeasurementHistory(final sweep)")
function Base.show(io::IO,::MIME"text/plain",h::PIVMeasurementHistory)
    d=_history_checked_data(h)
    print(io,"Planar measurement history: pass ",d["pass_index"],", sweep ",d["sweep_index"])
    d["pair_index"]===nothing || print(io,", pair ",d["pair_index"])
    print(io,"\n",count(!,d["mask"])," unmasked nodes; ",count(d["pre_substitution_outliers"])," flagged before alternatives")
    print(io,"\nAccepted alternatives: ",count(>(0),d["accepted_peak_rank"]),"; median assignments: ",count(d["fill_assigned"]),"; primary restorations: ",count(d["primary_restored"]))
    print(io,"\nFirst observed rejection stages only. Earlier sweeps/passes are not recorded.")
    print(io,"\nStored uncertainty uses the final deformed windows; alternatives are not reestimated and fills are not propagated. Applicability, accuracy and coverage are not established.")
end
