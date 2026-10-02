# A deliberately bounded first experiment format: file-based planar PIV.
# Native result files retain their independent format and existing APIs.
const EXPERIMENT_FORMAT_VERSION = 1

_experiment_error(message) = throw(ArgumentError(message))

function _experiment_keys(data, expected, context)
    data isa AbstractDict && Set(keys(data)) == Set(expected) ||
        _experiment_error("malformed $context: expected fields $(join(expected, ", "))")
    data
end

function _experiment_canonical(io, value)
    if value === nothing
        print(io, "nothing;")
    elseif value isa AbstractString
        print(io, "string:", ncodeunits(value), ':', value, ';')
    elseif value isa Union{Bool,Integer,AbstractFloat}
        print(io, string(typeof(value)), ':', repr(value), ';')
    elseif value isa AbstractDict
        all(k -> k isa String, keys(value)) || _experiment_error("record keys must be strings")
        print(io, "dict:", length(value), '{')
        for key in sort!(collect(keys(value)))
            _experiment_canonical(io, key)
            _experiment_canonical(io, value[key])
        end
        print(io, '}')
    elseif value isa AbstractArray
        print(io, "array:", string(eltype(value)), ':', join(size(value), ','), '[')
        for item in value  # Explicit column-major order, including BitMatrix.
            _experiment_canonical(io, item)
        end
        print(io, ']')
    else
        _experiment_error("unsupported experiment value $(typeof(value)); functions are never serialized")
    end
end

function _experiment_digest(data)
    io = IOBuffer()
    _experiment_canonical(io, data)
    bytes2hex(SHA.sha256(take!(io)))
end

_experiment_file_digest(path) = open(io -> bytes2hex(SHA.sha256(io)), path, "r")
_experiment_hash(value) = value isa String && occursin(r"^[0-9a-f]{64}$", value)

"""
    PreprocessStep(operation; kwargs...)

A serializable built-in image-conditioning operation for [`PIVRecipe`](@ref).
Supported operations are `:subtract_background` (`background` matrix),
`:intensity_cap` (`n_sigma=2`), `:highpass_filter` (`sigma=3`), `:clahe`
(`tiles=(8,8)`, `clip_limit=2`, `nbins=256`), `:percentile_stretch`
(`low=1`, `high=99`), `:invert_image`, and `:local_variance_normalize`
(`sigma=3`, `epsilon=1e-3`). Options are validated and defaults made explicit.
Background values are copied and embedded, not recomputed on replay.
"""
struct PreprocessStep
    operation::Symbol
    options::Dict{String,Any}
end

function PreprocessStep(operation::Symbol; kwargs...)
    defaults = operation === :subtract_background ? Dict{String,Any}() :
        operation === :intensity_cap ? Dict{String,Any}("n_sigma"=>2.0) :
        operation === :highpass_filter ? Dict{String,Any}("sigma"=>3.0) :
        operation === :clahe ? Dict{String,Any}("tiles"=>[8,8], "clip_limit"=>2.0, "nbins"=>256) :
        operation === :percentile_stretch ? Dict{String,Any}("low"=>1.0,"high"=>99.0) :
        operation === :invert_image ? Dict{String,Any}() :
        operation === :local_variance_normalize ? Dict{String,Any}("sigma"=>3.0,"epsilon"=>1e-3) :
        _experiment_error("unsupported built-in preprocessing operation $operation")
    allowed = operation === :subtract_background ? Set(["background"]) : Set(keys(defaults))
    for (key, value) in kwargs
        String(key) in allowed || _experiment_error("unsupported $operation option $key")
        defaults[String(key)] = deepcopy(value)
    end
    if operation === :subtract_background
        haskey(defaults, "background") || _experiment_error("subtract_background requires background")
        bg = defaults["background"]
        bg isa AbstractMatrix{<:Real} && all(isfinite, bg) ||
            _experiment_error("background must be a finite real matrix")
        defaults["background"] = Matrix(float.(bg))
    elseif operation === :clahe
        tiles, bins, limit = defaults["tiles"], defaults["nbins"], defaults["clip_limit"]
        (tiles isa Tuple || tiles isa AbstractVector) && length(tiles)==2 &&
            all(v -> v isa Integer && v >= 1, tiles) || _experiment_error("tiles must be two positive integers")
        bins isa Integer && bins >= 2 || _experiment_error("nbins must be at least 2")
        limit isa Real && isfinite(limit) && limit >= 1 || _experiment_error("clip_limit must be finite and at least 1")
        defaults["tiles"], defaults["nbins"], defaults["clip_limit"] = Int.(collect(tiles)), Int(bins), Float64(limit)
    elseif operation === :percentile_stretch
        low, high = defaults["low"], defaults["high"]
        low isa Real && high isa Real && 0 <= low < high <= 100 ||
            _experiment_error("percentiles must satisfy 0 <= low < high <= 100")
        defaults["low"], defaults["high"] = Float64(low), Float64(high)
    else
        for key in keys(defaults)
            v = defaults[key]
            v isa Real && isfinite(v) && v > 0 || _experiment_error("$key must be positive and finite")
            defaults[key] = Float64(v)
        end
    end
    PreprocessStep(operation, defaults)
end

"""
    ScriptReference(path; entrypoint)

Reference a custom preprocessing script by absolute locator, SHA-256 content,
and a nonempty entrypoint description. No script is included, evaluated, or
executed. [`replay_experiment`](@ref) requires a caller-provided function for
this reference, after verifying that the referenced bytes are unchanged.
The entrypoint documents the caller's intended function; it is not resolved
automatically and cannot prove that a supplied function implements the script.
"""
struct ScriptReference
    path::String
    sha256::String
    entrypoint::String
end

function ScriptReference(path::AbstractString; entrypoint::AbstractString)
    isempty(entrypoint) && _experiment_error("script entrypoint must not be empty")
    ScriptReference(realpath(path), _experiment_file_digest(path), String(entrypoint))
end

function _experiment_validator_data(spec)
    v = parse_validator(spec)
    name = v isa PeakRatioValidator ? "peak_ratio" :
        v isa CorrelationMomentValidator ? "correlation_moment" :
        v isa VelocityMagnitudeValidator ? "velocity_magnitude" :
        v isa UniversalOutlierValidator ? "uod" :
        _experiment_error("custom validators are not supported by experiment version 1")
    options = Dict{String,Any}(String(k)=>getfield(v,k) for k in fieldnames(typeof(v)))
    all(k -> options[k] isa Real && (isfinite(options[k]) ||
            (name=="velocity_magnitude" && k=="max" && options[k]==Inf)), keys(options)) ||
        _experiment_error("validator options must be finite")
    Dict{String,Any}("operation"=>name,"options"=>options)
end

function _experiment_validator(data)
    _experiment_keys(data, ["operation","options"], "validator")
    name, options = data["operation"], data["options"]
    if name in ("peak_ratio", "correlation_moment")
        _experiment_keys(options, ["threshold"], "validator options")
        v = name == "peak_ratio" ? PeakRatioValidator(options["threshold"]) : CorrelationMomentValidator(options["threshold"])
    elseif name == "velocity_magnitude"
        _experiment_keys(options, ["min","max"], "validator options")
        v = VelocityMagnitudeValidator(options["min"],options["max"])
    elseif name == "uod"
        _experiment_keys(options, ["threshold","neighborhood_size","epsilon"], "validator options")
        v = UniversalOutlierValidator(options["threshold"]; neighborhood_size=options["neighborhood_size"], epsilon=options["epsilon"])
    else
        _experiment_error("unknown experiment validator $name")
    end
    _experiment_validator_data(v)
    v
end

function _experiment_pass_data(p::PIVParameters)
    Dict{String,Any}(String(k) => k === :validation ? [_experiment_validator_data(v) for v in p.validation] :
        getfield(p,k) isa Symbol ? String(getfield(p,k)) :
        getfield(p,k) isa Tuple ? collect(getfield(p,k)) : getfield(p,k) for k in fieldnames(PIVParameters))
end

function _experiment_pass(data)
    _experiment_keys(data, String.(fieldnames(PIVParameters)), "PIV pass")
    args = Pair{Symbol,Any}[]
    for key in fieldnames(PIVParameters)
        value = data[String(key)]
        if key === :validation
            value isa AbstractVector || _experiment_error("validation must be a vector")
            value = Tuple(_experiment_validator(v) for v in value)
        elseif fieldtype(PIVParameters,key) === Symbol
            value isa String || _experiment_error("$key must be a string")
            value = Symbol(value)
        elseif fieldtype(PIVParameters,key) <: Tuple
            value isa AbstractVector && length(value)==2 && all(v -> v isa Int, value) ||
                _experiment_error("$key must contain two integers")
            value = Tuple(value)
        elseif fieldtype(PIVParameters,key) === Bool
            value isa Bool || _experiment_error("$key must be Bool")
        elseif fieldtype(PIVParameters,key) === Int
            value isa Int || _experiment_error("$key must be Int")
        else
            value isa Real && isfinite(value) || _experiment_error("$key must be finite")
        end
        push!(args, key=>value)
    end
    PIVParameters(; args...)
end

"""
    PIVRecipe(passes; preprocessing=PreprocessStep[], external_preprocess=nothing,
              mask=nothing, roi=nothing, scale=nothing, backend=:cpu,
              image_type=Float64, threaded=false, predictor_smoothing=true,
              mask_threshold=0.5, uncertainty_backend=:same)

Snapshot a complete explicit planar PIV pass schedule and replay settings.
Public fields expose the schedule, ordered built-in preprocessing, optional
script reference, original full-image static mask, ROI, scale, backend and
precision, plus deformation/mask options. Inputs are copied. Version 1 supports
CPU/KA and Float32/Float64, not GPU devices, dynamic masks, custom validators,
stereo/dewarping/self-calibration, PTV, or tracking. Resolve effort presets into
explicit [`multipass_parameters`](@ref) before constructing a recipe.
`recipe_id` is a location-independent SHA-256 of the settings; mutation after
construction is detected before saving or replaying.
"""
struct PIVRecipe
    passes::Vector{PIVParameters}
    preprocessing::Vector{PreprocessStep}
    external_preprocess::Union{Nothing,ScriptReference}
    mask::Union{Nothing,BitMatrix}
    roi::Union{Nothing,ROI}
    scale::Union{Nothing,PhysicalScale}
    backend::Symbol
    image_type::DataType
    threaded::Bool
    predictor_smoothing::Bool
    mask_threshold::Float64
    uncertainty_backend::Symbol
    recipe_id::String
end

function _experiment_recipe_data(r::PIVRecipe)
    external = r.external_preprocess
    Dict{String,Any}("kind"=>"planar_piv", "passes"=>[_experiment_pass_data(p) for p in r.passes],
        "preprocessing"=>[Dict{String,Any}("operation"=>String(s.operation),"options"=>deepcopy(s.options)) for s in r.preprocessing],
        "external_preprocess"=>external === nothing ? nothing : Dict{String,Any}("sha256"=>external.sha256,"entrypoint"=>external.entrypoint),
        "mask"=>r.mask, "roi"=>r.roi===nothing ? nothing : [first(r.roi.rows),last(r.roi.rows),first(r.roi.cols),last(r.roi.cols)],
        "scale"=>r.scale===nothing ? nothing : Dict{String,Any}(String(k)=>getfield(r.scale,k) for k in fieldnames(PhysicalScale)),
        "backend"=>String(r.backend), "image_type"=>string(r.image_type), "threaded"=>r.threaded,
        "predictor_smoothing"=>r.predictor_smoothing,"mask_threshold"=>r.mask_threshold,
        "uncertainty_backend"=>String(r.uncertainty_backend))
end

function PIVRecipe(passes::Union{PIVParameters,AbstractVector{PIVParameters}};
                   preprocessing::AbstractVector{PreprocessStep}=PreprocessStep[],
                   external_preprocess::Union{Nothing,ScriptReference}=nothing,
                   mask=nothing, roi=nothing, scale::Union{Nothing,PhysicalScale}=nothing,
                   backend::Symbol=:cpu, image_type::DataType=Float64,
                   threaded::Bool=false, predictor_smoothing::Bool=true,
                   mask_threshold::Real=0.5, uncertainty_backend::Symbol=:same)
    schedule = passes isa PIVParameters ? [passes] : collect(passes)
    isempty(schedule) && _experiment_error("recipe requires at least one explicit pass")
    schedule = [_experiment_pass(_experiment_pass_data(p)) for p in schedule]
    mask === nothing || mask isa AbstractMatrix{Bool} || _experiment_error("recipe mask must be a static full-image Bool matrix")
    backend in (:cpu,:ka) || _experiment_error("experiment version 1 supports :cpu and :ka only")
    image_type in (Float32,Float64) || _experiment_error("image_type must be Float32 or Float64")
    isfinite(mask_threshold) && 0 < mask_threshold <= 1 || _experiment_error("mask_threshold must be in (0,1]")
    uncertainty_backend in (:same,:cpu) || _experiment_error("uncertainty_backend must be :same or :cpu")
    steps = [PreprocessStep(s.operation; (Symbol(k)=>v for (k,v) in s.options)...) for s in preprocessing]
    external_preprocess === nothing || (_experiment_hash(external_preprocess.sha256) && !isempty(external_preprocess.entrypoint)) ||
        _experiment_error("invalid external script reference")
    rr = roi === nothing || roi isa ROI ? roi : ROI(roi)
    args = (schedule,steps,external_preprocess,mask===nothing ? nothing : BitMatrix(mask),rr,
            scale,backend,image_type,threaded,predictor_smoothing,Float64(mask_threshold),uncertainty_backend)
    provisional = PIVRecipe(args..., "")
    PIVRecipe(args..., _experiment_digest(_experiment_recipe_data(provisional)))
end

"""
    recipe_identity(recipe::PIVRecipe) -> String

Return the recipe's SHA-256 identity after checking that its copied settings
have not been changed. Paths are locators, not identity: a script's bytes and
entrypoint are included but its location is not.
"""
function recipe_identity(recipe::PIVRecipe)
    _experiment_digest(_experiment_recipe_data(recipe)) == recipe.recipe_id ||
        _experiment_error("recipe settings changed after snapshot; construct a new PIVRecipe")
    recipe.recipe_id
end

function _experiment_software()
    packages = Dict{String,Any}[]
    for (uuid, info) in sort!(collect(Pkg.dependencies()); by=p -> string(first(p)))
        push!(packages, Dict{String,Any}("uuid"=>string(uuid),"name"=>info.name,
            "version"=>info.version===nothing ? nothing : string(info.version),
            "tree_hash"=>info.tree_hash===nothing ? nothing : string(info.tree_hash)))
    end
    root = pkgdir(Hammerhead)
    source = Dict{String,Any}()
    for dir in ("src","ext"), (directory, _, files) in walkdir(joinpath(root,dir))
        for file in files
            endswith(file,".jl") || continue
            path = joinpath(directory,file)
            source[replace(relpath(path,root),'\\'=>'/')] = _experiment_file_digest(path)
        end
    end
    project = Base.active_project()
    project_text = project===nothing ? nothing : read(project,String)
    manifest = project===nothing ? nothing : joinpath(dirname(project),"Manifest.toml")
    versioned = project===nothing ? nothing : joinpath(dirname(project),"Manifest-v$(VERSION.major).$(VERSION.minor).toml")
    versioned!==nothing && isfile(versioned) && (manifest=versioned)
    manifest_text = manifest===nothing || !isfile(manifest) ? nothing : read(manifest,String)
    Dict{String,Any}("julia_version"=>string(VERSION),"hammerhead_version"=>string(Base.pkgversion(Hammerhead)),
        "core_source_sha256"=>_experiment_digest(source),"packages"=>packages,
        "kernel"=>string(Sys.KERNEL),"architecture"=>string(Sys.ARCH),
        "julia_threads"=>Threads.nthreads(),"fftw_threads"=>Int(FFTW.get_num_threads()),
        "project_path"=>project,"project_text"=>project_text,"manifest_text"=>manifest_text)
end

_experiment_environment_signature(env) = Dict{String,Any}(k=>v for (k,v) in env if !(k in ("project_path","project_text","manifest_text")))

"""
    ExperimentRun

Metadata for one execution: UUID `run_id`, recipe/input identities, UTC Unix
`started_at`/`finished_at`, `status` (`:completed` or `:failed`),
`completed_pairs`, output locator/digest, actual software `environment`, and
`error` summary. `completed_pairs` counts pairs whose result write completed,
not a durability or resumability guarantee. Run metadata contains no result
arrays. Creation and execution environments are recorded separately.
"""
struct ExperimentRun
    run_id::String
    recipe_id::String
    input_id::String
    started_at::Float64
    finished_at::Float64
    status::Symbol
    completed_pairs::Int
    output::String
    output_sha256::Union{Nothing,String}
    environment::Dict{String,Any}
    error::Union{Nothing,String}
end

"""
    ExperimentRecord(file_pairs, recipe::PIVRecipe)

Capture explicit ordered image-file pairs, their SHA-256 byte identities and
decoded sizes, the recipe, and creation software environment. Image data are
not embedded. `input_id` includes content, dimensions, and pairing order but
excludes path locations. `runs` holds optional execution metadata. Version 1
supports file-based planar PIV only, a full-image static mask, and embedded
built-in background images. This is not a checkpoint/resume format.
"""
struct ExperimentRecord
    recipe::PIVRecipe
    input_files::Vector{Dict{String,Any}}
    pairs::Vector{Vector{Int}}
    input_id::String
    creation_environment::Dict{String,Any}
    runs::Vector{ExperimentRun}
    record_paths::Vector{String}
end

function _experiment_input_data(files,pairs)
    descriptor(f)=Dict{String,Any}(k=>f[k] for k in ("sha256","size_bytes","image_size"))
    # Canonicalize observations rather than locator dedup: repeated use of one
    # file and separate files with identical bytes must have the same identity.
    Dict{String,Any}("pairs"=>[[descriptor(files[i]) for i in pair] for pair in pairs])
end

function _experiment_load_image(file, T)
    path = file["path"]
    isfile(path) && filesize(path)==file["size_bytes"] && _experiment_file_digest(path)==file["sha256"] ||
        _experiment_error("experiment input changed or is missing: $path")
    image = load_image(T,path)
    collect(size(image)) == file["image_size"] && _experiment_file_digest(path)==file["sha256"] ||
        _experiment_error("experiment input changed during loading: $path")
    image
end

function ExperimentRecord(file_pairs::AbstractVector, recipe::PIVRecipe)
    recipe_identity(recipe)
    isempty(file_pairs) && _experiment_error("experiment requires at least one file pair")
    files = Dict{String,Any}[]
    pairs = Vector{Int}[]
    for pair in file_pairs
        pair isa Tuple && length(pair)==2 && all(p -> p isa AbstractString, pair) ||
            _experiment_error("experiment inputs must be tuples of two image-file paths")
        ids = Int[]
        for name in pair
            path = realpath(name)
            index = findfirst(f -> f["path"]==path, files)
            if index===nothing
                file = Dict{String,Any}("path"=>path,"sha256"=>_experiment_file_digest(path),
                    "size_bytes"=>filesize(path),"image_size"=>collect(size(load_image(recipe.image_type,path))))
                _experiment_load_image(file,recipe.image_type)
                push!(files,file); index=length(files)
            end
            push!(ids,index)
        end
        push!(pairs,ids)
    end
    record = ExperimentRecord(deepcopy(recipe),files,pairs,
        _experiment_digest(_experiment_input_data(files,pairs)),_experiment_software(),ExperimentRun[],String[])
    _experiment_preflight(record; verify_files=true)
    record
end

function _experiment_preflight(record; verify_files=false)
    recipe_identity(record.recipe)
    recipe = record.recipe
    # Revalidate even manually constructed records, including mutable settings.
    _experiment_recipe_from_data(_experiment_recipe_data(recipe), recipe.external_preprocess===nothing ? nothing : recipe.external_preprocess.path)
    isempty(record.input_files) && _experiment_error("experiment has no input files")
    isempty(record.pairs) && _experiment_error("experiment has no pairs")
    for file in record.input_files
        _experiment_keys(file,["path","sha256","size_bytes","image_size"],"input file")
        file["path"] isa String && isabspath(file["path"]) && _experiment_hash(file["sha256"]) &&
            file["size_bytes"] isa Integer && file["size_bytes"]>=0 || _experiment_error("invalid input identity")
        dims=file["image_size"]
        dims isa Vector{Int} && length(dims)==2 && all(>(0),dims) || _experiment_error("invalid image dimensions")
        recipe.mask===nothing || size(recipe.mask)==Tuple(dims) || _experiment_error("recipe mask must match the original full image")
        selected = recipe.roi===nothing ? dims : [length(recipe.roi.rows),length(recipe.roi.cols)]
        recipe.roi===nothing || (last(recipe.roi.rows)<=dims[1] && last(recipe.roi.cols)<=dims[2]) ||
            _experiment_error("recipe ROI exceeds an input image")
        all(p -> all(collect(p.search_area_size).<=selected),recipe.passes) || _experiment_error("pass search area exceeds selected image/ROI")
        for step in recipe.preprocessing
            step.operation===:subtract_background && size(step.options["background"])!=Tuple(dims) &&
                _experiment_error("background must match the original full image")
        end
        verify_files && _experiment_load_image(file,recipe.image_type)
    end
    for pair in record.pairs
        length(pair)==2 && all(i -> 1<=i<=length(record.input_files),pair) || _experiment_error("invalid input pair indices")
        record.input_files[pair[1]]["image_size"]==record.input_files[pair[2]]["image_size"] || _experiment_error("paired image dimensions differ")
    end
    sort!(unique!(vcat(record.pairs...)))==collect(1:length(record.input_files)) ||
        _experiment_error("every input file entry must be referenced by a pair")
    _experiment_digest(_experiment_input_data(record.input_files,record.pairs))==record.input_id ||
        _experiment_error("input identities or pairing changed after snapshot")
    _check_backend_params(_resolve_backend(recipe.backend),recipe.passes)
    script=recipe.external_preprocess
    if verify_files && script!==nothing
        isfile(script.path) && _experiment_file_digest(script.path)==script.sha256 || _experiment_error("referenced preprocessing script changed or is missing")
    end
    record
end

function _experiment_recipe_from_data(data, script_path)
    expected_keys=["kind","passes","preprocessing","external_preprocess","mask","roi","scale","backend","image_type","threaded","predictor_smoothing","mask_threshold","uncertainty_backend"]
    _experiment_keys(data,expected_keys,"recipe")
    data["kind"]=="planar_piv" || _experiment_error("unsupported experiment kind")
    data["passes"] isa AbstractVector && data["preprocessing"] isa AbstractVector || _experiment_error("invalid recipe steps")
    steps = PreprocessStep[]
    for s in data["preprocessing"]
        _experiment_keys(s,["operation","options"],"preprocessing step")
        s["operation"] isa String && s["options"] isa AbstractDict || _experiment_error("invalid preprocessing step")
        step=PreprocessStep(Symbol(s["operation"]); (Symbol(k)=>v for (k,v) in s["options"])...)
        Set(keys(step.options))==Set(keys(s["options"])) || _experiment_error("preprocessing defaults must be explicit")
        push!(steps,step)
    end
    external=data["external_preprocess"]
    if external!==nothing
        _experiment_keys(external,["sha256","entrypoint"],"script reference")
        script_path isa String && isabspath(script_path) && _experiment_hash(external["sha256"]) && external["entrypoint"] isa String && !isempty(external["entrypoint"]) ||
            _experiment_error("invalid script reference")
        external=ScriptReference(script_path,external["sha256"],external["entrypoint"])
    else
        script_path===nothing || _experiment_error("unexpected script locator")
    end
    scale=data["scale"]
    if scale!==nothing
        _experiment_keys(scale,String.(fieldnames(PhysicalScale)),"scale")
        scale=PhysicalScale(scale["pixel_size"],scale["dt"],scale["length_unit"],scale["time_unit"])
    end
    roi=data["roi"]
    if roi!==nothing
        roi isa Vector{Int} && length(roi)==4 || _experiment_error("invalid ROI")
        roi=ROI(roi[1]:roi[2],roi[3]:roi[4])
    end
    for k in ("threaded","predictor_smoothing")
        data[k] isa Bool || _experiment_error("$k must be Bool")
    end
    data["image_type"] in ("Float32","Float64") || _experiment_error("unsupported image precision")
    data["backend"] in ("cpu","ka") || _experiment_error("unsupported experiment backend")
    data["uncertainty_backend"] in ("same","cpu") || _experiment_error("unsupported uncertainty backend")
    PIVRecipe([_experiment_pass(p) for p in data["passes"]]; preprocessing=steps,
        external_preprocess=external,mask=data["mask"],roi,scale,backend=Symbol(data["backend"]),
        image_type=data["image_type"]=="Float32" ? Float32 : Float64,
        threaded=data["threaded"],predictor_smoothing=data["predictor_smoothing"],
        mask_threshold=data["mask_threshold"],uncertainty_backend=Symbol(data["uncertainty_backend"]))
end

function _experiment_run_data(run::ExperimentRun)
    Dict{String,Any}(String(k)=>getfield(run,k) isa Symbol ? String(getfield(run,k)) : getfield(run,k) for k in fieldnames(ExperimentRun))
end

function _experiment_run(data, record)
    _experiment_keys(data,String.(fieldnames(ExperimentRun)),"run")
    data["run_id"] isa String || _experiment_error("invalid run ID")
    try UUIDs.UUID(data["run_id"]) catch; _experiment_error("invalid run UUID") end
    data["recipe_id"]==record.recipe.recipe_id && data["input_id"]==record.input_id || _experiment_error("run identities do not match its experiment")
    data["status"] in ("completed","failed") || _experiment_error("unsupported run status")
    n=data["completed_pairs"]
    n isa Int && 0<=n<=length(record.pairs) || _experiment_error("invalid completed pair count")
    data["status"]=="completed" && n!=length(record.pairs) && _experiment_error("completed run has an incomplete pair count")
    all(k -> data[k] isa Real && isfinite(data[k]),("started_at","finished_at")) && data["finished_at"]>=data["started_at"] || _experiment_error("invalid run times")
    data["output"] isa String && isabspath(data["output"]) || _experiment_error("invalid run output locator")
    data["output_sha256"]===nothing || _experiment_hash(data["output_sha256"]) || _experiment_error("invalid run output digest")
    data["error"]===nothing || data["error"] isa String || _experiment_error("invalid run error summary")
    (data["status"]=="completed" ? data["error"]===nothing : data["error"] isa String) || _experiment_error("run status and error disagree")
    _experiment_validate_environment(data["environment"])
    ExperimentRun(data["run_id"],data["recipe_id"],data["input_id"],Float64(data["started_at"]),Float64(data["finished_at"]),
        Symbol(data["status"]),n,data["output"],data["output_sha256"],deepcopy(data["environment"]),data["error"])
end

function _experiment_validate_environment(env)
    _experiment_keys(env,["julia_version","hammerhead_version","core_source_sha256","packages","kernel","architecture","julia_threads","fftw_threads","project_path","project_text","manifest_text"],"software environment")
    all(k -> env[k] isa String && !isempty(env[k]),("julia_version","hammerhead_version","kernel","architecture")) &&
        _experiment_hash(env["core_source_sha256"]) || _experiment_error("invalid software identity")
    all(k -> env[k] isa Int && env[k]>0,("julia_threads","fftw_threads")) || _experiment_error("invalid thread metadata")
    all(k -> env[k]===nothing || env[k] isa String,("project_path","project_text","manifest_text")) || _experiment_error("invalid project metadata")
    env["packages"] isa AbstractVector || _experiment_error("invalid package environment")
    for package in env["packages"]
        _experiment_keys(package,["uuid","name","version","tree_hash"],"package metadata")
        package["uuid"] isa String && package["name"] isa String &&
            all(k -> package[k]===nothing || package[k] isa String,("version","tree_hash")) || _experiment_error("invalid package metadata")
    end
    env
end

function _experiment_alias(path, other)
    abspath(path)==abspath(other) || (ispath(path) && ispath(other) && Base.samefile(path,other))
end

function _experiment_protect(path,record; records=false, results=false)
    protected=[f["path"] for f in record.input_files]
    record.recipe.external_preprocess===nothing || push!(protected,record.recipe.external_preprocess.path)
    records && append!(protected,record.record_paths)
    results && append!(protected,[r.output for r in record.runs])
    any(p -> _experiment_alias(path,p),protected) && _experiment_error("destination aliases a protected experiment input, script, record, or result file")
    nothing
end

"""
    save_experiment(path, record; runs=record.runs) -> path

Save a version-1 experiment as explicit primitive JLD2 mappings. Unknown/custom
objects and changed recipe/input identities are rejected before replacement.
Input images, referenced scripts, and recorded result outputs are protected
against path/same-file aliases. Settings/backgrounds/masks are embedded; input
images and custom functions are not. Known save locations are remembered on
the in-memory record to protect them from subsequent result output.
"""
function save_experiment(path::AbstractString,record::ExperimentRecord; runs=record.runs)
    _experiment_preflight(record)
    _experiment_validate_environment(record.creation_environment)
    validated=[_experiment_run(_experiment_run_data(r),record) for r in runs]
    protection=ExperimentRecord(record.recipe,record.input_files,record.pairs,record.input_id,record.creation_environment,validated,record.record_paths)
    _experiment_protect(path,protection; results=true)
    payload=Dict{String,Any}("recipe"=>_experiment_recipe_data(record.recipe),"recipe_id"=>record.recipe.recipe_id,
        "script_path"=>record.recipe.external_preprocess===nothing ? nothing : record.recipe.external_preprocess.path,
        "input_files"=>deepcopy(record.input_files),"pairs"=>deepcopy(record.pairs),"input_id"=>record.input_id,
        "creation_environment"=>deepcopy(record.creation_environment),"runs"=>[_experiment_run_data(r) for r in validated])
    _experiment_digest(payload) # Prove every value has a supported primitive encoding.
    jldopen(path,"w") do file
        file["experiment_format_version"]=EXPERIMENT_FORMAT_VERSION
        file["experiment"]=payload
    end
    locator=realpath(path)
    locator in record.record_paths || push!(record.record_paths,locator)
    path
end

"""
    load_experiment(path) -> ExperimentRecord

Read and validate a versioned experiment. Unknown versions, missing/extra
schema fields, malformed settings, and recipe/input identity mismatches are
rejected. Loading never executes scripts or reads input-image payloads;
content and software checks are repeated by [`replay_experiment`](@ref).
"""
function load_experiment(path::AbstractString)
    payload=jldopen(path,"r") do file
        haskey(file,"experiment_format_version") && file["experiment_format_version"]===EXPERIMENT_FORMAT_VERSION ||
            _experiment_error("missing or unsupported experiment_format_version")
        haskey(file,"experiment") || _experiment_error("missing experiment payload")
        file["experiment"]
    end
    _experiment_keys(payload,["recipe","recipe_id","script_path","input_files","pairs","input_id","creation_environment","runs"],"experiment")
    recipe=_experiment_recipe_from_data(payload["recipe"],payload["script_path"])
    recipe.recipe_id==payload["recipe_id"] || _experiment_error("saved recipe identity mismatch")
    payload["input_files"] isa AbstractVector && payload["pairs"] isa AbstractVector && payload["runs"] isa AbstractVector || _experiment_error("invalid experiment collections")
    files=Dict{String,Any}[Dict{String,Any}(f) for f in payload["input_files"]]
    all(p -> p isa Vector{Int},payload["pairs"]) || _experiment_error("invalid pair encoding")
    pairs=Vector{Int}[copy(p) for p in payload["pairs"]]
    _experiment_hash(payload["input_id"]) || _experiment_error("invalid input identity")
    _experiment_validate_environment(payload["creation_environment"])
    record=ExperimentRecord(recipe,files,pairs,payload["input_id"],deepcopy(payload["creation_environment"]),ExperimentRun[],[realpath(path)])
    _experiment_preflight(record)
    append!(record.runs,[_experiment_run(r,record) for r in payload["runs"]])
    record
end

function _experiment_preprocess(recipe,custom)
    function preprocess(image)
        original_size=size(image)
        for step in recipe.preprocessing
            options=(Symbol(k)=>v for (k,v) in step.options)
            image = step.operation===:subtract_background ? subtract_background(image,step.options["background"]) :
                step.operation===:intensity_cap ? intensity_cap(image; options...) :
                step.operation===:highpass_filter ? highpass_filter(image; options...) :
                step.operation===:clahe ? clahe(image; tiles=Tuple(step.options["tiles"]),clip_limit=step.options["clip_limit"],nbins=step.options["nbins"]) :
                step.operation===:percentile_stretch ? percentile_stretch(image; options...) :
                step.operation===:invert_image ? invert_image(image) : local_variance_normalize(image; options...)
        end
        custom===nothing || (image=custom(image))
        image isa AbstractMatrix{recipe.image_type} && all(isfinite,image) || _experiment_error("preprocessing must return a finite matrix in the recipe precision")
        size(image)==original_size || _experiment_error("preprocessing must preserve original image dimensions")
        image
    end
end

function _experiment_record_run(record,output,environment,started,status,completed,error)
    digest=isfile(output) ? try _experiment_file_digest(output) catch; nothing end : nothing
    ExperimentRun(string(UUIDs.uuid4()),record.recipe.recipe_id,record.input_id,started,time(),status,
        completed,abspath(output),digest,environment,error)
end

"""
    replay_experiment(record; output, custom_preprocess=nothing,
                      allow_environment_change=false, run_record=nothing) -> ExperimentRun

Rerun a saved file-based planar recipe from the beginning into a native result
file, without collecting result arrays. Before opening `output`, verify settings,
every input's bytes/dimensions, script content, backend capabilities, and software
compatibility. Output may not alias input images, scripts, any known saved
experiment, or `run_record`; same-file aliases are checked. Concurrent input
changes are unsupported; bytes are checked again before/after each image load.

By default Julia/core/package versions, core source content, platform, and
thread settings must match the creation environment. Set
`allow_environment_change=true` explicitly to rerun elsewhere; the actual run
environment is always recorded, with no promise of bitwise cross-environment
reproducibility. Recorded Project/Manifest text is provenance, never activated
or instantiated by replay. Recipe settings cannot be overridden by extra kwargs.

A referenced custom preprocessor requires an explicit `custom_preprocess`
function, applied after built-in steps. No `eval`, `include`, or automatic
entrypoint lookup occurs. Supplying an unreferenced custom function is rejected.
The caller is responsible for its correspondence to the referenced script and
for external state that a function uses.

Returns completed-run metadata; access numerical results with [`load_results`](@ref).
Optional `run_record` saves the experiment with appended run metadata, including
a failed run when processing fails. Preflight failures leave output and run
records unchanged. Processing failures rethrow their original exception and
leave the existing native completed prefix. A secondary run-record write failure
is logged without replacing the original processing exception. This is not a
resume API or an atomic transaction spanning the two files.
"""
function replay_experiment(record::ExperimentRecord; output::AbstractString,
                           custom_preprocess::Union{Nothing,Function}=nothing,
                           allow_environment_change::Bool=false,
                           run_record::Union{Nothing,AbstractString}=nothing)
    snapshot=deepcopy(record)
    _experiment_preflight(snapshot; verify_files=true)
    _experiment_validate_environment(snapshot.creation_environment)
    for run in snapshot.runs
        _experiment_run(_experiment_run_data(run),snapshot)
    end
    _experiment_protect(output,snapshot; records=true)
    if isfile(output)
        existing_record=try
            jldopen(output,"r") do file
                haskey(file,"experiment_format_version")
            end
        catch
            false
        end
        existing_record && _experiment_error("result output must not overwrite a saved experiment")
    end
    if run_record!==nothing
        _experiment_protect(run_record,snapshot; results=true)
        _experiment_alias(output,run_record) && _experiment_error("result output and run record must differ")
    end
    reference=snapshot.recipe.external_preprocess
    (reference===nothing)==(custom_preprocess===nothing) || _experiment_error("custom_preprocess requires exactly one recorded script reference")
    environment=_experiment_software()
    allow_environment_change || _experiment_environment_signature(environment)==_experiment_environment_signature(snapshot.creation_environment) ||
        _experiment_error("software environment differs from experiment creation; explicitly allow_environment_change to rerun")
    recipe=snapshot.recipe
    source=FrameSource(length(snapshot.input_files),i -> _experiment_load_image(snapshot.input_files[i],recipe.image_type))
    pairs=[(FrameRef(source,p[1]),FrameRef(source,p[2])) for p in snapshot.pairs]
    preprocess=_experiment_preprocess(recipe,custom_preprocess)
    started=time(); completed=Ref(0)
    run=try
        run_piv_sequence(pairs,recipe.passes; preprocess,output,collect_results=false,
            progress=(i,n)->(completed[]=i),backend=recipe.backend,image_type=recipe.image_type,
            mask=recipe.mask,roi=recipe.roi,scale=recipe.scale,threaded=recipe.threaded,
            predictor_smoothing=recipe.predictor_smoothing,mask_threshold=recipe.mask_threshold,
            uncertainty_backend=recipe.uncertainty_backend)
        _experiment_record_run(snapshot,output,environment,started,:completed,completed[],nothing)
    catch err
        failed=_experiment_record_run(snapshot,output,environment,started,:failed,completed[],sprint(showerror,err))
        if run_record!==nothing
            try save_experiment(run_record,snapshot; runs=[snapshot.runs;failed])
            catch recording_error
                @error "Failed to save experiment failure metadata" exception=recording_error
            end
        end
        rethrow()
    end
    run_record===nothing || save_experiment(run_record,snapshot; runs=[snapshot.runs;run])
    run
end
