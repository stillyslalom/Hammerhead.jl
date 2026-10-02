# Reproducible validation baseline; CPU processing timings are deliberately
# separate from fixture loading, calibration, generation, and report writing.
module ValidationScorecard

using Hammerhead
using Hammerhead.SyntheticData: generate_gaussian_particle!
using Statistics
using LinearAlgebra
using SHA
using TOML
using Dates

const ROOT = normpath(joinpath(@__DIR__, ".."))
const SCHEMA = "hammerhead-validation-scorecard-1"
const REPORT_MARKER = "Hammerhead validation scorecard report"

file_digest(path) = open(io -> bytes2hex(sha256(io)), path)

# A specified counter stream avoids Julia-version-dependent default RNGs.
# SplitMix64 constants/arithmetic, then the upper 53 bits map to [0, 1).
function uniform!(state::Base.RefValue{UInt64})
    state[] += 0x9e3779b97f4a7c15
    z = state[]
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return Float64((z ⊻ (z >> 31)) >> 11) * 0x1.0p-53
end

function pixel_digest(image)
    io = IOBuffer()
    for value in image # canonical Float64 little-endian, column-major
        bits = reinterpret(UInt64, Float64(value))
        for shift in 0:8:56
            write(io, UInt8((bits >> shift) & 0xff))
        end
    end
    return bytes2hex(sha256(take!(io)))
end

function synthetic_scene(; seed = 7321, size = 128, density = 0.02,
        diameter = 3.0, noise = 0.0, shear = 0.0, dropout = 0.0,
        du = 2.25, dv = -1.5)
    size >= 32 && density >= 0 && diameter > 0 && noise >= 0 &&
        0 <= dropout <= 1 || throw(ArgumentError("invalid synthetic scene settings"))
    state = Ref(UInt64(seed))
    a, b = zeros(size, size), zeros(size, size)
    n = round(Int, density * size^2)
    removed = 0
    center = (size + 1) / 2
    for _ in 1:n
        x, y = 1 + (size - 1) * uniform!(state), 1 + (size - 1) * uniform!(state)
        drop = uniform!(state) < dropout
        generate_gaussian_particle!(a, (x, y), diameter)
        if drop
            removed += 1
        else
            generate_gaussian_particle!(b, (x + du + shear * (y - center), y + dv), diameter)
        end
    end
    for i in eachindex(a)
        a[i] = max(0.0, a[i] + noise * (2uniform!(state) - 1))
        b[i] = max(0.0, b[i] + noise * (2uniform!(state) - 1))
    end
    spec = Dict("generator_version" => "splitmix64-gaussian-particles-1",
        "seed" => seed, "image_size" => [size, size], "particle_density" => density,
        "particle_count" => n, "diameter_px_4sigma" => diameter,
        "particle_intensity" => 1.0, "noise_uniform_half_width" => noise,
        "noise_clamp" => "max(0, intensity + independent uniform noise)",
        "dropout_requested_fraction" => dropout, "dropout_removed_count" => removed,
        "du_at_center_px" => du, "dv_px" => dv, "du_dy" => shear,
        "center_row" => center, "dt" => 1.0,
        "integration" => "one forward-Euler step from launch positions",
        "renderer" => "SyntheticData.generate_gaussian_particle!; diameter=4sigma; clipped 3sigma support",
        "pixel_hash_encoding" => "Float64 little-endian column-major",
        "image_a_sha256" => pixel_digest(a), "image_b_sha256" => pixel_digest(b))
    # For u(y), constant v, launch y = midpoint y - dv/2 exactly. This is
    # an analytic inverse of the synthetic step, not measured-vector truth.
    truth = (x, y) -> (du + shear * (y - dv / 2 - center), dv)
    return (; a, b, spec, truth)
end

serial(x::Symbol) = String(x)
serial(x::Union{AbstractString,Number}) = x
serial(x::Union{Tuple,AbstractVector}) = [serial(v) for v in x]
serial(x::Nothing) = "none"
pass_recipe(p) = Dict(String(k) => serial(getfield(p, k)) for k in fieldnames(PIVParameters))

function recipe(passes; kwargs...)
    result = Dict{String,Any}("passes" => [pass_recipe(p) for p in passes],
        "backend" => "cpu", "precision" => "Float64", "threaded" => false,
        "predictor_smoothing" => true, "mask_threshold" => 0.5,
        "uncertainty_backend" => "same", "preprocessing" => "none",
        "user_mask" => "none", "roi" => "none", "scale" => "none",
        "workspace" => "fresh per call")
    merge!(result, Dict(String(k) => v for (k, v) in kwargs))
    return result
end

function metrics(result; truth = nothing)
    arrays = result isa StereoPIVResult ? (result.u, result.v, result.w) : (result.u, result.v)
    dims = (length(result.y), length(result.x))
    all(a -> size(a) == dims, (arrays..., result.mask, result.outliers)) ||
        throw(ArgumentError("scorecard component/flag dimensions must match coordinates"))
    eligible = .!result.mask
    finite = map(i -> all(a -> isfinite(a[i]), arrays), CartesianIndices(result.u))
    selected = eligible .& .!result.outliers .& finite
    neligible, nvalid = count(eligible), count(selected)
    out = Dict{String,Any}("grid_nodes" => length(selected), "masked_count" => count(result.mask),
        "eligible_count" => neligible, "valid_count" => nvalid,
        "outlier_count_unmasked" => count(eligible .& result.outliers),
        "nonfinite_count_unmasked" => count(eligible .& .!finite),
        "yield_denominator" => "all unmasked grid nodes; flagged/nonfinite nodes remain in denominator",
        "selection" => "unmasked AND not flagged AND all stored components finite; replacement does not admit flagged vectors",
        "valid_yield" => neligible == 0 ? "unavailable: no eligible nodes" : nvalid / neligible)
    if truth === nothing
        out["error_status"] = "unavailable: no displacement truth/reference supplied"
    elseif nvalid == 0
        out["error_status"] = "unavailable: no valid vectors"
    else
        errors = [Float64[] for _ in arrays]
        for index in findall(selected)
            ref = truth(result.x[index[2]], result.y[index[1]])
            length(ref) == length(arrays) && all(isfinite, ref) ||
                throw(ArgumentError("truth must provide finite components for every selected node"))
            for k in eachindex(arrays)
                push!(errors[k], arrays[k][index] - ref[k])
            end
        end
        out["error_status"] = "measured against supplied truth"
        out["bias"] = Dict(String(k) => mean(e) for (k, e) in zip((:u, :v, :w), errors))
        out["rms_error"] = Dict(String(k) => sqrt(mean(abs2, e)) for (k, e) in zip((:u, :v, :w), errors))
        out["error_definition"] = "measured minus reference; RMS includes bias; population mean over identical selected nodes"
    end
    return out
end

function fixture_identity(paths)
    [Dict("path" => replace(relpath(path, ROOT), '\\' => '/'),
          "sha256" => file_digest(path), "bytes" => filesize(path)) for path in paths]
end

function synthetic_case(id, settings; final_window = 16)
    scene = synthetic_scene(; settings...)
    passes = multipass_parameters([2final_window, final_window, final_window];
        padding = true, apodization = :gauss)
    (; id, category = "synthetic_ground_truth", inputs = scene.spec,
       recipe = recipe(passes), truth = scene.truth, units = "px per image pair",
       reference = "analytic launch-to-midpoint inverse: y_launch=y_vector-dv/2; u=du+shear*(y_launch-center); v=dv; no measured displacement in reference",
       run = () -> run_piv(scene.a, scene.b, passes; threaded = false))
end

function case_a()
    dir = joinpath(ROOT, "test", "reference_images", "A")
    paths = [joinpath(dir, "A001_$i.tif") for i in 1:2]
    a, b = load_image.(paths)
    roi = ROI(257:768, 385:896)
    passes = multipass_parameters([64, 32, 32]; padding = true, apodization = :gauss)
    (; id = "challenge_2001_A_committed_crop", category = "real_data_smoke",
       inputs = Dict("files" => fixture_identity([paths; joinpath(dir, "readmeA.txt")]),
           "original_image_size" => collect(size(a)), "evaluated_roi" => [257, 768, 385, 896]),
       recipe = recipe(passes; roi = [257, 768, 385, 896]), truth = nothing,
       reference = "committed tip-vortex pair; no numerical ground truth or independent reference field supplied",
       units = "px per image pair", run = () -> run_piv(a, b, passes; threaded = false, roi))
end

function case_e()
    dir = joinpath(ROOT, "test", "reference_images", "E")
    plate_paths = [joinpath(dir, "E_camera_$(cam)_z_$(k).png") for cam in (1, 3) for k in (1, 4, 7)]
    particle_paths = [joinpath(dir, "E_camera_$(cam)_frame_000$(k).png") for cam in (1, 3) for k in (50, 51)]
    detect(path) = detect_calibration_grid(load_image(path); spacing = 15.0,
        two_level = true, level_separation = 3.0, origin_offset = (30.0, 7.5))
    cameras = [calibrate_camera(detect.(plate_paths[1:3]), [-3., 0., 3.]),
               calibrate_camera(detect.(plate_paths[4:6]), [-3., 0., 3.])]
    grid = DewarpGrid(x = -36.0:0.5:22.0, y = -32.5:0.5:29.0)
    dws = [ImageDewarper(cam, grid, (1024, 1024)) for cam in cameras]
    a1, b1, a3, b3 = load_image.(particle_paths)
    passes = multipass_parameters([32, 16, 16]; padding = true, apodization = :gauss)
    setup = Dict("camera_model" => "SoloffCamera", "plate_z_mm" => [-3., 0., 3.],
        "spacing_mm" => 15.0, "two_level" => true, "level_separation_mm" => 3.0,
        "origin_offset_mm" => [30.0, 7.5], "orientation" => "image", "invert" => false,
        "dewarp_grid" => Dict("x" => collect(grid.x), "y" => collect(grid.y), "z" => grid.z),
        "camera_fits" => [Dict(String(k) => collect(getfield(cam, k)) for k in fieldnames(typeof(cam))) for cam in cameras],
        "self_calibration" => "none: coarse smoke workflow, not the full real-data tutorial recipe",
        "synchronization" => "assumed from fixture readme; matrix inputs contain no timing metadata",
        "overlap_mask" => "union of both dewarpers' out-of-view masks")
    (; id = "challenge_2014_4E_committed_coarse_stereo", category = "real_data_smoke",
       inputs = Dict("files" => fixture_identity([plate_paths; particle_paths; joinpath(dir, "readmeE.txt")])),
       recipe = recipe(passes; stereo_setup = setup), truth = nothing,
       reference = "committed cameras 1+3 frames 50/51; no numerical displacement truth; coarse uncorrected calibration workflow",
       units = "mm per image pair", run = () -> run_piv_stereo(a1, b1, a3, b3, dws[1], dws[2], passes; threaded = false))
end

function evaluate(prepare; samples::Int = 3)
    samples >= 1 || throw(ArgumentError("samples must be positive"))
    case = nothing
    setup_seconds = @elapsed case = prepare()
    case.run() # one exact-recipe warmup, excluded from measurements
    times, bytes, gc_times = Float64[], Int[], Float64[]
    result = nothing
    for _ in 1:samples
        GC.gc() # controlled pre-sample collection, excluded from wall time
        measured = @timed case.run()
        result = measured.value
        push!(times, measured.time); push!(bytes, measured.bytes); push!(gc_times, measured.gctime)
    end
    measured_metrics = metrics(result; truth = case.truth)
    status = measured_metrics["valid_count"] == 0 ? "no_valid_vectors" : "measured"
    empty_scene = case.category == "synthetic_ground_truth" && case.inputs["particle_count"] == 0
    empty_scene && measured_metrics["valid_count"] > 0 && (status = "unexpected_valid_vectors_in_empty_scene")
    Dict{String,Any}("dataset_id" => case.id, "category" => case.category,
        "status" => status, "expected_failure_case" => empty_scene,
        "inputs" => case.inputs, "recipe" => case.recipe, "reference_convention" => case.reference,
        "component_units" => case.units, "metrics" => measured_metrics,
        "preparation_seconds" => setup_seconds,
        "performance" => Dict("scope" => "loaded-image CPU PIV call; stereo includes dewarping/reconstruction; excludes generation/loading/calibration/hashing/reporting",
            "warmup_calls" => 1, "gc_before_each_sample" => true,
            "runtime_seconds_samples" => times, "runtime_seconds_median" => median(times),
            "runtime_seconds_min" => minimum(times), "runtime_seconds_max" => maximum(times),
            "julia_allocated_bytes_samples" => bytes, "julia_allocated_bytes_median" => median(bytes),
            "gc_seconds_samples" => gc_times,
            "allocation_definition" => "cumulative bytes allocated by Julia during each call; NOT peak resident/host/device memory; native-library allocations may be absent"))
end

function environment_record()
    project = Base.active_project()
    manifest = project === nothing ? "" : joinpath(dirname(project), "Manifest.toml")
    source_paths = sort(filter(endswith(".jl"), readdir(joinpath(ROOT, "src"); join = true)))
    sources = fixture_identity([source_paths; @__FILE__])
    Dict{String,Any}("julia_version" => string(VERSION), "hammerhead_version" => string(Base.pkgversion(Hammerhead)),
        "os" => string(Sys.KERNEL), "architecture" => string(Sys.ARCH),
        "cpu_model" => first(Sys.cpu_info()).model, "julia_threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(), "fftw_threads" => Hammerhead.FFTW.get_num_threads(),
        "backend" => "cpu", "processing_threaded" => false,
        "active_project_path" => project === nothing ? "unavailable" : project,
        "active_project_toml" => project === nothing ? "unavailable" : read(project, String),
        "active_manifest_toml" => isfile(manifest) ? read(manifest, String) : "unavailable",
        "source_files" => sources)
end

function run_scorecard(; expanded::Bool = false, samples::Int = 3)
    environment = environment_record()
    preparations = Function[
        () -> synthetic_case("synthetic_translation", (;)),
        () -> synthetic_case("synthetic_linear_shear", (; shear = 0.025)),
        () -> synthetic_case("synthetic_no_particles_failure", (; density = 0.0)),
        case_a, case_e]
    if expanded
        for (id, settings, window) in (
            ("synthetic_sparse", (; density = 0.006), 16),
            ("synthetic_dense", (; density = 0.04), 16),
            ("synthetic_large_particles", (; diameter = 5.0), 16),
            ("synthetic_uniform_noise", (; noise = 0.03), 16),
            ("synthetic_stronger_shear", (; shear = 0.06), 16),
            ("synthetic_dropout", (; dropout = 0.35), 16),
            ("synthetic_translation_window32", (;), 32))
            push!(preparations, let id = id, settings = settings, window = window
                () -> synthetic_case(id, settings; final_window = window)
            end)
        end
    end
    rows = [evaluate(prepare; samples) for prepare in preparations]
    push!(rows, Dict("dataset_id" => "known_motion_experiment", "category" => "known_motion_experiment",
        "status" => "unavailable", "reason" => "no committed known-motion acquisition and independently measured motion reference supplied"))
    environment["source_files_unchanged_during_run"] = all(environment["source_files"]) do entry
        path = joinpath(ROOT, entry["path"])
        isfile(path) && file_digest(path) == entry["sha256"]
    end
    provenance_status = environment["source_files_unchanged_during_run"] ?
        "source hashes stable during run" : "source changed during run: freeze checkout and regenerate in a fresh Julia process"
    Dict{String,Any}("schema_version" => SCHEMA, "generated_utc" => string(now(UTC)),
        "expanded" => expanded, "samples_per_case" => samples, "environment" => environment,
        "provenance_status" => provenance_status, "cases" => rows,
        "limitations" => ["Synthetic errors validate the specified renderer/flow/recipe, not experimental accuracy.",
            "Real A and 4E rows are smoke checks with no accuracy reference; 4E uses a deliberately coarse grid and no self-calibration.",
            "Known-motion experiments and independent Challenge datasets beyond committed A/4E remain unavailable.",
            "Uncertainty coverage, PTV identity metrics, peak host/device memory, production-scale and GPU hardware evidence remain unmeasured.",
            "Timing samples show local runtime variation; these are not accuracy pass/fail gates or performance promises."])
end

function markdown_report(report)
    io = IOBuffer()
    println(io, "<!-- $REPORT_MARKER -->\n# Validation scorecard\n")
    println(io, "Schema: `", report["schema_version"], "`. Full provenance and recipes: `scorecard.toml`.\n")
    haskey(report, "provenance_status") && println(io, "Provenance: ", report["provenance_status"], ".\n")
    println(io, "| Dataset | Category/status | Valid/eligible | Bias u/v | RMS error u/v | Median CPU seconds | Julia allocated bytes (median) |")
    println(io, "|---|---|---:|---:|---:|---:|---:|")
    number(x) = string(round(x; sigdigits = 5))
    for row in report["cases"]
        m, p = get(row, "metrics", Dict()), get(row, "performance", Dict())
        pair(key) = haskey(m, key) ? join([number(m[key][k]) for k in ("u", "v")], " / ") : "unavailable"
        valid = haskey(m, "valid_count") ? "$(m["valid_count"])/$(m["eligible_count"])" : "unavailable"
        runtime = haskey(p, "runtime_seconds_median") ? number(p["runtime_seconds_median"]) : "unavailable"
        allocations = haskey(p, "julia_allocated_bytes_median") ? number(p["julia_allocated_bytes_median"]) : "unavailable"
        println(io, "| $(row["dataset_id"]) | $(row["category"])/$(row["status"]) | $valid | $(pair("bias")) | $(pair("rms_error")) | $runtime | $allocations |")
    end
    println(io, "\nBias/RMS are displacement errors in each row's component units. Allocations are cumulative Julia bytes, **not peak memory**. Preparation is excluded from CPU-call timing.\n")
    for limitation in report["limitations"]
        println(io, "- ", limitation)
    end
    return String(take!(io))
end

function write_report(directory, report)
    target = abspath(directory)
    function within(path, parent)
        p, base = splitpath(normpath(abspath(path))), splitpath(normpath(abspath(parent)))
        Sys.iswindows() && ((p, base) = (lowercase.(p), lowercase.(base)))
        return length(p) >= length(base) && p[1:length(base)] == base
    end
    if within(target, ROOT)
        allowed = normpath(joinpath(ROOT, "bench", "profile-output"))
        within(target, allowed) ||
            throw(ArgumentError("repository scorecard output must be under bench/profile-output; source/fixture directories are protected"))
    end
    paths = [joinpath(target, "scorecard.toml"), joinpath(target, "scorecard.md")]
    for path in paths
        if isfile(path)
            first_line = open(readline, path)
            occursin(REPORT_MARKER, first_line) ||
                throw(ArgumentError("refusing to overwrite an unrelated file: $path"))
        end
    end
    mkpath(target)
    open(paths[1], "w") do io
        println(io, "# $REPORT_MARKER")
        TOML.print(io, report; sorted = true)
    end
    write(paths[2], markdown_report(report))
    return paths
end

function main(args = ARGS)
    expanded, samples = false, 3
    output = joinpath(ROOT, "bench", "profile-output", "validation-scorecard")
    for arg in args
        if arg == "--expanded"
            expanded = true
        elseif startswith(arg, "--samples=")
            samples = parse(Int, split(arg, '='; limit = 2)[2])
        elseif startswith(arg, "--output=")
            output = split(arg, '='; limit = 2)[2]
        elseif arg == "--help"
            println("julia --project=. -t 4 bench/validation_scorecard.jl [--expanded] [--samples=3] [--output=directory]")
            return nothing
        else
            throw(ArgumentError("unknown scorecard option: $arg"))
        end
    end
    samples >= 1 || throw(ArgumentError("samples must be positive"))
    report = run_scorecard(; expanded, samples)
    paths = write_report(output, report)
    println(markdown_report(report))
    println("Report files: ", join(paths, ", "))
    return report
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    ValidationScorecard.main()
end
