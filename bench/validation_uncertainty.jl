# Bench-only synthetic evaluation. No production API or algorithm changes.
module ValidationUncertainty

using Hammerhead
using Statistics
using TOML
using Dates
include("validation_scorecard.jl")
const V = ValidationScorecard
const ROOT = V.ROOT
const SCHEMA = "hammerhead-synthetic-uncertainty-1"
const MARKER = "Hammerhead synthetic uncertainty report"
const LEVELS = (1.0, 2.0)
const DEFAULT_SEEDS = (7321, 7322, 7323)
const EXPANDED_SEEDS = Tuple(7321:7328)

# Count every attempted value, including arithmetic overflow. Scaled squares
# avoid overflow merely from squaring a large, representable error.
mutable struct Moment
    count::Int
    finite_count::Int
    mean::Float64
    scale::Float64
    squares::Float64
end
Moment() = Moment(0, 0, 0.0, 0.0, 0.0)
function add!(m::Moment, value)
    m.count += 1
    x = Float64(value)
    isfinite(x) || return m
    m.finite_count += 1
    delta = x - m.mean
    m.mean = isfinite(delta) ? m.mean + delta / m.finite_count :
        m.mean * ((m.finite_count - 1) / m.finite_count) + x / m.finite_count
    a = abs(x)
    if a > m.scale
        m.squares = 1 + m.squares * (m.scale / a)^2
        m.scale = a
    elseif a > 0
        m.squares += (a / m.scale)^2
    end
    m
end
function merge!(a::Moment, b::Moment)
    n = a.finite_count + b.finite_count
    if b.finite_count > 0
        a.mean = a.finite_count == 0 ? b.mean :
            a.mean * (a.finite_count / n) + b.mean * (b.finite_count / n)
        scale = max(a.scale, b.scale)
        a.squares = scale == 0 ? 0.0 :
            a.squares * (a.scale / scale)^2 + b.squares * (b.scale / scale)^2
        a.scale = scale
    end
    a.count += b.count
    a.finite_count = n
    a
end
function summary(m::Moment)
    out = Dict{String,Any}("count" => m.count, "finite_count" => m.finite_count,
        "arithmetic_unavailable_count" => m.count - m.finite_count)
    rms = m.finite_count == 0 ? 0.0 : m.scale * sqrt(m.squares / m.finite_count)
    available = m.count > 0 && m.finite_count == m.count && isfinite(m.mean) && isfinite(rms)
    out["available"] = available
    if available
        out["mean"] = m.mean; out["rms"] = rms
    else
        out["reason"] = m.count == 0 ? "empty_population" : "nonfinite_arithmetic"
    end
    out
end

const COMPONENT_COUNTS = (
    "primary_valid", "error_arithmetic_unavailable", "uq_available", "uq_nonfinite", "uq_negative",
    "sigma_zero", "sigma_positive", "sigma_above_0_3", "uq_error_arithmetic_unavailable",
    "zero_sigma_zero_error", "zero_sigma_nonzero_error", "zero_sigma_error_arithmetic_unavailable",
    "normalized_arithmetic_unavailable", "covered_1sigma", "covered_2sigma")
mutable struct ComponentPopulation
    counts::Dict{String,Int}
    primary_error::Moment
    uq_subset_error::Moment
    sigma::Moment
    normalized_error::Moment
end
ComponentPopulation() = ComponentPopulation(Dict(k => 0 for k in COMPONENT_COUNTS),
    Moment(), Moment(), Moment(), Moment())
mutable struct Population
    counts::Dict{String,Int}
    components::Dict{String,ComponentPopulation}
end
Population() = Population(Dict(k => 0 for k in ("grid_nodes", "masked", "unmasked",
    "flagged_unmasked", "nonfinite_output_unmasked", "flagged_finite", "flagged_nonfinite",
    "unflagged_nonfinite", "primary_valid")),
    Dict(k => ComponentPopulation() for k in ("u", "v")))
function merge!(a::Population, b::Population)
    for k in keys(a.counts)
        a.counts[k] += b.counts[k]
    end
    for k in ("u", "v")
        aa, bb = a.components[k], b.components[k]
        for key in keys(aa.counts)
            aa.counts[key] += bb.counts[key]
        end
        for field in (:primary_error, :uq_subset_error, :sigma, :normalized_error)
            merge!(getfield(aa, field), getfield(bb, field))
        end
    end
    a
end

function component!(p::ComponentPopulation, error, sigma, normalized_values)
    c = p.counts
    c["primary_valid"] += 1
    add!(p.primary_error, error)
    !isfinite(error) && (c["error_arithmetic_unavailable"] += 1)
    if !isfinite(sigma)
        c["uq_nonfinite"] += 1
        return
    elseif sigma < 0
        c["uq_negative"] += 1
        return
    end
    c["uq_available"] += 1
    sigma > 0.3 && (c["sigma_above_0_3"] += 1)
    add!(p.sigma, sigma); add!(p.uq_subset_error, error)
    !isfinite(error) && (c["uq_error_arithmetic_unavailable"] += 1)
    # Divide instead of forming k*sigma, which may overflow. Zero sigma has
    # exact-error coverage but never a normalized error (including 0/0).
    z = sigma == 0 ? NaN : error / sigma
    for k in LEVELS
        covered = isfinite(error) && (sigma == 0 ? error == 0 : abs(z) <= k)
        covered && (c[k == 1 ? "covered_1sigma" : "covered_2sigma"] += 1)
    end
    if sigma == 0
        c["sigma_zero"] += 1
        key = !isfinite(error) ? "zero_sigma_error_arithmetic_unavailable" :
            error == 0 ? "zero_sigma_zero_error" : "zero_sigma_nonzero_error"
        c[key] += 1
    else
        c["sigma_positive"] += 1
        add!(p.normalized_error, z)
        if isfinite(z)
            push!(normalized_values, z)
        else
            c["normalized_arithmetic_unavailable"] += 1
        end
    end
end

function controlled_passes(final_window = 16)
    multipass_parameters([2final_window, final_window, final_window];
        padding = true, apodization = :gauss, n_peaks = 1, replace_outliers = false,
        final = (uncertainty = true, max_iterations = 1,))
end
function check_primary(parameters)
    parameters.n_peaks == 1 && !parameters.replace_outliers &&
        parameters.max_iterations == 1 && parameters.uncertainty ||
        throw(ArgumentError("UQ evaluation requires a controlled primary-only final pass: n_peaks=1, replace_outliers=false, max_iterations=1, uncertainty=true"))
    all(v -> v isa Union{PeakRatioValidator,CorrelationMomentValidator,VelocityMagnitudeValidator,UniversalOutlierValidator},
        parameters.validation) || throw(ArgumentError("custom validators cannot establish primary measurement origin"))
    nothing
end

# Exact per-seed quantiles only. Convex interpolation avoids subtracting
# opposite large signs; a nonrepresentable summary stays unavailable.
function quantiles(values, expected)
    out = Dict{String,Any}("count" => expected, "finite_count" => length(values),
        "scope" => "this seed only; not pooled or averaged across seeds")
    out["available"] = expected > 0 && length(values) == expected
    if !out["available"]
        out["reason"] = expected == 0 ? "empty_population" : "nonfinite_arithmetic"
        return out
    end
    sort!(values)
    for (key, probability) in (("q05", 0.05), ("q50", 0.5), ("q95", 0.95))
        index = 1 + (length(values) - 1) * probability
        lo, hi = floor(Int, index), ceil(Int, index)
        alpha = index - lo
        value = lo == hi ? values[lo] : values[lo] * (1 - alpha) + values[hi] * alpha
        if !isfinite(value)
            return Dict{String,Any}("count" => expected, "finite_count" => length(values),
                "scope" => out["scope"], "available" => false, "reason" => "nonfinite_arithmetic")
        end
        out[key] = value
    end
    out
end

function summary(p::Population)
    out = Dict{String,Any}("counts" => copy(p.counts), "components" => Dict{String,Any}())
    n = p.counts["unmasked"]
    out["valid_yield"] = n == 0 ? Dict("available" => false, "reason" => "no_unmasked_nodes") :
        Dict("available" => true, "numerator" => p.counts["primary_valid"], "denominator" => n,
            "fraction" => p.counts["primary_valid"] / n)
    for key in ("u", "v")
        c = p.components[key]; counts = c.counts
        coverage = Dict{String,Any}()
        for level in (1, 2)
            available = counts["uq_available"] > 0 && counts["uq_error_arithmetic_unavailable"] == 0
            entry = Dict{String,Any}("available" => available,
                "covered_count" => counts["covered_$(level)sigma"], "denominator" => counts["uq_available"],
                "arithmetic_unavailable_count" => counts["uq_error_arithmetic_unavailable"])
            if available
                entry["fraction"] = counts["covered_$(level)sigma"] / counts["uq_available"]
            else
                entry["reason"] = counts["uq_available"] == 0 ? "empty_population" : "nonfinite_error_arithmetic"
            end
            coverage[string(level)] = entry
        end
        out["components"][key] = Dict("counts" => copy(counts),
            "primary_error" => summary(c.primary_error), "uq_subset_error" => summary(c.uq_subset_error),
            "sigma" => summary(c.sigma), "normalized_error" => summary(c.normalized_error), "coverage" => coverage)
    end
    out
end

# This bench helper assumes trusted synthetic measurement/test inputs. Stored
# parameters alone cannot certify the origin of an arbitrary historical result.
function metrics(result::PIVResult, truth)
    check_primary(result.parameters)
    dims = (length(result.y), length(result.x))
    all(a -> size(a) == dims, (result.u, result.v, result.mask, result.outliers,
        result.uncertainty_u, result.uncertainty_v)) || throw(ArgumentError("inconsistent result dimensions"))
    all(isfinite, result.x) && all(isfinite, result.y) || throw(ArgumentError("nonfinite grid coordinates"))
    p = Population()
    values = Dict(k => Float64[] for k in ("u", "v"))
    for index in CartesianIndices(result.u)
        p.counts["grid_nodes"] += 1
        if result.mask[index]
            p.counts["masked"] += 1
            continue
        end
        p.counts["unmasked"] += 1
        finite = isfinite(result.u[index]) && isfinite(result.v[index])
        result.outliers[index] && (p.counts["flagged_unmasked"] += 1)
        !finite && (p.counts["nonfinite_output_unmasked"] += 1)
        key = result.outliers[index] ? (finite ? "flagged_finite" : "flagged_nonfinite") :
            finite ? "primary_valid" : "unflagged_nonfinite"
        key != "primary_valid" && (p.counts[key] += 1)
        (!finite || result.outliers[index]) && continue
        reference = truth(result.x[index[2]], result.y[index[1]])
        length(reference) == 2 && all(isfinite, reference) ||
            throw(ArgumentError("truth must supply two finite components at each primary-valid node"))
        p.counts["primary_valid"] += 1
        for (k, output, uq, ref) in (("u", result.u[index], result.uncertainty_u[index], reference[1]),
                                    ("v", result.v[index], result.uncertainty_v[index], reference[2]))
            component!(p.components[k], Float64(output) - Float64(ref), Float64(uq), values[k])
        end
    end
    data = summary(p)
    for key in ("u", "v")
        data["components"][key]["normalized_error_quantiles"] =
            quantiles(values[key], p.components[key].counts["sigma_positive"])
    end
    (; data, population = p)
end

function conditions(expanded = false)
    rows = NamedTuple[(id = "baseline", settings = (;), window = 16),
        (id = "noise_0_03", settings = (; noise = 0.03), window = 16),
        (id = "dropout_0_35", settings = (; dropout = 0.35), window = 16),
        (id = "shear_0_025", settings = (; shear = 0.025), window = 16)]
    if expanded
        append!(rows, [
            (id = "density_0_006", settings = (; density = 0.006), window = 16),
            (id = "density_0_04", settings = (; density = 0.04), window = 16),
            (id = "diameter_2", settings = (; diameter = 2.0), window = 16),
            (id = "diameter_5", settings = (; diameter = 5.0), window = 16),
            (id = "noise_0_1", settings = (; noise = 0.1), window = 16),
            (id = "dropout_0_6", settings = (; dropout = 0.6), window = 16),
            (id = "shear_0_06", settings = (; shear = 0.06), window = 16),
            (id = "baseline_window32", settings = (;), window = 32)])
    end
    rows
end

# Exclude execution UUIDs from scientific content. Diagnostics describe the
# actual final primary residual population, not per-node UQ applicability.
function diagnostic_data(diagnostics)
    passes = [Dict{String,Any}("pass_index" => d.pass_index,
        "requested_iterations" => d.requested_iterations, "executed_iterations" => d.executed_iterations,
        "stop_reason" => String(d.stop_reason), "tolerance_checks" => d.checks,
        "residual" => Dict(String(k) => (v === nothing ? "unavailable" : v isa Symbol ? String(v) : v)
            for (k, v) in pairs(d.residual))) for d in diagnostics.passes]
    Dict("passes" => passes,
        "assumption_status" => "repeated final windows; zero-residual convergence is not proven; no per-node residual selection",
        "residual_selection" => "finite unmasked primary residuals before validation; differs from error/UQ populations")
end

function evaluate(condition, seed; samples = 1, size = 128)
    1 <= samples <= 10 || throw(ArgumentError("samples must be between 1 and 10"))
    passes = controlled_passes(condition.window)
    check_primary(last(passes))
    scene = nothing
    setup = @elapsed scene = V.synthetic_scene(; condition.settings..., seed, size)
    diagnostic = Ref{Any}(nothing)
    run() = run_piv(scene.a, scene.b, passes; threaded = false,
        on_diagnostics = d -> (diagnostic[] = d))
    run() # exact-recipe warmup; not an independent image replicate
    times, bytes, gctimes = Float64[], Int[], Float64[]
    result = nothing
    for _ in 1:samples
        GC.gc()
        measured = @timed run()
        result = measured.value
        push!(times, measured.time); push!(bytes, measured.bytes); push!(gctimes, measured.gctime)
    end
    measured = metrics(result, scene.truth)
    n = measured.data["counts"]["primary_valid"]
    empty = scene.spec["particle_count"] == 0
    status = n == 0 ? "no_valid_measurements" : empty ? "unexpected_valid_measurements_in_empty_scene" : "measured"
    row = Dict{String,Any}("condition" => condition.id, "seed" => seed,
        "category" => "synthetic_ground_truth", "status" => status, "expected_failure_case" => empty,
        "inputs" => scene.spec, "recipe" => V.recipe(passes; on_diagnostics = "final primary residual summary"),
        "metrics" => measured.data, "diagnostics" => diagnostic_data(diagnostic[]),
        "preparation_seconds" => setup,
        "performance" => Dict("runtime_seconds_samples" => times,
            "runtime_seconds_median" => median(times), "runtime_seconds_min" => minimum(times),
            "runtime_seconds_max" => maximum(times), "julia_allocated_bytes_samples" => bytes,
            "julia_allocated_bytes_median" => median(bytes), "gc_seconds_samples" => gctimes,
            "warmup_calls" => 1, "gc_before_each_sample" => true,
            "scope" => "loaded-image CPU PIV call including diagnostic callback/hashing; excludes rendering, setup, metric reduction and report writing",
            "allocation_definition" => "cumulative Julia allocations per call; NOT peak host/device memory; native allocations may be absent"))
    (; row, population = measured.population)
end

function environment_record()
    environment = V.environment_record()
    extensions = String[]
    for (directory, _, files) in walkdir(joinpath(ROOT, "ext"))
        append!(extensions, [joinpath(directory, f) for f in files if endswith(f, ".jl")])
    end
    append!(environment["source_files"], V.fixture_identity([sort(extensions); @__FILE__]))
    software = Hammerhead._experiment_software()
    environment["software_environment_sha256"] = Hammerhead._experiment_digest(software)
    environment["active_project_toml"] = something(software["project_text"], "unavailable")
    environment["active_manifest_toml"] = something(software["manifest_text"], "unavailable")
    environment["packages"] = [Dict(k => something(v, "unavailable") for (k, v) in package)
        for package in software["packages"]]
    environment
end
function stable_environment(environment)
    files_stable = all(environment["source_files"]) do entry
        path = joinpath(ROOT, entry["path"])
        isfile(path) && filesize(path) == entry["bytes"] && V.file_digest(path) == entry["sha256"]
    end
    files_stable && environment["software_environment_sha256"] ==
        Hammerhead._experiment_digest(Hammerhead._experiment_software())
end

function run_scorecard(; expanded = false, samples = 1)
    1 <= samples <= 10 || throw(ArgumentError("samples must be between 1 and 10"))
    environment = environment_record()
    seeds = expanded ? EXPANDED_SEEDS : DEFAULT_SEEDS
    groups = Dict{String,Any}[]
    for condition in conditions(expanded)
        pooled = Population()
        rows = Dict{String,Any}[]
        for seed in seeds
            measured = evaluate(condition, seed; samples)
            merge!(pooled, measured.population)
            push!(rows, measured.row) # primitive summaries only; no fields/node-error arrays
        end
        push!(groups, Dict("condition" => condition.id, "replicates" => rows,
            "pooled" => summary(pooled), "aggregation" => "pooled-node counts and moments; no pooled quantiles or vector-independence assumption"))
    end
    empty = evaluate((id = "empty_failure", settings = (; density = 0.0), window = 16), first(seeds); samples).row
    stable = stable_environment(environment)
    Dict{String,Any}("schema_version" => SCHEMA, "generated_utc" => string(now(UTC)),
        "expanded" => expanded, "seed_list" => collect(seeds), "samples_per_seed" => samples,
        "environment" => environment, "source_and_environment_stable" => stable,
        "provenance_status" => stable ? "on-disk sources/environment stable; regenerate in a fresh Julia process" :
            "sources/environment changed: evidence is unstable; freeze and regenerate in a fresh Julia process",
        "groups" => groups, "empty_failure" => empty,
        "conventions" => Dict(
            "units" => "px displacement per image pair; sigma in px; normalized error dimensionless",
            "truth" => "one forward-Euler step; y_launch=y_vector-dv/2; u=du+shear*(y_launch-center), v=dv; no measured-vector reference",
            "association" => "controlled final n_peaks=1, replace_outliers=false, max_iterations=1; uncertainty=true; intermediate predictor filling remains part of recipe",
            "yield" => "all unmasked grid nodes denominator; primary_valid finite u/v AND unflagged AND unmasked",
            "grid_partition" => "unmasked=primary_valid+flagged_finite+flagged_nonfinite+unflagged_nonfinite; flag/nonfinite marginal counts overlap",
            "uq_partition" => "per component: primary_valid=uq_available+uq_nonfinite+uq_negative; uq_available=sigma_zero+sigma_positive",
            "coverage" => "component abs(measured-truth)<=k*sigma, k=1,2; denominator all finite nonnegative component sigma on primary_valid nodes, including zero",
            "normalized_error" => "(measured-truth)/sigma; denominator positive finite component sigma; zero sigma excluded explicitly, without floor",
            "overflow" => "measurement and UQ denominators retain arithmetic-overflow nodes; affected moment/coverage/quantile is unavailable, not silently reduced; finite-error division overflow proves uncovered",
            "quantiles" => "per-seed normalized-error quantiles only; never averaged into pooled quantiles",
            "statistics" => "bias and RMS include systematic error; overlapping vectors are correlated; no Gaussian coverage acceptance gates or vector-based binomial intervals"),
        "unsupported" => Dict("known_motion_experiment" => "no supplied recording and independent motion reference",
            "real_A_4E_coverage" => "committed fixtures have no supplied displacement truth; see separate smoke scorecard",
            "independent_datasets" => "not supplied beyond committed A/4E", "stereo_coverage" => "not evaluated",
            "calibration_timing_budget" => "not evaluated", "peak_host_device_memory" => "not measured",
            "GPU_hardware" => "not evaluated", "spatial_resolution_transfer" => "window sensitivity on linear shear does not measure resolution transfer"),
        "limitations" => ["Synthetic conditions do not establish experimental accuracy or universally calibrated uncertainty.",
            "Repeated windows do not prove convergence; actual primary residual summaries are reported without per-node certification.",
            "Random correlation uncertainty omits systematic bias; full truth error is retained in coverage.",
            "Finite sigma above 0.3 px is counted and retained with a linearization caution.",
            "Seeded scene replicates describe condition variability; warmed timing repeats reuse the same images and are not accuracy replicates.",
            "Only one seed field/node-error workspace is needed at a time; report metadata grows with condition/seed count, not vector payloads."])
end

function markdown_report(report)
    io = IOBuffer()
    println(io, "<!-- $MARKER -->\n# Synthetic uncertainty scorecard\n")
    println(io, "Full recipes, per-seed metrics/quantiles, diagnostics and provenance: `uncertainty.toml`.\n")
    println(io, "Provenance: ", report["provenance_status"], ".\n")
    println(io, "| Condition | Primary valid/unmasked | Component | UQ subset count | Full error bias/RMS | UQ subset error bias/RMS | Coverage 1σ / 2σ | Normalized mean/RMS |")
    println(io, "|---|---:|---|---:|---:|---:|---:|---:|")
    number(x) = string(round(x; sigdigits = 5))
    moment(m) = m["available"] ? number(m["mean"]) * " / " * number(m["rms"]) : "unavailable"
    coverage(m) = m["available"] ? number(m["fraction"]) * " ($(m["covered_count"])/$(m["denominator"]))" : "unavailable ($(m["denominator"]))"
    for group in report["groups"], key in ("u", "v")
        p = group["pooled"]; c = p["components"][key]
        println(io, "| $(group["condition"]) | $(p["counts"]["primary_valid"])/$(p["counts"]["unmasked"]) | $key | $(c["counts"]["uq_available"]) | $(moment(c["primary_error"])) | $(moment(c["uq_subset_error"])) | $(coverage(c["coverage"]["1"])) / $(coverage(c["coverage"]["2"])) | $(moment(c["normalized_error"])) |")
    end
    empty = report["empty_failure"]
    println(io, "\nEmpty scene: ", empty["status"], " (", empty["metrics"]["counts"]["primary_valid"],
        "/", empty["metrics"]["counts"]["unmasked"], "). Unavailable errors are not passed accuracy.\n")
    println(io, "Component populations can differ. Coverage is observed synthetic coverage, without a Gaussian pass/fail target. Pooled quantiles are unavailable. Julia allocations are **not peak memory**.\n")
    for limitation in report["limitations"]
        println(io, "- ", limitation)
    end
    for (category, reason) in sort!(collect(report["unsupported"]); by = first)
        println(io, "- Unavailable `", category, "`: ", reason, ".")
    end
    String(take!(io))
end

function resolved_path(path)
    absolute = normpath(abspath(path))
    tail = String[]
    while !ispath(absolute)
        islink(absolute) && throw(ArgumentError("dangling symlink in output path"))
        push!(tail, basename(absolute))
        parent = dirname(absolute)
        parent == absolute && throw(ArgumentError("cannot resolve output directory"))
        absolute = parent
    end
    joinpath(realpath(absolute), reverse(tail)...)
end
function within(path, parent)
    p, base = splitpath(normpath(path)), splitpath(normpath(parent))
    Sys.iswindows() && ((p, base) = (lowercase.(p), lowercase.(base)))
    length(p) >= length(base) && p[1:length(base)] == base
end
function check_output(directory)
    target = resolved_path(directory)
    repository = realpath(ROOT)
    allowed = resolved_path(joinpath(ROOT, "bench", "profile-output"))
    within(target, repository) && !within(target, allowed) &&
        throw(ArgumentError("repository output must stay under bench/profile-output; source/fixture files are protected"))
    protected = String[@__FILE__, joinpath(@__DIR__, "validation_scorecard.jl")]
    for root in ("src", "ext", joinpath("test", "reference_images"))
        for (dir, _, files) in walkdir(joinpath(ROOT, root))
            append!(protected, joinpath.(dir, files))
        end
    end
    paths = [joinpath(target, "uncertainty.toml"), joinpath(target, "uncertainty.md")]
    for path in paths
        islink(path) && !ispath(path) && throw(ArgumentError("dangling report symlink is not an output file"))
        if ispath(path)
            isfile(path) || throw(ArgumentError("output path is not a regular file: $path"))
            resolved = realpath(path)
            within(resolved, repository) && !within(resolved, allowed) &&
                throw(ArgumentError("report path aliases protected repository content"))
            any(p -> isfile(p) && Base.samefile(path, p), protected) &&
                throw(ArgumentError("report aliases source or fixture file"))
            occursin(MARKER, open(readline, path)) || throw(ArgumentError("refusing to overwrite an unrelated file: $path"))
        end
    end
    paths
end
function write_report(directory, report)
    paths = check_output(directory)
    toml = IOBuffer(); println(toml, "# $MARKER"); TOML.print(toml, report; sorted = true)
    text = String(take!(toml)); markdown = markdown_report(report)
    mkpath(dirname(first(paths)))
    write(paths[1], text); write(paths[2], markdown)
    paths
end
function main(args = ARGS)
    expanded, samples = false, 1
    output = joinpath(ROOT, "bench", "profile-output", "validation-uncertainty")
    for arg in args
        if arg == "--expanded"
            expanded = true
        elseif startswith(arg, "--samples=")
            samples = parse(Int, split(arg, '='; limit = 2)[2])
        elseif startswith(arg, "--output=")
            output = split(arg, '='; limit = 2)[2]
        elseif arg == "--help"
            println("julia --project=. -t 4 bench/validation_uncertainty.jl [--expanded] [--samples=1] [--output=directory]")
            return nothing
        else
            throw(ArgumentError("unknown uncertainty scorecard option: $arg"))
        end
    end
    1 <= samples <= 10 || throw(ArgumentError("samples must be between 1 and 10"))
    check_output(output) # reject aliases before expensive numerical work
    report = run_scorecard(; expanded, samples)
    paths = write_report(output, report)
    println(markdown_report(report))
    println("Report files: ", join(paths, ", "))
    report
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    ValidationUncertainty.main()
end
