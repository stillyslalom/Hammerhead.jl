# Hammerhead.jl performance benchmarks.
#
# Run from the package root:
#   julia --project=. --threads=auto bench/run_benchmarks.jl
#
# Uses only package dependencies (no BenchmarkTools); timings are the minimum
# over several samples, which is robust against GC and scheduling noise.

using Hammerhead
using Hammerhead.SyntheticData
using Random
using Statistics

"Minimum elapsed time of `f()` over `samples` runs, optionally after a warmup call."
function time_min(f; samples::Int = 5, warmup::Bool = true)
    warmup && f()
    return minimum(@elapsed(f()) for _ in 1:samples)
end

format_time(t) = t < 1e-3 ? "$(round(t * 1e6, digits = 1)) μs" :
                 t < 1.0  ? "$(round(t * 1e3, digits = 2)) ms" :
                            "$(round(t, digits = 3)) s"

function parse_int_list(name, default)
    raw = get(ENV, name, default)
    vals = Int[]
    for part in split(raw, ',')
        s = strip(part)
        isempty(s) && continue
        push!(vals, parse(Int, s))
    end
    return vals
end

function high_effort_pair(n, rng)
    flow = vortex_flow(n / 2, n / 2, 4.0)
    generate_synthetic_piv_pair(flow, (n, n), 1.0;
                                particle_density = 0.006,
                                background_noise = 0.003,
                                rng)
end

function time_final_pass_buckets(imgA, imgB; threaded::Bool)
    T = float(promote_type(eltype(imgA), eltype(imgB)))
    predictor_result = run_piv(imgA, imgB,
                               multipass_parameters([128, 64];
                                   padding = true, apodization = :gauss,
                                   final = (max_iterations = 2,));
                               threaded)
    predictor = Hammerhead.build_predictor(predictor_result, true)
    params = PIVParameters(window_size = 32, overlap = 16,
                           padding = true, apodization = :gauss,
                           uncertainty = true, max_iterations = 2)
    grid = Hammerhead.pass_grid(T, size(imgA), params, nothing, 0.5)
    itpA = Hammerhead.image_interpolant(imgA, T)
    itpB = Hammerhead.image_interpolant(imgB, T)
    warpA = Matrix{T}(undef, size(imgA))
    warpB = Matrix{T}(undef, size(imgB))
    corr = Hammerhead.make_correlator(params, T)
    ny, nx = length(grid.y), length(grid.x)
    u = zeros(T, ny, nx)
    v = zeros(T, ny, nx)
    peak_ratio = zeros(T, ny, nx)
    moment = zeros(T, ny, nx)
    uu = fill(T(NaN), ny, nx)
    uv = fill(T(NaN), ny, nx)
    outliers = falses(ny, nx)

    deformed = Ref{Any}()
    deformation = time_min(samples = 1, warmup = false) do
        deformed[] = Hammerhead.apply_predictor(imgA, imgB, itpA, itpB, predictor,
                                                grid.x, grid.y, T; threaded,
                                                warpA, warpB)
    end
    warpedA, warpedB = deformed[][1], deformed[][2]

    correlation = time_min(samples = 1, warmup = false) do
        fill!(u, zero(T)); fill!(v, zero(T))
        fill!(peak_ratio, zero(T)); fill!(moment, zero(T))
        Hammerhead.process_windows!(u, v, peak_ratio, moment, nothing, nothing,
                                    nothing, nothing, grid.jobs, warpedA, warpedB,
                                    params, corr, nothing, nothing)
    end

    uq = time_min(samples = 1, warmup = false) do
        fill!(uu, T(NaN)); fill!(uv, T(NaN))
        Hammerhead.uncertainty_sweep!(uu, uv, grid.jobs, warpedA, warpedB,
                                      params, corr.apod, nothing)
    end

    result_sink = Ref{Any}()
    result_time = time_min(samples = 3, warmup = false) do
        result_sink[] = PIVResult(copy(grid.x), copy(grid.y), copy(u), copy(v),
                                  copy(peak_ratio), copy(moment), copy(uu), copy(uv),
                                  copy(outliers), copy(grid.grid_mask), params,
                                  nothing)
    end
    result = PIVResult(grid.x, grid.y, copy(u), copy(v), copy(peak_ratio),
                       copy(moment), copy(uu), copy(uv), copy(outliers),
                       copy(grid.grid_mask), params, nothing)
    validation = time_min(samples = 3, warmup = false) do
        r = PIVResult(result.x, result.y, copy(result.u), copy(result.v),
                      copy(result.peak_ratio), copy(result.correlation_moment),
                      copy(result.uncertainty_u), copy(result.uncertainty_v),
                      falses(size(result.outliers)), result.mask, params, nothing)
        Hammerhead.validate_and_replace!(r, params, false)
    end
    return [
        "loading/preprocess" => NaN,
        "image deformation" => deformation,
        "window correlation" => correlation,
        "validation/replacement" => validation,
        "UQ" => uq,
        "result assembly" => result_time,
        "host/device transfer" => NaN,
    ]
end

function benchmark_camera(; f = 3500.0, cx = 256.0, cy = 256.0,
                          yaw_deg = 0.0, dist = 500.0)
    th = deg2rad(yaw_deg)
    R = [cos(th) 0.0 -sin(th); 0.0 1.0 0.0; sin(th) 0.0 cos(th)]
    C = R' * [0.0, 0.0, -dist]
    K = [f 0.0 cx; 0.0 -f cy; 0.0 0.0 1.0]
    return PinholeCamera(K, R, -R * C)
end

function stereo_benchmark_frames(cams, pts, displacement;
                                 image_size = (512, 512), diameter = 6.0)
    dx, dy, dz = displacement
    frames = map(cams) do cam
        A = zeros(image_size)
        B = zeros(image_size)
        for (X, Y) in pts
            pa = world_to_pixel(cam, (X, Y, 0.0))
            pb = world_to_pixel(cam, (X + dx, Y + dy, dz))
            generate_gaussian_particle!(A, (pa[1], pa[2]), diameter)
            generate_gaussian_particle!(B, (pb[1], pb[2]), diameter)
        end
        (A, B)
    end
    return frames[1][1], frames[1][2], frames[2][1], frames[2][2]
end

rng = MersenneTwister(2026)
quick = get(ENV, "HAMMERHEAD_BENCH_QUICK", "0") == "1"
production_sizes = parse_int_list("HAMMERHEAD_BENCH_SIZES",
                                  quick ? "1024" : "1024,2048,4096")
ensemble_counts = parse_int_list("HAMMERHEAD_BENCH_ENSEMBLES",
                                 quick ? "10" : "10,100,1000")
production_samples = parse(Int, get(ENV, "HAMMERHEAD_BENCH_SAMPLES", "1"))

println("Hammerhead.jl benchmarks — Julia $VERSION, $(Threads.nthreads()) thread(s)")

# --- Correlator throughput -------------------------------------------------

println("\nCorrelation (per window pair, including subpixel refinement):")
for ws in (16, 32, 64)
    a = rand(rng, ws, ws)
    b = rand(rng, ws, ws)
    n = max(1, 10_000 ÷ ws)
    for (label, correlator) in (
        "cross        " => CrossCorrelator{Float64}((ws, ws)),
        "cross padded " => CrossCorrelator{Float64}((ws, ws); padding = true, apodization = :gauss),
        "phase        " => PhaseCorrelator{Float64}((ws, ws)),
    )
        t = time_min(() -> (for _ in 1:n; correlate(correlator, a, b); end)) / n
        println("  $(lpad(ws, 3))×$(rpad(ws, 3)) $label $(format_time(t))")
    end
end

# --- Full pipeline ----------------------------------------------------------

println("\nFull run_piv (512×512 synthetic vortex pair):")
flow = vortex_flow(256.0, 256.0, 4.0)
imgA, imgB, _, _ = generate_synthetic_piv_pair(flow, (512, 512), 1.0;
                                               particle_density = 0.03, rng = rng)

for (label, passes) in (
    "single pass 32/16     " => [PIVParameters(window_size = 32, overlap = 16)],
    "single pass 32/16 +pad" => [PIVParameters(window_size = 32, overlap = 16,
                                               padding = true, apodization = :gauss)],
    "multipass 64→32→16    " => multipass_parameters([64, 32, 16]),
    "  + final iterated ≤3 " => multipass_parameters([64, 32, 16];
                                                     final = (max_iterations = 3,)),
)
    serial = time_min(() -> run_piv(imgA, imgB, passes; threaded = false); samples = 3)
    if Threads.nthreads() > 1
        threaded = time_min(() -> run_piv(imgA, imgB, passes; threaded = true); samples = 3)
        speedup = round(serial / threaded, digits = 2)
        println("  $label serial $(format_time(serial)), threaded $(format_time(threaded)) ($(speedup)×)")
    else
        println("  $label serial $(format_time(serial))")
    end
end

# --- Effort presets ---------------------------------------------------------

function midpoint_truth(flow, x, y)
    x0, y0 = Float64(x), Float64(y)
    for _ in 1:8
        u, v, _ = flow(x0, y0, 0.0, 0.0)
        x0 = Float64(x) - 0.5u
        y0 = Float64(y) - 0.5v
    end
    return flow(x0, y0, 0.0, 0.0)
end

function rms_error(result, flow; midpoint::Bool = false)
    err2 = Float64[]
    for j in eachindex(result.x), i in eachindex(result.y)
        (result.mask[i, j] || result.outliers[i, j]) && continue
        uref, vref, _ = midpoint ? midpoint_truth(flow, result.x[j], result.y[i]) :
                                   flow(result.x[j], result.y[i], 0.0, 0.0)
        push!(err2, abs2(Float64(result.u[i, j]) - uref) +
                    abs2(Float64(result.v[i, j]) - vref))
    end
    return sqrt(mean(err2))
end

println("\nEffort presets (256×256 synthetic pairs, serial; high includes UQ):")
for (scene, flow, midpoint) in (
    ("uniform shift", linear_flow(2.4, -1.7, 0.0, 0.0, 0.0, 0.0, 0.0), false),
    ("vortex       ", vortex_flow(128.0, 128.0, 4.0), true),
)
    rng_scene = MersenneTwister(20260710)
    effort_imgA, effort_imgB, _, _ =
        generate_synthetic_piv_pair(flow, (256, 256), 1.0;
                                    particle_density = 0.045,
                                    background_noise = 0.005,
                                    rng = rng_scene)
    println("  $scene")
    base = nothing
    for level in (:low, :medium, :high)
        t = time_min(() -> run_piv(effort_imgA, effort_imgB; effort = level, threaded = false);
                     samples = 2)
        r = run_piv(effort_imgA, effort_imgB; effort = level, threaded = false)
        base === nothing && (base = t)
        multiple = round(t / base, digits = 2)
        rms = round(rms_error(r, flow; midpoint), digits = 4)
        println("    $(rpad(string(level), 7)) $(format_time(t)) ($(multiple)×), RMS $(rms) px")
    end
end

# --- High-effort production workloads --------------------------------------

println("\nHigh-effort production workloads:")
println("  sizes: $(join(production_sizes, ", ")) px; samples/run: $production_samples")
for n in production_sizes
    local_rng = MersenneTwister(2026 + n)
    imgP, imgQ, _, _ = high_effort_pair(n, local_rng)
    println("  $(n)x$n synthetic vortex, effort = :high")
    for maxiter in 2:5
        t = time_min(samples = production_samples, warmup = false) do
            run_piv(imgP, imgQ; effort = :high, max_iterations = maxiter,
                    threaded = Threads.nthreads() > 1)
        end
        println("    final max_iterations=$maxiter  $(format_time(t))")
    end
end

bucket_n = first(production_sizes)
bucket_rng = MersenneTwister(90210 + bucket_n)
bucketA, bucketB, _, _ = high_effort_pair(bucket_n, bucket_rng)
println("\nHigh-effort final-pass timing buckets ($(bucket_n)x$bucket_n):")
for (label, t) in time_final_pass_buckets(bucketA, bucketB;
                                          threaded = Threads.nthreads() > 1)
    value = isnan(t) ? "n/a on CPU" : format_time(t)
    println("  $(rpad(label, 24)) $value")
end

ens_n = first(production_sizes)
ens_rng = MersenneTwister(4242 + ens_n)
ensA, ensB, _, _ = high_effort_pair(ens_n, ens_rng)
println("\nHigh-effort ensemble correlation ($(ens_n)x$ens_n, repeated pair refs):")
for count in ensemble_counts
    pairs = fill((ensA, ensB), count)
    t = time_min(samples = production_samples, warmup = false) do
        run_piv_ensemble(pairs; effort = :high, progress = false,
                         threaded = Threads.nthreads() > 1)
    end
    println("  $(lpad(count, 4)) pairs  $(format_time(t))")
end

println("\nStereo high-effort path (dewarp + per-camera PIV):")
cams = (benchmark_camera(yaw_deg = -20.0), benchmark_camera(yaw_deg = 20.0))
grid = DewarpGrid(x = -25.0:0.2:25.0, y = -25.0:0.2:25.0)
dws = map(cam -> ImageDewarper(cam, grid, (512, 512)), cams)
stereo_rng = MersenneTwister(5150)
pts = [(56 * rand(stereo_rng) - 28, 56 * rand(stereo_rng) - 28) for _ in 1:350]
A1, B1, A2, B2 = stereo_benchmark_frames(cams, pts, (0.5, -0.3, 0.4))
dewarp_t = time_min(samples = production_samples, warmup = false) do
    dewarp(dws[1], A1); dewarp(dws[1], B1)
    dewarp(dws[2], A2); dewarp(dws[2], B2)
end
stereo_t = time_min(samples = production_samples, warmup = false) do
    run_piv_stereo(A1, B1, A2, B2, dws[1], dws[2];
                   effort = :high, threaded = Threads.nthreads() > 1)
end
println("  dewarp only        $(format_time(dewarp_t))")
println("  full stereo high   $(format_time(stereo_t))")

# --- Validation -------------------------------------------------------------

println("\nValidation (128×128 vector field):")
u = randn(rng, 128, 128)
v = randn(rng, 128, 128)
result = PIVResult(collect(1.0:128), collect(1.0:128), u, v,
                   fill(2.0, 128, 128), fill(0.2, 128, 128),
                   fill(NaN, 128, 128), fill(NaN, 128, 128),
                   falses(128, 128), falses(128, 128), PIVParameters())
for (label, validator) in (
    "peak ratio        " => PeakRatioValidator(1.2),
    "velocity magnitude" => VelocityMagnitudeValidator(0.0, 5.0),
    "UOD 5×5           " => UniversalOutlierValidator(2.0),
)
    t = time_min(() -> (fill!(result.outliers, false); apply_validator!(result, validator)))
    println("  $label $(format_time(t))")
end
