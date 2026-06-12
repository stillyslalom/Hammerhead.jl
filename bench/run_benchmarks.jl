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

"Minimum elapsed time of `f()` over `samples` runs, after a warmup call."
function time_min(f; samples::Int = 5)
    f()
    return minimum(@elapsed(f()) for _ in 1:samples)
end

format_time(t) = t < 1e-3 ? "$(round(t * 1e6, digits = 1)) μs" :
                 t < 1.0  ? "$(round(t * 1e3, digits = 2)) ms" :
                            "$(round(t, digits = 3)) s"

rng = MersenneTwister(2026)

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

# --- Validation -------------------------------------------------------------

println("\nValidation (128×128 vector field):")
u = randn(rng, 128, 128)
v = randn(rng, 128, 128)
result = PIVResult(collect(1.0:128), collect(1.0:128), u, v,
                   fill(2.0, 128, 128), fill(0.2, 128, 128), falses(128, 128),
                   PIVParameters())
for (label, validator) in (
    "peak ratio        " => PeakRatioValidator(1.2),
    "velocity magnitude" => VelocityMagnitudeValidator(0.0, 5.0),
    "UOD 5×5           " => UniversalOutlierValidator(2.0),
)
    t = time_min(() -> (fill!(result.outliers, false); apply_validator!(result, validator)))
    println("  $label $(format_time(t))")
end
