# Per-kernel profile of a UQ-enabled multipass run, to confirm whether the
# device UQ statistics kernel (`_ka_uq_stats!`) is genuinely the top
# optimization opportunity before restructuring it. Correlation FFTs run every
# pass; UQ runs on the final pass only, so a whole-pipeline profile is the
# right lens.
#
#     julia --project=bench/CUDA -t 4 bench/gpu_profile_uq.jl cuda
#
# Uses CUDA.@profile (device-side kernel timing). AMDGPU has no equivalent
# turnkey profiler here; on that backend fall back to rocprof externally.

using Hammerhead
using Hammerhead.SyntheticData
using Random, Statistics, Printf

backend = Symbol(get(ARGS, 1, "cuda"))
backend === :cuda || error("this profiler is CUDA-only (CUDA.@profile); got :$backend")
using CUDA
CUDA.functional() || error("CUDA is not functional — check the NVIDIA driver install")

println("backend: :", backend, "   threads: ", Threads.nthreads())
Hammerhead._resolve_backend(backend)

function make_scene(n::Int, ::Type{T}) where {T}
    rng = MersenneTwister(2026 + n)
    flow = vortex_flow(n / 2, n / 2, 4.0)
    A64, B64, _, _ = generate_synthetic_piv_pair(flow, (n, n), 1.0;
        particle_density = 0.006, background_noise = 0.003, rng)
    return T.(A64), T.(B64)
end

# Converged multipass with UQ on the final pass — the sanctioned way to get
# meaningful uncertainty (peak at ~zero residual).
uq_sched(; final_iters = 2) = Hammerhead.multipass_parameters([64, 32];
    padding = true, apodization = :gauss,
    final = (max_iterations = final_iters, uncertainty = true))

for n in (1024, 2048), T in (Float64, Float32)
    A, B = make_scene(n, T)
    sched = uq_sched()
    r = run_piv(A, B, sched; backend, threaded = false)   # warmup / compile
    nwin = length(r.u)
    nfinite = count(isfinite, r.uncertainty_u)
    @printf("\n===== %s  %d²  (%d windows, %d finite σ) =====\n",
            string(T), n, nwin, nfinite)
    prof = CUDA.@profile run_piv(A, B, sched; backend, threaded = false)
    show(stdout, MIME("text/plain"), prof)
    println()
end
