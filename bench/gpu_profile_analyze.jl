# Non-UQ multipass profile — `_ka_analyze!` runs on every pass, so a plain
# multipass run (the common case) is where it weighs most. Companion to
# gpu_profile_uq.jl; not committed (scratch for the analyze-kernel work).
#
#     julia --project=bench/CUDA -t 4 bench/gpu_profile_analyze.jl cuda

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

# Plain converged multipass, NO uncertainty — analyze runs on all passes.
sched() = Hammerhead.multipass_parameters([64, 32];
    padding = true, apodization = :gauss,
    final = (max_iterations = 2,))

for n in (1024, 2048), T in (Float64, Float32)
    A, B = make_scene(n, T)
    s = sched()
    r = run_piv(A, B, s; backend, threaded = false)   # warmup / compile
    nwin = length(r.u)
    @printf("\n===== %s  %d²  (%d windows) =====\n", string(T), n, nwin)
    prof = CUDA.@profile run_piv(A, B, s; backend, threaded = false)
    show(stdout, MIME("text/plain"), prof)
    println()
end
