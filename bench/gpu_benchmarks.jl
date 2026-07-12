# GPU-backend correctness check + benchmark vs. the CPU path.
#
# Needs an environment with Hammerhead (dev'd) plus the backend's trigger
# packages, e.g. for `:amdgpu`:
#
#     julia --project=<gpuenv> -t 4 bench/gpu_benchmarks.jl amdgpu
#
# where <gpuenv> has AMDGPU (or CUDA), KernelAbstractions, and AbstractFFTs
# installed. The backend selector defaults to :amdgpu; pass `cuda`, or `ka`
# for the hardware-free KernelAbstractions CPU tier, as ARGS[1].
#
# AMD note: on Windows, RDNA2 cards (e.g. RX 6800 XT / gfx1030) need ROCm 6.4 —
# ROCm/HIP 7.1 dropped RDNA2 — so point HIP_PATH/ROCM_PATH/PATH at the 6.4
# install before launching.

using Hammerhead
using Hammerhead.SyntheticData
using Random, Statistics, Printf

backend = Symbol(get(ARGS, 1, "amdgpu"))
if backend === :amdgpu
    using AMDGPU
    AMDGPU.functional() || error("AMDGPU is not functional — check the ROCm install")
elseif backend === :cuda
    using CUDA
    CUDA.functional() || error("CUDA is not functional — check the NVIDIA driver install")
end
using KernelAbstractions, AbstractFFTs

println("backend: :", backend, "   threads: ", Threads.nthreads())
Hammerhead._resolve_backend(backend)   # fails fast if the extension didn't load

function bench(backend, n::Int, ::Type{T}; samples = 4) where {T}
    rng = MersenneTwister(2026 + n)
    flow = vortex_flow(n / 2, n / 2, 4.0)
    A64, B64, _, _ = generate_synthetic_piv_pair(flow, (n, n), 1.0;
        particle_density = 0.006, background_noise = 0.003, rng)
    A = T.(A64); B = T.(B64)
    params = PIVParameters(window_size = 32, overlap = 16, padding = true, apodization = :gauss)

    r_cpu = run_piv(A, B, params; threaded = true)                # warmup + reference
    r_gpu = run_piv(A, B, params; backend, threaded = false)      # warmup
    valid = .!r_cpu.outliers .& .!r_cpu.mask
    dmax = max(maximum(abs.(r_gpu.u[valid] .- r_cpu.u[valid])),
               maximum(abs.(r_gpu.v[valid] .- r_cpu.v[valid])))

    tmin(f) = minimum(@elapsed(f()) for _ in 1:samples)
    tcpu = tmin(() -> run_piv(A, B, params; threaded = true))
    tgpu = tmin(() -> run_piv(A, B, params; backend, threaded = false))
    @printf("%-7s %4d²  %6d win   CPU %7.3f s   GPU %7.3f s   speedup %5.2fx   max|Δ| %.1e\n",
            string(T), n, length(r_cpu.u), tcpu, tgpu, tcpu / tgpu, dmax)
end

for n in (1024, 2048)
    bench(backend, n, Float64)
    bench(backend, n, Float32)
end
