# Portable CPU/device/hybrid configuration benchmark.
#
#   julia --project=<gpuenv> -t auto bench/gpu_configurations.jl \
#       <cuda|amdgpu|ka> [size=2048] [Float32|Float64=Float32] \
#       [low|medium|high=high] [samples=3]
#
# The public `benchmark_piv_configurations` helper accepts real images and
# explicit schedules; this CLI supplies a reproducible synthetic convenience
# workload so a new machine can be characterized immediately.

using Hammerhead
using Hammerhead.SyntheticData
using Random

backend = Symbol(get(ARGS, 1, "ka"))
if backend === :cuda
    @eval using CUDA
    CUDA.functional() || error("CUDA is not functional")
elseif backend === :amdgpu
    @eval using AMDGPU
    AMDGPU.functional() || error("AMDGPU is not functional")
elseif backend !== :ka
    error("backend must be cuda, amdgpu, or ka")
end

n = parse(Int, get(ARGS, 2, "2048"))
T = let name = get(ARGS, 3, "Float32")
    name == "Float32" ? Float32 : name == "Float64" ? Float64 :
        error("precision must be Float32 or Float64")
end
effort = Symbol(get(ARGS, 4, "high"))
samples = parse(Int, get(ARGS, 5, "3"))

println("Generating $(n)x$n $T synthetic pair")
rng = MersenneTwister(20260713 + n)
flow = vortex_flow(n / 2, n / 2, 4.0)
A64, B64, _, _ = generate_synthetic_piv_pair(flow, (n, n), 1.0;
    particle_density = 0.006, background_noise = 0.003, rng)
A, B = T.(A64), T.(B64)
A64 = B64 = nothing
GC.gc()

rows = benchmark_piv_configurations(A, B; effort,
                                    backends = (:cpu, backend), samples)
println("configuration  backend  UQ       seconds   speedup   max delta uv   max delta UQ")
for r in rows
    println(rpad(string(r.configuration), 15),
            rpad(string(r.backend), 9),
            rpad(string(r.uncertainty_backend), 9),
            lpad(round(r.seconds, digits = 3), 9),
            lpad(round(r.speedup, digits = 2), 10),
            lpad(string(round(r.max_vector_delta, sigdigits = 3)), 15),
            lpad(string(round(r.max_uncertainty_delta, sigdigits = 3)), 15))
end
best = rows[argmin(getfield.(rows, :seconds))]
println("Fastest: $(best.configuration) (backend=:$(best.backend), " *
        "uncertainty_backend=:$(best.uncertainty_backend))")
