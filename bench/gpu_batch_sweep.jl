# Sweep the device sub-batch size and report wall-time + free-VRAM low-water
# mark, so the memory-aware cap in the CUDA/AMDGPU extensions can be sanity
# checked and the occupancy-optimal batch tuned per GPU.
#
#     julia --project=bench/CUDA -t auto bench/gpu_batch_sweep.jl \
#         <cuda|amdgpu> [size=4096] [Float32|Float64=Float64] \
#         [batches=8192,4096,2048,1024] [effort=high] [samples=3]
#
# Background: the extensions size the batch to `_CUDA_MEM_FRACTION` of free VRAM
# (computed exactly from each pass's fft_size + precision — cuFFT workspace is
# ~0 for our power-of-two transforms), which eliminates the VRAM->shared-memory
# spill cliff (measured RTX 2000 Ada 4096^2 Float64 effort=:high: fixed bs=8192
# spilled to 33 s; the memory cap runs ~10.5 s). A byte-budget alone cannot see
# the *occupancy* plateau, though — the point past which a bigger batch stops
# helping (that machine's Float64 optimum was ~1024, ~7.5 s). Use this sweep to
# find that per-GPU optimum; set `HAMMERHEAD_CUDA_BATCH` (or the AMDGPU
# equivalent) to pin a chosen value.

using Hammerhead
using Hammerhead.SyntheticData
using Random

backend = Symbol(get(ARGS, 1, "cuda"))
if backend === :cuda
    @eval using CUDA
    CUDA.functional() || error("CUDA is not functional")
    free_memory = CUDA.free_memory
    total_memory = CUDA.total_memory
    device_name = () -> CUDA.name(CUDA.device())
    batch_env = "HAMMERHEAD_CUDA_BATCH"
elseif backend === :amdgpu
    @eval using AMDGPU
    AMDGPU.functional() || error("AMDGPU is not functional")
    free_memory = AMDGPU.free_memory
    total_memory = AMDGPU.total_memory
    device_name = () -> string(AMDGPU.device())
    batch_env = "HAMMERHEAD_AMDGPU_BATCH"
else
    error("backend must be cuda or amdgpu")
end

n = parse(Int, get(ARGS, 2, "4096"))
T = get(ARGS, 3, "Float64") == "Float32" ? Float32 : Float64
batches = parse.(Int, split(get(ARGS, 4, "8192,4096,2048,1024"), ","))
effort = Symbol(get(ARGS, 5, "high"))
samples = parse(Int, get(ARGS, 6, "3"))

println("device: ", device_name(), "  total ",
        round(total_memory() / 2^30, digits = 2), " GiB")
println("workload: ", n, "x", n, " ", T, "  effort=:", effort,
        "  backend=:", backend)

rng = MersenneTwister(20260713 + n)
flow = vortex_flow(n / 2, n / 2, 4.0)
A64, B64, _, _ = generate_synthetic_piv_pair(flow, (n, n), 1.0;
    particle_density = 0.006, background_noise = 0.003, rng)
A, B = T.(A64), T.(B64)
A64 = B64 = nothing
GC.gc()
passes = Hammerhead.effort_schedule(effort; image_size = size(A))

tcpu = Inf
for _ in 1:2
    global tcpu
    t0 = time_ns()
    run_piv(A, B, passes; backend = :cpu)
    tcpu = min(tcpu, (time_ns() - t0) / 1e9)
end
println("cpu baseline: ", round(tcpu, digits = 3), " s")

println("batch    seconds   speedup_vs_cpu   min_free_GiB")
for bs in batches
    ENV[batch_env] = string(bs)
    ws = piv_workspace(; backend)
    run_piv(A, B, passes; backend, workspace = ws)   # warmup + allocate
    minfree = free_memory()
    best = Inf
    for _ in 1:samples
        GC.gc()
        t0 = time_ns()
        run_piv(A, B, passes; backend, workspace = ws)
        best = min(best, (time_ns() - t0) / 1e9)
        minfree = min(minfree, free_memory())
    end
    println(rpad(bs, 9), lpad(round(best, digits = 3), 9),
            lpad(round(tcpu / best, digits = 2), 16),
            lpad(round(minfree / 2^30, digits = 3), 15))
end
delete!(ENV, batch_env)
println("(unset $batch_env -> memory-aware auto cap)")
