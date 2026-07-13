# GPU-backend correctness validation against `backend = :cpu`: single-pass,
# multipass (alt-peaks path), masked windows, and ensemble correlation
# (device-resident plane accumulator) on a synthetic uniform-shift scene.
# Expect max deviations at FFT round-off level (~1e-14 for Float64).
#
# Same environment requirements as gpu_benchmarks.jl (see its header):
#
#     julia --project=<gpuenv> -t 4 bench/gpu_validate.jl cuda

using Hammerhead
using Random, Statistics

backend = Symbol(get(ARGS, 1, "amdgpu"))
if backend === :amdgpu
    using AMDGPU
    AMDGPU.functional() || error("AMDGPU is not functional — check the ROCm install")
elseif backend === :cuda
    using CUDA
    CUDA.functional() || error("CUDA is not functional — check the NVIDIA driver install")
end

println("backend: :", backend, "   resolves to ", Hammerhead._resolve_backend(backend))

function add_particle!(a, c, d)
    s = d / 2
    for j in axes(a, 2), i in axes(a, 1)
        r2 = (i - c[1])^2 + (j - c[2])^2
        r2 > (5s)^2 && continue
        a[i, j] += exp(-0.5 * r2 / s^2)
    end
    a
end

rng = MersenneTwister(20260711)
imgA = zeros(256, 256); imgB = zeros(256, 256)
dv, du = 2.0, -1.25
for _ in 1:1500
    p = (rand(rng) * 276 - 10, rand(rng) * 276 - 10)
    add_particle!(imgA, p, 3.0)
    add_particle!(imgB, (p[1] + dv, p[2] + du), 3.0)
end
params = PIVParameters(window_size = 32, overlap = 16, padding = true, apodization = :gauss)

r_cpu = run_piv(imgA, imgB, params; threaded = false)
r_gpu = run_piv(imgA, imgB, params; backend, threaded = false)
valid = .!r_cpu.outliers .& .!r_cpu.mask
println("single-pass max |Δu| = ", maximum(abs.(r_gpu.u[valid] .- r_cpu.u[valid])),
        "  max |Δv| = ", maximum(abs.(r_gpu.v[valid] .- r_cpu.v[valid])))
println("gpu median (u,v) = (", median(r_gpu.u[valid]), ", ", median(r_gpu.v[valid]),
        ")  truth = (", du, ", ", dv, ")")

sched = Hammerhead.multipass_parameters([64, 32]; padding = true, apodization = :gauss)
m_cpu = run_piv(imgA, imgB, sched; threaded = false)
m_gpu = run_piv(imgA, imgB, sched; backend, threaded = false)
mvalid = .!m_cpu.outliers .& .!m_cpu.mask
println("multipass max |Δu| = ", maximum(abs.(m_gpu.u[mvalid] .- m_cpu.u[mvalid])),
        "  max |Δv| = ", maximum(abs.(m_gpu.v[mvalid] .- m_cpu.v[mvalid])))

mask = falses(size(imgA)); mask[100:160, 100:160] .= true
k_cpu = run_piv(imgA, imgB, params; mask, threaded = false)
k_gpu = run_piv(imgA, imgB, params; backend, mask, threaded = false)
kvalid = .!k_cpu.outliers .& .!k_cpu.mask
println("masked max |Δu| = ", maximum(abs.(k_gpu.u[kvalid] .- k_cpu.u[kvalid])),
        "  masked cells NaN: ", all(isnan, k_gpu.u[k_gpu.mask]))

# Ensemble: same flow, second pair with an independent particle set.
imgA2 = zeros(256, 256); imgB2 = zeros(256, 256)
for _ in 1:1500
    p = (rand(rng) * 276 - 10, rand(rng) * 276 - 10)
    add_particle!(imgA2, p, 3.0)
    add_particle!(imgB2, (p[1] + dv, p[2] + du), 3.0)
end
pairs = [(imgA, imgB), (imgA2, imgB2)]
e_cpu = run_piv_ensemble(pairs, params; progress = false, threaded = false)
e_gpu = run_piv_ensemble(pairs, params; backend, progress = false, threaded = false)
evalid = .!e_cpu.outliers .& .!e_cpu.mask
println("ensemble max |Δu| = ", maximum(abs.(e_gpu.u[evalid] .- e_cpu.u[evalid])),
        "  max |Δv| = ", maximum(abs.(e_gpu.v[evalid] .- e_cpu.v[evalid])))
me_cpu = run_piv_ensemble(pairs, sched; progress = false, threaded = false)
me_gpu = run_piv_ensemble(pairs, sched; backend, progress = false, threaded = false)
mevalid = .!me_cpu.outliers .& .!me_cpu.mask
println("ensemble multipass max |Δu| = ",
        maximum(abs.(me_gpu.u[mevalid] .- me_cpu.u[mevalid])),
        "  max |Δv| = ", maximum(abs.(me_gpu.v[mevalid] .- me_cpu.v[mevalid])))
# Phase 4b: Float64 UQ statistics on the device, including the post-iteration
# sweep and the device-resident ensemble accumulator.
uqparams = PIVParameters(window_size = 32, overlap = 16, padding = true,
                         apodization = :gauss, uncertainty = true,
                         max_iterations = 2)
uq_cpu = run_piv(imgA, imgB, uqparams; threaded = false)
uq_gpu = run_piv(imgA, imgB, uqparams; backend, threaded = false)
uvalid = isfinite.(uq_cpu.uncertainty_u) .& isfinite.(uq_gpu.uncertainty_u)
vvalid = isfinite.(uq_cpu.uncertainty_v) .& isfinite.(uq_gpu.uncertainty_v)
duq = maximum(abs.(uq_gpu.uncertainty_u[uvalid] .- uq_cpu.uncertainty_u[uvalid]))
dvq = maximum(abs.(uq_gpu.uncertainty_v[vvalid] .- uq_cpu.uncertainty_v[vvalid]))
println("UQ max abs error u = ", duq, "  v = ", dvq)
@assert any(uvalid) && any(vvalid) && duq < 1e-8 && dvq < 1e-8

euq_cpu = run_piv_ensemble(pairs, uqparams; progress = false, threaded = false)
euq_gpu = run_piv_ensemble(pairs, uqparams; backend, progress = false,
                           threaded = false)
euvalid = isfinite.(euq_cpu.uncertainty_u) .& isfinite.(euq_gpu.uncertainty_u)
evvalid = isfinite.(euq_cpu.uncertainty_v) .& isfinite.(euq_gpu.uncertainty_v)
deuq = maximum(abs.(euq_gpu.uncertainty_u[euvalid] .-
                    euq_cpu.uncertainty_u[euvalid]))
devq = maximum(abs.(euq_gpu.uncertainty_v[evvalid] .-
                    euq_cpu.uncertainty_v[evvalid]))
println("ensemble UQ max abs error u = ", deuq, "  v = ", devq)
@assert any(euvalid) && any(evvalid) && deuq < 1e-8 && devq < 1e-8
println("DONE")
