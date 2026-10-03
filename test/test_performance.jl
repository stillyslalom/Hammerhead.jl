using Hammerhead
using Test
using Random

# Guards against type instabilities in the per-window loops, which show up as
# allocations proportional to the number of pixels processed. A 256² three-pass
# run allocates about 80 MiB (mostly deformation buffers); an unstable inner
# loop pushes it well past the bound (about 2.5× at this size).
@testset "run_piv allocation budget" begin
    rng = MersenneTwister(1)
    n = 256
    A, B = particle_pair((n, n), [(rand(rng) * n, rand(rng) * n) for _ in 1:1500], 2.0, -1.0)
    passes = multipass_parameters([64, 32, 16])
    for T in (Float32, Float64), mask in (nothing, falses(n, n))
        a, b = T.(A), T.(B)
        ws = piv_workspace()
        run_piv(a, b, passes; workspace = ws, mask, threaded = false)
        bytes = @allocated run_piv(a, b, passes; workspace = ws, mask, threaded = false)
        @test bytes < 120 * 2^20
    end
end
