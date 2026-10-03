using Test
using LinearAlgebra
using Hammerhead

@testset "Manual affine registration validation" begin
    points = [(1., 2.), (5., 2.), (1., 7.), (5., 7.), (3., 4.)]
    matrix = [0.2 0.03; -0.04 -0.3]
    offset = [2., -1.]
    reference = [Tuple(matrix * collect(p) + offset) for p in points]
    fitted = calculate_manual_registration(points, reference)
    @test fitted.A ≈ matrix atol=1e-14
    @test fitted.b ≈ offset atol=1e-14
    @test calculate_manual_registration(collect.(points), collect.(reference)).A == fitted.A
    @test points == [(1., 2.), (5., 2.), (1., 7.), (5., 7.), (3., 4.)]

    # Preserve the preexisting least-squares expression exactly, including noise.
    noisy = [(r[1] + 0.002 * i, r[2] - 0.001 * i^2) for (i, r) in enumerate(reference)]
    design = zeros(2length(points), 6)
    rhs = zeros(2length(points))
    for i in eachindex(points)
        x, y = points[i]
        design[2i-1, :] .= (x, y, 1., 0., 0., 0.)
        design[2i, :] .= (0., 0., 0., x, y, 1.)
        rhs[2i-1], rhs[2i] = noisy[i]
    end
    coefficients = design \ rhs
    noisy_fit = calculate_manual_registration(points, noisy)
    @test noisy_fit.A == [coefficients[1] coefficients[2]; coefficients[4] coefficients[5]]
    @test noisy_fit.b == [coefficients[3], coefficients[6]]

    translated = [(x + 100_000, y - 70_000) for (x, y) in points]
    translated_reference = [Tuple(matrix * collect(p) + offset) for p in translated]
    @test calculate_manual_registration(translated, translated_reference).A ≈ matrix atol=1e-10

    @test_throws ArgumentError calculate_manual_registration(points[1:2], reference[1:2])
    @test_throws ArgumentError calculate_manual_registration(points, reference[1:3])
    collinear = [(0., 0.), (1., 2.), (2., 4.)]
    @test_throws ArgumentError calculate_manual_registration(collinear, points[1:3])
    @test_throws ArgumentError calculate_manual_registration(points[1:3], collinear)
    @test_throws ArgumentError calculate_manual_registration(fill((1., 1.), 3), points[1:3])
    @test_throws ArgumentError calculate_manual_registration(points[1:3], fill((1., 1.), 3))
    # Finite input coordinates can still require an unrepresentable fitted slope.
    edge = floatmax(Float64)
    @test_throws ArgumentError calculate_manual_registration(
        [(0., 0.), (1., 0.), (0., 1.)], [(-edge, 0.), (edge, 0.), (-edge, 1.)])
    for bad in (nothing, 1., (1.,), (1., 2., 3.), [1., 2., 3.],
                (NaN, 2.), (Inf, 2.), (1., -Inf), (1 + 2im, 2), (true, 2),
                (big"1e400", 2), (big"1e-400", 2))
        malformed = Any[points...]
        malformed[3] = bad
        @test_throws ArgumentError calculate_manual_registration(malformed, reference)
        @test_throws ArgumentError calculate_manual_registration(points, malformed)
    end
end
