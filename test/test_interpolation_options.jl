# Image and predictor interpolation choices for deforming passes.

@testset "Interpolation options" begin
    rng = MersenneTwister(21)
    n = 128
    du, dv = 2.3, -1.7
    positions = [(n * rand(rng), n * rand(rng)) for _ in 1:900]
    A, B = particle_pair((n, n), positions, dv, du)
    sched(; kw...) = multipass_parameters([32, 16]; padding = true, apodization = :gauss, kw...)
    inner(r) = (2:size(r.u, 1)-1, 2:size(r.u, 2)-1)
    med(x) = median(filter(isfinite, vec(x)))

    base = run_piv(A, B, sched())
    lin = run_piv(A, B, sched(image_interpolation = :linear))
    cub = run_piv(A, B, sched(predictor_interpolation = :cubic))
    @test !isequal(lin.u, base.u)
    for r in (base, cub)
        @test med(r.u[inner(r)...]) ≈ du atol = 0.05
        @test med(r.v[inner(r)...]) ≈ dv atol = 0.05
    end
    @test med(lin.u[inner(lin)...]) ≈ du atol = 0.1
    @test med(lin.v[inner(lin)...]) ≈ dv atol = 0.1
    # on a near-uniform field the two predictors agree closely
    @test maximum(abs.(filter(isfinite, cub.u .- base.u))) < 0.05

    # the cubic predictor reproduces a linear field between its nodes, and
    # stays flat outside the grid; coarse grids fall back to linear
    ys, xs = collect(10.0:20.0:130.0), collect(5.0:10.0:75.0)
    F = [0.3y - 0.2x for y in ys, x in xs]
    itp = Hammerhead.predictor_interpolant(ys, xs, F; method = :cubic)
    @test itp isa Hammerhead._CubicPredictor
    @test itp(37.0, 41.5) ≈ 0.3 * 37.0 - 0.2 * 41.5 atol = 1e-9
    @test itp(0.0, 41.5) ≈ itp(10.0, 41.5)
    @test !(Hammerhead.predictor_interpolant(ys[1:3], xs, F[1:3, :]; method = :cubic) isa
            Hammerhead._CubicPredictor)

    # the ensemble path honors the choice too
    ens = run_piv_ensemble([(A, B)], sched(image_interpolation = :linear); progress = false)
    @test med(ens.u) ≈ du atol = 0.1

    @test_throws ArgumentError PIVParameters(image_interpolation = :quintic)
    @test_throws ArgumentError PIVParameters(predictor_interpolation = :nearest)
    @test_throws ArgumentError run_piv(A, B, sched(image_interpolation = :linear); backend = :ka)
    p = PIVParameters(image_interpolation = :linear, predictor_interpolation = :cubic)
    @test occursin("image_interpolation=:linear", sprint(show, p))
    mktempdir() do dir
        r = PIVRecipe(sched(predictor_interpolation = :cubic))
        @test load_recipe(save_recipe(joinpath(dir, "r.jld2"), r)) == r
    end
end
