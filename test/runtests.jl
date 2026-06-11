using Hammerhead
using Test
using LinearAlgebra
using Random
using Statistics

"""
Add a Gaussian particle image at `centroid = (row, col)` with the given
diameter (≈ 2σ) to `array`.
"""
function add_particle!(array::AbstractMatrix, centroid::Tuple{<:Real,<:Real}, diameter::Real)
    sigma = diameter / 2
    for j in axes(array, 2), i in axes(array, 1)
        d2 = (i - centroid[1])^2 + (j - centroid[2])^2
        d2 > (5sigma)^2 && continue
        array[i, j] += exp(-0.5 * d2 / sigma^2)
    end
    return array
end

"""
Create an image pair of size `image_size` containing Gaussian particles at
`positions` (vector of `(row, col)`), displaced by `(dv, du) = (drow, dcol)`
in the second image.
"""
function particle_pair(image_size, positions, dv, du; diameter = 3.0)
    imgA = zeros(Float64, image_size)
    imgB = zeros(Float64, image_size)
    for p in positions
        add_particle!(imgA, p, diameter)
        add_particle!(imgB, (p[1] + dv, p[2] + du), diameter)
    end
    return imgA, imgB
end

@testset "Hammerhead.jl" begin

@testset "PIVParameters validation" begin
    p = PIVParameters()
    @test p.window_size == (32, 32)
    @test p.overlap == (16, 16)
    @test PIVParameters(window_size = 64).window_size == (64, 64)
    @test PIVParameters(overlap = 0).overlap == (0, 0)
    @test_throws ArgumentError PIVParameters(window_size = 2)
    @test_throws ArgumentError PIVParameters(overlap = (32, 16))     # overlap == window
    @test_throws ArgumentError PIVParameters(overlap = -1)
    @test_throws ArgumentError PIVParameters(correlation_method = :fancy)
    @test_throws ArgumentError PIVParameters(apodization = :hann)
    @test_throws ArgumentError PIVParameters(subpixel_method = :spline)
    @test_throws ArgumentError PIVParameters(uod_threshold = 0)
    @test_throws ArgumentError PIVParameters(uod_neighborhood = 0)
    @test_throws ArgumentError PIVParameters(min_peak_ratio = -1)
end

@testset "multipass_parameters" begin
    passes = multipass_parameters([64, 32, 16]; padding = true, apodization = :gauss)
    @test length(passes) == 3
    @test passes[1].window_size == (64, 64) && passes[1].overlap == (32, 32)
    @test passes[3].window_size == (16, 16) && passes[3].overlap == (8, 8)
    @test all(p.padding && p.apodization === :gauss for p in passes)
    @test multipass_parameters([(64, 32)])[1].window_size == (64, 32)
    @test multipass_parameters([32]; overlap_fraction = 0.75)[1].overlap == (24, 24)
    @test_throws ArgumentError multipass_parameters([32]; overlap_fraction = 1.0)
    @test_throws ArgumentError multipass_parameters(Int[])
end

@testset "Correlators: known displacement" begin
    image_size = (64, 64)
    center = image_size .÷ 2
    for (dv, du) in ((2.2, 1.3), (-3.4, 2.6), (0.0, 0.0))
        imgA, imgB = particle_pair(image_size, [center], dv, du)
        for C in (CrossCorrelator, PhaseCorrelator)
            c = C{Float64}(image_size)
            res = correlate(c, imgA, imgB)
            @test res.du ≈ du atol = 0.1
            @test res.dv ≈ dv atol = 0.1
            @test res.peak > 0
        end
    end
end

@testset "Subpixel methods" begin
    image_size = (64, 64)
    dv, du = 2.7, -1.4
    imgA, imgB = particle_pair(image_size, [image_size .÷ 2], dv, du)
    c = CrossCorrelator{Float64}(image_size)

    res_none = correlate(c, imgA, imgB; subpixel = :none)
    @test res_none.du == round(du) && res_none.dv == round(dv)

    res_g3 = correlate(c, imgA, imgB; subpixel = :gauss3)
    @test res_g3.du ≈ du atol = 0.1
    @test res_g3.dv ≈ dv atol = 0.1

    res_g2d = correlate(c, imgA, imgB; subpixel = :gauss2d)
    @test res_g2d.du ≈ du atol = 0.15
    @test res_g2d.dv ≈ dv atol = 0.15

    @test_throws ArgumentError correlate(c, imgA, imgB; subpixel = :spline)
    @test_throws DimensionMismatch correlate(c, imgA[1:32, 1:32], imgB[1:32, 1:32])
end

@testset "Padded and apodized correlation" begin
    image_size = (64, 64)
    dv, du = 2.2, 1.3
    imgA, imgB = particle_pair(image_size, [image_size .÷ 2], dv, du)
    for kwargs in ((padding = true,), (apodization = :gauss,),
                   (padding = true, apodization = :gauss))
        for C in (CrossCorrelator, PhaseCorrelator)
            c = C{Float64}(image_size; kwargs...)
            res = correlate(c, imgA, imgB)
            @test size(res.correlation) ==
                  (get(kwargs, :padding, false) ? 2 .* image_size : image_size)
            @test res.du ≈ du atol = 0.1
            @test res.dv ≈ dv atol = 0.1
        end
    end
    @test_throws ArgumentError CrossCorrelator{Float64}(image_size; apodization = :hann)
end

@testset "Deformable correlation" begin
    image_size = (64, 64)
    dv, du = 3.6, -2.3
    imgA, imgB = particle_pair(image_size, [image_size .÷ 2], dv, du)
    c = CrossCorrelator{Float64}(image_size)
    res = correlate_deformable(c, imgA, imgB; iterations = 4)
    @test res.du ≈ du atol = 0.05
    @test res.dv ≈ dv atol = 0.05
    @test_throws ArgumentError correlate_deformable(c, imgA, imgB; iterations = 0)
end

@testset "run_piv: uniform displacement" begin
    rng = MersenneTwister(42)
    image_size = (128, 128)
    dv, du = 3.0, 2.0
    # Draw particles over an extended domain so particles enter at the edges.
    positions = [(rand(rng) * 148 - 10, rand(rng) * 148 - 10) for _ in 1:250]
    imgA, imgB = particle_pair(image_size, positions, dv, du)

    # Plain circular correlation carries a small bias toward zero displacement
    # (wrap-around noise tilts the subpixel fit), hence the looser tolerance.
    for method in (:cross, :phase)
        params = PIVParameters(window_size = 32, overlap = 16, correlation_method = method)
        result = run_piv(imgA, imgB, params)
        @test size(result.u) == (length(result.y), length(result.x))
        @test median(result.u) ≈ du atol = 0.25
        @test median(result.v) ≈ dv atol = 0.25
        good_u = count(abs.(result.u .- du) .< 0.5)
        @test good_u >= 0.85 * length(result.u)
        @test sum(result.outliers) <= 0.15 * length(result.u)
        @test all(result.peak_ratio .> 1)
    end

    # Padding + overlap normalization + apodization removes the bias entirely.
    params_pa = PIVParameters(window_size = 32, overlap = 16,
                              padding = true, apodization = :gauss)
    result_pa = run_piv(imgA, imgB, params_pa)
    @test median(result_pa.u) ≈ du atol = 0.05
    @test median(result_pa.v) ≈ dv atol = 0.05
    @test count(abs.(result_pa.u .- du) .< 0.2) == length(result_pa.u)

    # Threaded execution matches serial exactly.
    r_ser = run_piv(imgA, imgB, params_pa; threaded = false)
    r_thr = run_piv(imgA, imgB, params_pa; threaded = true)
    @test r_ser.u == r_thr.u
    @test r_ser.v == r_thr.v
    @test r_ser.peak_ratio == r_thr.peak_ratio

    @test_throws DimensionMismatch run_piv(imgA, imgB[1:64, 1:64])
    @test_throws ArgumentError run_piv(imgA[1:16, 1:16], imgB[1:16, 1:16],
                                       PIVParameters(window_size = 32))
end

@testset "run_piv: multi-pass on linear shear" begin
    # u(y) = a*(y - 64), v = 0: ±4 px at the image edges, with strong
    # in-window gradients — the case where image deformation matters.
    rng = MersenneTwister(11)
    a = 1 / 16
    n = 128
    imgA = zeros(n, n)
    imgB = zeros(n, n)
    for _ in 1:900
        p = (rand(rng) * (n + 20) - 10, rand(rng) * (n + 20) - 10)
        add_particle!(imgA, p, 3.0)
        add_particle!(imgB, (p[1], p[2] + a * (p[1] - 64)), 3.0)
    end
    kw = (padding = true, apodization = :gauss)
    rms_u(r) = sqrt(mean((r.u .- [a * (yi - 64) for yi in r.y, _ in r.x]) .^ 2))

    single = run_piv(imgA, imgB, PIVParameters(; window_size = 16, overlap = 8, kw...))
    multi = run_piv(imgA, imgB, multipass_parameters([64, 32, 16, 16]; kw...))
    @test size(multi.u) == size(single.u)
    @test rms_u(multi) < 0.08          # measured ≈ 0.04 px
    @test rms_u(multi) < rms_u(single) # deformation beats direct correlation
    @test sqrt(mean(multi.v .^ 2)) < 0.05
    @test sum(multi.outliers) <= 3     # no false-positive validation on smooth shear
end

@testset "Vector replacement and smoothing" begin
    u = [Float64(j) for i in 1:6, j in 1:6]  # u = column index
    v = -copy(u)
    u[3, 4] = 99.0
    v[3, 4] = -99.0
    invalid = falses(6, 6)
    invalid[3, 4] = true
    Hammerhead.replace_vectors!(u, v, invalid)
    @test u[3, 4] ≈ 4.0  # median of valid neighbors restores the local value
    @test v[3, 4] ≈ -4.0

    # Fewer than 3 valid vectors anywhere: flagged entries are left unchanged.
    u2 = fill(7.0, 3, 3)
    v2 = zeros(3, 3)
    invalid2 = trues(3, 3)
    invalid2[1, 1] = false
    u2[2, 2] = 42.0
    Hammerhead.replace_vectors!(u2, v2, invalid2)
    @test u2[2, 2] == 42.0
    @test_throws ArgumentError Hammerhead.replace_vectors!(u2, zeros(2, 2), invalid2)

    # smooth_field: constants exact, linear fields preserved in the interior.
    @test Hammerhead.smooth_field(fill(3.0, 5, 5)) ≈ fill(3.0, 5, 5)
    lin = [Float64(i + 2j) for i in 1:6, j in 1:6]
    sm = Hammerhead.smooth_field(lin)
    @test sm[2:5, 2:5] ≈ lin[2:5, 2:5]
end

@testset "Universal outlier detection" begin
    rng = MersenneTwister(7)
    u = 3.0 .+ 0.05 .* randn(rng, 8, 8)  # uniform field with subpixel noise
    v = 0.05 .* randn(rng, 8, 8)
    @test !any(universal_outlier_detection(u, v, 2.0))
    u_spiked = copy(u)
    u_spiked[4, 5] += 50.0
    mask = universal_outlier_detection(u_spiked, v, 2.0)
    @test mask[4, 5]
    @test sum(mask) <= 3  # the spike, perhaps a neighbor or two
    @test_throws ArgumentError universal_outlier_detection(u, zeros(4, 4), 2.0)
    @test_throws ArgumentError universal_outlier_detection(u, v, 2.0; neighborhood_size = 0)
    @test_throws ArgumentError universal_outlier_detection(u, v, 2.0; epsilon = 0)
end

@testset "Peak ratio and correlation moment" begin
    R = zeros(32, 32)
    R[16, 16] = 1.0
    R[8, 24] = 0.25
    @test calculate_peak_ratio(R, (16, 16)) ≈ 4.0
    @test calculate_peak_ratio(R, (16, 16); exclusion_radius = 10) == Inf
    @test_throws ArgumentError calculate_peak_ratio(R, (0, 0))

    # A wider Gaussian peak has a larger moment than a narrow one.
    narrow = zeros(33, 33)
    wide = zeros(33, 33)
    add_particle!(narrow, (17.0, 17.0), 2.0)
    add_particle!(wide, (17.0, 17.0), 6.0)
    m_narrow = calculate_correlation_moment(narrow, (17.0, 17.0))
    m_wide = calculate_correlation_moment(wide, (17.0, 17.0))
    @test 0 < m_narrow < m_wide
    @test isnan(calculate_correlation_moment(narrow, (NaN, NaN)))
    @test_throws ArgumentError calculate_correlation_moment(narrow, (17.0, 17.0);
                                                            neighborhood_size = 4)
end

@testset "Affine transforms" begin
    @test_throws ArgumentError AffineTransform(zeros(3, 3), zeros(2))
    @test_throws ArgumentError AffineTransform(zeros(2, 2), zeros(3))
    @test AffineTransform().A == I(2)

    # Registration round-trip: recover a known affine map from point pairs.
    A_true = [1.1 0.2; -0.15 0.95]
    b_true = [3.0, -2.0]
    pts = [(2.0, 3.0), (10.0, 4.0), (5.0, 12.0), (8.0, 8.0), (1.0, 9.0)]
    refs = [Tuple(A_true * [p[1], p[2]] + b_true) for p in pts]
    tform = calculate_manual_registration(pts, refs)
    @test tform.A ≈ A_true atol = 1e-10
    @test tform.b ≈ b_true atol = 1e-10
    @test_throws ArgumentError calculate_manual_registration(pts[1:2], refs[1:2])
    @test_throws ArgumentError calculate_manual_registration(pts, refs[1:3])

    # warp_image: pure translation moves a particle by (b_y, b_x).
    img = zeros(48, 48)
    add_particle!(img, (24.0, 20.0), 4.0)
    warped = warp_image(img, AffineTransform([1.0 0.0; 0.0 1.0], [5.0, -3.0]))
    expected = zeros(48, 48)
    add_particle!(expected, (21.0, 25.0), 4.0)
    @test isapprox(warped[5:44, 5:44], expected[5:44, 5:44], atol = 1e-3)

    # transform_vector_field: 90° rotation maps (u, v) = (1, 0) to (0, 1).
    rot = AffineTransform([0.0 -1.0; 1.0 0.0], [0.0, 0.0])
    nx, ny, nu, nv = transform_vector_field([1.0], [0.0], [1.0], [0.0], rot)
    @test nx ≈ [0.0] atol = 1e-12
    @test ny ≈ [1.0] atol = 1e-12
    @test nu ≈ [0.0] atol = 1e-12
    @test nv ≈ [1.0] atol = 1e-12
    @test_throws ArgumentError transform_vector_field([1.0], [0.0], [1.0], zeros(2), rot)
end

@testset "Makie extension stub" begin
    # Without a Makie backend loaded, the stubs raise a helpful error.
    err = try
        plot_vector_field([1.0], [1.0], ones(1, 1), ones(1, 1))
        nothing
    catch e
        e
    end
    @test err isa ErrorException && occursin("Makie", err.msg)
    @test_throws ErrorException plot_vector_field!(nothing)
end

end # top-level testset
