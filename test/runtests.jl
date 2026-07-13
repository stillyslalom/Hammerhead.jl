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
    @test PIVParameters().peak_finder === :exclusion
    @test PIVParameters(peak_finder = :regionalmax).peak_finder === :regionalmax
    @test_throws ArgumentError PIVParameters(peak_finder = :watershed)
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

    # `final` overrides apply only to the last entry; shared kwargs still apply.
    fp = multipass_parameters([64, 32, 16]; padding = true,
                              final = (n_peaks = 2, keep_correlation_planes = true))
    @test all(p.padding for p in fp)                          # shared kwarg on every pass
    @test fp[1].n_peaks == 3 && fp[2].n_peaks == 3            # default, unaffected
    @test fp[3].n_peaks == 2 && fp[3].keep_correlation_planes # override on last only
    @test !fp[1].keep_correlation_planes && !fp[2].keep_correlation_planes
    # Empty `final` is a no-op.
    a = multipass_parameters([64, 32]; padding = true, final = (;))
    b = multipass_parameters([64, 32]; padding = true)
    @test [string(p) for p in a] == [string(p) for p in b]
    # Applies even to a length-1 schedule.
    @test multipass_parameters([32]; final = (n_peaks = 1,))[1].n_peaks == 1
end

@testset "effort schedules" begin
    low = Hammerhead.effort_schedule(:low)
    medium = Hammerhead.effort_schedule(:medium)
    high = Hammerhead.effort_schedule(:high)
    @test [p.window_size[1] for p in low] == [32]
    @test [p.window_size[1] for p in medium] == [64, 32]
    @test [p.window_size[1] for p in high] == [128, 64, 32]
    @test all(p.max_iterations >= 2 for p in high)
    @test all(p.padding && p.apodization === :gauss for p in high)
    @test high[end].uncertainty
    @test !any(p.uncertainty for p in high[1:end-1])

    ensemble_high = Hammerhead.effort_schedule(:high; ensemble = true)
    @test [p.window_size[1] for p in ensemble_high] == [128, 64, 32, 32]
    @test ensemble_high[end].uncertainty

    unpadded = Hammerhead.effort_schedule(:high; padding = false)
    @test !any(p.padding for p in unpadded)
    small_final = Hammerhead.effort_schedule(:high; window_size = 16)
    @test [p.window_size[1] for p in small_final] == [64, 32, 16]
    final_wins = Hammerhead.effort_schedule(:high; uncertainty = false,
                                            final = (uncertainty = true,
                                                     max_iterations = 4))
    @test final_wins[end].uncertainty
    @test final_wins[end].max_iterations == 4
    @test all(p.max_iterations == 2 for p in final_wins[1:end-1])
    clamped = Hammerhead.effort_schedule(:high; image_size = (64, 48))
    @test [p.window_size for p in clamped] == [(64, 48), (64, 48), (32, 32)]
    @test_throws ArgumentError Hammerhead.effort_schedule(:extreme)
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

    # Effort schedules are exactly the corresponding explicit pass vectors.
    effort = run_piv(imgA, imgB; effort = :medium, threaded = false)
    manual = run_piv(imgA, imgB,
                     Hammerhead.effort_schedule(:medium; image_size = size(imgA));
                     threaded = false)
    @test isequal(effort.u, manual.u)
    @test isequal(effort.v, manual.v)
    @test effort.parameters.window_size == manual.parameters.window_size
    @test_throws ArgumentError run_piv(imgA, imgB, params_pa; effort = :low)

    seq = run_piv_sequence([(imgA, imgB), (imgA, imgB)]; effort = :low,
                           progress = false)
    @test length(seq) == 2
    @test seq[1] isa PIVResult{Float64}
    @test_throws ArgumentError run_piv_sequence([(imgA, imgB)], params_pa;
                                                effort = :low, progress = false)

    ens = run_piv_ensemble([(imgA, imgB), (imgA, imgB)]; effort = :low,
                           progress = false)
    @test ens isa PIVResult{Float64}
    @test ens.parameters.window_size == (32, 32)
    @test_throws ArgumentError run_piv_ensemble([(imgA, imgB)], params_pa;
                                                effort = :low, progress = false)
end

@testset "Float32 pipeline" begin
    rng = MersenneTwister(21)
    image_size = (128, 128)
    dv, du = 3.0, 2.0
    positions = [(rand(rng) * 148 - 10, rand(rng) * 148 - 10) for _ in 1:250]
    imgA, imgB = particle_pair(image_size, positions, dv, du)
    imgA32, imgB32 = Float32.(imgA), Float32.(imgB)

    # correlate: du/dv/peak share the correlator's precision.
    c32 = CrossCorrelator{Float32}((64, 64))
    pairA, pairB = particle_pair((64, 64), [(32.0, 32.0)], dv, du)
    res32 = correlate(c32, Float32.(pairA), Float32.(pairB))
    @test res32.du isa Float32 && res32.dv isa Float32 && res32.peak isa Float32
    @test res32.du ≈ du atol = 0.1
    @test res32.dv ≈ dv atol = 0.1
    res32_2d = correlate(c32, Float32.(pairA), Float32.(pairB); subpixel = :gauss2d)
    @test res32_2d.du isa Float32
    res32_none = correlate(c32, Float32.(pairA), Float32.(pairB); subpixel = :none)
    @test res32_none.du isa Float32

    # run_piv: precision follows the images, single- and multi-pass.
    params = PIVParameters(window_size = 32, overlap = 16)
    result32 = run_piv(imgA32, imgB32, multipass_parameters([64, 32]))
    @test result32 isa PIVResult{Float32}
    @test eltype(result32.u) == eltype(result32.x) == eltype(result32.peak_ratio) == Float32
    @test median(result32.u) ≈ du atol = 0.25
    @test median(result32.v) ≈ dv atol = 0.25
    # ... and agrees with the Float64 analysis to well below measurement noise.
    result64 = run_piv(imgA, imgB, multipass_parameters([64, 32]))
    @test maximum(abs, result32.u .- result64.u) < 0.05
    # Mixed inputs promote to Float64.
    @test run_piv(imgA32, imgB, params) isa PIVResult{Float64}

    # Preprocessing preserves Float32; wrappers still copy.
    f32 = rand(rng, Float32, 32, 32)
    @test subtract_background(f32, zeros(Float32, 32, 32)) isa Matrix{Float32}
    @test intensity_cap(f32) isa Matrix{Float32}
    @test highpass_filter(f32) isa Matrix{Float32}
    @test clahe(f32) isa Matrix{Float32}
    buf32 = copy(f32)
    @test highpass_filter!(buf32) === buf32
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

@testset "iterative multipass (max_iterations)" begin
    @test PIVParameters().max_iterations == 1
    @test PIVParameters().convergence_tol == 0.05
    @test_throws ArgumentError PIVParameters(max_iterations = 0)
    @test_throws ArgumentError PIVParameters(convergence_tol = -0.1)
    @test occursin("max_iterations=3", string(PIVParameters(max_iterations = 3)))
    @test !occursin("max_iterations", string(PIVParameters()))

    # Linear shear u(y) = a*(y - 64) with particles crossing the edges — the
    # deformation-sensitive scene of the multi-pass testset above.
    rng = MersenneTwister(17)
    n = 128
    a = 1 / 16
    imgA = zeros(n, n)
    imgB = zeros(n, n)
    for _ in 1:900
        p = (rand(rng) * (n + 20) - 10, rand(rng) * (n + 20) - 10)
        add_particle!(imgA, p, 3.0)
        add_particle!(imgB, (p[1], p[2] + a * (p[1] - 64)), 3.0)
    end
    # isequal: masked cells hold NaN, where array `==` is always false.
    same(r1, r2) = isequal(r1.u, r2.u) && isequal(r1.v, r2.v) &&
                   isequal(r1.peak_ratio, r2.peak_ratio) &&
                   isequal(r1.correlation_moment, r2.correlation_moment) &&
                   isequal(r1.uncertainty_u, r2.uncertainty_u) &&
                   isequal(r1.uncertainty_v, r2.uncertainty_v) &&
                   r1.outliers == r2.outliers && r1.mask == r2.mask

    # An iterated final stage with the early exit disabled (`convergence_tol
    # = 0`) is exactly a schedule that repeats the final pass — including the
    # Wieneke uncertainty, which iterating stages estimate in a post-loop
    # sweep instead of fused into the correlation sweep.
    r_rep = run_piv(imgA, imgB,
                    multipass_parameters([32, 16, 16, 16]; uncertainty = true))
    r_it = run_piv(imgA, imgB,
                   multipass_parameters([32, 16]; uncertainty = true,
                       final = (max_iterations = 3, convergence_tol = 0.0)))
    @test same(r_rep, r_it)
    @test any(isfinite, r_it.uncertainty_u)   # the post-loop UQ sweep ran

    # A single-pass schedule iterates too (re-deforming by its own field),
    # which requires run_piv to build the image interpolants it would
    # otherwise skip.
    s_rep = run_piv(imgA, imgB, multipass_parameters([32, 32]))
    s_it = run_piv(imgA, imgB, PIVParameters(window_size = 32, overlap = 16,
                                             max_iterations = 2,
                                             convergence_tol = 0.0))
    @test same(s_rep, s_it)

    # Convergence early exit: an absurdly large tolerance stops after the
    # second sweep (the first sweep with a previous field to compare to).
    e2 = run_piv(imgA, imgB, multipass_parameters([32, 16];
                 final = (max_iterations = 2, convergence_tol = 0.0)))
    eb = run_piv(imgA, imgB, multipass_parameters([32, 16];
                 final = (max_iterations = 5, convergence_tol = 1e6)))
    @test same(e2, eb)

    # replace_outliers = false on an iterating final pass: sweeps replace
    # internally (the next predictor must be well behaved), but flagged cells
    # of the returned field still hold the measured data — identical to
    # repeating the final pass with replacement off.
    f_it = run_piv(imgA, imgB, multipass_parameters([32, 16];
                   min_peak_ratio = 1.3,
                   final = (max_iterations = 3, convergence_tol = 0.0,
                            replace_outliers = false, min_peak_ratio = 1.3)))
    f_rep = run_piv(imgA, imgB, multipass_parameters([32, 16, 16, 16];
                    min_peak_ratio = 1.3,
                    final = (replace_outliers = false, min_peak_ratio = 1.3)))
    @test sum(f_it.outliers) > 0          # non-vacuous
    @test same(f_it, f_rep)

    # Iterating with the default tolerance stays accurate on the shear, and
    # the threaded/workspace paths are bitwise identical to plain serial.
    passes = multipass_parameters([64, 32, 16]; padding = true,
                                  apodization = :gauss,
                                  final = (max_iterations = 3,))
    r_ser = run_piv(imgA, imgB, passes; threaded = false)
    @test sqrt(mean((r_ser.u .- [a * (yi - 64) for yi in r_ser.y, _ in r_ser.x]) .^ 2)) < 0.08
    r_thr = run_piv(imgA, imgB, passes; threaded = true)
    @test same(r_ser, r_thr)
    r_ws = run_piv(imgA, imgB, passes; threaded = false, workspace = piv_workspace())
    @test same(r_ser, r_ws)

    # Masked windows stay NaN/dropped through the iteration loop, and the
    # iterated≡repeated equivalence holds with a mask in play.
    mask = falses(n, n)
    mask[1:48, 1:48] .= true
    m_it = run_piv(imgA, imgB, multipass_parameters([32, 16];
                   final = (max_iterations = 3, convergence_tol = 0.0)); mask)
    m_rep = run_piv(imgA, imgB, multipass_parameters([32, 16, 16, 16]); mask)
    @test same(m_it, m_rep)
    @test any(m_it.mask) && all(isnan, m_it.u[m_it.mask])

    # Float32 pipeline: the convergence norm follows the image precision.
    r32 = run_piv(Float32.(imgA), Float32.(imgB),
                  multipass_parameters([32, 16]; final = (max_iterations = 2,)))
    @test r32 isa PIVResult{Float32}
end

@testset "keep_correlation_planes" begin
    rng = MersenneTwister(3)
    n = 96
    imgA = zeros(n, n); imgB = zeros(n, n)
    for _ in 1:400
        p = (rand(rng) * n, rand(rng) * n)
        add_particle!(imgA, p, 3.0)
        add_particle!(imgB, (p[1] + 2.0, p[2] + 1.0), 3.0)
    end
    @test !PIVParameters().keep_correlation_planes                 # default off
    @test PIVParameters(keep_correlation_planes = true).keep_correlation_planes

    r0 = run_piv(imgA, imgB, PIVParameters(window_size = 32, overlap = 16))
    @test r0.correlation_planes === nothing                        # opt-in

    passes = multipass_parameters([32, 32]; final = (keep_correlation_planes = true,))
    r = run_piv(imgA, imgB, passes)
    @test r.correlation_planes isa Matrix{Union{Nothing,Matrix{Float64}}}
    @test size(r.correlation_planes) == size(r.u)
    @test all(p -> p isa Matrix{Float64} && size(p) == (32, 32), r.correlation_planes)

    # Threaded and serial store identical planes.
    rs = run_piv(imgA, imgB, passes; threaded = false)
    rt = run_piv(imgA, imgB, passes; threaded = true)
    @test rs.u == rt.u && rs.correlation_planes == rt.correlation_planes

    # A masked window stores `nothing`, and JLD2 round-trips the planes.
    mask = falses(n, n); mask[1:40, 1:40] .= true
    rmask = run_piv(imgA, imgB, passes; mask)
    @test any(isnothing, rmask.correlation_planes)
    @test all(i -> !rmask.mask[i] || isnothing(rmask.correlation_planes[i]),
              eachindex(rmask.mask))     # masked ⟹ nothing plane
    tmp = tempname() * ".jld2"
    save_results(tmp, r)
    r2 = load_results(tmp)[1]
    @test r2.correlation_planes == r.correlation_planes
    Base.rm(tmp; force = true)
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

@testset "Preprocessing" begin
    a = [1.0 5.0; 3.0 2.0]
    b = [2.0 4.0; 1.0 6.0]
    @test compute_background([a, b]) == [1.0 4.0; 1.0 2.0]
    @test compute_background([a, b]; method = :mean) == [1.5 4.5; 2.0 4.0]
    @test_throws ArgumentError compute_background([a]; method = :median)
    @test_throws DimensionMismatch compute_background([a, zeros(3, 3)])
    sub = subtract_background(a, b)
    @test sub == [0.0 1.0; 2.0 0.0]  # clamped at zero

    img = fill(1.0, 8, 8)
    img[4, 4] = 100.0
    capped = intensity_cap(img)
    @test capped[4, 4] < 100.0 && capped[1, 1] == 1.0
    @test_throws ArgumentError intensity_cap(img; n_sigma = 0)

    # Highpass kills a uniform background but keeps a particle.
    flat = fill(5.0, 32, 32)
    @test maximum(highpass_filter(flat)) ≈ 0 atol = 1e-12
    part = copy(flat)
    add_particle!(part, (16.0, 16.0), 3.0)
    hp = highpass_filter(part; sigma = 4)
    @test hp[16, 16] > 0.5
    @test hp[1, 1] ≈ 0 atol = 0.01

    # CLAHE: output in [0,1], constant input is safe, contrast is stretched
    # in a dim region.
    rng = MersenneTwister(3)
    dim = 0.1 .* rand(rng, 64, 64)
    eq = clahe(dim)
    @test size(eq) == size(dim)
    @test all(0 .<= eq .<= 1)
    @test std(eq) > std(dim)
    @test clahe(zeros(16, 16)) == fill(0.5, 16, 16)
    @test_throws ArgumentError clahe(dim; clip_limit = 0.5)
    @test_throws ArgumentError clahe(dim; tiles = (0, 8))

    # Mutating variants return their (reused) buffer and match the allocating
    # versions exactly.
    raw = rand(rng, 16, 16) .+ 0.5
    bg2 = fill(0.2, 16, 16)
    buf = copy(raw)
    @test subtract_background!(buf, bg2) === buf
    @test buf == subtract_background(raw, bg2)

    buf = copy(raw)
    buf[5, 5] = 50.0
    reference = intensity_cap(buf)
    @test intensity_cap!(buf) === buf
    @test buf == reference

    buf = copy(part)
    @test highpass_filter!(buf; sigma = 4) === buf
    @test buf == hp

    buf = copy(dim)
    @test clahe!(buf) === buf
    @test buf == eq
    z = zeros(16, 16)
    @test clahe!(z) === z && z == fill(0.5, 16, 16)
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

@testset "arrow_lengthscale (plot auto scaling)" begin
    # 0.99-quantile magnitude maps to the target length.
    u = fill(1.0, 4, 4); v = zeros(4, 4)       # 15 vectors of magnitude 1
    u[1, 1] = 10.0                              # one large (magnitude 10)
    q_all = quantile([hypot(u[i], v[i]) for i in eachindex(u)], 0.99)
    @test Hammerhead.arrow_lengthscale(u, v, nothing, 10.0) ≈ 10.0 / q_all
    # `valid` excluding the large vector ⟹ smaller quantile ⟹ larger scale.
    valid = trues(4, 4); valid[1, 1] = false
    @test Hammerhead.arrow_lengthscale(u, v, valid, 10.0) >
          Hammerhead.arrow_lengthscale(u, v, nothing, 10.0)
    # Zero field ⟹ no scaling; all-NaN ⟹ no scaling (NaNs are ignored).
    @test Hammerhead.arrow_lengthscale(zeros(3, 3), zeros(3, 3), nothing, 5.0) === nothing
    @test Hammerhead.arrow_lengthscale(fill(NaN, 2, 2), fill(NaN, 2, 2), nothing, 5.0) === nothing
    # Empty `valid` selection falls back to all-finite vectors.
    @test Hammerhead.arrow_lengthscale(u, v, falses(4, 4), 10.0) ≈ 10.0 / q_all
    @test Hammerhead.grid_axis_step([2.0, 5.0, 8.0]) ≈ 3.0
    @test Hammerhead.grid_axis_step([1.0]) == 1.0
end

include("test_synthetic.jl")
include("test_calibration.jl")
include("test_dewarp.jl")
include("test_stereo.jl")
include("test_backend.jl")
include("test_ka.jl")
include("test_selfcal.jl")
include("test_validation.jl")
include("test_masking.jl")
include("test_peaks.jl")
include("test_accuracy.jl")
include("test_uncertainty.jl")
include("test_ensemble.jl")
include("test_ptv.jl")
include("test_io.jl")
include("test_scaling.jl")
include("test_reference.jl")

end # top-level testset
