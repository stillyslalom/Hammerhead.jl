using Hammerhead
using Hammerhead.SyntheticData
using Test
using Random
using StableRNGs
using Statistics
using LinearAlgebra
using FileIO: save
using ImageCore: Gray, N0f16
using JLD2: jldopen

# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------

# Map each detected particle to the nearest ground-truth particle within `tol`
# px (−1 if none). `generate_synthetic_piv_pair` returns index-aligned truth
# (particles1[i] ↔ particles2[i]), so a match (a, b) is a correct
# correspondence when the two detections map to the same truth index.
function truth_map(p::Particles, tx, ty; tol = 0.5)
    map = fill(-1, length(p))
    for i in 1:length(p)
        best = Inf; bj = 0
        for j in eachindex(tx)
            d2 = (p.x[i] - tx[j])^2 + (p.y[i] - ty[j])^2
            d2 < best && (best = d2; bj = j)
        end
        sqrt(best) < tol && (map[i] = bj)
    end
    return map
end

# Fraction of correct matches and the displacement error over correct ones.
function matching_accuracy(res, pf1, pf2)
    mA = truth_map(res.particles_a, pf1.x, pf1.y)
    mB = truth_map(res.particles_b, pf2.x, pf2.y)
    correct = 0; total = 0; derr = Float64[]
    for k in eachindex(res.index_a)
        ta = mA[res.index_a[k]]; tb = mB[res.index_b[k]]
        (ta > 0 && tb > 0) || continue
        total += 1
        if ta == tb
            correct += 1
            push!(derr, hypot(res.u[k] - (pf2.x[tb] - pf1.x[ta]),
                              res.v[k] - (pf2.y[tb] - pf1.y[ta])))
        end
    end
    med = isempty(derr) ? NaN : median(derr)
    rms = isempty(derr) ? NaN : sqrt(mean(derr .^ 2))
    return (frac = correct / total, n = total, med = med, rms = rms)
end

# A grid of jittered, non-overlapping Gaussian particles at known positions.
function detection_scene(; noise, rng, imgsize = (256, 256), spacing = 16.0, jitter = 4.0)
    img = zeros(imgsize)
    tx = Float64[]; ty = Float64[]; td = Float64[]
    for cx in 20.0:spacing:(imgsize[2] - 18), cy in 20.0:spacing:(imgsize[1] - 18)
        px = cx + jitter * (rand(rng) - 0.5)
        py = cy + jitter * (rand(rng) - 0.5)
        d = 2.5 + 1.5 * rand(rng)
        generate_gaussian_particle!(img, (px, py), d, 1.0)
        push!(tx, px); push!(ty, py); push!(td, d)
    end
    if noise > 0
        img .+= noise .* randn(rng, imgsize...)
        img .= max.(img, 0.0)
    end
    return img, tx, ty, td
end

@testset "PTV" begin
    @testset "PTVParameters validation" begin
        p = PTVParameters()
        @test p.threshold === :auto
        @test p.search_radius == 3.0
        @test PTVParameters(threshold = 0.2).threshold == 0.2
        @test_throws ArgumentError PTVParameters(threshold_k = 0)
        @test_throws ArgumentError PTVParameters(min_separation = -1)
        @test_throws ArgumentError PTVParameters(search_radius = 0)
        @test_throws ArgumentError PTVParameters(uod_threshold = 0)
        @test_throws ArgumentError PTVParameters(uod_epsilon = 0)
        @test_throws ArgumentError PTVParameters(min_diameter = 5, max_diameter = 2)
        @test_throws ArgumentError PTVParameters(min_diameter = 0)
        @test_throws ArgumentError PTVParameters(uod_neighbors = 2)
        @test occursin("PTVParameters", sprint(show, p))
    end

    @testset "Detection accuracy" begin
        img, tx, ty, td = detection_scene(; noise = 0.0, rng = StableRNG(1))
        p = detect_particles(img)
        # Recall and subpixel position/diameter accuracy against truth.
        perr = Float64[]; derr = Float64[]; found = 0
        for i in eachindex(tx)
            best = Inf; bj = 0
            for j in 1:length(p)
                d2 = (p.x[j] - tx[i])^2 + (p.y[j] - ty[i])^2
                d2 < best && (best = d2; bj = j)
            end
            if bj > 0 && sqrt(best) < 1.0
                found += 1
                push!(perr, sqrt(best))
                isnan(p.diameter[bj]) || push!(derr, abs(p.diameter[bj] - td[i]) / td[i])
            end
        end
        @test found == length(tx)                       # 100% recall, noiseless
        @test sqrt(mean(perr .^ 2)) < 0.02              # measured ≈ 0
        @test median(derr) < 0.15                       # measured ≈ 0

        # With modest noise: still (nearly) complete recall and sub-0.15 px RMS.
        imgn, txn, tyn, _ = detection_scene(; noise = 0.01, rng = StableRNG(2))
        pn = detect_particles(imgn)
        perrn = Float64[]; foundn = 0
        for i in eachindex(txn)
            best = Inf; bj = 0
            for j in 1:length(pn)
                d2 = (pn.x[j] - txn[i])^2 + (pn.y[j] - tyn[i])^2
                d2 < best && (best = d2; bj = j)
            end
            if bj > 0 && sqrt(best) < 1.0
                foundn += 1
                push!(perrn, sqrt(best))
            end
        end
        @test foundn >= 0.98 * length(txn)
        @test sqrt(mean(perrn .^ 2)) < 0.15
    end

    @testset "Detection edge cases" begin
        # Dedupe keeps the brighter of a close pair.
        img = zeros(64, 64)
        generate_gaussian_particle!(img, (32.0, 30.0), 3.0, 1.0)
        generate_gaussian_particle!(img, (33.1, 30.5), 3.0, 0.4)
        p = detect_particles(img, PTVParameters(min_separation = 3.0))
        @test length(p) == 1
        @test hypot(p.x[1] - 32.0, p.y[1] - 30.0) < 0.5   # the bright one survived

        # Masked region yields no detections there.
        img2 = zeros(128, 128)
        for c in 20:20:120, r in 20:20:120
            generate_gaussian_particle!(img2, (Float64(c), Float64(r)), 3.0, 1.0)
        end
        mask = falses(128, 128); mask[1:64, :] .= true
        pm = detect_particles(img2, PTVParameters(); mask = mask)
        @test length(pm) > 0
        @test !any(pm.y .< 64)

        # A saturated (clipped-top) particle is still detected once, via the
        # centroid fallback.
        img3 = zeros(64, 64)
        generate_gaussian_particle!(img3, (32.0, 32.0), 5.0, 3.0)
        img3 .= min.(img3, 1.0)
        p3 = detect_particles(img3, PTVParameters(max_diameter = 20.0))
        @test count(hypot.(p3.x .- 32.0, p3.y .- 32.0) .< 2.0) == 1

        # A blank frame detects nothing without throwing.
        @test length(detect_particles(zeros(64, 64))) == 0
    end

    @testset "Two-frame matching (hybrid)" begin
        rng = MersenneTwister(42)
        vf = vortex_flow(128.0, 128.0, 8.0, 0.0)
        img1, img2, pf1, pf2 = generate_synthetic_piv_pair(vf, (256, 256), 1.0;
            particle_density = 0.02, rng = rng, laser_sheet = GaussianLaserSheet(0.0, 8.0, 1.0))
        res = run_ptv(img1, img2)
        acc = matching_accuracy(res, pf1, pf2)
        @test acc.frac >= 0.95           # measured ≈ 0.98
        @test acc.med < 0.05             # measured ≈ 0.03
        @test acc.rms < 0.15             # measured ≈ 0.06
        # Frame-A attribution: x/y are frame-A positions of the matches.
        @test all(1 .<= res.x .<= 256) && all(1 .<= res.y .<= 256)
        @test length(res.match_residual) == length(res.x)
    end

    @testset "Pure nearest-neighbor" begin
        # Small uniform displacement: pure NN works.
        rng = MersenneTwister(3)
        lf = linear_flow(2.0, 1.0, 0.0, 0, 0, 0, 0)
        i1, i2, f1, f2 = generate_synthetic_piv_pair(lf, (256, 256), 1.0;
            particle_density = 0.01, rng = rng, laser_sheet = GaussianLaserSheet(0.0, 8.0, 1.0))
        res = run_ptv(i1, i2, PTVParameters(search_radius = 4.0); predictor = nothing)
        @test matching_accuracy(res, f1, f2).frac >= 0.75    # measured ≈ 0.85

        # Large displacement: pure NN fails — this is why hybrid is the default.
        lf2 = linear_flow(8.0, 0.0, 0.0, 0, 0, 0, 0)
        j1, j2, g1, g2 = generate_synthetic_piv_pair(lf2, (256, 256), 1.0;
            particle_density = 0.01, rng = MersenneTwister(3), laser_sheet = GaussianLaserSheet(0.0, 8.0, 1.0))
        res2 = run_ptv(j1, j2, PTVParameters(search_radius = 4.0); predictor = nothing)
        @test matching_accuracy(res2, g1, g2).frac < 0.5     # measured ≈ 0
    end

    @testset "Predictor pass-through" begin
        rng = MersenneTwister(17)
        vf = vortex_flow(128.0, 128.0, 6.0, 0.0)
        img1, img2, = generate_synthetic_piv_pair(vf, (256, 256), 1.0;
            particle_density = 0.02, rng = rng, laser_sheet = GaussianLaserSheet(0.0, 8.0, 1.0))
        passes = multipass_parameters([64, 32])
        pr = run_piv(img1, img2, passes)
        rA = run_ptv(img1, img2; predictor = :piv, piv_passes = passes)
        rB = run_ptv(img1, img2; predictor = pr)
        @test rA.index_a == rB.index_a
        @test rA.index_b == rB.index_b
        @test rA.u == rB.u && rA.v == rB.v
    end

    @testset "Scattered UOD" begin
        # Uniform field with subpixel noise plus a few corrupted vectors: exactly
        # the corrupted ones are flagged.
        rng = StableRNG(11)
        n = 300
        x = 256 .* rand(rng, n); y = 256 .* rand(rng, n)
        u = fill(3.0, n) .+ 0.05 .* randn(rng, n)
        v = fill(2.0, n) .+ 0.05 .* randn(rng, n)
        corrupt = [5, 60, 120, 200, 280]
        for c in corrupt; u[c] += 12.0; v[c] -= 10.0; end
        flags = Hammerhead.scattered_uod(x, y, u, v, PTVParameters())
        @test all(flags[c] for c in corrupt)
        @test sum(flags) == length(corrupt)

        # A smooth shear field is not flagged.
        us = 0.01 .* (y .- 128) .+ 0.02 .* randn(rng, n)
        vs = 0.02 .* randn(rng, n)
        @test sum(Hammerhead.scattered_uod(x, y, us, vs, PTVParameters())) == 0

        # Too few matches → nothing flagged, no throw.
        @test sum(Hammerhead.scattered_uod([1.0, 2.0], [1.0, 2.0], [0.0, 9.0], [0.0, 0.0],
                                           PTVParameters())) == 0
    end

    @testset "ptv_to_grid" begin
        rng = MersenneTwister(11)
        lf = linear_flow(3.0, 2.0, 0.0, 0, 0, 0, 0)
        img1, img2, = generate_synthetic_piv_pair(lf, (256, 256), 1.0;
            particle_density = 0.02, rng = rng, laser_sheet = GaussianLaserSheet(0.0, 8.0, 1.0))
        res = run_ptv(img1, img2)
        g = ptv_to_grid(res, (256, 256))
        @test g isa PIVResult
        unmasked = [i for i in eachindex(g.u) if !g.mask[i]]
        @test !isempty(unmasked)
        @test maximum(abs.(g.u[unmasked] .- 3.0)) < 0.1     # measured ≈ 0.045
        @test maximum(abs.(g.v[unmasked] .- 2.0)) < 0.1
        @test all(isnan, g.peak_ratio)
        @test !any(g.outliers)
        # Masked-result conventions plug into field_statistics without error.
        fs = field_statistics([g])
        @test size(fs.mean_u) == size(g.u)

        # Sparse corners get masked by min_count.
        sparse = PTVResult{Float64}([10.0, 12.0, 11.0], [10.0, 11.0, 12.0],
            [3.0, 3.0, 3.0], [2.0, 2.0, 2.0], zeros(3), falses(3),
            Int[1, 2, 3], Int[1, 2, 3], res.particles_a, res.particles_b, res.parameters)
        gs = ptv_to_grid(sparse, (256, 256); min_count = 3)
        @test count(gs.mask) > count(.!gs.mask)              # mostly masked
    end

    @testset "Sequence and serialization" begin
        rng = MersenneTwister(9)
        vf = vortex_flow(128.0, 128.0, 5.0, 0.0)
        img1, img2, = generate_synthetic_piv_pair(vf, (256, 256), 1.0;
            particle_density = 0.02, rng = rng, laser_sheet = GaussianLaserSheet(0.0, 8.0, 1.0))
        mktempdir() do dir
            files = [joinpath(dir, "f$i.png") for i in 1:4]
            for (f, img) in zip(files, (img1, img2, img1, img2))
                save(f, Gray{N0f16}.(clamp.(img, 0, 1)))
            end
            out = joinpath(dir, "ptv.jld2")
            res = run_ptv_sequence(Hammerhead.image_pairs(files), PTVParameters();
                                   output = out, progress = false)
            @test length(res) == 2
            @test all(r -> r isa PTVResult, res)
            loaded = load_results(out)
            @test length(loaded) == 2
            @test loaded[1] isa PTVResult
            @test loaded[1].index_a == res[1].index_a
            jldopen(out, "r") do f
                @test f["format_version"] == 4
                @test f["sources/000001"] == [files[1], files[2]]
            end

            # Mixed-type save_results round-trips.
            pr = run_piv(img1, img2, PIVParameters(window_size = 32, overlap = 16))
            p = joinpath(dir, "mixed.jld2")
            save_results(p, Union{PIVResult,PTVResult}[pr, res[1]])
            mixed = load_results(p)
            @test mixed[1] isa PIVResult && mixed[2] isa PTVResult
        end
    end

    @testset "Tracking" begin
        # A 6-frame vortex sequence (center off-frame for smooth rotation, thick
        # sheet so no dropout) tracked end to end.
        rng = MersenneTwister(20)
        vf = vortex_flow(128.0, 360.0, 2.0, 0.0)
        sheet = GaussianLaserSheet(0.0, 40.0, 1.0)
        f0 = generate_particle_field((256, 256), 0.005; z_range = (-0.5, 0.5), rng = rng)
        nframes = 6
        frames = Matrix{Float64}[]
        f = copy(f0)
        px = [Float64[] for _ in 1:length(f.x)]
        py = [Float64[] for _ in 1:length(f.x)]
        for k in 1:nframes
            push!(frames, render_particle_image(f, (256, 256), sheet;
                background_noise = 0.0, rng = MersenneTwister(2000 + k)))
            for i in eachindex(f.x); push!(px[i], f.x[i]); push!(py[i], f.y[i]); end
            f = displace_particles(f, vf, 1.0)
        end
        tr = track_particles(frames, PTVParameters(); min_track_length = 4, progress = false)
        @test tr isa TrackingResult
        @test tr.n_frames == nframes
        @test issorted([t.start_frame for t in tr.trajectories])

        fullin = [all(2 .<= px[i] .<= 255) && all(2 .<= py[i] .<= 255) for i in eachindex(px)]
        recalled = falses(length(px))
        within = 0; ntotfull = 0; maxstep = 0.0; stepderr = Float64[]
        for t in tr.trajectories
            length(t) == nframes || continue
            ntotfull += 1
            best = Inf; bi = 0
            for i in eachindex(px)
                d2 = (t.x[1] - px[i][1])^2 + (t.y[1] - py[i][1])^2
                d2 < best && (best = d2; bi = i)
            end
            dev = maximum(hypot(t.x[k] - px[bi][k], t.y[k] - py[bi][k]) for k in 1:nframes)
            for k in 1:(nframes - 1)
                se = hypot((t.x[k + 1] - t.x[k]) - (px[bi][k + 1] - px[bi][k]),
                           (t.y[k + 1] - t.y[k]) - (py[bi][k + 1] - py[bi][k]))
                maxstep = max(maxstep, se)
                push!(stepderr, se)
            end
            dev <= 0.5 && (within += 1; fullin[bi] && (recalled[bi] = true))
        end
        @test count(recalled) / count(fullin) >= 0.85    # measured ≈ 0.89–0.94
        @test within / ntotfull >= 0.95                  # ≥95% within 0.5 px
        @test maxstep < 1.0                              # no identity switches (measured ≤ 0.4)
        @test median(stepderr) < 0.05                    # measured ≈ 0

        # trajectory_velocities matches finite differences of the positions.
        t1 = tr.trajectories[1]
        u, v = trajectory_velocities(t1)
        @test length(u) == length(t1)
        @test u[1] ≈ t1.x[2] - t1.x[1]
        @test u[end] ≈ t1.x[end] - t1.x[end - 1]
        n = length(t1)
        @test u[2] ≈ (t1.x[3] - t1.x[1]) / 2
        @test v[n - 1] ≈ (t1.y[n] - t1.y[n - 2]) / 2
        @test_throws ArgumentError track_particles(frames[1:1])
        @test_throws ArgumentError track_particles(frames; min_track_length = 1)
    end

    @testset "Edge cases and types" begin
        # Blank images give a valid empty PTVResult (no throw).
        blank = run_ptv(zeros(96, 96), zeros(96, 96))
        @test blank isa PTVResult
        @test length(blank.x) == 0 && length(blank.particles_a) == 0

        # Float32 images run end to end in single precision.
        rng = MersenneTwister(5)
        vf = vortex_flow(128.0, 128.0, 5.0, 0.0)
        img1, img2, = generate_synthetic_piv_pair(vf, (256, 256), 1.0;
            particle_density = 0.02, rng = rng, laser_sheet = GaussianLaserSheet(0.0, 8.0, 1.0))
        res32 = run_ptv(Float32.(img1), Float32.(img2))
        @test res32 isa PTVResult{Float32}
        @test eltype(res32.u) == Float32
        @test detect_particles(Float32.(img1)) isa Particles{Float32}

        # Dimension checks.
        @test_throws DimensionMismatch run_ptv(zeros(64, 64), zeros(32, 32))
        @test occursin("PTVResult", sprint(show, res32))
    end
end
