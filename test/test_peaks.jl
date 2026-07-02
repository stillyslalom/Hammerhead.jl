using Hammerhead
using Test
using Random
using Statistics

@testset "Multi-peak detection & substitution" begin
    @testset "find_peaks" begin
        R = zeros(32, 32)
        add_particle!(R, (16.0, 16.0), 3.0)          # primary, height 1
        R[8, 24] = 0.6                                # secondary
        R[26, 6] = 0.3                                # tertiary
        peaks = find_peaks(R, 3)
        @test length(peaks) == 3
        @test peaks[1].location == (16, 16) && peaks[1].value ≈ 1.0
        @test peaks[2].location == (8, 24) && peaks[2].value == 0.6
        @test peaks[3].location == (26, 6) && peaks[3].value == 0.3

        # The primary's shoulder is excluded; peaks beyond the list are not
        # invented when nothing positive remains.
        only2 = zeros(16, 16)
        only2[8, 8] = 1.0
        only2[8, 9] = 0.9    # inside the exclusion box
        @test length(find_peaks(only2, 3)) == 1
        @test length(find_peaks(only2, 3; exclusion_radius = 0)) == 3 ||
              length(find_peaks(only2, 3; exclusion_radius = 0)) == 2

        # Consistency with the documented peak-ratio semantics.
        p2 = find_peaks(R, 2)
        @test peaks[1].value / p2[2].value ≈ calculate_peak_ratio(R, p2[1].location)

        @test_throws ArgumentError find_peaks(R, 0)
    end

    @testset "n_peaks parameter" begin
        @test PIVParameters().n_peaks == 3
        @test PIVParameters(n_peaks = 1).n_peaks == 1
        @test_throws ArgumentError PIVParameters(n_peaks = 0)
    end

    @testset "peak substitution end-to-end" begin
        # Uniform flow, but one interrogation window also contains a strong
        # static reflection (identical bright pattern in both frames): its
        # zero-displacement peak beats the true one, UOD flags the vector,
        # and the secondary peak recovers the real displacement.
        # replace_outliers = false distinguishes substitution (measured
        # value, unflagged) from median replacement.
        rng = MersenneTwister(23)
        n = 128
        dv, du = 4.0, 5.0
        positions = [(rand(rng) * (n + 20) - 10, rand(rng) * (n + 20) - 10) for _ in 1:250]
        imgA, imgB = particle_pair((n, n), positions, dv, du)
        # Densify true seeding inside the reflection window so the true peak
        # clearly beats reflection×particle cross-talk (but not the reflection).
        for _ in 1:10
            p = (66 + rand(rng) * 26, 66 + rand(rng) * 26)
            add_particle!(imgA, p, 3.0)
            add_particle!(imgB, (p[1] + dv, p[2] + du), 3.0)
        end
        for p in [(72.0, 74.0), (78.0, 86.0), (86.0, 71.0), (90.0, 84.0),
                  (74.0, 90.0), (82.0, 78.0)]
            for img in (imgA, imgB)
                add_particle!(img, p, 3.0)
                add_particle!(img, p, 3.0)   # doubled amplitude, static
            end
        end

        params_sub = PIVParameters(window_size = 32, overlap = 0,
                                   n_peaks = 3, replace_outliers = false)
        params_off = PIVParameters(window_size = 32, overlap = 0,
                                   n_peaks = 1, replace_outliers = false)
        r_off = run_piv(imgA, imgB, params_off)
        r_sub = run_piv(imgA, imgB, params_sub)
        # Without substitution the reflection window reads ≈ (0, 0), flagged;
        # with it, the true displacement returns (the reflection shoulder
        # biases the secondary's subpixel fit, hence the ~1 px tolerance).
        wi = findfirst(i -> abs(r_off.u[i]) < 1 && abs(r_off.v[i]) < 1,
                       eachindex(r_off.u))
        @test wi !== nothing               # the reflection peak did win somewhere
        @test r_off.outliers[wi]
        @test abs(r_sub.u[wi] - du) < 1.0  # substituted with the true peak
        @test abs(r_sub.v[wi] - dv) < 1.0
        @test !r_sub.outliers[wi]          # accepted alternatives are unflagged
        @test sum(r_sub.outliers) < sum(r_off.outliers)

        # Substitution leaves clean fields bitwise unchanged, threaded or not.
        imgC, imgD = particle_pair((n, n), positions, dv, du)
        pc = PIVParameters(window_size = 32, overlap = 16)
        r1 = run_piv(imgC, imgD, pc; threaded = false)
        r3 = run_piv(imgC, imgD, pc; threaded = true)
        @test isequal(r1.u, r3.u) && isequal(r1.peak_ratio, r3.peak_ratio)
    end

    @testset "peak_locking" begin
        rng = MersenneTwister(3)
        locked = 5.0 .+ 0.03 .* randn(rng, 4000)          # fractions pile on 0
        uniform = 5.0 .+ (rand(rng, 4000) .- 0.5)         # uniform fractions
        @test peak_locking(locked).index > 0.8
        @test abs(peak_locking(uniform).index) < 0.15
        @test sum(peak_locking(uniform).counts) == 4000
        with_nan = [1.0, 2.5, NaN]
        @test sum(peak_locking(with_nan).counts) == 2
        @test isnan(peak_locking([NaN]).index)
        @test_throws ArgumentError peak_locking(locked; nbins = 2)
    end
end
