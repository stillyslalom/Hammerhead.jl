@testset "Stereo exposure synchronization" begin
    # A small rig keeps these metadata/driver tests independent of the large
    # calibration and rendering fixtures in test_stereo.jl.
    img = rand(MersenneTwister(482), 40, 40)
    grid = DewarpGrid(x = 2.0:1.0:33.0, y = 2.0:1.0:33.0)
    cams = map((-8.0, 8.0)) do slope
        PinholeCamera([64.0 0.0 slope 0.0; 0.0 64.0 0.0 0.0;
                       0.0 0.0 1.0 64.0])
    end
    dw1, dw2 = map(cam -> ImageDewarper(cam, grid, size(img)), cams)
    params = PIVParameters(window_size = 16, overlap = 8, uod_enable = false)
    loads = Ref(0)
    source(ts) = FrameSource(2, i -> (loads[] += 1; img); timestamps = ts)
    pairs(ts) = image_pairs(source(ts))
    acquisition(p1, p2) = [(p1[1][1], p1[1][2], p2[1][1], p2[1][2])]
    check(p1, p2; kw...) = Hammerhead._check_stereo_pair_times(p1, p2; kw...)

    exact = pairs([10.0, 11.0])
    shifted = pairs([10.25, 11.25])
    near = pairs([10.0625, 11.0625])
    @test check(exact, exact; missing_timestamps = :error) === nothing
    @test check(exact, near; sync_atol = 0.0625) === nothing
    @test check(exact, near; sync_rtol = 0.0625) === nothing
    @test check(exact, near; sync_atol = 0.03125, sync_rtol = 0.03125) === nothing
    @test_throws ArgumentError check(exact, near; sync_atol = 0.03125)
    @test_throws ArgumentError check(exact, shifted) # equal dt, offset exposures
    @test loads[] == 0

    # Relative tolerance scales with the delay rather than a large clock epoch.
    epoch = 2.0^40
    @test_throws ArgumentError check(pairs([epoch, epoch + 1]),
        pairs([epoch + 0.25, epoch + 1.25]); sync_rtol = 0.1)
    @test check(pairs([epoch, epoch + 1]), pairs([epoch + 0.0625, epoch + 1.0625]);
                sync_rtol = 0.1) === nothing

    unequal_delay = pairs([10.0, 11.125])
    @test_throws ArgumentError check(exact, unequal_delay)
    @test check(exact, unequal_delay; sync_atol = 0.125) === nothing
    conflicting = [FramePair(exact[1][1], exact[1][2], 2.0)]
    @test_throws ArgumentError check(conflicting, conflicting)
    @test check(conflicting, conflicting; sync_atol = 1.0) === nothing
    # Declared dt cannot hide a reversed or zero actual exposure delay.
    for ts in ([11.0, 10.0], [10.0, 10.0])
        refs = image_pairs(source(ts))[1]
        false_delay = [FramePair(refs[1], refs[2], 1.0)]
        @test_throws ArgumentError check(false_delay, false_delay)
    end
    # Existing FramePair.dt consistency also applies without exposure metadata.
    @test_throws ArgumentError check([FramePair(img, img, 1.0)],
                                     [FramePair(img, img, 2.0)])
    for dt in (NaN, Inf, 0.0, -1.0)
        @test_throws ArgumentError check([FramePair(img, img, dt)], [(img, img)])
    end

    no_times = pairs(nothing)
    @test check(exact, no_times) === nothing
    @test check(no_times, no_times) === nothing
    @test_throws ArgumentError check(exact, no_times; missing_timestamps = :error)
    @test_throws ArgumentError check([(img, img)], [(img, img)];
                                     missing_timestamps = :error)
    partial = [(FrameRef(source([10.0, nothing]), 1), img)]
    @test check(exact, partial) === nothing
    partial_offset = [(FrameRef(source([10.25, missing]), 1), img)]
    @test_throws ArgumentError check(exact, partial_offset)
    @test_throws ArgumentError check(exact, partial; missing_timestamps = :error)
    partial_b_offset = [(img, FrameRef(source([nothing, 11.25]), 2))]
    @test_throws ArgumentError check(exact, partial_b_offset)
    missing_value = [(FrameRef(source([missing, missing]), 1), img)]
    @test check(exact, missing_value) === nothing
    for t in (NaN, Inf, -Inf, "invalid")
        invalid = [(FrameRef(source([t, nothing]), 1), img)]
        @test_throws ArgumentError check(no_times, invalid)
    end
    for bad in (-1.0, Inf, NaN)
        @test_throws ArgumentError check(exact, exact; sync_atol = bad)
        @test_throws ArgumentError check(exact, exact; sync_rtol = bad)
    end
    @test_throws ArgumentError check(exact, exact; missing_timestamps = :skip)
    @test loads[] == 0

    # Exercise each public overload, including effort keyword splitting. A bad
    # later acquisition must reject the whole run before pixels/output/hooks.
    drivers = (
        (p1, p2; kw...) -> run_piv_stereo_sequence(p1, p2, dw1, dw2, params;
                                                 progress = false, kw...),
        (p1, p2; kw...) -> run_piv_stereo_sequence(p1, p2, dw1, dw2;
                                                 progress = false, kw...),
        (p1, p2; kw...) -> run_piv_stereo_sequence(p1, p2, dw1, dw2;
                                                 effort = :low, progress = false, kw...),
        (p1, p2; kw...) -> run_piv_stereo_sequence(acquisition(p1, p2), dw1, dw2, params;
                                                 progress = false, kw...),
        (p1, p2; kw...) -> run_piv_stereo_sequence(acquisition(p1, p2), dw1, dw2;
                                                 progress = false, kw...),
        (p1, p2; kw...) -> run_piv_stereo_sequence(acquisition(p1, p2), dw1, dw2;
                                                 effort = :low, progress = false, kw...),
        (p1, p2; kw...) -> run_piv_stereo_ensemble(p1, p2, dw1, dw2, params;
                                                 progress = false, kw...),
        (p1, p2; kw...) -> run_piv_stereo_ensemble(p1, p2, dw1, dw2;
                                                 progress = false, kw...),
        (p1, p2; kw...) -> run_piv_stereo_ensemble(p1, p2, dw1, dw2;
                                                 effort = :low, progress = false, kw...))
    for driver in drivers
        loads[] = 0
        @test_throws ArgumentError driver(exact, shifted)
        @test_throws ArgumentError driver(exact, no_times; missing_timestamps = :error)
        @test loads[] == 0
        result = driver(exact, near; sync_atol = 0.0625, missing_timestamps = :error)
        @test result isa StereoPIVResult ||
              (length(result) == 1 && first(result) isa StereoPIVResult)
        @test loads[] == 4
    end
    loads[] = 0
    @test_throws ArgumentError run_piv_stereo_sequence(conflicting, conflicting, dw1, dw2, params;
                                                     progress = false)
    @test_throws ArgumentError run_piv_stereo_ensemble(conflicting, conflicting, dw1, dw2, params;
                                                     progress = false)
    @test_throws ArgumentError run_piv_stereo_ensemble(conflicting, conflicting, dw1, dw2;
                                                     effort = :low, progress = false)
    @test loads[] == 0
    loads[] = 0
    mktempdir() do dir
        path = joinpath(dir, "untouched.jld2")
        write(path, "existing output")
        hook_calls = Ref(0)
        @test_throws ArgumentError run_piv_stereo_sequence(
            [exact[1], exact[1]], [exact[1], shifted[1]], dw1, dw2, params;
            output = path, preprocess = x -> (hook_calls[] += 1; x), progress = false)
        @test read(path, String) == "existing output"
        @test loads[] == 0 && hook_calls[] == 0
    end
end
