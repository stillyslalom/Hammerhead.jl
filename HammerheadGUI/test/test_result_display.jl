# Results display settings (no GL): physical units, color limits, the
# diverging scale, re-validation, flagged vectors in derived fields, tool
# point editing, the particle image, and saving in-memory results.

@testset "Results display settings (no GL)" begin
    C = HammerheadGUI.Controllers
    n, Ω, ctr = 21, 0.05, 10.0
    xs = collect(0.0:20.0)
    u = [-Ω * (y - ctr) for y in xs, x in xs]
    v = [Ω * (x - ctr) for y in xs, x in xs]
    rot = PIVResult(xs, copy(xs), u, v, fill(3.0, n, n), ones(n, n),
                    fill(NaN, n, n), fill(NaN, n, n), falses(n, n),
                    falses(n, n), PIVParameters(window_size = 16, overlap = (8, 8)))

    @testset "physical units toggle" begin
        scaled = with_scale(rot, PhysicalScale(2.0, 0.5, "mm", "s"))
        ex = ResultExplorer(scaled)
        @test has_scale(ex) && stored_result(ex) === scaled
        @test current_result(ex).x == physical(scaled).x
        @test current_result(ex) === current_result(ex)            # cached
        set_tool!(ex, :profile)
        C.click!(ex, 10.0, 20.0)
        set_physical_units!(ex, false)
        @test current_result(ex).u == scaled.u && current_result(ex).scale === nothing
        @test isempty(ex.tool_points[])
        @test field_label(current_result(ex), :u) == "u (px)"
        @test !has_scale(ResultExplorer(rot))
    end

    @testset "color limits: percentile and absolute" begin
        ex = ResultExplorer(rot)
        set_field!(ex, :u)
        @test color_scale_mode(ex) === :percentile
        lo, hi = current_color_limits(ex)
        set_color_percentiles!(ex, "10", "90")
        lo2, hi2 = current_color_limits(ex)
        @test lo < lo2 < hi2 < hi
        @test_throws ArgumentError set_color_percentiles!(ex, 50, 40)
        @test_throws ArgumentError set_color_percentiles!(ex, "x", 40)
        set_color_scale_mode!(ex, :absolute)                       # starts from the shown limits
        @test color_scale_mode(ex) === :absolute && current_color_limits(ex) == (lo2, hi2)
        set_color_limits!(ex; min = -1, max = 1)
        @test current_color_limits(ex) == (-1.0, 1.0)
        set_frame!(ex, 1)
        @test current_color_limits(ex) == (-1.0, 1.0)              # pinned
        set_color_scale_mode!(ex, :percentile)
        @test ex.color_min[] === nothing && current_color_limits(ex) == (lo2, hi2)
        # signed fields are centered on zero
        set_field!(ex, :vorticity)
        @test is_diverging(:vorticity) && !is_diverging(:u)
        l, h = current_color_limits(ex)
        @test l ≈ -h
    end

    @testset "re-validation and flagged vectors" begin
        bad = deepcopy(rot)
        bad.u[10, 10] = 5.0                                        # an unflagged spike
        bad.peak_ratio[3, 3] = 1.1
        ex = ResultExplorer(bad)
        @test count(current_result(ex).outliers) == 0
        @test revalidation_settings(ex).uod_threshold == bad.parameters.uod_threshold
        set_revalidation!(ex; uod_threshold = 2.0, min_peak_ratio = 1.5, replace = true)
        r = current_result(ex)
        @test r.outliers[10, 10] && r.outliers[3, 3] && count(r.outliers) == 2
        @test r.u[10, 10] ≈ u[10, 10] atol = 1e-9                   # replaced from neighbors
        @test stored_result(ex).u[10, 10] == 5.0                   # the results are unchanged
        edit_revalidation!(ex, :min_peak_ratio, "1")
        @test count(current_result(ex).outliers) == 1
        @test_throws ArgumentError edit_revalidation!(ex, :uod_threshold, "x")
        # derived fields leave flagged cells out unless asked to include them
        set_field!(ex, :vorticity)
        @test !isfinite(C.current_field_values(ex)[10, 10])
        set_include_flagged!(ex, true)
        @test C.current_field_values(ex)[10, 10] ≈ 2Ω atol = 1e-9
        set_revalidation!(ex, nothing)
        @test count(current_result(ex).outliers) == 0 && current_result(ex).u == bad.u
    end

    @testset "moving and deleting tool points" begin
        ex = ResultExplorer(rot)
        set_field!(ex, :v)
        set_tool!(ex, :profile)
        C.click!(ex, 5.0, 10.0); C.click!(ex, 15.0, 10.0)
        @test tool_point_near(ex, 15.2, 10.1, 0.5) == 2 && tool_point_near(ex, 12, 10, 0.5) === nothing
        C.click!(ex, 15.1, 10.0; tol = 0.5)                        # selects instead of restarting
        @test ex.tool_selected[] == 2 && length(ex.tool_points[]) == 2
        move_tool_point!(ex, 2, 10.0, 10.0)
        @test ex.profile_data[].values[end] ≈ 0 atol = 1e-12       # recomputed: v = 0 at x = 10
        @test delete_tool_point!(ex) && length(ex.tool_points[]) == 1 && ex.profile_data[] === nothing
        set_tool!(ex, :circulation)
        for p in ((5.0, 5.0), (15.0, 5.0), (15.0, 15.0), (5.0, 15.0), (10.0, 18.0))
            C.click!(ex, p...)
        end
        C.alt_click!(ex)
        a5 = ex.circulation_result[].area
        C.click!(ex, 10.0, 18.0; tol = 0.5)
        @test C.canvas_key!(ex, :delete)                          # delete the selected vertex
        @test length(ex.tool_points[]) == 4
        @test ex.circulation_result[].area ≈ 2Ω * 100 atol = 1e-9
        @test !(ex.circulation_result[].area ≈ a5)
        move_tool_point!(ex, 3, 15.0, 25.0 - 10.0 + 0.0)
        @test ex.circulation_result[] !== nothing
    end

    @testset "particle image field" begin
        ex = ResultExplorer(rot)
        @test !(:image in explorer_fields(ex))
        @test_throws ArgumentError set_field!(ex, :image)
        ex.image_available[] = true
        @test :image in explorer_fields(ex)
        set_field!(ex, :image)
        img = rand(Float32, 30, 40)
        ex.image[] = img
        @test C.current_field_values(ex) === img
        @test image_extent(ex) == (1.0 .* (1:40), 1.0 .* (1:30))
        lo, hi = current_color_limits(ex)
        @test lo >= minimum(img) && hi <= maximum(img)
        ex.image_available[] = false                              # falls back to a result field
        @test ex.field[] !== :image
    end

    @testset "workflow: images under results, saving in-memory results" begin
        frames = Any[imgA, imgB, imgB, imgA]
        wf = PlanarWorkflow(; files = frames)
        fill_preset!(wf.passes, :low)
        start_run!(wf; spawn = false)
        ex = wf.explorer[]
        @test results_in_memory(wf) && ex.image_available[]
        set_step!(wf, :results)
        set_field!(ex, :image)
        @test ex.image[] == Float32.(imgA)
        switch_frame!(wf, :b)
        @test ex.image[] == Float32.(imgB) && wf.frames.shown[] === :a
        set_frame!(ex, 2)
        @test ex.image[] == Float32.(imgA)                        # pair 2 is (imgB, imgA)
        mktempdir() do dir
            path = save_run_results!(wf, joinpath(dir, "later.jld2"))
            @test !results_in_memory(wf) && wf.results_path[] == path
            @test load_recipe(path) == workflow_recipe(wf)
            @test length(load_results(path)) == 2
            @test load_sources(path) == [String[], String[]]       # in-memory frames
            @test_throws ArgumentError save_run_results!(PlanarWorkflow(), joinpath(dir, "x.jld2"))
        end
    end
end
