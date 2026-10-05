# The stereo workflow canvas (offscreen GLMakie): what each step shows, the
# overlays on the dewarped grid, and Makie clicks reaching the controllers.
# The Qt stereo window itself runs, opt-in, from test_planar_window.jl.

@testset "stereo canvas (offscreen)" begin
    fx = stereo_rig(acquisitions = 2)
    wf = StereoWorkflow(; files1 = fx.files1, files2 = fx.files2)
    cal = wf.calibration
    c = HammerheadGUI.stereo_canvas(wf)
    finite(pts) = count(p -> all(isfinite, p), pts)
    nplots = length(c.ax.scene.plots)

    # Images: the shown camera's raw frame
    @test c.space[] === :camera && c.frame_size[] == (512, 512)
    @test c.frame[3][] ≈ permutedims(Float32.(fx.files1[1]))
    set_camera!(wf, 2)
    @test c.frame[3][] ≈ permutedims(Float32.(fx.files2[1]))
    @test startswith(c.ax.title[], "camera 2 · frame A")
    set_camera!(wf, 1)
    @test !isempty(colorbuffer(c.fig; px_per_unit = 1))

    # Calibration: the plate with its dots and magnified residuals
    set_step!(wf, :calibration)
    @test c.space[] === :plate && c.frame_size[] === nothing   # no plates yet
    add_fixture_plates!(cal, fx)
    @test c.frame_size[] == (512, 512)                         # in-memory plate shown before the fit
    @test finite(c.dots[1][]) == 0
    fit_calibration!(cal)
    cr = cal.reviews[1][]
    @test c.shown[] === cr.images[1] && finite(c.dots[1][]) == length(cr.grids[1].pixels)
    @test finite(c.residuals[1][]) == 2 * length(cr.grids[1].pixels)
    @test finite(c.square[1][]) == 1 && finite(c.triangle[1][]) == 1
    res = plane_residuals(cr)
    g = c.residual_gain[]
    @test g >= 1 && c.residuals[1][][2] ≈ Point2f(res.pixels[1] .+ g .* res.residuals[1])
    @test occursin("residual arrows ×", c.ax.title[])
    set_plane!(cr, 3)
    @test c.shown[] === cr.images[3] && occursin("plate 3 of 3", c.ax.title[])
    set_camera!(wf, 2)
    @test c.shown[] === cal.reviews[2][].images[cal.reviews[2][].plane[]]
    set_camera!(wf, 1)
    @test !isempty(colorbuffer(c.fig; px_per_unit = 1))

    # Prepare: the dewarped frame with the out-of-view union shaded
    dws = cal.dewarpers[]
    gsz = size(dws[1].grid)
    set_step!(wf, :prepare)
    @test finite(c.dots[1][]) == 0 && finite(c.residuals[1][]) == 0
    @test c.space[] === :grid && c.frame_size[] == gsz
    # the whole dewarped frame, not the plate's limits (plot data applies lazily)
    @test c.ax.targetlimits[].widths ≈ Vec2f(gsz[2], gsz[1])
    @test c.frame[3][] ≈ permutedims(wf.dewarped[][1])
    @test count(isfinite, c.out_of_view[3][]) == count(out_of_view(wf))
    add_step!(wf.prepare.preview, :highpass_filter)
    wf.prepare.show_processed[] = true
    @test c.frame[3][] ≈ permutedims(Float32.(wf.prepare.preview.processed[]))
    @test canvas_click!(wf, 120.0, 120.0)
    @test finite(c.probe_box[1][]) == 5
    set_prepare_page!(wf, :mask)
    @test finite(c.probe_box[1][]) == 0
    @test c.frame[3][] ≈ permutedims(wf.dewarped[][1])           # raw dewarped off Preprocess

    # real Makie clicks and keys on the figure reach the mask editor
    colorbuffer(c.fig; px_per_unit = 1)                          # lay out the axis
    scene, ev = c.ax.scene, events(c.fig)
    function mouse_click(x, y, button = Mouse.left)
        p = Makie.project(scene, Point2f(x, y)) .+ scene.viewport[].origin
        ev.mouseposition[] = (Float64(p[1]), Float64(p[2]))
        ev.mousebutton[] = Makie.MouseButtonEvent(button, Mouse.press)
        ev.mousebutton[] = Makie.MouseButtonEvent(button, Mouse.release)
    end
    key(k) = (ev.keyboardbutton[] = Makie.KeyEvent(k, Keyboard.press))
    me = wf.prepare.mask[]
    for (x, y) in ((30.0, 30.0), (90.0, 30.0), (90.0, 90.0), (30.0, 90.0))
        mouse_click(x, y)
    end
    @test length(me.active[]) == 4 && all(isapprox.(me.active[][2], (90.0, 30.0); atol = 0.5))
    @test finite(c.active_points[1][]) == 4
    key(Keyboard.backspace)
    @test length(me.active[]) == 3
    mouse_click(60.0, 60.0, Mouse.right)
    @test length(me.polygons[]) == 1 && size(wf.mask[]) == gsz && wf.mask[][40, 80]
    @test finite(c.polygons[1][]) == 4
    @test count(isfinite, c.mask[3][]) == count(wf.mask[])

    # Passes: window outlines and the test's in-plane vectors in dewarped px; Run too
    fill_preset!(wf.passes, :low)
    set_step!(wf, :passes)
    @test finite(c.polygons[1][]) == 0 && c.box_labels.text[] == ["32 px"]
    test_pair!(wf; spawn = false)
    r = wf.test.result[]
    @test r isa StereoPIVResult
    d = HammerheadGUI.grid_vector_data(r, dws[1].grid)
    @test finite(c.shafts[1][]) == 2 * length(d.x) > 0
    # vector nodes land on the per-camera grid, displacement in dewarped px
    on_node(a, nodes) = all(x -> minimum(abs.(nodes .- x)) < 1e-6, a)
    @test on_node(d.x, r.cam1.x) && on_node(d.y, r.cam1.y)
    @test median(d.u) ≈ fx.disp[1] / step(dws[1].grid.x) rtol = 0.1
    @test median(d.v) ≈ fx.disp[2] / step(dws[1].grid.y) rtol = 0.1     # descending y: up
    @test count(isfinite, c.mask[3][]) == count(wf.mask[])               # the mask stays
    start_run!(wf; spawn = false)
    set_step!(wf, :run)
    @test finite(c.shafts[1][]) > 0

    # Calibration › Self-calibration: the dewarped frame under a disparity map,
    # one arrow scale for every pass
    set_step!(wf, :calibration)
    set_calibration_page!(cal, :selfcal)
    @test c.space[] === :grid && occursin("self-calibrate to see", c.ax.title[])
    @test finite(c.shafts[1][]) == 0
    start_selfcal!(wf; pairs = 1)
    m1 = disparity_map(cal)
    @test m1 isa PIVResult && occursin("pass 1 of", c.ax.title[])
    @test finite(c.shafts[1][]) == 2 * length(HammerheadGUI.vector_data(m1).x) > 0
    @test finite(c.dots[1][]) == 0
    seg_length(pts) = median([hypot((pts[2k] - pts[2k - 1])...) for k in 1:length(pts) ÷ 2 if all(isfinite, pts[2k])])
    long1 = seg_length(c.shafts[1][])
    set_disparity_pass!(cal, 99)
    np = length(cal.selfcal[].report.passes)
    @test np > 1 && cal.disparity_pass[] == np && occursin("pass $np of $np", c.ax.title[])
    @test seg_length(c.shafts[1][]) < long1                  # corrected: shorter arrows
    set_calibration_page!(cal, "plates")
    @test c.space[] === :plate && finite(c.shafts[1][]) == 0
    @test_throws ArgumentError set_calibration_page!(cal, :prepare)
    set_step!(wf, :images)
    @test finite(c.shafts[1][]) == 0 && c.space[] === :camera
    @test count(isfinite, c.mask[3][]) == 0                              # no mask on camera frames
    @test length(c.ax.scene.plots) == nplots
    @test !isempty(colorbuffer(c.fig; px_per_unit = 1))

    # Results: stereo results on world axes (+Y up)
    rc = HammerheadGUI.results_canvas()
    HammerheadGUI.set_explorer!(rc, wf.explorer[])
    @test !rc.ax.yreversed[] && rc.ax.xlabel[] == "x (world units)"
    set_field!(wf.explorer[], :w)
    @test occursin("w", rc.colorbar.label[])
    @test !isempty(colorbuffer(rc.fig; px_per_unit = 1))
    HammerheadGUI.set_explorer!(rc, ResultExplorer(r_unc))               # planar again
    @test rc.ax.yreversed[] && rc.ax.xlabel[] == "x (px)"
end
