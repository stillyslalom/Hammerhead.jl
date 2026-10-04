# Workflow canvases (offscreen GLMakie) and, opt-in, the Qt window itself.
# Uses the `imgA`/`imgB` synthetic pair from runtests.jl.

@testset "workflow canvases (offscreen)" begin
    wf = PlanarWorkflow(files = Any[imgA, imgB, imgA, imgB])
    fill_preset!(wf.passes, :low)
    c = HammerheadGUI.planar_canvas(wf)
    finite(pts) = count(p -> all(isfinite, p), pts)
    @test c.frame_size[] == (128, 128)
    @test finite(c.boxes[1][]) == 0              # boxes only on the Passes step
    set_step!(wf, :passes)
    @test finite(c.boxes[1][]) == 5              # one 32 px outline (closed polygon)
    @test c.box_labels.text[] == ["32 px"]
    @test !isempty(colorbuffer(c.fig; px_per_unit = 1))

    test_pair!(wf; spawn = false)
    set_step!(wf, :test)
    n = count(i -> !(isnan(wf.test.result[].u[i]) || isnan(wf.test.result[].v[i])),
              eachindex(wf.test.result[].u))
    @test finite(c.shafts[1][]) == 2n
    set_step!(wf, :images)
    @test finite(c.shafts[1][]) == 0
    show_frame!(wf.frames, :b)                   # same size: data swap only
    @test c.frame_size[] == (128, 128)
    wf.roi[] = ROI(10:100, 20:110)
    @test finite(c.roi[1][]) == 5
    @test !isempty(colorbuffer(c.fig; px_per_unit = 1))

    start_run!(wf; spawn = false)
    rc = HammerheadGUI.results_canvas()
    HammerheadGUI.set_explorer!(rc, wf.explorer[])
    @test rc.grid_size[] == size(wf.explorer[].results[1].u)
    @test finite(rc.shafts[1][]) > 0
    set_field!(wf.explorer[], :vorticity)
    @test occursin("vorticity", lowercase(rc.colorbar.label[]))
    r = current_result(wf.explorer[])
    select_nearest!(wf.explorer[], r.x[3], r.y[2])
    @test rc.selection[1][] == [Point2f(r.x[3], r.y[2])]
    @test !isempty(colorbuffer(rc.fig; px_per_unit = 1))
    HammerheadGUI.set_explorer!(rc, nothing)
    @test finite(rc.shafts[1][]) == 0
end

@testset "Prepare overlays and gestures (offscreen)" begin
    wf = PlanarWorkflow(files = Any[imgA, imgB, imgA, imgB])
    c = HammerheadGUI.planar_canvas(wf)
    finite(pts) = count(p -> all(isfinite, p), pts)
    ps = wf.prepare
    nplots = length(c.ax.scene.plots)
    @test !isempty(colorbuffer(c.fig; px_per_unit = 1))

    # Overlays only change their inputs: the plot count never changes.
    set_step!(wf, :prepare)
    add_step!(ps.preview, :invert_image)
    ps.show_processed[] = true                       # processed frame shown
    @test c.frame[3][] ≈ permutedims(ps.preview.processed[])
    canvas_click!(wf, 64.0, 64.0)
    @test finite(c.probe_box[1][]) == 5
    set_prepare_page!(wf, :mask)
    @test finite(c.probe_box[1][]) == 0              # editing overlays: own page only
    @test c.frame[3][] ≈ permutedims(Float32.(imgA)) # raw frame off the Preprocess page
    for (x, y) in ((20.0, 10.0), (60.0, 10.0), (60.0, 50.0))
        canvas_click!(wf, x, y)
    end
    @test finite(c.active_points[1][]) == 3 && finite(c.active_line[1][]) == 3
    canvas_alt_click!(wf)
    @test finite(c.polygons[1][]) == 4 && finite(c.active_points[1][]) == 0
    @test count(isfinite, c.mask[3][]) == count(wf.mask[])
    canvas_click!(wf, 50.0, 20.0)                    # select: highlighted
    @test all(==(RGBAf(HammerheadGUI.SELECTED_COLOR)), c.polygons.color[][1:4])
    set_prepare_page!(wf, :roi)
    @test finite(c.polygons[1][]) == 0
    canvas_click!(wf, 10.0, 20.0)
    @test c.roi_corner[1][] == [Point2f(10, 20)]
    canvas_click!(wf, 100.0, 110.0)
    @test finite(c.roi[1][]) == 5 && finite(c.roi_corner[1][]) == 0
    set_prepare_page!(wf, :scale)
    canvas_click!(wf, 10.0, 20.0); canvas_click!(wf, 10.0, 70.0)
    @test finite(c.scale_line[1][]) == 2 && c.scale_label.text[] == ["50.0 px"]
    set_step!(wf, :passes)                           # mask and ROI stay; tools hide
    @test finite(c.scale_line[1][]) == 0 && finite(c.roi[1][]) == 5
    @test length(c.ax.scene.plots) == nplots
    @test !isempty(colorbuffer(c.fig; px_per_unit = 1))

    # Real Makie mouse and key events on the figure reach the controller
    # through the canvas's interaction (fast clicks arrive as double clicks).
    set_step!(wf, :prepare)
    set_prepare_page!(wf, :mask)
    clear_polygons!(ps.mask[])
    colorbuffer(c.fig; px_per_unit = 1)              # lay out the axis
    scene, ev = c.ax.scene, events(c.fig)
    function mouse_click(x, y, button = Mouse.left)
        p = Makie.project(scene, Point2f(x, y)) .+ scene.viewport[].origin
        ev.mouseposition[] = (Float64(p[1]), Float64(p[2]))
        ev.mousebutton[] = Makie.MouseButtonEvent(button, Mouse.press)
        ev.mousebutton[] = Makie.MouseButtonEvent(button, Mouse.release)
    end
    key(k) = (ev.keyboardbutton[] = Makie.KeyEvent(k, Keyboard.press))
    for (x, y) in ((80.0, 80.0), (110.0, 80.0), (110.0, 110.0), (80.0, 110.0))
        mouse_click(x, y)
    end
    verts = ps.mask[].active[]
    @test length(verts) == 4 && all(isapprox.(verts[2], (110.0, 80.0); atol = 0.5))
    key(Keyboard.backspace)
    @test length(ps.mask[].active[]) == 3
    mouse_click(90.0, 90.0, Mouse.right)
    @test length(ps.mask[].polygons[]) == 1 && wf.mask[][90, 100]
    mouse_click(100.0, 90.0)
    @test ps.mask[].selected[] == 1
    key(Keyboard.delete)
    @test wf.mask[] === nothing
    limits = c.ax.finallimits[]
    mouse_click(64.0, 64.0, Mouse.right)              # nothing to use: the Axis keeps it
    @test isempty(ps.mask[].active[]) && c.ax.finallimits[] == limits
    @test length(c.ax.scene.plots) == nplots
end

@testset "Results tools and profile row (offscreen)" begin
    wf = PlanarWorkflow(files = Any[imgA, imgB, imgA, imgB])
    fill_preset!(wf.passes, :low)
    start_run!(wf; spawn = false)
    ex = wf.explorer[]
    rc = HammerheadGUI.results_canvas()
    HammerheadGUI.set_explorer!(rc, ex)
    finite(pts) = count(p -> all(isfinite, p), pts)
    allplots(scene) = length(scene.plots) + sum(allplots, scene.children; init = 0)
    nmain, nall = length(rc.ax.scene.plots), allplots(rc.fig.scene)
    @test !isempty(colorbuffer(rc.fig; px_per_unit = 1))  # lay out the figure
    @test !rc.profile_shown[] && !rc.profile_ax.scene.visible[] && !rc.profile_ax.blockscene.visible[]
    h0 = rc.ax.scene.viewport[].widths[2]

    scene, ev = rc.ax.scene, events(rc.fig)
    function mouse_click(x, y, button = Mouse.left)
        p = Makie.project(scene, Point2f(x, y)) .+ scene.viewport[].origin
        ev.mouseposition[] = (Float64(p[1]), Float64(p[2]))
        ev.mousebutton[] = Makie.MouseButtonEvent(button, Mouse.press)
        ev.mousebutton[] = Makie.MouseButtonEvent(button, Mouse.release)
    end
    key(k) = (ev.keyboardbutton[] = Makie.KeyEvent(k, Keyboard.press))
    r = current_result(ex)
    x0, x1 = extrema(r.x); y0, y1 = extrema(r.y)
    xm, ym = (x0 + x1) / 2, (y0 + y1) / 2

    # inspect: a click selects the nearest vector
    mouse_click(r.x[3], r.y[2])
    @test ex.selection[] == CartesianIndex(2, 3)

    # profile: two real clicks draw the line and open the profile row
    set_tool!(ex, :profile)
    mouse_click(x0 + 5, ym); mouse_click(x1 - 5, ym)
    @test length(ex.tool_points[]) == 2 && all(isapprox.(ex.tool_points[][2], (x1 - 5, ym); atol = 1))
    @test finite(rc.tool_line[1][]) == 2 && finite(rc.tool_points[1][]) == 2
    @test rc.profile_shown[] && rc.profile_ax.scene.visible[] && rc.profile_legend.blockscene.visible[]
    @test length(rc.profile_lines[1][1][]) == 100
    @test occursin("distance along the line", rc.profile_ax.xlabel[])
    @test !isempty(colorbuffer(rc.fig; px_per_unit = 1))
    @test rc.ax.scene.viewport[].widths[2] < h0       # the row takes space from the field
    @test rc.profile_ax.scene.viewport[].widths[2] >= 100
    @test length(rc.ax.scene.plots) == nmain && allplots(rc.fig.scene) == nall

    # Escape clears the line and collapses the row
    key(Keyboard.escape)
    @test isempty(ex.tool_points[]) && finite(rc.tool_line[1][]) == 0
    @test !rc.profile_shown[] && !rc.profile_ax.scene.visible[] && !rc.profile_legend.blockscene.visible[]
    @test !isempty(colorbuffer(rc.fig; px_per_unit = 1))
    @test rc.ax.scene.viewport[].widths[2] == h0

    # circulation: clicks add vertices, a right click closes the contour
    set_tool!(ex, :circulation)
    for (x, y) in ((x0 + 5, y0 + 5), (x1 - 5, y0 + 5), (x1 - 5, y1 - 5), (x0 + 5, y1 - 5))
        mouse_click(x, y)
    end
    @test finite(rc.tool_line[1][]) == 4             # open path while drawing
    mouse_click(xm, ym, Mouse.right)
    @test ex.circulation_result[] !== nothing
    @test finite(rc.tool_line[1][]) == 5 && rc.tool_line[1][][1] == rc.tool_line[1][][end]
    @test occursin("Γ (line)", tool_summary(ex)) && !rc.profile_shown[]
    @test !isempty(colorbuffer(rc.fig; px_per_unit = 1))
    set_tool!(ex, :inspect)                           # switching tools clears the path
    @test finite(rc.tool_line[1][]) == 0 && finite(rc.tool_points[1][]) == 0
    limits = rc.ax.finallimits[]
    mouse_click(xm, ym, Mouse.right)                  # not used: the Axis keeps it
    @test rc.ax.finallimits[] == limits
    @test length(rc.ax.scene.plots) == nmain && allplots(rc.fig.scene) == nall

    # a frame switch clears a profile, and the row with it
    set_tool!(ex, :profile)
    HammerheadGUI.Controllers.click!(ex, x0 + 5, ym)
    HammerheadGUI.Controllers.click!(ex, x1 - 5, ym)
    @test rc.profile_shown[]
    set_frame!(ex, 2)
    @test !rc.profile_shown[] && finite(rc.tool_line[1][]) == 0
    @test length(rc.ax.scene.plots) == nmain && allplots(rc.fig.scene) == nall
end

# Opens a real Qt window: needs a display and an OpenGL 3.3 context. It runs in
# a separate Julia process: on some drivers (seen with AMD on Windows) a Qt GL
# context crashes once GLFW/GLMakie has created a context in the same process,
# which the offscreen tests above do.
if get(ENV, "HAMMERHEADGUI_QT_TESTS", "") == "true"
    for (name, file) in (("planar_window", "qt_window.jl"), ("stereo_window", "qt_stereo_window.jl"))
        @testset "$name (Qt, separate process)" begin
            script = joinpath(@__DIR__, file)
            cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -t 4 $script`
            @test success(pipeline(cmd; stdout, stderr))
        end
    end
else
    @info "Skipping the Qt window tests; set HAMMERHEADGUI_QT_TESTS=true on a machine with a display to run them."
end
