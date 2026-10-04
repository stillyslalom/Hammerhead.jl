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

# Opens a real Qt window: needs a display and an OpenGL 3.3 context. It runs in
# a separate Julia process: on some drivers (seen with AMD on Windows) a Qt GL
# context crashes once GLFW/GLMakie has created a context in the same process,
# which the offscreen tests above do.
if get(ENV, "HAMMERHEADGUI_QT_TESTS", "") == "true"
    @testset "planar_window (Qt, separate process)" begin
        script = joinpath(@__DIR__, "qt_window.jl")
        cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) -t 4 $script`
        @test success(pipeline(cmd; stdout, stderr))
    end
else
    @info "Skipping the Qt window test; set HAMMERHEADGUI_QT_TESTS=true on a machine with a display to run it."
end
