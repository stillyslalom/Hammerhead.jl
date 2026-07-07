# Calibration-diagnostics views: plate-by-plate grid-detection and
# reprojection-error review, and the self-calibration report summary with
# its disparity maps embedded through the result explorer.

"""
    calibration_review(images, zs; model = :soloff, detect_kwargs...) -> Figure
    calibration_review(cr::CalibrationReview) -> Figure

Open the calibration review: each plate image with its detected dots
colored by reprojection error (fiducial markers outlined in cyan), a plane
slider, a camera-model menu (`soloff` / `pinhole` — switching refits), and
per-plane + overall fit summaries. `detect_kwargs` (`spacing`,
`origin_offset`, `two_level`, …) are forwarded to
`Hammerhead.detect_calibration_grid`.
"""
calibration_review(images, zs; model = :soloff, size = (1000, 720), detect_kwargs...) =
    calibration_review(CalibrationReview(images, zs; model, detect_kwargs...); size)

function calibration_review(cr::CalibrationReview; size = (1000, 720))
    fig = Figure(; size)
    ax = Axis(fig[1, 1]; xlabel = "x (px)", ylabel = "y (px)",
              yreversed = true, aspect = DataAspect())
    crange = Observable((0.0, 1.0))
    Colorbar(fig[1, 2]; colormap = :plasma, limits = crange,
             label = "reprojection error (px)")

    controls = GridLayout(fig[1, 3]; tellheight = false, valign = :top)
    Label(controls[1, 1], "camera model"; halign = :left, font = :bold)
    model_menu = Menu(controls[2, 1]; tellwidth = false,
                      options = [("soloff", :soloff), ("pinhole", :pinhole)])
    plane_info = lift((args...) -> plane_summary(cr), cr.plane, cr.camera)
    Label(controls[3, 1], plane_info; halign = :left, justification = :left,
          word_wrap = true, width = 200)
    fit_info = lift(_ -> fit_summary(cr), cr.camera)
    Label(controls[4, 1], fit_info; halign = :left, justification = :left,
          word_wrap = true, width = 200)
    colsize!(fig.layout, 3, Fixed(220))

    Label(fig[2, 1:3][1, 1], "plane")
    plane_slider = Slider(fig[2, 1:3][1, 2]; range = 1:nplanes(cr),
                          startvalue = cr.plane[])
    Label(fig[2, 1:3][1, 3], lift(i -> "z = $(cr.zs[i])", cr.plane))

    _sync_menu!(model_menu, cr.model)
    on(plane_slider.value) do i
        i == cr.plane[] || set_plane!(cr, i)
    end
    on(cr.plane) do i
        i == plane_slider.value[] || set_close_to!(plane_slider, i)
    end

    plots = Any[]
    function refresh!()
        for p in plots
            delete!(ax, p)
        end
        empty!(plots)
        img = cr.images[cr.plane[]]
        nr, nc = Base.size(img)
        bg = heatmap!(ax, 1:nc, 1:nr, permutedims(img); colormap = :grays)
        translate!(bg, 0, 0, -1)
        push!(plots, bg)
        g = cr.grids[cr.plane[]]
        pe = plane_errors(cr)
        if pe === nothing
            push!(plots, scatter!(ax, [Point2f(p[1], p[2]) for p in g.pixels];
                                  color = :orange, markersize = 7))
        else
            # colorrange over all planes, so scrubbing stays comparable
            cmax = maximum(i -> maximum(plane_errors(cr, i).errors), 1:nplanes(cr))
            crange[] = (0.0, max(cmax, eps()))
            push!(plots, scatter!(ax, [Point2f(p[1], p[2]) for p in pe.pixels];
                                  color = pe.errors, colormap = :plasma,
                                  colorrange = crange[], markersize = 9))
        end
        for (pos, marker) in ((g.square, :rect), (g.triangle, :utriangle))
            pos === nothing && continue
            push!(plots, scatter!(ax, [Point2f(pos[1], pos[2])]; marker,
                                  color = :transparent, strokecolor = :cyan,
                                  strokewidth = 2, markersize = 18))
        end
        return
    end
    onany((args...) -> refresh!(), cr.plane, cr.camera)
    refresh!()
    return fig
end

"""
    selfcal_review(report::SelfCalibrationReport; size = (1150, 650)) -> Figure

Open the self-calibration report: per-pass disparity/triangulation
statistics, the fitted sheet planes, convergence (with the
judge-by-signed-median reminder for real data), and the size of the rigid
correction. When the report carries disparity maps
(`self_calibrate(...; keep_disparity_maps = true)`), they open in an
embedded result explorer with the frame slider stepping through the passes.
"""
function selfcal_review(report::SelfCalibrationReport; size = (1150, 650))
    fig = Figure(; size)
    left = GridLayout(fig[1, 1]; tellheight = false, valign = :top)
    Label(left[1, 1], "self-calibration"; halign = :left, font = :bold)
    Label(left[2, 1], selfcal_summary(report); halign = :left,
          justification = :left, word_wrap = true, width = 330)
    if isempty(report.disparity_maps)
        Label(left[3, 1],
              "run self_calibrate with keep_disparity_maps = true to inspect the disparity maps";
              halign = :left, justification = :left, word_wrap = true, width = 330)
    else
        Label(left[3, 1], "disparity maps (frame = pass):"; halign = :left)
        result_explorer!(fig[1, 2], ResultExplorer(report.disparity_maps))
        colsize!(fig.layout, 1, Fixed(350))
    end
    return fig
end
