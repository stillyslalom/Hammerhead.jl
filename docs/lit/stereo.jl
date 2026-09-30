# # Stereo particle image velocimetry (PIV) end to end
#
# Planar PIV measures two in-plane displacement components. Two
# cameras viewing the same light sheet from different angles provide enough
# information to recover the out-of-plane component. This tutorial uses a
# synthetic rig to take calibration-plate images through to a three-component
# vector field. The known geometry and displacement let you check each step:
#
# 1. photograph a calibration target ([`render_calibration_target`](@ref)),
# 2. detect and index its dots ([`detect_calibration_grid`](@ref)),
# 3. fit camera models ([`calibrate_camera`](@ref)),
# 4. dewarp onto a common world plane ([`ImageDewarper`](@ref)),
# 5. correct the plate-to-sheet misalignment ([`self_calibrate`](@ref)),
# 6. reconstruct ``(u, v, w)`` ([`run_piv_stereo`](@ref)).
#
# The concepts are explained in
# [Stereo geometry and self-calibration](../explanation/stereo.md); the
# real-data version of this workflow is the
# [stereo-rig how-to](../howto/stereo_rig.md).
#
# ## A synthetic stereo rig
#
# The rig has two pinhole cameras yawed ±20° about the vertical axis,
# looking at the world origin from 500 mm. The camera matrices and particle
# rendering live in the
# [scene code](https://github.com/stillyslalom/Hammerhead.jl/blob/main/docs/lit/helpers/advanced_tutorials.jl),
# so the steps below begin with the images those cameras would record.

using Hammerhead
using Random
using Statistics: median
include(joinpath(pkgdir(Hammerhead), "docs", "lit", "helpers", "advanced_tutorials.jl"))

image_size = (384, 384)
true_cams = (tutorial_camera(-20.0), tutorial_camera(20.0))

# The two cameras see different in-plane shifts when a particle moves in Z.
# The diagram is a schematic X–Z section: dashed rays show the first position,
# solid rays the second. Since the viewing slopes differ, two measured shifts
# can separate the common X motion from the Z motion.

using CairoMakie

let
    fig = Figure(size = (620, 430))
    ax = Axis(fig[1, 1]; xlabel = "world X", ylabel = "world Z",
              title = "Two views of one particle displacement (schematic)",
              aspect = DataAspect())
    A, B = (0.0, 0.0), (2.0, 1.5)
    for (cam, label) in (((-6.0, -8.0), "camera 1"), ((6.0, -8.0), "camera 2"))
        lines!(ax, [cam[1], A[1]], [cam[2], A[2]]; color = :gray, linestyle = :dash)
        lines!(ax, [cam[1], B[1]], [cam[2], B[2]]; color = :steelblue)
        scatter!(ax, [cam[1]], [cam[2]]; color = :black, markersize = 12)
        text!(ax, cam[1], cam[2] - 0.7; text = label, align = (:center, :top))
    end
    arrows2d!(ax, [A[1]], [A[2]], [B[1] - A[1]], [B[2] - A[2]];
              color = :darkorange)
    text!(ax, A[1] - 0.3, A[2] + 0.3; text = "A")
    text!(ax, B[1] + 0.2, B[2]; text = "B")
    xlims!(ax, -8, 8); ylims!(ax, -10, 4)
    fig
end

# ## Photograph the calibration target
#
# The target is a LaVision-style two-level dot plate: dots every 15 mm on
# each level, the back level 3 mm behind the front, plus a filled square
# marker that anchors the origin and a triangle for orientation
# diagnostics. Render it at three traverse positions (z = −3, 0, +3 mm):

zs = [-3.0, 0.0, 3.0]
target_kwargs = (spacing = 15.0, two_level = true, level_separation = 3.0,
                 marker_square = (-30.0, -7.5), marker_triangle = (-15.0, -7.5))

plates = [[render_calibration_target(cam, image_size; z, target_kwargs...)
           for z in zs] for cam in true_cams]

let
    fig = Figure(size = (720, 380))
    for (i, title) in enumerate(("camera 1 (−20°)", "camera 2 (+20°)"))
        ax = Axis(fig[1, i]; title, yreversed = true, aspect = DataAspect())
        image!(ax, plates[i][2]'; colormap = :grays)
    end
    fig
end

# The cameras see different perspectives of the same dots. That difference
# lets stereo reconstruction resolve out-of-plane motion.
#
# ## Detect the grid and calibrate
#
# [`detect_calibration_grid`](@ref) finds the dots (subpixel
# intensity-weighted centroids), indexes them on the lattice, and anchors
# the world frame to the square marker. `origin_offset = (30.0, 7.5)` says
# "the origin dot sits 30 mm right of and 7.5 mm above the marker." This
# matches the PIV Challenge case 4E convention and places both cameras in
# the same world frame. Fit a default Soloff model to each camera:

grids = map(plates) do cam_plates
    [detect_calibration_grid(img; spacing = 15.0,
         two_level = true, level_separation = 3.0,
         origin_offset = (30.0, 7.5))
     for img in cam_plates]
end
cams = [calibrate_camera(g, zs) for g in grids]
cams[1]

# Check the calibration residuals. These synthetic plates have known dot
# positions, so the residuals mainly check detection and fitting here. With
# recorded plates, inspect where errors occur before assigning a cause:
# detection, indexing, camera-model limits, refraction, and target geometry
# can all contribute (see the [stereo-rig how-to](../howto/stereo_rig.md)):

[calibration_quality(cam, gs, zs) for (cam, gs) in zip(cams, grids)]

# ## Dewarp onto a common world plane
#
# A [`DewarpGrid`](@ref) defines the measurement plane as a regular grid of
# world coordinates, shared by both cameras; an [`ImageDewarper`](@ref) per
# camera precomputes the resampling map once and reuses it for every frame.
#
# The dewarped image is indexed `out[r, c] = world (x[c], y[r], z)` in the
# order the ranges are given, so a **descending** `y` range puts world +Y at
# the top and displays the image upright. An ascending `y` would put +Y
# down the image. Displacements remain consistent because `dv * step(y)`
# carries the sign; see
# [Stereo geometry and self-calibration](../explanation/stereo.md).

grid = DewarpGrid(x = -25.0:0.2:25.0, y = 25.0:-0.2:-25.0)
dw1 = ImageDewarper(cams[1], grid, image_size)
dw2 = ImageDewarper(cams[2], grid, image_size)

let
    fig = Figure(size = (720, 380))
    for (i, (dw, plate)) in enumerate(((dw1, plates[1][2]), (dw2, plates[2][2])))
        ax = Axis(fig[1, i]; title = "camera $i, dewarped",
                  yreversed = true, aspect = DataAspect())
        image!(ax, dewarp(dw, plate)'; colormap = :grays)
    end
    fig
end

# After dewarping, the two cameras' views of the z = 0 plate align dot
# for dot: the same world point sits at the same pixel in both images.
# Zero-filled corners are regions that camera cannot see; they are recorded
# in `dw.mask` and excluded from the analysis automatically.
#
# ## Self-calibration: find the actual light sheet
#
# Calibration uses the plate position, while measurement uses the light sheet.
# Here the particle sheet is offset by 0.8 mm and tilted relative to the
# calibrated z = 0 plane. Both cameras record the *same* instants so their
# image disparity can locate the sheet:

sheet = (a = 0.8, b = 0.010, c = -0.006)   # z = a + b·X + c·Y

rng = MersenneTwister(7)
instants = [[(56 * rand(rng) - 28, 56 * rand(rng) - 28) for _ in 1:400]
            for _ in 1:3]
frames1 = [tutorial_sheet_image(true_cams[1], pts, image_size, sheet) for pts in instants]
frames2 = [tutorial_sheet_image(true_cams[2], pts, image_size, sheet) for pts in instants]

dw1c, dw2c, report = self_calibrate(frames1, frames2, dw1, dw2)
report

# Inspect the measured disparity, fitted plane, and final residual in the
# report. Self-calibration triangulates the initial disparity and adjusts
# both camera models to the fitted sheet. The final pass checks the residual
# disparity. Compare the fitted plane with the simulated sheet:

report.passes[1].plane

# ## Reconstruct three components
#
# Render a frame pair in which every particle moves by a known world
# displacement, including 0.25 mm *out of plane*:

truth = (0.30, -0.20, 0.25)   # (dx, dy, dz) in mm per frame interval
pts = [(56 * rand(rng) - 28, 56 * rand(rng) - 28) for _ in 1:400]
A1 = tutorial_sheet_image(true_cams[1], pts, image_size, sheet)
B1 = tutorial_sheet_image(true_cams[1], pts, image_size, sheet; displacement = truth)
A2 = tutorial_sheet_image(true_cams[2], pts, image_size, sheet)
B2 = tutorial_sheet_image(true_cams[2], pts, image_size, sheet; displacement = truth)

params = PIVParameters(window_size = 32, overlap = 16,
                       padding = true, apodization = :gauss)
stereo = run_piv_stereo(A1, B1, A2, B2, dw1c, dw2c, params)

# The [`StereoPIVResult`](@ref) carries `(u, v, w)` in world units on the
# shared grid. Compare the medians over valid vectors with the truth:

sel = .!stereo.mask .& .!stereo.outliers
(u = median(stereo.u[sel]), v = median(stereo.v[sel]), w = median(stereo.w[sel]),
 truth = truth)

# Compare all three medians with `truth`. The self-calibrated geometry lets
# the reconstruction account for the sheet offset, including in `w`.
#
# The in-plane field with the out-of-plane component as background:

let
    fig = Figure(size = (560, 460))
    ax = Axis(fig[1, 1]; title = "stereo field: arrows (u, v), color w (mm)",
              xlabel = "X (mm)", ylabel = "Y (mm)", aspect = DataAspect())
    hm = heatmap!(ax, stereo.x, stereo.y, permutedims(stereo.w); colormap = :viridis)
    u = copy(stereo.u); u[.!sel] .= NaN
    v = copy(stereo.v); v[.!sel] .= NaN
    plot_vector_field!(ax, stereo.x, stereo.y, u, v)
    Colorbar(fig[1, 2], hm)
    fig
end

# ## Where to go next
#
# - Apply the same steps to recorded images and interpret the calibration
#   residuals, self-calibration report, and σw/σu:
#   [Stereo on a real recording](stereo_real.md).
# - The rig-calibration checklist, including what residuals to expect
#   from physical plates: [Calibrate a real stereo rig](../howto/stereo_rig.md).
# - How disparity self-calibration works:
#   [Stereo geometry and self-calibration](../explanation/stereo.md).
# - Per-vector uncertainty propagation into ``(u, v, w)``: enable
#   `uncertainty = true` with a converged multi-pass schedule — see
#   [Uncertainty quantification](../explanation/uncertainty.md).
