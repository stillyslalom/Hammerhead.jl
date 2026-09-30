# # Your first vector field
#
# This tutorial follows one particle-image pair from a single measured
# displacement to a vector field. You will inspect the correlation peak,
# compare two interrogation schedules, mark an image region that should be
# excluded, and convert the result to physical units.
#
# ## Make an image pair
#
# Particle image velocimetry (PIV) measures how a pattern of tracer particles
# moves between two exposures. We use a smooth synthetic vortex so we can later
# compare the measured displacements with known particle motion. The code
# below makes two 256 × 256 pixel frames separated by one frame interval.

using Hammerhead
using Hammerhead.SyntheticData
using Random
using CairoMakie
using Statistics: median

function flow(x, y, z, t)
    dx, dy = x - 128.0, y - 128.0
    rate = 3.0 / sqrt(dx^2 + dy^2 + 30.0^2)
    return (-rate * dy, rate * dx, 0.0)
end
imgA, imgB, _, _ = generate_synthetic_piv_pair(
    flow, (256, 256), 1.0;
    particle_density = 0.05,
    background_noise = 0.03,
    z_range = (-1.0, 1.0),
    rng = MersenneTwister(42),
)

# You can use `load_image("frame_0001.tif")` and the next recorded frame
# in place of these arrays. Each image is a matrix: rows run downward and
# columns run rightward.

let
    fig = Figure(size = (720, 350))
    for (k, (img, name)) in enumerate(((imgA, "frame A"), (imgB, "frame B")))
        ax = Axis(fig[1, k]; title = name, yreversed = true, aspect = DataAspect())
        image!(ax, img'; colormap = :grays)
    end
    fig
end

# ## What does one vector measure?
#
# [`run_piv`](@ref) divides the pair into *interrogation windows*. Within each
# window it finds the shift that best aligns the particle patterns. That
# gives one displacement for a group of particles, not one path per particle.
# The first pass below uses 32 × 32 pixel windows at 50% overlap. We retain
# its correlation planes so we can inspect one; turn this option off for a
# large dataset because it stores a matrix for every vector.

basic = run_piv(imgA, imgB,
    PIVParameters(window_size = 32, overlap = 16,
                  keep_correlation_planes = true))

# Choose a window in the upper-right part of the vortex. Its grid coordinates
# are in pixels, with `x` along columns and `y` along rows. The window starts
# half a window width before its center in each direction.

i, j = 5, 9
row0 = round(Int, basic.y[i] - 15.5)
col0 = round(Int, basic.x[j] - 15.5)
rows, cols = row0:(row0 + 31), col0:(col0 + 31)
windowA, windowB = @view(imgA[rows, cols]), @view(imgB[rows, cols])

# Predict the sign before inspecting the result: particles above and right
# of the vortex center move mostly rightward, so `u` should be positive.
# The two windows show the local pattern that contributes to this vector.

let
    fig = Figure(size = (650, 330))
    for (k, (win, name)) in enumerate(((windowA, "window A"), (windowB, "window B")))
        ax = Axis(fig[1, k]; title = name, yreversed = true, aspect = DataAspect(),
                  xlabel = "column in window", ylabel = "row in window")
        image!(ax, win'; colormap = :grays)
    end
    fig
end

# The retained plane scores possible shifts. Its center is zero shift; the
# brightest peak is the best alignment. Correlating the same two windows
# with the exported low-level [`correlate`](@ref) function also gives the
# integer peak and its three-point Gaussian subpixel fit. The plane retained
# by `run_piv` and the one returned by `correlate` use the same settings.

plane = basic.correlation_planes[i, j]
probe = correlate(CrossCorrelator{Float64}((32, 32)), windowA, windowB)
center_lag = size(plane) .÷ 2 .+ 1
(integer_peak = probe.peakloc,
 refined_peak = probe.refined_peakloc,
 zero_lag = center_lag,
 measured = (basic.u[i, j], basic.v[i, j]),
 probe = (probe.du, probe.dv))

# Read horizontal lag from the peak's column offset and vertical lag from
# its row offset. Subpixel fitting places the estimate between matrix cells.
# A particle at `(row, col)` in A found at `(row + v, col + u)` in B has
# positive `u` to the right and positive `v` downward.

let
    lags = (1:size(plane, 1)) .- center_lag[1]
    fig = Figure(size = (430, 380))
    ax = Axis(fig[1, 1]; title = "correlation of the selected windows",
              xlabel = "horizontal shift (px)", ylabel = "vertical shift (px)",
              aspect = DataAspect(), yreversed = true)
    heatmap!(ax, lags, lags, plane'; colormap = :viridis)
    scatter!(ax, [probe.du], [probe.dv]; color = :red, markersize = 15)
    fig
end

# Zoom in on the three samples around the peak along the horizontal axis.
# The default subpixel method fits a Gaussian to their heights. Its maximum
# falls between the sampled shifts, giving the fractional part of `u`.

peak_row, peak_col = probe.peakloc
neighbor_cols = (peak_col - 1):(peak_col + 1)
heights = plane[peak_row, neighbor_cols]
log_heights = log.(heights)
a = (log_heights[1] + log_heights[3]) / 2 - log_heights[2]
b = (log_heights[3] - log_heights[1]) / 2
subpixel_offset = -b / (2a)
δ = -1.0:0.02:1.0
lag_x = peak_col - center_lag[2]

let
    fig = Figure(size = (470, 320))
    ax = Axis(fig[1, 1]; title = "horizontal subpixel peak fit",
              xlabel = "horizontal shift (px)", ylabel = "correlation")
    scatter!(ax, neighbor_cols .- center_lag[2], heights;
             color = :black, markersize = 12, label = "sampled shifts")
    lines!(ax, lag_x .+ δ,
           exp.(a .* δ.^2 .+ b .* δ .+ log_heights[2]);
           color = :blue, label = "Gaussian fit")
    scatter!(ax, [lag_x + subpixel_offset],
             [exp(log_heights[2] - b^2 / (4a))];
             color = :red, markersize = 14, label = "subpixel maximum")
    axislegend(ax; position = :lb)
    fig
end

# The peak needs a particle pattern. A constant pair contains no shift
# information: mean subtraction leaves a flat correlation plane. Its first
# matrix entry is still an `argmax`, but that index is not a measurement.
# Inspect image quality before interpreting an isolated vector; the
# [image-quality guide](../howto/image_quality.md) covers recorded images.

blank = fill(0.5, 32, 32)
blank_probe = correlate(CrossCorrelator{Float64}((32, 32)), blank, blank)
(useful_peak = maximum(plane), blank_peak = maximum(blank_probe.correlation),
 blank_plane_range = extrema(blank_probe.correlation))

let
    fig = Figure(size = (730, 340))
    for (k, (R, title)) in enumerate(((plane, "particle pattern"),
                                     (blank_probe.correlation, "constant window")))
        ax = Axis(fig[1, k]; title, xlabel = "horizontal shift (px)",
                  ylabel = "vertical shift (px)", aspect = DataAspect(),
                  yreversed = true)
        heatmap!(ax, (1:size(R, 2)) .- center_lag[2],
                 (1:size(R, 1)) .- center_lag[1], R';
                 colormap = :viridis, colorrange = (0, maximum(plane)))
    end
    fig
end

# ## From one vector to a field
#
# Plot the first-pass field. Neighboring 32 px windows overlap by 16 px, so
# vector centers are 16 px apart. Overlap samples the field more often; it
# does not make each measurement sensitive to features smaller than its
# 32 px particle-sampling window.

plot_vector_field(basic)

# A multi-pass schedule begins with 64 px windows and uses each measured
# field to deform the images for the next pass. Later 32 and 16 px windows
# measure the remaining shift with more local detail. Repeating 16 px gives
# the final deformation a convergence sweep before estimating uncertainty.
# Predict where the 16 px field will show more variation than the basic one.

passes = multipass_parameters([64, 32, 16, 16];
    padding = true, apodization = :gauss, uncertainty = true)
fine = run_piv(imgA, imgB, passes)

# Use the same color range and arrow length scale in both panels. Each
# panel shows displacement magnitude below the arrows. The finer grid
# resolves changes over shorter distances, especially near the vortex core.

let
    fig = Figure(size = (850, 390))
    for (k, (r, title)) in enumerate(((basic, "32 px, one pass"),
                                     (fine, "16 px, multi-pass")))
        ax = Axis(fig[1, k]; title, yreversed = true, aspect = DataAspect(),
                  xlabel = "x (px)", ylabel = "y (px)")
        speed = hypot.(r.u, r.v)
        heatmap!(ax, r.x, r.y, speed'; colorrange = (0, 4), colormap = :viridis)
        plot_vector_field!(ax, r; stride = k == 1 ? 1 : 2, lengthscale = 2)
    end
    fig
end

# Both fields report displacement in pixels per frame interval. The
# difference in grid density reflects the final window size and overlap;
# compare local changes and flagged vectors before deciding which resolution
# serves your measurement.

# ## Validate a difficult region
#
# Add a static bright patch to both frames, as a reflection might appear in
# recorded images. A zero-shift peak inside it can disagree with neighboring
# flow vectors. Universal outlier detection flags vectors that differ from
# nearby measurements. With `replace_outliers = true`, the returned `u` and
# `v` at flagged positions hold local-median replacements, while
# `outliers` retains the flag. Replacements are estimates, not new particle
# measurements.

imgA_refl, imgB_refl = copy(imgA), copy(imgB)
for img in (imgA_refl, imgB_refl)
    img[97:144, 41:104] .= 1.0
end
with_reflection = run_piv(imgA_refl, imgB_refl, passes)
(flagged = count(with_reflection.outliers),
 affected_region = count(with_reflection.outliers[:, 1:7]))

# Overlay the vectors on the image. Orange arrows are flagged replacements;
# cyan arrows are retained measurements. A partly covered window can still
# be biased without being flagged.

let
    fig = Figure(size = (520, 470))
    ax = Axis(fig[1, 1]; title = "reflection and validated vectors",
              yreversed = true, aspect = DataAspect())
    image!(ax, imgA_refl'; colormap = :grays)
    plot_vector_field!(ax, with_reflection; stride = 2,
                       color = :cyan, replaced_color = :orangered,
                       lengthscale = 2)
    fig
end

# Mask the known reflection. Windows whose masked fraction reaches the
# configured threshold are excluded from the field, validation, and
# statistics. `mask` and `outliers` remain distinct in the result.

mask = falses(size(imgA))
mask[97:144, 41:104] .= true
masked = run_piv(imgA_refl, imgB_refl, passes; mask)
(masked_windows = count(masked.mask), remaining_outliers = count(masked.outliers))

# ## Read uncertainty and convert units
#
# `uncertainty = true` estimates the random displacement uncertainty of
# each final-pass correlation [Wieneke2015](@cite). It is stored separately
# from `outliers`: a vector can have an uncertainty estimate without being
# flagged, and a replacement remains flagged. The estimate does not measure
# systematic errors such as calibration bias. Inspect valid, finite values:

valid = .!(fine.mask .| fine.outliers)
σu = filter(isfinite, fine.uncertainty_u[valid])
(valid_vectors = count(valid), median_uncertainty_u_px = median(σu))

# To express velocity, suppose a calibration target established 0.02 mm per
# pixel and the camera exposures were 1 ms apart. These values illustrate
# the conversion; use your measured calibration and timing for real data.
# Attach them to the pixel result, then convert to mm and mm/s.
# Select the fine-grid point nearest the window inspected earlier.

scale = PhysicalScale(pixel_size = 0.02, dt = 0.001,
                      length_unit = "mm", time_unit = "s")
scaled = with_scale(fine, scale)
velocity = physical(scaled)
fine_i = argmin(abs.(fine.y .- basic.y[i]))
fine_j = argmin(abs.(fine.x .- basic.x[j]))
(pixel_displacement = fine.u[fine_i, fine_j],
 velocity_mm_per_s = velocity.u[fine_i, fine_j],
 x_mm = velocity.x[fine_j],
 uncertainty_mm_per_s = velocity.uncertainty_u[fine_i, fine_j])

# `fine.u` remains in pixels. The converted velocity uses
# `u × 0.02 / 0.001`; `physical` also scales positions and uncertainty.
# See [Scale results to physical units](../howto/scaling.md) for attaching
# calibration to saved results and [Uncertainty quantification](../explanation/uncertainty.md)
# for interpreting the uncertainty estimate.

# ## Optional: check against synthetic truth
#
# The image generator advances each particle from its starting position.
# Multi-pass PIV attributes a vector to the trajectory midpoint, so a
# reference evaluated at `x - u/2, y - v/2` is a closer comparison for
# curved flow. [`error_statistics`](@ref) excludes masked and flagged
# vectors. This check is available because the flow was prescribed; on a
# recording, use the diagnostics above to assess the measurement.

midpoint_reference(r) = (
    [flow(x - r.u[i, j] / 2, y - r.v[i, j] / 2, 0.0, 0.0)[1]
     for (i, y) in enumerate(r.y), (j, x) in enumerate(r.x)],
    [flow(x - r.u[i, j] / 2, y - r.v[i, j] / 2, 0.0, 0.0)[2]
     for (i, y) in enumerate(r.y), (j, x) in enumerate(r.x)],
)

u_ref, v_ref = midpoint_reference(fine)
err = error_statistics(fine, u_ref, v_ref)
(rms_u_px = err.rms_u, rms_v_px = err.rms_v, bias_u_px = err.bias_u)

# ## Where to go next
#
# - Apply this workflow to a [wind-tunnel recording](real_data.md).
# - Inspect seeding, focus, and illumination with the
#   [image-quality guide](../howto/image_quality.md).
# - Process a recording with many pairs and summarize it in the
#   [sequence tutorial](sequence_statistics.md).
# - Draw masks with the [masking guide](../howto/masking.md).
# - Tune outlier handling with [Tune validation](../howto/validation.md).
