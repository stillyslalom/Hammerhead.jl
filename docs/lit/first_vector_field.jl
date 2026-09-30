# # Your first vector field
#
# Generate a particle-image pair, measure its displacement field, then check
# validation and uncertainty estimates against the known flow. The synthetic
# images let you measure the error directly.
#
# ## Generate a synthetic image pair
#
# Particle image velocimetry (PIV) measures displacement between two images
# of a flow seeded with small tracer particles. A short light pulse freezes
# the particles in each exposure; their pattern shifts between exposures as
# the fluid moves. Hammerhead's `SyntheticData` submodule renders such image
# pairs from a prescribed velocity field. Each particle moves by its local
# velocity times ``\Delta t``, giving us a reference field for comparison.
#
# A velocity field is any function `(x, y, z, t) -> (u, v, w)`. Ready-made
# fields exist ([`vortex_flow`](@ref Hammerhead.SyntheticData.vortex_flow),
# [`shear_flow`](@ref Hammerhead.SyntheticData.shear_flow),
# [`linear_flow`](@ref Hammerhead.SyntheticData.linear_flow)); here we
# define a Lamb–Oseen vortex with a smooth core:

using Hammerhead
using Hammerhead.SyntheticData
using Random

center = (128.0, 128.0)
rc = 40.0        # core radius, px
Γ = 1200.0       # circulation, px² per frame interval (≈3 px peak displacement)

function flow(x, y, z, t)
    dx, dy = x - center[1], y - center[2]
    r² = dx^2 + dy^2
    k = r² < 1e-9 ? Γ / (2π * rc^2) : Γ / (2π * r²) * (1 - exp(-r² / rc^2))
    return (-k * dy, k * dx, 0.0)
end

rng = MersenneTwister(42)
imgA, imgB, particles1, particles2 = generate_synthetic_piv_pair(
    flow, (256, 256), 1.0;
    particle_density = 0.05,     # particles per pixel
    background_noise = 0.03,     # sensor noise level
    z_range = (-1.0, 1.0),       # keep particles inside the laser sheet
    rng,
)
size(imgA), extrema(imgA)

# Plot the pair. To analyze image files, load each frame with
# `load_image("frame_0001.tif")`.

using CairoMakie

let
    fig = Figure(size = (720, 380))
    for (i, (img, title)) in enumerate(((imgA, "frame A"), (imgB, "frame B")))
        ax = Axis(fig[1, i]; title, yreversed = true, aspect = DataAspect())
        image!(ax, img'; colormap = :grays)
    end
    fig
end

# ## Run a basic analysis
#
# [`run_piv`](@ref) tiles the images into interrogation windows, correlates
# each window pair, and refines every correlation peak to subpixel
# precision. With no configuration it runs a single pass of 32×32 windows
# at 50% overlap:

result = run_piv(imgA, imgB)

# An *interrogation window* is a small image region containing several
# particles. Correlation finds the shift that best aligns its particle pattern
# between frames, so each window produces one representative displacement
# vector rather than one vector per particle.
#
# The [`PIVResult`](@ref) holds the interrogation grid (`x` along columns,
# `y` along rows, in pixels) and the displacement fields (`u` along x, `v`
# along y). A particle at `(row, col)` in frame A is found at
# `(row + v, col + u)` in frame B. The Makie extension plots the result:

plot_vector_field(result)

# ## Multi-pass with image deformation
#
# A multi-pass schedule starts with
# large windows, then uses each pass's validated field to *deform* the
# images before the next, finer pass. Later passes measure a smaller
# residual and can afford small windows. Two more switches, `padding` and
# `apodization`, reduce the systematic bias of plain fast Fourier transform
# (FFT) correlation (see
# [Correlation accuracy](../explanation/correlation.md)):

passes = multipass_parameters([64, 32, 16, 16];
    padding = true,
    apodization = :gauss,
    uncertainty = true,     # per-vector uncertainty, estimated on the final pass
)
result = run_piv(imgA, imgB, passes)

#-

plot_vector_field(result)

# The final windows are 16 px instead of 32 px. Repeating 16 in the schedule
# `[64, 32, 16, 16]` adds a convergence
# sweep, which the uncertainty estimator requires (it assumes the
# deformation has converged).
#
# ## Check against the ground truth
#
# The generator displaces each particle by its velocity at the *launch
# point*, while symmetric image deformation attributes each measured
# vector to the *midpoint* of the particle trajectory (that midpoint
# attribution is what makes the scheme second-order accurate; see
# [Multi-pass interrogation](../explanation/multipass.md)). To compare
# like with like, we evaluate the reference velocity at the launch point
# `x - d/2`, and hand the reference fields to
# [`error_statistics`](@ref):

midpoint_reference(r) = (
    [flow(x - r.u[i, j] / 2, y - r.v[i, j] / 2, 0.0, 0.0)[1]
     for (i, y) in enumerate(r.y), (j, x) in enumerate(r.x)],
    [flow(x - r.u[i, j] / 2, y - r.v[i, j] / 2, 0.0, 0.0)[2]
     for (i, y) in enumerate(r.y), (j, x) in enumerate(r.x)],
)

u_ref, v_ref = midpoint_reference(result)
err = error_statistics(result, u_ref, v_ref)
(bias_u = err.bias_u, rms_u = err.rms_u, rms_v = err.rms_v, n = err.n)

# The padded, apodized multi-pass result has about 0.03 pixels (px) of
# root-mean-square (RMS) error and negligible bias over the field. Plain
# unpadded single-pass correlation has a systematic bias of about 0.15 px
# on this pair.
#
# ## Per-vector uncertainty
#
# With `uncertainty = true`, the final pass estimates each vector's random
# error from correlation statistics (Wieneke 2015) into `uncertainty_u` /
# `uncertainty_v`. The median estimate should sit at the noise-driven
# share of the root-mean-square error measured above. The estimator uses
# image correlations, not the reference field:

using Statistics: median

valid = .!(result.outliers .| result.mask)
σu = filter(isfinite, result.uncertainty_u[valid])
(median_uncertainty_u = median(σu), measured_rms_u = err.rms_u)

# The estimator captures the random error of each correlation; the
# remaining gap to the measured RMS is residual deformation error that no
# per-window estimator can see. See
# [Uncertainty quantification](../explanation/uncertainty.md) for what
# the numbers mean and when to trust them.
#
# ## Outliers, validation, and masking
#
# This pair has no flagged outliers (`count(result.outliers) == 0`). To see
# what validation catches, add a saturated reflection: a bright static patch in *both*
# frames:

imgA_refl, imgB_refl = copy(imgA), copy(imgB)
for img in (imgA_refl, imgB_refl)
    img[97:144, 41:104] .= 1.0
end
result_refl = run_piv(imgA_refl, imgB_refl, passes)
count(result_refl.outliers)

# A static reflection can produce high-confidence *zero* vectors that
# disagree with their neighbors. Universal outlier detection flags vectors
# that differ enough from nearby measurements. Flagged vectors are replaced by
# the local median of their valid neighbors (`replace_outliers = true`),
# and the flag identifies values interpolated rather than
# measured:

plot_vector_field(result_refl)

# Windows partly covering the reflection can still produce biased vectors
# that pass validation. Mark the reflection with an image mask. Windows with
# sufficient masked coverage are excluded from the vector field, including
# subsequent validation, replacement, and statistics:

mask = falses(size(imgA))
mask[97:144, 41:104] .= true
result_masked = run_piv(imgA_refl, imgB_refl, passes; mask)

u_ref, v_ref = midpoint_reference(result_masked)
err_masked = error_statistics(result_masked, u_ref, v_ref)
(rms_u = err_masked.rms_u,
 outliers = count(result_masked.outliers),
 masked = count(result_masked.mask))

# Compare the new RMS error with the unmasked result above. The excluded
# windows no longer contribute biased vectors.
# `result.mask` records the excluded windows, separately
# from `result.outliers` (see
# [The masking model](../explanation/masking.md)):

plot_vector_field(result_masked)

# ## Where to go next
#
# - Apply the workflow to a wind-tunnel recording:
#   [the real-data tutorial](real_data.md).
# - Real image files: [`load_image`](@ref) and the
#   [batch-processing guide](../howto/batch.md).
# - Polygon and image-file masks: the [masking guide](../howto/masking.md).
# - When the defaults flag too much or too little:
#   [Tune validation](../howto/validation.md).
# - Noisy recordings: the
#   [preprocessing](../howto/preprocessing.md) and
#   [ensemble-correlation](../howto/ensemble.md) guides.
# - Two cameras: the [stereo tutorial](stereo.md).
