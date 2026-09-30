# # A real recording: tip vortex with seeding dropout
#
# Analyze a wind-tunnel image pair to calculate a vector field and identify
# where its measurements are less reliable. You will inspect the images,
# then compare validation flags, peak ratios, and per-vector uncertainty.
#
# The data is case A of the first International Particle Image Velocimetry
# (PIV) Challenge
# [Stanislas2003](@cite): a wing-tip vortex 1.64 m behind a transport
# aircraft half-model in the German–Dutch Wind Tunnels Large Low-Speed
# Facility (DNW-LLF), recorded by C. Kähler of the German Aerospace Center
# (DLR). The images contain strong velocity gradients, varying particle
# image sizes, and loss of seeding in the vortex core.
#
# ## Load and inspect
#
# [`load_image`](@ref) reads any FileIO-supported image (here 12-bit
# grayscale TIFF) into a `Matrix{Float64}` scaled to ``[0, 1]``:

using Hammerhead

dir = joinpath(pkgdir(Hammerhead), "test", "reference_images", "A")
imgA = load_image(joinpath(dir, "A001_1.tif"))
imgB = load_image(joinpath(dir, "A001_2.tif"))
size(imgA), extrema(imgA)

# Inspect the particle images and illumination before choosing analysis
# settings:

using CairoMakie

let
    fig = Figure(size = (860, 400))
    ax1 = Axis(fig[1, 1]; title = "frame A", yreversed = true, aspect = DataAspect())
    image!(ax1, imgA'; colormap = :grays)
    ax2 = Axis(fig[1, 2]; title = "vortex core (closeup)", yreversed = true,
               aspect = DataAspect())
    image!(ax2, imgA[395:695, 425:725]'; colormap = :grays)
    fig
end

# Most of the image has dense seeding and large, bright particle images.
# Illumination varies across the frame: the row-wise mean intensity changes
# by a factor of four. The dark disk near the center is the vortex core,
# roughly 200 px across. Swirling flow has pushed most particles out of
# that region, so its vectors will have less particle information to use.
#
# ## Run the analysis
#
# In-plane displacements in this pair reach about 10 px, so the first pass
# needs 64 px windows (the quarter-window rule); later passes refine at
# 32 px. Repeating the final window size adds the convergence sweep that
# the uncertainty estimator requires. This is the same accuracy
# configuration as in the first tutorial:

passes = multipass_parameters([64, 32, 32];
    padding = true,
    apodization = :gauss,
    uncertainty = true,
)
result = run_piv(imgA, imgB, passes)

#-

plot_vector_field(result)

# The in-plane field shows a tip vortex centered on the dark disk. The
# free-stream flow is perpendicular to the light sheet, so the measured
# in-plane motion is mostly swirl.
#
# ## Judging a measurement without ground truth
#
# There is no reference displacement field for this recording. Use these
# three diagnostics in [`PIVResult`](@ref) to assess the calculated vectors:
#
# 1. **`result.outliers`**: vectors rejected by validation (universal outlier
#    detection by default; peak-ratio rejection requires an explicit threshold).
# 2. **`result.peak_ratio`**: the height ratio of the primary to the
#    secondary correlation peak. A low ratio means the displacement peak
#    has a strong competitor.
# 3. **`result.uncertainty_u` / `uncertainty_v`**: the Wieneke (2015)
#    per-vector random-error estimate [Wieneke2015](@cite).
#
# Start with the flags:

count(result.outliers), length(result.outliers)

# Fewer than one percent of the nearly 5000 vectors are flagged. Most flags
# are near the frame edges, with few in the sparsely seeded core. A core
# window can still pass validation if its remaining particles produce a
# vector consistent with its neighbors. Peak ratio and uncertainty show
# differences among the vectors that pass:

valid = .!(result.outliers .| result.mask);

let
    fig = Figure(size = (900, 330))
    pr = copy(result.peak_ratio)
    pr[.!valid] .= NaN
    ax1 = Axis(fig[1, 1]; title = "peak ratio", yreversed = true, aspect = DataAspect())
    hm1 = heatmap!(ax1, result.x, result.y, pr'; colorrange = (1, 5))
    Colorbar(fig[1, 2], hm1)
    σu = copy(result.uncertainty_u)
    σu[.!valid] .= NaN
    ax2 = Axis(fig[1, 3]; title = "σᵤ (px)", yreversed = true, aspect = DataAspect())
    hm2 = heatmap!(ax2, result.x, result.y, σu'; colorrange = (0, 0.3))
    Colorbar(fig[1, 4], hm2)
    fig
end

# In the core, peak ratios are lower and estimated uncertainty is higher.
# Compare the core with the far field using medians, which are less affected
# by the very large estimates in a few near-outlier windows:

using Statistics: median, quantile

r_core = [hypot(x - 577, y - 545) for y in result.y, x in result.x]  # px from disk center

function region_quality(sel)
    pr = filter(isfinite, result.peak_ratio[sel .& valid])
    σu = filter(isfinite, result.uncertainty_u[sel .& valid])
    (median_pr = round(median(pr); digits = 2),
     median_σu = round(median(σu); digits = 3),
     q90_σu = round(quantile(σu, 0.9); digits = 3))
end
(core = region_quality(r_core .< 120), far_field = region_quality(r_core .> 300))

# In the far field, the median estimated random error is near 0.09 px.
# Inside the core, the median is about 50% higher and the 90th percentile
# roughly doubles because fewer particle images contribute to each
# correlation. The validation flags identify rejected vectors; the
# uncertainty estimates help assess the vectors that remain. See
# [Uncertainty quantification](../explanation/uncertainty.md) for what the
# estimate does and doesn't cover.
#
# ## Compare preprocessing options
#
# Compare each processing option on the same image pair using peak ratios
# and outlier counts. The [preprocessing guide](../howto/preprocessing.md)
# describes the methods. Here, try
# [`highpass_filter`](@ref) for the illumination gradient,
# [`intensity_cap`](@ref) for the bright particles [Shavit2007](@cite),
# and contrast-limited adaptive histogram equalization ([`clahe`](@ref),
# commonly abbreviated CLAHE) for the dim core:

candidates = [
    "raw"              => identity,
    "highpass (σ = 8)" => img -> highpass_filter(img; sigma = 8),
    "intensity cap"    => img -> intensity_cap(img),
    "CLAHE"            => img -> clahe(img),
]

function chain_quality(f)
    res = run_piv(f(imgA), f(imgB), passes)
    ok = .!(res.outliers .| res.mask)
    pr = filter(isfinite, res.peak_ratio[ok])
    (median_pr = round(median(pr); digits = 2),
     q10_pr = round(quantile(pr, 0.1); digits = 2),
     outliers = count(res.outliers))
end
[name => chain_quality(f) for (name, f) in candidates]

# High-pass filtering lowers the median peak ratio and triples the outlier
# count. These large particle images lose signal along with the smooth
# background, while each correlation window already subtracts its own mean
# intensity. Intensity capping doubles the outlier count because the bright
# particle images contribute useful signal. CLAHE provides a modest gain:
# it reduces the visible banding and raises peak ratios in dim regions.
#
# For this pair, use the raw images or consider CLAHE for the dim regions;
# the tested high-pass and intensity-cap settings make the correlations
# worse. The core remains less certain because it contains few particles.
# If you have a sequence of a statistically stationary flow, ensemble
# correlation can combine information from particles passing through the
# core at different times; see
# [Ensemble correlation for low signal-to-noise ratio (SNR)](../howto/ensemble.md).
#
# ## Where to go next
#
# - Static background removal needs a frame *sequence*
#   ([`compute_background`](@ref)): the
#   [preprocessing guide](../howto/preprocessing.md).
# - If the default checks flag too much or too little:
#   [Tune validation](../howto/validation.md).
# - Whole recordings, incremental result files:
#   [Batch processing](../howto/batch.md).
# - What σᵤ means and when to trust it:
#   [Uncertainty quantification](../explanation/uncertainty.md).
# - Two cameras: the [stereo tutorial](stereo.md).
