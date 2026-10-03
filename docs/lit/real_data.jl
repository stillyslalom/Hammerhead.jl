# # Find a vortex in a real recording
#
# The dark disk in these images is not an obstacle. It is the core of a
# wing-tip vortex, where swirling flow has swept many particles away.
# Can we recover the swirl—and see where the images give weaker evidence?
#
# Use case A of the first International PIV Challenge [Stanislas2003](@cite),
# recorded by C. Kähler (DLR) in the German–Dutch Wind Tunnels. The image pair
# is included with Hammerhead; no download is needed.
#
# ## Look before you calculate
#
# Load both exposures and compare their particle patterns.

using Hammerhead
using CairoMakie
using Statistics: median, quantile

dir = joinpath(pkgdir(Hammerhead), "test", "reference_images", "A")
imgA = load_image(joinpath(dir, "A001_1.tif"))
imgB = load_image(joinpath(dir, "A001_2.tif"))

let
    fig = Figure(size = (900, 390))
    for (k, img, title) in ((1, imgA, "First exposure"), (2, imgB, "Next exposure"))
        ax = Axis(fig[1, k]; title, yreversed = true, aspect = DataAspect(),
                  xlabel = "x (px)", ylabel = "y (px)")
        image!(ax, (0.5, size(img, 2) + 0.5), (0.5, size(img, 1) + 0.5), img';
               colormap = :grays, colorrange = (0, 0.6))
    end
    fig
end

# Notice the illumination bands, bright particle images and dark center.
# Compare a well-seeded patch with the core at the **same contrast**.
# Which patch would you expect to give the clearer correlation peak?

let
    fig = Figure(size = (650, 320))
    for (k, rows, cols, title) in ((1, 200:295, 200:295, "Many particles"),
                                   (2, 497:592, 529:624, "Inside the core"))
        ax = Axis(fig[1, k]; title, yreversed = true, aspect = DataAspect(),
                  xlabel = "column in patch", ylabel = "row in patch")
        image!(ax, (0.5, length(cols) + 0.5), (0.5, length(rows) + 0.5),
               imgA[rows, cols]'; colormap = :grays, colorrange = (0, 0.6))
    end
    fig
end

# A core vector can agree with its neighbors even when few particles
# contribute. Keep this image comparison in mind as you inspect the field.
#
# ## Recover the swirl
#
# First estimate how far particles move using generous 96 px windows.
# Summarize only finite vectors without outlier or mask flags.

preview = run_piv(imgA, imgB,
    PIVParameters(window_size = 96, overlap = 48,
                  padding = true, apodization = :gauss))
preview_ok = .!(preview.outliers .| preview.mask) .&
             isfinite.(preview.u) .& isfinite.(preview.v)
shifts = hypot.(preview.u[preview_ok], preview.v[preview_ok])
(p95_shift_px = round(quantile(shifts, 0.95); digits = 1),
 largest_shift_px = round(maximum(shifts); digits = 1))

# Compare this range with your first window's width. Shifts below roughly
# one quarter of that width are a useful starting point, not a guarantee.
# Here we begin at 64 px, refine to 32 px, and repeat the final window size
# to estimate uncertainty.

passes = multipass_parameters([64, 32, 32];
    padding = true, apodization = :gauss, uncertainty = true)
result = run_piv(imgA, imgB, passes)

let
    fig = Figure(size = (700, 540))
    ax = Axis(fig[1, 1]; title = "The particle pattern reveals a vortex",
              yreversed = true, aspect = DataAspect(),
              xlabel = "x (px)", ylabel = "y (px)")
    image!(ax, (0.5, size(imgA, 2) + 0.5), (0.5, size(imgA, 1) + 0.5), imgA';
           colormap = :grays, colorrange = (0, 0.6))
    plot_vector_field!(ax, result; stride = 5, color = :cyan,
                       replaced_color = :orangered, lengthscale = 3)
    fig
end

# Follow the arrows around the dark core. These are in-plane displacements;
# the free-stream direction is perpendicular to the light sheet.
# Arrow lengths are enlarged threefold. Orange arrows carry outlier flags.
#
# **Try it:** reduce `stride` to 2 to see more vectors, then zoom your attention
# to the dark core. Does a smooth-looking field mean equally good evidence
# everywhere?
#
# ## Where is the measurement weaker?
#
# A validation flag finds vectors that fail a check, such as disagreement
# with neighbors. A low peak ratio says another correlation peak competes
# with the chosen shift. Estimated uncertainty describes random correlation
# error [Wieneke2015](@cite). Use these together rather than treating one
# map as a verdict.

valid = .!(result.outliers .| result.mask) .&
        isfinite.(result.u) .& isfinite.(result.v)
(accepted_vectors = count(valid), flagged_vectors = count(result.outliers))

let
    fig = Figure(size = (900, 350))
    pr = copy(result.peak_ratio)
    pr[.!valid] .= NaN
    ax1 = Axis(fig[1, 1]; title = "Peak ratio: competing shifts",
               yreversed = true, aspect = DataAspect())
    hm1 = heatmap!(ax1, result.x, result.y, pr'; colorrange = (1, 5))
    Colorbar(fig[1, 2], hm1)
    sigma = copy(result.uncertainty_u)
    sigma[.!valid] .= NaN
    ax2 = Axis(fig[1, 3]; title = "Horizontal uncertainty (px)",
               yreversed = true, aspect = DataAspect())
    hm2 = heatmap!(ax2, result.x, result.y, sigma'; colorrange = (0, 0.3))
    Colorbar(fig[1, 4], hm2)
    fig
end

# Find the core in both maps. Compare it with the seeded outer region.
# Are the uncertainty and peak ratio telling the same story?
# Summarize each region with medians, excluding flagged and nonfinite values.

distance_from_core = [hypot(x - 577, y - 545) for y in result.y, x in result.x]
function region_quality(region)
    selected = region .& valid
    pr = filter(isfinite, result.peak_ratio[selected])
    sigma = filter(isfinite, result.uncertainty_u[selected])
    (vectors = count(selected), median_peak_ratio = median(pr),
     median_uncertainty_px = median(sigma))
end
(core = region_quality(distance_from_core .< 120),
 outer_flow = region_quality(distance_from_core .> 300))

# Fewer visible particles can weaken a measurement that still passes a
# neighbor check. This recording has no reference displacement field:
# these diagnostics show evidence and sensitivity, not the true error.
#
# ## Does window size change the feature you care about?
#
# Use 48 px final windows on the same pair. Larger windows use more particles
# but average motion over a larger footprint. Compare both fields along the
# same horizontal line through the vortex.

passes48 = multipass_parameters([64, 48, 48];
    padding = true, apodization = :gauss, uncertainty = true)
result48 = run_piv(imgA, imgB, passes48)
line = [(330.0, 545.0), (820.0, 545.0)]
profile32 = extract_profile(result, line; n = 100)
profile48 = extract_profile(result48, line; n = 100)

let
    fig = Figure(size = (680, 360))
    ax = Axis(fig[1, 1]; title = "Across the vortex core",
              xlabel = "x (px)", ylabel = "vertical displacement (px)",
              limits = (300, 850, -12, 12))
    lines!(ax, profile32.x, profile32.v; label = "32 px windows")
    lines!(ax, profile48.x, profile48.v; label = "48 px windows")
    axislegend(ax)
    fig
end

# Both profiles are interpolated onto common positions; that makes comparison
# possible, but cannot restore detail lost inside a large window.
# Look near the steep changes, where a core-width or gradient estimate would
# be most sensitive to processing choices.
#
# **Try it:** move the line above the core to `y = 400`. Does the difference
# between window sizes change? Then increase overlap without changing window
# size. More samples do not necessarily mean more resolved detail.
#
# ## Will a cleaner-looking background help?
#
# The illumination bands suggest high-pass filtering. Test that idea on the
# same pair before adopting it: filters can remove useful particle signal too.

filteredA = highpass_filter(imgA; sigma = 8)
filteredB = highpass_filter(imgB; sigma = 8)
filtered_result = run_piv(filteredA, filteredB, passes)

function pair_quality(r)
    accepted = .!(r.outliers .| r.mask) .& isfinite.(r.u) .& isfinite.(r.v)
    ratios = filter(isfinite, r.peak_ratio[accepted])
    (accepted = count(accepted), flagged = count(r.outliers),
     median_peak_ratio = median(ratios),
     lower_peak_ratio = quantile(ratios, 0.1))
end
(raw = pair_quality(result), highpass = pair_quality(filtered_result))

# Did the competing peaks weaken or strengthen? Did more vectors get flagged?
# This comparison can reject an unhelpful setting, but a higher peak ratio
# alone cannot prove a more accurate field.
#
# **Try it:** change the filter's `sigma`, or substitute
# `clahe(imgA)` and `clahe(imgB)`. Compare the images and diagnostics on the
# same region each time; the [preprocessing guide](../howto/preprocessing.md)
# explains what each operation changes.
#
# This lesson reports pixels between exposures. The sample data supplies no
# spatial calibration or exposure delay for velocity conversion. For your own
# recording, measure both and follow [physical scaling](../howto/scaling.md).
#
# Next, [process a sequence](sequence_statistics.md) to distinguish a persistent
# flow feature from pair-to-pair variation. For a difficult image region, use
# [image inspection](../howto/image_quality.md) before adding more processing.
