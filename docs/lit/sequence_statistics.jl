# # From image pairs to flow statistics
#
# Measure a series of particle-image pairs, then calculate a mean field and
# the variation between pairs. This example generates a short sequence so
# the sampling times and imposed motion are known. The same analysis works
# on image paths; see [Batch processing](../howto/batch.md).
#
# ## Make a sampled recording
#
# Each pair has two exposures separated by 0.01 s. New pairs start every
# 0.02 s. The exposure separation converts displacement to velocity; the
# interval between pairs tells us how often the velocity field was sampled.
# They need not be equal. The imposed horizontal velocity varies periodically
# from pair to pair, while the vertical velocity stays fixed.

using Hammerhead
using Hammerhead.SyntheticData
using Random
using Statistics: median
using CairoMakie

exposure_dt = 0.01             # seconds between images A and B
sample_dt = 0.02               # seconds between successive image pairs
sample_times = (0:15) .* sample_dt
flow(x, y, z, t) = (200 + 60sin(2π * t / 0.16), 50.0, 0.0)  # px/s

rng = MersenneTwister(2026)
pairs = [begin
    a, b, _, _ = generate_synthetic_piv_pair(flow, (128, 128), exposure_dt;
        t0 = t, particle_density = 0.05, background_noise = 0.03,
        z_range = (-1.0, 1.0), rng)
    (a, b)
end for t in sample_times]

# The prescribed horizontal displacement is 2 ± 0.6 px per pair. We can
# compare the results with it here because we generated the recording;
# a measured sequence normally has no known reference field.

true_u = [flow(64, 64, 0, t)[1] * exposure_dt for t in sample_times]
(first = true_u[1], range = extrema(true_u), pairs = length(pairs))

# ## Analyze every pair
#
# A 32 px window contains enough particles for this synthetic example.
# Repeating it allows a converged deformation before estimating per-vector
# correlation uncertainty. The same mask excludes one corner throughout
# the sequence, as a wall or glare region might in a recording.

passes = multipass_parameters([32, 32];
    padding = true, apodization = :gauss, uncertainty = true)
mask = falses(128, 128)
mask[1:48, 1:48] .= true
results = run_piv_sequence(pairs, passes; mask)

# Each result is a displacement field in pixels per exposure interval.
# Divide by `exposure_dt` for px/s if those units are useful. Do not divide
# by `sample_dt`: it describes the time between fields, not particle motion
# within a pair.

(fields = length(results), grid = size(results[1].u),
 first_pair_valid = count(.!(results[1].outliers .| results[1].mask)))

# At the center, compare measured displacement with the imposed variation.
# Sampling every eighth pair would take one field per 0.16 s cycle at the
# same phase, hiding the fluctuation even though the individual pairs are
# valid measurements.

center_y = argmin(abs.(results[1].y .- 64))
center_x = argmin(abs.(results[1].x .- 64))
center_u = [r.outliers[center_y, center_x] ? NaN :
            r.u[center_y, center_x] for r in results]

let
    fig = Figure(size = (650, 340))
    ax = Axis(fig[1, 1]; xlabel = "pair start time (s)",
              ylabel = "u (px per pair)", title = "Center displacement")
    lines!(ax, sample_times, true_u; label = "imposed")
    scatter!(ax, sample_times, center_u; label = "measured")
    scatter!(ax, sample_times[1:8:end], center_u[1:8:end];
             label = "every eighth pair", markersize = 16)
    axislegend(ax)
    fig
end

# ## Calculate pointwise statistics
#
# [`field_statistics`](@ref) uses only finite, unmasked, unflagged vectors
# by default. `count` records how many pairs contributed at each grid point.
# A location with no accepted samples has `NaN` statistics rather than a
# fabricated zero.

stats = field_statistics(results)
(min_count = minimum(stats.count), max_count = maximum(stats.count),
 missing_locations = count(==(0), stats.count))

let
    fig = Figure(size = (820, 340))
    ax1 = Axis(fig[1, 1]; title = "mean u (px per pair)",
               yreversed = true, aspect = DataAspect())
    hm1 = heatmap!(ax1, stats.x, stats.y, stats.mean_u'; colorrange = (1.4, 2.6))
    Colorbar(fig[1, 2], hm1)
    ax2 = Axis(fig[1, 3]; title = "accepted pairs", yreversed = true,
               aspect = DataAspect())
    hm2 = heatmap!(ax2, stats.x, stats.y, stats.count'; colorrange = (0, length(pairs)))
    Colorbar(fig[1, 4], hm2)
    fig
end

# The unmasked region has a mean displacement near 2 px. The masked corner
# has no estimate. Where `count` is below 16 elsewhere, one or more pairs
# were rejected, so compare locations using their counts as well as their
# means.
#
# ## Watch a mean stabilize
#
# The center stays unmasked. Recompute its running mean as more pairs enter
# the analysis, and compare it with the known mean of the same sampled times.

iy, ix = center_y, center_x
running = [field_statistics(results[1:n]).mean_u[iy, ix] for n in 1:length(results)]
reference_running = [sum(true_u[1:n]) / n for n in 1:length(true_u)]

let
    fig = Figure(size = (720, 340))
    ax = Axis(fig[1, 1]; xlabel = "number of pairs", ylabel = "mean u (px per pair)",
              title = "Running mean at the center")
    lines!(ax, 1:length(running), running; label = "measured")
    lines!(ax, 1:length(reference_running), reference_running;
           label = "mean of imposed samples", linestyle = :dash)
    axislegend(ax)
    fig
end

# The running mean returns near the cycle average after complete cycles.
# Its movement depends on which phases are included, so 16 closely spaced
# pairs are not 16 independent samples of the long-term flow. A flat-looking
# mean alone does not prove accuracy; calibration bias or a missed flow
# feature can remain. Also check accepted-sample counts.
#
# ## Fluctuation RMS and measurement uncertainty
#
# `stats.rms_u` describes how much accepted displacement varies between
# pairs around its local mean. Per-pair `uncertainty_u` instead estimates
# random error in each correlation measurement. In this recording, the
# imposed time variation contributes to RMS even with perfect measurements.

center_σ = [r.uncertainty_u[iy, ix] for r in results
            if !r.outliers[iy, ix] && !r.mask[iy, ix] &&
               isfinite(r.uncertainty_u[iy, ix])]
(mean_u = stats.mean_u[iy, ix], rms_u = stats.rms_u[iy, ix],
 median_pair_σu = median(center_σ), accepted = stats.count[iy, ix],
 imposed_rms = sqrt(sum(abs2, true_u .- sum(true_u) / length(true_u)) /
                    length(true_u)))

# A large RMS relative to correlation uncertainty suggests variation in
# the flow or another source of pair-to-pair change; it is not itself an
# uncertainty interval on the mean. If the flow changes over time, inspect
# the individual fields before treating a single mean as representative.
#
# ## Average vectors or average correlations?
#
# `field_statistics` averages accepted displacement vectors and preserves
# their time variation. [`run_piv_ensemble`](@ref) sums each window's
# correlation planes first, then finds one peak for the whole sequence.
# To compare them for a stationary flow, generate eight independent pairs
# with a constant imposed displacement. Use the same correlation settings:

steady_flow(x, y, z, t) = (200.0, 50.0, 0.0)  # px/s
steady_pairs = [begin
    a, b, _, _ = generate_synthetic_piv_pair(steady_flow, (128, 128), exposure_dt;
        particle_density = 0.05, background_noise = 0.03,
        z_range = (-1.0, 1.0), rng)
    (a, b)
end for _ in 1:8]

steady_results = run_piv_sequence(steady_pairs, passes; mask)
steady_stats = field_statistics(steady_results)
ensemble = run_piv_ensemble(steady_pairs, passes; mask)
(mean_of_vectors = steady_stats.mean_u[iy, ix],
 ensemble_u = ensemble.u[iy, ix], reference_displacement = 2.0,
 valid_pairs = steady_stats.count[iy, ix])

# Both estimate the fixed displacement here, but they need not give identical
# numbers: peak finding after summing correlations differs from averaging
# individual peaks. Ensemble correlation is useful when single-pair peaks
# are too weak to measure reliably. It does not return the pair-to-pair
# fluctuation RMS. The periodic sequence above needs a phase-aware analysis
# if the changing velocity matters. Use ensemble correlation when a mean
# displacement over a statistically stationary interval answers your question.
# A broad distribution of displacements can broaden or skew the summed peak,
# so statistical stationarity alone does not make it an arithmetic mean; see
# [Ensemble correlation](../howto/ensemble.md).
#
# ## Apply this to a recording
#
# For files, pass image-path pairs from [`image_pairs`](@ref) to
# [`run_piv_sequence`](@ref). Record the exposure separation and the field
# sampling interval separately. Check image quality and validation, inspect
# `stats.count`, and compare results across sensible window sizes before
# interpreting local gradients or other derived quantities. The
# [real recording tutorial](real_data.md) shows that window comparison;
# [Scale to physical units](../howto/scaling.md) explains velocity units.
