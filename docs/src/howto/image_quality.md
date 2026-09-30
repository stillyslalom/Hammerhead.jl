# Inspect images before choosing PIV settings

**Goal:** decide which regions contain usable particle patterns and which
acquisition or processing choices need attention. Inspect both images at
native pixel resolution, including the dimmest region, the fastest flow,
and any wall or reflection. A whole-frame preview can hide individual
particle shapes.

## Compare particle images

These illustrations use the same particle positions where possible and the
same grayscale limits. Compare particle width, flat-topped bright regions,
and the number of particles that would fit inside an interrogation window.

```@setup image_quality
using Hammerhead.SyntheticData
using CairoMakie

centers = [(9.4, 12.7), (25.2, 9.6), (48.6, 16.4), (17.8, 32.3),
           (38.3, 35.7), (55.1, 44.2), (10.6, 52.4), (32.7, 53.8)]
function particle_patch(points; diameter = 5.0, offsets = [0.0])
    img = zeros(64, 64)
    for (x, y) in points, dx in offsets
        generate_gaussian_particle!(img, (x + dx, y), diameter, 1 / length(offsets))
    end
    return img
end
resolved = particle_patch(centers)
small = particle_patch(centers; diameter = 1.2)
sparse = particle_patch(centers[[2, 6]])
saturated = min.(2.5 .* resolved, 1.0)
blurred = particle_patch(centers; offsets = collect(-3.0:1.5:3.0))
glare = copy(resolved)
glare[23:41, 16:49] .= 1.0
quality_figure = Figure(size = (840, 560))
for (k, (title, img)) in enumerate(zip(
        ["Resolved particle images", "Under-resolved images", "Sparse seeding",
         "Clipped bright particles", "Motion during exposure", "Static reflection"],
        [resolved, small, sparse, saturated, blurred, glare]))
    ax = Axis(quality_figure[fld(k - 1, 3) + 1, mod(k - 1, 3) + 1];
              title, xlabel = "x (px)", ylabel = "y (px)",
              yreversed = true, aspect = DataAspect())
    heatmap!(ax, 1:64, 1:64, permutedims(img); colormap = :grays,
             colorrange = (0, 1))
end
```

```@example image_quality
quality_figure # hide
```

| Observation | Consequence for the measurement | What to check or change |
|---|---|---|
| Most particle images occupy only one pixel | Subpixel displacements are poorly resolved; peak locking can occur | Inspect enlarged crops and the fractional-displacement distribution with [`peak_locking`](@ref). For a new acquisition, adjust imaging so particle images span several pixels. |
| Few particle images in a window | The correlation may have several competing peaks or no reproducible peak | Try a larger window and check whether the resulting spatial averaging is acceptable. For stationary flow, consider [ensemble correlation](ensemble.md). |
| Particle centers have a flat intensity plateau | Saturation has changed the particle shape | Check the original pixel values and camera bit depth. Reduce gain, illumination, or exposure in a new acquisition; intensity capping cannot reconstruct clipped shapes. |
| Particles form streaks | Motion during the exposure has blurred their positions | Check exposure or illumination-pulse duration. Shortening the exposure is a different adjustment from changing the delay between the two exposures. |
| A bright feature stays fixed while particles move | The feature can create a strong zero-displacement peak | Use [background subtraction](preprocessing.md) for a separable static background, or [mask](masking.md) an obscured region. |
| Particles disappear or change brightness between frames | Fewer particles contribute to the same displacement peak | Check whether particles leave the image or light sheet, and compare the two exposures' illumination. A shorter pair delay may preserve more particle pairs. |

Particle-image diameter is measured in pixels; it is different from the
physical tracer diameter. Whether tracers follow the fluid depends on their
response time and the flow timescales. Image sharpness alone cannot establish
that they are suitable tracers. These acquisition effects contribute to the
measurement uncertainty [Sciacchitano2019](@cite).

## Relate displacement to the window and timing

Estimate displacement from a few recognizable patterns in the two images,
or start with a coarse pass and inspect its reliable vectors. For an
initial pass with equal-sized windows, keeping displacement below roughly a
quarter of the window width is a useful starting guideline: larger motion
loses more particle pairs at the window boundary. It is not a hard search
limit or a guarantee of a valid measurement. Enlarged search areas and
multi-pass deformation change the capture conditions; see
[Correlation accuracy](../explanation/correlation.md).

For a spatial calibration of 0.02 mm/px and an expected speed of 100 mm/s,
a 1 ms delay gives an expected displacement of 5 px. A shorter delay gives
smaller displacements, making a fixed pixel error larger relative to the
motion. A longer delay can increase particle loss and the variation of
displacement within a window. Try a representative fast and slow region
before selecting one delay for an entire recording.

Use the delay between the paired illumination pulses or exposures for
velocity conversion. The time between successive image pairs sets the
sampling interval for time-series analysis and may be much larger. Keep
both in the acquisition metadata; the
[sequence tutorial](../tutorials/sequence_statistics.md) uses them separately.

## Choose what the result must resolve

A 32 px window with 50% overlap places vectors every 16 px, but those vectors
still use particle patterns across 32 px windows. Increasing overlap adds
sample locations; it does not make each measurement independent or reduce
its interrogation footprint. Strong gradients within that footprint can
broaden the correlation peak and bias the displacement [Westerweel2008](@cite).

Compare two final window sizes on the same images and inspect a velocity
profile through the feature of interest. Check rejected-vector counts as
well as the profile: a finer grid is useful only where sufficient particle
information remains. The [tip-vortex tutorial](../tutorials/real_data.md)
works through this comparison.

Keep a run log alongside the result with the full pass schedule, overlap,
validation settings, input mask, spatial calibration, and both the image-pair
delay and field-sampling interval. A result stores its measured fields, but
these acquisition and processing choices are needed to reproduce and
interpret them. Assess correlation quality, sensitivity to settings, and
physical plausibility together.
