# Prepare the images

Image problems are easiest to diagnose in the images themselves. An enlarged
view of a pair of frames — a bright region, a dim region, and the region of
largest expected motion — shows whether the same particle patterns are
recognizable in both exposures, which often explains a poor vector field
before any setting is changed. The
[wing-tip vortex tutorial](../tutorials/real_data.md) works through this
inspection on a real recording.

## Common image problems

| In the image | Remedy |
|:--|:--|
| A wall or reflection hides the particles | [Draw an exclusion mask](../howto/masking.md) |
| Illumination varies slowly across the image | [Compare a high-pass filter with the raw image](../howto/preprocessing.md) |
| A stationary bright background appears in every frame | [Estimate and subtract the background](../howto/preprocessing.md#Estimate-a-background-from-the-sequence) |
| Particles look clipped, blurred or barely sampled | [Inspect particle-image quality](../howto/image_quality.md) before filtering |

## Evaluate one change at a time

Keep the PIV settings fixed while evaluating a filter:

```julia
using Hammerhead

passes = multipass_parameters([32, 32]; padding=true, apodization=:gauss)
raw = run_piv(imgA, imgB, passes)
filtered = run_piv(highpass_filter(imgA; sigma=5),
                   highpass_filter(imgB; sigma=5), passes)
```

Choose windows that suit the images, and compare the same region in both
fields. A filter can remove a spurious pattern, but it can also remove
particle detail; the particle images and the vectors together show which.
[Preprocess images](../howto/preprocessing.md) describes the available
operations and how to reuse a chain for a recording.

Masks and filters solve different problems. A mask says “do not measure here”;
a filter changes the intensities used to measure elsewhere. Keep masks at the
original image size, even when analyzing a smaller region.

Next: [choose a processing method](processing.md). The
[GUI](../howto/gui.md) offers the same masks and filters with a live preview.
