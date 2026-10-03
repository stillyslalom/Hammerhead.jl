# Start with the images

Before changing a PIV setting, zoom in on a pair of frames. Can you recognize
the same particle patterns in both? Look at a bright region, a dim region and
the area where you expect the largest motion. Those three views often explain
more than a first vector plot.

If you want a recording to practice on, follow
[the tip-vortex example](../tutorials/real_data.md). It includes the images,
processing code and resulting field.

## Pick the problem you can see

| In the image | First thing to try |
|:--|:--|
| A wall or reflection hides the particles | [Draw an exclusion mask](../howto/masking.md) |
| Illumination varies slowly across the image | [Compare a high-pass filter with the raw image](../howto/preprocessing.md) |
| A stationary bright background appears in every frame | [Estimate and subtract the background](../howto/preprocessing.md#Estimate-a-background-from-the-sequence) |
| Particles look clipped, blurred or barely sampled | [Inspect particle-image quality](../howto/image_quality.md) before filtering |

## Make one change and compare

With `imgA` and `imgB` loaded, keep the PIV settings fixed while trying a filter:

```julia
using Hammerhead

passes = multipass_parameters([32, 32]; padding=true, apodization=:gauss)
raw = run_piv(imgA, imgB, passes)
filtered = run_piv(highpass_filter(imgA; sigma=5),
                   highpass_filter(imgB; sigma=5), passes)
```

Use windows that fit your images. Compare the same region in both fields:
did a suspicious pattern disappear, or did the filter also remove useful
particle detail? Judge the change through both the particle images and the vectors.
The [preprocessing guide](../howto/preprocessing.md) explains the available
operations and how to reuse a chain for a recording.

Masks and filters solve different problems. A mask says “do not measure here”;
a filter changes the intensities used to measure elsewhere. Keep masks at the
original image size, even when analyzing a smaller region.

**Next:** [choose a processing method](processing.md), or draw and compare
interactively in the [GUI](../howto/gui.md).
