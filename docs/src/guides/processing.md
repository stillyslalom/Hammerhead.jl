# Choose how to measure the motion

Settings are best chosen on a representative pair that contains the flow
feature of interest: measure it, inspect the vectors over the image, and
change one setting at a time before processing the whole recording.

## Begin with planar PIV

For two images from one camera, an effort preset is a useful first pass:

```julia
using Hammerhead
imgA = load_image("frame_0001.tif")
imgB = load_image("frame_0002.tif")
result = run_piv(imgA, imgB; effort=:medium)
```

[A first vector field](../tutorials/first_vector_field.md) explains how each
vector is obtained. [Choose an effort level](../howto/effort.md) covers the
trade-off between spatial detail and the particle signal within each window.
Where the signal stays weak, mask obscured regions and check the image
quality.

## Methods

| Quantity | Example |
|:--|:--|
| A field of particle-pattern displacements | [PIV on a real tip-vortex recording](../tutorials/real_data.md) |
| A representative field when individual pairs are weak | [Pool correlations across pairs](../howto/ensemble.md) |
| Individual particles and their paths | [Particle tracking](../tutorials/ptv.md) |
| Three velocity components from two cameras | [Stereo PIV](../tutorials/stereo.md), then [calibrate your rig](../howto/stereo_rig.md) |

## Decide whether a settings change helps

Inspect rejected vectors as well as the remaining ones. A smoother field can
come from filling rejected measurements, and a smaller window can reveal more
detail while leaving too few particles for reliable correlation.
[Validation](../howto/validation.md) explains the flags and how to tune the checks.

Run both settings on the same representative pair and compare the fields
side by side; [`recipe_diff`](@ref) lists exactly which settings differ
between two saved recipes.

Once the settings are useful, [run the recording](repeat.md).
For supported hardware, [GPU processing](../howto/gpu.md) can accelerate that work.
