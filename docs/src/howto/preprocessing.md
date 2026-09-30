# Build a preprocessing chain

**Goal:** reduce static glare, uneven illumination, saturation, or low
contrast before correlation. Check each change against the resulting vectors;
processing that helps one recording may hurt another.

## The building blocks

| Function | Removes | Typical use |
|---|---|---|
| [`subtract_background`](@ref) | static background (walls, glare) | first, with a [`compute_background`](@ref) image |
| [`highpass_filter`](@ref) | low-frequency illumination gradients | sheet inhomogeneity; `sigma` a few × particle diameter |
| [`intensity_cap`](@ref) | overexposed particles and reflections [Shavit2007](@cite) | limit unusually bright pixels before correlation |
| [`clahe`](@ref) (contrast-limited adaptive histogram equalization, CLAHE) | poor local contrast | dim regions next to bright ones |

Each has a mutating form (`subtract_background!`, `highpass_filter!`,
`intensity_cap!`, `clahe!`) that operates in place on a floating-point
image — use those in hot loops so chains don't allocate per step.

## Estimate a background from the sequence

For sparse particles over static background, the pixel-wise minimum over
many frames is a robust background estimate:

```julia
using Hammerhead

files = readdir("run_042"; join = true)
bg = compute_background(load_image(f) for f in files)   # method = :min
```

## Chain in-place transforms

The mutating forms return the image, so they nest naturally:

```julia
preprocess(img) = clahe!(highpass_filter!(subtract_background!(img, bg); sigma = 8))
```

Order matters: remove the background first (so the high-pass filter doesn't
smear glare into halos), cap or equalize last.

## Hook into the batch drivers

[`run_piv_sequence`](@ref), [`run_piv_ensemble`](@ref), and
[`self_calibrate`](@ref) accept a `preprocess` function applied to each
frame after loading:

```julia
pairs = image_pairs(files; mode = :chained)
results = run_piv_sequence(pairs, passes; preprocess)
```

Frames loaded from file paths arrive as fresh buffers, so mutating
preprocessors are safe there. If you pass in-memory matrices, they are
handed to `preprocess` as-is — use the allocating forms in that case to
leave your arrays untouched.

Preprocessing currently runs on the central processing unit (CPU) even when a
particle image velocimetry (PIV) driver uses a graphics processing unit (GPU)
backend. The processed pair is uploaded afterward; keep mutating chains
allocation-light and reuse batch-driver workspaces as described in
[Run PIV on a GPU](gpu.md).

## Check the effect before committing

An aggressive high-pass filter can remove large particle images along with
the background. Before running a batch, process one pair with and without
the chain. Compare the vector fields, outlier flags, and `result.peak_ratio`
distributions. Keep a step only if it improves the measurements you need.
The [real-data tutorial](../tutorials/real_data.md) shows this comparison on
a wind-tunnel recording.
