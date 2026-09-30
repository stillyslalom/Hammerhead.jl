# Build a preprocessing chain

**Goal:** reduce static glare, uneven illumination, saturation, or low
contrast before correlation. Check each change against the resulting vectors;
processing that helps one recording may hurt another.

First [inspect particle images](image_quality.md) to distinguish illumination
variation from saturation, motion blur, or insufficient particle sampling.

## The building blocks

| Function | Removes | Typical use |
|---|---|---|
| [`subtract_background`](@ref) | a static background | estimate it from frames in which moving particles leave each pixel uncovered |
| [`highpass_filter`](@ref) | broad illumination gradients | start with `sigma` a few times the particle-image diameter in pixels |
| [`intensity_cap`](@ref) | the influence of unusually bright pixels [Shavit2007](@cite) | try when a small number of bright particles dominate the correlation; it cannot recover clipped shapes |
| [`clahe`](@ref) (contrast-limited adaptive histogram equalization, CLAHE) | poor local contrast | compare dim and bright regions after equalization; it can also amplify noise |

Each has a mutating form (`subtract_background!`, `highpass_filter!`,
`intensity_cap!`, `clahe!`) that overwrites a floating-point image. These
forms avoid a copy of the input at each step, though filters and CLAHE still
use temporary buffers internally.

## Estimate a background from the sequence

For moving, sparse particles over a static background, the pixel-wise minimum
over many frames can estimate that background if most pixels are uncovered at
least once:

```julia
using Hammerhead

files = readdir("run_042"; join = true)
bg = compute_background(load_image(f) for f in files)   # method = :min
```

Preview `bg` next to several input frames. If a bright feature remains in
the estimate, collect more frames or use `method = :mean` only when particle
contributions are weak enough that averaging is appropriate. A minimum can
also pick unusually dark noise values, especially from a long sequence.

## Chain in-place transforms

The mutating forms return the image, so they nest naturally:

```julia
preprocess(img) = clahe!(highpass_filter!(subtract_background!(img, bg); sigma = 8))
```

Order matters: remove a separable static background first, then address
broad illumination variation. Add capping or local contrast adjustment only
if the comparison below shows a benefit.

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
the background. Before running a batch, process representative pairs with
and without each proposed step, using the same PIV parameters and masks.
Compare vectors, the valid-vector count, and `peak_ratio` in dim, bright, and
high-gradient regions. A higher peak ratio alone is insufficient if vectors
shift implausibly or valid coverage falls. Keep only steps that improve the
measurements you need.
The [real-data tutorial](../tutorials/real_data.md) shows this comparison on
a wind-tunnel recording.
