# Build a preprocessing chain

**Goal:** condition raw recordings — static glare, uneven illumination,
overexposed particles, low contrast — before correlation. Applied well,
preprocessing is often the difference between an unusable and a clean
vector field on real data.

## The building blocks

| Function | Removes | Typical use |
|---|---|---|
| [`subtract_background`](@ref) | static background (walls, glare) | first, with a [`compute_background`](@ref) image |
| [`highpass_filter`](@ref) | low-frequency illumination gradients | sheet inhomogeneity; `sigma` a few × particle diameter |
| [`intensity_cap`](@ref) | overexposed particles and reflections [Shavit2007](@cite) | before correlation, cheap and safe |
| [`clahe`](@ref) | poor local contrast | dim regions next to bright ones |

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

## Check the effect before committing

Preprocessing that helps correlation can also destroy information (an
aggressive high-pass erases large particles). Sanity-check on one pair
before running a batch: process with and without the chain and compare
`result.peak_ratio` distributions — the chain should raise it.
