# Correlation accuracy

Particle image velocimetry (PIV) estimates a window's displacement from the
location of its correlation peak. The defaults of [`PIVParameters`](@ref)
favor speed; two switches — `padding` and `apodization` — trade roughly 4×
the fast Fourier transform (FFT) cost for a large
accuracy gain. This page explains what each one does and what accuracy to
expect.

## Plain circular correlation biases toward zero

FFT-based cross-correlation is *circular*: content shifted out of one side
of the window wraps around to the other. With the default un-padded
correlation, the wrap-around overlap decreases linearly as displacement
grows, which systematically pulls the measured peak toward zero — about
**0.15 px** of bias on typical synthetic data. That is acceptable for
exploratory work, but it is the dominant error source long before random
noise matters.

## Zero padding and overlap-gain normalization

Setting `padding = true` zero-pads each window to twice its size before the
FFT, turning circular correlation into true linear correlation — no
wrap-around. Linear correlation has its own bias, though: a displacement of
`d` leaves only `N - d` overlapping pixels, so raw peak heights fall off
linearly with displacement, skewing the peak shape. Hammerhead therefore
normalizes each correlation plane by the overlap gain (the number of
contributing pixel pairs at each shift), which restores an unbiased peak.
This normalization is built in — `padding = true` always includes it.

## Gaussian apodization

Sharp window edges leak spectral energy across the correlation plane and
distort the peak's shape at subpixel scale. `apodization = :gauss` applies a
Gaussian taper to each interrogation window before correlating, suppressing
edge effects.

The combination `padding = true, apodization = :gauss` is the accuracy
configuration: on noise-free synthetic data with a converged multi-pass
schedule it reaches about **0.03 px root-mean-square (RMS)** error. All
tutorials use it.

## Subpixel peak fitting and peak locking

The integer peak location is refined by fitting a Gaussian to the peak's
neighborhood (`subpixel_method`):

- `:gauss3` (default) — two independent 3-point Gaussian fits, one per
  axis. Fast and accurate for round peaks.
- `:gauss9` — closed-form 2D Gaussian regression on the 3×3 neighborhood
  [NobachHonkanen2005](@cite). Exact for rotated elliptical peaks and
  measurably less prone to *peak locking* (the tendency of subpixel
  estimates to cluster at integer displacements), at nearly the same cost.
- `:gauss2d` — iterative least-squares 2D Gaussian fit; the most flexible
  and by far the slowest.

Diagnose peak locking in your own data with [`peak_locking`](@ref), which
histograms the fractional parts of the displacements and reports a locking
index.

## Phase correlation

`correlation_method = :phase` whitens the cross-power spectrum before the
inverse transform, sharpening the peak and suppressing broadband intensity
variations at some cost in noise robustness. The standard `:cross` method
with good preprocessing is usually the better starting point.
