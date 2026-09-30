# Correlation accuracy

Particle image velocimetry (PIV) estimates a window's displacement from the
location of its correlation peak. The `padding` and `apodization` settings
change how window boundaries affect that peak. Use this page to choose a
correlation method and interpret its limitations.

## Why circular correlation can bias a shift

FFT-based cross-correlation without padding is *circular*: content shifted
out of one side of a window wraps around to the other. These artificial
particle pairings can distort the peak, especially when displacement is
large relative to the window. Inspect the peak and compare configurations
on representative image pairs before relying on a single-pass result.

## Zero padding and overlap compensation

Setting `padding = true` zero-pads each window to twice its size before the
fast Fourier transform (FFT), giving linear correlation without wraparound.
Linear correlation has its own effect: a shifted pair of equal-sized windows
has less overlap at larger shifts, so raw peak heights decrease. For those
windows Hammerhead compensates using the overlap of the window weights at
each shift. It applies this gain only where overlap is at least half of its
zero-lag value, avoiding large amplification of distant, weak peaks.

This compensation for equal-sized windows is built into `padding = true`.

## Gaussian apodization

Sharp window edges can distort the correlation peak at subpixel scale.
`apodization = :gauss` tapers the interrogation window before correlating,
reducing that edge effect.

Try `padding = true, apodization = :gauss` when window-edge bias matters.
Compare it with the default on representative pairs: the measured error
also depends on seeding, image quality, displacement, and velocity
gradients. Padding increases the FFT size and memory use.

## Enlarged search areas

Set `search_area_size` larger than `window_size` when the first-pass
displacement can exceed the interrogation window's useful capture range. The
frame-A interrogation window remains small, preserving its particle-sampling
footprint, and is compared with a larger frame-B search area. Each size
difference must be even so the two footprints share a pixel-grid center. Grid
stride remains `window_size - overlap`; outer vector centers move inward to
keep the search area inside the images.

The correlation plane is search-area-sized, or twice that size with `padding`.
Only shifts for which the interrogation footprint lies fully inside the
search area can be selected as peaks. Their overlap weight is constant, so
no additional gain is needed. Gaussian apodization is applied to the
interrogation footprint; the larger search area remains untapered so
candidates near its boundary are not suppressed. Enlarged search areas
currently require `backend = :cpu`.

## Subpixel peak fitting and peak locking

The integer peak location is refined from neighboring correlation values
(`subpixel_method`):

- `:gauss3` (default) uses independent three-point Gaussian fits along the
  horizontal and vertical axes.
- `:gauss9` uses closed-form two-dimensional Gaussian regression on the 3×3
  neighborhood [NobachHonkanen2005](@cite). It can represent a rotated
  elliptical peak.
- `:gauss2d` uses an iterative least-squares two-dimensional Gaussian fit
  over the 3×3 neighborhood, with more computation per peak.

Use [`peak_locking`](@ref) to see whether fractional displacements cluster
near integers. Such clustering can indicate *peak locking*, but interpret
the histogram alongside the flow's actual displacement distribution.

## Phase correlation

`correlation_method = :phase` whitens the cross-power spectrum before the
inverse transform. This can sharpen a broad peak, but it also changes the
weight given to weak and noisy frequencies. Compare its vectors, peak
ratios, and rejected-vector counts with `:cross` on representative images
before using it for a full sequence.
