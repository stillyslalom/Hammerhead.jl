# Choose an effort level

**Goal:** choose a particle image velocimetry (PIV) schedule, then check
whether its final window resolves the flow without losing too many valid
vectors.

For routine runs, omit the explicit pass vector and use `effort`:

```julia
quick = run_piv(imgA, imgB; effort = :low)
draft = run_piv(imgA, imgB; effort = :medium)
final = run_piv(imgA, imgB; effort = :high)
```

The same keyword is accepted by [`run_piv_sequence`](@ref),
[`run_piv_ensemble`](@ref), and [`run_piv_stereo`](@ref). The current presets
are:

| Effort | Schedule | Options | Use it for |
|---|---|---|---|
| `:low` | `[32]` | [`PIVParameters`](@ref) defaults | A quick field and image-quality check when displacements are modest |
| `:medium` | `[64, 32]` | defaults | A first analysis with a coarse predictor and finer final grid |
| `:high` | `[128, 64, 32]` | `padding = true`, `apodization = :gauss`, `max_iterations = 2`; final pass `uncertainty = true` | A more expensive analysis with iterative deformation and per-vector uncertainty |

Each added pass or convergence sweep processes the image again; padding also
increases FFT work. On images too small for a listed window, Hammerhead
reduces the preset's window sizes to fit. For planar PIV with `roi`, those
sizes are based on the selected rectangle, and output coordinates still refer
to the full image. Explicit pass schedules must fit the rectangle. A predictor
with only one vector along an axis is extended constantly along that axis.
None of the presets guarantees a
particular error on your recording.

Start with `:medium` on representative pairs. Compare its field and rejected
vectors with `:low`; use `:high` when smaller residuals or per-vector
uncertainty matter enough to justify the extra computation. Check a profile
through the smallest feature you need to resolve and inspect the particle
images in windows with poor peaks. The [image-quality guide](image_quality.md)
explains the window-size and seeding tradeoff.

Effort and execution backend are independent choices. For example,
`run_piv(imgA, imgB; effort = :high, backend = :amdgpu)` runs the same
high-effort schedule through the AMD GPU extension, including final-pass
Float64 uncertainty statistics. Small `:low` and `:medium` jobs often remain
faster on the CPU because GPU setup and transfers do not amortize. See
[Run PIV on a GPU](gpu.md) before selecting a device backend.

## Override the parts that matter

When `effort` is set, `PIVParameters` keyword arguments override the preset:

```julia
result = run_piv(imgA, imgB;
    effort = :high,
    window_size = 16,        # final pass is 16 px; pyramid rescales with it
    uncertainty = false,     # final-pass override
)
```

Most field keywords apply to every pass. `window_size` sets the final window
size and rescales the pyramid. `search_area_size` similarly sets the final
search area, subject to its centered-footprint constraints; enlarged search
areas currently require `backend = :cpu`. `uncertainty`, `max_iterations`, and
`keep_correlation_planes` apply only to the final pass. Use `final = (;)` when
you need an explicit last-pass override; it wins over both the preset and the
field keywords.

Smaller final windows give denser vectors but contain fewer particle images.
For example, compare 32 px and 16 px on the same pair: look at rejected
vectors and a profile across the feature of interest. Choose 16 px only if
it adds useful spatial detail without losing reliable coverage.

To see what a preset runs, call [`effort_schedule`](@ref) with the same
arguments; it returns the pass vector, which you can edit and pass explicitly.
Use an explicit `multipass_parameters(...)` schedule when you need per-pass
control beyond those overrides. `effort` and an explicit `PIVParameters` or
pass vector cannot be combined in one call.
