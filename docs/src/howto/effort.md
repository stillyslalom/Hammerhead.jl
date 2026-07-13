# Choose an effort level

**Goal:** pick a PIV schedule quickly, then override only the knobs your data
actually requires.

For routine runs, omit the explicit pass vector and use `effort`:

```julia
quick = run_piv(imgA, imgB; effort = :low)
draft = run_piv(imgA, imgB; effort = :medium)
final = run_piv(imgA, imgB; effort = :high)
```

The same keyword is accepted by [`run_piv_sequence`](@ref),
[`run_piv_ensemble`](@ref), and [`run_piv_stereo`](@ref). The current presets
are:

| Effort | Schedule | Options | Typical synthetic result |
|---|---|---|---|
| `:low` | `[32]` | [`PIVParameters`](@ref) defaults | 1× time; ~0.10 px uniform-shift RMS, ~0.43 px vortex RMS |
| `:medium` | `[64, 32]` | defaults | ~4× time; ~0.02 px uniform-shift RMS, ~0.22 px vortex RMS |
| `:high` | `[128, 64, 32]` | `padding = true`, `apodization = :gauss`, `max_iterations = 2`; final pass `uncertainty = true` | ~32× time; ~0.02 px uniform-shift RMS, ~0.09 px vortex RMS |

Those numbers are typical single-threaded timings on 256×256 synthetic pairs,
not guarantees for every seeding density or flow.

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
size and rescales the pyramid, while `uncertainty`, `max_iterations`, and
`keep_correlation_planes` apply only to the final pass. Use `final = (;)` when
you need an explicit last-pass override; it wins over both the preset and the
field keywords.

Final window size is a physics and image-quality choice. Smaller windows give
denser vectors, but they also contain fewer particles and can be noisier; the
default `:high` preset stops at 32 px for that reason. Use `window_size = 16`
only when the seeding density and signal quality support it.

Use an explicit `multipass_parameters(...)` schedule when you need per-pass
control beyond those overrides. `effort` and an explicit `PIVParameters` or
pass vector cannot be combined in one call.
