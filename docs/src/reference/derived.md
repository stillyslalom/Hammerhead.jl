```@meta
CurrentModule = Hammerhead
```

# Derived flow analysis

Use the functions below for profiles, regions, circulation, spatial
derivatives, and time spectra. Derivatives use immediate valid neighbors;
they do not cross a mask or excluded outlier. The 2D swirling-strength and
Q values describe the measured in-plane gradient tensor and do not include
unmeasured 3D terms.

[`flow_derivatives`](@ref) preserves the existing neighboring-secant and
one-sided quotients with `stencil=:available`. Use `stencil=:centered` to
require both immediate neighbors; boundaries or gaps without them remain
unavailable. On nonuniform spacing the two-sided formula is the exact secant
between neighbors, not the general three-point derivative at the center.
Axes must be finite and strictly monotonic, with native adjacent/two-neighbor
spans that are finite, nonzero and preserve direction. Geometry is validated
regardless of masks or stencil policy.

`return_support=true` adds actual per-axis contributor indices, stencil kinds,
signed spans and algebraic weights. Center eligibility, structural support,
representable metadata and finite component output are distinct: `valid`
describes eligible input centers, while `support.finite` describes each returned
gradient. A valid direct quotient can exist even when the reciprocal span
overflows, so unavailable weights do not invalidate its derivative. Metadata
uses a promoted floating geometry type; calculation retains native subtraction
and division order. The [support guide](../howto/derivative_support.md)
illustrates the effect of missing neighbors. The support schema is listed below. This supplies stored
value support, without measurement-origin or uncertainty-calibration claims.

`extract_profile` interpolates only from corners with positive weight. A
masked or invalid corner does not invalidate an exact node or edge sample
when its weight is zero; a contributing invalid corner yields `NaN`.
`extract_region` returns `included`, a Boolean grid where `true` marks a
returned node. Its legacy `mask` field aliases the same inclusion grid;
`result.mask` uses the opposite convention (`true` means excluded).

Area-form circulation integrates vorticity over the portion of each grid cell
inside a requested rectangle or polygon, including fractional boundary cells.
Cells with a masked, excluded outlier, or non-finite vorticity corner are
omitted; `include_invalid=true` admits outliers but still excludes masks.
By default, an incomplete requested region raises an error rather than
returning an unmarked partial integral. Use `coverage=:report` to receive
`(; value, valid_area, requested_area, coverage_fraction, complete)`. When
some area is missing, `value` integrates only the valid area; when none is
valid, it is `NaN`. Check `complete` to decide whether the full region was
integrated; a displayed fraction can round to 1 even with a small gap.

For [`result_spectrum`](@ref), pass `dt` as the interval between successive
results. `PhysicalScale.dt` is the delay between images within each pair;
it converts displacement to velocity and may differ from the sequence
cadence. Alternatively provide explicit `sample_times`; the
[spectrum timing guide](../howto/spectrum_timing.md) describes exact uniformity
checks, explicit tolerances and optional timing provenance. Results must share
grid dimensions, scale factors and unit labels. Stored values are analyzed
without conversion; sampling-unit labels stay independent of component units.
Invalid interpolation operates on the accepted regular FFT grid, not on a
separately resampled irregular timeline.

## Derivative support fields

Opt-in support contains `policy`, `center_eligible`, `x`, `y` and `finite`.
Each axis has this schema:

| Field | Meaning |
|---|---|
| `dimension` | Array dimension: x=2 (columns), y=1 (rows). |
| `kind`, `legend` | Grid-shaped UInt8 codes: unavailable=0, centered secant=1, forward=2, backward=3. |
| `first_index`, `second_index` | Contributor indices along that axis; the other grid index is unchanged. Index 0 means no stencil. |
| `signed_span` | Second coordinate minus first coordinate, preserving descending-axis signs and supplied coordinate units. |
| `first_weight`, `second_weight` | Algebraic coefficients `(-1/span,+1/span)`, not a different numerical evaluation rule. |
| `structural_supported` | Eligible center and neighbors exist under the requested policy. |
| `span_available`, `weights_available` | Descriptor/reciprocal arithmetic is representable. Unavailable descriptors are `NaN`, not fabricated zeros. |

`support.finite` contains separate Boolean matrices `dudx`, `dudy`, `dvdx`
and `dvdy`, equal to finite availability of the four returned gradients.
Component arithmetic can fail while geometric support remains valid.
Combinations such as vorticity, strain magnitude or Q must also check their
own finite output: finite input gradients do not prevent later overflow.

Axes need at least two finite real values and must be strictly increasing or
decreasing. All adjacent and two-neighbor spans are validated before output
calculation, independently of masks or policy. A nonfinite, zero,
direction-inconsistent or overflowing native integer coordinate span raises
`ArgumentError`. There is no silent promotion that repairs an overflowing
coordinate denominator. Integer component-subtraction overflow instead gives
an unavailable `NaN` gradient; floating component arithmetic retains its native
nonfinite result and has a false finite-availability mask.

Descriptor types promote floating native spans with `Float64`, retaining
`BigFloat`. Actual gradients retain the existing Float64 output convention and
native difference/division order, rather than multiplying metadata weights.
Reciprocal overflow does not discard a usable direct quotient. For example,
`nextfloat(0.0)/nextfloat(0.0)` is 1, although the reciprocal coordinate span
overflows; the derivative and structural support remain available while
weights are `NaN` and `weights_available=false`.

Stored arrays are analyzed directly. Call `physical(result)` first for physical
velocity gradients; spans then use the physical coordinate units, gradients
use velocity per coordinate unit, and metadata indices still address the same
grid. Attaching a scale alone does not convert stored values. No calibration,
timing or random-uncertainty contributions are propagated by this API.

The default returns exactly `(; dudx, dudy, dvdx, dvdy, valid)` and does not
allocate the rich support matrices. Opt-in support stores only the current
grid's maps and owned snapshots, without a sequence cache or new dependencies.
Existing regular-grid arithmetic is preserved; malformed geometry that used
to produce misleading denominators is now refused. The
[GUI support inspector](../howto/gui_derivative_support.md) displays these maps and
contributors from the current physical display. Measurement-origin association
and uncertainty propagation remain separate analyses.

```@index
Pages = ["derived.md"]
```

```@autodocs
Modules = [Hammerhead]
Pages = ["derived.jl"]
Private = false
```
