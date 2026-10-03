# Inspect derivative support before interpreting a flow gradient

An accepted vector can lack usable neighbors for a derivative. The
[spatial-response study](spatial_transfer.md), for example, retained only 43/256
and 59/256 common interior vectors at its shortest wavelength; those counts
alone do not say how much vorticity has usable stencils. Compute actual support
from the field you analyze, keeping missing locations and arithmetic failures
visible.

```julia
d = flow_derivatives(result; return_support=true)
omega = vorticity(d)

eligible_centers = count(d.valid)
supported_x = count(d.support.x.structural_supported)
supported_y = count(d.support.y.structural_supported)
finite_vorticity = count(isfinite, omega)
```

Default `stencil=:available` uses both immediate eligible neighbors when
possible, with the exact neighboring secant
`(f[k+1]-f[k-1])/(axis[k+1]-axis[k-1])`. Otherwise it uses one eligible
immediate neighbor and the center in a one-sided quotient. No stencil skips
over an excluded neighbor. Even the two-neighbor secant requires an eligible
center, although that center has zero algebraic weight. At irregular spacing
this is not the general three-point derivative at the center; second-order
accuracy on a nonuniform grid is not claimed.

Use `stencil=:centered` to refuse one-sided fallback:

```julia
centered = flow_derivatives(result; stencil=:centered, return_support=true)
omega_centered = vorticity(centered)
```

Boundary and gap locations lacking both eligible neighbors remain `NaN`.
This is a requested analysis policy, not a claim that the remaining locations
are spatially resolved. It does not fill missing vectors. Both policies are
also accepted by result-based `vorticity`, `divergence`, `strain_rate`,
`swirling_strength` and `q_criterion` through their derivative keywords.

A small analytic field demonstrates the difference without processing images:

```julia
using Hammerhead
x, y = collect(0.0:4.0), collect(0.0:2.0)
u = [2xx + 3yy for yy in y, xx in x]
v = [5xx - 2yy for yy in y, xx in x]
eligible = trues(3, 5)
eligible[2, 3] = false

d = flow_derivatives(x, y, u, v; valid=eligible, return_support=true)
@assert count(d.valid) == 14
@assert count(isfinite, vorticity(d)) == 12

centered = flow_derivatives(x, y, u, v;
    valid=eligible, stencil=:centered, return_support=true)
@assert count(centered.valid) == 14
@assert count(isfinite, vorticity(centered)) == 0
```

The centered-only policy cannot combine both required axes at any location
on this small gapped grid. It keeps the same eligible centers; it does not
declare them all usable for vorticity or manufacture replacement neighbors.

The result method always excludes masks and nonfinite components, and excludes
flagged nodes unless `include_invalid=true`. The array method's explicit
`valid` mask is authoritative: it supplies eligibility without automatically
intersecting it with finite components. Returned `valid` is an owned eligibility
copy; `support.center_eligible` aliases that copy. Neither is a derivative
availability mask. Stored vectors are not automatically identified as original
measurements or replacements from their flags or finite values.

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
to produce misleading denominators is now refused. GUI support inspection,
measurement-origin association and uncertainty propagation are separate work.
