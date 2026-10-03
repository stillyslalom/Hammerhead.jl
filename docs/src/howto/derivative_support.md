# Where can you measure vorticity?

Vorticity uses differences between nearby velocity vectors. Removing one bad
vector therefore affects more than one location in the vorticity map. This
small example makes that effect visible before you try it on a recording.

## Put a gap in a rotating field

Use the analytic field `u = -y`, `v = x`. Its vorticity is exactly 2 in the
chosen coordinate and velocity units. Exclude the center vector, as if it lay
inside a reflection mask:

```@example derivative_support
using Hammerhead
using CairoMakie
x = collect(0.0:6.0)
y = collect(0.0:6.0)
u = [-yy for yy in y, xx in x]
v = [xx for yy in y, xx in x]
valid = trues(7, 7)
valid[4, 4] = false

available = flow_derivatives(x, y, u, v; valid, return_support=true)
centered = flow_derivatives(x, y, u, v;
    valid, stencil=:centered, return_support=true)
omega = vorticity(available)
omega_centered = vorticity(centered)
@assert all(value -> value ≈ 2, filter(isfinite, vec(omega))) # hide
@assert all(value -> value ≈ 2, filter(isfinite, vec(omega_centered))) # hide
nothing # hide
```

```@example derivative_support
fig = Figure(size=(850, 280))
for (column, values, title) in (
    (1, Float64.(valid), "Input vectors: one gap"),
    (2, Float64.(isfinite.(omega)), "Allow one-sided differences"),
    (3, Float64.(isfinite.(omega_centered)), "Require neighbors on both sides"))
    ax = Axis(fig[1, column]; title, xlabel="x", ylabel="y", aspect=DataAspect())
    heatmap!(ax, x, y, permutedims(values);
        colormap=[:lightgray, :teal], colorrange=(0, 1))
end
Label(fig[2, 1:3], "Teal: usable here     Gray: unavailable")
fig
```

Both methods recover 2 wherever they can compute vorticity. Their difference
is the usable area: the default can use a one-sided difference near the gap
and boundaries; `stencil=:centered` requires both immediate neighbors along
each axis. Neither jumps across the excluded center.

**Try it:** remove the gap with `valid[4, 4] = true` and rerun the two
calculations. Which gray locations remain, and why?

## Apply the same check to your result

For a PIV `result` with a physical scale attached:

```julia
velocity = physical(result)
d = flow_derivatives(velocity; return_support=true)
omega = vorticity(d)
(eligible_vectors=count(d.valid), usable_vorticity=count(isfinite, omega))
```

An eligible vector is not necessarily a usable derivative. Plot missing
vorticity as a gap, and compare the usable area when changing the stencil.
The result-based method excludes masked, nonfinite and flagged vectors by
default. On an irregular grid, the two-neighbor calculation is a secant;
it is not a general second-order derivative formula.

For individual contributor indices, spans and numerical-availability fields,
use the [derivative reference](../reference/derived.md#Derivative-support-fields).
The [GUI inspector](gui_derivative_support.md) lets you select a location and
see its neighbors. To build a time-dependent analysis, continue with
[flow statistics](../tutorials/sequence_statistics.md).
