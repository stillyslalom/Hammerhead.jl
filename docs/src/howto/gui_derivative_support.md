```@meta
CurrentModule = HammerheadGUI
```

# Inspect derivative support in the GUI

Choose **derivative support** in the result explorer's tool menu for a planar
grid with at least two points per axis. The lower details drawer shows support
counts and a paged explanation of the selected node. Click a node to inspect
its immediate contributors, including excluded locations. Other result kinds
and singleton axes have no supported planar derivative analysis.

The field menu adds four maps while this tool is active:

| Map | Interpretation |
|---|---|
| Eligible centers | The center is unmasked, unflagged and has finite u/v. |
| x stencil / y stencil | Unavailable, neighbor secant, forward or backward, with a discrete legend. |
| Finite gradient components | Number of finite dudx, dudy, dvdx, dvdy quotients, from 0 to 4. |

Excluded/unavailable nodes remain visible as categories; they are not erased
by the scalar color-range filters. Manual scalar color limits remain stored
while these fixed legends are displayed. In a derived scalar field, gray marks
nonfinite output while the support tool is open. Four finite gradients still
do not guarantee that a combined quantity such as Q is finite.

**Available neighbors** preserves the default derivative arithmetic: two
eligible immediate neighbors give their secant, otherwise one eligible
neighbor permits a one-sided quotient. **Require both neighbors** selects
`:centered`, refusing fallback at boundaries and gaps. The policy persists
across tools and frames; derived-field labels show the two-neighbor requirement
even after closing support details. Closed area-circulation contours recompute
under the policy. Profiles and line circulation sample u/v and remain independent
of derivative stencils. No excluded neighbor is skipped to find a farther one.

For example, this analytic controller setup needs no window or image processing:

```@example gui_derivative_support
using Hammerhead, HammerheadGUI

x = collect(0.0:4.0)
u = [2xx + 3yy for yy in x, xx in x]
v = [5xx - 2yy for yy in x, xx in x]
flags = falses(5, 5)
flags[3, 3] = true
result = PIVResult(x, copy(x), u, v, ones(5, 5), ones(5, 5),
                   fill(NaN, 5, 5), fill(NaN, 5, 5), flags, falses(5, 5),
                   PIVParameters(window_size=16, overlap=8))
explorer = ResultExplorer(result)
set_tool!(explorer, :derivative_support)
set_derivative_stencil!(explorer, :centered)
set_field!(explorer, :vorticity)
select_nearest!(explorer, 1.0, 2.0)
summary = derivative_support_summary(explorer)
(eligible=summary.eligible, x_supported=summary.x_supported,
 finite_vorticity=count(isfinite, HammerheadGUI.Controllers.current_field_values(explorer)))
```

The contributor explanation uses `(row, column)` grid indices and the displayed
coordinates. After physical conversion, spans use the length unit and gradients
use inverse time; the conversion happens once in the explorer. Descending axes
keep signed spans. A centered neighbor secant excludes the center's algebraic
value while still requiring center eligibility. On irregular grids it is not
the general quadratic three-point derivative at the center and carries no
second-order accuracy claim.

Representable spans and weights are separate from finite quotient outputs.
Unrepresentable reciprocal weights can coexist with a usable native quotient.
A current outlier flag means exclusion, not proof of replacement. Support for
stored displayed values does not establish measurement origin, spatial
resolution, propagated uncertainty or uncertainty applicability. Use
[recorded processing details](gui_companions.md) for actual persisted events.

The support tool occupies the existing details drawer. Recorded details return
when another tool is selected if their toggle remains enabled. Profiles keep
their existing vector sampling semantics. Rich support metadata is released
when leaving the tool and evicted on frame changes; stencil policy persists.
Each complete display result must fit memory. Rich Float64 support and four
gradient arrays cost roughly 110 MiB per million nodes, excluding the display,
plots and temporary scalar maps. No per-recording support cache is retained.

Inspection hashes current displayed analysis inputs in O(nodes) to detect
mutation when accessed, rather than continuously monitoring them. If the
display arrays were edited, reselect the support tool to rebuild the analysis.
These checks do not reload or certify the original images. See the
[core stencil contract](derivative_support.md) for arithmetic and eligibility
details.
