```@meta
CurrentModule = HammerheadGUI
```

# Inspect derivative support in the GUI

For a planar grid with at least two points per axis, choose **derivative support**
in the explorer's tool menu. Click a node to inspect its immediate contributors,
including excluded locations. The lower drawer shows counts and a paged explanation.

The field menu adds four maps while this tool is active:

| Map | Interpretation |
|---|---|
| Eligible centers | The center is unmasked, unflagged and has finite u/v. |
| x stencil / y stencil | Unavailable, neighbor secant, forward or backward, with a discrete legend. |
| Finite gradient components | Number of finite dudx, dudy, dvdx, dvdy quotients, from 0 to 4. |

The maps retain excluded/unavailable nodes as categories with fixed legends.
Manual scalar limits remain stored for your other fields. Gray marks nonfinite
derived output. Inspect both the gradient count and the selected quantity's
finite status.

**Available neighbors** uses a neighboring secant when both immediate neighbors
are eligible, or a one-sided quotient when one is eligible. **Require both
neighbors** selects `:centered`, keeping boundaries and gaps unavailable.
This policy persists across tools and frames; field labels show the selection.
Closed area-circulation contours use the policy, while profiles and line
circulation sample u/v directly. Stencils use immediate neighbors.

Try the tool on this analytic field:

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

Contributor details use `(row, column)` indices and displayed coordinates.
After physical conversion, spans use the length unit and gradients use inverse
time. Descending axes retain signed spans. A neighboring secant subtracts the
two neighbors and divides by their separation, with an eligible center required.
The [core stencil contract](derivative_support.md) gives the exact weights for
regular and irregular axes.

Inspect span/weight availability separately from finite quotients: a direct
quotient can remain usable when its reciprocal metadata weight overflows.
Current outlier flags exclude a node. For recorded rejection and replacement
events, open [recorded processing details](gui_companions.md).

Support uses the existing details drawer. Select another tool to return to
recorded details; close details to restore a profile graph. Frame changes release
the previous support arrays. Allow roughly 110 MiB per million nodes for rich
support and four gradients, plus the display and plot arrays.

After editing displayed arrays, reselect the support tool to rebuild its
analysis. Access checks the current analysis inputs; the
[core stencil contract](derivative_support.md) explains eligibility and arithmetic.