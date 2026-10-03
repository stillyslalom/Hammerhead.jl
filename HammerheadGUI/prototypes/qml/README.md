# Isolated Qt6/QML shell evaluation

This opt-in prototype evaluates forms and window management over HammerheadGUI's
framework-free controllers. It does not replace the production GUI or change its
dependencies. See [the framework evaluation](../../../docs/src/explanation/gui_framework.md)
for requirements and adoption gates.

The shell supports synthetic raw planar batches, a 16,384-vector demo over a
1024-square image, completed-file lazy browsing, picking, mask commands, inline
errors, cancellation, and integrated/separate views. Browsing accepts unscaled
planar PIV entries only; unsupported entries fail before replacing the display.
The separate saved-experiment lane preserves the complete planar recipe with
`ExperimentController`: explicit record/result/history paths, environment policy,
paged recipe/history, written-pair progress, replay/cancel, and verified completed
run inspection. Saved planar output displays physical units through the existing
explorer conversion. Generic completed-file browsing retains its unscaled-only
restriction. Active recipe/input identity and displayed run/output identity are
shown separately; failure retains a labelled previous display. Referenced scripts
are inspectable but never loaded/executed by this shell. The production explorer
supports all four result types and dedicated actual-time tracking artifacts.
Masks belong to the synthetic 96-square demo and are hidden/refused on experiment
output. Saved ROI/mask/preprocessing settings are preserved, not projected into
the demo's form. There is no experiment tree, ROI/preprocessing editor, resumability
or concurrent-writer support.

## Reproduction

Run from the repository root:

```powershell
julia HammerheadGUI/prototypes/qml/setup.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/shell_actions_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/adapter_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/experiment_adapter_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/experiment_transaction_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/viewport_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/test_lifecycle_contract.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/viewport_ownership_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/lifecycle_runner.jl --self-test
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/lifecycle_runner.jl --cases=shell-software --timeout=180
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/lifecycle_runner.jl --cases=shell-experiment-software --timeout=240
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/lifecycle_runner.jl --cases=construction,baseline,single-context --timeout=90
```

The runner creates hidden child processes with Qt's offscreen platform and an
explicit software or RHI/OpenGL adaptation in their startup environment. It
normalizes Windows environment keys and removes inherited selectors, including
`QMLSCENE_DEVICE`. Use it for automated Qt validation instead of relying on
Julia ENV changes reaching Qt's separate Windows C runtime. Scientific ownership
tests request invisible GLFW screens explicitly. No desktop probe is automated.
An explicit `run.jl --desktop` remains an unvalidated interactive opt-in.

`setup.jl` resolves only this environment and restores portable project source
paths. Its Manifest is ignored. If the local core gains dependencies, refresh
this environment offline with `Pkg.offline(true); Pkg.resolve()` before checking
application imports. `artifacts/` contains ignored reports, logs and captures;
no generated image, downloaded binary or machine-specific manifest belongs in
a commit.

## Lifecycle evidence contract

The saved-experiment smoke writes actual temporary PNG inputs and a saved recipe
with ordered passes, preprocessing, ROI, mask and physical scale. It closes a view
while replay is owned by the shell, cancels after the first native pair write,
reruns to completion (a last-pair cancel completes), inspects/navigates/picks
physical output, and exercises retained display after an open error. Shutdown
requests cancellation and pumps until both controllers finish cleanup before
shell subscriptions and the viewport are disposed. Cooperative tasks can pause
the Qt event loop during preflight, pair work or cleanup; this is not a
responsiveness benchmark or checkpoint workflow.

Preview arrows normalize their display length to grid spacing; position axes
and selected component values retain their true units. Physical padding derives
from coordinate spacing/span, without a fixed distance in arbitrary units.
Arrow normalization uses available axis spacings; a fully singleton grid uses
the same local extent policy as padding. Demo
images/polygons are absent on file/experiment plots. Fallback picking accounts for
image letterboxing. A failed plot refresh restores the previous model and view;
if restoration fails, the plot is hidden and labelled unavailable until a
successful open/inspection. Plot leases own figures/pick subscriptions; shell
subscriptions and replay lifetime survive view closure.

The runner preserves invocation, whitelisted Qt environment variants, prototype
source hashes before/after each child, numbered flushed stage files, complete
stdout/stderr, captures and OS process status in a fresh
`artifacts/lifecycle-*` directory. That directory survives parent exit.
Timeout termination targets only the child it owns. The summary is written
before returning nonzero when any requested case fails or is gated. Flushing
evidence does not establish power-loss durability. Source/QML/manifest edits
during a child invalidate its acceptance evidence.

Pass requires the expected generations, first frames with attached GL screens,
release acknowledgements and detachment, nonempty PNG capture, disposed
subscriptions, zero retained figures, baseline screen counts where native
release is requested, no recorded render/teardown errors, and zero final OS
exit status. Capture or a pre-shutdown completion marker alone cannot pass.
Inspect captures as well as their signatures and stages.

Native cases are `construction`, `baseline`, `single-context`, `single-observe`,
`reopen-context`, `separate-context`, `resize-context`, `failure-context`
and `reuse-observe`. Observe mode acknowledges application disposal only.
Context mode attempts GPU disposal from the render callback after checking its
Windows wgl context handle. Reopen/separate/resize/failure/reuse cases are skipped
until a clean single-render/release/exit case passes; a requested matrix adds
that prerequisite. `--cycles=20` is available for later validation, but the
current Windows platform cannot pass the native prerequisite.

Native probes import QML/QMLMakie/GLMakie/Observables only. Resolved core/GUI
dependency hashes describe declared environment packages, not imported code.
Parent drift checks cover the prototype sources/QML/manifest. Diagnostic elapsed
times under concurrent tests are not startup benchmarks. No depot package or
production dependency is modified.

## Current blocker and ownership repair

Current checks pass 248 focused assertions: 11 queued-action, 30 demo-adapter,
63 saved-experiment adapter, 26 display-transaction, 13 physical-geometry,
15 viewport, 30 scientific ownership and 60 ownership-contract checks.
The process/environment harness passes 24 assertions, including refusal of
leftover subscriptions or replay work. Released leases retain neither
subscriptions nor their old figures; invisible GLFW cycles return the screen
registry to baseline.

The final software child exits zero with five fresh viewport generations, five
application releases including shutdown, cancellation after one of three pairs,
and no render/model/teardown errors. Its capture was inspected: Qt controls and
scientific preview are readable. An explicit FontLoader uses Makie's existing
TeX Gyre Heros asset; its path, SHA-256 and loaded family are recorded. The direct
shell accepts `--font=<path>` for an existing alternative font; no font is copied
or installed. These are software/application ownership results, not native Qt
GPU, gesture or performance evidence.

The saved-experiment software child also exits zero after five generations.
It records cancellation after one written pair, a completed three-pair rerun,
view release while replaying, and retained display after an open error. Loading
the completed history leaves the controller ready with zero current-run progress;
the separately recorded historical run remains completed with three pairs.
Final captures show readable recipe/ROI controls, displayed identity, physical
component units and vector picking. Both software cases require no remaining
shell subscriptions and no running replay after disposal. Their reports verify
unchanged prototype, core and GUI source maps before/after execution.

Exploratory saved-replay children faulted in Qt's JS-stack collection while a
GLFW render loop polled events. Those logs are preserved. The software path now
copies Qt callback arguments into Julia primitives and queues heavy actions
outside JS callbacks. It renders an explicitly owned hidden screen directly,
without starting a background GLFW loop. The final children have no recorded
render/model/teardown errors and exit zero; this does not resolve the native
bridge gate below.

Explicit startup RHI/OpenGL reaches a platform failure on Windows:
`This plugin does not support createPlatformOpenGLContext!`,
`QRhiGles2: Failed to create context` and `Failed to create RHI (backend 2)`.
The owned children time out without a native first frame. Earlier diagnostic
trials selected Qt's software adaptation, produced inspected black PNGs and
acquired zero GL screens; the native gate rejects those. This capability blocker
precedes bridge teardown, so the new trial cannot establish safe native cleanup.

The existing `run.jl` / `main.qml` ownership path now creates a fresh figure for
each viewport generation, disposes its mouse subscription and refresh callback,
clears its owned Makie current-figure reference, and serializes integrated/
separate handoff. Acknowledgements certify application ownership only. Hidden
GLFW preview screens have an implemented context switch; Qt native screens
do not gain that guarantee. Retaining a released lease must not retain its old
figure. QMLMakie's destruction hook, context-valid flag and no-op context switch
remain separate upstream risks; the local opt-in trial is not a production fix.

## Historical evidence

Observed Windows versions: Julia 1.11.4, QML 0.13.2, QMLMakie 0.3.3,
CxxWrap 0.17.5, jlqml_jll 0.10.4+0, Qt6Base_jll/Qt6Declarative_jll 6.10.2+2,
GLMakie 0.13.15, Makie 0.24.15 and HammerheadGUI 0.1.1. These are observed versions,
not a compatibility matrix. Linux/macOS, minimum Julia, packaging, HiDPI,
accessibility and native input latency remain untested.

Before the ownership repair and enforced process environment, adapter tests
passed 30/30 and a software shell exited 0 after five viewport creations,
form/open errors and cancellation after one pair. One historical warm run
measured import 8.99 s, QML load 0.40 s and total 30.66 s; estimated controller/
figure sizes were 231,249/22,085,754 bytes and peak RSS about 1.76 GiB including
JIT/workload. The initial bridge import took 235.66 s including precompilation.
These are historical single-machine observations, not current-source results
or performance comparisons.

Historical `bridge_framebuffer.png` contains a real embedded plot, but shutdown
reported ModernGL.ContextNotAvailable during glDeleteBuffers. The native shell
printed `exception in render` during reopen and faulted during context
destruction on exception unwind. Those scripts recorded Julia environment flags
without verifying Qt's native C-runtime platform/backend; their captures do not
establish an offscreen native lifecycle. Their earlier bridge errors remain
unresolved evidence, not a demonstrated consequence of the current platform.

Programmatic viewport tests cover zoom/pan, picking, mask overlays, reversed
image coordinates and file geometry; they do not certify native gestures.
The software shell is a static scientific preview refreshed after controller
actions, not GPU bridge parity. Keep the production GLMakie shell while native
lifetime, input, accessibility, distribution and platform gates remain open.
