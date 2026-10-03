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
output. A previously checked demo drawing mode is ignored/cleared for file or
saved-experiment picking, so it cannot swallow result inspection. Saved
ROI/mask/preprocessing settings are preserved, not projected into
the demo's form. Saved planar replay executes in a core-only subprocess while
the Qt owner retains controls and its owned scientific screen. This worker
requires tested 64-bit Windows Job Object ownership; other hosts refuse before
processing. The synthetic batch remains cooperative in-process work; production
GUI workflows and dependencies remain unchanged. There is no experiment tree, ROI/preprocessing editor, resumability
or concurrent-writer support.

## Reproduction

The path fields also offer Browse. An accepted choice changes only the draft;
opening a record/result or replaying remains a separate explicit action. Save
choices do not create files or prompt about overwriting, because no write happens
until replay. Each chooser captures its own request token and purpose, so an old
chooser cannot accept into or dismiss a newer request. While a chooser is open,
root navigation, viewport close, replay and cancellation shortcuts are suspended.

The isolated dialog checks use actual Quick FileDialogs with
`DontUseNativeDialog` and `Popup.Item`, under offscreen/software Qt. They exercise
Qt URL conversion rather than manually decoding paths. Native OS dialogs,
desktop keyboard integration and other platform behavior remain unvalidated.

The local Windows dialog evidence in ignored `artifacts/file-dialog-Cw5sYt`
passed with Julia 1.11.4, Qt 6.10.2 and QML 0.13.2. The owned child exited zero
in 40.09 seconds without timeout. Thirty stages cover four actual choices,
Escape/rejection, each modal navigation/close/run key separately, same-picker
reopening followed by old-instance callbacks, busy/opening-failure refusal and
shutdown. OpenFile automation clicks the actual fallback list delegate; SaveFile
automation edits its actual filename control. These implementation object names
are a pinned Quick-fallback harness detail, not a portable native-dialog API.
All assertions and input hashing run in the owner loop after primitive callback
capture. Four original files remained byte-identical, three fresh destinations
remained absent, and the displayed frame 2 and selected vector were retained.

Independent checks verified all 54 prototype, 47 core and 31 GUI source hashes
against the final files, plus the three PNG digests. `dialog.png` shows a settled
real modal chooser; `small.png` and `large.png` show labelled paths and persistent
controls at 900 × 600 and 1100 × 800 window sizes. The image surfaces exclude the
40 px menu bar. The small sidebar intentionally scrolls explanatory text.
The focused checks passed 76 path/state, 15 evidence-refusal and 38 existing
process-owner assertions; the latter retain their original defaults.
The existing saved-planar software child also passed on these exact source maps
in ignored `artifacts/lifecycle-dNcVIZ`: exit zero, 99.80 seconds, no timeout,
with verified replay/cancellation and observer cleanup. Native dialog behavior
and the embedded Qt/OpenGL gate remain open.

```powershell
julia --startup-file=no --threads=1 --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/file_path_tests.jl
julia --startup-file=no --threads=1 --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/file_dialog_tests.jl
julia --startup-file=no --threads=1 --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/file_dialog_runner.jl --timeout=240
```

Run from the repository root:

```powershell
julia HammerheadGUI/prototypes/qml/setup.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/shell_actions_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/adapter_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/experiment_adapter_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/experiment_error_pages_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/experiment_transaction_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/viewport_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/test_lifecycle_contract.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/viewport_ownership_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/owned_glfw_tests.jl
julia --startup-file=no --threads=1 --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/worker_protocol_tests.jl
julia --startup-file=no --threads=1 --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/worker_client_tests.jl
julia --startup-file=no --threads=1 --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/worker_replay_tests.jl
julia --startup-file=no --threads=1 --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/worker_lifecycle_tests.jl
julia --startup-file=no --threads=1 --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/worker_lifecycle_runner.jl --timeout=480
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/lifecycle_runner.jl --self-test
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/lifecycle_runner.jl --cases=shell-software --timeout=180
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/lifecycle_runner.jl --cases=shell-experiment-software --timeout=240
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/lifecycle_runner.jl --cases=shell-glfw,shell-experiment-glfw --timeout=240
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/lifecycle_runner.jl --cases=construction,baseline,single-context --timeout=90
```

The runner creates hidden child processes with Qt's offscreen platform and an
explicit software or RHI/OpenGL adaptation in their startup environment. It
normalizes Windows environment keys and removes inherited selectors, including
`QMLSCENE_DEVICE`. Use it for automated Qt validation instead of relying on
Julia ENV changes reaching Qt's separate Windows C runtime. Scientific ownership
tests request invisible GLFW screens explicitly. No desktop probe is automated.
An explicit `run.jl --desktop` remains an unvalidated interactive opt-in.

## Independent interactive scientific window

`run.jl --plot=glfw` uses Qt software controls with a separately owned OpenGL
GLMakie window. It preserves the same demo and saved-planar controllers,
physical units, selection and displayed-run identity. The settings window shows
the identity and ownership status; the scientific figure is not a static Qt
preview. Automatic non-demo limits include the full normalized arrow vertices,
including singleton axes, while preserving subsequent manual pan/zoom. The
demo keeps its original image-domain limits. The default remains the offscreen embedded diagnostic, `--software`
selects the existing static preview, and `--plot=preview` / `--plot=embedded`
name those modes explicitly.

The new mode allocates a dedicated GLFW screen rather than Makie's singleton;
an unrelated preexisting screen is preserved. One owner thread pumps Qt events,
drains queued controller actions, then polls GLFW, updates render objects,
renders and swaps buffers. No background GLFW rendering task is started.
Closing the plot is observed after polling returns and synchronizes the Qt
controls; one reopen action creates a fresh figure and screen. Plot closure
leaves replay owned by the shell. Shutdown requests cancellation, waits for
processing cleanup with a deadline, then disposes callbacks, the owned screen
and Qt. A cleanup error is recorded separately without replacing an original
processing/render failure. If processing remains busy after the deadline, the
child fails without claiming safe model/context disposal. The desktop flag may
make both windows visible, but no visible desktop
trial is part of the automated checks.

QMLMakie and its QML plugin remain loaded by the shared shell. This mode does
not construct a `MakieArea` or attach the figure to a Qt screen; it avoids bridge
rendering, rather than removing the candidate dependency. A passing independent
GLFW lane does not satisfy the embedded native-Qt rendering/release gate.

The hidden `shell-glfw` and `shell-experiment-glfw` children capture Qt controls
as `framebuffer.png` and the actual independent scientific framebuffer as
`scientific.png`, with a digest recorded in `shell_report.toml`. Their gates
require rendered frames, hidden screens, no background renderer, matching
generation/release counts, restored screen count, no retained old figures,
disposed shell observers, completed cleanup and zero final process exit.
The demo also checks a simulated unsolicited close followed by exactly one
reopen. Logged nonthrowing render-object update failures fail the evidence gate.

Focused checks deliver scroll, right-drag pan, selection and demo-mask events
through Makie's event observables, and exercise real saved replay/cancellation,
physical selection and retained display after failure. They are synthetic event
routing checks, not native mouse/keyboard or desktop focus evidence. Hidden
OpenGL screens still require a working graphics context; Linux validation would
need a display such as Xvfb plus compatible GL support. Windows desktop,
other platforms, HiDPI/accessibility, Qt GPU embedding and packaging remain open.
The recorded maximum pump gap is a workload observation; preflight/computation,
I/O, JIT and concurrent validation can delay both event loops. It is not an
interaction-latency benchmark or a claim of responsive processing. Frame counts
in these reports count explicit owner-pump renders; capture can perform another
render and is not included in that count.

The pre-worker local Windows evaluation on 2026-10-03 passed **87 focused independent
window checks** and **38 process/evidence checks**. All four requested children
in the ignored local `artifacts/lifecycle-hoXTGy` directory exited zero:
independent demo/saved experiment and the existing static-preview demo/saved
experiment. No timeout, retained observer/replay, render-error pattern or
source drift was recorded. The independent demo completed four generations and
four releases, including simulated close generation 1 followed by one reopen
at generation 2. Its saved-experiment counterpart completed five generations
and five releases, cancelled after one written pair, reran all three pairs,
inspected the completed output and retained its displayed identity after an
open error. Both restored the screen registry and released old figures.
Inspected paired captures show all arrowheads inside automatic limits and
readable physical mm/mm/s selection at frame 3/3. No visible window was launched.
Recorded maximum pump gaps of about 2.95 s (demo) and 1.43 s (saved experiment)
include JIT/workload and concurrent validation; they do not establish interactive
latency or a performance target. These results support the independent-window
candidate only; they do not close the embedded native-QML or desktop-input gate.
The source scope is the recorded maps in `lifecycle-hoXTGy`, before subprocess
replay integration; these captures are not current worker-source acceptance.

`setup.jl` resolves only this environment and restores portable project source
paths. Its Manifest is ignored. If the local core gains dependencies, refresh
this environment offline with `Pkg.offline(true); Pkg.resolve()` before checking
application imports. `artifacts/` contains ignored reports, logs and captures;
no generated image, downloaded binary or machine-specific manifest belongs in
a commit.

## Lifecycle evidence contract

The worker request is detached before observers or Qt actions can change next
settings. One pair-write progress event is outstanding at a time; the owner
applies it and acknowledges it explicitly, without waiting inside the Qt event
callback. Cancellation at an earlier write retains the ordinary planar failed
prefix. Cancellation at the final write completes; an explicit observer abort
can fail even then. Native bytes and total write counts do not invent a returned
completed run after a history-save failure. Terminal status is checked separately
from final OS exit, and a missing or crossed terminal packet fails the request.

`worker_lifecycle_runner.jl` retains the existing process owner and adds worker,
compact-sidebar and active-control checks. An explicitly injected child barrier
keeps the worker alive while controls and scientific callbacks are serviced;
this interval is not PIV work. Actual PIV intervals are recorded separately and
include only the exclusions explicitly identified in the report. A warmed parent
does not establish a warmed fresh worker. There is no universal desktop or
interaction-latency claim. An outer timeout or unverified worker lifetime leaves
an incomplete-owner marker and forbids later children; direct Qt-child reaping
does not alone establish descendant cleanup.

The Windows ownership fixture holds a synchronization handle to the exact live
worker before abruptly terminating its test owner, then verifies worker exit.
Its scope is an enrolled injected waiting boundary, not mid-PIV interruption or
native-file recovery. A separate injected enrollment failure verifies an
unassigned child is reaped before another launch. Equivalent non-Windows process
ownership remains open. The manual matrix retains unsupported worker capability
as a failure rather than falling back to the legacy cooperative lane.

The current Windows checks pass 63 protocol, 48 client/ownership, 90 real replay
and 26 lifecycle-validator assertions. The isolated enrollment-failure helper
passes seven checks separately. GUI focused checks total 370 across separate
invocations: 97 final cached-error/adapter checks, 39 display transactions,
147 other contracts and 87 owned GLFW checks. Only the 97 were rerun after the
last error-cache repair; the total is not a single final-source invocation.

The corrected active-worker child in ignored `artifacts/worker-lifecycle-KemBhR`
exited zero in 97.33 seconds. Its 46 prototype source hashes and executed core/GUI
source maps remained unchanged. Twenty stages retain exact owner/worker identities,
including nine injected-barrier interaction stages and three actual native writes.
The owner services controls, rendering, pick/pan/zoom and close/reopen while that
barrier is active, retaining the labelled prior physical result. The original
failed-open error appears in cached text before navigation and is reconstructable
across the inspection pages. Active/small/large controls and the scientific plot
were visually inspected; all five PNG digests match the report. Persistent
cancel/progress controls fit, both scroll areas reach their bottom, four GLFW
generations/releases restore the screen registry, and no old figures,
subscriptions or replay remain after disposal. The provisional
`worker-lifecycle-8mtkLy` capture precedes the error-cache fix and is superseded.

Recorded owner service from request-startup return to joined terminal spans
29.261 seconds with a maximum pump gap of 1.051 seconds. This includes injected
waiting work and fresh-worker import/JIT/rendering; it excludes prior fixture
replay and request capture/spawn. From first native-write acknowledgement to
joined terminal, the interval is 2.397 seconds with maximum gap 0.036 seconds;
imports and the first pair are excluded, but later JIT and I/O may remain.
These are workload observations, not desktop latency targets, portable
responsiveness or proof of a fully warmed fresh worker.

The final saved-experiment static-preview and separate-GLFW children in
`artifacts/lifecycle-qhqTp3` also exited zero (95.62 and 94.98 seconds), without
timeouts or render/model/teardown errors. They cancelled, reran three pairs and
retained the labelled completed display through a failed open. Both completed
five viewport generations; the GLFW case released all five screens, restored
the registry and cleared old figure references. Their 46 prototype source hashes
match the active-worker child exactly, with unchanged recorded core/GUI sources.
The old demo captures remain pre-error-fix evidence; they were not rerun as part
of this final saved-planar acceptance.

The saved-experiment smoke writes actual temporary PNG inputs and a saved recipe
with ordered passes, preprocessing, ROI, mask and physical scale. It closes a view
while replay is owned by the shell, cancels after the first native pair write,
reruns to completion (a last-pair cancel completes), inspects/navigates/picks
physical output, and exercises retained display after an open error. Shutdown
requests cancellation and pumps until both controllers finish cleanup before
shell subscriptions and the viewport are disposed. Snapshotting, startup,
result I/O and rendering can still pause the Qt owner; saved-planar pair work
executes in its worker. This is not a universal responsiveness benchmark or
checkpoint workflow.

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

## Pre-worker ownership repair and native blocker

The focused counts and software captures below describe the earlier ownership
repair, before the saved-planar subprocess lane. Their retained source maps
define that historical scope; current worker evidence is recorded separately.

Earlier preview/bridge checks passed 250 focused assertions: 11 queued-action, 31 demo-adapter,
64 saved-experiment adapter, 26 display-transaction, 13 physical-geometry,
15 viewport, 30 scientific ownership and 60 ownership-contract checks.
The earlier process/environment harness passed 24 assertions, including refusal of
leftover subscriptions or replay work. Released leases retain neither
subscriptions nor their old figures; invisible GLFW cycles return the screen
registry to baseline.

The then-final software child exits zero with five fresh viewport generations, five
application releases including shutdown, cancellation after one of three pairs,
and no render/model/teardown errors. Its capture was inspected: Qt controls and
scientific preview are readable. An explicit FontLoader uses Makie's existing
TeX Gyre Heros asset; its path, SHA-256 and loaded family are recorded. The direct
shell accepts `--font=<path>` for an existing alternative font; no font is copied
or installed. These are software/application ownership results, not native Qt
GPU, gesture or performance evidence.

The historical saved-experiment software child also exits zero after five generations.
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
without starting a background GLFW loop. Those source-stable children have no recorded
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
