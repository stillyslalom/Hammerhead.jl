# Desktop GUI framework evaluation

HammerheadGUI currently uses GLMakie views over framework-free controllers.
The next desktop shell needs stronger forms, file navigation, focus handling,
and window management while preserving the existing analysis and result
contracts. The first implementation scope is planar PIV, including physical-unit
inspection of saved experiment output. Stereo, PTV and tracking remain supported
by the production GUI; the isolated candidate does not reproduce those views.

## Interaction requirements

| Requirement | Acceptance exercise | Candidate evidence |
|:--|:--|:--|
| Experiment and file browser | Open a completed recording; show path/read errors beside input; retain previous result on failure | Prototype has explicit path entry and lazy open; experiment tree remains pending |
| Editable parameter forms | Invalid schedules leave controller parameters unchanged; valid correction clears the inline error | Controller adapter tested; native QML TextField and error label wired |
| Large recordings | Browse indexed completed files while retaining one display payload and current view arrays | Existing ResultFile/ResultExplorer reused; O(number of entries) key index; no live-file tailing |
| Keyboard and focus | Tab through labeled inputs; arrow keys browse; Esc cancels; Ctrl+Return runs; Ctrl+W reopens viewport | QML bindings and accessible input names present; native input/accessibility audit pending |
| Layout and windows | Resize panels; compare settings plus separate visualization with an integrated layout using the same controllers | Static-preview and separate-GLFW children exit cleanly; embedded Qt bridge lifetime remains blocked |
| Image navigation | Pan/zoom a 1024-square image and retain precise data-coordinate picking | Makie axis interactions in bridge; image-transform fallback implemented; native gesture latency pending |
| Dense vectors | Render 16,384 vectors over an image, inspect one vector, avoid frame-history caches | Hidden standalone and real embedded framebuffer captures inspected; native bridge reports render exceptions during reopen |
| Masks and ROI | Draw/close a polygon on the demo image; edits reach the batch controller; retain production ROI semantics | Prototype mask adapter tested; native mask gesture/ROI editor parity pending |
| Live results and cancellation | Completed pairs update the explorer; cancellation preserves the completed prefix while controls remain serviceable | Synthetic BatchRunner remains cooperative; saved planar replay uses an owned Windows worker with acknowledged native-write boundaries; portable and desktop responsiveness remain pending |
| Saved recipe replay | Preserve the full saved recipe; explicit output/history paths and environment policy; progress/cancel; verify completed run before lazy inspection | Core-only saved planar worker passes hidden static-preview and separate-GLFW checks; no automatic script execution, ensemble replay or checkpoint resume in this prototype |
| HiDPI and accessibility | Test 100/150/200% scaling, focus indicators, screen-reader labels, menus and shortcuts | Pending real desktop checks |
| Resource lifetime | Close/reopen views repeatedly; exit without stale GL contexts or leaked payloads | Static-preview and dedicated GLFW screens release cleanly in hidden trials; embedded Qt shutdown has exposed GL cleanup errors |
| Distribution and compatibility | Record startup/memory and build/install on supported Julia versions and each target OS | Windows Julia 1.11.4 tested; manual three-OS workflow prepared but not dispatched; other-platform results, lower Julia versions and packaging pending |

## Isolated Qt6 candidate

The opt-in source lives in `HammerheadGUI/prototypes/qml/`, outside the
production dependency graph. It combines Qt Quick Controls with existing
`BatchRunner`, `ExperimentController`, `ResultExplorer`, and `MaskEditor` controllers. QML functions
and property maps are registered at runtime. This follows the upstream
[QML.jl callback/property-map contract](https://juliagraphics.github.io/QML.jl/dev/).
The native plotting component follows
[QMLMakie's MakieArea embedding API](https://github.com/JuliaGraphics/QMLMakie.jl).

The measured resolver result on Windows used Julia 1.11.4, QML 0.13.2,
QMLMakie 0.3.3, CxxWrap 0.17.5, jlqml_jll 0.10.4+0, Qt6Base_jll and
Qt6Declarative_jll 6.10.2+2, GLMakie 0.13.15, and Makie 0.24.15.
These versions are an observed compatible environment, not a claim that all
versions admitted by the prototype's compatibility ranges work. Production
HammerheadGUI dependencies were not updated. The generated manifest and
logs/images are ignored local artifacts.

The lifecycle harness launches hidden child processes with
`QT_QPA_PLATFORM=offscreen` in their startup environment. It normalizes Windows
environment names, removes the legacy `QMLSCENE_DEVICE` selector, and explicitly
selects `QT_QUICK_BACKEND=rhi` with `QSG_RHI_BACKEND=opengl` for native trials.
The older scripts set Julia's `ENV` before imports; those recorded values do
not prove which environment Qt's separate Windows C runtime observed. Their
captures therefore do not establish a verified offscreen native lifecycle.
Invisible construction does not establish rendering: Qt documents that
[hidden windows stop scene-graph rendering](https://doc.qt.io/qt-6/qquickwindow.html).
Captures use a QML-defined item's
[grabToImage](https://doc.qt.io/qt-6/qml-qtquick-item.html#grabToImage-method).

An earlier native bridge run produced an inspected plot capture, but
exiting that render probe reported
`ModernGL.ContextNotAvailable("glDeleteBuffers, ... no valid OpenGL context available")`.
This is a concrete lifecycle gate for adoption. A passing construction or
capture must not be described as a clean native application lifecycle.
The shell also provides a separate Qt software/image-preview path to isolate
ordinary controls and controller behavior from bridge resource management.
That fallback is a static rendered preview refreshed on controller actions;
it does not establish GPU bridge performance or equivalent native input.

## Lifecycle harness and application ownership

`lifecycle_runner.jl` owns each offscreen child, a finite timeout and termination,
and complete stdout/stderr files. It preserves a fresh evidence directory after
parent exit. Reports include OS exit status, timeout/error reasons, package and
prototype source hashes, a manifest digest, relevant Qt environment variants,
numbered flushed stage files and captures. Parent source hashes before and after
execution reject prototype edits during a trial. These files aid crash diagnosis;
flushing them does not establish power-loss durability. Capture success or a
pre-shutdown completion marker is insufficient: required stages, attached native
screens, capture files, resource counts, logs and final exit must agree.

The minimal native probe imports QML, QMLMakie, GLMakie and Observables only.
Its environment report also identifies declared Hammerhead/GUI dependencies,
which it does not execute; their source hashes are environment provenance rather
than a claim about imported application code. Startup times under concurrent
validation activity are diagnostic elapsed times, not performance measurements.

The application ownership repair gives each viewport generation a fresh figure,
removes its mouse subscription and refresh callback before detachment, and
serializes integrated/separate transitions. It clears Makie's current-figure
reference when that reference belongs to the departing viewport. The existing
`run.jl` / `main.qml` shell uses this path rather than reusing one scene across
screens. Release acknowledgement there certifies application callback disposal.
It does **not** certify native Qt GL cleanup. The software preview owns ordinary
hidden GLFW screens and can destroy those through their implemented context switch.

Pre-worker ownership checks passed 60 application-state assertions, including a
retained released lease, and 30 scientific viewport assertions across three
explicitly invisible GLFW render/pick/dispose cycles. Each cycle returns the
screen registry to its baseline and old figure weak references clear after GC.
The hidden-process harness passes 24 checks for environment normalization,
invalid PNG refusal, complete logs, nonzero exits and owned timeout termination.
It also rejects retained shell subscriptions or running replay at disposal.
These checks establish application ownership bounds; they do not test Qt GL
resource release or native pointer/keyboard behavior.

The pre-worker enforced software child exited zero after five fresh viewport generations
and five application releases including shutdown, with cancellation after one
of three pairs. Its inspected capture has readable Qt controls and a correct
scientific preview. Qt's offscreen font database initially supplied missing
glyphs; an explicit FontLoader now uses an existing Makie font asset and records
its path, digest and loaded family. No font binary is installed or committed.
There are no recorded render, model-binding or teardown errors in this software
run. This remains a static preview workflow, without native Qt GPU or gesture
evidence, and its elapsed time includes compilation and concurrent validation.

The opt-in native trial records the Windows `wglGetCurrentContext` handle in the
actual render callback and attempts screen destruction in that callback only
when the same nonzero context is current. QML item destruction waits for the
release acknowledgement. Full render exceptions are recorded and fail the child;
they never escape a Cvoid callback. This is a diagnostic experiment, not a
production bridge patch. QMLMakie's empty destruction hook, its context-valid
flag and no-op context switch remain upstream lifecycle concerns; no depot
package is modified.

With the explicit startup environment, the current Windows Qt offscreen plugin
reports `This plugin does not support createPlatformOpenGLContext!`, fails to
create the OpenGL RHI context, and cannot produce a native frame. Earlier trials
that selected the software adaptation produced inspected black captures and
zero GL screens; the harness rejects those as native evidence. The native
single-render/release/exit gate remains failed. Reopen, resize/FBO replacement,
separate-window transitions and injected processing-failure scenarios are defined
but remain unvalidated and are skipped until that minimal gate passes. Nothing
here establishes safe bridge teardown, desktop input, a threaded render loop,
cross-platform compatibility or GPU-memory bounds.

The historical Windows software-shell smoke, before the ownership repair and
enforced child environment, exited with status zero,
created five viewports across the integrated/separate layouts, corrected an
invalid schedule, displayed an open failure, and cancelled after one of three
pairs. Its inspected `software_shell.png` has readable controls and a downward
image y-axis. The adapter suite passed 30 assertions. The native shell's earlier
pre-shutdown smoke also reached five viewport creations and captured dense
vectors, but printed `exception in render` during reopen and faulted during
context destruction on exception unwind. Its `lifecycle_complete` report flag
describes the smoke sequence before teardown; it is not a clean-lifecycle pass.

| Historical run on Windows Julia 1.11.4, before the ownership repair | Evidence |
|:--|:--|
| First hidden bridge import, including initial candidate precompilation | 235.66 s; QML load 0.94 s; construction only |
| Warm software-shell process import | 8.99 s; not a production GUI comparison |
| Software-shell QML load | 0.40 s; excludes controller/figure construction and first rendered frame |
| Whole software-shell smoke | 30.66 s, including JIT, rendering, batch and lifecycle exercises |
| Estimated retained Julia controller state / figure | 231,249 / 22,085,754 bytes at report time; excludes native allocations |
| Peak process memory reported by Sys.maxrss | About 1.76 GiB; includes JIT/workload, not steady-state memory |

These are single-machine observations, not benchmark distributions or startup
targets. Generated TOML reports record the resolved versions and workload
flags. Hidden programmatic pan/zoom, vector picking, mask overlays, reversed
image coordinates, and file-coordinate limits have separate viewport tests;
they do not certify native mouse/keyboard input.

Generic completed-file browsing in this prototype accepts unscaled planar PIV
entries. Unsupported scaled or mixed non-planar entries report an error
before changing the selected frame. File views use result coordinates and
do not show the unrelated demo image. The index is a snapshot of a completed
file; concurrent writes and resumable processing are outside this contract.

The saved-experiment lane preserves full recipes separately from demo forms,
including ROI, embedded masks/backgrounds, ordered preprocessing and every pass
setting. It uses explicit result/run-record destinations and environment policy,
with written-pair progress and boundary cancellation from core replay in an
owned subprocess. Referenced scripts can be inspected but are never automatically
loaded or executed. Recipe/input identity and the retained displayed run/output
identity appear separately. Failed open/replay/inspection cannot relabel old
vectors as a new run; a failed view restoration hides the plot and marks it
unavailable. Completed inspection verifies the recorded output and retains one
lazy display payload. Physical planar positions/values use the explorer's single
conversion, unit-aware axis labels and spacing-derived limits; arrow lengths are
normalized for display. Demo images and mask overlays are absent on experiment
data. Fallback picks map through image letterboxing and the plotted axis bounds.

Replay belongs to the shell, while a viewport lease owns its figure and picking
subscription. Closing/reopening a view does not cancel processing. Shell shutdown
requests cancellation and waits through loading/output/history cleanup before
disposing subscriptions and views. Owner-side request capture, startup, result
I/O and rendering can pause the event loop; hidden tests do not establish desktop
responsiveness.
The saved-experiment child also rejects changes to core/GUI source maps during
its trial. These checks do not certify Qt native GL cleanup or cross-platform
availability.

The pre-worker focused prototype checks passed 250 assertions, including 64 saved-recipe
adapter checks, 26 rendering transactions, 13 physical-geometry checks and
11 queued-action checks alongside the existing adapter/view/lifetime tests.
Both software children exit zero after five viewport generations and inspected
readable captures. The saved child records one-pair cancellation, a completed
three-pair rerun, view closure while replaying, lazy navigation/picking and
retained display on failure. After opening saved history, the controller is
ready with zero new-run progress; the historical run is separately reported as
completed with three pairs. Disposal requires zero remaining shell subscriptions
and no running replay. Tiny-unit and singleton fixtures check that arrow lengths
and limits derive from local coordinate spacing/extent, without an arbitrary
distance in world units.

Exploratory saved-replay children faulted during Qt JS-stack collection while
GLFW polled events. Their logs remain failure evidence. The final software path
copies callback arguments before queueing heavy work outside Qt JS callbacks,
then renders an explicitly owned hidden GLFW screen without a background render
loop. Final children verify unchanged source maps and clean OS exit. This
software event-loop boundary is not evidence of native bridge cleanup or desktop
responsiveness.

## Separate interactive scientific window

The explicit `--plot=glfw` mode couples Qt software-rendered controls to a
dedicated GLMakie window. It uses the same controllers and saved planar recipe
workflow. Qt event processing returns before queued application work and GLFW
events/rendering run on the owning Julia thread. No background GLMakie renderer
is started. Each view owns its screen and figure; closing the scientific window
keeps replay running, and Ctrl+W reopens the view. Native keyboard/mouse delivery
still needs a desktop exercise. This environment still imports QMLMakie and its
plugin; it does not prove that dependency can be removed.

The dedicated owner avoids the GLMakie scene constructor's singleton screen,
preserving unrelated screens in focused tests. Destruction uses GLFW's own
context switch and retains ownership if release fails. The shell preserves the
original exception when cleanup also fails and records the cleanup failure.
Observed window closure is acknowledged before another transition. Render-error
logs fail the lifecycle gate even when the underlying update did not throw.

The pre-worker separate-window focused suite passed 87 checks for cleanup failure handling, screen
ownership/events, saved replay/selection and rendered glyph bounds. Bounds allow
for arrow tips and stroke width, including singleton grids and unequal axis
spacing; changing frame values preserves a manually adjusted view. The lifecycle
harness passes 38 checks, including refusal of inconsistent window counts and
nonthrowing render errors.

Four pre-worker source-stable hidden Windows children exited zero: demo and saved-experiment
lanes in both separate-GLFW and static-preview modes. In the separate-window
lanes, demo creates/releases four screens and saved replay creates/releases five.
Both return the screen registry to baseline, clear old figure weak references,
dispose all shell subscriptions and finish with no active replay or background
renderer. Independent visual review confirms readable controls, selected-vector
agreement, physical mm/mm/s labels and complete arrowheads in the saved capture.
Saved plots have no unrelated demo image or mask overlay.

Before saved-planar subprocess integration, the observed maximum event-pump
gaps were about 2.95 seconds for demo and 1.43
seconds for saved replay under compilation and concurrent validation load. These
are diagnostic observations, not responsiveness acceptance. This mode establishes
bounded application and GLFW ownership in hidden trials; it leaves desktop
input, HiDPI, accessibility, embedded Qt GL lifetime and other platforms open.

## Saved-planar subprocess ownership

The implemented isolated lane moves existing saved planar replay into a core-only
subprocess. The Qt owner retains event handling, GLFW rendering and result
inspection. A captured request retains the complete recipe, ordered inputs,
destinations and environment policy; it does not substitute a simplified form
or use the new saved-ensemble workflow. Progress is acknowledged at each native
pair-write boundary. Owner polling is bounded and never waits for a numerical
pair to finish. These boundaries do not promise an upper bound on GUI action or
rendering latency.

This worker ownership implementation is Windows-only. Its kill-on-close Job
Object must enroll the owned child before processing is permitted. The ownership
mechanism follows Microsoft's [Job Object contract](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects).
Other hosts
refuse the unsupported ownership capability rather than silently falling back
to cooperative replay on the Qt thread. The manual three-platform workflow
must retain this refusal as a failed capability gate. Equivalent tested Linux
and macOS enrollment, parent-loss cleanup and descendant reaping remain open;
the production GUI is unchanged.

Terminal metadata and final OS exit are separate evidence. A process exit or a
written-pair count cannot establish a completed experiment run. Ordinary planar
cancellation retains the existing failed-prefix run semantics; cancellation
requested at the final write completes, whereas an explicit callback abort can
fail even after the final write. A history-save exception can leave all native
results intact without returning any completed run. The worker must report that
failure without inventing completion from the surviving file.

Concurrency checks use an explicitly injected worker barrier to keep the child
active while the owner acknowledges controls, rendering, selection, pan/zoom
and close/reopen. Real PIV event-pump gaps are reported separately as observations
of that workload, not universal responsiveness. Parent-loss tests at an
acknowledgement boundary do not establish interruption during numerical work.
An outer Qt-owner timeout does not itself verify replay-descendant cleanup;
incomplete evidence prevents further child launches.

The current Windows checks comprise 63 protocol, 48 client/ownership, 90 real
replay parity/cancellation/failure and 26 evidence-validator assertions. The
isolated startup-enrollment failure helper adds seven checks separately. GUI
focused checks total 370 across separate invocations: 97 final cached-error and
adapter checks, 39 display transactions, 147 remaining contracts and 87 owned
GLFW checks. Only the 97 cached-error/adapter checks were rerun after the final
error-cache repair; these counts are not one final-source test invocation.

The corrected hidden worker child in the ignored
`artifacts/worker-lifecycle-KemBhR` evidence directory exited zero in 97.33 seconds
with unchanged prototype, core and GUI source maps. Twenty stages bind the owner
and worker identities; nine demonstrate control/render/pick/pan/zoom and
close/reopen acknowledgements during explicitly injected waiting work. All
three real native pair writes subsequently completed. The prior physical display
remained identified while active, and the original failed-open error was checked
in cached page content before navigation and reconstructed across all pages.
Inspected active, small and large Qt captures show persistent progress/cancel
controls and reachable scroll content; the separate scientific capture shows
complete arrowheads and physical mm/mm/s values. Five capture digests are bound
to the report. Four scientific generations/releases restore the screen registry,
clear old figure weak references and leave no shell subscriptions or worker.

Owner servicing after request startup through joined terminal lasted 29.261
seconds with a maximum observed pump gap of 1.051 seconds; it includes the
injected barrier, fresh-worker import/JIT and rendering, and excludes earlier
fixture replay and request capture/spawn. The first native-write acknowledgement
through joined terminal lasted 2.397 seconds with a maximum gap of 0.036 seconds;
this excludes imports and the first pair but may include later JIT and I/O.
These single-workload observations are not responsiveness thresholds or evidence
of a generally warmed worker. The earlier `worker-lifecycle-8mtkLy` trial is
superseded for final acceptance because it preceded the error-cache repair.

The final saved-experiment static-preview and separate-GLFW children in
`artifacts/lifecycle-qhqTp3` exited zero in 95.62 and 94.98 seconds, with no
timeout or recorded render/model/teardown errors. Both exercise cancellation,
a three-pair rerun, completed physical inspection and retained display through
a failed open, finishing five viewport generations. The GLFW case releases all
five screens, restores its registry and clears old figure references. Their
46 prototype source hashes match the active-worker child exactly; recorded
core/GUI source maps also remain unchanged. Demo lifecycle captures were not
rerun after the final error-cache fix and retain their earlier source scope.

## Manual platform evidence

The opt-in [workflow guide](https://github.com/stillyslalom/Hammerhead.jl/blob/main/HammerheadGUI/prototypes/qml/ci_validation.md)
describes a manually triggered Julia 1.11 matrix on Ubuntu, Windows and macOS.
It resolves only the prototype environment, runs focused and hidden lifecycle
checks, retains the native Qt prerequisite as a failing gate when unsupported,
and uploads logs, captures and the resolved environment. No hosted run has been
dispatched or observed. Local validation passes 18 process-owner checks and 20
static workflow checks; this does not establish platform compatibility.

The workflow owner records child exit, timeout and error evidence. Any
unsuccessful child command prevents every subsequent child launch, since cleanup of possible
descendants is unverified. It never treats an incomplete native run as a software
pass. Hard runner loss can prevent artifact upload even with an always-run step.

## Reproduction and decision

The Qt prototype now provides Browse beside the saved-record, native-result,
replay-output and run-history drafts. Acceptance stages the native path; it does
not open a recording, change the controller destinations or write a file. Save
selection therefore disables overwrite confirmation. Replay retains its own
alias and publication guards. Qt converts local URLs, including Unicode and
literal reserved filename characters, before the owner queues other work.

Each opening creates a separate chooser with a captured token and purpose.
Rejected, stale or closing chooser callbacks cannot clear a newer request.
Modal root shortcuts are disabled while choosing a path. The automated lane
forces offscreen software Qt, `DontUseNativeDialog` and `Popup.Item`; these checks
do not validate native desktop dialogs, accessibility or every host path format.

The source-stable Windows child in ignored `artifacts/file-dialog-Cw5sYt`
exited zero in 40.09 seconds. Its 30 stages preserve the existing recipe,
controller destinations, lazy explorer frame 2 and vector selection through
actual OpenFile/SaveFile choices, rejection, Escape, separate modal keys,
old-instance callbacks after same-picker reopening, an injected opening failure,
busy refusal and shutdown. Selecting an existing history destination preserved
its bytes; selecting fresh output did not create it. Processing was never
started. Actual fallback list clicks and filename-control edits are automation
for the installed Qt version, with native OS selection still an open gate.

Independent source/capture checks matched 54 prototype, 47 core and 31 GUI
hashes and all three PNG digests. The modal chooser and compact labelled rows
were visually inspected at 900 × 600 and 1100 × 800 window sizes; captures omit
the 40 px menu bar. Explanatory text scrolls while the cancel/progress/status
footer remains visible. Focused path/state checks passed 76 assertions, the
dialog evidence validator 15, and the existing process-owner self-tests 38.
The existing saved-planar software regression in ignored
`artifacts/lifecycle-dNcVIZ` then exited zero in 99.80 seconds without timeout.
Its prototype/core/GUI source maps match the dialog child exactly, preserving
the previously tested replay, cancellation and subscription-cleanup workflow.

From the repository root:

```powershell
julia HammerheadGUI/prototypes/qml/setup.jl
julia --startup-file=no --threads=1 --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/file_path_tests.jl
julia --startup-file=no --threads=1 --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/file_dialog_tests.jl
julia --startup-file=no --threads=1 --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/file_dialog_runner.jl --timeout=240
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/shell_actions_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/adapter_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/experiment_adapter_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/experiment_error_pages_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/experiment_transaction_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/viewport_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/test_lifecycle_contract.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/viewport_ownership_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/owned_glfw_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/lifecycle_runner.jl --self-test
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/lifecycle_runner.jl --cases=construction,baseline,single-context --timeout=90
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/lifecycle_runner.jl --cases=shell-software --timeout=180
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/lifecycle_runner.jl --cases=shell-experiment-software --timeout=240
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/lifecycle_runner.jl --cases=shell-glfw,shell-experiment-glfw --timeout=240
```

The lifecycle runner writes its summary before returning nonzero for failed or
gated scenarios. The probe README identifies expected failures and generated evidence. Native
startup, resource-lifetime, gesture, accessibility, and platform gates must
all be satisfied before choosing Qt as the production shell. GTK4 and a
Bonito/WGLMakie browser shell remain comparison candidates; neither has been
implemented or measured here. Keep the current production GLMakie shell
while collecting this evidence. Any migration should replace shell/view
code incrementally and retain the tested controller boundary, rather than
coupling analysis logic to the candidate toolkit.
