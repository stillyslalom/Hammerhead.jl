# Desktop GUI framework evaluation

HammerheadGUI currently uses GLMakie views over framework-free controllers.
The next desktop shell needs stronger forms, file navigation, focus handling,
and window management while preserving the existing analysis and result
contracts. The first implementation scope is raw planar PIV. Stereo, PTV,
tracking, and physical-unit displays remain supported by the production GUI;
the isolated candidate does not yet reproduce those views.

## Interaction requirements

| Requirement | Acceptance exercise | Candidate evidence |
|:--|:--|:--|
| Experiment and file browser | Open a completed recording; show path/read errors beside input; retain previous result on failure | Prototype has explicit path entry and lazy open; experiment tree remains pending |
| Editable parameter forms | Invalid schedules leave controller parameters unchanged; valid correction clears the inline error | Controller adapter tested; native QML TextField and error label wired |
| Large recordings | Browse indexed completed files while retaining one display payload and current view arrays | Existing ResultFile/ResultExplorer reused; O(number of entries) key index; no live-file tailing |
| Keyboard and focus | Tab through labeled inputs; arrow keys browse; Esc cancels; Ctrl+Return runs; Ctrl+W reopens viewport | QML bindings and accessible input names present; native input/accessibility audit pending |
| Layout and windows | Resize panels; compare settings plus separate visualization with an integrated layout using the same controllers | Software/offscreen smoke recreated five viewports across both layouts and exited cleanly; native bridge lifetime remains blocked |
| Image navigation | Pan/zoom a 1024-square image and retain precise data-coordinate picking | Makie axis interactions in bridge; image-transform fallback implemented; native gesture latency pending |
| Dense vectors | Render 16,384 vectors over an image, inspect one vector, avoid frame-history caches | Hidden standalone and real embedded framebuffer captures inspected; native bridge reports render exceptions during reopen |
| Masks and ROI | Draw/close a polygon on the demo image; edits reach the batch controller; retain production ROI semantics | Prototype mask adapter tested; native mask gesture/ROI editor parity pending |
| Live results and cancellation | Completed pairs update the explorer; cancellation preserves the completed prefix and leaves shell responsive | Existing cooperative BatchRunner reused; cancellation adapter regression tested; CPU/GPU responsiveness audit pending |
| HiDPI and accessibility | Test 100/150/200% scaling, focus indicators, screen-reader labels, menus and shortcuts | Pending real desktop checks |
| Resource lifetime | Close/reopen views repeatedly; exit without stale GL contexts or leaked payloads | Software shell completes and exits cleanly; native shutdown exposes GL context cleanup errors and has faulted during exception unwind |
| Distribution and compatibility | Record startup/memory and build/install on supported Julia versions and each target OS | Isolated Windows Julia 1.11.4 resolution tested; Linux/macOS, lower Julia versions and packaging pending |

## Isolated Qt6 candidate

The opt-in source lives in `HammerheadGUI/prototypes/qml/`, outside the
production dependency graph. It combines Qt Quick Controls with existing
`BatchRunner`, `ResultExplorer`, and `MaskEditor` controllers. QML functions
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

Current ownership checks pass 60 application-state assertions, including a
retained released lease, and 30 scientific viewport assertions across three
explicitly invisible GLFW render/pick/dispose cycles. Each cycle returns the
screen registry to its baseline and old figure weak references clear after GC.
The hidden-process harness passes 18 checks for environment normalization,
invalid PNG refusal, complete logs, nonzero exits and owned timeout termination.
These checks establish application ownership bounds; they do not test Qt GL
resource release or native pointer/keyboard behavior.

The final enforced software child exits zero after five fresh viewport generations
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

Completed-file browsing in this prototype accepts unscaled planar PIV
entries. Unsupported scaled or mixed non-planar entries report an error
before changing the selected frame. File views use result coordinates and
do not show the unrelated demo image. The index is a snapshot of a completed
file; concurrent writes and resumable processing are outside this contract.

## Reproduction and decision

From the repository root:

```powershell
julia HammerheadGUI/prototypes/qml/setup.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/adapter_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/viewport_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/test_lifecycle_contract.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/viewport_ownership_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/lifecycle_runner.jl --self-test
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/lifecycle_runner.jl --cases=construction,baseline,single-context --timeout=90
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/lifecycle_runner.jl --cases=shell-software --timeout=180
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
