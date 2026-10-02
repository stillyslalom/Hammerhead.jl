# Isolated Qt6/QML shell evaluation

This opt-in prototype evaluates desktop forms and window management around
HammerheadGUI's existing framework-free controllers. It does not replace the
production GUI or alter its dependencies. The requirements and adoption gates
are in [the framework evaluation](../../../docs/src/explanation/gui_framework.md).

The prototype supports synthetic raw planar PIV batches, a dense 16,384-vector
demo over a 1024-square image, completed-file lazy browsing, vector inspection,
mask polygon commands, inline schedule/open errors, cancellation, and integrated
or separate visualization windows using the same controller state. Lazy browsing
accepts **unscaled planar PIV entries only**. Other entries report an error before
changing the displayed frame; the production explorer supports all four types
and physical units. File views use result coordinates and hide the demo image.
Masks apply only to the synthetic demo's 96-square images. There is no experiment
tree, ROI form, preprocessing form, resumability, or concurrent-writer support.

## Run from the repository root

```powershell
julia HammerheadGUI/prototypes/qml/setup.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/adapter_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/viewport_tests.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/bridge_probe.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/render_probe.jl
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/run.jl --software
julia --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/run.jl
```

All commands above use hidden GLMakie rendering or Qt's `offscreen` platform.
`run.jl` defaults to a finite automated smoke, not a desktop window. An explicit
`--desktop` argument opts into an interactive native desktop window; that mode
has not been exercised in this evaluation. Do not use it in automated validation.

`setup.jl` resolves only this candidate environment, develops the local core and
GUI packages, and restores the portable project source paths after Pkg's path
canonicalization. Its generated Manifest is ignored. `artifacts/` contains
ignored TOML reports, logs, and PNG captures. No downloaded binary or generated
artifact belongs in a commit. Windows PowerShell can treat redirected native
stderr as a shell error; use `exit $LASTEXITCODE` after a redirected command when
recording the native process status.

## Evidence and limitations

Resolved/tested on Windows with Julia 1.11.4: QML 0.13.2, QMLMakie 0.3.3,
CxxWrap 0.17.5, jlqml_jll 0.10.4+0, Qt6Base_jll and Qt6Declarative_jll
6.10.2+2, GLMakie 0.13.15, Makie 0.24.15, and local HammerheadGUI 0.1.1.
These are observed versions, not a tested compatibility matrix. Linux/macOS,
older Julia versions, packaging, HiDPI, screen readers and native input latency
remain untested.

Final adapter checks passed 30/30. The corrected software/offscreen shell smoke
exited zero after five viewport creations, error callbacks, and cancellation
after one of three pairs; its capture has readable controls and image-down y.
One warm run measured import 8.99 s, QML load 0.40 s, whole smoke 30.66 s,
estimated state 231,249 bytes, figure 22,085,754 bytes, and about 1.76 GiB
peak process memory including JIT/workload. These are single-machine
observations, not steady-state memory or a performance comparison. The first
hidden bridge import took 235.66 s including candidate precompilation.

`adapter_tests.jl` checks form correction without parameter mutation on errors,
lazy frame navigation/picking, rejected scaled entries and preserved frame,
failed open preserving the old explorer, mask application, and cooperative
cancellation retaining the first completed pair. `viewport_tests.jl` checks
hidden dense rendering, programmatic zoom/pan, picking without resetting the
view, mask overlay changes, image-coordinate reversal, and file geometry.
These programmatic checks do not measure native pointer/keyboard behavior.

`bridge_probe.jl` proves invisible component construction and an event-loop
timer. `render_probe.jl` proves a **real embedded native framebuffer** through
`bridge_framebuffer.png`. It also reproduces a shutdown warning/error involving
`ModernGL.ContextNotAvailable("glDeleteBuffers, ... no valid OpenGL context available")`.
The QMLMakie bridge catches some render errors internally and prints
`exception in render`, so successful capture/report flags alone are insufficient
to call a run successful. Inspect stderr and exit status.

`run.jl` exercises invalid/valid form callbacks, explicit open errors, repeated
Loader destruction/recreation, separate/integrated windows, a cooperative batch,
cancellation, and a shell capture. `lifecycle_complete` in a TOML report means
the pre-shutdown smoke sequence finished; **it does not certify clean teardown**.
The native bridge has reported render exceptions during reopen and unsafe
context destruction on shutdown. Those are adoption blockers, not suppressed
exceptions. The separate `--software` path uses Qt software controls plus a
static scientific image rendered by hidden GLMakie and refreshed after controller
actions. Wheel/drag handlers transform that preview, and picking maps through
the saved axis rectangle. It does not establish native bridge GPU performance.

Reports include import/load/whole-smoke timing, estimated Julia state/view sizes,
and `Sys.maxrss()` peak process memory. Peak RSS includes JIT compilation and
the demo workload; state/view sizes exclude native allocations and are not
steady-state process memory. First-run precompile timing is separate from warm
startup. No responsiveness claim follows from an event timer merely running
between frame pairs; CPU/GPU load and cancellation latency need a desktop audit.

Keep the production GLMakie shell while the native lifetime, input, accessibility,
distribution, and cross-platform gates remain open.
