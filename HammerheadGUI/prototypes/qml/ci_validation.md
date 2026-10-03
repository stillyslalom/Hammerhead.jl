# Manual cross-platform prototype evidence

The [QML prototype workflow](../../../.github/workflows/qml-prototype.yml) is
manually triggered with `workflow_dispatch` only. It does not run on pushes or
pull requests, publish packages, change production dependencies, or adopt Qt as
the production framework. Adding the workflow is not evidence that any platform
has passed it; inspect the actual run and downloaded artifacts.

The fixed matrix is Ubuntu 24.04, Windows 2025 and macOS 15, using Julia 1.11
and one Julia thread. It uses the runner's native architecture; GitHub currently
documents `macos-15` as ARM64. This is a three-platform experiment at one Julia
minor version, not validation of the prototype's full supported Julia range.
The image version, architecture, commit and run attempt are saved in `runner.txt`.
[GitHub's runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
and [setup-julia's architecture documentation](https://github.com/julia-actions/setup-julia)
describe the runner and installer choices.

`setup.jl` develops the checked-out core and GUI into the isolated prototype
environment and restores its portable Project file. Explicit prototype
precompilation runs during bounded setup so cold imports are not charged to
individual focused-test deadlines. The resolved Manifest is
uploaded, making the actual bridge versions inspectable; different runs can
resolve different compatible versions. No production environment is resolved
or edited. The workflow uses read-only repository permission and does not retain
checkout credentials or use a dependency cache.

Ubuntu installs Xvfb, Mesa and the X11/GL libraries used by the existing GUI
workflow. A persistent Xvfb display is checked with `glxinfo -B` before setup.
`LIBGL_ALWAYS_SOFTWARE=1` selects Mesa software GL in this lane. This supplies
an OpenGL context for invisible GLFW windows; it is not hardware GPU evidence.
The library requirements follow [GLMakie's documentation](https://docs.makie.org/stable/explanations/backends/glmakie).
Windows and macOS use their hosted runner's graphics environment. Missing or
unsupported graphics capability fails the requested lane rather than skipping it.

Each job requests these independent checks:

| Lane | Requested checks |
|---|---|
| Focused contracts | Queued actions, demo and experiment adapters, display transactions, viewport geometry, ownership contracts, invisible-screen ownership and owned GLFW |
| Harness contracts | Hidden-process exit/timeout ownership and rejection of incomplete evidence |
| Software Qt | Demo and saved-experiment lifecycle children |
| Owned GLFW | Demo and saved-experiment lifecycle children with invisible directly rendered scientific screens |
| Native Qt prerequisite | Construction, native baseline frame, single render/release/exit |

The native lane deliberately retains its current possible Windows offscreen
platform failure. There is no expected-failure allowance, `continue-on-error`,
or software fallback that turns a native failure green. Longer native reopen,
separate, resize and repeated-cycle scenarios are not requested by this bounded
workflow; those remain separate acceptance work after a clean prerequisite.
The runner itself owns Qt startup environment selection and does not infer
native success from an application-release acknowledgement or capture alone.

After successful setup, a failed check does not prevent independent later lanes
from running. Every failed step still makes its job fail. The focused loop also
collects subsequent focused outcomes after one assertion failure. Cancellation
stops new work; setup failure prevents meaningless application checks.

The inline workflow process owner writes complete stdout and stderr plus a TOML
invocation, child PID, OS exit status, timeout and owner-error report for every
requested script. It hides Windows console windows and terminates only its
owned child on timeout. An owner timeout/error publishes an incomplete-owner
marker: every later requested child launch is refused with failure evidence,
including later focused files and lanes. This prevents overlapping new work
with a potentially surviving descendant without killing unrelated processes.
The lifecycle harness additionally preserves its own
child provenance, staged acknowledgements, logs, captures and summary. A timeout
of the outer harness owner does not certify cleanup of that harness's descendants;
inspect its partial evidence. Step/job deadlines and runner termination remain
outer limits, not successful disposal evidence.

The upload step uses `always()` and a unique OS/run/attempt name. It includes the
entire prototype `artifacts/` tree and Project/Manifest, with 14-day retention.
It does not upload the checkout, Julia depot or inherited environment. Capture
PNGs should be visually inspected alongside their signatures and recorded stage
counts. [upload-artifact's documentation](https://github.com/actions/upload-artifact)
describes immutable artifact names and retention. Hard runner loss or job timeout
can prevent even an `always()` upload; GitHub's job log is then the remaining
evidence. No successful upload, shell completion or isolated green lane alone
establishes native Qt lifecycle, accessibility, desktop input, packaging,
performance or hardware support.
