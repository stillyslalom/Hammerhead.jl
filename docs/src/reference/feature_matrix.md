# Feature matrix

The columns describe the `backend` selector used by planar PIV. CUDA and
AMDGPU require their package extensions; `:ka` runs the shared device
kernels on a CPU. For stereo drivers, the selector applies to each camera's
two-component PIV analysis. Image dewarping and three-component
reconstruction remain CPU operations.

| Workflow | CPU | KA CPU | CUDA | AMDGPU |
|---|:---:|:---:|:---:|:---:|
| Planar single-pair / multipass | yes | yes | yes | yes |
| Planar sequence | yes | yes | yes | yes |
| Ensemble correlation | yes | yes | yes | yes |
| Stereo 2D3C, sequence, ensemble (per-camera PIV) | yes | yes | yes | yes |
| Cross and filtered phase correlation | yes | yes | yes | yes |
| `:gauss3` and `:gauss9` peak fits | yes | yes | yes | yes |
| Correlation-statistics uncertainty | yes | yes | yes* | yes* |
| Enlarged search areas | yes | no | no | no |
| `:gauss2d` peak fit | yes | no | no | no |
| Retained correlation planes | yes | no | no | no |
| File-based planar experiment replay | yes | yes | no | no |
| Built-in planar recipe checkpoint/resume | yes | yes | no | no |
| Representative-pair recipe comparison | yes | yes | no | no |

*Device uncertainty estimation uses Float64 accumulation and requires a GPU
with suitable Float64 support. A backend rejects an unsupported setting
instead of silently changing algorithms.

Particle tracking, trajectory linking, vector validation, image loading,
exports, and field statistics run on the CPU; their functions do not use
the PIV backend selector. Planar PIV accepts `Float32` and `Float64` images.
See [Run PIV on a GPU](../howto/gpu.md) for setup and performance tradeoffs,
or [KernelAbstractions and GPU backends](backends.md) for the execution
boundary.

[Experiment records](../howto/experiments.md) and
[checkpoints](../howto/checkpoints.md) currently cover planar file-based recipes;
the broader stereo/PTV/GPU pipelines above do not imply saved-recipe support.
[Quality reports](../howto/run_quality.md) summarize stored planar/stereo results
on the CPU, independently of the backend that produced them. The shared
original-stencil guard has CPU/KA regression coverage; its vendor GPU paths
still need fresh hardware validation after this change.

[Execution diagnostics](../howto/execution_diagnostics.md) observe planar passes
and can accompany planar sequence/replay results. CPU/KA tests establish the
software behavior; new vendor-device execution evidence remains separate.
[Stereo execution diagnostics](../howto/stereo_execution_diagnostics.md) use a
separate two-camera companion, explicit dewarped-pixel residual basis and common
world-grid geometry. Native inspection can verify reconstructed/camera measurement
fields; it does not verify calibration or acquisition sources. Ensemble iteration
semantics require a separate model and remain unsupported. GUI companion panels
and quality reports do not yet consume the stereo companion.

[Measurement history](../howto/measurement_history.md) observes the final
planar pass's final sweep and can accompany sequence/replay results. It records
validation, alternative-peak and filling events on the host result grid.
Stereo, ensemble, PTV and checkpoint companions remain outside this API;
vendor-device correctness still requires hardware evidence.

Opt-in quality-report version 2 aggregates verified planar history with explicit
missing/unsupported coverage. [Lazy GUI inspection](../howto/gui_companions.md)
verifies raw history before physical display; eager inputs and checkpoints lack
this supported association. Numerical UQ availability is not applicability.

[Pair timing](../howto/pair_timing.md) preserves supplied timestamp/source
metadata for planar sequences through an optional native companion. It does not
add timing-aware tracking, exports, replay, checkpoint or stereo persistence;
those workflows require separate timing semantics.

[Actual-time tracking](../howto/tracking_timing.md) is a separate explicit CPU
workflow. Its wrapper, dedicated native artifact and CSV schema preserve sample
times through elapsed-time linking and secant velocity evaluation. Default
tracking remains ordinal. [GUI timed-trajectory inspection](../howto/gui_tracking_timing.md)
accepts one timed wrapper or an explicitly selected dedicated artifact, retaining
its timing through physical display. Mixed native/timed sequences and lazy timed
indexes remain unsupported. Extracting the ordinary payload discards its timing.

[Calibrated particle and trajectory tables](../howto/calibrated_scattered_export.md)
are a separate CPU export with affine position/vector conversion and a verified
CSV/TOML pair. They accept unscaled PTV, ordinal tracking and timed tracking;
existing grid exports and ordinary `export_table` keep their contracts.

Timed artifacts and calibrated companions preserve foreign Windows/POSIX source
locators as provenance. Local relocation and consumed-file protection are explicit;
this does not provide automatic source-file discovery or source-byte verification.

[Temporal spectra](../howto/spectrum_timing.md) accept an explicit interval or
sample times checked for uniformity within declared tolerances. They do not
resample irregular data or infer sampling intervals from image-pair delays.

[Ordinary GUI replay](../howto/gui_experiment_replay.md) reports completed pairs
and supports cooperative cancellation after pair writes and loader cleanup.
Its native prefix has no checkpoint resume guarantee.

[GUI recipe comparison](../howto/gui_comparison.md) uses the shared selected-pair
comparison and report contracts. It runs saved CPU/KA planar recipes; richer
GUI controls do not expand the core recipe scope or supply accuracy ground truth.
