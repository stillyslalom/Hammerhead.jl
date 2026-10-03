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
| Pooled ensemble execution diagnostics | yes | yes | no | no |
| Stereo 2D3C, sequence, ensemble (per-camera PIV) | yes | yes | yes | yes |
| Cross and filtered phase correlation | yes | yes | yes | yes |
| `:gauss3` and `:gauss9` peak fits | yes | yes | yes | yes |
| Correlation-statistics uncertainty | yes | yes | yes* | yes* |
| Enlarged search areas | yes | no | no | no |
| `:gauss2d` peak fit | yes | no | no | no |
| Retained correlation planes | yes | no | no | no |
| File-based planar experiment replay | yes | yes | no | no |
| Frozen-camera stereo experiment replay | yes | yes | no | no |
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

[Planar experiment records](../howto/experiments.md) and
[checkpoints](../howto/checkpoints.md) retain their file-based planar scope.
Separate [stereo experiment APIs](../howto/stereo_experiments.md) replay CPU/KA stereo
sequences from frozen fitted cameras and a common dewarp grid. They do not rerun
calibration fitting or self-calibration or extend checkpoint recovery. A separate
[saved-stereo GUI workflow](../howto/gui_stereo_experiments.md) snapshots the
supported batch form and reopens richer recipes intact with read-only settings.
PTV/tracking and vendor-GPU recipes remain separate
from the supported processing pipelines.
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
fields; it does not verify calibration or acquisition sources.
[Ensemble diagnostics](../howto/ensemble_execution_diagnostics.md) separately
record one pooled sweep per pass, ignored iteration settings, numerical
contributions and pre-addition residuals. Capture supports CPU/KA; vendor devices
are explicitly refused. Separate [ensemble quality reports](../howto/ensemble_quality_reports.md)
use explicit format-4 opt-in and verify raw binding; lazy GUI inspection also
checks ensemble and stereo raw binding before physical conversion. Opt-in quality
report version 3 retains planar/per-camera execution coverage and support counts;
it does not pool residual amplitudes or reverify files when a saved report loads.

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
metadata for planar sequences through an optional native companion.
[Stereo pair timing](../howto/stereo_pair_timing.md) provides a separate companion
with both cameras' delays/midpoints, synchronization policy and reconstructed
scaling provenance. Frozen-camera stereo replay preserves and verifies its timing
companion. Timing-aware exports, checkpoints and GUI timing inspection require
separate integration.

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

[Calibrated PIV/PLIF resampling](../howto/calibrated_resampling.md) is CPU
postprocessing of raw planar fields and scalar images on an explicit shared
grid. Affine vector conversion, masks and contributor availability accompany
the sampled values. This does not propagate uncertainty, establish synchronized
acquisition or equalize the measurement support of the two modalities.

Timed artifacts and calibrated companions preserve foreign Windows/POSIX source
locators as provenance. Local relocation and consumed-file protection are explicit;
this does not provide automatic source-file discovery or source-byte verification.

[Temporal spectra](../howto/spectrum_timing.md) accept an explicit interval or
sample times checked for uniformity within declared tolerances. They do not
resample irregular data or infer sampling intervals from image-pair delays.

[Derived flow analysis](derived.md) can describe the actual planar derivative
contributors and require two-sided stencils. Geometry, input eligibility and
finite output are distinct; these diagnostics do not establish spatial resolution
or propagate measurement uncertainty. The planar GUI explorer exposes discrete
support maps and selected contributors, using one stencil policy across scalar
analysis and area circulation. Component profiles remain independent.

[Ordinary GUI replay](../howto/gui_experiment_replay.md) reports completed pairs
and supports cooperative cancellation after pair writes and loader cleanup.
Its native prefix has no checkpoint resume guarantee.

[GUI recipe comparison](../howto/gui_comparison.md) uses the shared selected-pair
comparison and report contracts. It runs saved CPU/KA planar recipes; richer
GUI controls do not expand the core recipe scope or supply accuracy ground truth.

The [GUI recipe revision editor](../howto/gui_recipe_revision.md) edits ordered
planar passes while retaining imported preprocessing, masks, ROI, scale and
execution options. A metadata difference describes changed settings; it does
not rerun images or measure their effect. Saving a new record verifies unchanged
inputs and captures the current environment without copying prior run history.

The [preprocessing revision view](../howto/gui_preprocessing_revision.md) retains
ordered duplicate steps, every built-in option and embedded background precision.
Explicit verified-pair previews apply the saved core operations to full images
before ROI and show masks as overlays. Shared display ranges support comparison;
these images do not establish improved PIV accuracy or uncertainty coverage.
