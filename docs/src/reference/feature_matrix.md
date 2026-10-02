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
Stereo needs distinct per-camera diagnostics, and ensemble iteration semantics
require a separate model; neither currently accepts this planar diagnostics API.
