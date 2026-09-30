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

*Device uncertainty estimation uses Float64 accumulation and requires a GPU
with suitable Float64 support. A backend rejects an unsupported setting
instead of silently changing algorithms.

Particle tracking, trajectory linking, vector validation, image loading,
exports, and field statistics run on the CPU; their functions do not use
the PIV backend selector. Planar PIV accepts `Float32` and `Float64` images.
See [Run PIV on a GPU](../howto/gpu.md) for setup and performance tradeoffs,
or [KernelAbstractions and GPU backends](backends.md) for the execution
boundary.
