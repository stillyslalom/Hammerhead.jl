# Feature matrix

| Workflow | CPU | KA CPU | CUDA | AMDGPU |
|---|:---:|:---:|:---:|:---:|
| Planar single-pair / multipass | yes | yes | yes | yes |
| Planar sequence | yes | yes | yes | yes |
| Ensemble correlation | yes | yes | yes | yes |
| Stereo 2D3C, sequence, ensemble | yes | yes | yes | yes |
| Correlation-statistics uncertainty | yes | yes | yes | yes |
| Enlarged search areas | yes | no | no | no |
| PTV and trajectory linking | yes | CPU | CPU | CPU |
| Retained correlation planes | yes | no | no | no |

`KA CPU` is the hardware-free proving backend selected by `backend = :ka`.
CUDA and AMDGPU accelerate correlation, deformation, peak analysis, and the
Wieneke uncertainty statistics; image loading, dewarping, 3C reconstruction,
PTV, validation, exports, and statistics remain host operations. GPU backends
support cross/phase correlation and gauss3/gauss9 subpixel fits; enlarged
search areas, gauss2d, and retained planes are rejected explicitly.

All workflows accept `Float32` and `Float64`. See [KernelAbstractions and GPU
backends](backends.md) for the precise device boundary and supported options.
