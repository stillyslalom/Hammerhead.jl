```@meta
CurrentModule = Hammerhead
```

# Internals

Non-exported functions and types. These are implementation details and may
change between releases.

Execution backend plumbing is internal. Public driver/workspace keywords use
selectors such as `backend = :cpu`; concrete backend implementation types are
not exported and ordinary user code should rely on the default CPU behavior.
The core package provides `backend = :cpu` (FFTW) and the portable
`backend = :ka` (the same KernelAbstractions kernels the GPU backends run,
executed on the CPU — a hardware-free proving tier). GPU backends are
provided by package extensions that activate when the device package is
loaded: `using AMDGPU` enables `backend = :amdgpu` (ROCm) and `using CUDA`
enables `backend = :cuda` (NVIDIA). An extension registers its selector by
adding a `_resolve_backend(::Val{:amdgpu})`-style method, so unknown
selectors report which package to load rather than silently failing. Device backends currently
cover cross-correlation with `:gauss3`/`:gauss9` subpixel refinement;
phase correlation, `:gauss2d`, uncertainty quantification, and
correlation-plane storage stay on `:cpu` and error with a clear message.

```@index
Pages = ["internals.md"]
```

```@autodocs
Modules = [Hammerhead]
Order = [:function, :type, :constant, :macro]
Public = false
```

## Synthetic Data

```@autodocs
Modules = [Hammerhead.SyntheticData]
Order = [:function, :type, :constant, :macro]
Public = false
```
