```@meta
CurrentModule = Hammerhead
```

# Hammerhead

Documentation for [Hammerhead](https://github.com/stillyslalom/Hammerhead.jl),
a Julia package for particle image velocimetry (PIV).

[`run_piv`](@ref) operates on in-memory image pairs (any equally sized
real-valued matrices):

```julia
using Hammerhead

# Multi-pass with symmetric image deformation: each pass uses the previous
# validated field as a predictor and shrinks the window.
passes = multipass_parameters([64, 32, 16, 16];
    padding = true,         # zero-padded (linear) correlation
    apodization = :gauss,   # Gaussian window on each interrogation window
)
result = run_piv(imgA, imgB, passes)

result.u, result.v    # displacement field (px), u along x/columns
result.x, result.y    # interrogation grid centers (px)
result.outliers       # validation mask
```

See the [Function Reference](function_ref.md) for the public API and
[Internals](internals.md) for implementation details.
