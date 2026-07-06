```@meta
CurrentModule = Hammerhead
```

# Preprocessing

Image conditioning before correlation: background removal, intensity
capping, high-pass filtering, and contrast equalization, plus affine
registration utilities. The mutating forms (`f!`) are the implementations
and operate in place on floating-point buffers; the allocating names accept
any real-valued matrix. See the
[preprocessing how-to](../howto/preprocessing.md) for chaining recipes.

```@index
Pages = ["preprocessing.md"]
```

## Image conditioning

```@autodocs
Modules = [Hammerhead]
Pages = ["preprocessing.jl"]
Private = false
```

## Registration and warping

```@autodocs
Modules = [Hammerhead]
Pages = ["transforms.jl"]
Private = false
```
