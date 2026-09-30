```@meta
CurrentModule = Hammerhead
```

# Preprocessing

Image conditioning before correlation: background removal, intensity
capping, high-pass filtering, and contrast equalization, plus affine
registration utilities. The mutating forms (`f!`) modify floating-point
buffers in place; the allocating forms accept any real-valued matrix and
return a processed copy. See the
[preprocessing how-to](../howto/preprocessing.md) for combining operations.

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
