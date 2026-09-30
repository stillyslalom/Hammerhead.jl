```@meta
CurrentModule = Hammerhead
```

# Preprocessing

Choose an operation for a specific image problem: subtract a stable
background, reduce an illumination gradient, cap unusually bright pixels,
or raise local contrast. Compare correlation results before and after
processing; an operation can also remove particle signal. Mutating forms
(`f!`) modify floating-point arrays in place, while allocating forms
return a processed array. Registration and warping functions are listed
below. See [Build a preprocessing chain](../howto/preprocessing.md) for
examples.

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
