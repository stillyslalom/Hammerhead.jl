```@meta
CurrentModule = Hammerhead
```

# Synthetic data

The `SyntheticData` submodule renders particle images from a supplied
velocity function. It moves particles directly rather than warping an image,
so their launch positions and imposed displacements are known. The
laser-sheet profile can also vary particle brightness with depth. Use these
inputs to check an analysis against known motion, as in the
[first tutorial](../tutorials/first_vector_field.md).

```@index
Pages = ["synthetic.md"]
```

```@autodocs
Modules = [Hammerhead.SyntheticData]
Order = [:module, :function, :type, :constant, :macro]
Private = false
```
