```@meta
CurrentModule = Hammerhead
```

# Synthetic data

The `SyntheticData` submodule generates particle images with exact ground
truth: particle positions are displaced directly through a velocity
function (no interpolation warping), including out-of-plane motion through a
laser-sheet intensity profile. Used throughout the test suite and the
[tutorials](../tutorials/first_vector_field.md).

```@index
Pages = ["synthetic.md"]
```

```@autodocs
Modules = [Hammerhead.SyntheticData]
Order = [:module, :function, :type, :constant, :macro]
Private = false
```
