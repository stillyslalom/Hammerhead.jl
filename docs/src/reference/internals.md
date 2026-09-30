```@meta
CurrentModule = Hammerhead
```

# Internals

This page lists non-exported functions and types. Their signatures may
change between releases. Use the public driver keywords such as
`backend = :cpu` for normal analysis; see [Run PIV on a GPU](../howto/gpu.md)
for backend selection and the [feature matrix](feature_matrix.md) for
supported options. The [backend reference](backends.md) describes the
private execution interface.

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
