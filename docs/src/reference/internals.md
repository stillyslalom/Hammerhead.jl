```@meta
CurrentModule = Hammerhead
```

# Internals

Non-exported functions and types. These are implementation details and may
change between releases.

Execution backend plumbing is internal. Public driver/workspace keywords use
selectors such as `backend = :cpu`; concrete backend implementation types are
not exported and ordinary user code should rely on the default CPU behavior.

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
