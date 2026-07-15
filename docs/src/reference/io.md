```@meta
CurrentModule = Hammerhead
```

# Input/output (I/O) and batch processing

Image and mask loading (FileIO/ImageIO), result serialization in the JLD2
Julia data format, and the
batch sequence driver. See the [batch-processing how-to](../howto/batch.md)
for workflow recipes and [Run PIV on a GPU](../howto/gpu.md) for persistent
device-workspace behavior.

```@index
Pages = ["io.md"]
```

```@autodocs
Modules = [Hammerhead]
Pages = ["io.jl", "interoperability.jl"]
Private = false
```
