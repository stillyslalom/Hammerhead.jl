```@meta
CurrentModule = Hammerhead
```

# Input/output (I/O) and batch processing

Load images with [`load_image`](@ref), build path pairs with
[`image_pairs`](@ref), and process them with [`run_piv_sequence`](@ref).
[`save_results`](@ref) and [`load_results`](@ref) write and read JLD2 result
files. The sequence driver accepts file paths or in-memory arrays and can
write results as pairs finish. See [Batch processing](../howto/batch.md)
for an end-to-end workflow. The time between images in a pair and the time
between successive pairs serve different purposes; see the
[sequence tutorial](../tutorials/sequence_statistics.md).

```@index
Pages = ["io.md"]
```

```@autodocs
Modules = [Hammerhead]
Pages = ["io.jl", "interoperability.jl"]
Private = false
```
