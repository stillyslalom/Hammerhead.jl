```@meta
CurrentModule = Hammerhead
```

# Saved settings (recipes)

A [`PIVRecipe`](@ref) holds the settings needed to process a recording again:
pass schedule, preprocessing steps, mask, ROI, physical scale, sequence or
ensemble mode, and precision. Save it with [`save_recipe`](@ref), run it with
[`apply_recipe`](@ref), and recover it from either a recipe file or a results
file with [`load_recipe`](@ref). [Save settings and reuse them](../howto/recipes.md)
walks through each step.

A recipe is written as TOML text carrying a `recipe_format_version` and the
Hammerhead version that wrote it. A `.toml` settings file keeps its arrays in
image files beside it: `<name>.mask.png` (white = excluded) and
`<name>.background.tif` (Float64 TIFF; per camera
`<name>.camera1.background.tif`, …). A JLD2 recipe or results file embeds the
same text as `recipe_toml` with the arrays under `recipe_arrays/`. Settings
left at `nothing` (no mask, ROI or scale) are omitted, and keys missing from a
file take their defaults. Custom validators and arbitrary preprocessing
functions cannot be saved; use the built-in validators and
[`PreprocessStep`](@ref) operations.

Per-pair masks passed to `apply_recipe(...; masks)` are input data, not part of
the recipe; a results file lists their image paths
([`load_sources`](@ref)`(path; masks = true)`).

```@index
Pages = ["recipes.md"]
```

```@autodocs
Modules = [Hammerhead]
Pages = ["recipes.jl"]
Private = false
```
