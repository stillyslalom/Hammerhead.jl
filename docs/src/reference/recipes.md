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

Recipe files are JLD2 files holding plain dictionaries and a
`recipe_format_version`; they also record the Hammerhead and Julia versions
that wrote them. Custom validators and arbitrary preprocessing functions cannot
be saved; use the built-in validators and [`PreprocessStep`](@ref) operations.

```@index
Pages = ["recipes.md"]
```

```@autodocs
Modules = [Hammerhead]
Pages = ["recipes.jl"]
Private = false
```
