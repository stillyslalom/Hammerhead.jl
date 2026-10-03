```@meta
CurrentModule = HammerheadGUI
```

# GUI representative-pair comparison

The [comparison workflow](../howto/gui_comparison.md) delegates scientific
execution and persistence to [`compare_recipe_pair`](@ref) and
[`save_pair_comparison`](@ref). Complete saved recipes remain separate from
editable batch forms. Current request choices and the last report's historical
provenance are distinct. Loading a report is metadata-only inspection, not input
reverification or recipe replay.

```@index
Pages = ["gui_comparison.md"]
```

```@autodocs
Modules = [HammerheadGUI, HammerheadGUI.Controllers]
Pages = ["recipe_comparison.jl"]
Order = [:type, :function]
```
