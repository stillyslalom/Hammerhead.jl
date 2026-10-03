```@meta
CurrentModule = HammerheadGUI
```

# GUI planar recipe revisions

The [pass-revision editor](../howto/gui_recipe_revision.md) creates a distinct
version-1 [`ExperimentRecord`](@ref) from a complete saved planar recipe. It
changes the ordered pass schedule while retaining every other recipe option
and each pass's validation tuple. It does not modify the source record or add
a new lineage, checkpoint or replay schema.

The same controller also holds ordered preprocessing drafts for the
[preprocessing editor](../howto/gui_preprocessing_revision.md). Validation and
saving compose both draft sequences; every field outside those edits is retained.
Explicit image-pair conditioning uses a separate preview controller and does not
turn a metadata difference into a numerical assessment.

`revision_recipe` and `revision_diff` inspect the current draft without input
image verification. Creating/saving a revised record verifies the original
ordered inputs, preserves `input_id`, captures the current environment, and
starts with no runs. Input verification and image decoding have their normal
I/O and memory costs; a task is not a responsiveness guarantee.

The controller retains separate current drafts, last valid preview and last
saved record. Raw invalid text survives edits, while preview/save actions refuse
it. Requests are detached before callbacks. Additional source/history/result
protection belongs to this revision workflow because the ordinary core
[`save_experiment`](@ref) API also supports updating an existing record.
Previously saved revision destinations remain protected. Observer failures do
not roll back values or a file that were already published; the status identifies
a save followed by notification failure separately from input/validation failure.

```@index
Pages = ["gui_recipe_revision.md"]
```

```@autodocs
Modules = [HammerheadGUI, HammerheadGUI.Controllers]
Pages = ["controllers/recipe_revision.jl", "views/recipe_revision.jl"]
Order = [:type, :function]
```
