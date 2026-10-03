```@meta
CurrentModule = HammerheadGUI
```

# Saved-recipe preprocessing revisions and image previews

The [preprocessing workflow](../howto/gui_preprocessing_revision.md) extends
[`RecipeRevisionController`](@ref) with an ordered chain of the seven core
built-ins. [`preprocessing_fields`](@ref), [`set_revision_preprocess!`](@ref),
[`insert_revision_preprocess!`](@ref), [`move_revision_preprocess!`](@ref) and
[`delete_revision_preprocess!`](@ref) expose the editable schema and sequence.
[`set_revision_background!`](@ref) accepts a typed embedded replacement;
[`load_revision_background!`](@ref) explicitly decodes a file in recipe precision.
Shared validation, diff and distinct-save behavior follows
[`apply_recipe_revision!`](@ref) and [`save_recipe_revision!`](@ref).

These actions preserve every other imported recipe setting. Ordering, duplicates
and embedded background values/precision contribute to the recipe identity.
No new core schema, ancestry field or history migration is introduced. New
saved records preserve original ordered input identity and start with empty
runs. External script references are retained and verified as bytes, never
executed automatically.

[`RecipeImagePreviewController`](@ref) retains one detached successful selected
pair bundle. [`preview_recipe_images!`](@ref) captures the request before
callbacks, verifies original selected file bytes and dimensions around decoding,
conditions full frames in saved precision, and checks bytes again at completion.
ROI and mask remain overlays rather than numerical preprocessing stages. Any
external script reference refuses pixel preview while leaving metadata
inspection and distinct saving available. Failures before publication preserve
the previous bundle's captured identity; notification failures after publication
retain the new bundle without promising rollback. Published snapshots are mutable and do not certify
fresh source integrity on later access.

The new view and controller synthesize no vectors, result artifact, measurement
history or uncertainty. Async scheduling can still pause rendering and offers
no processing cancellation. The view supports saved built-in revision and an
original-pair preview; arbitrary preprocessing/plugin execution and scientific
preprocessing recommendations remain outside this workflow.

```@index
Pages = ["gui_preprocessing_revision.md"]
```

```@autodocs
Modules = [HammerheadGUI, HammerheadGUI.Controllers]
Pages = ["controllers/recipe_image_preview.jl", "views/preprocessing_revision.jl"]
Order = [:type, :function]
```
