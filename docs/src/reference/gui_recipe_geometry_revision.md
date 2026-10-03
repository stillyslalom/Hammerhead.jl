```@meta
CurrentModule = HammerheadGUI
```

# Saved-recipe ROI and isotropic scale revisions

The [geometry revision workflow](../howto/gui_recipe_geometry_revision.md)
uses the shared [`RecipeRevisionController`](@ref).
[`revision_roi_fields`](@ref) and [`revision_scale_fields`](@ref) return detached
field schemas. [`set_revision_roi!`](@ref) and [`set_revision_scale!`](@ref)
merge raw text and explicitly enable/disable their sections. Drafts hold
`(enabled, values)`; disabling composes `nothing` while retaining text.

ROI keys are `row_first`, `row_last`, `col_first`, `col_last`: inclusive
original-image bounds. Scale keys are `pixel_size`, `dt`, `length_unit`,
`time_unit`: isotropic spatial factor, image-pair delay and opaque labels.
There are no inferred line endpoints or implicit label conversions. Enabled
numeric text is parsed into integer bounds or positive finite Float64 factors.
All input dimensions and saved pass/search geometry must remain compatible.

[`revision_recipe`](@ref), [`revision_diff`](@ref),
[`apply_recipe_revision!`](@ref) and [`save_recipe_revision!`](@ref) compose
geometry, pass and preprocessing drafts together. Every unedited scientific
setting is preserved, including full-image backgrounds/masks and exact imported
scale labels. Metadata validation does not read source images. Constructing or
saving a new record verifies original inputs, preserves ordered `input_id`,
captures the current environment and starts with no runs. No lineage or history
schema is added.

[`preview_recipe_images!`](@ref) captures revised geometry and recipe identity
with its explicit selected-pair request. Raw/processed matrices remain full-frame
unscaled pixels; ROI/mask are overlays. Core replay filters before cropping and
attaches scale to pixel-native PIV output. [`ResultExplorer`](@ref) converts raw
results through [`physical`](@ref) for display exactly once. Pixel diagnostics
and imported unit assertions are not independently calibrated by this editor.

The geometry view does not duplicate the image preview panel. Validation/save
work is queued outside native callbacks but is not a responsiveness or
cancellation guarantee. Original, candidate, saved and displayed-result
identities remain distinct; post-publication observer errors do not roll back
published values.

Shared controller actions are documented in the
[recipe revision reference](gui_recipe_revision.md); only the new view is
listed below.

```@index
Pages = ["gui_recipe_geometry_revision.md"]
```

```@autodocs
Modules = [HammerheadGUI]
Pages = ["views/recipe_geometry_revision.jl"]
Order = [:type, :function]
```
