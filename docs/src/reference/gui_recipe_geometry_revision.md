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
Supply calibrated numeric factors in the chosen units. Enabled numeric text is
parsed into integer bounds or positive finite Float64 factors. All input dimensions
and saved pass/search geometry must fit the ROI; adjust the bounds or pass schedule
when validation reports an incompatible crop. Invalid raw text remains visible;
disabled sections retain it but compose
`nothing`. Reset actions remove the corresponding optional metadata.

[`revision_recipe`](@ref), [`revision_diff`](@ref),
[`apply_recipe_revision!`](@ref) and [`save_recipe_revision!`](@ref) compose
geometry, pass and preprocessing drafts together. Every unedited scientific
setting is preserved, including full-image backgrounds/masks and exact imported
scale labels. Metadata validation uses the stored dimensions and settings.
Saving a new record verifies original inputs, preserves ordered `input_id`,
captures the current environment and starts an empty run history.

[`preview_recipe_images!`](@ref) captures revised geometry and recipe identity
with its explicit selected-pair request. Raw/processed matrices remain full-frame
unscaled pixels; ROI/mask are overlays. Core replay filters before cropping and
attaches scale to pixel-native PIV output. [`ResultExplorer`](@ref) converts raw
results through [`physical`](@ref) for display exactly once.
Native centers retain the original-image ROI offset. Physical conversion
multiplies coordinates by `pixel_size`, and displacement/uncertainty by
`pixel_size / dt`; converted results carry an identity scale with the labels.
Use raw results for pixel-based overlays and diagnostics. Full-frame filtering
and background subtraction happen before cropping; embedded backgrounds retain
their full-image dimensions.

Use the preprocessing preview to inspect the ROI overlay. The form queues
validation and saving after native callbacks; large operations can pause rendering.
Original, candidate, saved and displayed-result identities remain distinct.
If a notification fails after publication, the published values remain available
and the controller reports the notification error.

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
