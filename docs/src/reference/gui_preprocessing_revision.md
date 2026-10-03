```@meta
CurrentModule = HammerheadGUI
```

# Preprocessing revision API

Follow [Try high-pass filtering before replay](../howto/gui_preprocessing_revision.md)
for a runnable example and original/processed image comparison. These actions
share a [`RecipeRevisionController`](@ref) with pass, mask, ROI and scale edits.

## Ordered steps and options

The supported built-ins are background subtraction, intensity cap, high-pass
filtering, CLAHE, percentile stretching, inversion and local-variance
normalization. [`preprocessing_fields`](@ref) exposes each operation's editable
options, including CLAHE tile counts/bin count and local-variance epsilon.
Tuple fields use row, column order. Raw invalid text remains visible for
correction; validation parses the supported numeric and tuple fields.

[`set_revision_preprocess!`](@ref), [`insert_revision_preprocess!`](@ref),
[`move_revision_preprocess!`](@ref) and [`delete_revision_preprocess!`](@ref)
edit the ordered sequence. Duplicates remain separate applications. Ordering,
duplicates and embedded values/precision contribute to recipe identity.
Other imported settings remain intact: full passes/validation, mask, ROI,
scale, backend, image precision, threading, predictor and uncertainty options.
Sharing a controller composes all current drafts at validation/save time.

## Background values and precision

[`set_revision_background!`](@ref) accepts a typed embedded replacement. Imported
background matrices retain their stored precision, even when different from
`recipe.image_type`. [`load_revision_background!`](@ref) explicitly decodes a
chosen file in the recipe precision, verifies bytes across decoding and requires
the original full-frame size. Replay uses this supplied background at its stored
size and precision. The existing version-1 recipe embeds the matrix, and the
controller protects the consumed file path from revision-output aliases.

## Metadata, pixels and capture

[`apply_recipe_revision!`](@ref) parses and validates complete drafts and produces
a settings diff using recorded metadata. It works offline. Choose
[`preview_recipe_images!`](@ref) explicitly to inspect conditioned pixels.

[`preview_recipe_images!`](@ref) captures all drafts, the original record,
recipe/input identities and ordered pair selection before observer callbacks.
It validates that request, verifies both original file byte digests/dimensions
before and after decoding, conditions full frames in saved precision, then
checks bytes again after conditioning. Filtering takes place **before ROI
cropping**. Mask and ROI appear as read-only overlays on the complete images.

[`RecipeImagePreviewController`](@ref) retains only one detached successful
bundle: `raw_a`, `raw_b`, `processed_a`, `processed_b`, `roi`, `mask`,
`recipe_id`, `source_recipe_id`, `input_id`, `pair_index`, `input_paths`,
`input_descriptors` and `image_type`. A/B selection uses this same bundle.
Captured identity labels stay with their pixels when drafts change. The mutable
full-resolution arrays retain both raw and processed frames. Refresh the preview
to check current files and settings.

The view uses an explicit shared original/processed intensity range and shows
it above the panels. Constant-image ranges are padded for display. Cyan marks
the captured ROI; red marks exclusions. Prepublication failure retains the
previous bundle and its identities;
a notification failure after publication leaves the new bundle visible and
reports the error.

## Scripts and saving

Numerical preview supports recipes composed entirely of built-in steps. Recipes
with external scripts remain available for metadata inspection and saving, with
script bytes verified on save. Supply the preprocessing function explicitly
when replaying a scripted recipe.

[`save_recipe_revision!`](@ref) verifies original input/script bytes and creates
a distinct record with unchanged ordered input identity, current creation
environment and empty run history. Original records, known input/script/output/
history/report paths, consumed backgrounds and previous revision destinations
stay protected. Open the new record in a separate workflow for replay.

The saved workflow queues editor construction; the form queues validation,
decoding and conditioning after native callbacks. Direct constructors are
synchronous. Wait for queued decoding/conditioning to finish before another
action. Preview pixels first, then replay the saved recipe to obtain vectors.

## Preview controller and view

Shared actions are documented in the [recipe revision API](gui_recipe_revision.md).

```@autodocs
Modules = [HammerheadGUI, HammerheadGUI.Controllers]
Pages = ["controllers/recipe_image_preview.jl", "views/preprocessing_revision.jl"]
Order = [:type, :function]
```
