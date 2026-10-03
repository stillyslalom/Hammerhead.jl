```@meta
CurrentModule = HammerheadGUI
```

# Saved mask revision API

Use [Exclude another reflection from a saved recipe](../howto/gui_recipe_mask_revision.md)
for an executed example. The editor shares a [`RecipeRevisionController`](@ref)
with pass, preprocessing, ROI and scale revisions.

## Draft and composition

`mask_draft` holds `(enabled::Bool, raster::Union{Nothing,BitMatrix})`.
[`set_revision_mask!`](@ref) copies supplied Bool bits. Disabling retains those
bits but composes literal `nothing`; enabling an all-false raster is a different
recipe from an absent mask. [`reset_revision_mask!`](@ref) restores the exact
imported optional mask. Saved raster bits form the editor's initial exclusions.

The complete recipe retains passes and validation tuples, ordered preprocessing,
embedded background precision, ROI, scale, backend, image precision, threading,
predictor and uncertainty settings. The retained `mask_threshold` determines
how masked pixels exclude interrogation windows.
Masks and backgrounds stay full-frame; replay conditions full images before
cropping the ROI. Apply updates recipe bits; replay produces the new vector field.

## Raw reference and editing session

[`load_recipe_mask_reference!`](@ref) captures the full draft and selected
original pair/frame before observers. It validates recipe metadata, then loads
only that selected full image in the saved precision. Original bytes and
recorded dimensions are checked before and after decoding, including the editor
copy. The reference displays decoded original pixels. Use the
[preprocessing preview](gui_preprocessing_revision.md) for a built-in chain's effect.

The detached bundle contains `editor`, `raw_image`, `roi`, `mask_draft`,
`recipe_id`, `source_recipe_id`, `input_id`, `pair_index`, `frame` and
`input_descriptor`. ROI is an original-coordinate overlay. The source identity
and displayed ROI remain those captured at loading, even after later edits.
Arrays are mutable snapshots. Reload the reference to check current file bytes.
Only the current bundle is retained; a failed load before publication retains
it. Observer failure after publication reports the error alongside the new values.

[`MaskEditor`](@ref) starts from a copied full-frame raster. Each exclusion
polygon unions with the accumulated raster; each hole removes its pixels from
that raster, including imported exclusions. Order matters. Grow/shrink replaces
the combined polygons with raster bits.
Clear-all removes both imported exclusions and polygons.

[`apply_revision_mask!`](@ref) captures the combined bits and complete current
recipe before callbacks. It refuses unfinished drawing, a different original
experiment, or a mask whose enabled state/bits differ from the session baseline.
Restoring identical content is permitted. Newer non-mask draft edits compose
normally. Apply validates full-frame geometry against all inputs and passes,
then advances the baseline for further edits. Apply checks the detached editing
session; loading and record saving check current input-file bytes.

## Import and save

[`load_revision_mask!`](@ref) uses core `load_mask`: grayscale values at or above
`threshold` are excluded, or values below it when `invert=true`. It verifies
bytes across decoding and requires matching full-image dimensions. The existing
version-1 recipe embeds the accepted bits; replay uses that raster snapshot.
Consumed mask paths remain protected against save aliases for the controller's
lifetime. A failed/cancelled import before publication retains the draft.

[`save_recipe_revision!`](@ref) verifies original acquisition/script bytes and
saves a distinct record with unchanged ordered input identity, current creation
environment and empty history. Known original/input/output/history/report paths
and previous saved revision destinations remain protected. Open the new record
in a separate workflow to replay it. Metadata inspection works offline;
reference loading and saving require the original acquisition identities.

The saved workflow queues editor construction and the form queues I/O and
validation outside native callbacks. Direct constructors are synchronous.
Wait for queued loading/validation to finish before another action. Recipes with
scripts support raw-reference editing, metadata inspection and saving; script
execution belongs to explicitly configured replay.

## Reference controller and view

The shared draft actions are documented in the
[recipe revision API](gui_recipe_revision.md).

```@autodocs
Modules = [HammerheadGUI, HammerheadGUI.Controllers]
Pages = ["controllers/recipe_mask_reference.jl", "views/recipe_mask_revision.jl"]
```
