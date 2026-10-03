```@meta
CurrentModule = HammerheadGUI
```

# GUI planar recipe revisions

The [pass-revision editor](../howto/gui_recipe_revision.md) creates a distinct
version-1 [`ExperimentRecord`](@ref) from a complete saved planar recipe. It
changes the ordered pass schedule while retaining every other recipe option
and each pass's validation tuple. The source record stays unchanged.

The same controller also holds ordered preprocessing drafts for the
[preprocessing editor](../howto/gui_preprocessing_revision.md), raw ROI/scale
drafts for the [geometry form](../howto/gui_recipe_geometry_revision.md), and
full-image bits for the [mask editor](../howto/gui_recipe_mask_revision.md).
Validation and saving compose all these drafts; every field outside those edits
is retained. Disabled geometry sections retain text but compose `nothing`.
Explicit image-pair conditioning uses a separate preview controller; metadata
differences describe settings, while pixel previews show their conditioning effect.

## Editable drafts and validation

[`revision_fields`](@ref) returns the pass editor's ordered field schema.
The editable fields cover window/search geometry, overlap, correlation and peak
settings, validation controls, iteration settings and retained correlation
planes. Each pass's imported validation tuple, including its order and options,
remains read-only. Insertion copies a complete raw draft and validation template;
moving a pass moves both together. At least one pass must remain.

Tuple text follows row, column order; booleans are `true` or `false`. Symbol
fields accept names such as `cross` or `:cross`. Real-valued fields use finite
Float64 parsing. Text is parsed as data. Raw invalid text survives
pass/group switching and duplication. Preview and save parse the complete draft,
including fields not currently visible, and refuse invalid active settings.

[`revision_recipe`](@ref) and [`revision_diff`](@ref) inspect the current draft
from the saved metadata. The preview hashes embedded recipe values and can work
offline. An invalid draft retains the last valid preview with its own settings;
the dirty indicator distinguishes it from the current draft. To assess numerical
changes, save the revision and [compare a representative pair](../howto/gui_comparison.md).

Every unedited imported option is retained: preprocessing and embedded background
precision/content, script reference, full-image mask, ROI, scale, CPU/KA backend,
Float32/Float64 image precision, threading, predictor smoothing, mask threshold
and uncertainty backend. The complete saved recipe supplies these values directly.
Scripts remain inspectable file references; inspection and saving handle their
metadata and byte identity.

## Saving and opening a revision

Creating/saving a revised record validates the complete current recipe and
verifies that the original ordered input bytes and dimensions are unchanged.
The fresh record preserves `input_id` and pairing order, captures the current
creation environment, and starts with empty run history and fresh record paths.
Keep the original and revised files as the explicit before/after records in the
ordinary version-1 format. Unchanged scientific settings keep the same recipe
identity even when saved to a different path.

The distinct-save action protects source record paths, image/script files,
original run outputs and additional known paths supplied through
`protected_paths`, including filesystem aliases. Supply additional known history
and report destinations when embedding the editor. Successfully consumed background/mask paths and
previously saved revision destinations remain protected for the controller's
lifetime, even if the public path list changes. This adds protection beyond
ordinary [`save_experiment`](@ref), which also supports updating an existing
record. Choose a new destination for each revision: an earlier saved record may
have acquired its own history in another workflow.

**Open saved revision** uses the last saved record, even when newer drafts are
unsaved, and creates a separate experiment workflow. The original workflow keeps
its displayed result. Choose Replay explicitly to process the new recipe under
the ordinary environment/progress/cancellation contract. Replaying a referenced
script requires an explicit caller-supplied preprocessing function.

## Captured requests and failures

The controller retains separate current drafts, last valid preview and last
saved record. Preview/save capture detached drafts, original identity and known
protected paths before observer notifications or picker calls. Direct relative
save destinations are resolved before those notifications; picker destinations
are resolved on return. Busy edit/action requests are refused.

Validation, input and destination failures before publication retain the source,
previous valid preview and last saved revision. Observer failures after values or
a file are published leave them in place: the status distinguishes a successful
save followed by notification failure.

The view queues work on the GUI owner task after native callbacks return. Input
hashing, embedded-array hashing, image decoding and file I/O share that task and
can delay event handling. Let the action finish before editing again. Use edit
helpers to update drafts while preserving the imported recipe's arrays.

```@index
Pages = ["gui_recipe_revision.md"]
```

```@autodocs
Modules = [HammerheadGUI, HammerheadGUI.Controllers]
Pages = ["controllers/recipe_revision.jl", "views/recipe_revision.jl"]
Order = [:type, :function]
```
