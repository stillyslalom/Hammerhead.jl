# Revise the passes of a saved planar recipe

Choose **revise pass schedule** in the [saved-experiment workflow](gui_experiments.md).
It captures a separate copy of the selected record. The source recipe, its run
history and the displayed result remain separate from the revision.

Select a pass and a field group. Edit the interrogation window, search area,
overlap, correlation and peak settings, validation controls, iteration settings
or retained-plane option. Tuple entries follow **row, column** order. Boolean
entries are `true` or `false`. The imported validation tuple remains read-only,
including its order and options. Duplicate, move or delete passes to change the
ordered schedule; at least one pass must remain.
Names such as `cross` or `:cross` are accepted for symbol fields. Real-valued
entries use finite Float64 parsing; text is never evaluated as Julia code.

Choose **validate / preview changes** to inspect the current complete draft.
Invalid visible text is retained and refused; the editor does not substitute a
previously valid value. Switching pass or field group preserves the text you
entered. Invalid drafts keep the last valid preview, which describes its own
settings rather than certifying the current draft.

The preview compares settings, not numerical results. It can inspect a valid
saved record without reopening its image files. It does not establish accuracy,
uncertainty coverage or replay equivalence. To rerun a selected pair, save the
revision and use [representative-pair comparison](gui_comparison.md).

All other recipe options stay intact: built-in preprocessing and embedded
backgrounds, a referenced script, the full-image mask, ROI, scale, CPU/KA
backend, Float32/Float64 precision, threading, predictor smoothing, mask
threshold and uncertainty backend. Scripts are retained as references and are
never executed by revision inspection or saving. This editor does not translate
the imported recipe into the ordinary batch form or an effort preset.

Choose **save distinct revision** and a separate experiment destination. Saving
rechecks the current draft, verifies that the original ordered inputs are
available and unchanged, and captures the current creation environment. A
successful revision has the same input identity and pairing order, its revised
recipe identity, and an empty run history. No parent/revision lineage is added
to the core version-1 record schema; keeping both records provides the explicit
before/after artifacts.
Unchanged settings keep the same recipe identity; saving does not force a new
scientific identity merely because the destination differs.

The source record, image files, referenced script, prior result files and known
history/output destinations are protected against aliases. Paths successfully
saved by this editor also stay protected: choose another path for a subsequent
revision, since an opened revision may have acquired its own history. Saving
does not overwrite the original experiment. Validation, input or protected-path
failures retain the source, previous preview and last saved revision. File writes
are not an atomic publication or a durability guarantee. If an observer throws
after new values or a successful save have been published, the error is reported
without rolling those values or the file back; their displayed identities remain
explicit.

**Open saved revision** creates a separate saved-experiment workflow. It opens
the last saved record, even if newer draft edits have not been saved. Replay
still uses the existing environment checks, progress and cancellation contract;
editing or saving a recipe does not execute PIV or inherit a completed run.
Replaying a referenced script still requires an explicit caller-supplied
preprocessing function; opening the revision does not load code from that file.

```julia
using HammerheadGUI

editor = RecipeRevisionController("original-experiment.jld2")
draft = copy(editor.drafts[][1])
draft[:max_iterations] = "3"
set_revision_pass!(editor, 1, draft)
apply_recipe_revision!(editor; async=false)
save_recipe_revision!(editor, "revised-experiment.jld2"; async=false)
figure = recipe_revision(editor)
```

Applications embedding the editor should pass their additional known artifact
paths through `protected_paths`. Use the controller's edit helpers rather than
mutating imported recipe arrays. Inspect `revision_fields()` for the editable
field schema; each pass retains its read-only validation tuple.

The view queues preview/save work after native callbacks return. Input hashing,
image decoding and file I/O can still pause rendering. Asynchronous scheduling
does not establish background responsiveness, cancellation or desktop
accessibility. Busy actions are refused, and the captured save request remains
independent of later widget or observer changes.
