# Revise saved preprocessing and preview an original pair

Open the preprocessing revision view for a saved planar experiment. It keeps
the imported recipe and history separate from the editable draft. The existing
[pass revision editor](gui_recipe_revision.md) uses the same controller; neither
projects an imported recipe into an effort preset or the narrower batch form.

The ordered list supports adding, duplicating, moving and deleting steps. Every
listed step runs in order: two copies of an operation are two applications,
not a single enabled toggle. The seven supported built-ins are background
subtraction, intensity cap, high-pass filtering, CLAHE, percentile stretching,
intensity inversion and local-variance normalization. Edit all options for the
selected operation, including CLAHE's tile counts and bin count and local
variance's epsilon. Tuple fields use row, column order. Raw invalid text stays
visible and is refused by validation; text is never evaluated as Julia code.

Background subtraction uses a complete embedded matrix. An imported background
retains its stored precision, even if it differs from the recipe's image
precision. **Load background as recipe Float32/64** explicitly decodes a chosen
image in the recipe precision and replaces the selected step's background.
There is no automatic estimation, resizing or recomputation during replay.
The replacement must match the original full frame. Its values and precision
are embedded in the ordinary version-1 recipe, rather than stored as a new
external preprocessing dependency. The consumed file remains protected against
revision-output aliases.

Choose **validate / metadata diff** for a settings diff. This action hashes
embedded recipe values but does not open acquisition images or execute PIV.
The metadata preview can therefore work offline. It describes settings, not
improved correlation, accuracy or uncertainty coverage. Other recipe options
remain intact: the complete pass schedule and validation tuples, mask, ROI,
scale, backend, image precision, threading and predictor/uncertainty options.

Choose an original ordered pair index and **preview verified pair** to perform
pixel conditioning explicitly. The request captures the current complete draft,
recipe/input identities and pair selection before observer callbacks. Both
files must match the original byte digests and dimensions before and after
decoding; their bytes are checked again after conditioning. The preview uses
the core operations in the saved image precision on each complete frame.

ROI and mask are read-only overlays. They do not crop, zero or otherwise alter
preview pixels: full-frame filtering occurs **before** the ROI stage used by
PIV. This matters for filter boundaries, percentile statistics and CLAHE tile
layout. A/B selection shows the two captured original/processed images from
one bundle. Captured recipe, input and pair labels belong to that bundle; edits
do not relabel old pixels as a new draft. Original and processed panels use one
explicit shared intensity range, shown above the images. Cyan outlines the ROI;
red marks excluded pixels. The display does not independently normalize the two
panels. Check numerical pixel values when assessing contrast, not just appearance.

A changed or missing selected input, invalid option or failure before publication
retains the last successful bundle with its own identities. A notification error
after publication leaves the newly published bundle in place and reports the
failure; publication is not rolled back. Availability of
that preview is not a continuous assertion that files remain unchanged. Only
one current pair bundle is retained; full-resolution raw and processed A/B
matrices still have their normal memory cost. They are publicly mutable
snapshots, not a persisted result or integrity proof.

Referenced scripts remain inspectable and saveable, but **any external script
reference refuses numerical image preview**. No script is included or evaluated,
even if its built-in prefix could run independently. Saving verifies the script
bytes; actual replay still requires the existing explicit caller-supplied
preprocessing function.

Choose **save distinct revision** to create a new record with the original
ordered input identity, the revised complete recipe, current creation environment
and empty run history. Original records, input/script files, known output/history
paths, consumed background files and previously saved revision destinations
remain protected. There is no new lineage schema. Open the saved revision in a
separate workflow; it does not replace the original workflow's displayed result.

```julia
using HammerheadGUI

editor = RecipeRevisionController("original-experiment.jld2")
index = length(editor.preprocessing_drafts[]) + 1
insert_revision_preprocess!(editor, index, :highpass_filter)
set_revision_preprocess!(editor, index, Dict(:sigma => "2.0"))
apply_recipe_revision!(editor; async=false)

images = RecipeImagePreviewController()
preview_recipe_images!(images, editor; pair_index=1, async=false)
figure = preprocessing_revision(editor; preview_controller=images)
```

The view queues expensive validation, file decoding and conditioning after
native callbacks return. This is not a background responsiveness or cancellation
guarantee. Previewing pixels performs no PIV, matching, uncertainty estimation,
registration or PLIF concentration calibration. Static appearance does not
establish stationarity or measurement accuracy. If an observer fails after
values or a saved file were published, the error is reported without promising
rollback or an atomic file transaction.
