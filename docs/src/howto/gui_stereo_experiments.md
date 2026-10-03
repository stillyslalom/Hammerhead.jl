# Save and replay a fitted stereo experiment in the GUI

Use `stereo_experiment_workflow` to reopen fitted-stereo settings in their own
window. The record retains fitted cameras, dewarp grid, passes, camera-specific
preprocessing, static mask, ROI, scale, synchronization policy, and supplied
provenance. Replay reuses those saved camera coefficients and maps.

```julia
using Hammerhead, HammerheadGUI

# The record was created with fitted cameras and image-file acquisitions.
controller = StereoExperimentController("saved-stereo.jld2")
figure = stereo_experiment_workflow(controller)
```

The stereo batch view's **saved stereo workflow** button opens this window with
the current form attached. **Snapshot stereo batch** captures file-based camera
lists, fitted dewarpers, the effective window schedule, and scale. Effort presets
use dewarp-grid dimensions. For preprocessing, masks, or ROI, construct a complete
core `StereoPIVRecipe` and open it here. Snapshot validation reports unsupported
input or camera configurations before saving.

Choose **Files**, **Replay**, or **Reports** for that section's actions. Cancel,
written-acquisition progress, and status stay visible. Text pages show full
recipes, history, reports, and long paths. Embedded arrays appear as compact
dimensions/precision/content summaries; the record retains their exact values.
Calibration notes and self-calibration summaries identify the saved setup.

In **Files**, save the experiment and choose separate native-result and optional
run-history destinations. Replay starts at acquisition 1 with the captured
recipe and options. Explicit toggles record camera execution or pair timing.
Camera diagnostics use dewarped-pixel units; reconstructed fields use world units.

**Cancel after acquisition** stops at the current native-write boundary. The
workflow stays busy while prefetched loading, output closure, hashing, and history
handling finish. A request at the final write completes the recording; earlier
cancellation can leave a failed core run and readable native prefix. A request
before scheduled work retains existing files. Another replay starts at acquisition 1.

Select a historical run independently of the latest attempt. An older completed
run stays selectable after a later failure. **View completed results** checks its
native association, content, raw measurements, settings, and recorded packets,
then opens a separate lazy explorer with a fixed run identity. Use history pages
to inspect failed attempts.

In **Reports**, **save quality report** generates a TOML report for the selected
completed run. Current-field counts are the default; enable **include recorded
execution in report** for camera support/count summaries. **Check current input
bytes** also verifies local source files. Choices are captured before the dialog,
and a failed scan or save retains the prior report and its run/recipe/input IDs.

Choose separate destinations for exports. Guards protect known inputs, records,
and run outputs, including filesystem aliases. Use the core export APIs for the
current result. See the [stereo workflow reference](../reference/gui_stereo_experiments.md)
for verification and supported recipe settings.