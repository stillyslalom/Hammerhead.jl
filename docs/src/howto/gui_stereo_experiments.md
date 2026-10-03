# Save and replay a fitted stereo experiment in the GUI

Use `stereo_experiment_workflow` for saved fitted-stereo settings. It is separate
from the planar workflow. Opening a record preserves its complete recipe: fitted
camera coefficients, dewarp grid, exact passes, camera-specific preprocessing,
static grid mask, ROI, scale, synchronization policy and supplied provenance.
The workflow never refits cameras or loads scripts.

```julia
using Hammerhead, HammerheadGUI

# The record was created with fitted cameras and image-file acquisitions.
controller = StereoExperimentController("saved-stereo.jld2")
figure = stereo_experiment_workflow(controller)
```

The stereo batch view's **saved stereo workflow** button opens this lane with
the current form attached. **Snapshot stereo batch** captures idle file-based
camera lists, fitted dewarpers, exact effective window schedule and scale.
Effort presets resolve using the dewarp-grid dimensions. In-memory images,
arbitrary loaders, unsupported camera models and undocumented map edits refuse
before saving. The current form does not configure preprocessing, masks or ROI;
those richer settings are supplied through a core `StereoPIVRecipe` and reopened
intact, rather than silently added to a form snapshot.

Choose Files, Replay or Reports to expose that section's actions. Cancellation,
written-acquisition progress and a status preview remain visible. Complete
recipe, history and report text uses pages sized to the available space,
including long paths, IDs and error messages at 900×600 and larger sizes.
Embedded arrays are summarized by dimensions, precision and content digest;
their exact values remain in the saved record. Supplied calibration notes or
self-calibration summaries are provenance, not verified calibration accuracy.

In Files, save the intact experiment and choose separate native result and
optional run-history destinations. Replay starts from acquisition 1 with the
captured recipe and options. The environment override is unchecked by default.
Recording camera execution or pair timing is explicit and does not change the
scientific recipe identity. Camera diagnostics retain dewarped-pixel units;
they are separate from reconstructed world-coordinate fields.

**Cancel after acquisition** requests cancellation after the current native
write. The workflow stays busy until prefetched loading, output closure,
hashing and history handling finish. A request after the final write completes
normally. Earlier cancellation can leave a failed core run and a written native
prefix; it is not a resumable checkpoint. Cancellation before scheduled work
starts leaves files untouched. Cooperative Julia tasks yield at acquisition
boundaries; validation, current computation and I/O can pause rendering.
Offscreen checks do not establish native desktop responsiveness.

The run selector chooses a historical run independently of the latest attempt.
The paged text distinguishes the captured active request from next replay
destinations, the latest recorded attempt and the selected historical run.
An older completed run remains selectable after a later failure. **View
completed results** verifies the selected native association, file content,
raw measurement fields, expected grids/masks/settings and recorded companions
before opening a lazy explorer. Its displayed run ID remains fixed when the
workflow selection changes. Only one display frame is retained; physical
conversion occurs once. Failed runs are inspected as history, not as verified
completed output. Concurrent file writers are unsupported.

In Reports, **save quality report** generates the associated core TOML report
for the selected completed run. The default format 1 counts current stored
fields; **include recorded execution in report** selects format 3 support/count
summaries. Stereo per-node measurement history is unavailable. **Check current
input bytes** additionally verifies original local inputs; it does not recompute
PIV. Report choices, selected record/run and protected destinations are captured
before the save dialog. Failed scans or saves keep the prior report and its own
run/recipe/input IDs. Verification describes generation time, not continuing
validity, uncertainty applicability or measurement accuracy.

The core and GUI guards protect known input, experiment-record, run-output and
selected destination aliases. Programmatic exports of a current result use the
existing core export APIs; this workflow adds no new calibrated-table or
trajectory export semantics. Stereo representative-pair comparison, checkpoint
resume, calibration fitting and native Qt adoption remain separate work.
