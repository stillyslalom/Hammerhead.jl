# Record and replay a fitted stereo experiment

A `StereoExperimentRecord` binds an ordered collection of four-file acquisitions
to frozen fitted cameras and processing settings. It has a separate versioned
format from planar `ExperimentRecord`; neither format changes registered result
layouts. Replay rebuilds dewarp maps and runs stereo PIV. It does not fit cameras
or rerun self-calibration.

Start with two fitted `ImageDewarper`s sharing a `DewarpGrid`, and list each
acquisition as `(camera1_A, camera1_B, camera2_A, camera2_B)`:

```julia
using Hammerhead

passes = multipass_parameters([64, 32]; padding=true)
recipe = StereoPIVRecipe(passes, dw1, dw2;
    preprocessing=([PreprocessStep(:intensity_cap; n_sigma=2)], PreprocessStep[]),
    image_type=Float32, backend=:cpu, threaded=false,
    world_unit="mm", coordinate_frame="fitted laser sheet",
    calibration_note="Supplied fitted camera models; target files not bound")
record = StereoExperimentRecord(acquisitions, recipe)
save_experiment("stereo-experiment.jld2", record)

saved = load_stereo_experiment("stereo-experiment.jld2")
run = replay_experiment(saved; output="stereo-results.jld2",
    run_record="stereo-experiment.jld2",
    record_diagnostics=true, record_pair_timing=true)
inspection = verify_stereo_experiment_run(saved, run;
    verify_results=true, verify_inputs=true)
```

`world_unit` and `coordinate_frame` describe the fitted world grid. They are
independent of `PhysicalScale` factors and display labels: setting a label does
not convert coordinates or displacement. Supply `scale` only when the ordinary
stereo driver should apply that scale. Camera fields retain dewarped pixel
coordinates and displacement; reconstructed fields follow existing signed-grid
and scale conventions.

## Capture acquisition metadata explicitly

Metadata vectors follow each camera's supplied A/B stream: two entries per
acquisition, including repeated files if they are selected again. Optional
source/frame IDs are opaque labels, not verified byte identities. Actual file
bytes and dimensions have separate identities.

```julia
epoch = big(2)^75
times = BigInt[epoch, epoch + 2, epoch + 5, epoch + 8]
record = StereoExperimentRecord(acquisitions, recipe;
    timestamps=(times, times), time_units=("ns", "ns"),
    clock_ids=("capture-clock", "capture-clock"),
    source_ids=("camera-1", "camera-2"),
    frame_ids=(["A1", "B1", "A2", "B2"], ["A1", "B1", "A2", "B2"]))
```

Provided integer, rational and supported floating timestamps retain their exact
encoding. Known inconsistent units/clocks, nonpositive delays, conflicting
declared delays and excessive camera skew reject. Missing labels/timestamps stay
unknown under the recorded `missing_timestamps=:allow` policy. Metadata is
snapshotted; subsequent caller edits do not change the recipe or source selection.
Capture reads the selected files to record bytes/dimensions. Replay validates all
metadata and destinations before decoding inputs or opening output.

Four-tuples retain the supplied `PhysicalScale.dt`, even if an observed timestamp
delay differs. To preserve explicit per-pair overrides, use the two-list
constructor instead:

```julia
camera1 = [FramePair(a[1], a[2], 2.0) for a in acquisitions]
camera2 = [FramePair(a[3], a[4], 2.0) for a in acquisitions]
record = StereoExperimentRecord(camera1, camera2, recipe)
```

In pair-list mode camera 1's declared delay replaces the scale delay as in the
ordinary driver; without scale it is only declared timing metadata. Timestamp
midpoints remain separate per camera. Any reconstructed reference time is
explicitly camera 1's midpoint, not an invented common exposure time. See
[stereo pair timing](stereo_pair_timing.md) for tolerance arithmetic and precision.

## Inspect a run without recomputing PIV

Replay is noncollecting: pixel/result arrays are bounded by current processing
and prefetch work, while scalar input descriptors and measurement hashes grow
with the number of acquisitions. CPU/KA Float32/64, builtin passes/validators,
two builtin raw-image preprocessing pipelines, a static common-grid mask, ROI
and scale are supported. Fitted Pinhole/Soloff cameras and a single rigid wrapper
are supported; arbitrary camera implementations, edited maps and scripts are not.

The strict default compares creation and execution environments and rebuilt map
digests. `allow_environment_change=true` is an explicit rerun override: actual
environment/map identities are recorded without promising numerical equivalence.
Input bytes are checked around loads and again at completion. Known destination
aliases are refused before writing. This does not lock sources against outside
writers or certify acquisition authenticity.

The native artifact contains a separate run association. Verification first
checks its whole-file hash and association. `verify_results=true` streams one raw
result at a time and independently checks precision, axes, masks, final pass
settings, ordered four-source labels, measurement-field hashes and requested
companions. Hash binding excludes parameters/correlation planes; final pass
parameters are checked separately. It does not verify calibration accuracy or
recompute PIV. `verify_inputs=true` additionally checks current local input bytes.
For a relocated result artifact, pass `output=new_path`; the historical locator
in the run remains unchanged.

Failures join outstanding loading, preserve the original error and can save an
`ExperimentRun` through `run_record`. `completed_pairs` counts completed four-file
acquisitions, not individual cameras. A failed native file may contain one extra
partially published entry; verification checks only the completed prefix and
reports `measurement_fields_checked_acquisitions` and
`unverified_trailing_entries`. More than one trailing entry, or any trailing
entry on a completed run, rejects. Saving the run record and native output is
sequential, not atomic publication or a resumable checkpoint.

An optional supplied `SelfCalibrationReport` contributes only detached scalar
descriptive provenance. Neither its initial calibration inputs nor its cumulative
transformation are verified. Calibration fitting/self-calibration reruns, stereo
GUI recipe editing, checkpointing and broader script recipes remain separate
workflows.
