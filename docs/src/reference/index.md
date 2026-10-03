# API reference

Look up functions, arguments and saved-data contracts here. For a worked
example, start with [your first vector field](../tutorials/first_vector_field.md)
or [the GUI tour](../tutorials/gui_tour.md).

## Images to measurements

| Task | Reference |
|:--|:--|
| Run planar PIV and configure passes | [Pipeline and parameters](pipeline.md) |
| Filter images and estimate backgrounds | [Preprocessing](preprocessing.md) |
| Validate or replace vectors | [Validation](validation.md) |
| Pool correlations or summarize a sequence | [Ensemble and statistics](ensemble.md) |
| Calibrate cameras and reconstruct stereo fields | [Calibration and stereo](stereo.md) |
| Detect, match and track individual particles | [PTV and trajectories](ptv.md) |
| Use GPU or portable kernel backends | [Backends](backends.md) |
| Generate images with known motion | [Synthetic data](synthetic.md) |

## Analysis, timing and export

- [Flow analysis](derived.md): profiles, circulation, gradients and spectra.
- [Image and result I/O](io.md): loading, batch processing and native files.
- [Calibrated grids](calibrated_resampling.md) and
  [particle/trajectory tables](calibrated_table.md): common-coordinate analysis
  and export.
- [Frame-pair timing](pair_timing.md), [stereo timing](stereo_pair_timing.md)
  and [actual-time tracking](tracking_timing.md): recorded times and their use.

## Repeating and reviewing experiments

| Need | Reference |
|:--|:--|
| Save a recipe and its runs | [Planar](experiments.md), [stereo](stereo_experiments.md), [ensemble](ensemble_experiments.md) |
| Resume interrupted work | [Checkpoints](checkpoints.md) |
| Compare settings on the same images | [Pair comparisons](pair_comparison.md) |
| Summarize completed results | [Run-quality reports](run_quality.md), [pooled results](ensemble_quality_reports.md), [saved ensemble results](ensemble_experiment_quality.md) |
| Inspect recorded pass behavior | [Planar](execution_diagnostics.md), [stereo](stereo_execution_diagnostics.md), [ensemble](ensemble_execution_diagnostics.md) |
| Distinguish measurement, rejection and filling | [Vector history](measurement_history.md) |

## Desktop tools

Start with the [GUI API](gui.md) for image preparation and batch controls, or
the [result explorer](gui_results.md) for fields and trajectories.

| Workflow | Reference |
|:--|:--|
| Open, save and replay experiments | [Planar](gui_experiments.md), [stereo](gui_stereo_experiments.md), [ensemble](gui_ensemble_experiments.md) |
| Revise a saved recipe | [Passes](gui_recipe_revision.md), [preprocessing](gui_preprocessing_revision.md), [ROI and scale](gui_recipe_geometry_revision.md), [masks](gui_recipe_mask_revision.md) |
| Compare saved recipes | [Comparison](gui_comparison.md) |
| Resume from a checkpoint | [Checkpoint tools](gui_checkpoints.md) |
| Inspect pooled results | [Ensemble inspection](gui_ensemble_companions.md) |

The [feature matrix](feature_matrix.md) records the scope of each method.
[Internal APIs](internals.md) are intended for contributors and may change.
