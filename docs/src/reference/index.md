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

## Analysis and export

- [Flow analysis](derived.md): profiles, circulation, gradients and spectra.
- [Image and result I/O](io.md): loading, batch processing, native files and
  table/VTK export.
- [Saved settings](recipes.md): recipes that rerun the same processing on new
  recordings.
- [Calibrated grids](calibrated_resampling.md): common-coordinate analysis of
  PIV and image data.

## Desktop tools

Start with the [GUI API](gui.md) for image preparation and batch controls, or
the [result explorer](gui_results.md) for fields and trajectories.

The [feature matrix](feature_matrix.md) records the scope of each method.
[Internal APIs](internals.md) are intended for contributors and may change.
