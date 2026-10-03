# Development and validation

This section is for contributors and readers investigating how Hammerhead is
tested. To learn the analysis workflow, start with
[the tutorials](../tutorials/first_vector_field.md). To assess your own output,
start with [a run-quality report](../howto/run_quality.md).

## Test a scientific change

The [validation scorecard](../howto/validation_scorecard.md) is the starting
point for comparing a change with known synthetic motion and real-data smoke
cases. The studies below investigate particular sources of measurement error
and show how the results depend on their inputs and processing settings.

| Investigation | Study |
|:--|:--|
| Particle detection, matching and trajectory continuity | [Annotated particle tracks](../howto/validation_ptv_tracking.md) and [independent VSJ301 images](../howto/validation_vsj301.md) |
| Estimated uncertainty versus known motion | [Synthetic uncertainty](../howto/validation_uncertainty.md) |
| Where an uncertainty estimate comes from | [Estimator diagnostics](../howto/diagnostic_uncertainty.md) |
| Variation when only image noise changes | [Conditional noise experiments](../howto/conditional_uncertainty.md) |
| Sensitivity to image formation and interpolation | [Rendering experiments](../howto/rendering_uncertainty.md) |
| Which spatial variations a processing schedule retains | [Spatial response](../howto/spatial_transfer.md) |

## Change the software

The [GUI architecture](../explanation/gui.md) separates application logic from
its widgets. The [desktop framework evaluation](../explanation/gui_framework.md)
records the Qt prototype and its remaining adoption requirements.
The [compatibility policy](../explanation/compatibility.md) covers public APIs
and saved data when preparing a release.

The repository's `ROADMAP.md` is the development backlog. Learning pages should
teach a useful task; keep test inventories, validation evidence and delivery
status in the development and reference material.
