# Understand the measurements

These pages explain the methods and conventions behind the results. For a
worked example, see [A first vector field](../tutorials/first_vector_field.md).

| Question | Read |
|:--|:--|
| Why does a positive vertical vector point down? | [Coordinates, signs and units](conventions.md) |
| How does a correlation peak measure displacement? | [Correlation and its sources of bias](correlation.md) |
| Why use more than one pass? | [Image deformation and smaller windows](multipass.md) |
| What happens inside an excluded region? | [How masks affect correlation and validation](masking.md) |
| Why can a window produce no measurement? | [Windows without displacement information](noninformative_windows.md) |
| What do the uncertainty estimates mean? | [Correlation uncertainty](uncertainty.md) |
| How accurate are the measurements on known motion? | [Synthetic accuracy, uncertainty and tracking tests](validation_results.md) |
| How can two cameras recover three components? | [Stereo geometry and self-calibration](stereo.md) |
| When should I use Float32 or Float64? | [Numerical precision](precision.md) |

Looking for arguments or return values? Use the [API reference](../reference/index.md).
Implementation choices live in the
[development documentation](../development/index.md).
