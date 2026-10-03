# Revise a saved ROI and physical scale

Choose **revise ROI / scale...** in the saved planar experiment workflow, or
open `recipe_geometry_revision(editor)` for a shared
[`RecipeRevisionController`](@ref). This editor retains the complete imported
recipe rather than rebuilding it from an effort preset. Ordered passes,
validation tuples, preprocessing, embedded backgrounds, the full-image mask,
precision and backend options remain intact. Pass and preprocessing edits in
the same controller are composed with the geometry edits.

Enable **ROI enabled** and enter inclusive row and column bounds in the
**original image's pixel coordinates**. Rows correspond to image y and columns
to image x. Every recorded input must contain those bounds, and every saved
pass's window and search area must fit the crop. Validation refuses an
incompatible schedule; it does not resize windows, shift the crop or regenerate
passes. **reset ROI to nothing** disables cropping. Disabling retains the raw
draft text for inspection but puts literal `nothing` in the composed recipe.

Enable **Physical scale enabled** to enter one isotropic length-per-pixel factor,
the positive image-pair delay `dt`, and length/time labels. Numeric factors must
be finite, positive and consistent with the labels you choose. The labels are
opaque assertions, not parsed units: changing `mm` to `m` does not convert a
factor. This form does not infer a calibration line, endpoints, rotation,
anisotropic registration or acquisition timestamps from a saved scale.
**reset scale to nothing** removes the conversion metadata; it does not select
an identity physical calibration. Imported factors round-trip their stored
Float64 values and imported labels retain their exact text.

Invalid raw text remains visible. Choose **validate / metadata diff** to parse
the complete draft afresh and inspect settings changes. Disabled sections
compose `nothing` even if retained text is invalid. Enabled sections must pass
validation. This metadata action hashes embedded arrays and checks recorded
dimensions without opening acquisition images; it can work offline. A settings
diff does not assess correlation, accuracy or uncertainty coverage.

Geometry editing does not automatically process or display an image. Use the
existing [preprocessing image preview](gui_preprocessing_revision.md) explicitly
with the same controller to inspect a verified original ordered pair. It captures
the revised recipe identity and ROI before callbacks, conditions each complete
frame, and shows the ROI in original coordinates with the unchanged full-image
mask. Neither ROI nor physical scale alters preview pixels. Filtering and
background subtraction occur on the **full frame before PIV crops the ROI**;
embedded backgrounds are not resized when the crop changes.

```julia
using HammerheadGUI

editor = RecipeRevisionController("original-experiment.jld2")
set_revision_roi!(editor,
    Dict(:row_first => "33", :row_last => "224",
         :col_first => "49", :col_last => "304"); enabled=true)
set_revision_scale!(editor,
    Dict(:pixel_size => "0.025", :dt => "0.001",
         :length_unit => "mm", :time_unit => "s"); enabled=true)
apply_recipe_revision!(editor; async=false)
figure = recipe_geometry_revision(editor)

images = RecipeImagePreviewController()
preview_recipe_images!(images, editor; pair_index=1, async=false)
```

Choose **save distinct revision...** to verify original input/script bytes and
create a separate version-1 record with the original ordered input identity,
the revised recipe, current creation environment and empty run history. Original
records and known inputs, histories, result/report paths and prior saved
revision destinations remain protected. The original workflow and displayed
result stay separate. Opening the new record does not invent parent-recipe
lineage or reinterpret the original run history.

When that record is replayed, native PIV centers retain the original pixel
offset of the ROI. Displacements and uncertainty stay in pixels, with the
chosen scale attached as metadata. The result explorer applies core
[`physical`](@ref) once: positions multiply by `pixel_size`, and displacements
and uncertainty multiply by `pixel_size / dt`. Converted display results carry
an identity scale with the labels so another `physical` call does not multiply
again. Removing scale metadata from an already converted result does **not**
restore pixel values. Use the raw native result when checking image-coordinate
overlays or pixel-based diagnostics.

The saved workflow queues editor construction, and the form queues validation
and saving after native callbacks. Calling the view constructor directly is
synchronous. Cooperative tasks can still pause rendering and offer no
cancellation or latency guarantee. Pre-publication validation/input failures preserve the prior
preview and saved revision. Observer failure after publication reports the
failure without rolling back the published values or file. This workflow does
not edit masks, estimate a calibration, validate unit assertions or change the
native result/experiment schemas. External scripts remain inspectable and
saveable, while numerical image preview refuses them.
