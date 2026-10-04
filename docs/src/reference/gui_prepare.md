```@meta
CurrentModule = HammerheadGUI
```

# GUI Prepare editors

Part of the [GUI reference](gui.md).

```@index
Pages = ["gui_prepare.md"]
```

## Editors

The Prepare step's pages edit the workflow through these controllers
(`wf.prepare.preview`, `wf.prepare.mask[]`, `wf.prepare.roi[]`,
`wf.prepare.scale[]`). [`PreprocessPreview`](@ref Controllers.PreprocessPreview)
holds core `PreprocessStep`s and previews them with `recipe_preprocess`;
[`MaskEditor`](@ref Controllers.MaskEditor) exports its polygons with
`polygon_mask`; [`ROIEditor`](@ref Controllers.ROIEditor) edits core `ROI`
bounds; [`ScaleTool`](@ref Controllers.ScaleTool) measures a pixel size from
two points of known separation. Each editor also works on its own, given an
image or an image size.

```@autodocs
Modules = [HammerheadGUI.Controllers]
Order = [:module, :type, :function, :constant, :macro]
Pages = ["controllers/preprocess_preview.jl", "controllers/mask_editor.jl",
         "controllers/roi_editor.jl", "controllers/scale_tool.jl", "controllers/shared.jl"]
```
