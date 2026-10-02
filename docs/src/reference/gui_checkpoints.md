```@meta
CurrentModule = HammerheadGUI
```

# GUI checkpoint workflows

The [checkpoint workflow](../howto/gui_checkpoints.md) retains complete planar
recipes in a separate controller and resumes only the missing committed pairs.
Cancellation is acknowledged at a committed pair boundary. Recovery requires
an explicit stopped-writer assertion, and native attempt status remains distinct
from data completeness. Result browsing uses a fixed lazy prefix with one
display frame; refreshing the store does not append to an existing explorer.

```@index
Pages = ["gui_checkpoints.md"]
```

```@autodocs
Modules = [HammerheadGUI, HammerheadGUI.Controllers]
Pages = ["checkpoint_controller.jl", "checkpoint_workflow.jl"]
Order = [:type, :function]
```
