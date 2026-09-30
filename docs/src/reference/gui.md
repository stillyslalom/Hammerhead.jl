```@meta
CurrentModule = HammerheadGUI
```

# Graphical user interface (GUI; HammerheadGUI)

The desktop GUI companion package uses GLMakie. Each component has a
controller (plain Julia + Observables, in the
`HammerheadGUI.Controllers` submodule) and a GLMakie view that renders it
and forwards user input.

```@index
Pages = ["gui.md"]
```

## Application and views

```@autodocs
Modules = [HammerheadGUI]
Order = [:module, :type, :function, :constant, :macro]
```

## Controllers

Controllers hold application state and logic. You can use them without
opening a window or creating a GL context.

```@autodocs
Modules = [HammerheadGUI.Controllers]
Order = [:module, :type, :function, :constant, :macro]
```
