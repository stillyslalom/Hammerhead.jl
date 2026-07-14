```@meta
CurrentModule = HammerheadGUI
```

# Graphical user interface (GUI; HammerheadGUI)

The desktop GUI companion package, built on GLMakie. Each component is a
framework-free controller (plain Julia + Observables, in the
`HammerheadGUI.Controllers` submodule) paired with a GLMakie view
function that renders it and forwards user input into it.

```@index
Pages = ["gui.md"]
```

## Application and views

```@autodocs
Modules = [HammerheadGUI]
Order = [:module, :type, :function, :constant, :macro]
```

## Controllers

Application state and logic, testable without a GL context — the submodule
never imports Makie.

```@autodocs
Modules = [HammerheadGUI.Controllers]
Order = [:module, :type, :function, :constant, :macro]
```
