```@meta
CurrentModule = HammerheadGUI
```

# Graphical user interface (GUI; HammerheadGUI)

The desktop GUI uses GLMakie. The application and view functions below open
interactive tools; `HammerheadGUI.Controllers` holds their state and actions
for scripted use without a display. Start with the
[GUI tour](../tutorials/gui_tour.md) for a worked session.

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
