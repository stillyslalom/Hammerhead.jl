"""
    HammerheadGUI

Desktop GUI for Hammerhead.jl (particle image velocimetry) built on GLMakie.

Application state and logic live in a framework-free controller layer (plain
Julia + Observables); Makie code renders controllers and pushes user input
into them, but controllers never depend on Makie, so the widget shell stays
swappable and the logic testable without a GL context.
"""
module HammerheadGUI

using Hammerhead
using GLMakie
using Observables
using NativeFileDialog

end # module HammerheadGUI
