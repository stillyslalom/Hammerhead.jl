```@meta
CurrentModule = Hammerhead
```

# Particle tracking velocimetry (PTV)

Use [`detect_particles`](@ref) to locate particle images,
[`run_ptv`](@ref) to match particles across a pair, and
[`track_particles`](@ref) to link matches across frames. A [`PTVResult`](@ref)
stores each displacement at its particle's position in frame A; its flagged
matches remain in the result for inspection. This page also covers
scattered validation and binning to a grid. See the
[PTV tutorial](../tutorials/ptv.md) for a worked example and the
[conventions page](../explanation/conventions.md) for coordinates and units.
For irregular sample intervals, use explicit
[actual-time tracking](../howto/tracking_timing.md); its wrapper and dedicated
file format preserve timing independently of ordinal tracking results.

```@index
Pages = ["ptv.md"]
```

```@autodocs
Modules = [Hammerhead]
Pages = ["particles.jl", "ptv.jl", "tracking.jl"]
Private = false
```
