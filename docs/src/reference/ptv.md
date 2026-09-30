```@meta
CurrentModule = Hammerhead
```

# Particle tracking velocimetry (PTV)

Per-frame particle detection, hybrid particle image velocimetry (PIV)-guided
two-frame matching, scattered
validation, grid binning, and multi-frame trajectory linking. See the
[PTV tutorial](../tutorials/ptv.md) for a worked example and the
[conventions explanation](../explanation/conventions.md) for choosing between
PTV and PIV and interpreting the frame-A positions attached to PTV vectors.

```@index
Pages = ["ptv.md"]
```

```@autodocs
Modules = [Hammerhead]
Pages = ["particles.jl", "ptv.jl", "tracking.jl"]
Private = false
```
