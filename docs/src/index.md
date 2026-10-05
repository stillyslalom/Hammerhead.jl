```@meta
CurrentModule = Hammerhead
```

# Hammerhead.jl

```@raw html
<img src="assets/logo.svg" alt="Hammerhead logo" width="112" style="float: right; margin: 0 0 1em 1.5em;">
```

Hammerhead measures fluid motion from particle images: planar particle image
velocimetry (PIV), stereo PIV with camera calibration and self-calibration,
and particle tracking (PTV), with uncertainty estimates, statistics and
derived quantities. HammerheadGUI provides a desktop workflow for the same
analyses.

```@example welcome
using Hammerhead, CairoMakie, Random # hide
using Hammerhead.SyntheticData # hide
swirl(x,y,z,t) = (-(y-64)*2/hypot(x-64,y-64,18), (x-64)*2/hypot(x-64,y-64,18), 0.0) # hide
a,b,_,_ = generate_synthetic_piv_pair(swirl,(128,128),1.0;particle_density=.06,background_noise=.02,z_range=(-1.,1.),rng=MersenneTwister(42)) # hide
r = run_piv(a,b,multipass_parameters([32,16];padding=true,apodization=:gauss)) # hide
fig = Figure(size=(960,310)) # hide
for (k,image,title) in ((1,a,"First exposure"),(2,b,"Next exposure")) # hide
    ax = Axis(fig[1,k];title,yreversed=true,aspect=DataAspect(),xlabel="x (px)",ylabel="y (px)") # hide
    image!(ax,(0.5,128.5),(0.5,128.5),image';colormap=:grays) # hide
end # hide
ax = Axis(fig[1,3];title="Measured motion",yreversed=true,aspect=DataAspect(),xlabel="x (px)",ylabel="y (px)") # hide
image!(ax,(0.5,128.5),(0.5,128.5),a';colormap=:grays) # hide
plot_vector_field!(ax,r;stride=2,color=:cyan,lengthscale=3) # hide
fig # hide
```

*A synthetic particle pair and its measured displacement field; arrow
lengths are enlarged.*

[A first vector field](tutorials/first_vector_field.md) follows one image pair
from particle images to a correlation peak and a complete field, then covers
window size, masking and conversion to velocity, using synthetic images.

## Install

In Julia 1.10 or later, press `]` to enter package mode:

```julia
pkg> add Hammerhead CairoMakie
```

`CairoMakie` draws the tutorial figures. The desktop workflow is in the
separate `HammerheadGUI` package (`pkg> add HammerheadGUI`); start Julia with
several threads (`julia -t auto`) and open it with `using HammerheadGUI;
hammerhead()`.

## Tutorials

- [A wing-tip vortex from a real recording](tutorials/real_data.md): PIV on
  PIV Challenge images, including a vortex core with few particles.
- [From image pairs to flow statistics](tutorials/sequence_statistics.md):
  sequences, mean and fluctuation fields, and ensemble correlation.
- [Stereo PIV end to end](tutorials/stereo.md) and
  [a real stereo recording](tutorials/stereo_real.md): calibration, dewarping,
  self-calibration and three-component reconstruction.
- [Particle tracking](tutorials/ptv.md): detection, matching and trajectories.
- [A PIV session in the GUI](tutorials/gui_tour.md): the same workflow in the
  desktop window.

PIV measures the displacement of particle *patterns* within interrogation
windows; PTV follows individual particles. A spatial calibration and the
exposure delay convert displacement to velocity.
