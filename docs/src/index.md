```@meta
CurrentModule = Hammerhead
```

# See the flow in your images

Two exposures of illuminated particles reveal how a fluid moves. Hammerhead
turns the change in their patterns into a field of displacement vectors.

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

*A synthetic particle pair and its measured swirl. Arrows show direction;
their lengths are enlarged for visibility.*

**[Start here: make your first vector field →](tutorials/first_vector_field.md)**

Follow one image pair from particles to a correlation peak and a complete
field. Then change the window size, mask a reflection, and convert pixels to
velocity. No recording or calibration equipment is needed to try it.

## Install

In Julia 1.10 or later, press `]` to enter package mode:

```julia
pkg> add Hammerhead CairoMakie
```

`CairoMakie` draws the lesson figures. For desktop tools, also install
`HammerheadGUI`.

## Choose your next experiment

- **Have a recording?** [Find a tip vortex](tutorials/real_data.md), including
  the region where particles disappear and the vectors become harder to judge.
- **Prefer desktop tools?** [Take the GUI tour](tutorials/gui_tour.md) to load
  images, draw masks and explore a field.
- **Have many frames?** [Measure flow statistics](tutorials/sequence_statistics.md)
  from a sequence rather than a single pair.
- **Need another view of motion?** [Use two cameras for stereo PIV](tutorials/stereo.md)
  or [follow individual particles](tutorials/ptv.md).

PIV follows particle *patterns* within small windows; PTV follows individual
particles. A spatial calibration and the exposure delay turn displacement into
velocity. Start with the images and check the measurement before interpreting
small flow features.
