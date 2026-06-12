"""
    SyntheticData

Synthetic PIV test data with particle-based image generation. Ground truth
comes from displacing particle positions directly (no interpolation warping),
including out-of-plane motion and a laser-sheet intensity profile.
"""
module SyntheticData

using Random

export ParticleField3D, generate_gaussian_particle!, generate_particle_field,
       displace_particles, displace_particles!, render_particle_image
export generate_synthetic_piv_pair, linear_flow, vortex_flow, shear_flow
export LaserSheet, GaussianLaserSheet

"""
    ParticleField3D

A collection of tracer particles with 3D positions and per-particle properties.
`x` is the in-plane column coordinate, `y` the in-plane row coordinate, and `z`
the out-of-plane coordinate (the laser sheet typically sits at `z = 0`).
"""
struct ParticleField3D{T<:Real}
    x::Vector{T}
    y::Vector{T}
    z::Vector{T}
    intensity::Vector{T}
    diameter::Vector{T}
end

Base.copy(p::ParticleField3D) =
    ParticleField3D(copy(p.x), copy(p.y), copy(p.z), copy(p.intensity), copy(p.diameter))

"""
    LaserSheet

Abstract type for laser sheet intensity profiles. Subtypes implement
`laser_intensity(sheet, z)`.
"""
abstract type LaserSheet end

"""
    GaussianLaserSheet(z_center, thickness, peak_intensity)

Gaussian laser sheet profile: intensity falls off as a Gaussian in `z` with
full width at half maximum `thickness`, centered at `z_center`.
"""
struct GaussianLaserSheet{T<:Real} <: LaserSheet
    z_center::T
    thickness::T
    peak_intensity::T
end

"""
    laser_intensity(laser::LaserSheet, z) -> Real

Laser intensity multiplier at out-of-plane position `z`.
"""
function laser_intensity(laser::GaussianLaserSheet, z::Real)
    σ = laser.thickness / (2 * sqrt(2 * log(2)))  # FWHM → standard deviation
    return laser.peak_intensity * exp(-(z - laser.z_center)^2 / (2 * σ^2))
end

# Velocity field functions have signature (x, y, z, t) -> (u, v, w).

"""
    linear_flow(u₀, v₀, w₀, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y) -> velocity_function

Create a linear velocity field `(x, y, z, t) -> (u, v, w)`.
"""
function linear_flow(u₀, v₀, w₀, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y)
    return (x, y, z, t) -> (u₀ + ∂u∂x * x + ∂u∂y * y,
                            v₀ + ∂v∂x * x + ∂v∂y * y,
                            w₀)
end

"""
    vortex_flow(center_x, center_y, strength, w₀ = 0.0) -> velocity_function

Create a circular vortex velocity field with constant azimuthal speed
`strength` around `(center_x, center_y)`.
"""
function vortex_flow(center_x, center_y, strength, w₀ = 0.0)
    return (x, y, z, t) -> begin
        dx = x - center_x
        dy = y - center_y
        r = sqrt(dx^2 + dy^2)
        r > 0 || return (0.0, 0.0, w₀)
        return (-strength * dy / r, strength * dx / r, w₀)
    end
end

"""
    shear_flow(shear_rate, w₀ = 0.0) -> velocity_function

Create a simple shear velocity field `u = shear_rate * y`.
"""
function shear_flow(shear_rate, w₀ = 0.0)
    return (x, y, z, t) -> (shear_rate * y, 0.0, w₀)
end

"""
    generate_particle_field(image_size, particle_density; kwargs...) -> ParticleField3D

Generate a random 3D particle field for synthetic PIV images.

# Arguments
- `image_size::Tuple{Int,Int}`: image size `(rows, cols)`
- `particle_density::Real`: average particles per pixel

# Keywords
- `z_range = (-5.0, 5.0)`: range of out-of-plane positions
- `diameter_mean = 3.0`, `diameter_std = 0.5`: particle image diameter
  distribution in pixels (normal, clipped to ≥ 0.5)
- `intensity_mean = 1.0`, `intensity_std = 0.2`: particle intensity
  distribution (normal, clipped to ≥ 0.1)
- `rng = Random.default_rng()`: random number generator
"""
function generate_particle_field(
    image_size::Tuple{Int,Int},
    particle_density::Real;
    z_range::Tuple{Real,Real} = (-5.0, 5.0),
    diameter_mean::Real = 3.0,
    diameter_std::Real = 0.5,
    intensity_mean::Real = 1.0,
    intensity_std::Real = 0.2,
    rng::AbstractRNG = Random.default_rng(),
)
    height, width = image_size
    n = round(Int, particle_density * height * width)

    x = rand(rng, n) * width
    y = rand(rng, n) * height
    z = z_range[1] .+ rand(rng, n) * (z_range[2] - z_range[1])
    diameter = max.(diameter_mean .+ diameter_std .* randn(rng, n), 0.5)
    intensity = max.(intensity_mean .+ intensity_std .* randn(rng, n), 0.1)

    return ParticleField3D(x, y, z, intensity, diameter)
end

"""
    displace_particles!(particles::ParticleField3D, velocity_function, dt, t = 0.0)

Displace particles in place by one step `dt` of `velocity_function`
(signature `(x, y, z, t) -> (u, v, w)`, evaluated at time `t`).
"""
function displace_particles!(particles::ParticleField3D, velocity_function, dt::Real, t::Real = 0.0)
    for i in eachindex(particles.x)
        u, v, w = velocity_function(particles.x[i], particles.y[i], particles.z[i], t)
        particles.x[i] += u * dt
        particles.y[i] += v * dt
        particles.z[i] += w * dt
    end
    return particles
end

"""
    displace_particles(particles::ParticleField3D, velocity_function, dt, t = 0.0) -> ParticleField3D

Non-mutating version of [`displace_particles!`](@ref).
"""
displace_particles(particles::ParticleField3D, velocity_function, dt::Real, t::Real = 0.0) =
    displace_particles!(copy(particles), velocity_function, dt, t)

"""
    generate_gaussian_particle!(image, (cx, cy), diameter, intensity = 1.0)

Add a Gaussian particle image centered at `(cx, cy)` — `cx` along columns (x),
`cy` along rows (y) — with `σ = diameter / 4`. Intensity is accumulated within
a 3σ bounding box, clipped to the image.
"""
function generate_gaussian_particle!(image::AbstractMatrix{T}, center::Tuple{Real,Real},
                                     diameter::Real, intensity::Real = 1.0) where {T<:Real}
    cx, cy = center
    σ = diameter / 4.0
    radius = ceil(Int, 3 * σ)
    j_range = max(1, round(Int, cx - radius)):min(size(image, 2), round(Int, cx + radius))
    i_range = max(1, round(Int, cy - radius)):min(size(image, 1), round(Int, cy + radius))

    @inbounds for j in j_range, i in i_range
        r² = (j - cx)^2 + (i - cy)^2
        image[i, j] += T(intensity * exp(-r² / (2 * σ^2)))
    end
    return image
end

"""
    render_particle_image(particles::ParticleField3D, image_size, laser::LaserSheet;
                          kwargs...) -> Matrix{Float64}

Render a particle field into an image, modulating each particle's intensity by
the laser sheet profile at its `z` position. Particles whose centers lie just
outside the frame still contribute their in-frame tails, so pairs of rendered
images stay consistent near the borders.

# Keywords
- `background_noise = 0.01`: Gaussian background noise level (0 disables)
- `intensity_threshold = 0.01`: minimum laser intensity multiplier for a
  particle to be rendered
- `rng = Random.default_rng()`: random number generator (noise only)
"""
function render_particle_image(
    particles::ParticleField3D,
    image_size::Tuple{Int,Int},
    laser::LaserSheet;
    background_noise::Real = 0.01,
    intensity_threshold::Real = 0.01,
    rng::AbstractRNG = Random.default_rng(),
)
    height, width = image_size
    image = zeros(Float64, height, width)

    for i in eachindex(particles.x)
        margin = 3 * particles.diameter[i] / 4  # 3σ render radius
        (1 - margin <= particles.x[i] <= width + margin &&
         1 - margin <= particles.y[i] <= height + margin) || continue
        laser_mult = laser_intensity(laser, particles.z[i])
        laser_mult > intensity_threshold || continue
        generate_gaussian_particle!(image, (particles.x[i], particles.y[i]),
                                    particles.diameter[i], particles.intensity[i] * laser_mult)
    end

    if background_noise > 0
        image .+= background_noise .* randn(rng, height, width)
        image .= max.(image, 0.0)
    end
    return image
end

"""
    generate_synthetic_piv_pair(velocity_function, image_size, dt;
                                kwargs...) -> (img1, img2, particles1, particles2)

Generate a synthetic PIV image pair with known ground truth: a random particle
field is rendered at time `t0`, displaced through `velocity_function`
(signature `(x, y, z, t) -> (u, v, w)`) over `dt`, and rendered again. Out-of-
plane motion changes particle brightness via the laser sheet profile.

# Keywords
- `particle_density = 0.05`: particles per pixel
- `laser_sheet = GaussianLaserSheet(0.0, 2.0, 1.0)`
- `background_noise = 0.01`: per-image Gaussian noise level
- `z_range = (-5.0, 5.0)`: range of particle z-positions
- `t0 = 0.0`: time at the first image
- `rng = Random.default_rng()`
- remaining keywords are forwarded to [`generate_particle_field`](@ref)

Returns the two images and the two particle fields (for ground-truth analysis).
"""
function generate_synthetic_piv_pair(
    velocity_function,
    image_size::Tuple{Int,Int},
    dt::Real;
    particle_density::Real = 0.05,
    laser_sheet::LaserSheet = GaussianLaserSheet(0.0, 2.0, 1.0),
    background_noise::Real = 0.01,
    z_range::Tuple{Real,Real} = (-5.0, 5.0),
    t0::Real = 0.0,
    rng::AbstractRNG = Random.default_rng(),
    kwargs...,
)
    particles1 = generate_particle_field(image_size, particle_density; z_range, rng, kwargs...)
    particles2 = displace_particles(particles1, velocity_function, dt, t0)

    img1 = render_particle_image(particles1, image_size, laser_sheet; background_noise, rng)
    img2 = render_particle_image(particles2, image_size, laser_sheet; background_noise, rng)

    return (img1, img2, particles1, particles2)
end

end # module SyntheticData
