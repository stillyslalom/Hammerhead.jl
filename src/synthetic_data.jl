"""
    SyntheticData

Submodule for generating synthetic PIV test data with proper particle-based image generation.
Provides accurate ground truth by directly displacing particle positions rather than
using bilinear interpolation warping. Includes realistic effects of out-of-plane motion
and laser sheet intensity profiles.
"""
module SyntheticData

using Random
using Distributions

export ParticleField3D, generate_particle_field, displace_particles!, render_particle_image
export generate_synthetic_piv_pair, linear_flow, vortex_flow, shear_flow
export LaserSheet, GaussianLaserSheet

# ============================================================================
# Core Data Structures
# ============================================================================

"""
    ParticleField3D

Represents a collection of particles with 3D positions and properties.
Includes z-coordinate for out-of-plane motion effects.
"""
struct ParticleField3D{T<:Real}
    x::Vector{T}          # X positions (in-plane)
    y::Vector{T}          # Y positions (in-plane)
    z::Vector{T}          # Z positions (out-of-plane)
    intensity::Vector{T}  # Base particle intensities
    diameter::Vector{T}   # Particle diameters
end

"""
    LaserSheet

Abstract type for laser sheet intensity profiles.
"""
abstract type LaserSheet end

"""
    GaussianLaserSheet(z_center, thickness, peak_intensity)

Gaussian laser sheet profile with specified center, thickness, and peak intensity.
Intensity falls off as exp(-(z-z_center)²/(2σ²)) where σ = thickness/4.
"""
struct GaussianLaserSheet{T<:Real} <: LaserSheet
    z_center::T      # Z-position of laser sheet center
    thickness::T     # Full-width at half-maximum of laser sheet
    peak_intensity::T # Peak intensity multiplier
end

# ============================================================================
# Flow Field Functions
# ============================================================================

"""
Velocity field functions should have signature: (x, y, z, t) -> (u, v, w)

Example flows are provided as functions:
- `linear_flow(u₀, v₀, w₀, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y)` - Linear velocity field
- `vortex_flow(center_x, center_y, strength, w₀=0.0)` - Circular vortex
- `shear_flow(shear_rate, w₀=0.0)` - Simple shear flow
"""

"""
    linear_flow(u₀, v₀, w₀, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y) -> velocity_function

Create a linear velocity field function.
"""
function linear_flow(u₀, v₀, w₀, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y)
    return (x, y, z, t) -> (
        u₀ + ∂u∂x * x + ∂u∂y * y,
        v₀ + ∂v∂x * x + ∂v∂y * y,
        w₀
    )
end

"""
    vortex_flow(center_x, center_y, strength, w₀=0.0) -> velocity_function

Create a circular vortex velocity field function.
"""
function vortex_flow(center_x, center_y, strength, w₀=0.0)
    return (x, y, z, t) -> begin
        dx = x - center_x
        dy = y - center_y
        r = sqrt(dx^2 + dy^2)
        if r > 0
            u = -strength * dy / r
            v = strength * dx / r
        else
            u = v = 0.0
        end
        return (u, v, w₀)
    end
end

"""
    shear_flow(shear_rate, w₀=0.0) -> velocity_function

Create a simple shear velocity field function.
"""
function shear_flow(shear_rate, w₀=0.0)
    return (x, y, z, t) -> (shear_rate * y, 0.0, w₀)
end

"""
    laser_intensity(laser::LaserSheet, z) -> intensity_multiplier

Calculate the laser intensity multiplier at z-position.
"""
function laser_intensity(laser::GaussianLaserSheet, z::Real)
    σ = laser.thickness / (2 * sqrt(2 * log(2)))  # Convert FWHM to standard deviation
    intensity = laser.peak_intensity * exp(-(z - laser.z_center)^2 / (2 * σ^2))
    return intensity
end

# ============================================================================
# Particle Field Generation
# ============================================================================

"""
    generate_particle_field(image_size, particle_density; kwargs...) -> ParticleField3D

Generate a random 3D particle field for synthetic PIV images.

# Arguments
- `image_size::Tuple{Int,Int}`: Size of the image (height, width)
- `particle_density::Real`: Average particles per pixel

# Keywords
- `z_range::Tuple{Real,Real}=(-5.0, 5.0)`: Range of z-positions (laser sheet typically at z=0)
- `diameter_mean::Real=3.0`: Mean particle diameter in pixels
- `diameter_std::Real=0.5`: Standard deviation of particle diameter
- `intensity_mean::Real=1.0`: Mean particle intensity
- `intensity_std::Real=0.2`: Standard deviation of particle intensity
- `rng::AbstractRNG=Random.GLOBAL_RNG`: Random number generator
"""
function generate_particle_field(
    image_size::Tuple{Int,Int}, 
    particle_density::Real;
    z_range::Tuple{Real,Real} = (-5.0, 5.0),
    diameter_mean::Real = 3.0,
    diameter_std::Real = 0.5,
    intensity_mean::Real = 1.0,
    intensity_std::Real = 0.2,
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    height, width = image_size
    n_particles = round(Int, particle_density * height * width)
    
    # Generate random positions
    x = rand(rng, n_particles) * width
    y = rand(rng, n_particles) * height
    z = z_range[1] .+ rand(rng, n_particles) * (z_range[2] - z_range[1])
    
    # Generate particle properties (normal distributions, clipped to positive)
    diameter_dist = Normal(diameter_mean, diameter_std)
    intensity_dist = Normal(intensity_mean, intensity_std)
    
    diameter = max.(rand(rng, diameter_dist, n_particles), 0.5)  # Min diameter 0.5
    intensity = max.(rand(rng, intensity_dist, n_particles), 0.1)  # Min intensity 0.1
    
    return ParticleField3D(x, y, z, intensity, diameter)
end

"""
    displace_particles!(particles::ParticleField3D, velocity_function, dt::Real, t::Real=0.0)

Displace particles according to the velocity function for time step dt.
Modifies particles in-place.

# Arguments
- `particles`: ParticleField3D to modify
- `velocity_function`: Function with signature (x, y, z, t) -> (u, v, w)
- `dt`: Time step
- `t`: Current time (default 0.0)
"""
function displace_particles!(particles::ParticleField3D, velocity_function, dt::Real, t::Real=0.0)
    for i in eachindex(particles.x)
        u, v, w = velocity_function(particles.x[i], particles.y[i], particles.z[i], t)
        particles.x[i] += u * dt
        particles.y[i] += v * dt
        particles.z[i] += w * dt
    end
    return particles
end

"""
    displace_particles(particles::ParticleField3D, velocity_function, dt::Real, t::Real=0.0) -> ParticleField3D

Create a new ParticleField3D with particles displaced according to the velocity function.

# Arguments
- `particles`: ParticleField3D to displace
- `velocity_function`: Function with signature (x, y, z, t) -> (u, v, w)
- `dt`: Time step
- `t`: Current time (default 0.0)
"""
function displace_particles(particles::ParticleField3D, velocity_function, dt::Real, t::Real=0.0)
    new_x = copy(particles.x)
    new_y = copy(particles.y)
    new_z = copy(particles.z)
    
    for i in eachindex(new_x)
        u, v, w = velocity_function(particles.x[i], particles.y[i], particles.z[i], t)
        new_x[i] += u * dt
        new_y[i] += v * dt
        new_z[i] += w * dt
    end
    
    return ParticleField3D(new_x, new_y, new_z, particles.intensity, particles.diameter)
end

# ============================================================================
# Image Rendering
# ============================================================================

"""
    generate_gaussian_particle!(image, center, diameter, intensity=1.0)

Add a Gaussian particle to an image at the specified center position.
"""
function generate_gaussian_particle!(
    image::AbstractMatrix{T}, 
    center::Tuple{Real,Real}, 
    diameter::Real, 
    intensity::Real = 1.0
) where T<:Real
    cx, cy = center
    σ = diameter / 4.0  # Standard deviation from diameter
    
    # Determine bounding box (3σ radius should capture >99% of intensity)
    radius = ceil(Int, 3 * σ)
    i_min = max(1, round(Int, cx - radius))  # i maps to x-coordinate
    i_max = min(size(image, 1), round(Int, cx + radius))
    j_min = max(1, round(Int, cy - radius))  # j maps to y-coordinate  
    j_max = min(size(image, 2), round(Int, cy + radius))
    
    # Add Gaussian intensity
    for i in i_min:i_max, j in j_min:j_max
        dx = i - cx  # i maps to x-coordinate
        dy = j - cy  # j maps to y-coordinate
        r² = dx^2 + dy^2
        value = intensity * exp(-r² / (2 * σ^2))
        image[i, j] += T(value)
    end
    
    return image
end

"""
    render_particle_image(particles::ParticleField3D, image_size, laser::LaserSheet; kwargs...) -> Matrix

Render a 3D ParticleField into an image with laser sheet intensity modulation.

# Keywords
- `background_noise::Real=0.01`: Background noise level
- `intensity_threshold::Real=0.01`: Minimum laser intensity for particle visibility
- `rng::AbstractRNG=Random.GLOBAL_RNG`: Random number generator
"""
function render_particle_image(
    particles::ParticleField3D, 
    image_size::Tuple{Int,Int},
    laser::LaserSheet;
    background_noise::Real = 0.01,
    intensity_threshold::Real = 0.01,
    rng::AbstractRNG = Random.GLOBAL_RNG
)
    height, width = image_size
    image = zeros(Float64, height, width)
    
    # Render each particle with laser intensity modulation
    for i in eachindex(particles.x)
        # Skip particles outside image bounds
        if 1 <= particles.x[i] <= width && 1 <= particles.y[i] <= height
            # Calculate laser intensity at particle z-position
            laser_mult = laser_intensity(laser, particles.z[i])
            
            # Only render if laser intensity is above threshold
            if laser_mult > intensity_threshold
                effective_intensity = particles.intensity[i] * laser_mult
                generate_gaussian_particle!(
                    image, 
                    (particles.x[i], particles.y[i]), 
                    particles.diameter[i], 
                    effective_intensity
                )
            end
        end
    end
    
    # Add background noise
    if background_noise > 0
        image .+= background_noise * randn(rng, height, width)
        image = max.(image, 0.0)  # Ensure non-negative
    end
    
    return image
end

# ============================================================================
# High-Level Interface
# ============================================================================

"""
    generate_synthetic_piv_pair(velocity_function, image_size, dt; kwargs...) -> (img1, img2, particles1, particles2)

Generate a synthetic PIV image pair with known ground truth displacement.
Includes realistic effects of out-of-plane motion and laser sheet illumination.

# Arguments
- `velocity_function`: Function with signature (x, y, z, t) -> (u, v, w)
- `image_size::Tuple{Int,Int}`: Size of images (height, width)  
- `dt::Real`: Time step between images

# Keywords
- `particle_density::Real=0.05`: Particles per pixel
- `laser_sheet::LaserSheet=GaussianLaserSheet(0.0, 2.0, 1.0)`: Laser sheet profile
- `background_noise::Real=0.01`: Background noise level
- `z_range::Tuple{Real,Real}=(-5.0, 5.0)`: Range of particle z-positions
- `t0::Real=0.0`: Initial time
- `rng::AbstractRNG=Random.GLOBAL_RNG`: Random number generator
- Additional keywords passed to `generate_particle_field`

# Returns
- `img1, img2`: The two synthetic PIV images
- `particles1, particles2`: The 3D particle fields for ground truth analysis
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
    rng::AbstractRNG = Random.GLOBAL_RNG,
    kwargs...
)
    # Generate initial 3D particle field
    particles1 = generate_particle_field(image_size, particle_density; z_range=z_range, rng=rng, kwargs...)
    
    # Displace particles according to velocity function
    particles2 = displace_particles(particles1, velocity_function, dt, t0)
    
    # Render both images with laser sheet effects
    img1 = render_particle_image(particles1, image_size, laser_sheet; background_noise=background_noise, rng=rng)
    img2 = render_particle_image(particles2, image_size, laser_sheet; background_noise=background_noise, rng=rng)
    
    return (img1, img2, particles1, particles2)
end

end # module SyntheticData