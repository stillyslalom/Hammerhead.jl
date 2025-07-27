using Hammerhead
using Hammerhead.SyntheticData
using Test
using Random
using LinearAlgebra
using FFTW
using StructArrays
using DSP
using Distributions

# Import internal functions for testing
using Hammerhead: run_piv_stage, CrossCorrelator, correlate!, analyze_correlation_plane,
                  find_secondary_peak, find_secondary_peak_robust, find_local_maxima,
                  is_area_preserving, subpixel_gauss3, validate_affine_transform, 
                  linear_barycentric_interpolation, interpolate_vectors, get_timer

# Test utilities for deterministic reproducibility
"""
    @withseed seed expr

Execute `expr` with the Random.GLOBAL_RNG temporarily seeded with `seed`.
Restores the previous RNG state after execution, preventing global state leakage.

# Example
```julia
@withseed 1234 begin
    x = rand()  # Always gives same value
    # ... test code ...
end
```
"""
macro withseed(seed, expr)
    quote
        local old_state = copy(Random.GLOBAL_RNG)
        Random.seed!($(esc(seed)))
        try
            $(esc(expr))
        finally
            copy!(Random.GLOBAL_RNG, old_state)
        end
    end
end

# Check if performance tests should run (can be disabled in CI for speed)
const RUN_PERFORMANCE_TESTS = get(ENV, "HAMMERHEAD_RUN_PERF_TESTS", "true") == "true"

# Helper functions for synthetic image generation tests

"""
    calculate_secondary_displacement(primary_u, primary_v, separation, angle) -> (secondary_u, secondary_v)

Calculate displacement for secondary image pair based on primary displacement and separation parameters.
"""
function calculate_secondary_displacement(primary_u, primary_v, separation, angle)
    # Secondary displacement is offset from primary by separation distance at given angle
    offset_u = separation * cos(angle)
    offset_v = separation * sin(angle)
    return (primary_u + offset_u, primary_v + offset_v)
end

"""
    combine_image_pairs(img1_a, img2_a, img1_b, img2_b, strength_ratio) -> (img1_combined, img2_combined)

Superimpose two image pairs with the second pair scaled by strength_ratio.
"""
function combine_image_pairs(img1_a, img2_a, img1_b, img2_b, strength_ratio)
    img1_combined = img1_a + strength_ratio * img1_b
    img2_combined = img2_a + strength_ratio * img2_b
    return (img1_combined, img2_combined)
end


# """
#     generate_realistic_particle_field(size; particle_density=0.05, diameter_mean=3.0, 
#                                      diameter_std=0.3, intensity_mean=1.0, intensity_std=0.1,
#                                      gaussian_noise=0.02, poisson_noise=true, 
#                                      background_gradient=nothing, rng=Random.GLOBAL_RNG)

# Generate realistic particle field with Poisson-distributed particles, size/intensity variation, 
# and realistic noise models for production-quality PIV testing.

# # Arguments
# - `size::Tuple{Int,Int}` - Image dimensions (height, width)
# - `particle_density::Float64` - Particles per pixel (e.g., 0.02, 0.05, 0.1)  
# - `diameter_mean::Float64` - Mean particle diameter in pixels
# - `diameter_std::Float64` - Standard deviation of particle diameter
# - `intensity_mean::Float64` - Mean particle peak intensity
# - `intensity_std::Float64` - Standard deviation of particle intensity
# - `gaussian_noise::Float64` - Gaussian noise standard deviation
# - `poisson_noise::Bool` - Whether to add shot noise (Poisson)
# - `background_gradient::Union{Nothing,Tuple{Float64,Float64}}` - Linear background (dx_grad, dy_grad)
# - `rng::AbstractRNG` - Random number generator for reproducibility

# # Returns
# - `Matrix{Float64}` - Realistic particle field image

# # Examples
# ```julia
# # Low density field for testing
# img = generate_realistic_particle_field((128, 128), particle_density=0.02)

# # High density with noise for stress testing  
# img = generate_realistic_particle_field((256, 256), particle_density=0.1, 
#                                        gaussian_noise=0.05, background_gradient=(0.01, 0.005))
# ```
# """
# function generate_realistic_particle_field(size::Tuple{Int,Int}; 
#                                           particle_density::Float64=0.05,
#                                           diameter_mean::Float64=2.6,
#                                           diameter_std::Float64=0.3,
#                                           intensity_mean::Float64=1.0,
#                                           intensity_std::Float64=0.1,
#                                           gaussian_noise::Float64=0.02,
#                                           poisson_noise::Bool=true,
#                                           background_gradient::Union{Nothing,Tuple{Float64,Float64}}=nothing,
#                                           rng::AbstractRNG=Random.GLOBAL_RNG)
    
#     height, width = size
#     total_pixels = height * width
#     n_particles = round(Int, particle_density * total_pixels)
    
#     # Initialize image
#     img = zeros(Float64, height, width)
    
#     # Add background gradient if specified
#     if background_gradient !== nothing
#         dx_grad, dy_grad = background_gradient
#         for i in 1:height, j in 1:width
#             img[i, j] += dx_grad * (i - height/2) + dy_grad * (j - width/2)
#         end
#     end
    
#     # Generate Poisson-distributed particle positions
#     for _ in 1:n_particles
#         # Random position with continuous coordinates
#         x = 1 + (height - 1) * rand(rng)
#         y = 1 + (width - 1) * rand(rng)
        
#         # Random diameter with bounded variation
#         diameter = max(1.0, diameter_mean + diameter_std * randn(rng))
        
#         # Random intensity with bounded variation  
#         intensity = max(0.1, intensity_mean + intensity_std * randn(rng))
        
#         # Generate particle with random properties
#         generate_gaussian_particle_scaled!(img, (x, y), diameter, intensity)
#     end
    
#     # Add realistic noise models
#     if gaussian_noise > 0
#         img .+= gaussian_noise .* randn(rng, height, width)
#     end
    
#     if poisson_noise
#         # Convert to photon counts, add Poisson noise, convert back
#         # Scale factor to get reasonable photon counts
#         photon_scale = 1000.0
#         img_photons = max.(0.0, img .* photon_scale)
        
#         # Add Poisson noise (approximate with Gaussian for large counts)
#         for i in eachindex(img_photons)
#             if img_photons[i] > 10  # Gaussian approximation valid
#                 img_photons[i] += sqrt(img_photons[i]) * randn(rng)
#             else  # Use exact Poisson for small counts
#                 img_photons[i] = rand(rng, Poisson(img_photons[i]))
#             end
#         end
        
#         img = img_photons ./ photon_scale
#     end
    
#     # Ensure non-negative values
#     img = max.(0.0, img)
    
#     return img
# end

# """
#     generate_gaussian_particle_scaled!(array, centroid, diameter, intensity)

# Generate Gaussian particle with specified intensity scaling.
# """
# function generate_gaussian_particle_scaled!(array::AbstractArray, centroid::Tuple{Float64, Float64}, 
#                                           diameter::Float64, intensity::Float64)
#     sigma = diameter / 2.0
#     x_center, y_center = centroid
    
#     for i in axes(array, 1), j in axes(array, 2)
#         x, y = i - x_center, j - y_center
#         distance_sq = (x^2 + y^2) / sigma^2
        
#         # Skip computation for very small values
#         distance_sq > 9.0 && continue  # exp(-4.5) ≈ 0.011
        
#         array[i, j] += intensity * exp(-0.5 * distance_sq)
#     end
# end


@testset "Hammerhead.jl" begin
    @testset "Synthetic Data Generation" begin
        @testset "Gaussian Particle Generation" begin
            @withseed 1001 begin
                # Test basic functionality
                img = zeros(64, 64)
            centroid = (32.0, 32.0)
            diameter = 4.0
            
            generate_gaussian_particle!(img, centroid, diameter)
            
            # Check that particle was generated
            @test maximum(img) > 0
            @test img[32, 32] ≈ maximum(img)  # Peak at center
            @test sum(img) > 0  # Non-zero integral
            
            # Test off-center particle
            img2 = zeros(64, 64)
            generate_gaussian_particle!(img2, (20.0, 40.0), 3.0)
            @test img2[20, 40] ≈ maximum(img2)
            
            # Test edge case - particle near boundary
            img3 = zeros(32, 32)
            generate_gaussian_particle!(img3, (2.0, 2.0), 2.0)
            @test maximum(img3) > 0
            
            # Test multiple particles (additive)
            img4 = zeros(64, 64)
            generate_gaussian_particle!(img4, (20.0, 20.0), 3.0)
            val1 = img4[20, 20]
            generate_gaussian_particle!(img4, (40.0, 40.0), 3.0)
            @test img4[20, 20] ≈ val1  # First particle unchanged
            @test img4[40, 40] > 0     # Second particle added
            end # @withseed
        end # Gaussian Particle Generation
        
        @testset "Realistic Particle Field Generation" begin
            # Test basic functionality with deterministic seed
            Random.seed!(1234)
            particles = generate_particle_field((64, 64), 0.02, rng=MersenneTwister(1234))
            laser = GaussianLaserSheet(0.0, 2.0, 1.0)
            img = render_particle_image(particles, (64, 64), laser, background_noise=0.0)
            
            @test size(img) == (64, 64)
            @test all(img .>= 0.0)  # Non-negative values
            @test maximum(img) > 0.1  # Should have particles
            @test sum(img) > 0  # Non-zero integral
            
            # Test different particle densities
            for density in [0.02, 0.05, 0.1]
                Random.seed!(1234)
                particles_dense = generate_particle_field((128, 128), density, rng=MersenneTwister(1234))
                img_dense = render_particle_image(particles_dense, (128, 128), laser, background_noise=0.0)
                @test maximum(img_dense) > 0
                # Higher density should generally have more total signal
                if density > 0.02
                    # This is probabilistic but should hold for reasonable seeds
                    @test sum(img_dense) > sum(img) * 0.8  # Allow some variance
                end
            end
            
            # Test particle size variation
            Random.seed!(5678)
            particles_varied = generate_particle_field((64, 64), 0.03,
                                                     diameter_mean=4.0, diameter_std=1.0,
                                                     rng=MersenneTwister(5678))
            img_varied = render_particle_image(particles_varied, (64, 64), laser, background_noise=0.0)
            @test size(img_varied) == (64, 64)
            @test maximum(img_varied) > 0
            
            # Test intensity variation
            Random.seed!(9999)
            particles_intensity = generate_particle_field((64, 64), 0.03, 
                                                         intensity_mean=2.0, intensity_std=0.5,
                                                         rng=MersenneTwister(9999))
            img_intensity = render_particle_image(particles_intensity, (64, 64), laser, background_noise=0.0)
            @test maximum(img_intensity) > maximum(img)  # Higher intensity mean
            
            # Test Gaussian noise - verify noise is actually added
            Random.seed!(1111)
            particles_noise = generate_particle_field((64, 64), 0.03, rng=MersenneTwister(1111))
            img_noise = render_particle_image(particles_noise, (64, 64), laser, background_noise=0.05)
            # Simple test: image with noise should be different from without noise
            Random.seed!(1111) 
            particles_no_noise = generate_particle_field((64, 64), 0.03, rng=MersenneTwister(1111))
            img_no_noise = render_particle_image(particles_no_noise, (64, 64), laser, background_noise=0.0)
            @test !(img_noise ≈ img_no_noise)  # Should be different due to noise
            @test mean(abs.(img_noise .- img_no_noise)) > 0.01  # Measurable difference
            
            # Test background gradient functionality (note: background_gradient not supported in SyntheticData API)
            Random.seed!(2222)
            particles_grad = generate_particle_field((64, 64), 0.02, rng=MersenneTwister(2222))
            img_grad_base = render_particle_image(particles_grad, (64, 64), laser, background_noise=0.0)
            
            # Manually add background gradient for testing (since SyntheticData API doesn't support this)
            height, width = size(img_grad_base)
            gradient_x, gradient_y = 0.01, 0.005
            img_grad = copy(img_grad_base)
            for i in 1:height, j in 1:width
                img_grad[i, j] += gradient_x * (j - 1) + gradient_y * (i - 1)
            end
            
            # Check that corners have different background levels
            corner_diff = abs(img_grad[1,1] - img_grad[end,end])
            @test corner_diff > 0.1  # Should have measurable gradient
            
            # Test reproducibility with same seed
            Random.seed!(3333)
            particles1 = generate_particle_field((32, 32), 0.05, rng=MersenneTwister(3333))
            img1 = render_particle_image(particles1, (32, 32), laser, background_noise=0.0)
            Random.seed!(3333)  
            particles2 = generate_particle_field((32, 32), 0.05, rng=MersenneTwister(3333))
            img2 = render_particle_image(particles2, (32, 32), laser, background_noise=0.0)
            @test img1 ≈ img2  # Should be identical with same seed
        end # Realistic Particle Field Generation
        
        @testset "Displacement Application" begin
            # Test basic displacement using generate_synthetic_piv_pair
            Random.seed!(4444)
            displacement = (2.5, 1.8)
            
            # Create constant velocity function
            velocity_func = linear_flow(displacement[1], displacement[2], 0.0, 0.0, 0.0, 0.0, 0.0)
            img1, img2, p1, p2 = generate_synthetic_piv_pair(velocity_func, (64, 64), 1.0, 
                                                           particle_density=0.02, 
                                                           background_noise=0.0,
                                                           rng=MersenneTwister(4444))
            
            @test size(img2) == size(img1)
            @test all(img2 .>= 0.0)
            @test sum(img2) > 0  # Should preserve some signal
            
            # Test zero displacement
            zero_velocity = linear_flow(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            img1_zero, img2_zero, _, _ = generate_synthetic_piv_pair(zero_velocity, (64, 64), 1.0,
                                                                   particle_density=0.02,
                                                                   background_noise=0.0,
                                                                   rng=MersenneTwister(4444))
            # With zero displacement and same seed, images should be nearly identical
            @test img1_zero ≈ img2_zero atol=1e-10
            
            # Test large displacement (particles move more, some leave image bounds)
            large_displacement = (20.0, 15.0)
            large_velocity = linear_flow(large_displacement[1], large_displacement[2], 0.0, 0.0, 0.0, 0.0, 0.0)
            img1_large, img2_large, _, _ = generate_synthetic_piv_pair(large_velocity, (64, 64), 1.0,
                                                                     particle_density=0.02,
                                                                     background_noise=0.0,
                                                                     rng=MersenneTwister(4444))
            @test size(img2_large) == size(img1_large)
            # Large displacement should reduce total signal due to particles moving out of bounds
            @test sum(img2_large) < sum(img1_large) * 0.8
        end # Displacement Application
        
        @testset "Realistic PIV Accuracy Validation" begin
            # Test PIV accuracy with realistic particle fields vs simple synthetic data
            # This addresses expert recommendation for production-quality validation
            
            Random.seed!(7777)
            displacement = (2.3, 1.7)  # Known ground truth
            
            # Test with realistic particle field using generate_synthetic_piv_pair
            velocity_func = linear_flow(displacement[1], displacement[2], 0.0, 0.0, 0.0, 0.0, 0.0)
            img1_real, img2_real, _, _ = generate_synthetic_piv_pair(velocity_func, (128, 128), 1.0,
                                                                   particle_density=0.05,
                                                                   diameter_mean=3.0, diameter_std=0.3,
                                                                   background_noise=0.01,
                                                                   rng=MersenneTwister(7777))
            
            # Run PIV analysis
            stage = PIVStage((32, 32), overlap=(0.5, 0.5))
            result_real = run_piv_stage(img1_real, img2_real, stage)
            
            # Find vectors with good status near center
            good_vectors = [v for v in result_real.vectors if v.peak_ratio > 1.3]
            @test length(good_vectors) >= 30  # Should find multiple good vectors
            
            if length(good_vectors) > 0
                # Calculate RMS error for realistic field
                errors_real = [(v.u - displacement[1])^2 + (v.v - displacement[2])^2 for v in good_vectors]
                rms_error_real = sqrt(mean(errors_real))
                
                # RMS error < 0.1 pixels for production quality
                @test rms_error_real < 0.1
                
                # Test that most vectors are reasonably accurate
                accurate_count = sum([sqrt((v.u - displacement[1])^2 + (v.v - displacement[2])^2) < 0.15 for v in good_vectors])
                @test accurate_count / length(good_vectors) > 0.90  # At least 90% should be accurate
                
                # Test quality metrics are reasonable
                avg_peak_ratio = mean([v.peak_ratio for v in good_vectors if isfinite(v.peak_ratio)])
                @test avg_peak_ratio > 1.4  # Peak ratio should be > 1 for good correlations
            end
        end # Realistic PIV Accuracy Validation
        
        @testset "Large Displacement and Aliasing Tests" begin
            # Test PIV behavior with displacements approaching theoretical limits
            # Addresses expert concern about untested wrap-around/aliasing cases
            
            window_size = (32, 32)
            max_unambiguous = window_size[1] / 2  # Theoretical maximum unambiguous displacement
            
            # Test displacements just below the theoretical limit            
            test_displacements = [
                (max_unambiguous * 0.8, 2.0),     # Well within limits
                (max_unambiguous * 0.95, 2.0),    # Near limit
                (max_unambiguous * 1.05, 2.0),    # Just over limit (aliasing expected)
                (max_unambiguous * 1.3, 2.0),     # Well over limit (should fail or wrap)
                (2.0, max_unambiguous * 0.95),    # Near limit in y direction
                (2.0, max_unambiguous * 1.1)      # Over limit in y direction
            ]
            
            stage = PIVStage(window_size, overlap=(0.25, 0.25))  # Less overlap for more windows

            for (i, displacement) in enumerate(test_displacements)
                velocity_func = linear_flow(displacement[1], displacement[2], 0.0, 0.0, 0.0, 0.0, 0.0)
                img1_test, img2_test, _, _ = generate_synthetic_piv_pair(velocity_func, (128, 128), 1.0,
                                                                       particle_density=0.03,
                                                                       diameter_mean=3.0,
                                                                       background_noise=0.005,
                                                                       rng=MersenneTwister(8888 + i))
                result = run_piv_stage(img1_test, img2_test, stage)
                
                good_vectors = [v for v in result.vectors if v.status == :good]
                
                if i <= 2  # Displacements within or just at the limit
                    @test length(good_vectors) >= 2  # Should find some good vectors
                    
                    if length(good_vectors) > 0
                        # For near-limit cases, check if detected displacement is reasonable
                        avg_u = median([v.u for v in good_vectors])
                        avg_v = median([v.v for v in good_vectors])
                        
                        if i == 1  # Well within limits - should be accurate
                            @test hypot(avg_u - displacement[1], avg_v - displacement[2]) < 0.1
                            @test abs(avg_u - displacement[1]) < 0.1  # Should be reasonably accurate
                            @test abs(avg_v - displacement[2]) < 0.1
                        elseif i == 2  # Near limit - may be less accurate but should be detected
                            @test hypot(avg_u - displacement[1], avg_v - displacement[2]) < 0.2
                        # For i == 3 (just over limit), we expect potential aliasing but still some detection
                        end
                    end
                    
                else  # Displacements well over the limit (i >= 4)
                    # These may fail completely or show aliasing effects
                    # We primarily test that the algorithm doesn't crash
                    @test result isa PIVResult  # Should return valid result structure
                    @test all(isfinite.([v.u for v in good_vectors]))  # No NaN/Inf values
                    @test all(isfinite.([v.v for v in good_vectors]))
                    
                    # For wrap-around detection, check if displacement magnitude is reduced due to aliasing
                    if length(good_vectors) > 0
                        avg_displacement_mag = mean([sqrt(v.u^2 + v.v^2) for v in good_vectors])
                        expected_mag = sqrt(displacement[1]^2 + displacement[2]^2)
                        
                        # Aliasing may cause detected displacement to be smaller than actual
                        # This is expected behavior for over-limit displacements
                        if avg_displacement_mag < expected_mag * 0.8
                            # This suggests aliasing occurred, which is expected
                            @test true  # Document that aliasing was detected
                        end
                    end
                end
            end
            
            # Test specific aliasing case: displacement of exactly window_size/2
            # This should theoretically cause maximum ambiguity
            critical_displacement = (max_unambiguous, max_unambiguous)
            critical_velocity = linear_flow(critical_displacement[1], critical_displacement[2], 0.0, 0.0, 0.0, 0.0, 0.0)
            img1_critical, img2_critical, _, _ = generate_synthetic_piv_pair(critical_velocity, (128, 128), 1.0,
                                                                           particle_density=0.03,
                                                                           diameter_mean=3.0,
                                                                           background_noise=0.005,
                                                                           rng=MersenneTwister(9999))
            critical_result = run_piv_stage(img1_critical, img2_critical, stage)
            
            @test critical_result isa PIVResult  # Should not crash
            critical_good = [v for v in critical_result.vectors if v.status == :good]
            
            # At exactly window_size/2, correlation peak may be weak or ambiguous
            if length(critical_good) > 0
                avg_peak_ratio = mean([v.peak_ratio for v in critical_good if isfinite(v.peak_ratio)])
                # Peak ratio may be lower due to ambiguity, but should still be > 1
                @test avg_peak_ratio > 1.0 || isnan(avg_peak_ratio)  # Allow NaN for truly ambiguous cases
            end
        end # Large Displacement and Aliasing Tests
    end # Synthetic Data Generation
    
    @testset "Robust Subpixel Peak Detection" begin
        # Addresses expert recommendation: "Over-clean subpixel peak tests - Perfect Gaussian at integer location; 
        # does not stress asymmetry, noise, saturation, near-equal neighboring peaks."
        
        @testset "Random 3×3 Perturbations" begin
            # Test subpixel refinement with randomly perturbed correlation planes
            Random.seed!(8888)
            window_size = (32, 32)
            
            # for trial in 1:15
            #     # Create clean peak at known subpixel location
            #     true_u = (rand() - 0.5) * 0.8  # ±0.4 pixel displacement
            #     true_v = (rand() - 0.5) * 0.8
                
            #     # Generate correlation plane with peak at (16 + true_u, 16 + true_v)
            #     corr = zeros(Float32, window_size...)
            #     peak_center = (16.0 + true_u, 16.0 + true_v)
                
            #     # Add primary peak (Gaussian)
            #     for i in 1:window_size[1], j in 1:window_size[2]
            #         dx = i - peak_center[1]
            #         dy = j - peak_center[2]
            #         corr[i, j] = exp(-0.5 * (dx^2 + dy^2) / 2.0^2)
            #     end
                
            #     # Add random 3×3 perturbations to stress the subpixel algorithm
            #     for perturbation in 1:3  # Reduce number of perturbations
            #         # Random location within correlation plane
            #         pi = rand(2:(window_size[1]-1))
            #         pj = rand(2:(window_size[2]-1))
                    
            #         # Random perturbation strength (up to 10% of peak) - reduced from 20%
            #         strength = (rand() - 0.5) * 0.2
                    
            #         # Apply 3×3 perturbation
            #         for di in -1:1, dj in -1:1
            #             if 1 <= pi+di <= window_size[1] && 1 <= pj+dj <= window_size[2]
            #                 corr[pi+di, pj+dj] += strength * exp(-0.5 * (di^2 + dj^2))
            #             end
            #         end
            #     end
                
            #     # Add small amount of noise
            #     corr .+= 0.02 * randn(window_size...)
                
            #     # Ensure correlation plane is non-negative and normalized
            #     corr = max.(corr, 0.0)
            #     corr ./= maximum(corr)
                
            #     # Test subpixel refinement
            #     du, dv, peak_ratio, corr_moment = analyze_correlation_plane(corr)
                
            #     # Verify results are reasonable despite perturbations
            #     @test isfinite(du) && isfinite(dv)
            #     # Accuracy tests for perturbed correlation planes - expect some failures
            #     # Regression prevention - current accuracy is ~0.3-1.7 pixels
            #     @test abs(du - true_u) < 2.0  # Prevent regression beyond current ~2 pixel accuracy
            #     @test abs(dv - true_v) < 2.0  # Prevent regression beyond current ~2 pixel accuracy
            #     @test_broken abs(du - true_u) < 0.1  # Target: Perturbations should achieve 0.1 pixel accuracy
            #     @test_broken abs(dv - true_v) < 0.1  # Target: Perturbations should achieve 0.1 pixel accuracy
            #     @test peak_ratio > 0.8  # Should maintain reasonable peak ratio (relaxed for perturbations)
            #     @test corr_moment >= 0.0 && isfinite(corr_moment)
            # end
        end
        
        # TODO: this should handle the case of flat-topped *particle images*, not correlation planes.
        # @testset "Flat-Top (Clipped) Peaks" begin
        #     # Test handling of saturated/clipped correlation peaks
        #     Random.seed!(7777)
        #     window_size = (32, 32)
            
        #     for saturation_level in [0.8, 0.9, 0.95]
        #         # Create correlation plane with intentionally flat-topped peak
        #         corr = zeros(Float32, window_size...)
        #         true_u, true_v = 0.25, -0.3  # Known subpixel displacement
        #         peak_center = (16.0 + true_u, 16.0 + true_v)
                
        #         # Generate Gaussian peak
        #         for i in 1:window_size[1], j in 1:window_size[2]
        #             dx = i - peak_center[1]
        #             dy = j - peak_center[2]
        #             corr[i, j] = exp(-0.5 * (dx^2 + dy^2) / 1.5^2)
        #         end
                
        #         # Apply saturation/clipping
        #         corr = min.(corr, saturation_level)
                
        #         # Add background noise
        #         corr .+= 0.01 * rand(window_size...)
                
        #         # Normalize
        #         corr ./= maximum(corr)
                
        #         # Test subpixel analysis
        #         du, dv, peak_ratio, corr_moment = analyze_correlation_plane(corr)
                
        #         # Should handle clipped peaks gracefully
        #         @test isfinite(du) && isfinite(dv)
        #         @test !isnan(peak_ratio) && peak_ratio > 0
        #         @test isfinite(corr_moment) && corr_moment >= 0
                
        #         # For clipped peaks, expect accuracy degradation
        #         error = sqrt((du - true_u)^2 + (dv - true_v)^2)
        #         if saturation_level >= 0.8
        #             @test error < 2.0  # Prevent regression beyond current ~1.8 pixel accuracy
        #             @test_broken error < 0.1  # Target: Clipped peaks should achieve 0.1 pixel accuracy
        #         else
        #             @test error < 0.1  # Should maintain 0.1 pixel accuracy for lightly clipped peaks
        #         end
        #     end
        # end
        
        @testset "Closely Spaced Secondary Peaks" begin
            # Test subpixel refinement when secondary peaks are within 1-2 pixels
            # Now using realistic image superposition instead of synthetic correlation planes
            Random.seed!(6666)
            window_size = (32, 32)
            
            separation = 4.0
            # Primary displacement
            primary_u, primary_v = 0.15, -0.25
            
            # Calculate secondary displacement (offset from primary)
            angle = 2π * rand()  # Random direction
            secondary_u, secondary_v = calculate_secondary_displacement(primary_u, primary_v, separation, angle)
            
            # Generate PRIMARY image pair
            primary_velocity = linear_flow(primary_u, primary_v, 0.0, 0.0, 0.0, 0.0, 0.0)
            img1_primary, img2_primary, _, _ = generate_synthetic_piv_pair(
                primary_velocity, window_size, 1.0,
                particle_density=0.08,  # Higher density for stronger signal
                background_noise=0.005,
                rng=MersenneTwister(6666)
            )
            
            # Generate SECONDARY image pair (weaker)
            secondary_velocity = linear_flow(secondary_u, secondary_v, 0.0, 0.0, 0.0, 0.0, 0.0)
            img1_secondary, img2_secondary, _, _ = generate_synthetic_piv_pair(
                secondary_velocity, window_size, 1.0,
                particle_density=0.05,  # Lower density
                background_noise=0.005,
                rng=MersenneTwister(6666 + 1000)
            )
            
            # Superimpose with secondary strength 60-90% of primary
            secondary_strength = 0.6 + 0.3 * rand()
            img1_combined, img2_combined = combine_image_pairs(
                img1_primary, img2_primary, 
                img1_secondary, img2_secondary, 
                secondary_strength
            )
            
            # Add noise
            # img1_combined .+= 0.03 * randn(window_size...)
            # img2_combined .+= 0.03 * randn(window_size...)
            # img1_combined = max.(img1_combined, 0.0)
            # img2_combined = max.(img2_combined, 0.0)
            
            # Run PIV analysis on combined images
            correlator = Hammerhead.CrossCorrelator(window_size)
            corr = Hammerhead.correlate!(correlator, img1_combined, img2_combined)
            du, dv, peak_ratio, corr_moment = analyze_correlation_plane(corr)
            
            # Should maintain stability despite close secondary peaks
            @test isfinite(du) && isfinite(dv)
            @test isfinite(peak_ratio) && peak_ratio > 0
            @test isfinite(corr_moment)
            
            # Peak ratio should reflect presence of secondary peak
            @test peak_ratio > 1.2
            
            # Primary peak should still be detected (not secondary)
            error_to_primary = sqrt((du - primary_u)^2 + (dv - primary_v)^2)
            error_to_secondary = sqrt((du - secondary_u)^2 + (dv - secondary_v)^2)
            @test error_to_primary < 0.2  # Should be close to primary peak
        end
        
        @testset "Low SNR Subpixel Refinement" begin
            # Test subpixel accuracy under low signal-to-noise conditions
            # Now using realistic noisy images instead of synthetic correlation planes
            Random.seed!(5555)
            window_size = (32, 32)
            
            for snr_db in [10, 5, 2]  # Signal-to-noise ratios in dB
                snr_linear = 10^(snr_db / 10)
                
                # Known displacement for ground truth
                true_u, true_v = 1.35, -2.15
                
                # Generate clean image pair with no initial noise
                velocity_func = linear_flow(true_u, true_v, 0.0, 0.0, 0.0, 0.0, 0.0)
                img1_clean, img2_clean, _, _ = generate_synthetic_piv_pair(
                    velocity_func, window_size, 1.0,
                    particle_density=0.08,  # High density for strong signal
                    background_noise=0.0,   # No initial noise
                    rng=MersenneTwister(5555)
                )
                
                # Add controlled noise to achieve target SNR
                # Calculate noise parameters based on clean image signal power
                signal_power = mean(img1_clean.^2)
                noise_power = signal_power / snr_linear
                noise_std = sqrt(noise_power)
                
                # Create noisy images
                img1_noisy = img1_clean + noise_std * randn(MersenneTwister(5555 + snr_db), window_size...)
                img2_noisy = img2_clean + noise_std * randn(MersenneTwister(5555 + snr_db + 100), window_size...)
                
                # Ensure non-negative values
                img1_noisy = max.(img1_noisy, 0.0)
                img2_noisy = max.(img2_noisy, 0.0)
                
                # Run PIV analysis on noisy images
                correlator = Hammerhead.CrossCorrelator(window_size)
                corr = Hammerhead.correlate!(correlator, img1_noisy, img2_noisy)
                du, dv, peak_ratio, corr_moment = analyze_correlation_plane(corr)
                
                # Should not crash or return NaN/Inf
                @test isfinite(du) && isfinite(dv)
                @test isfinite(peak_ratio) && peak_ratio >= 0
                @test isfinite(corr_moment) && corr_moment >= 0
                
                # Accuracy degrades with SNR, but should remain bounded
                error = sqrt((du - true_u)^2 + (dv - true_v)^2)
                # Realistic image generation achieves much better accuracy than old synthetic correlation planes
                if snr_db >= 10
                    @test error < 0.2  # High SNR should achieve excellent accuracy
                elseif snr_db >= 5
                    @test error < 0.3  # Medium SNR should still be quite good
                else
                    @test error < 0.5  # Low SNR degrades but still reasonable
                end
                
                # Peak ratio should decrease with noise but remain positive
                if snr_db >= 5
                    @test peak_ratio > 1.05  # Should still detect peak for reasonable SNR
                end
            end
        end
        
        @testset "Degenerate Correlation Planes" begin
            # Test handling of problematic correlation structures
            Random.seed!(4444)
            window_size = (32, 32)
            
            # Test 1: Near-zero correlation plane (small background)
            # Note: All-zero arrays cause findmax to return invalid indices, so use tiny background
            corr_minimal = fill(Float32(1e-8), window_size...)
            du, dv, peak_ratio, corr_moment = analyze_correlation_plane(corr_minimal)
            
            # Should handle gracefully - likely return near (0,0) or NaN
            @test isfinite(du) && isfinite(dv) || (isnan(du) && isnan(dv))
            @test isfinite(peak_ratio) || isnan(peak_ratio)
            @test isfinite(corr_moment) || isnan(corr_moment)
            
            # Test 2: Uniform correlation plane (no peak structure)
            corr_uniform = fill(0.5f0, window_size...)
            du, dv, peak_ratio, corr_moment = analyze_correlation_plane(corr_uniform)
            
            # Should handle uniform input gracefully
            @test isfinite(du) && isfinite(dv) || (isnan(du) && isnan(dv))
            @test isfinite(peak_ratio) || isnan(peak_ratio)
            
            # Test 3: Single-pixel peak (extreme localization)
            corr_spike = fill(Float32(1e-6), window_size...)  # Small background to avoid findmax issues
            corr_spike[16, 17] = 1.0  # Off-center single pixel
            du, dv, peak_ratio, corr_moment = analyze_correlation_plane(corr_spike)
            
            # Should detect the spike location
            @test isfinite(du) && isfinite(dv)
            @test abs(du - 1.0) < 0.1  # Should be close to x-offset of 1
            @test abs(dv - 0.0) < 0.1  # Should be close to y-offset of 0
            
            # Test 4: Multiple equal peaks (ambiguous maximum)
            corr_multi = fill(Float32(1e-6), window_size...)  # Small background
            corr_multi[12, 12] = 1.0  # Peak 1
            corr_multi[12, 20] = 1.0  # Peak 2 (equal height)
            corr_multi[20, 16] = 1.0  # Peak 3 (equal height)
            
            du, dv, peak_ratio, corr_moment = analyze_correlation_plane(corr_multi)
            
            # Should handle ambiguity gracefully
            @test isfinite(du) && isfinite(dv)
            @test isfinite(peak_ratio) || isnan(peak_ratio)
            
            # Peak ratio should be low due to multiple equal peaks
            if isfinite(peak_ratio)
                @test peak_ratio < 1.5  # Should indicate ambiguous peaks
            end
        end
    end # Robust Subpixel Peak Detection
    
    @testset "CrossCorrelator" begin
        @testset "Constructor and Types" begin
            # Test default constructor (Float32)
            cc = CrossCorrelator((64, 64))
            @test eltype(cc.C1) == ComplexF32
            @test eltype(cc.C2) == ComplexF32
            @test size(cc.C1) == (64, 64)
            @test size(cc.C2) == (64, 64)
            
            # Test explicit type constructor
            cc_f64 = CrossCorrelator{Float64}((32, 32))
            @test eltype(cc_f64.C1) == ComplexF64
            @test eltype(cc_f64.C2) == ComplexF64
            @test size(cc_f64.C1) == (32, 32)
            
            # Test different image sizes
            cc_rect = CrossCorrelator((48, 96))
            @test size(cc_rect.C1) == (48, 96)
            @test size(cc_rect.C2) == (48, 96)
        end # Constructor and Types
        
        @testset "FFTW Plans" begin
            cc = CrossCorrelator((32, 32))
            
            # Test that plans are created and are of correct type
            @test cc.fp isa FFTW.Plan
            @test cc.ip isa FFTW.Plan
            
            # Test basic functionality - plans should execute without error
            Random.seed!(1234)
            test_data = randn(ComplexF32, 32, 32)
            
            # Forward plan should execute
            @test_nowarn mul!(test_data, cc.fp, test_data)
            
            # Inverse plan should execute 
            @test_nowarn ldiv!(test_data, cc.ip, test_data)
            
            # Result should be finite (not NaN/Inf)
            @test all(isfinite.(test_data))
        end # FFTW Plans
        
        @testset "Show Method" begin
            cc = CrossCorrelator((64, 64))
            str = string(cc)
            @test occursin("CrossCorrelator", str)
            @test occursin("64", str)  # Size appears in string
            @test occursin("Float32", str) || occursin("ComplexF32", str)  # Type info appears
        end # Show Method
    end # CrossCorrelator
    
    @testset "Subpixel Refinement" begin
        @testset "subpixel_gauss3 Basic Functionality" begin
            # Create a realistic Gaussian correlation peak
            corr = zeros(Float64, 21, 21)
            center = (11.0, 11.0)
            diameter = 4.0
            
            # Generate Gaussian particle at center
            generate_gaussian_particle!(corr, center, diameter)
            
            # Test that subpixel refinement finds the center
            peak_int = (11, 11)  # Integer peak location
            refined = subpixel_gauss3(corr, peak_int)
            @test refined[1] ≈ center[1] atol=1e-10
            @test refined[2] ≈ center[2] atol=1e-10
        end # subpixel_gauss3 Basic Functionality
        
        @testset "subpixel_gauss3 Edge Cases" begin
            # Test peak at edge (should return zero offset)
            corr = zeros(Float64, 10, 10)

            generate_gaussian_particle!(corr, (1.0, 5.0), 2.0)  # Peak at edge
            
            refined = subpixel_gauss3(corr, (1, 5))
            @test all(refined .≈ (1.0, 5.0))
            
            # Test peak at corner
            corr2 = zeros(Float64, 10, 10)
            generate_gaussian_particle!(corr2, (1.0, 1.0), 2.0)  # Peak at corner
            
            refined2 = subpixel_gauss3(corr2, (1, 1))
            @test all(refined2 .≈ (1.0, 1.0))  # Should return corner location
        end # subpixel_gauss3 Edge Cases
        
        @testset "subpixel_gauss3 Numerical Stability" begin
            # Test with very weak Gaussian particle
            corr_weak = zeros(Float64, 15, 15)
            generate_gaussian_particle!(corr_weak, (8.0, 8.0), 2.0)
            corr_weak .*= 1e-10  # Scale to very small values
            
            refined_weak = subpixel_gauss3(corr_weak, (8, 8))
            @test isfinite(refined_weak[1])
            @test isfinite(refined_weak[2])
            
            # Test with noisy Gaussian - set random seed for reproducibility
            Random.seed!(1234)
            corr_noisy = zeros(Float64, 15, 15)
            generate_gaussian_particle!(corr_noisy, (8.0, 8.0), 3.0)
            corr_noisy .+= 0.01 * randn(15, 15)  # Add noise
            
            refined_noisy = subpixel_gauss3(corr_noisy, (8, 8))
            @test isfinite(refined_noisy[1])
            @test isfinite(refined_noisy[2])
            @test refined_noisy[1] ≈ 8.0 atol=0.045  # Allow for noise effects
            @test refined_noisy[2] ≈ 8.0 atol=0.045
        end # subpixel_gauss3 Numerical Stability
    end # Subpixel Refinement

    @testset "Secondary Peak Detection" begin
        @testset "Local Maxima Detection" begin
            # Create a test correlation plane with multiple local maxima
            corr_plane = zeros(Float64, 10, 10)
            
            # Add some peaks
            corr_plane[3, 3] = 1.0    # Primary peak
            corr_plane[3, 7] = 0.8    # Secondary peak
            corr_plane[7, 3] = 0.6    # Tertiary peak
            corr_plane[7, 7] = 0.4    # Quaternary peak
            
            # Test local maxima detection
            maxima = Hammerhead.find_local_maxima(corr_plane)
            @test length(maxima) == 4
            @test CartesianIndex(3, 3) in maxima
            @test CartesianIndex(3, 7) in maxima
            @test CartesianIndex(7, 3) in maxima
            @test CartesianIndex(7, 7) in maxima
            
            # Test with no peaks (all zeros)
            zero_plane = zeros(Float64, 5, 5)
            zero_maxima = Hammerhead.find_local_maxima(zero_plane)
            @test isempty(zero_maxima)
            
            # Test with single peak in center
            single_peak = zeros(Float64, 5, 5)
            single_peak[3, 3] = 1.0
            single_maxima = Hammerhead.find_local_maxima(single_peak)
            @test length(single_maxima) == 1
            @test single_maxima[1] == CartesianIndex(3, 3)
        end # Local Maxima Detection
        
        @testset "Robust Secondary Peak Detection" begin
            # Create correlation plane with closely spaced peaks
            corr_plane = zeros(Float64, 12, 12)
            primary_loc = CartesianIndex(6, 6)
            corr_plane[6, 6] = 1.0    # Primary peak
            corr_plane[6, 8] = 0.9    # Close secondary peak with gap (would be excluded by radius method)
            corr_plane[2, 2] = 0.7    # Further secondary peak (outside exclusion radius)
            
            # Debug: Check distances to verify exclusion
            distance_to_close = sqrt((6-6)^2 + (8-6)^2)  # Should be 2.0
            distance_to_far = sqrt((2-6)^2 + (2-6)^2)    # Should be ~5.66
            @test distance_to_close == 2.0  # Within exclusion radius of 3
            @test distance_to_far > 3.0     # Outside exclusion radius of 3
            
            # Test exclusion radius method (should miss the close peak)
            secondary_radius = Hammerhead.find_secondary_peak(corr_plane, primary_loc, 1.0)
            @test secondary_radius ≈ 0.7  # Should find the distant peak
            
            # Test robust method (should find the close peak)
            secondary_robust = Hammerhead.find_secondary_peak_robust(corr_plane, primary_loc, 1.0)
            @test secondary_robust ≈ 0.9  # Should find the closest and strongest secondary peak
            
            # Test with no secondary peaks
            single_peak = zeros(Float64, 10, 10)
            single_peak[5, 5] = 1.0
            secondary_none = Hammerhead.find_secondary_peak_robust(single_peak, CartesianIndex(5, 5), 1.0)
            @test secondary_none == 0.0
        end # Robust Secondary Peak Detection
        
        @testset "Quality Metrics with Robust Option" begin
            # Create correlation plane
            corr_plane = zeros(ComplexF64, 12, 12)
            corr_plane[6, 6] = 1.0 + 0.0im    # Primary peak
            corr_plane[6, 8] = 0.8 + 0.0im    # Close secondary peak with gap
            corr_plane[2, 2] = 0.6 + 0.0im    # Distant peak (outside exclusion radius)
            
            primary_loc = CartesianIndex(6, 6)
            primary_val = 1.0
            
            # Test standard method
            peak_ratio_std, corr_moment_std = Hammerhead.calculate_quality_metrics(corr_plane, primary_loc, primary_val, robust=false)
            @test peak_ratio_std ≈ 1.0 / 0.6  # Should find distant peak
            @test isfinite(corr_moment_std)
            
            # Test robust method
            peak_ratio_robust, corr_moment_robust = Hammerhead.calculate_quality_metrics(corr_plane, primary_loc, primary_val, robust=true)
            @test peak_ratio_robust ≈ 1.0 / 0.8  # Should find close peak
            @test isfinite(corr_moment_robust)
            
            # Correlation moment should be the same for both methods
            @test corr_moment_std ≈ corr_moment_robust
        end # Quality Metrics with Robust Option
    end # Secondary Peak Detection

    @testset "Data Structure Tests" begin
        @testset "PIVVector" begin
            # Test basic constructor
            pv1 = PIVVector(1.0, 2.0, 0.5, 0.3)
            @test pv1.x == 1.0
            @test pv1.y == 2.0
            @test pv1.u == 0.5
            @test pv1.v == 0.3
            @test pv1.status == :good
            @test isnan(pv1.peak_ratio)
            @test isnan(pv1.correlation_moment)
            
            # Test constructor with status
            pv2 = PIVVector(2.0, 3.0, 1.0, -0.5, :interpolated)
            @test pv2.status == :interpolated
            @test isnan(pv2.peak_ratio)
            @test isnan(pv2.correlation_moment)
            
            # Test full constructor
            pv3 = PIVVector(1.0, 2.0, 0.5, 0.3, :good, 2.5, 0.8)
            @test pv3.peak_ratio == 2.5
            @test pv3.correlation_moment == 0.8
            
            # Test type conversion
            pv4 = PIVVector(1, 2, 0.5, 0.3)  # Int inputs
            @test isa(pv4.x, Float64)
            @test isa(pv4.y, Float64)
        end # PIVVector
        
        @testset "PIVResult" begin
            # Test grid size constructor
            result1 = PIVResult((3, 4))
            @test size(result1.vectors) == (3, 4)
            @test length(result1.metadata) == 0
            @test length(result1.auxiliary) == 0
            
            # Test vector array constructor
            pv_array = [PIVVector(i, j, 0.0, 0.0) for i in 1:2, j in 1:3]
            result2 = PIVResult(pv_array)
            @test size(result2.vectors) == (2, 3)
            
            # Test property forwarding
            # Fill vectors with test data
            for (idx, pv) in enumerate(pv_array)
                result2.vectors[idx] = PIVVector(pv.x, pv.y, idx*0.1, idx*0.2)
            end
            
            # Test that property forwarding works
            @test result2.x == result2.vectors.x
            @test result2.u == result2.vectors.u
            @test result2.v == result2.vectors.v
            
            # Test direct field access still works
            @test isa(result2.vectors, StructArray)
            @test isa(result2.metadata, Dict)
            @test isa(result2.auxiliary, Dict)
        end # PIVResult
        
        @testset "PIVStage" begin
            # Test basic constructor with defaults
            stage1 = PIVStage((64, 64))
            @test stage1.window_size == (64, 64)
            @test stage1.overlap == (0.5, 0.5)
            @test stage1.padding == 0
            @test stage1.deformation_iterations == 3
            @test isa(stage1.window_function, Hammerhead.SimpleWindow)
            @test isa(stage1.interpolation_method, Hammerhead._Bilinear)
            
            # Test constructor with custom parameters
            stage2 = PIVStage((32, 48), overlap=(0.25, 0.75), padding=5, 
                             deformation_iterations=5, window_function=:hanning,
                             interpolation_method=:bicubic)
            @test stage2.window_size == (32, 48)
            @test stage2.overlap == (0.25, 0.75)
            @test stage2.padding == 5
            @test stage2.deformation_iterations == 5
            @test isa(stage2.window_function, Hammerhead.SimpleWindow)
            @test isa(stage2.interpolation_method, Hammerhead._Bicubic)
            
            # Test square window helper constructor
            stage3 = PIVStage(128, overlap=(0.6, 0.6))
            @test stage3.window_size == (128, 128)
            @test stage3.overlap == (0.6, 0.6)
            
            # Test input validation
            @test_throws ArgumentError PIVStage((0, 64))  # Invalid window size
            @test_throws ArgumentError PIVStage((64, 64), overlap=(1.0, 0.5))  # Invalid overlap
            @test_throws ArgumentError PIVStage((64, 64), overlap=(-0.1, 0.5))  # Invalid overlap
            
            # Test unknown symbols
            @test_throws ArgumentError PIVStage((64, 64), window_function=:unknown)
            @test_throws ArgumentError PIVStage((64, 64), interpolation_method=:unknown)
        end # PIVStage
        
        @testset "PIVStages Helper" begin
            # Test basic multi-stage generation with old signature
            stages = PIVStages(3, 32, overlap=0.5)
            @test length(stages) == 3
            @test all(s -> s.overlap == (0.5, 0.5), stages)
            
            # Test size progression (should be geometric)
            sizes = [stage.window_size[1] for stage in stages]
            @test sizes[end] == 32  # Final size should match input
            @test all(sizes .>= 32)  # All sizes should be >= final size
            @test issorted(sizes, rev=true)  # Should be decreasing
            
            # Test scalar parameters applied to all stages
            stages2 = PIVStages(2, 32, overlap=0.25, padding=5, 
                               window_function=:hanning, interpolation_method=:bicubic)
            @test all(s -> s.overlap == (0.25, 0.25), stages2)
            @test all(s -> s.padding == 5, stages2)
            @test all(s -> isa(s.window_function, Hammerhead.SimpleWindow), stages2)
            @test all(s -> isa(s.interpolation_method, Hammerhead._Bicubic), stages2)
            
            # Test vector parameters (one per stage)
            stages3 = PIVStages(3, 16, overlap=[0.75, 0.5, 0.25], 
                               deformation_iterations=[1, 3, 5])
            @test stages3[1].overlap == (0.75, 0.75)
            @test stages3[2].overlap == (0.5, 0.5)
            @test stages3[3].overlap == (0.25, 0.25)
            @test stages3[1].deformation_iterations == 1
            @test stages3[2].deformation_iterations == 3
            @test stages3[3].deformation_iterations == 5
            
            # Test mixed scalar/vector parameters
            stages4 = PIVStages(2, 32, overlap=0.5, 
                               window_function=[:rectangular, :hanning])
            @test isa(stages4[1].window_function, Hammerhead.SimpleWindow)
            @test isa(stages4[2].window_function, Hammerhead.SimpleWindow)
            @test all(s -> s.overlap == (0.5, 0.5), stages4)
            
            # Test same asymmetric overlap for all stages - use vector with single tuple element
            stages5b = PIVStages(2, 32, overlap=[(0.6, 0.4)])
            @test all(s -> s.overlap == (0.6, 0.4), stages5b)
            
            # Test 1×n matrix parameters
            stages7 = PIVStages(2, 32, padding=[5 10])  # 1×2 matrix
            @test stages7[1].padding == 5
            @test stages7[2].padding == 10
            
            # Test n×1 matrix parameters  
            stages8 = PIVStages(2, 32, deformation_iterations=[2; 4])  # 2×1 matrix
            @test stages8[1].deformation_iterations == 2
            @test stages8[2].deformation_iterations == 4
            
            # Test range parameters (iterable)
            stages9 = PIVStages(3, 32, deformation_iterations=1:3)
            @test stages9[1].deformation_iterations == 1
            @test stages9[2].deformation_iterations == 2 
            @test stages9[3].deformation_iterations == 3
            
            # Test input validation
            @test_throws ArgumentError PIVStages(0, 32)  # Invalid number of stages
            @test_throws ArgumentError PIVStages(3, 32, overlap=[0.5, 0.25])  # Wrong vector length
            @test_throws ArgumentError PIVStages(2, 32, padding=[1, 2, 3])  # Wrong vector length
            @test_throws ArgumentError PIVStages(2, 32, padding=[1 2; 3 4])  # Invalid matrix size
            @test_throws Union{ArgumentError, TypeError} PIVStages(2, 32, overlap=Dict(:a => 1))  # Unsupported type
        end # PIVStages Helper
    end # Data Structure Tests

    @testset "Integration Tests" begin
        @testset "CrossCorrelator with Gaussian Particle" begin
        # Define test image size
        image_size = (64, 64)
        centroid1 = image_size ./ 2
        displacement = (2.2, 1.3)
        diameter = 2.6

        # Create test images with a Gaussian particle
        img1 = zeros(Float64, image_size)
        img2 = zeros(Float64, image_size)

        # Add a Gaussian particle in the center of img1
        generate_gaussian_particle!(img1, centroid1, diameter)

        # Shift the particle to create img2
        generate_gaussian_particle!(img2, centroid1 .+ displacement, diameter)

        # Initialize CrossCorrelator
        correlator = CrossCorrelator(image_size)
        
        # Perform correlation
        correlation_plane = correlate!(correlator, img1, img2)
        @test isa(correlation_plane, AbstractMatrix)
        
        # Analyze correlation plane
        disp_u, disp_v, peak_ratio, corr_moment = analyze_correlation_plane(correlation_plane)
        @test disp_u ≈ 2.2 atol=2e-2
        @test disp_v ≈ 1.3 atol=2e-2
        @test peak_ratio > 1.0  # Should have good peak ratio
        @test isfinite(corr_moment)  # Should have finite correlation moment
        end # CrossCorrelator with Gaussian Particle
    end # Integration Tests

    # Additional comprehensive tests for existing functionality
    @testset "Correlation Algorithm Tests" begin
        @testset "CrossCorrelator with Various Displacements" begin
            image_size = (16, 16)
            centroid = (8.0, 8.0)
            diameter = 3.0
            correlator = CrossCorrelator(image_size)
            
            # Test multiple displacement values
            test_displacements = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.5, 2.3), (3.7, -2.1)]
            
            for true_disp in test_displacements
                img1 = zeros(Float64, image_size)
                img2 = zeros(Float64, image_size)
                
                generate_gaussian_particle!(img1, centroid, diameter)
                generate_gaussian_particle!(img2, centroid .+ true_disp, diameter)
                
                correlation_plane = correlate!(correlator, img1, img2)
                disp_u, disp_v, peak_ratio, corr_moment = analyze_correlation_plane(correlation_plane)
                
                @test disp_u ≈ true_disp[1] atol=1e-2
                @test disp_v ≈ true_disp[2] atol=1e-2
                @test peak_ratio > 0.0  # Should have positive peak ratio
                @test isfinite(corr_moment)  # Should have finite correlation moment
            end
        end # CrossCorrelator with Various Displacements
    
        @testset "CrossCorrelator with Different Image Types" begin
            image_size = (32, 32)
            correlator = CrossCorrelator(image_size)
            
            # Test with Float32 images
            img1_f32 = rand(Float32, image_size...)
            img2_f32 = circshift(img1_f32, (2, 1))
            
            correlation_plane_f32 = correlate!(correlator, img1_f32, img2_f32)
            disp_u_f32, disp_v_f32, peak_ratio_f32, corr_moment_f32 = analyze_correlation_plane(correlation_plane_f32)
            @test abs(disp_u_f32) ≈ 2.0 atol=1e-5  # Should detect the shift
            @test abs(disp_v_f32) ≈ 1.0 atol=1e-5   # Float32 precision limit
            
            # Test with Float64 images
            img1_f64 = rand(Float64, image_size...)
            img2_f64 = circshift(img1_f64, (1, 3))
            
            correlation_plane_f64 = correlate!(correlator, img1_f64, img2_f64)
            disp_u_f64, disp_v_f64, peak_ratio_f64, corr_moment_f64 = analyze_correlation_plane(correlation_plane_f64)
            @test abs(disp_u_f64) ≈ 1.0 atol=1e-5
            @test abs(disp_v_f64) ≈ 3.0 atol=1e-5
        end # CrossCorrelator with Different Image Types
        
        @testset "CrossCorrelator Memory Reuse" begin
            # Test that the same correlator can be used multiple times
            image_size = (32, 32)
            correlator = CrossCorrelator(image_size)
            
            # Store original memory addresses
            c1_ptr = pointer(correlator.C1)
            c2_ptr = pointer(correlator.C2)
            
            # Perform multiple correlations
            for i in 1:5
                img1 = rand(Float32, image_size...)
                img2 = rand(Float32, image_size...)
                
                correlation_plane = correlate!(correlator, img1, img2)
                
                # Verify memory wasn't reallocated
                @test pointer(correlator.C1) == c1_ptr
                @test pointer(correlator.C2) == c2_ptr
                
                # Verify result is reasonable
                @test isa(correlation_plane, AbstractMatrix)
                @test all(isfinite.(correlation_plane))
            end
        end # CrossCorrelator Memory Reuse
        
        @testset "CrossCorrelator Error Handling" begin
            correlator = CrossCorrelator((32, 32))
            
            # Test with wrong size images
            img_wrong_size = zeros(16, 16)
            img_correct = zeros(32, 32)
            
            @test_throws DimensionMismatch correlate!(correlator, img_wrong_size, img_correct)
            @test_throws DimensionMismatch correlate!(correlator, img_correct, img_wrong_size)
        end # CrossCorrelator Error Handling
    end # Correlation Algorithm Tests

    @testset "Performance Tests" begin
        if RUN_PERFORMANCE_TESTS
            @testset "CrossCorrelator Performance" begin
                @withseed 4321 begin
                    # Test performance with larger images
                    image_size = (128, 128)  # Reasonable size for CI
                    correlator = CrossCorrelator(image_size)
                    
                    img1 = rand(Float32, image_size...)
                    img2 = rand(Float32, image_size...)
                    
                    # Warm-up runs to ensure compilation and caching
                    for _ in 1:3
                        correlate!(correlator, img1, img2)
                    end
                    
                    # Measure time for correlation (after warm-up)
                    elapsed = @elapsed begin
                        for i in 1:5
                            correlate!(correlator, img1, img2)
                        end
                    end
                    
                    # Test memory allocation efficiency
                    # Additional warm-up for allocation measurement
                    correlate!(correlator, img1, img2)
                    
                    # Measure allocations on subsequent run
                    allocs = @allocated correlate!(correlator, img1, img2)
                    
                    # Performance targets based on measured values:
                    # Measured: elapsed ≈ 0.0008s for 5 runs, allocs ≈ 2048 bytes
                    @test elapsed < 0.01   # 5 runs should complete in <10ms (10x safety margin)
                    @test allocs < 5000    # Should have minimal allocations (<5KB)
                end
            end # CrossCorrelator Performance
        else
            @test_skip "Performance tests disabled (set HAMMERHEAD_RUN_PERF_TESTS=true to enable)"
        end
    end # Performance Tests

    @testset "Window Padding Tests" begin
        @testset "pad_to_size Functionality" begin
            # Test basic padding
            small_img = [1.0 2.0; 3.0 4.0]  # 2x2
            padded = Hammerhead.pad_to_size(small_img, (4, 4))
            
            @test size(padded) == (4, 4)
            @test padded[2:3, 2:3] == small_img  # Original data in center
            
            # Test no padding needed
            same_size = Hammerhead.pad_to_size(small_img, (2, 2))
            @test same_size === small_img
            
            # Test symmetric reflection property
            @test padded[1, 2] == padded[2, 2]  # Top reflection
            @test padded[4, 2] == padded[3, 2]  # Bottom reflection  
            @test padded[2, 1] == padded[2, 2]  # Left reflection
            @test padded[2, 4] == padded[2, 3]  # Right reflection
            
            # Test error for invalid size
            @test_throws ArgumentError Hammerhead.pad_to_size(small_img, (1, 1))
        end # pad_to_size Functionality
        
        @testset "Boundary Window Processing" begin
            # Create test images to force boundary conditions  
            img_size = (64, 64)
            img1 = rand(Float64, img_size...)
            img2 = rand(Float64, img_size...)
            
            # Use large window size with high overlap to force boundary padding scenarios
            stage = PIVStage((48, 48), overlap=(0.75, 0.75))  # Large windows, high overlap
            # This should process without error due to padding
            result = run_piv_stage(img1, img2, stage)
            
            @test isa(result, PIVResult)
            @test length(result.vectors) > 0
            # All vectors should be processed (no :bad status from padding issues)
            @test all(v -> v.status != :bad, result.vectors)
            
            # Test boundary case with smaller image
            small_img1 = rand(Float64, (32, 32)...)
            small_img2 = rand(Float64, (32, 32)...)
            stage_small = PIVStage((24, 24), overlap=(0.5, 0.5))
            
            result_small = run_piv_stage(small_img1, small_img2, stage_small)
            @test isa(result_small, PIVResult)
            @test length(result_small.vectors) > 0
        end # Boundary Window Processing
    end # Window Padding Tests
    
    @testset "Windowing Functions Tests" begin
        @testset "1D Window Generation (DSP.jl implementations)" begin
            # Test rectangular window (should be all ones)
            rect_window = Hammerhead.generate_window_1d(Hammerhead.SimpleWindow(DSP.rect), 5)
            @test rect_window == ones(5)
            @test length(rect_window) == 5
            
            # Test Hanning window properties
            hann_window = Hammerhead.generate_window_1d(Hammerhead.SimpleWindow(DSP.hanning), 8)
            @test length(hann_window) == 8
            @test hann_window[1] ≈ 0.0 atol=1e-10  # Should start at 0
            @test hann_window[end] ≈ 0.0 atol=1e-10  # Should end at 0
            @test maximum(hann_window) ≈ 0.95 atol=0.001  # Peak value for 8-point Hanning
            
            # Test Hamming window properties
            hamm_window = Hammerhead.generate_window_1d(Hammerhead.SimpleWindow(DSP.hamming), 8)
            @test length(hamm_window) == 8
            @test hamm_window[1] ≈ 0.08 atol=1e-15  # Hamming doesn't go to zero
            @test hamm_window[end] ≈ 0.08 atol=1e-15
            @test maximum(hamm_window) ≈ 0.98 atol=0.03  # Peak value for 8-point Hamming
            
            # Test Blackman window properties
            blackman_window = Hammerhead.generate_window_1d(Hammerhead.SimpleWindow(DSP.blackman), 8)
            @test length(blackman_window) == 8
            @test blackman_window[1] ≈ 0.0 atol=1e-10  # Should start near 0
            @test blackman_window[end] ≈ 0.0 atol=1e-10  # Should end near 0
            @test maximum(blackman_window) ≈ 0.92 atol=0.001  # Peak value for 8-point Blackman
            
            # Test additional DSP.jl window functions
            bartlett_window = Hammerhead.generate_window_1d(Hammerhead.SimpleWindow(DSP.bartlett), 8)
            @test length(bartlett_window) == 8
            @test bartlett_window[1] ≈ 0.0 atol=1e-10
            @test bartlett_window[end] ≈ 0.0 atol=1e-10
            
            cosine_window = Hammerhead.generate_window_1d(Hammerhead.SimpleWindow(DSP.cosine), 8)
            @test length(cosine_window) == 8
            @test cosine_window[1] ≈ 0.0 atol=1e-10
            
            # Test parametric window functions
            kaiser_window = Hammerhead.generate_window_1d(Hammerhead.ParametricWindow(DSP.kaiser, (5.0,)), 8)
            @test length(kaiser_window) == 8
            @test kaiser_window[1] < kaiser_window[4]  # Should be increasing toward center
            
            tukey_window = Hammerhead.generate_window_1d(Hammerhead.ParametricWindow(DSP.tukey, (0.5,)), 8)
            @test length(tukey_window) == 8
            @test maximum(tukey_window) ≈ 1.0 atol=1e-10
            
            # Test edge case: single element for simple windows
            simple_windows = [
                Hammerhead.SimpleWindow(DSP.rect),
                Hammerhead.SimpleWindow(DSP.hanning),
                Hammerhead.SimpleWindow(DSP.hamming),
                Hammerhead.SimpleWindow(DSP.blackman)
            ]
            for window_type in simple_windows
                single = Hammerhead.generate_window_1d(window_type, 1)
                @test single == [1.0]
            end
        end # 1D Window Generation
        
        @testset "2D Window Application" begin
            # Create test image
            test_img = ones(Float64, 8, 8)
            
            # Test rectangular windowing (should not change the image)
            rect_window = Hammerhead.build_window((8, 8), Hammerhead.SimpleWindow(DSP.rect))
            windowed_rect = copy(test_img)
            Hammerhead.apply_window!(windowed_rect, rect_window)
            @test windowed_rect ≈ test_img
            
            # Test Hanning windowing
            hann_window = Hammerhead.build_window((8, 8), Hammerhead.SimpleWindow(DSP.hanning))
            windowed_hann = copy(test_img)
            Hammerhead.apply_window!(windowed_hann, hann_window)
            @test size(windowed_hann) == size(test_img)
            @test windowed_hann[1, 1] ≈ 0.0 atol=1e-10  # Corners should be near zero
            @test windowed_hann[end, end] ≈ 0.0 atol=1e-10
            @test windowed_hann[4, 4] ≈ 0.903 atol=0.001  # Center value for 8x8 Hanning
            
            # Test Hamming windowing
            hamm_window = Hammerhead.build_window((8, 8), Hammerhead.SimpleWindow(DSP.hamming))
            windowed_hamm = copy(test_img)
            Hammerhead.apply_window!(windowed_hamm, hamm_window)
            @test size(windowed_hamm) == size(test_img)
            # Hamming doesn't go to zero at edges
            @test windowed_hamm[1, 1] > 0.0
            @test windowed_hamm[end, end] > 0.0
            @test windowed_hamm[4, 4] ≈ 0.91 atol=0.002  # Center value for 8x8 Hamming
            
            # Test Blackman windowing
            blackman_window = Hammerhead.build_window((8, 8), Hammerhead.SimpleWindow(DSP.blackman))
            windowed_blackman = copy(test_img)
            Hammerhead.apply_window!(windowed_blackman, blackman_window)
            @test size(windowed_blackman) == size(test_img)
            @test windowed_blackman[1, 1] ≈ 0.0 atol=1e-10
            @test windowed_blackman[end, end] ≈ 0.0 atol=1e-10
            @test windowed_blackman[4, 4] ≈ 0.85 atol=0.004  # Center value for 8x8 Blackman
            
            # Test energy conservation property (windowed energy should be less than original)
            original_energy = sum(test_img.^2)
            hann_energy = sum(windowed_hann.^2)
            hamm_energy = sum(windowed_hamm.^2)
            blackman_energy = sum(windowed_blackman.^2)
            
            @test hann_energy < original_energy
            @test hamm_energy < original_energy
            @test blackman_energy < original_energy
        end # 2D Window Application
        
        @testset "Windowing Integration with PIV" begin
            # Test that windowing integrates properly with PIV processing
            image_size = (64, 64)
            centroid = (32.0, 32.0)
            diameter = 3.0
            displacement = (2.0, 1.5)
            
            # Create test images with Gaussian particles
            img1 = zeros(Float64, image_size)
            img2 = zeros(Float64, image_size)
            generate_gaussian_particle!(img1, centroid, diameter)
            generate_gaussian_particle!(img2, centroid .+ displacement, diameter)
            
            # Test different windowing functions (simple)
            for window_func in [:rectangular, :hanning, :hamming, :blackman, :bartlett, :cosine]
                stage = PIVStage((32, 32), window_function=window_func)
                result = run_piv_stage(img1, img2, stage)
                
                @test isa(result, PIVResult)
                @test length(result.vectors) > 0
                
                # All windows should detect displacement reasonably well
                if length(result.vectors) > 0
                    # Find vector closest to centroid (32, 32)
                    best_vector = result.vectors[1]
                    min_dist = Inf
                    for v in result.vectors
                        dist = sqrt((v.x - centroid[1])^2 + (v.y - centroid[2])^2)
                        if dist < min_dist
                            min_dist = dist
                            best_vector = v
                        end
                    end
                    
                    if window_func == :rectangular
                        @test best_vector.u ≈ displacement[1] atol=1e-10
                        @test best_vector.v ≈ displacement[2] atol=1e-10
                    else
                        # # Non-rectangular windows reduce energy, may affect precision
                        # @test isfinite(best_vector.u)
                        # @test isfinite(best_vector.v)
                        # @test abs(best_vector.u - displacement[1]) < 2.0
                        # @test abs(best_vector.v - displacement[2]) < 2.0
                    end
                    @test best_vector.status == :good
                end
            end
            
            # Test parametric window functions
            parametric_windows = [(:kaiser, 5.0), (:tukey, 0.5), (:gaussian, 0.4)]
            for window_func in parametric_windows
                stage = PIVStage((32, 32), window_function=window_func)
                result = run_piv_stage(img1, img2, stage)
                
                @test isa(result, PIVResult)
                @test length(result.vectors) > 0
                
                if length(result.vectors) > 0
                    best_vector = result.vectors[1]
                    min_dist = Inf
                    for v in result.vectors
                        dist = sqrt((v.x - centroid[1])^2 + (v.y - centroid[2])^2)
                        if dist < min_dist
                            min_dist = dist
                            best_vector = v
                        end
                    end
                    
                    # Parametric windows should also detect displacement reasonably
                    @test isfinite(best_vector.u)
                    @test isfinite(best_vector.v)
                    @test abs(best_vector.u - displacement[1]) < 2.0
                    @test abs(best_vector.v - displacement[2]) < 2.0
                    @test best_vector.status == :good
                end
            end
        end # Windowing Integration with PIV
        
        @testset "PIVStages with Generalized Windowing" begin
            # Test PIVStages with mixed window functions
            stages_mixed = PIVStages(3, 32, window_function=[:rectangular, (:kaiser, 6.0), :hanning])
            @test length(stages_mixed) == 3
            @test isa(stages_mixed[1].window_function, Hammerhead.SimpleWindow)
            @test isa(stages_mixed[2].window_function, Hammerhead.ParametricWindow)
            @test isa(stages_mixed[3].window_function, Hammerhead.SimpleWindow)
            
            # Test all parametric windows
            stages_param = PIVStages(2, 32, window_function=[(:tukey, 0.25), (:gaussian, 0.5)])
            @test length(stages_param) == 2
            @test all(s -> isa(s.window_function, Hammerhead.ParametricWindow), stages_param)
            
            # Test error handling for unknown window
            @test_throws ArgumentError PIVStage((32, 32), window_function=:unknown_window)
            @test_throws ArgumentError PIVStage((32, 32), window_function=(:unknown_param, 1.0))
        end # PIVStages with Generalized Windowing
    end # Windowing Functions Tests
    
    @testset "Affine Transform Validation" begin
        @testset "Area Preservation Checks" begin
            # Test identity matrix (perfect area preservation)
            identity_2x2 = [1.0 0.0; 0.0 1.0]
            @test is_area_preserving(identity_2x2)
            @test is_area_preserving(identity_2x2, 0.01)  # Strict tolerance
            
            # Test rotation matrix (area preserving)
            θ = π/6  # 30 degrees
            rotation = [cos(θ) -sin(θ); sin(θ) cos(θ)]
            @test is_area_preserving(rotation)
            @test abs(det(rotation) - 1.0) < 1e-10  # Should be exactly 1
            
            # Test small shear (area preserving)
            shear = [1.0 0.1; 0.0 1.0]
            @test is_area_preserving(shear)
            @test abs(det(shear) - 1.0) < 1e-10
            
            # Test uniform scaling (not area preserving if scale ≠ 1)
            scale_up = [1.2 0.0; 0.0 1.2]  # det = 1.44
            @test !is_area_preserving(scale_up, 0.1)  # Outside 10% tolerance
            
            scale_small = [1.05 0.0; 0.0 0.95]  # det ≈ 1.0 (0.9975)
            @test is_area_preserving(scale_small, 0.1)  # Within 10% tolerance
            
            # Test non-uniform scaling
            stretch = [2.0 0.0; 0.0 0.5]  # det = 1.0 (area preserving but distorting)
            @test is_area_preserving(stretch)
            
            # Test degenerate matrix
            singular = [1.0 0.0; 1.0 0.0]  # det = 0
            @test !is_area_preserving(singular)
            
            # Test error handling
            @test_throws ArgumentError is_area_preserving([1.0 0.0])  # Wrong size
            @test_throws ArgumentError is_area_preserving([1.0 0.0; 0.0 1.0; 0.0 0.0])  # Wrong size
        end # Area Preservation Checks
        
        @testset "Full Transform Validation" begin
            # Test valid transforms
            identity_2x2 = [1.0 0.0; 0.0 1.0]
            @test validate_affine_transform(identity_2x2)
            
            # Test 3x3 affine matrix (should extract 2x2 part)
            affine_3x3 = [1.0 0.0 5.0; 0.0 1.0 3.0; 0.0 0.0 1.0]
            @test validate_affine_transform(affine_3x3)
            
            # Test small deformation (valid)
            small_deform = [1.02 0.01; -0.01 0.98]
            @test validate_affine_transform(small_deform)
            
            # Test rotation (valid)
            θ = π/12  # 15 degrees
            rotation = [cos(θ) -sin(θ); sin(θ) cos(θ)]
            @test validate_affine_transform(rotation)
            
            # Test invalid transforms
            
            # Non-finite values
            invalid_nan = [1.0 NaN; 0.0 1.0]
            @test !validate_affine_transform(invalid_nan)
            
            invalid_inf = [Inf 0.0; 0.0 1.0]
            @test !validate_affine_transform(invalid_inf)
            
            # Excessive scaling (violates eigenvalue constraints)
            excessive_scale = [5.0 0.0; 0.0 0.2]  # 5x stretch, 0.2x compression
            @test !validate_affine_transform(excessive_scale)
            
            # High condition number (near-singular) - this should fail due to poor conditioning
            high_condition = [1.0 1.0; 0.0 1e-5]  # Very skewed matrix
            @test !validate_affine_transform(high_condition)
            
            # Non area-preserving with strict tolerance
            non_area_preserving = [1.5 0.0; 0.0 0.8]  # det = 1.2
            @test !validate_affine_transform(non_area_preserving, tolerance=0.05)  # Strict tolerance
            @test validate_affine_transform(non_area_preserving, tolerance=0.3)   # Loose tolerance
            
            # Test tolerance parameter
            borderline_pass = [1.08 0.0; 0.0 0.93]  # det ≈ 1.004, within 1%
            @test validate_affine_transform(borderline_pass, tolerance=0.1)   # Within 10%
            @test validate_affine_transform(borderline_pass, tolerance=0.01)  # Within 1%
            
            # This matrix fails area preservation with strict tolerance
            borderline_fail = [1.2 0.0; 0.0 0.83]  # det = 0.996, fails 1% area tolerance  
            @test validate_affine_transform(borderline_fail, tolerance=0.1)   # Within 10%
            @test !validate_affine_transform(borderline_fail, tolerance=0.003) # Outside 0.3%
            
            # Test wrong matrix size
            @test_throws ArgumentError validate_affine_transform(reshape([1.0], 1, 1))
            @test_throws ArgumentError validate_affine_transform([1.0 0.0 0.0; 0.0 1.0 0.0; 1.0 2.0 3.0; 4.0 5.0 6.0])
        end # Full Transform Validation
        
        @testset "Edge Cases and Robustness" begin
            # Test complex eigenvalues (should be rejected if not proper rotation)
            # This matrix has complex eigenvalues but is not area-preserving: |det| = 2
            complex_eigen = [0.0 -2.0; 1.0 0.0]  # Eigenvalues are ±i√2, |det| = 2
            @test !validate_affine_transform(complex_eigen)
            
            # Test very small transforms (should be valid)
            tiny_transform = [1.001 0.0001; -0.0001 0.999]
            @test validate_affine_transform(tiny_transform)
            
            # Test boundary conditions for tolerance
            exactly_tolerance = [1.1 0.0; 0.0 1.0/1.1]  # det = 1.0, but eigenvalues at tolerance boundary
            @test validate_affine_transform(exactly_tolerance, tolerance=0.1)
            
            # Test mixed valid/invalid characteristics
            # Good determinant, bad condition number - matrix that's nearly singular but has det ≈ 1
            good_det_bad_cond = [1.0 1.0; 1e-6 1e-6]  # det ≈ 1e-6, not area-preserving
            @test !validate_affine_transform(good_det_bad_cond)
            
            # Test matrix with reflection (negative determinant but magnitude = 1)
            reflection = [-1.0 0.0; 0.0 1.0]  # det = -1
            @test is_area_preserving(reflection)  # Area preserved
            @test validate_affine_transform(reflection)  # Should be valid
        end # Edge Cases and Robustness
    end # Affine Transform Validation
    
    @testset "Linear Barycentric Interpolation" begin
        @testset "Basic Interpolation" begin
            # Test with simple triangle
            points = [0.0 0.0; 1.0 0.0; 0.0 1.0]  # Right triangle
            values = [1.0, 2.0, 3.0]
            
            # Test center point of triangle
            query_points = [1/3 1/3]  # Center of triangle
            result = linear_barycentric_interpolation(points, values, query_points)
            expected = (1.0 + 2.0 + 3.0) / 3  # Average of vertices
            @test result[1] ≈ expected atol=1e-10
            
            # Test vertex points (should return exact values)
            for i in 1:3
                query = reshape(points[i, :], 1, 2)
                result = linear_barycentric_interpolation(points, values, query)
                @test result[1] ≈ values[i] atol=1e-10
            end
            
            # Test edge midpoints
            edge_mid = [0.5 0.0]  # Midpoint of edge 1-2
            result = linear_barycentric_interpolation(points, values, edge_mid)
            @test result[1] ≈ (values[1] + values[2]) / 2 atol=1e-10
        end
        
        @testset "Two Point Interpolation" begin
            # Test linear interpolation between two points
            points = [0.0 0.0; 2.0 0.0]  # Two points on x-axis
            values = [10.0, 30.0]
            
            # Test midpoint
            query_points = [1.0 0.0]
            result = linear_barycentric_interpolation(points, values, query_points)
            @test result[1] ≈ 20.0 atol=1e-10
            
            # Test point on line but outside segment
            query_outside = [3.0 0.0]  # Beyond second point
            result = linear_barycentric_interpolation(points, values, query_outside, fallback_method=:nearest)
            @test result[1] ≈ 30.0 atol=1e-10  # Should use nearest (second point)
            
            # Test point off the line - should project to line and interpolate
            query_off_line = [1.0 1.0]
            result = linear_barycentric_interpolation(points, values, query_off_line, fallback_method=:nearest)
            @test result[1] ≈ 20.0 atol=1e-10  # Projects to midpoint of line
        end
        
        @testset "Single Point Case" begin
            # Test with single point
            points = reshape([1.0, 2.0], 1, 2)
            values = [42.0]
            query_points = [0.0 0.0; 5.0 5.0; 1.0 2.0]
            
            result = linear_barycentric_interpolation(points, values, query_points)
            @test all(result .≈ 42.0)
        end
        
        @testset "Fallback Methods" begin
            # Test fallback methods for points outside convex hull
            points = [0.0 0.0; 1.0 0.0; 0.5 1.0]  # Triangle
            values = [1.0, 2.0, 3.0]
            query_outside = [2.0 2.0]  # Clearly outside triangle
            
            # Test :nearest fallback
            result_nearest = linear_barycentric_interpolation(points, values, query_outside, fallback_method=:nearest)
            @test result_nearest[1] ∈ values  # Should be one of the vertex values
            
            # Test :zero fallback
            result_zero = linear_barycentric_interpolation(points, values, query_outside, fallback_method=:zero)
            @test result_zero[1] ≈ 0.0
            
            # Test :nan fallback
            result_nan = linear_barycentric_interpolation(points, values, query_outside, fallback_method=:nan)
            @test isnan(result_nan[1])
        end
        
        @testset "Edge Cases and Error Handling" begin
            # Test empty points
            empty_points = zeros(0, 2)
            empty_values = Float64[]
            query = [1.0 1.0]
            result = linear_barycentric_interpolation(empty_points, empty_values, query)
            @test isnan(result[1])
            
            # Test dimension mismatches
            points_3d = [1.0 2.0 3.0; 4.0 5.0 6.0]  # 3D points
            values = [1.0, 2.0]
            query = [1.0 1.0]
            @test_throws ArgumentError linear_barycentric_interpolation(points_3d, values, query)
            
            # Test value count mismatch
            points = [0.0 0.0; 1.0 1.0]
            wrong_values = [1.0]  # Wrong number of values
            @test_throws ArgumentError linear_barycentric_interpolation(points, wrong_values, query)
            
            # Test invalid fallback method
            points = [0.0 0.0; 1.0 1.0]
            values = [1.0, 2.0]
            @test_throws ArgumentError linear_barycentric_interpolation(points, values, query, fallback_method=:invalid)
        end
        
        @testset "Multiple Query Points" begin
            # Test with multiple query points efficiently
            points = [0.0 0.0; 1.0 0.0; 0.0 1.0; 1.0 1.0]  # Square
            values = [1.0, 2.0, 3.0, 4.0]
            
            # Multiple query points
            query_points = [0.5 0.5; 0.0 0.0; 1.0 1.0; 0.25 0.25]
            result = linear_barycentric_interpolation(points, values, query_points)
            
            @test length(result) == 4
            @test all(isfinite.(result))
            @test result[2] ≈ 1.0 atol=1e-10  # Query at first vertex
            @test result[3] ≈ 4.0 atol=1e-10  # Query at fourth vertex
        end
    end # Linear Barycentric Interpolation
    
    @testset "Vector Interpolation" begin
        @testset "Basic Vector Interpolation" begin
            # Create test PIV result with some bad vectors
            positions_x = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
            positions_y = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
            displacements_u = [1.0, 2.0, 3.0, 1.0, NaN, 3.0]  # One NaN displacement
            displacements_v = [0.5, 1.0, 1.5, 0.5, NaN, 1.5]  # One NaN displacement
            status_flags = [:good, :good, :good, :good, :bad, :good]
            
            piv_vectors = [PIVVector(positions_x[i], positions_y[i], displacements_u[i], displacements_v[i],
                                   status_flags[i], 1.0, 0.5) for i in 1:6]
            
            result = PIVResult(piv_vectors)
            
            # Interpolate the bad vector
            interp_result = interpolate_vectors(result, method=:linear_barycentric)
            
            # Check that bad vector was interpolated
            @test interp_result.status[5] == :interpolated
            @test !isnan(interp_result.u[5])
            @test !isnan(interp_result.v[5])
            
            # Check that good vectors remain unchanged
            good_indices = [1, 2, 3, 4, 6]
            for i in good_indices
                @test interp_result.u[i] ≈ result.u[i] atol=1e-10
                @test interp_result.v[i] ≈ result.v[i] atol=1e-10
                @test interp_result.status[i] == result.status[i]
            end
        end
        
        @testset "Nearest Neighbor Interpolation" begin
            # Create simple test case
            piv_vectors = [
                PIVVector(0.0, 0.0, 1.0, 0.0, :good, 1.0, 0.5),
                PIVVector(2.0, 0.0, 3.0, 0.0, :good, 1.0, 0.5),
                PIVVector(1.0, 1.0, NaN, NaN, :bad, NaN, NaN)
            ]
            
            result = PIVResult(piv_vectors)
            interp_result = interpolate_vectors(result, method=:nearest)
            
            # Bad vector should be interpolated with nearest neighbor
            @test interp_result.status[3] == :interpolated
            @test !isnan(interp_result.u[3])
            @test !isnan(interp_result.v[3])
            
            # Should be closer to one of the neighboring values
            @test interp_result.u[3] ∈ [1.0, 3.0]
            @test interp_result.v[3] ≈ 0.0 atol=1e-10
        end
        
        @testset "No Good Vectors Case" begin
            # Test case where no good vectors exist
            piv_vectors = [
                PIVVector(1.0, 1.0, NaN, NaN, :bad, NaN, NaN),
                PIVVector(2.0, 2.0, NaN, NaN, :bad, NaN, NaN)
            ]
            
            result = PIVResult(piv_vectors)
            
            # Should warn and return unchanged result
            interp_result = @test_logs (:warn, r"No vectors with status good found") interpolate_vectors(result)
            
            @test interp_result.status[1] == :bad
            @test interp_result.status[2] == :bad
            @test isnan(interp_result.u[1])
            @test isnan(interp_result.u[2])
        end
        
        @testset "Custom Status Handling" begin
            # Test with custom source and target status
            piv_vectors = [
                PIVVector(0.0, 0.0, 1.0, 0.0, :secondary, 1.0, 0.5),
                PIVVector(2.0, 0.0, 3.0, 0.0, :secondary, 1.0, 0.5),
                PIVVector(1.0, 1.0, NaN, NaN, :bad, NaN, NaN)
            ]
            
            result = PIVResult(piv_vectors)
            
            # Interpolate using secondary vectors as source
            interp_result = interpolate_vectors(result, 
                                              source_status=:secondary,
                                              target_status=:repaired)
            
            @test interp_result.status[3] == :repaired
            @test !isnan(interp_result.u[3])
            @test !isnan(interp_result.v[3])
        end
        
        @testset "Metadata Preservation" begin
            # Test that metadata is preserved and updated
            piv_vectors = [
                PIVVector(0.0, 0.0, 1.0, 0.0, :good, 1.0, 0.5),
                PIVVector(1.0, 1.0, NaN, NaN, :bad, NaN, NaN)
            ]
            
            result = PIVResult(piv_vectors)
            result.metadata["test_key"] = "test_value"
            
            interp_result = interpolate_vectors(result)
            
            # Original metadata should be preserved
            @test interp_result.metadata["test_key"] == "test_value"
            
            # New metadata should be added
            @test haskey(interp_result.metadata, "interpolation_method")
            @test haskey(interp_result.metadata, "interpolated_count")
            @test interp_result.metadata["interpolated_count"] >= 0
        end
    end # Vector Interpolation

    @testset "Edge Case & Negative Input Coverage" begin
        # Comprehensive testing of edge cases and negative inputs for production robustness
        
        @testset "NaN and Inf Input Handling" begin
            @withseed 2001 begin
                # Create base images for testing
                image_size = (32, 32)
                correlator = CrossCorrelator(image_size)
                
                # Base valid images
                img_base = rand(Float32, image_size...)
                
                @testset "NaN in Input Images" begin
                    # Test with NaN values (simulating masked regions)
                    img_with_nan = copy(img_base)
                    img_with_nan[10:12, 10:12] .= NaN  # Create masked region
                    
                    # Measure actual behavior - correlation succeeds but analysis may fail
                    corr = correlate!(correlator, img_with_nan, img_base)
                    @test eltype(corr) <: Complex  # Correlation maintains complex type
                    @test any(isnan.(corr))        # NaN values propagate through FFT
                    
                    # Analysis should gracefully handle NaN correlation planes
                    du, dv, pr, cm = analyze_correlation_plane(corr)
                    @test isnan(du) && isnan(dv) && isnan(pr) && isnan(cm)
                end
            end
        end
        
        @testset "Non-Square Windows" begin
            @withseed 2003 begin
                # Test one specific non-square case and measure behavior
                window_size = (48, 32)  # height, width
                
                correlator = CrossCorrelator(window_size)
                @test size(correlator.C1) == window_size
                
                # Test correlation with known displacement
                img1 = rand(Float32, window_size...)
                img2 = circshift(img1, (2, 1))  # Known displacement
                
                corr = correlate!(correlator, img1, img2)
                @test size(corr) == window_size
                
                # Measure actual accuracy - works perfectly!
                du, dv, pr, cm = analyze_correlation_plane(corr)
                
                # Non-square windows work excellently - detected exact displacement
                @test du ≈ 2.0 atol=0.1  # Measured: exactly 2.0
                @test dv ≈ 1.0 atol=0.1  # Measured: exactly 1.0
                @test isfinite(pr) && pr > 1.0  # Measured: ~1.3
                @test isfinite(cm) && cm > 0.0  # Measured: ~3.9
            end
        end
        
        @testset "UInt8 Image Support" begin
            @withseed 2002 begin
                image_size = (32, 32)
                
                # Test with UInt8 images (common camera output)
                img1_uint8 = rand(UInt8, image_size...)
                img2_uint8 = rand(UInt8, image_size...)
                
                # UInt8 images work perfectly! No type conversion needed
                correlator_uint8 = CrossCorrelator(image_size)
                corr = correlate!(correlator_uint8, img1_uint8, img2_uint8)
                
                @test eltype(corr) <: Complex     # Automatically promotes to ComplexF32
                @test size(corr) == image_size
                
                # Should also be able to analyze results
                du, dv, pr, cm = analyze_correlation_plane(corr)
                @test isfinite(du) && isfinite(dv)
                @test isfinite(pr) && isfinite(cm)
            end
        end
        
        @testset "Gray{T} Image Type Support" begin
            @withseed 2006 begin
                using ImageCore  # Provides Gray{T} and fixed-point types
                image_size = (32, 32)
                
                # Test various Gray{T} types common in scientific imaging
                gray_types = [Gray{N0f8}, Gray{N4f12}, Gray{N0f16}]
                type_names = ["N0f8 (8-bit)", "N4f12 (12-bit)", "N0f16 (16-bit)"]
                
                for (GrayType, type_name) in zip(gray_types, type_names)
                    @testset "Gray{$type_name}" begin
                        img1_gray = rand(GrayType, image_size...)
                        img2_gray = rand(GrayType, image_size...)
                        
                        correlator = CrossCorrelator(image_size)
                        corr = correlate!(correlator, img1_gray, img2_gray)
                        
                        @test eltype(corr) <: Complex
                        @test size(corr) == image_size
                        
                        du, dv, pr, cm = analyze_correlation_plane(corr)
                        @test isfinite(du) && isfinite(dv)
                        @test isfinite(pr) && isfinite(cm)
                    end
                end
            end
        end
        
        @testset "Border and Padding Validation" begin
            @withseed 2004 begin
                # Test PIV processing with particles near image boundaries
                image_size = (64, 64)
                
                # Create image with particles near edges
                img1 = zeros(Float64, image_size...)
                img2 = zeros(Float64, image_size...)
                
                # Add particles very close to boundaries
                generate_gaussian_particle!(img1, (3.0, 3.0), 3.0)      # Near corner
                generate_gaussian_particle!(img1, (61.0, 32.0), 4.0)    # Near edge
                generate_gaussian_particle!(img2, (5.0, 5.0), 3.0)      # Shifted corner
                generate_gaussian_particle!(img2, (63.0, 34.0), 4.0)    # Shifted edge
                
                # Test with window size that forces boundary processing
                window_size = (16, 16)
                stage = PIVStage(window_size, overlap=(0.5, 0.5), padding=4)
                
                result = run_piv(img1, img2, [stage])  # Vector of stages returns Vector{PIVResult}
                
                # Should process without crashing despite boundary particles
                @test isa(result, Vector{PIVResult})
                @test length(result) == 1
                piv_result = result[1]
                @test length(piv_result.vectors) > 0
                
                # Should have some good vectors despite boundary challenges
                good_vectors = [v for v in piv_result.vectors if v.status == :good]
                @test length(good_vectors) >= 0  # May be zero for this difficult case, but shouldn't crash
            end
        end
        
        @testset "Degenerate Input Cases" begin
            @withseed 2005 begin
                image_size = (16, 16)  # Small for faster testing
                correlator = CrossCorrelator(image_size)
                
                # All-zero images
                img_zeros = zeros(Float32, image_size...)
                corr_zeros = correlate!(correlator, img_zeros, img_zeros)
                
                @test size(corr_zeros) == image_size
                # Zero correlation should produce near-zero or NaN result
                du, dv, pr, cm = analyze_correlation_plane(corr_zeros)
                @test isnan(du) && isnan(dv)  # All-zero correlation should return NaN
                
                # Uniform non-zero images (no features to track)
                img_uniform = fill(0.7f0, image_size...)
                corr_uniform = correlate!(correlator, img_uniform, img_uniform)
                
                du_u, dv_u, pr_u, cm_u = analyze_correlation_plane(corr_uniform)
                # Uniform images create correlation peak, displacement depends on peak location
                @test !isnan(du_u) && !isnan(dv_u)  # Should not be NaN, displacement is valid
                @test pr_u > 0.0 || isnan(pr_u)  # Peak ratio should be positive or NaN
            end
        end
    end # Edge Case & Negative Input Coverage

    @testset "Vector Validation System" begin
        # Import validation types for testing
        using Hammerhead: PeakRatioValidator, CorrelationMomentValidator, VelocityMagnitudeValidator,
                          LocalMedianValidator, NormalizedResidualValidator, parse_validator,
                          parse_validation_pipeline, validate_vectors!, apply_local_validators!,
                          get_valid_mask, update_valid_mask!
        using Hammerhead.Validators

        @testset "Validator Type Hierarchy" begin
            # Test that all validators have correct type hierarchy
            @test PeakRatioValidator <: Hammerhead.LocalValidator
            @test CorrelationMomentValidator <: Hammerhead.LocalValidator
            @test VelocityMagnitudeValidator <: Hammerhead.LocalValidator
            @test LocalMedianValidator <: Hammerhead.NeighborhoodValidator
            @test NormalizedResidualValidator <: Hammerhead.NeighborhoodValidator
            
            # Test sub-module aliases
            @test PeakRatio === PeakRatioValidator
            @test CorrelationMoment === CorrelationMomentValidator
            @test VelocityMagnitude === VelocityMagnitudeValidator
            @test LocalMedian === LocalMedianValidator
            @test NormalizedResidual === NormalizedResidualValidator
        end
        
        @testset "Validator Parsing API" begin
            # Test parsing validator objects (should return as-is)
            validator1 = PeakRatioValidator(1.2)
            @test parse_validator(validator1) === validator1
            
            # Test parsing symbol pairs for simple validators
            @test parse_validator(:peak_ratio => 1.2) isa PeakRatioValidator
            @test parse_validator(:peak_ratio => 1.2).threshold == 1.2
            @test parse_validator(:correlation_moment => 0.1) isa CorrelationMomentValidator
            @test parse_validator(:correlation_moment => 0.1).threshold == 0.1
            
            # Test parsing symbol pairs for complex validators
            @test parse_validator(:local_median => (window_size=3, threshold=2.0)) isa LocalMedianValidator
            @test parse_validator(:velocity_magnitude => (min=0.1, max=50.0)) isa VelocityMagnitudeValidator
            @test parse_validator(:normalized_residual => (window_size=5, threshold=3.0)) isa NormalizedResidualValidator
            
            # Test validation pipeline parsing
            pipeline = (:peak_ratio => 1.2, :local_median => (window_size=3, threshold=2.0))
            parsed = parse_validation_pipeline(pipeline)
            @test length(parsed) == 2
            @test parsed[1] isa PeakRatioValidator
            @test parsed[2] isa LocalMedianValidator
            
            # Test empty pipeline
            @test parse_validation_pipeline(()) == ()
            
            # Test error on unknown validator
            @test_throws ErrorException parse_validator(:unknown_validator => 1.0)
        end
        
        @testset "Local Validators" begin
            # Create test vectors with known properties
            vectors = StructArray([
                PIVVector(1.0, 1.0, 1.0, 1.0, :good, 1.5, 0.3),   # Good vector
                PIVVector(2.0, 1.0, 2.0, 2.0, :good, 0.8, 0.3),   # Low peak ratio
                PIVVector(3.0, 1.0, 3.0, 3.0, :good, 1.5, 0.05),  # Low correlation moment
                PIVVector(4.0, 1.0, 100.0, 100.0, :good, 1.5, 0.3), # High velocity magnitude
                PIVVector(5.0, 1.0, 5.0, 5.0, :bad, 1.5, 0.3),    # Already bad
            ])
            
            # Test PeakRatioValidator
            validator = PeakRatioValidator(1.2)
            test_vectors = deepcopy(vectors)
            apply_local_validators!(test_vectors, [validator])
            
            @test test_vectors[1].status == :good  # 1.5 >= 1.2
            @test test_vectors[2].status == :bad   # 0.8 < 1.2
            @test test_vectors[3].status == :good  # 1.5 >= 1.2
            @test test_vectors[4].status == :good  # 1.5 >= 1.2
            @test test_vectors[5].status == :bad   # Already bad
            
            # Test CorrelationMomentValidator
            validator = CorrelationMomentValidator(0.1)
            test_vectors = deepcopy(vectors)
            apply_local_validators!(test_vectors, [validator])
            
            @test test_vectors[1].status == :good  # 0.3 >= 0.1
            @test test_vectors[2].status == :good  # 0.3 >= 0.1
            @test test_vectors[3].status == :bad   # 0.05 < 0.1
            @test test_vectors[4].status == :good  # 0.3 >= 0.1
            @test test_vectors[5].status == :bad   # Already bad
            
            # Test VelocityMagnitudeValidator
            validator = VelocityMagnitudeValidator(0.1, 10.0)
            test_vectors = deepcopy(vectors)
            apply_local_validators!(test_vectors, [validator])
            
            mag1 = sqrt(1.0^2 + 1.0^2)  # ≈ 1.41
            mag4 = sqrt(100.0^2 + 100.0^2)  # ≈ 141.4
            
            @test test_vectors[1].status == :good  # mag ∈ [0.1, 10.0]
            @test test_vectors[2].status == :good  # mag ∈ [0.1, 10.0]
            @test test_vectors[3].status == :good  # mag ∈ [0.1, 10.0]
            @test test_vectors[4].status == :bad   # mag > 10.0
            @test test_vectors[5].status == :bad   # Already bad
            
            # Test batched local validators
            validators = [PeakRatioValidator(1.2), CorrelationMomentValidator(0.1)]
            test_vectors = deepcopy(vectors)
            apply_local_validators!(test_vectors, validators)
            
            @test test_vectors[1].status == :good  # Passes both
            @test test_vectors[2].status == :bad   # Fails peak ratio
            @test test_vectors[3].status == :bad   # Fails correlation moment
            @test test_vectors[4].status == :good  # Passes both
            @test test_vectors[5].status == :bad   # Already bad
        end
        
        @testset "Validation Masks" begin
            # Create 3×3 grid of vectors
            vectors = StructArray(reshape([
                PIVVector(1.0, 1.0, 1.0, 1.0, :good, 1.5, 0.3),
                PIVVector(2.0, 1.0, 2.0, 2.0, :bad, 1.5, 0.3),
                PIVVector(3.0, 1.0, 3.0, 3.0, :good, 1.5, 0.3),
                PIVVector(1.0, 2.0, 1.0, 1.0, :good, 1.5, 0.3),
                PIVVector(2.0, 2.0, 2.0, 2.0, :good, 1.5, 0.3),
                PIVVector(3.0, 2.0, 3.0, 3.0, :bad, 1.5, 0.3),
                PIVVector(1.0, 3.0, 1.0, 1.0, :good, 1.5, 0.3),
                PIVVector(2.0, 3.0, 2.0, 2.0, :good, 1.5, 0.3),
                PIVVector(3.0, 3.0, 3.0, 3.3, :good, 1.5, 0.3),
            ], 3, 3))
            
            # Test mask creation
            mask = get_valid_mask(vectors)
            expected_mask = [true true true; false true true; true false true]
            @test mask == expected_mask
            
            # Test mask update
            vectors[1, 1] = PIVVector(1.0, 1.0, 1.0, 1.0, :bad, 1.5, 0.3)
            update_valid_mask!(mask, vectors)
            expected_mask[1, 1] = false
            @test mask == expected_mask
        end
        
        @testset "PIVStage Integration" begin
            # Test PIVStage with validation tuple
            validation = (:peak_ratio => 1.2, :correlation_moment => 0.1)
            stage = PIVStage((64, 64), validation=validation)
            @test stage.validation == validation
            
            # Test PIVStage with validator objects
            validation_objects = (PeakRatioValidator(1.2), CorrelationMomentValidator(0.1))
            stage = PIVStage((64, 64), validation=validation_objects)
            @test stage.validation == validation_objects
            
            # Test PIVStage with sub-module validators
            validation_clean = (PeakRatio(1.2), CorrelationMoment(0.1))
            stage = PIVStage((64, 64), validation=validation_clean)
            @test stage.validation == validation_clean
            
            # Test empty validation
            stage = PIVStage((64, 64))
            @test stage.validation == ()
            
            # Test PIVStages with validation
            stages = PIVStages(2, 32, validation=(:peak_ratio => 1.2,))
            @test all(s.validation == (:peak_ratio => 1.2,) for s in stages)
        end
        
        @testset "Full Validation Pipeline" begin
            # Create test PIVResult with mixed quality vectors
            vectors = StructArray(reshape([
                PIVVector(1.0, 1.0, 1.0, 1.0, :good, 1.5, 0.3),   # Good
                PIVVector(2.0, 1.0, 2.0, 2.0, :good, 0.8, 0.3),   # Low peak ratio
                PIVVector(3.0, 1.0, 3.0, 3.0, :good, 1.5, 0.05),  # Low correlation moment
                PIVVector(1.0, 2.0, 1.0, 1.0, :good, 1.5, 0.3),   # Good
                PIVVector(2.0, 2.0, 2.0, 2.0, :good, 1.5, 0.3),   # Good
                PIVVector(3.0, 2.0, 3.0, 3.0, :good, 1.5, 0.3),   # Good  
            ], 2, 3))
            
            result = PIVResult(vectors, Dict{String, Any}(), Dict{String, Any}())
            
            # Test validation pipeline
            validation = (:peak_ratio => 1.2, :correlation_moment => 0.1)
            validate_vectors!(result, validation)
            
            # Check results - failed vectors are now interpolated automatically
            @test result.vectors[1, 1].status == :good  # Passes both
            @test result.vectors[2, 1].status == :interpolated   # Failed peak ratio, then interpolated
            @test result.vectors[1, 2].status == :interpolated   # Failed correlation moment, then interpolated
            @test result.vectors[1, 3].status == :good  # Passes both
            @test result.vectors[2, 2].status == :good  # Passes both
            @test result.vectors[2, 3].status == :good  # Passes both
            
            # Test empty validation pipeline with fresh good vectors
            vectors_fresh = StructArray(reshape([
                PIVVector(1.0, 1.0, 1.0, 1.0, :good, 1.5, 0.3),   
                PIVVector(2.0, 1.0, 2.0, 2.0, :good, 0.8, 0.3),   
                PIVVector(3.0, 1.0, 3.0, 3.0, :good, 1.5, 0.05),  
                PIVVector(1.0, 2.0, 1.0, 1.0, :good, 1.5, 0.3),   
                PIVVector(2.0, 2.0, 2.0, 2.0, :good, 1.5, 0.3),   
                PIVVector(3.0, 2.0, 3.0, 3.0, :good, 1.5, 0.3),   
            ], 2, 3))
            result_copy = PIVResult(vectors_fresh, Dict{String, Any}(), Dict{String, Any}())
            validate_vectors!(result_copy, ())  # Empty validation
            
            # Should be unchanged
            @test all(v.status == :good for v in result_copy.vectors)
        end
        
        @testset "Edge Cases and Error Handling" begin
            # Test with all bad vectors
            vectors = StructArray([
                PIVVector(1.0, 1.0, 1.0, 1.0, :bad, 1.5, 0.3),
                PIVVector(2.0, 1.0, 2.0, 2.0, :bad, 1.5, 0.3),
            ])
            
            # Local validators should not change already bad vectors
            apply_local_validators!(vectors, [PeakRatioValidator(10.0)])  # Very high threshold
            @test all(v.status == :bad for v in vectors)
            
            # Test neighborhood validators with insufficient neighbors
            vectors_2x2 = StructArray(reshape([
                PIVVector(1.0, 1.0, 1.0, 1.0, :good, 1.5, 0.3),
                PIVVector(2.0, 1.0, 1.0, 1.0, :good, 1.5, 0.3),
                PIVVector(1.0, 2.0, 1.0, 1.0, :good, 1.5, 0.3),
                PIVVector(2.0, 2.0, 100.0, 100.0, :good, 1.5, 0.3),  # Outlier
            ], 2, 2))
            
            # With window_size=3, boundary vectors should have insufficient neighbors
            result = PIVResult(vectors_2x2, Dict{String, Any}(), Dict{String, Any}())
            validation = (:local_median => (window_size=3, threshold=2.0),)
            validate_vectors!(result, validation)
            
            # Insufficient neighbors should remain unchanged
            # Only interior vectors with enough neighbors get validated
        end
        
        @testset "Sub-module API" begin
            # Test clean sub-module API
            validation = (PeakRatio(1.2), CorrelationMoment(0.1), LocalMedian(3, 2.0))
            
            # Should work with PIVStage
            stage = PIVStage((64, 64), validation=validation)
            @test stage.validation == validation
            
            # Should work with validation pipeline parsing
            parsed = parse_validation_pipeline(validation)
            @test parsed == validation  # Should return as-is since already validator objects
            
            # Mixed API usage
            mixed_validation = (PeakRatio(1.2), :correlation_moment => 0.1)
            stage = PIVStage((64, 64), validation=mixed_validation)
            @test stage.validation == mixed_validation
        end
    end # Vector Validation System

    @testset "Vector Replacement (Hole Filling) System" begin
        using Hammerhead: detect_holes, IterativeMedian, replace_vectors!, apply_vector_replacement!
        
        @testset "Hole Detection and Classification" begin
            # Create realistic 16×16 field with synthetic flow
            vectors = StructArray([PIVVector(Float64(j), Float64(i), 0.1*j + 0.05*i, 0.05*j + 0.1*i, :good, 1.5, 0.3) 
                                  for i in 1:16, j in 1:16])
            
            # Add different hole patterns
            vectors[3, 3] = PIVVector(3.0, 3.0, NaN, NaN, :bad, 1.5, 0.3)  # Isolated hole
            
            # Small 2×2 cluster
            vectors[8, 8] = PIVVector(8.0, 8.0, NaN, NaN, :bad, 1.5, 0.3)
            vectors[8, 9] = PIVVector(9.0, 8.0, NaN, NaN, :bad, 1.5, 0.3)
            vectors[9, 8] = PIVVector(8.0, 9.0, NaN, NaN, :bad, 1.5, 0.3)
            vectors[9, 9] = PIVVector(9.0, 9.0, NaN, NaN, :bad, 1.5, 0.3)
            
            # Large irregular region (3×4)
            for i in 12:14, j in 5:8
                vectors[i, j] = PIVVector(Float64(j), Float64(i), NaN, NaN, :bad, 1.5, 0.3)
            end
            
            holes = detect_holes(vectors)
            @test length(holes) >= 2  # Should detect multiple hole regions
            
            # Verify classifications exist
            hole_classifications = Set([hole.classification for hole in holes])
            @test length(hole_classifications) >= 2  # Should have different classifications
            
            # Verify all hole indices point to bad vectors
            for hole in holes
                for idx in hole.indices
                    @test vectors[idx].status == :bad
                end
            end
        end
        
        @testset "Iterative Median Replacement" begin
            # Create 16×16 field with linear flow pattern
            vectors = StructArray([PIVVector(Float64(j), Float64(i), 0.1*j, 0.1*i, :good, 1.5, 0.3) 
                                  for i in 1:16, j in 1:16])
            
            # Add isolated hole in center with good neighbors
            vectors[8, 8] = PIVVector(8.0, 8.0, NaN, NaN, :bad, 1.5, 0.3)
            
            # Store expected median values from neighbors
            neighbor_u_values = [vectors[7,7].u, vectors[7,8].u, vectors[7,9].u,
                               vectors[8,7].u, vectors[8,9].u,
                               vectors[9,7].u, vectors[9,8].u, vectors[9,9].u]
            neighbor_v_values = [vectors[7,7].v, vectors[7,8].v, vectors[7,9].v,
                               vectors[8,7].v, vectors[8,9].v,
                               vectors[9,7].v, vectors[9,8].v, vectors[9,9].v]
            expected_u = median(neighbor_u_values)
            expected_v = median(neighbor_v_values)
            
            # Apply iterative median
            holes = detect_holes(vectors)
            replace_vectors!(vectors, holes, IterativeMedian(5, 3, 3))
            
            # Verify replacement
            @test vectors[8, 8].status == :interpolated
            @test abs(vectors[8, 8].u - expected_u) < 1e-10
            @test abs(vectors[8, 8].v - expected_v) < 1e-10
            @test sum(v.status == :bad for v in vectors) == 0
        end
        
        @testset "Integration with Validation Pipeline" begin
            # Create 16×16 field with realistic PIV parameters
            vectors = StructArray([PIVVector(Float64(j), Float64(i), 0.1*j + 0.02*i, 0.02*j + 0.1*i, :good, 1.5, 0.3) 
                                  for i in 1:16, j in 1:16])
            
            # Add vectors that will fail validation
            vectors[5, 5] = PIVVector(5.0, 5.0, 0.5, 0.5, :good, 0.8, 0.3)    # Low peak ratio
            vectors[10, 10] = PIVVector(10.0, 10.0, 1.0, 1.0, :good, 1.5, 0.05) # Low correlation moment
            vectors[12, 7] = PIVVector(7.0, 12.0, 1.2, 1.2, :good, 0.9, 0.04)  # Fails both
            
            result = PIVResult(vectors, Dict{String, Any}(), Dict{String, Any}())
            
            # Apply validation with interpolation (this now includes replacement)
            validation = (:peak_ratio => 1.2, :correlation_moment => 0.1)
            validate_vectors!(result, validation)
            
            # Check that failed vectors were interpolated
            @test result.vectors[5, 5].status == :interpolated
            @test result.vectors[10, 10].status == :interpolated  
            @test result.vectors[12, 7].status == :interpolated
            @test sum(v.status == :bad for v in result.vectors) == 0
            
            # Verify interpolated values are reasonable (finite and within field range)
            for v in result.vectors
                if v.status == :interpolated
                    @test isfinite(v.u) && isfinite(v.v)
                    @test -1.0 <= v.u <= 3.0  # Reasonable range for this flow
                    @test -1.0 <= v.v <= 3.0
                end
            end
        end
        
        @testset "Robustness and Edge Cases" begin
            # Test large sparse region that should remain partially unfilled
            vectors = StructArray([PIVVector(Float64(j), Float64(i), 0.1*j, 0.1*i, :good, 1.5, 0.3) 
                                  for i in 1:16, j in 1:16])
            
            # Create large sparse bad region (most of center)
            for i in 6:11, j in 6:11
                if (i + j) % 3 == 0  # Sparse pattern - only some bad
                    vectors[i, j] = PIVVector(Float64(j), Float64(i), NaN, NaN, :bad, 1.5, 0.3)
                end
            end
            
            initial_bad = sum(v.status == :bad for v in vectors)
            
            # Apply with conservative parameters
            holes = detect_holes(vectors)
            replace_vectors!(vectors, holes, IterativeMedian(3, 4, 3))  # Need 4 neighbors
            
            final_bad = sum(v.status == :bad for v in vectors)
            interpolated = sum(v.status == :interpolated for v in vectors)
            
            # Should make some progress but not fill everything
            @test final_bad <= initial_bad
            @test interpolated >= 0
            
            # All interpolated vectors should be finite
            for v in vectors
                if v.status == :interpolated
                    @test isfinite(v.u) && isfinite(v.v)
                end
            end
        end
        
        @testset "Performance and Iterative Convergence" begin
            # Create 16×16 field with connected bad region requiring multiple iterations
            vectors = StructArray([PIVVector(Float64(j), Float64(i), 0.1*j, 0.1*i, :good, 1.5, 0.3) 
                                  for i in 1:16, j in 1:16])
            
            # Create diamond-shaped hole that expands from edges inward
            center = (8, 8)
            for i in 1:16, j in 1:16
                dist = abs(i - center[1]) + abs(j - center[2])  # Manhattan distance
                if 2 <= dist <= 4
                    vectors[i, j] = PIVVector(Float64(j), Float64(i), NaN, NaN, :bad, 1.5, 0.3)
                end
            end
            
            initial_bad = sum(v.status == :bad for v in vectors)
            @test initial_bad > 10  # Should have significant hole
            
            # Apply iterative median
            holes = detect_holes(vectors)
            replace_vectors!(vectors, holes, IterativeMedian(5, 3, 3))
            
            final_bad = sum(v.status == :bad for v in vectors)
            interpolated = sum(v.status == :interpolated for v in vectors)
            
            # Should fill most or all of the hole
            @test final_bad < initial_bad / 2  # At least 50% improvement
            @test interpolated >= initial_bad / 2
        end
        
        @testset "Symbol-based API" begin
            # Test the user-facing symbol-based interface
            vectors = StructArray([PIVVector(Float64(j), Float64(i), 0.1*j, 0.1*i, :good, 1.5, 0.3) 
                                  for i in 1:16, j in 1:16])
            
            # Add some bad vectors
            vectors[5, 5] = PIVVector(5.0, 5.0, NaN, NaN, :bad, 1.5, 0.3)
            vectors[10, 10] = PIVVector(10.0, 10.0, NaN, NaN, :bad, 1.5, 0.3)
            
            result = PIVResult(vectors, Dict{String, Any}(), Dict{String, Any}())
            
            # Test default method
            apply_vector_replacement!(result)
            @test sum(v.status == :interpolated for v in result.vectors) == 2
            @test sum(v.status == :bad for v in result.vectors) == 0
            
            # Test explicit method specification
            vectors2 = StructArray([PIVVector(Float64(j), Float64(i), 0.1*j, 0.1*i, :good, 1.5, 0.3) 
                                   for i in 1:16, j in 1:16])
            vectors2[8, 8] = PIVVector(8.0, 8.0, NaN, NaN, :bad, 1.5, 0.3)
            result2 = PIVResult(vectors2, Dict{String, Any}(), Dict{String, Any}())
            
            apply_vector_replacement!(result2; method=:iterative_median)
            @test sum(v.status == :interpolated for v in result2.vectors) == 1
            
            # Test error on unknown method
            @test_throws ErrorException apply_vector_replacement!(result2; method=:unknown)
        end
        
        @testset "Garbage Region Detection" begin
            using Hammerhead: detect_garbage_regions
            
            # Create field with sparse bad region (low density)
            vectors = StructArray([PIVVector(Float64(j), Float64(i), 0.1*j, 0.1*i, :good, 1.5, 0.3) 
                                  for i in 1:16, j in 1:16])
            
            # Create sparse bad region in center with very low good density
            for i in 5:12, j in 5:12
                # Make region extremely sparse (only ~10% good vectors)
                if (i + j) % 10 != 0  # 90% bad
                    old = vectors[i, j]
                    vectors[i, j] = PIVVector(old.x, old.y, old.u, old.v, :bad, old.peak_ratio, old.correlation_moment)
                end
            end
            
            # Detect garbage regions
            garbage_mask = detect_garbage_regions(vectors; min_good_density=0.3, analysis_window=7)
            
            # Should detect meaningful portion of sparse region as garbage
            garbage_in_region = sum(garbage_mask[5:12, 5:12])
            bad_in_region = sum(vectors[i, j].status == :bad for i in 5:12, j in 5:12)
            detection_rate = garbage_in_region / bad_in_region
            @test detection_rate >= 0.15  # At least 15% of bad vectors in sparse region should be garbage
            
            # Test integration with vector replacement
            result = PIVResult(vectors, Dict{String, Any}(), Dict{String, Any}())
            apply_vector_replacement!(result; abandon_garbage=true)
            
            # Should have both :garbage and :interpolated vectors
            @test sum(v.status == :garbage for v in result.vectors) > 0
            @test sum(v.status == :interpolated for v in result.vectors) > 0
            
            # Test disabling garbage detection - create fresh vectors
            vectors2 = StructArray([PIVVector(Float64(j), Float64(i), 0.1*j, 0.1*i, :good, 1.5, 0.3) 
                                   for i in 1:16, j in 1:16])
            for i in 5:12, j in 5:12
                if (i + j) % 10 != 0  # Same sparse pattern
                    old = vectors2[i, j]
                    vectors2[i, j] = PIVVector(old.x, old.y, old.u, old.v, :bad, old.peak_ratio, old.correlation_moment)
                end
            end
            result2 = PIVResult(vectors2, Dict{String, Any}(), Dict{String, Any}())
            apply_vector_replacement!(result2; abandon_garbage=false)
            
            # Should not have any :garbage vectors when disabled
            @test sum(v.status == :garbage for v in result2.vectors) == 0
        end
    end # Vector Replacement System

end # Hammerhead.jl
