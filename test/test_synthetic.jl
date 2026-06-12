using Hammerhead
using Hammerhead.SyntheticData
using Test
using Random
using Statistics

@testset "SyntheticData" begin
    @testset "Particle Field Generation" begin
        laser = GaussianLaserSheet(0.0, 1.0, 1.0)
        
        # Test basic generation
        rng = MersenneTwister(1234)
        particles = generate_particle_field((64, 64), 0.05, rng=rng)
        @test particles isa ParticleField3D
        @test length(particles.x) > 0
        
        # Test rendering
        img = render_particle_image(particles, (64, 64), laser, background_noise=0.0)
        @test size(img) == (64, 64)
        @test all(img .>= 0)
        @test sum(img) > 0
        
        # Test reproducibility
        particles1 = generate_particle_field((32, 32), 0.05, rng=MersenneTwister(3333))
        img1 = render_particle_image(particles1, (32, 32), laser, background_noise=0.0)
        particles2 = generate_particle_field((32, 32), 0.05, rng=MersenneTwister(3333))
        img2 = render_particle_image(particles2, (32, 32), laser, background_noise=0.0)
        @test img1 ≈ img2
    end

    @testset "Displacement Application" begin
        displacement = (2.5, 1.8)
        velocity_func = linear_flow(displacement[1], displacement[2], 0.0, 0.0, 0.0, 0.0, 0.0)
        
        rng = MersenneTwister(4444)
        img1, img2, p1, p2 = generate_synthetic_piv_pair(velocity_func, (64, 64), 1.0,
                                                       particle_density=0.02,
                                                       background_noise=0.0,
                                                       rng=rng)
        @test size(img1) == size(img2) == (64, 64)
        
        # Zero displacement
        zero_velocity = linear_flow(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        img1_z, img2_z, _, _ = generate_synthetic_piv_pair(zero_velocity, (64, 64), 1.0,
                                                         particle_density=0.02,
                                                         background_noise=0.0,
                                                         rng=MersenneTwister(4444))
        @test img1_z ≈ img2_z atol=1e-10
    end

    @testset "PIV Accuracy Validation" begin
        # Test PIV accuracy with realistic particle fields
        rng = MersenneTwister(7777)
        true_du, true_dv = 1.2, 0.8
        velocity_func = linear_flow(true_du, true_dv, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        img1, img2, _, _ = generate_synthetic_piv_pair(velocity_func, (128, 128), 1.0,
                                                     particle_density=0.05,
                                                     diameter_mean=3.0, diameter_std=0.3,
                                                     background_noise=0.01,
                                                     rng=rng)
        
        params = PIVParameters(window_size=32, overlap=16)
        result = run_piv(img1, img2, params)
        
        # Filter out outliers (there shouldn't be many)
        valid_u = result.u[.!result.outliers]
        valid_v = result.v[.!result.outliers]
        
        @test length(valid_u) > 0
        @test mean(valid_u) ≈ true_du atol=0.1
        @test mean(valid_v) ≈ true_dv atol=0.1
        @test std(valid_u) < 0.1
        @test std(valid_v) < 0.1
    end
end
