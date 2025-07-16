using Hammerhead
using Test
using Random
using LinearAlgebra
using FFTW

function generate_gaussian_particle!(array::AbstractArray, centroid::Tuple{Float64, Float64}, diameter::Float64)
    sigma = diameter / 2.0  # Approximation: diameter covers ~1 standard deviation
    for i in axes(array, 1), j in axes(array, 2)
        x, y = i - centroid[1], j - centroid[2]
        v = exp(-0.5 * (x^2 + y^2) / sigma^2)
        v < 1e-10 && continue  # Avoid numerical issues
        array[i, j] += exp(-0.5 * (x^2 + y^2) / sigma^2)
    end
end

@testset "Hammerhead.jl" begin
    @testset "Synthetic Data Generation" begin
        @testset "Gaussian Particle Generation" begin
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
        end
    end
    
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
        end
        
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
        end
        
        @testset "Show Method" begin
            cc = CrossCorrelator((64, 64))
            str = string(cc)
            @test occursin("CrossCorrelator", str)
            @test occursin("64", str)  # Size appears in string
            @test occursin("Float32", str) || occursin("ComplexF32", str)  # Type info appears
        end
    end
    
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
            @test refined[1] ≈ center[1] atol=0.1
            @test refined[2] ≈ center[2] atol=0.1
        end
        
        @testset "subpixel_gauss3 Edge Cases" begin
            # Test peak at edge (should return zero offset)
            corr = zeros(Float64, 10, 10)
            corr[1, 5] = 1.0  # Peak at edge
            
            refined = subpixel_gauss3(corr, (1, 5))
            @test refined[1] == 1.0  # No x-refinement possible at edge
            
            # Test peak at corner
            corr2 = zeros(Float64, 10, 10)
            corr2[1, 1] = 1.0
            
            refined2 = subpixel_gauss3(corr2, (1, 1))
            @test refined2 == (1.0, 1.0)  # No refinement possible at corner
        end
        
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
            @test refined_noisy[1] ≈ 8.0 atol=0.5  # Allow for noise effects
            @test refined_noisy[2] ≈ 8.0 atol=0.5
        end
    end
end

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
    displacement = correlate(correlator, img1, img2)
    @test all(displacement .≈ (2.2, 1.3))
end

# Additional comprehensive tests for existing functionality
@testset "Correlation Algorithm Tests" begin
    @testset "CrossCorrelator with Various Displacements" begin
        image_size = (64, 64)
        centroid = (32.0, 32.0)
        diameter = 3.0
        correlator = CrossCorrelator(image_size)
        
        # Test multiple displacement values
        test_displacements = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.5, 2.3), (3.7, -2.1)]
        
        for true_disp in test_displacements
            img1 = zeros(Float64, image_size)
            img2 = zeros(Float64, image_size)
            
            generate_gaussian_particle!(img1, centroid, diameter)
            generate_gaussian_particle!(img2, centroid .+ true_disp, diameter)
            
            measured_disp = correlate(correlator, img1, img2)
            
            @test measured_disp[1] ≈ true_disp[1] atol=0.15
            @test measured_disp[2] ≈ true_disp[2] atol=0.15
        end
    end
    
    @testset "CrossCorrelator with Different Image Types" begin
        image_size = (32, 32)
        correlator = CrossCorrelator(image_size)
        
        # Test with Float32 images
        img1_f32 = rand(Float32, image_size...)
        img2_f32 = circshift(img1_f32, (2, 1))
        
        disp_f32 = correlate(correlator, img1_f32, img2_f32)
        @test abs(disp_f32[1]) ≈ 2.0 atol=0.3  # Should detect the shift
        @test abs(disp_f32[2]) ≈ 1.0 atol=0.3
        
        # Test with Float64 images
        img1_f64 = rand(Float64, image_size...)
        img2_f64 = circshift(img1_f64, (1, 3))
        
        disp_f64 = correlate(correlator, img1_f64, img2_f64)
        @test abs(disp_f64[1]) ≈ 1.0 atol=0.3
        @test abs(disp_f64[2]) ≈ 3.0 atol=0.3
    end
    
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
            
            disp = correlate(correlator, img1, img2)
            
            # Verify memory wasn't reallocated
            @test pointer(correlator.C1) == c1_ptr
            @test pointer(correlator.C2) == c2_ptr
            
            # Verify result is reasonable
            @test isa(disp, Tuple{Float64, Float64})
            @test isfinite(disp[1])
            @test isfinite(disp[2])
        end
    end
    
    @testset "CrossCorrelator Error Handling" begin
        correlator = CrossCorrelator((32, 32))
        
        # Test with wrong size images
        img_wrong_size = zeros(16, 16)
        img_correct = zeros(32, 32)
        
        @test_throws BoundsError correlate(correlator, img_wrong_size, img_correct)
        @test_throws BoundsError correlate(correlator, img_correct, img_wrong_size)
    end
end

@testset "Performance Tests" begin
    @testset "CrossCorrelator Performance" begin
        # Test performance with larger images
        image_size = (128, 128)  # Reasonable size for CI
        correlator = CrossCorrelator(image_size)
        
        img1 = rand(Float32, image_size...)
        img2 = rand(Float32, image_size...)
        
        # Measure time for correlation
        elapsed = @elapsed begin
            for i in 1:5
                correlate(correlator, img1, img2)
            end
        end
        
        # Should be reasonably fast
        @test elapsed < 2.0
        
        # Test memory allocation efficiency
        # First run to compile
        correlate(correlator, img1, img2)
        
        # Measure allocations on subsequent run
        allocs = @allocated correlate(correlator, img1, img2)
        # Should have minimal allocations after compilation
        @test allocs < 5000  # bytes - allow some headroom for different Julia versions
    end
end
