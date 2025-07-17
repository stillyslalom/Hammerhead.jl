using Hammerhead
using Test
using Random
using LinearAlgebra
using FFTW
using StructArrays
using DSP

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
        end # Gaussian Particle Generation
    end # Synthetic Data Generation
    
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
            corr[1, 5] = 1.0  # Peak at edge
            
            refined = subpixel_gauss3(corr, (1, 5))
            @test refined[1] == 1.0  # No x-refinement possible at edge
            
            # Test peak at corner
            corr2 = zeros(Float64, 10, 10)
            corr2[1, 1] = 1.0
            
            refined2 = subpixel_gauss3(corr2, (1, 1))
            @test refined2 == (1.0, 1.0)  # No refinement possible at corner
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
            @test refined_noisy[1] ≈ 8.0 atol=0.5  # Allow for noise effects
            @test refined_noisy[2] ≈ 8.0 atol=0.5
        end # subpixel_gauss3 Numerical Stability
    end # Subpixel Refinement

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
            
            # Test tuple overlap - when tuple has 2 elements for 2 stages, each stage gets one value
            stages5 = PIVStages(2, 32, overlap=(0.6, 0.4))
            @test stages5[1].overlap == (0.6, 0.6)  # First stage gets 0.6 as symmetric overlap
            @test stages5[2].overlap == (0.4, 0.4)  # Second stage gets 0.4 as symmetric overlap
            
            # Test same asymmetric overlap for all stages - use vector with single tuple element
            stages5b = PIVStages(2, 32, overlap=[(0.6, 0.4)])
            @test all(s -> s.overlap == (0.6, 0.4), stages5b)
            
            # Test tuple parameters
            stages6 = PIVStages(3, 32, overlap=(0.75, 0.5, 0.25), deformation_iterations=(1, 3, 5))
            @test stages6[1].overlap == (0.75, 0.75)
            @test stages6[2].overlap == (0.5, 0.5) 
            @test stages6[3].overlap == (0.25, 0.25)
            @test stages6[1].deformation_iterations == 1
            @test stages6[2].deformation_iterations == 3
            @test stages6[3].deformation_iterations == 5
            
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
        @test disp_u ≈ 2.2 atol=1e-5
        @test disp_v ≈ 1.3 atol=1e-5
        @test peak_ratio > 1.0  # Should have good peak ratio
        @test isfinite(corr_moment)  # Should have finite correlation moment
        end # CrossCorrelator with Gaussian Particle
    end # Integration Tests

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
                
                correlation_plane = correlate!(correlator, img1, img2)
                disp_u, disp_v, peak_ratio, corr_moment = analyze_correlation_plane(correlation_plane)
                
                @test disp_u ≈ true_disp[1] atol=2e-6
                @test disp_v ≈ true_disp[2] atol=2e-6
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
            @test abs(disp_u_f32) ≈ 2.0 atol=0.3  # Should detect the shift
            @test abs(disp_v_f32) ≈ 1.0 atol=0.3
            
            # Test with Float64 images
            img1_f64 = rand(Float64, image_size...)
            img2_f64 = circshift(img1_f64, (1, 3))
            
            correlation_plane_f64 = correlate!(correlator, img1_f64, img2_f64)
            disp_u_f64, disp_v_f64, peak_ratio_f64, corr_moment_f64 = analyze_correlation_plane(correlation_plane_f64)
            @test abs(disp_u_f64) ≈ 1.0 atol=0.3
            @test abs(disp_v_f64) ≈ 3.0 atol=0.3
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
        @testset "CrossCorrelator Performance" begin
            # Test performance with larger images
            image_size = (128, 128)  # Reasonable size for CI
            correlator = CrossCorrelator(image_size)
            
            img1 = rand(Float32, image_size...)
            img2 = rand(Float32, image_size...)
            
            # Measure time for correlation
            elapsed = @elapsed begin
                for i in 1:5
                    correlate!(correlator, img1, img2)
                end
            end
            
            # Should be reasonably fast
            @test elapsed < 2.0
            
            # Test memory allocation efficiency
            # First run to compile
            correlate!(correlator, img1, img2)
            
            # Measure allocations on subsequent run
            allocs = @allocated correlate!(correlator, img1, img2)
            # Should have minimal allocations (no copy, just returns view to correlation plane)
            @test allocs < 5000  # bytes - back to original threshold since no copy
        end # CrossCorrelator Performance
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
            result = run_piv_stage(img1, img2, stage, CrossCorrelator)
            
            @test isa(result, PIVResult)
            @test length(result.vectors) > 0
            # All vectors should be processed (no :bad status from padding issues)
            @test all(v -> v.status != :bad, result.vectors)
            
            # Test boundary case with smaller image
            small_img1 = rand(Float64, (32, 32)...)
            small_img2 = rand(Float64, (32, 32)...)
            stage_small = PIVStage((24, 24), overlap=(0.5, 0.5))
            
            result_small = run_piv_stage(small_img1, small_img2, stage_small, CrossCorrelator)
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
            @test maximum(hann_window) ≈ 0.95 atol=0.05  # Peak value for 8-point Hanning
            
            # Test Hamming window properties
            hamm_window = Hammerhead.generate_window_1d(Hammerhead.SimpleWindow(DSP.hamming), 8)
            @test length(hamm_window) == 8
            @test hamm_window[1] ≈ 0.08 atol=1e-2  # Hamming doesn't go to zero
            @test hamm_window[end] ≈ 0.08 atol=1e-2
            @test maximum(hamm_window) ≈ 0.98 atol=0.05  # Peak value for 8-point Hamming
            
            # Test Blackman window properties
            blackman_window = Hammerhead.generate_window_1d(Hammerhead.SimpleWindow(DSP.blackman), 8)
            @test length(blackman_window) == 8
            @test blackman_window[1] ≈ 0.0 atol=1e-10  # Should start near 0
            @test blackman_window[end] ≈ 0.0 atol=1e-10  # Should end near 0
            @test maximum(blackman_window) ≈ 0.92 atol=0.05  # Peak value for 8-point Blackman
            
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
            windowed_rect = Hammerhead.apply_window_function(test_img, Hammerhead.SimpleWindow(DSP.rect))
            @test windowed_rect ≈ test_img
            
            # Test Hanning windowing
            windowed_hann = Hammerhead.apply_window_function(test_img, Hammerhead.SimpleWindow(DSP.hanning))
            @test size(windowed_hann) == size(test_img)
            @test windowed_hann[1, 1] ≈ 0.0 atol=1e-10  # Corners should be near zero
            @test windowed_hann[end, end] ≈ 0.0 atol=1e-10
            @test windowed_hann[4, 4] ≈ 0.903 atol=1e-2  # Center value for 8x8 Hanning
            
            # Test Hamming windowing
            windowed_hamm = Hammerhead.apply_window_function(test_img, Hammerhead.SimpleWindow(DSP.hamming))
            @test size(windowed_hamm) == size(test_img)
            # Hamming doesn't go to zero at edges
            @test windowed_hamm[1, 1] > 0.0
            @test windowed_hamm[end, end] > 0.0
            @test windowed_hamm[4, 4] ≈ 0.91 atol=1e-2  # Center value for 8x8 Hamming
            
            # Test Blackman windowing
            windowed_blackman = Hammerhead.apply_window_function(test_img, Hammerhead.SimpleWindow(DSP.blackman))
            @test size(windowed_blackman) == size(test_img)
            @test windowed_blackman[1, 1] ≈ 0.0 atol=1e-10
            @test windowed_blackman[end, end] ≈ 0.0 atol=1e-10
            @test windowed_blackman[4, 4] ≈ 0.85 atol=1e-2  # Center value for 8x8 Blackman
            
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
                result = run_piv_stage(img1, img2, stage, CrossCorrelator)
                
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
                        # Non-rectangular windows reduce energy, may affect precision
                        @test isfinite(best_vector.u)
                        @test isfinite(best_vector.v)
                        @test abs(best_vector.u - displacement[1]) < 2.0
                        @test abs(best_vector.v - displacement[2]) < 2.0
                    end
                    @test best_vector.status == :good
                end
            end
            
            # Test parametric window functions
            parametric_windows = [(:kaiser, 5.0), (:tukey, 0.5), (:gaussian, 0.4)]
            for window_func in parametric_windows
                stage = PIVStage((32, 32), window_function=window_func)
                result = run_piv_stage(img1, img2, stage, CrossCorrelator)
                
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
end # Hammerhead.jl
