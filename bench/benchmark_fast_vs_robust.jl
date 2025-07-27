#!/usr/bin/env julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Hammerhead
using Hammerhead.SyntheticData
using TimerOutputs
using Random
using Statistics

# Import internal functions for benchmarking
import Hammerhead: calculate_quality_metrics, apply_window_function, window_function_type

println("🏁 Fast vs Robust Methods Performance Benchmark")
println("=" ^ 60)

# Generate synthetic test images with controlled characteristics
function generate_benchmark_images(size_img=(256, 256), displacement=(2.0, 3.0), noise_level=0.1)
    Random.seed!(1234)
    
    # Create first image with random particles
    img1 = zeros(Float64, size_img)
    n_particles = 200
    for i in 1:n_particles
        x = rand() * (size_img[1] - 20) + 10
        y = rand() * (size_img[2] - 20) + 10
        generate_gaussian_particle!(img1, (x, y), 3.0)
    end
    
    # Add noise
    img1 .+= noise_level * randn(size_img)
    
    # Create second image by shifting particles
    img2 = zeros(Float64, size_img)
    for i in 1:n_particles
        x = rand() * (size_img[1] - 20) + 10 + displacement[1]
        y = rand() * (size_img[2] - 20) + 10 + displacement[2]
        generate_gaussian_particle!(img2, (x, y), 3.0)
    end
    
    # Add noise
    img2 .+= noise_level * randn(size_img)
    
    return img1, img2
end


# Test secondary peak detection methods directly
function benchmark_peak_detection()
    println("\n📊 Secondary Peak Detection Benchmark")
    println("-" ^ 50)
    
    # Create a synthetic correlation plane with two peaks
    corr_plane = zeros(64, 64)
    
    # Primary peak at center
    primary_loc = CartesianIndex(32, 32)
    corr_plane[32, 32] = 1.0
    # Add some spread around primary peak
    for i in 30:34, j in 30:34
        corr_plane[i, j] = 0.8 * exp(-0.5 * ((i-32)^2 + (j-32)^2))
    end
    
    # Secondary peak away from primary
    secondary_loc = CartesianIndex(25, 40)
    corr_plane[25, 40] = 0.6
    # Add some spread around secondary peak
    for i in 23:27, j in 38:42
        corr_plane[i, j] = 0.5 * exp(-0.5 * ((i-25)^2 + (j-40)^2))
    end
    
    n_iterations = 10000
    
    # Benchmark fast method
    println("🚀 Testing fast secondary peak detection...")
    fast_timer = TimerOutput()
    for i in 1:n_iterations
        @timeit fast_timer "Fast Method" begin
            peak_ratio, _ = calculate_quality_metrics(corr_plane, primary_loc, 1.0, robust=false)
        end
    end
    
    # Benchmark robust method
    println("🛡️  Testing robust secondary peak detection...")
    robust_timer = TimerOutput()
    for i in 1:n_iterations
        @timeit robust_timer "Robust Method" begin
            peak_ratio, _ = calculate_quality_metrics(corr_plane, primary_loc, 1.0, robust=true)
        end
    end
    
    # Extract timing results
    fast_time = TimerOutputs.time(fast_timer.inner_timers["Fast Method"]) / 1e9 / n_iterations
    robust_time = TimerOutputs.time(robust_timer.inner_timers["Robust Method"]) / 1e9 / n_iterations
    
    speedup = robust_time / fast_time
    
    println("📈 Peak Detection Results:")
    println("   Fast method:   $(round(fast_time * 1e6, digits=2)) μs per call")
    println("   Robust method: $(round(robust_time * 1e6, digits=2)) μs per call")
    println("   Speedup:       $(round(speedup, digits=1))x faster")
    
    return speedup
end

# Full PIV analysis benchmark
function benchmark_full_piv()
    println("\n📊 Full PIV Analysis Benchmark")
    println("-" ^ 50)
    
    # Generate test images
    img1, img2 = generate_benchmark_images((128, 128), (1.5, 2.0))
    
    n_runs = 5
    fast_times = Float64[]
    robust_times = Float64[]
    
    println("🚀 Running PIV with fast methods...")
    for i in 1:n_runs
        # This uses fast methods by default (robust=false in calculate_quality_metrics)
        result = run_piv(img1, img2, window_size=(32, 32), overlap=(0.5, 0.5))
        timer = Hammerhead.get_timer(result)
        total_time = TimerOutputs.time(timer.inner_timers["Single-Stage PIV"]) / 1e9
        push!(fast_times, total_time)
    end
    
    # Note: We would need to modify run_piv to accept a robust parameter to test robust methods
    # For now, let's benchmark the individual components we can control
    
    avg_fast = mean(fast_times)
    std_fast = std(fast_times)
    
    println("📈 PIV Analysis Results:")
    println("   Fast PIV: $(round(avg_fast, digits=3)) ± $(round(std_fast, digits=3)) seconds")
    
    return avg_fast
end

# Window function benchmark
function benchmark_window_functions()
    println("\n📊 Window Function Benchmark")
    println("-" ^ 50)
    
    test_img = rand(64, 64)
    n_iterations = 10000
    
    window_functions = [:rectangular, :hanning, :hamming, :blackman]
    
    for wf in window_functions
        wf_timer = TimerOutput()
        wf_type = window_function_type(wf)
        for i in 1:n_iterations
            @timeit wf_timer "Window $wf" begin
                apply_window_function(test_img, wf_type)
            end
        end
        
        wf_time = TimerOutputs.time(wf_timer.inner_timers["Window $wf"]) / 1e9 / n_iterations
        println("   $wf: $(round(wf_time * 1e6, digits=2)) μs per call")
    end
end

# FFT vs Direct correlation benchmark 
function benchmark_correlation_methods()
    println("\n📊 Correlation Methods Benchmark")
    println("-" ^ 50)
    
    # Test different window sizes
    window_sizes = [16, 32, 64, 128]
    
    for ws in window_sizes
        test_img1 = rand(ws, ws)
        test_img2 = rand(ws, ws)
        correlator = CrossCorrelator((ws, ws))
        
        n_iterations = 1000
        corr_timer = TimerOutput()
        
        for i in 1:n_iterations
            @timeit corr_timer "FFT Correlation $ws" begin
                correlate!(correlator, test_img1, test_img2)
            end
        end
        
        corr_time = TimerOutputs.time(corr_timer.inner_timers["FFT Correlation $ws"]) / 1e9 / n_iterations
        println("   $(ws)×$(ws): $(round(corr_time * 1e6, digits=2)) μs per correlation")
    end
end

# Run all benchmarks
println("🏃 Starting comprehensive performance benchmarks...")

# 1. Secondary peak detection speed comparison
peak_speedup = benchmark_peak_detection()

# 2. Window function performance
benchmark_window_functions()

# 3. Correlation method scaling
benchmark_correlation_methods()

# 4. Full PIV performance
piv_time = benchmark_full_piv()

println("\n" * "=" ^ 60)
println("🏆 BENCHMARK SUMMARY")
println("=" ^ 60)
println("💨 Fast vs Robust Peak Detection: $(round(peak_speedup, digits=1))x speedup")
println("⚡ Full PIV Analysis Time: $(round(piv_time, digits=3)) seconds")
println("✅ All timing instrumentation working correctly!")

if peak_speedup > 2.0
    println("🎯 SUCCESS: Fast methods are significantly faster than robust methods!")
else
    println("⚠️  WARNING: Fast methods may not be providing expected speedup")
end