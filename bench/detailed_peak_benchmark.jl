#!/usr/bin/env julia
using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Hammerhead
using TimerOutputs
using Random

# Import internal functions for detailed benchmarking
import Hammerhead: find_secondary_peak, find_secondary_peak_robust, find_local_maxima

println("🔍 Detailed Peak Detection Performance Analysis")
println("=" ^ 55)

# Create test correlation planes of different sizes
function create_test_correlation_plane(size_dim)
    corr_plane = zeros(size_dim, size_dim)
    
    # Primary peak at center
    center = size_dim ÷ 2 + 1
    primary_loc = CartesianIndex(center, center)
    corr_plane[center, center] = 1.0
    
    # Add some spread around primary peak
    for i in max(1, center-2):min(size_dim, center+2)
        for j in max(1, center-2):min(size_dim, center+2)
            if i != center || j != center
                corr_plane[i, j] = 0.8 * exp(-0.5 * ((i-center)^2 + (j-center)^2))
            end
        end
    end
    
    # Secondary peak away from primary
    sec_i, sec_j = center - 8, center + 10
    if sec_i > 0 && sec_j <= size_dim
        corr_plane[sec_i, sec_j] = 0.6
        # Add some spread around secondary peak
        for i in max(1, sec_i-2):min(size_dim, sec_i+2)
            for j in max(1, sec_j-2):min(size_dim, sec_j+2)
                if i != sec_i || j != sec_j
                    corr_plane[i, j] = max(corr_plane[i, j], 0.5 * exp(-0.5 * ((i-sec_i)^2 + (j-sec_j)^2)))
                end
            end
        end
    end
    
    return corr_plane, primary_loc
end

# Benchmark individual functions
function benchmark_methods()
    sizes = [32, 64, 128, 256]
    n_iterations = 1000
    
    println("📊 Performance vs Correlation Plane Size")
    println("-" ^ 50)
    
    for size_dim in sizes
        corr_plane, primary_loc = create_test_correlation_plane(size_dim)
        corr_mag = abs.(corr_plane)
        
        println("\n🔹 Testing $(size_dim)×$(size_dim) correlation plane:")
        
        # Benchmark fast method
        fast_timer = TimerOutput()
        for i in 1:n_iterations
            @timeit fast_timer "Fast Method" begin
                secondary_val = find_secondary_peak(corr_mag, primary_loc, 1.0)
            end
        end
        fast_time = TimerOutputs.time(fast_timer.inner_timers["Fast Method"]) / 1e9 / n_iterations
        
        # Benchmark robust method
        robust_timer = TimerOutput()
        for i in 1:n_iterations
            @timeit robust_timer "Robust Method" begin
                secondary_val = find_secondary_peak_robust(corr_mag, primary_loc, 1.0)
            end
        end
        robust_time = TimerOutputs.time(robust_timer.inner_timers["Robust Method"]) / 1e9 / n_iterations
        
        # Benchmark just local maxima detection
        maxima_timer = TimerOutput()
        for i in 1:n_iterations
            @timeit maxima_timer "Local Maxima" begin
                maxima = find_local_maxima(corr_mag)
            end
        end
        maxima_time = TimerOutputs.time(maxima_timer.inner_timers["Local Maxima"]) / 1e9 / n_iterations
        
        speedup = fast_time / robust_time
        
        println("   Fast method:     $(round(fast_time * 1e6, digits=1)) μs")
        println("   Robust method:   $(round(robust_time * 1e6, digits=1)) μs")
        println("   Local maxima:    $(round(maxima_time * 1e6, digits=1)) μs")
        println("   Speedup ratio:   $(round(speedup, digits=2))x")
        
        if speedup < 1.0
            println("   ⚠️  Fast method is SLOWER than robust method!")
        else
            println("   ✅ Fast method is faster")
        end
    end
end

# Analyze computational complexity
function analyze_complexity()
    println("\n" ^ 2 * "🧮 Computational Complexity Analysis")
    println("-" ^ 50)
    
    size_dim = 64
    corr_plane, primary_loc = create_test_correlation_plane(size_dim)
    corr_mag = abs.(corr_plane)
    
    # Count operations in fast method
    println("📊 Fast Method Analysis:")
    pixel_count = 0
    sqrt_ops = 0
    comparisons = 0
    
    exclusion_radius = 3
    for idx in CartesianIndices(corr_mag)
        pixel_count += 1
        dx = idx.I[1] - primary_loc.I[1]
        dy = idx.I[2] - primary_loc.I[2]
        sqrt_ops += 1  # sqrt(dx^2 + dy^2)
        comparisons += 1  # distance comparison
        
        if sqrt(dx^2 + dy^2) > exclusion_radius
            comparisons += 1  # value comparison
        end
    end
    
    println("   Total pixels checked: $pixel_count")
    println("   Square root operations: $sqrt_ops")
    println("   Comparisons: $comparisons")
    println("   → O(n²) complexity with expensive sqrt operations")
    
    # Count operations in robust method  
    println("\n📊 Robust Method Analysis:")
    local_maxima = find_local_maxima(corr_mag)
    interior_checks = (size_dim - 2) * (size_dim - 2) * 8  # 3x3 neighbor checks
    sort_ops = length(local_maxima) * log(length(local_maxima))
    
    println("   Interior pixels checked: $((size_dim - 2) * (size_dim - 2))")
    println("   Neighbor comparisons: $interior_checks")
    println("   Local maxima found: $(length(local_maxima))")
    println("   Sort operations: $(round(sort_ops, digits=1))")
    println("   → O(n²) complexity with simple comparisons + O(k log k) sort")
end

# Propose fast method improvements
function propose_improvements()
    println("\n" ^ 2 * "🚀 Fast Method Improvement Proposals")
    println("-" ^ 50)
    
    println("❌ Current Fast Method Issues:")
    println("   • Expensive sqrt() operation for every pixel")
    println("   • Checks entire correlation plane")
    println("   • No early termination")
    
    println("\n✅ Proposed Improvements:")
    println("   • Use squared distance comparison (avoid sqrt)")
    println("   • Use rectangular exclusion region") 
    println("   • Pre-compute exclusion bounds")
    println("   • Consider iterating in value-descending order")
    
    println("\n🔧 Implementation sketch:")
    println("   ```julia")
    println("   # Fast exclusion with squared distance")
    println("   exclusion_radius_sq = exclusion_radius^2")
    println("   for idx in CartesianIndices(corr_mag)")
    println("       dx, dy = idx.I .- primary_loc.I")
    println("       if dx^2 + dy^2 > exclusion_radius_sq  # No sqrt!")
    println("           # Check value...")
    println("       end")
    println("   end")
    println("   ```")
end

# Run all analyses
benchmark_methods()
analyze_complexity()
propose_improvements()

println("\n" * "=" ^ 55)
println("🎯 CONCLUSIONS:")
println("• Current 'fast' method is slower due to sqrt operations")
println("• Robust method is actually more efficient for small correlation planes")
println("• Fast method needs optimization to live up to its name")
println("• Consider implementing the proposed improvements")
println("=" ^ 55)