#!/usr/bin/env julia

"""
Critical robustness and performance benchmarks for vector interpolation.

Focus areas:
1. Interpolated vector robustness and error propagation
2. Per-iteration performance for frequent interpolation calls
3. Cascading failure detection through multiple iterations
4. Validation that interpolated vectors won't corrupt deformation

Usage:
    julia --project=.. interpolation_robustness_benchmarks.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Hammerhead
using Hammerhead: detect_holes, replace_vectors!, VectorMedian, InverseDistanceWeighted, DelaunayBarycentric
using StructArrays
using Statistics: mean, std, median
using Random
using Printf
using BenchmarkTools
using LinearAlgebra: norm

# ============================================================================
# Robustness Test Framework
# ============================================================================

"""
    simulate_iterative_interpolation(vectors, iterations=5) -> Dict

Simulate multiple iterations of validation + interpolation to test error propagation.
"""
function simulate_iterative_interpolation(vectors, iterations=5; contamination_rate=0.05)
    results = Dict(
        :initial_accuracy => Float64[],
        :final_accuracy => Float64[],
        :error_propagation => Float64[],
        :performance_times => Float64[],
        :interpolated_counts => Int[],
        :cascade_failures => Int
    )
    
    # Store original ground truth
    original_vectors = deepcopy(vectors)
    cascade_failures = 0
    
    for iter in 1:iterations
        println("  Iteration $iter:")
        
        # Simulate validation failures (random bad vectors)
        n_vectors = length(vectors)
        n_contaminate = max(1, round(Int, contamination_rate * n_vectors))
        
        # Randomly mark good vectors as bad
        good_indices = findall(v -> v.status == :good, vectors)
        if length(good_indices) >= n_contaminate
            contaminate_indices = good_indices[randperm(length(good_indices))[1:n_contaminate]]
            for idx in contaminate_indices
                i, j = Tuple(idx)
                old = vectors[i, j]
                vectors[i, j] = PIVVector(old.x, old.y, old.u, old.v, :bad, old.peak_ratio, old.correlation_moment)
            end
        end
        
        # Detect holes and interpolate
        timing_result = @benchmark begin
            holes = detect_holes($vectors)
            for hole in holes
                if hole.classification == :isolated
                    replace_vectors!($vectors, hole, VectorMedian())
                elseif hole.classification == :small_cluster
                    replace_vectors!($vectors, hole, InverseDistanceWeighted())
                else
                    replace_vectors!($vectors, hole, DelaunayBarycentric())
                end
            end
        end setup=(vectors = deepcopy($vectors))
        
        iteration_time = BenchmarkTools.median(timing_result).time / 1e6  # ms
        
        # Apply interpolation to actual vectors
        holes = detect_holes(vectors)
        interpolated_count = 0
        for hole in holes
            if hole.classification == :isolated
                replace_vectors!(vectors, hole, VectorMedian())
            elseif hole.classification == :small_cluster
                replace_vectors!(vectors, hole, InverseDistanceWeighted())
            else
                replace_vectors!(vectors, hole, DelaunayBarycentric())
            end
            interpolated_count += sum(v.status == :interpolated for v in vectors if v in [vectors[idx] for idx in hole.indices])
        end
        
        # Calculate accuracy metrics
        total_errors = calculate_field_errors(vectors, original_vectors)
        interpolated_errors = calculate_interpolated_errors(vectors, original_vectors)
        
        push!(results[:performance_times], iteration_time)
        push!(results[:interpolated_counts], interpolated_count)
        push!(results[:initial_accuracy], total_errors[:rmse])
        
        # Check for cascade failure (large error growth)
        if iter > 1 && total_errors[:rmse] > 2.0 * results[:initial_accuracy][1]
            cascade_failures += 1
            println("    ⚠️  Cascade failure detected! RMSE: $(total_errors[:rmse])")
        end
        
        @printf("    🔄 Interpolated: %d vectors, Time: %.3f ms, RMSE: %.4f\\n", 
                interpolated_count, iteration_time, total_errors[:rmse])
    end
    
    # Final accuracy assessment
    final_errors = calculate_field_errors(vectors, original_vectors)
    results[:final_accuracy] = [final_errors[:rmse]]
    results[:error_propagation] = [final_errors[:rmse] / results[:initial_accuracy][1]]
    results[:cascade_failures] = cascade_failures
    
    return results
end

"""
    calculate_field_errors(current, original) -> Dict

Calculate error metrics between current and original vector fields.
"""
function calculate_field_errors(current, original)
    u_errors = Float64[]
    v_errors = Float64[]
    magnitude_errors = Float64[]
    
    for idx in eachindex(current)
        if current[idx].status != :bad  # Only compare non-bad vectors
            curr = current[idx]
            orig = original[idx]
            
            u_err = abs(curr.u - orig.u)
            v_err = abs(curr.v - orig.v)
            
            curr_mag = sqrt(curr.u^2 + curr.v^2)
            orig_mag = sqrt(orig.u^2 + orig.v^2)
            mag_err = abs(curr_mag - orig_mag)
            
            push!(u_errors, u_err)
            push!(v_errors, v_err)
            push!(magnitude_errors, mag_err)
        end
    end
    
    return Dict(
        :rmse => sqrt(mean(u_errors.^2 + v_errors.^2)),
        :u_rmse => sqrt(mean(u_errors.^2)),
        :v_rmse => sqrt(mean(v_errors.^2)),
        :max_error => maximum([maximum(u_errors), maximum(v_errors)]),
        :mean_error => mean(sqrt.(u_errors.^2 + v_errors.^2))
    )
end

"""
    calculate_interpolated_errors(current, original) -> Dict

Calculate errors specifically for interpolated vectors.
"""
function calculate_interpolated_errors(current, original)
    u_errors = Float64[]
    v_errors = Float64[]
    
    for idx in eachindex(current)
        if current[idx].status == :interpolated
            curr = current[idx]
            orig = original[idx]
            
            push!(u_errors, abs(curr.u - orig.u))
            push!(v_errors, abs(curr.v - orig.v))
        end
    end
    
    if isempty(u_errors)
        return Dict(:rmse => 0.0, :count => 0)
    end
    
    return Dict(
        :rmse => sqrt(mean(u_errors.^2 + v_errors.^2)),
        :count => length(u_errors),
        :max_error => maximum(sqrt.(u_errors.^2 + v_errors.^2))
    )
end

# ============================================================================
# Performance Stress Tests
# ============================================================================

"""
    benchmark_frequent_interpolation(sizes, hole_densities) -> Dict

Benchmark performance for frequent interpolation calls at various scales.
"""
function benchmark_frequent_interpolation(sizes, hole_densities)
    println("\\n📊 FREQUENT INTERPOLATION PERFORMANCE")
    println("=" ^ 50)
    
    results = []
    
    for size in sizes
        for density in hole_densities
            println("\\n🔍 Testing $(size[1])×$(size[2]) field, $(density*100)% holes")
            
            # Generate synthetic field
            vectors = StructArray([PIVVector(Float64(j), Float64(i), 0.1*j + 0.05*i, 0.05*j + 0.1*i, :good, 1.5, 0.3) 
                                  for i in 1:size[1], j in 1:size[2]])
            
            # Create holes at specified density
            n_holes = round(Int, density * length(vectors))
            hole_indices = randperm(length(vectors))[1:n_holes]
            
            for idx in hole_indices
                i, j = Tuple(CartesianIndices(vectors)[idx])
                old = vectors[i, j]
                vectors[i, j] = PIVVector(old.x, old.y, old.u, old.v, :bad, old.peak_ratio, old.correlation_moment)
            end
            
            # Benchmark the complete interpolation pipeline
            timing_result = @benchmark begin
                holes = detect_holes(test_vectors)
                for hole in holes
                    if hole.classification == :isolated
                        replace_vectors!(test_vectors, hole, VectorMedian())
                    elseif hole.classification == :small_cluster
                        replace_vectors!(test_vectors, hole, InverseDistanceWeighted())
                    else
                        replace_vectors!(test_vectors, hole, DelaunayBarycentric())
                    end
                end
            end setup=(test_vectors = deepcopy($vectors))
            
            median_time = BenchmarkTools.median(timing_result).time / 1e6  # ms
            memory_alloc = BenchmarkTools.median(timing_result).memory / 1024  # KB
            
            # Count hole types
            holes = detect_holes(vectors)
            isolated_count = sum(h.classification == :isolated for h in holes)
            cluster_count = sum(h.classification == :small_cluster for h in holes)
            large_count = sum(h.classification == :large_region for h in holes)
            
            @printf("    ⚡ Performance: %.3f ms (%.1f KB)\\n", median_time, memory_alloc)
            @printf("    🕳️  Holes: %d isolated, %d clusters, %d large\\n", isolated_count, cluster_count, large_count)
            
            push!(results, Dict(
                :size => size,
                :density => density,
                :time_ms => median_time,
                :memory_kb => memory_alloc,
                :holes => Dict(:isolated => isolated_count, :cluster => cluster_count, :large => large_count),
                :total_holes => length(holes)
            ))
        end
    end
    
    return results
end

# ============================================================================
# Deformation Corruption Tests
# ============================================================================

"""
    test_deformation_corruption(vectors) -> Dict

Test if interpolated vectors would corrupt image deformation transformations.
"""
function test_deformation_corruption(vectors)
    println("\\n🔬 DEFORMATION CORRUPTION ANALYSIS")
    println("=" ^ 40)
    
    # Apply interpolation
    holes = detect_holes(vectors)
    for hole in holes
        if hole.classification == :isolated
            replace_vectors!(vectors, hole, VectorMedian())
        elseif hole.classification == :small_cluster
            replace_vectors!(vectors, hole, InverseDistanceWeighted())
        else
            replace_vectors!(vectors, hole, DelaunayBarycentric())
        end
    end
    
    # Check for problematic interpolated vectors
    interpolated_vectors = [v for v in vectors if v.status == :interpolated]
    
    if isempty(interpolated_vectors)
        return Dict(:corruption_risk => :none, :issues => [])
    end
    
    issues = []
    
    # Check for extreme displacements
    u_values = [v.u for v in interpolated_vectors]
    v_values = [v.v for v in interpolated_vectors]
    magnitudes = sqrt.(u_values.^2 + v_values.^2)
    
    # Statistical outlier detection
    mag_median = median(magnitudes)
    mag_std = std(magnitudes)
    outlier_threshold = mag_median + 3*mag_std
    
    extreme_count = sum(magnitudes .> outlier_threshold)
    if extreme_count > 0
        push!(issues, "$(extreme_count) extreme displacement outliers (>3σ)")
    end
    
    # Check for NaN/Inf values
    nan_count = sum(isnan.(u_values) .| isnan.(v_values))
    inf_count = sum(isinf.(u_values) .| isinf.(v_values))
    
    if nan_count > 0
        push!(issues, "$(nan_count) NaN values")
    end
    if inf_count > 0
        push!(issues, "$(inf_count) Inf values")
    end
    
    # Assess overall corruption risk
    corruption_risk = if nan_count > 0 || inf_count > 0
        :high
    elseif extreme_count > length(interpolated_vectors) * 0.1
        :medium
    elseif extreme_count > 0
        :low
    else
        :none
    end
    
    return Dict(
        :corruption_risk => corruption_risk,
        :issues => issues,
        :interpolated_count => length(interpolated_vectors),
        :stats => Dict(
            :magnitude_median => mag_median,
            :magnitude_std => mag_std,
            :max_magnitude => maximum(magnitudes)
        )
    )
end

# Utility functions
using Random: randperm

# ============================================================================
# Main Benchmark Suite
# ============================================================================

function main()
    println("🚀 INTERPOLATION ROBUSTNESS & PERFORMANCE BENCHMARKS")
    println("=" ^ 60)
    
    Random.seed!(42)  # Reproducible results
    
    # Test configurations focused on iteration workflow
    test_sizes = [(32, 32), (64, 64), (128, 128)]
    hole_densities = [0.02, 0.05, 0.10, 0.15]  # 2% to 15% holes
    
    # 1. Iterative robustness test
    println("\\n🔄 ITERATIVE ROBUSTNESS TEST")
    println("=" ^ 40)
    
    vectors = StructArray([PIVVector(Float64(j), Float64(i), 0.1*j + 0.05*i, 0.05*j + 0.1*i, :good, 1.5, 0.3) 
                          for i in 1:64, j in 1:64])
    
    robustness_results = simulate_iterative_interpolation(vectors, 5; contamination_rate=0.03)
    
    println("\\n📈 Robustness Summary:")
    @printf("  Initial RMSE: %.4f\\n", robustness_results[:initial_accuracy][1])
    @printf("  Final RMSE: %.4f\\n", robustness_results[:final_accuracy][1])
    @printf("  Error amplification: %.2fx\\n", robustness_results[:error_propagation][1])
    @printf("  Cascade failures: %d\\n", robustness_results[:cascade_failures])
    @printf("  Avg iteration time: %.3f ms\\n", mean(robustness_results[:performance_times]))
    
    # 2. Performance scaling test
    performance_results = benchmark_frequent_interpolation(test_sizes, hole_densities)
    
    # 3. Deformation corruption test
    test_vectors = StructArray([PIVVector(Float64(j), Float64(i), 0.1*j, 0.1*i, i+j > 10 ? :bad : :good, 1.5, 0.3) 
                               for i in 1:32, j in 1:32])
    
    corruption_results = test_deformation_corruption(test_vectors)
    
    println("\\n🔬 Deformation Corruption Risk: $(corruption_results[:corruption_risk])")
    if !isempty(corruption_results[:issues])
        println("  Issues found:")
        for issue in corruption_results[:issues]
            println("    - $issue")
        end
    end
    
    # 4. Summary recommendations
    println("\\n\\n🎯 PERFORMANCE RECOMMENDATIONS")
    println("=" ^ 50)
    
    # Find performance sweet spot
    fast_configs = filter(r -> r[:time_ms] < 1.0, performance_results)
    if !isempty(fast_configs)
        best_config = fast_configs[argmax([r[:size][1] * r[:size][2] * r[:density] for r in fast_configs])]
        println("✅ Recommended for frequent iteration:")
        @printf("   Size: %dx%d, Hole density: %.1f%%, Time: %.3f ms\\n", 
                best_config[:size]..., best_config[:density]*100, best_config[:time_ms])
    end
    
    if robustness_results[:error_propagation][1] < 1.5 && robustness_results[:cascade_failures] == 0
        println("✅ Interpolation methods are stable across iterations")
    else
        println("⚠️  Caution: Error propagation detected")
    end
    
    if corruption_results[:corruption_risk] == :none
        println("✅ Interpolated vectors safe for deformation")
    else
        println("⚠️  Interpolated vectors may corrupt deformation")
    end
    
    println("\\n✅ Robustness benchmarks complete!")
    
    return Dict(
        :robustness => robustness_results,
        :performance => performance_results,
        :corruption => corruption_results
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end