#!/usr/bin/env julia

"""
Comprehensive benchmarks for vector replacement (hole filling) methods.

Tests accuracy and performance of:
1. Vector median (isolated 1×1 holes)
2. Inverse distance weighted (≤2×2 clusters) 
3. Delaunay barycentric (large/irregular holes)

Usage:
    julia --project=.. vector_replacement_benchmarks.jl
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Hammerhead
using Hammerhead: detect_holes, replace_vectors!, VectorMedian, InverseDistanceWeighted, DelaunayBarycentric
using StructArrays
using Statistics: mean, std
using Random
using Printf
using BenchmarkTools

# ============================================================================
# Synthetic Data Generation
# ============================================================================

"""
    generate_synthetic_field(rows::Int, cols::Int; flow_type=:linear) -> StructArray{PIVVector}

Generate synthetic vector field with known ground truth.
"""
function generate_synthetic_field(rows::Int, cols::Int; flow_type=:linear)
    vectors = PIVVector[]
    
    for i in 1:rows, j in 1:cols
        x, y = Float64(j), Float64(i)  # Grid positions
        
        # Generate synthetic flow based on type
        if flow_type == :linear
            u = 0.1 * x + 0.05 * y  # Linear velocity field
            v = 0.05 * x + 0.1 * y
        elseif flow_type == :vortex
            cx, cy = rows/2, cols/2  # Vortex center
            dx, dy = x - cx, y - cy
            r = sqrt(dx^2 + dy^2)
            u = -0.1 * dy / max(r, 1.0)  # Tangential velocity
            v = 0.1 * dx / max(r, 1.0)
        elseif flow_type == :shear
            u = 0.2 * y  # Shear flow
            v = 0.0
        else
            error("Unknown flow type: $flow_type")
        end
        
        # Add some realistic quality metrics
        peak_ratio = 2.0 + 0.5 * randn()
        correlation_moment = 0.8 + 0.2 * randn()
        
        push!(vectors, PIVVector(x, y, u, v, :good, peak_ratio, correlation_moment))
    end
    
    return StructArray(reshape(vectors, rows, cols))
end

"""
    create_holes!(vectors::StructArray{PIVVector}, hole_patterns::Vector) -> Vector{HoleRegion}

Create holes in vector field according to specified patterns.
Returns the ground truth vectors that were removed.
"""
function create_holes!(vectors::StructArray{PIVVector}, hole_patterns::Vector)
    ground_truth = Dict{CartesianIndex{2}, PIVVector}()
    
    for pattern in hole_patterns
        if pattern[:type] == :isolated
            # Create isolated 1×1 holes
            for _ in 1:pattern[:count]
                i = rand(2:size(vectors,1)-1)  # Avoid edges
                j = rand(2:size(vectors,2)-1)
                idx = CartesianIndex(i, j)
                ground_truth[idx] = vectors[i, j]
                # Mark as bad but preserve other data
                vectors[i, j] = PIVVector(vectors[i, j].x, vectors[i, j].y, NaN, NaN, 
                                        :bad, vectors[i, j].peak_ratio, vectors[i, j].correlation_moment)
            end
        elseif pattern[:type] == :cluster
            # Create small 2×2 clusters
            for _ in 1:pattern[:count]
                i = rand(2:size(vectors,1)-2)  # Leave room for 2×2
                j = rand(2:size(vectors,2)-2)
                for di in 0:1, dj in 0:1
                    idx = CartesianIndex(i+di, j+dj)
                    ground_truth[idx] = vectors[i+di, j+dj]
                    vectors[i+di, j+dj] = PIVVector(vectors[i+di, j+dj].x, vectors[i+di, j+dj].y, NaN, NaN,
                                                  :bad, vectors[i+di, j+dj].peak_ratio, vectors[i+di, j+dj].correlation_moment)
                end
            end
        elseif pattern[:type] == :large
            # Create large irregular holes
            for _ in 1:pattern[:count]
                center_i = rand(4:size(vectors,1)-4)
                center_j = rand(4:size(vectors,2)-4)
                radius = pattern[:radius]
                
                for i in (center_i-radius):(center_i+radius)
                    for j in (center_j-radius):(center_j+radius)
                        if 1 <= i <= size(vectors,1) && 1 <= j <= size(vectors,2)
                            # Create irregular shape (not perfect circle)
                            dist = sqrt((i-center_i)^2 + (j-center_j)^2)
                            if dist <= radius + 0.5*randn()  # Add noise to shape
                                idx = CartesianIndex(i, j)
                                ground_truth[idx] = vectors[i, j]
                                vectors[i, j] = PIVVector(vectors[i, j].x, vectors[i, j].y, NaN, NaN,
                                                        :bad, vectors[i, j].peak_ratio, vectors[i, j].correlation_moment)
                            end
                        end
                    end
                end
            end
        end
    end
    
    return ground_truth
end

# ============================================================================
# Accuracy Metrics
# ============================================================================

"""
    calculate_accuracy_metrics(replaced::StructArray{PIVVector}, ground_truth::Dict) -> Dict

Calculate accuracy metrics comparing replaced vectors to ground truth.
"""
function calculate_accuracy_metrics(replaced::StructArray{PIVVector}, ground_truth::Dict)
    u_errors = Float64[]
    v_errors = Float64[]
    magnitude_errors = Float64[]
    
    for (idx, true_vector) in ground_truth
        if replaced[idx].status == :interpolated
            replaced_vector = replaced[idx]
            
            u_error = abs(replaced_vector.u - true_vector.u)
            v_error = abs(replaced_vector.v - true_vector.v)
            
            true_mag = sqrt(true_vector.u^2 + true_vector.v^2)
            replaced_mag = sqrt(replaced_vector.u^2 + replaced_vector.v^2)
            magnitude_error = abs(replaced_mag - true_mag)
            
            push!(u_errors, u_error)
            push!(v_errors, v_error)
            push!(magnitude_errors, magnitude_error)
        end
    end
    
    return Dict(
        :u_rmse => sqrt(mean(u_errors.^2)),
        :v_rmse => sqrt(mean(v_errors.^2)),
        :magnitude_rmse => sqrt(mean(magnitude_errors.^2)),
        :u_mean_error => mean(u_errors),
        :v_mean_error => mean(v_errors),
        :magnitude_mean_error => mean(magnitude_errors),
        :success_rate => length(u_errors) / length(ground_truth)
    )
end

# ============================================================================
# Benchmark Functions
# ============================================================================

"""
    benchmark_method(method, vectors, holes; description="") -> Dict

Benchmark a specific replacement method for accuracy and performance.
"""
function benchmark_method(method, vectors, holes, ground_truth; description="")
    println("\\n🔍 Benchmarking: $description")
    
    # Create copy for testing
    test_vectors = deepcopy(vectors)
    
    # Performance benchmark
    @printf("   ⏱️  Performance: ")
    timing_result = @benchmark begin
        for hole in $holes
            if hole.classification == $(method isa VectorMedian ? :isolated : 
                                        method isa InverseDistanceWeighted ? :small_cluster : :large_region)
                replace_vectors!($test_vectors, hole, $method)
            end
        end
    end setup=(test_vectors = deepcopy($vectors))
    
    median_time = BenchmarkTools.median(timing_result).time / 1e6  # Convert to ms
    memory_alloc = BenchmarkTools.median(timing_result).memory
    
    @printf("%.3f ms (%.1f KB allocated)\\n", median_time, memory_alloc / 1024)
    
    # Accuracy benchmark
    test_vectors = deepcopy(vectors)
    relevant_holes = filter(h -> h.classification == (method isa VectorMedian ? :isolated : 
                                                     method isa InverseDistanceWeighted ? :small_cluster : :large_region), holes)
    
    for hole in relevant_holes
        replace_vectors!(test_vectors, hole, method)
    end
    
    accuracy = calculate_accuracy_metrics(test_vectors, ground_truth)
    
    @printf("   📊 Accuracy: RMSE(u)=%.4f, RMSE(v)=%.4f, Success=%.1f%%\\n", 
            accuracy[:u_rmse], accuracy[:v_rmse], accuracy[:success_rate] * 100)
    
    return Dict(
        :method => description,
        :median_time_ms => median_time,
        :memory_kb => memory_alloc / 1024,
        :accuracy => accuracy,
        :relevant_holes => length(relevant_holes)
    )
end

# ============================================================================
# Main Benchmark Suite
# ============================================================================

function main()
    println("🚀 Vector Replacement Benchmarks")
    println("=" ^ 50)
    
    Random.seed!(42)  # Reproducible results
    
    # Test configurations
    configurations = [
        (flow=:linear, size=(32, 32), name="Linear Flow (32×32)"),
        (flow=:vortex, size=(32, 32), name="Vortex Flow (32×32)"),
        (flow=:linear, size=(64, 64), name="Linear Flow (64×64)"),
        (flow=:shear, size=(48, 48), name="Shear Flow (48×48)")
    ]
    
    hole_patterns = [
        Dict(:type => :isolated, :count => 10),
        Dict(:type => :cluster, :count => 3),
        Dict(:type => :large, :count => 2, :radius => 3)
    ]
    
    methods = [
        (VectorMedian(), "Vector Median"),
        (InverseDistanceWeighted(2.0), "Inverse Distance Weighted (p=2)"),
        (InverseDistanceWeighted(1.0), "Inverse Distance Weighted (p=1)"),
        (DelaunayBarycentric(), "Delaunay Barycentric")
    ]
    
    all_results = []
    
    for config in configurations
        println("\\n🔬 Testing: $(config.name)")
        println("-" ^ 30)
        
        # Generate synthetic field
        vectors = generate_synthetic_field(config.size...; flow_type=config.flow)
        
        # Create holes and store ground truth
        ground_truth = create_holes!(vectors, hole_patterns)
        
        # Detect holes
        holes = detect_holes(vectors)
        println("   Detected $(length(holes)) hole regions")
        
        # Benchmark each method
        for (method, description) in methods
            result = benchmark_method(method, vectors, holes, ground_truth; description=description)
            result[:flow_type] = config.flow
            result[:grid_size] = "$(config.size[1])×$(config.size[2])"
            push!(all_results, result)
        end
    end
    
    # Summary analysis
    println("\\n\\n📈 SUMMARY ANALYSIS")
    println("=" ^ 50)
    
    # Group by method for comparison
    methods_summary = Dict()
    for result in all_results
        method = result[:method]
        if !haskey(methods_summary, method)
            methods_summary[method] = []
        end
        push!(methods_summary[method], result)
    end
    
    for (method, results) in methods_summary
        println("\\n$method:")
        avg_time = mean([r[:median_time_ms] for r in results])
        avg_memory = mean([r[:memory_kb] for r in results])
        avg_u_rmse = mean([r[:accuracy][:u_rmse] for r in results])
        avg_success = mean([r[:accuracy][:success_rate] for r in results])
        
        @printf("  ⚡ Performance: %.3f ms, %.1f KB\\n", avg_time, avg_memory)
        @printf("  🎯 Accuracy: RMSE(u)=%.4f, Success=%.1f%%\\n", avg_u_rmse, avg_success * 100)
    end
    
    println("\\n✅ Benchmark complete!")
    return all_results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end