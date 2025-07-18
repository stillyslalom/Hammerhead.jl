#!/usr/bin/env julia
"""
Hammerhead.jl Benchmark Runner

Runs all performance benchmarks and provides a summary report.
Use this for regression testing and performance validation.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

println("🏁 Hammerhead.jl Performance Benchmark Suite")
println("=" ^ 60)
println("Julia Version: $(VERSION)")
println("Timestamp: $(now())")
println("=" ^ 60)

# List of benchmark files to run
benchmarks = [
    "benchmark_fast_vs_robust.jl" => "Core Performance Benchmark",
    "detailed_peak_benchmark.jl" => "Algorithm Analysis Benchmark"
]

results = Dict{String, Any}()

for (benchmark_file, description) in benchmarks
    println("\n🚀 Running: $description")
    println("-" ^ 50)
    
    benchmark_path = joinpath(@__DIR__, benchmark_file)
    
    if isfile(benchmark_path)
        try
            # Capture start time
            start_time = time()
            
            # Run benchmark in separate process to avoid interference
            result = read(`julia --project=$(joinpath(@__DIR__, "..")) $benchmark_path`, String)
            
            # Capture end time
            end_time = time()
            duration = end_time - start_time
            
            println(result)
            println("⏱️  Benchmark completed in $(round(duration, digits=2)) seconds")
            
            results[benchmark_file] = Dict(
                "description" => description,
                "duration" => duration,
                "output" => result,
                "status" => "success"
            )
            
        catch e
            println("❌ Benchmark failed: $e")
            results[benchmark_file] = Dict(
                "description" => description,
                "status" => "failed",
                "error" => string(e)
            )
        end
    else
        println("⚠️  Benchmark file not found: $benchmark_path")
        results[benchmark_file] = Dict(
            "description" => description,
            "status" => "missing"
        )
    end
end

# Summary report
println("\n" * "=" ^ 60)
println("📊 BENCHMARK SUMMARY REPORT")
println("=" ^ 60)

total_time = 0.0
successful_benchmarks = 0
failed_benchmarks = 0

for (file, result) in results
    status_emoji = result["status"] == "success" ? "✅" : 
                   result["status"] == "failed" ? "❌" : "⚠️"
    
    println("$status_emoji $(result["description"])")
    
    if result["status"] == "success"
        println("   Duration: $(round(result["duration"], digits=2))s")
        total_time += result["duration"]
        successful_benchmarks += 1
    elseif result["status"] == "failed"
        println("   Error: $(result["error"])")
        failed_benchmarks += 1
    else
        failed_benchmarks += 1
    end
end

println("\n📈 Overall Statistics:")
println("   Successful: $successful_benchmarks")
println("   Failed: $failed_benchmarks") 
println("   Total time: $(round(total_time, digits=2))s")

if failed_benchmarks == 0
    println("\n🎉 All benchmarks completed successfully!")
    exit(0)
else
    println("\n⚠️  Some benchmarks failed - check results above")
    exit(1)
end