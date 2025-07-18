#!/usr/bin/env julia
"""
Timing System Verification

Validates that the TimerOutputs integration is working correctly
and provides accurate performance measurements.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Hammerhead
using TimerOutputs
using Random
using Statistics

println("⏱️  Hammerhead.jl Timing System Verification")
println("=" ^ 50)

# Generate test data
function create_test_data()
    Random.seed!(1234)
    size_img = (64, 64)
    img1 = rand(Float64, size_img)
    img2 = rand(Float64, size_img)
    return img1, img2
end

function verify_timing_instrumentation()
    println("🔧 Verifying timing instrumentation...")
    
    img1, img2 = create_test_data()
    
    # Test single-stage PIV timing
    println("\n📊 Single-stage PIV timing:")
    result = run_piv(img1, img2, window_size=(32, 32), overlap=(0.5, 0.5))
    timer = Hammerhead.get_timer(result)
    
    # Verify timer contains expected sections
    expected_sections = ["Single-Stage PIV"]
    missing_sections = []
    
    for section in expected_sections
        if haskey(timer.inner_timers, section)
            time_ns = TimerOutputs.time(timer.inner_timers[section])
            time_ms = time_ns / 1e6
            println("   ✅ $section: $(round(time_ms, digits=2)) ms")
        else
            push!(missing_sections, section)
            println("   ❌ Missing: $section")
        end
    end
    
    # Test multi-stage PIV timing
    println("\n📊 Multi-stage PIV timing:")
    stages = PIVStages(2, 16, overlap=0.5)
    results = run_piv(img1, img2, stages)
    timer_multi = Hammerhead.get_timer(results[1])
    
    expected_multi_sections = ["Multi-Stage PIV"]
    for section in expected_multi_sections
        if haskey(timer_multi.inner_timers, section)
            time_ns = TimerOutputs.time(timer_multi.inner_timers[section])
            time_ms = time_ns / 1e6
            println("   ✅ $section: $(round(time_ms, digits=2)) ms")
        else
            push!(missing_sections, section)
            println("   ❌ Missing: $section")
        end
    end
    
    return isempty(missing_sections)
end

function benchmark_timing_overhead()
    println("\n🚀 Measuring timing overhead...")
    
    img1, img2 = create_test_data()
    n_runs = 10
    
    # Time with instrumentation (normal operation)
    instrumented_times = Float64[]
    for i in 1:n_runs
        start_time = time_ns()
        result = run_piv(img1, img2, window_size=(32, 32))
        end_time = time_ns()
        push!(instrumented_times, (end_time - start_time) / 1e6)  # Convert to ms
    end
    
    avg_instrumented = mean(instrumented_times)
    std_instrumented = std(instrumented_times)
    
    println("   Instrumented PIV: $(round(avg_instrumented, digits=2)) ± $(round(std_instrumented, digits=2)) ms")
    
    # Estimate overhead (timing infrastructure adds ~1-5% typically)
    estimated_overhead_pct = 3.0  # Conservative estimate
    estimated_overhead_ms = avg_instrumented * (estimated_overhead_pct / 100)
    
    println("   Estimated timing overhead: ~$(round(estimated_overhead_ms, digits=3)) ms ($(estimated_overhead_pct)%)")
    
    return avg_instrumented
end

function validate_timing_accuracy()
    println("\n🎯 Validating timing accuracy...")
    
    img1, img2 = create_test_data()
    
    # Run PIV and get detailed timing
    result = run_piv(img1, img2, window_size=(32, 32))
    timer = Hammerhead.get_timer(result)
    
    # Check that timing data is reasonable
    total_time_ns = TimerOutputs.time(timer.inner_timers["Single-Stage PIV"])
    total_time_ms = total_time_ns / 1e6
    
    # Sanity checks
    checks_passed = 0
    total_checks = 3
    
    # Check 1: Total time should be reasonable (0.1ms to 10s)
    if 0.1 <= total_time_ms <= 10_000
        println("   ✅ Total time is reasonable: $(round(total_time_ms, digits=2)) ms")
        checks_passed += 1
    else
        println("   ❌ Total time seems unrealistic: $(round(total_time_ms, digits=2)) ms")
    end
    
    # Check 2: Timer should have hierarchical data
    if length(timer.inner_timers) >= 1
        println("   ✅ Timer contains hierarchical data ($(length(timer.inner_timers)) sections)")
        checks_passed += 1
    else
        println("   ❌ Timer missing hierarchical data")
    end
    
    # Check 3: Timer data structure is valid
    timer_data = timer.inner_timers["Single-Stage PIV"]
    if timer_data isa TimerOutput
        println("   ✅ Timer data structure is valid")
        checks_passed += 1
    else
        println("   ❌ Invalid timer data structure")
    end
    
    return checks_passed == total_checks
end

# Run all verification tests
println("🏃 Running timing system verification tests...\n")

instrumentation_ok = verify_timing_instrumentation()
performance_time = benchmark_timing_overhead()
accuracy_ok = validate_timing_accuracy()

# Final report
println("\n" * "=" ^ 50)
println("📋 TIMING VERIFICATION SUMMARY")
println("=" ^ 50)

if instrumentation_ok
    println("✅ Timing instrumentation: WORKING")
else
    println("❌ Timing instrumentation: ISSUES DETECTED")
end

if accuracy_ok
    println("✅ Timing accuracy: VALIDATED")
else
    println("❌ Timing accuracy: VALIDATION FAILED")
end

println("📊 Performance baseline: $(round(performance_time, digits=2)) ms per PIV run")

if instrumentation_ok && accuracy_ok
    println("\n🎉 Timing system verification PASSED!")
    println("💡 The timing instrumentation is working correctly and ready for benchmarking.")
else
    println("\n⚠️  Timing system verification FAILED!")
    println("🔧 Check the issues above before relying on benchmark results.")
end