# Hammerhead.jl Benchmarks

This directory contains performance benchmarks for Hammerhead.jl to ensure optimal performance and detect regressions.

## Running Benchmarks

From the package root directory:

```bash
# Run all benchmarks
julia --project=. bench/run_benchmarks.jl

# Run specific benchmark
julia --project=. bench/benchmark_fast_vs_robust.jl
julia --project=. bench/detailed_peak_benchmark.jl
```

## Benchmark Categories

### Core Performance (`benchmark_fast_vs_robust.jl`)
- **Secondary peak detection**: Fast vs robust methods comparison
- **Window function performance**: Different windowing functions
- **Correlation scaling**: FFT-based correlation at different sizes
- **Full PIV analysis**: End-to-end performance measurement

### Algorithm Analysis (`detailed_peak_benchmark.jl`)
- **Computational complexity**: Operation counting and analysis
- **Scaling behavior**: Performance vs correlation plane size
- **Implementation optimization**: Identifies bottlenecks

## Expected Performance Targets

Based on current implementation:

### Peak Detection (64×64 correlation plane)
- **Fast method**: ~3-7 μs per call
- **Robust method**: ~10-15 μs per call
- **Target**: Fast method should be 2-5x faster than robust

### FFT Correlation
- **32×32**: ~5 μs
- **64×64**: ~20 μs  
- **128×128**: ~100 μs
- **256×256**: ~400 μs

### Full PIV Analysis (128×128 images, 32×32 windows)
- **Single-stage**: ~0.1-0.2 seconds
- **Multi-stage (3 stages)**: ~0.3-0.5 seconds

## Regression Detection

When making performance-critical changes:

1. **Run baseline**: `julia --project=. bench/run_benchmarks.jl > baseline.log`
2. **Make changes**
3. **Run comparison**: `julia --project=. bench/run_benchmarks.jl > new.log`
4. **Compare results**: Look for >20% performance regressions

## Performance Issues Found

### Current Findings
- **Fast vs Robust Peak Detection**: The "fast" method is currently slower than "robust" for typical correlation plane sizes
- **Root cause**: Fast method processes every pixel with exclusion checks; robust method only processes local maxima
- **Status**: Optimization opportunities identified, implementation improvements ongoing

### Improvement Opportunities
1. **Fast peak detection**: Use early termination strategies
2. **Window functions**: Rectangular vs DSP.jl functions have different performance characteristics
3. **Memory allocation**: Some operations could benefit from pre-allocation

## Adding New Benchmarks

To add a new benchmark:

1. Create `bench/benchmark_<name>.jl`
2. Follow the existing pattern:
   - Import required internal functions if needed
   - Use TimerOutputs for precise measurements
   - Include multiple iterations for statistical significance
   - Report results in μs/ms/s as appropriate
3. Add to `bench/run_benchmarks.jl`
4. Update this README with performance targets

## Integration with CI/CD

Future: Integrate benchmarks into release process to automatically detect performance regressions before releases.