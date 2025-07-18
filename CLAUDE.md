# CLAUDE.md

Development guidance for Hammerhead.jl - a modern Julia PIV analysis package.

## Core Principles

1. **No placeholders**: Only implement working, testable functionality
2. **User-friendly APIs**: Symbol-based public APIs with type-stable internals
3. **Property forwarding**: `result.x` instead of `result.vectors.x`
4. **Comprehensive testing**: Realistic test data, edge cases, immediate test fixes
5. **Performance first**: Type stability, pre-allocation, benchmarking

## Key Patterns

### Symbol-to-Type Mapping
```julia
# Public API uses symbols
PIVStage((64,64), window_function=:hanning)

# Internal dispatch uses types
struct _Hanning <: WindowFunction end
window_function_type(s::Symbol) = s == :hanning ? _Hanning() : error("Unknown: $s")
```

### Property Forwarding
```julia
function Base.getproperty(r::PIVResult, s::Symbol)
    s in (:vectors, :metadata, :auxiliary) ? getfield(r, s) : getproperty(r.vectors, s)
end
```

## Development Workflow

### Dependencies
```bash
julia --project=. -e "using Pkg; Pkg.add([\"PackageName\"])"  # Never edit Project.toml manually
```

### Testing
```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

### Performance Analysis
```bash
julia --project=. bench/run_benchmarks.jl  # Comprehensive benchmarks
```

## Architecture

**Current Status (Production Ready)**:
- **PIVVector/PIVResult**: Core data structures with property forwarding
- **PIVStage/PIVStages**: Type-safe configuration with flexible parameter handling
- **run_piv**: Single/multi-stage analysis with automatic timing
- **Timing infrastructure**: TimerOutputs.jl integration for performance monitoring
- **Benchmark suite**: Professional regression testing in `bench/`

## Critical Guidelines

### Implementation
- Use multiple dispatch instead of `isa()` checks
- Handle flexible input types (scalars, vectors, tuples, matrices)
- Use `joinpath()` for cross-platform path handling
- Avoid global state - pass timers/state through function calls

### Testing
- Use `generate_gaussian_particle!` for realistic test data
- Test all constructors, edge cases, and error conditions
- Use appropriate tolerances for data types (Float32 vs Float64)
- Fix broken tests immediately

### Performance
- Validate performance assumptions with benchmarks
- Provide both fast and robust method options
- Use TimerOutputs.jl for detailed performance analysis
- Maintain `bench/` directory for regression testing

## Key Insights

- **Fast ≠ Actually Fast**: Benchmark everything - our "fast" peak detection was slower than "robust"
- **Global state breaks concurrency**: Use local timers passed through function calls
- **Edge cases matter**: Users pass unexpected input types - handle gracefully
- **Domain expertise critical**: Listen to experts about algorithm robustness needs
- **Professional tooling pays off**: Comprehensive benchmarking prevents regressions

## Dependencies

**Core**: FFTW, ImageFiltering, LinearAlgebra, StructArrays, Interpolations, ImageIO, DSP, TimerOutputs

## Performance Targets

- Single-stage PIV: ~0.1-0.2 seconds (128×128 images, 32×32 windows)
- Timing overhead: <3% with detailed performance insights
- Memory efficient for large datasets (200-500 image pairs)