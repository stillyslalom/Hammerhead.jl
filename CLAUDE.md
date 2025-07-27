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
- **Transform validation**: Comprehensive affine transform validation for iterative deformation
- **Vector replacement**: Robust iterative median hole-filling integrated with validation pipeline
- **Validation system**: Comprehensive validator hierarchy with automatic interpolation

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
- **Complex eigenvalues need special handling**: 2D rotations have complex eigenvalues with |λ| = 1
- **Validation is multifaceted**: Area preservation + condition number + eigenvalue bounds all matter
- **Iterative robustness trumps single-pass accuracy**: For inter-stage methods, stability over multiple iterations is more critical than perfect accuracy on one pass
- **Commercial practices > academic sophistication**: Proven industry approaches (iterative median) often outperform complex academic methods in production workflows
- **Error propagation analysis is essential**: Test not just single-pass performance but multi-iteration stability - 9× error amplification would be catastrophic
- **Frequency determines performance requirements**: Inter-stage methods run between every iteration - sub-millisecond performance becomes critical
- **Benchmark realistic workflows, not synthetic cases**: Test the actual usage patterns (frequent calls, iterative workflows) not just isolated method performance
- **Morphological operations for spatial analysis**: ImageMorphology.jl's connected components and dilation are cleaner than manual neighbor searching
- **Realistic test data scales matter**: Use 16×16+ synthetic fields rather than tiny hand-crafted arrays for meaningful integration tests

## Dependencies

**Core**: FFTW, ImageFiltering, LinearAlgebra, StructArrays, Interpolations, ImageIO, DSP, TimerOutputs, ImageMorphology

## Performance Targets

- Single-stage PIV: ~0.1-0.2 seconds (128×128 images, 32×32 windows)
- Timing overhead: <3% with detailed performance insights
- Memory efficient for large datasets (200-500 image pairs)