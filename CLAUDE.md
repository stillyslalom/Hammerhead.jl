# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hammerhead.jl is a Julia package for Particle Image Velocimetry (PIV) analysis, designed to provide modern, high-performance capabilities on par with the Prana library. The project targets shock tube experiments with turbulent mixing regions involving subsonic post-shock flow with significant shear and compressibility effects.

## Development Philosophy and Key Design Decisions

### 1. No Placeholders or Stubs
**Critical Rule**: Never add placeholder functions or stub implementations. Only implement functionality that actually works and can be tested. If a feature isn't ready to be built, don't add empty functions for it.

**Example**: Don't create empty `apply_windowing_function()` - instead, implement the actual windowing when ready.

### 2. Symbol-to-Type Mapping for Clean APIs
Use symbols in public APIs but convert to concrete types internally for type-stable dispatch and to avoid namespace collisions.

**Pattern**:
```julia
# Internal types (not exported)
abstract type WindowFunction end
struct _Rectangular <: WindowFunction end
struct _Hanning <: WindowFunction end

# Symbol mapping function
function window_function_type(s::Symbol)
    s == :rectangular && return _Rectangular()
    s == :hanning && return _Hanning()
    throw(ArgumentError("Unknown window function: $s"))
end

# Constructor uses symbols, stores types
PIVStage((64,64), window_function=:hanning)  # User-friendly
```

**Rationale**: Provides clean user interface while leveraging robust implementations from packages like DSP.jl internally.

### 3. Property Forwarding for Ergonomic APIs
Implement `Base.getproperty` to provide direct access to nested data structures.

**Pattern**:
```julia
# Instead of result.vectors.x, allow result.x
function Base.getproperty(r::PIVResult, s::Symbol)
    if s in (:vectors, :metadata, :auxiliary)
        return getfield(r, s)
    else
        return getproperty(getfield(r, :vectors), s)
    end
end
```

### 4. Use Proper Dependency Management
Always use `Pkg.add` instead of manual Project.toml editing to ensure correctness and proper version resolution.

**Command**: `julia --project=. -e "using Pkg; Pkg.add([\"StructArrays\", \"Interpolations\"])"`

### 5. Comprehensive Testing Strategy
Test functionality that exists thoroughly, but don't test placeholders.

**Approach**:
- Use realistic test data (`generate_gaussian_particle!`) instead of artificial patterns
- Test all constructors and edge cases
- Test property forwarding and API ergonomics
- Test error conditions and validation

### 6. Organized Exports by Functionality
Group exports by functionality rather than alphabetically:

```julia
# Data structures
export PIVVector, PIVResult, PIVStage

# Core functionality  
export run_piv, CrossCorrelator, correlate

# Utilities
export subpixel_gauss3
```

### 7. Type Stability and Performance
- Use parametric types where beneficial (`PIVStage{W<:WindowFunction, I<:InterpolationMethod}`)
- Pre-allocate arrays for performance-critical operations
- Use `Vector{<:PIVStage}` to handle parametric type collections

## Common Development Tasks

### Running Tests
```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

### Adding Dependencies
```bash
julia --project=. -e "using Pkg; Pkg.add([\"PackageName\"])"
```

### Testing Individual Components
```bash
julia --project=. -e "using Hammerhead; # test specific functionality"
```

## Architecture Notes

### Current Status (Phase 1 Complete)
- **PIVVector**: Individual vector data with position, displacement, quality metrics
- **PIVResult**: StructArray container with property forwarding 
- **PIVStage**: Type-safe configuration with symbol-to-type mapping
- **PIVStages**: Multi-stage helper with flexible parameter handling (scalars/vectors/tuples/matrices)
- **run_piv**: Single and multi-stage PIV analysis
- **generate_interrogation_grid**: Grid generation with overlap handling

### Recent Enhancements
- **Flexible parameter APIs**: PIVStages accepts scalars, vectors, tuples, matrices, ranges
- **Robust dispatch**: Multiple dispatch for parameter handling instead of runtime checks
- **Comprehensive edge case handling**: Proper conversion and validation for all input types
- **Enhanced testing**: 165 tests covering all functionality and edge cases

### Implementation Guidelines

**When implementing new features**:
1. Check existing patterns in the codebase first
2. Follow the symbol-to-type mapping pattern for user APIs
3. Implement complete, testable functionality - no stubs
4. Add comprehensive tests alongside implementation
5. Use realistic test data (Gaussian particles, not synthetic matrices)
6. Handle edge cases and errors gracefully
7. Use multiple dispatch instead of runtime type checking (`isa()`)
8. Design flexible APIs that accept multiple input formats (scalars, vectors, tuples, matrices)
9. Fix any broken tests before adding new functionality

**When encountering namespace conflicts**:
- Use internal types with underscores (`_Rectangular`)
- Provide symbol-based constructors for users
- Keep internal types private (don't export)

**When adding tests**:
- Test actual functionality, not placeholders
- Use `generate_gaussian_particle!` for realistic test scenarios
- Test all constructors and validation logic
- Verify property forwarding works correctly

## Performance Targets

- Process 29 megapixel image pair in <30 seconds (single-stage)
- Memory efficient with large datasets (200-500 pairs)
- Type-stable implementations throughout

## Dependencies

Core dependencies (managed via Pkg.add):
- **FFTW**: Fast Fourier Transform library for correlation
- **ImageFiltering**: Image processing utilities  
- **LinearAlgebra**: Matrix operations (mul!, ldiv!, inv)
- **StructArrays**: Efficient storage for vector fields
- **Interpolations**: Grid interpolation and resampling
- **ImageIO**: Flexible image loading

## Testing Strategy

- **Unit tests**: Each data structure and function
- **Integration tests**: Complete PIV workflows with synthetic data
- **Realistic data**: Gaussian particles with known displacements
- **Edge cases**: Boundary conditions, error scenarios, alternative input formats
- **Performance**: Memory allocation and timing benchmarks
- **Test structure**: Use end-block comments for complex nested testsets
- **Comprehensive coverage**: Test tuples, matrices, ranges, and error conditions
- **Fix failures immediately**: Don't accumulate broken tests

## Key Lessons from Development

1. **Start with data structures**: Get the foundation right before building algorithms
2. **User experience matters**: Property forwarding and symbol APIs improve usability
3. **Type safety enables performance**: Use Julia's type system for dispatch and optimization
4. **Test with realistic data**: Gaussian particles reveal issues that synthetic data doesn't
5. **Avoid premature optimization**: Build working functionality first, optimize later
6. **Namespace management**: Internal types prevent conflicts with ecosystem packages

## Recent Development Insights

7. **Fix broken tests immediately**: Don't accumulate technical debt - address failing tests before adding features
8. **Use dispatch over runtime type checks**: Replace `isa()` checks with multiple dispatch for cleaner, more performant code
9. **Handle edge cases comprehensively**: Users will pass tuples, matrices, ranges - design APIs to be flexible and convert appropriately
10. **Test edge cases extensively**: Include tests for tuples, matrices, ranges, and error cases to ensure robust behavior
11. **Document parameter flexibility**: When functions accept multiple input types, document all supported formats clearly
12. **CI compatibility matters**: Keep Julia version support current (LTS minimum, not ancient versions like 1.6)
13. **Test structure is critical**: Use end-block comments (`end # TestsetName`) to debug complex nested test structures
14. **Use @show for tolerance optimization**: Rather than iteratively guessing test tolerances, add temporary @show statements to measure actual accuracy and set appropriate tolerances based on real performance
15. **Documentation should be general**: Avoid mentioning specific experimental details (image sizes, experimental setups) in public-facing documentation to keep it broadly applicable
16. **Commit messages should be factual**: Avoid hyperbolic language about "improvements" when previous values weren't carefully set - just state what was done
17. **Domain expertise matters for robustness**: Listen to domain experts about edge cases (e.g., closely spaced peaks in high shear flows) and implement robust alternatives alongside fast methods
18. **Local maxima detection needs careful design**: Simple 3x3 neighborhood checking fails for broad peaks - need to merge adjacent maxima and sort by magnitude to find true peaks
19. **Provide speed vs robustness options**: In performance-critical applications, offer both fast and robust methods rather than one-size-fits-all solutions
20. **Global state is problematic for concurrency**: Avoid global timers/state that cause issues in multiprocessing environments - use local state passed through function calls instead
21. **Performance assumptions need validation**: Methods labeled "fast" should actually be benchmarked - we discovered our "fast" peak detection was slower than the "robust" method due to expensive sqrt operations
22. **Comprehensive timing infrastructure pays dividends**: TimerOutputs.jl with hierarchical instrumentation provides invaluable performance insights with minimal overhead (<3%)
23. **Professional benchmarking infrastructure enables regression testing**: A dedicated bench/ directory with organized benchmarks helps maintain performance standards over time
24. **Path concatenation should use joinpath()**: Use Julia's builtin joinpath() function instead of string concatenation for cross-platform compatibility