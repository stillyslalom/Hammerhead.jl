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

**Rationale**: Avoids conflicts with packages like DSP.jl that export `Hanning`, while providing clean user interface.

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
- **run_piv**: Single and multi-stage PIV analysis
- **generate_interrogation_grid**: Grid generation with overlap handling

### Implementation Guidelines

**When implementing new features**:
1. Check existing patterns in the codebase first
2. Follow the symbol-to-type mapping pattern for user APIs
3. Implement complete, testable functionality - no stubs
4. Add comprehensive tests alongside implementation
5. Use realistic test data (Gaussian particles, not synthetic matrices)
6. Handle edge cases and errors gracefully

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
- **Edge cases**: Boundary conditions, error scenarios
- **Performance**: Memory allocation and timing benchmarks

## Key Lessons from Development

1. **Start with data structures**: Get the foundation right before building algorithms
2. **User experience matters**: Property forwarding and symbol APIs improve usability
3. **Type safety enables performance**: Use Julia's type system for dispatch and optimization
4. **Test with realistic data**: Gaussian particles reveal issues that synthetic data doesn't
5. **Avoid premature optimization**: Build working functionality first, optimize later
6. **Namespace management**: Internal types prevent conflicts with ecosystem packages