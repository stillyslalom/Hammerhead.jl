# Handoff Plan for Core PIV Engine Development

## Current Status

**Session Date**: 2025-01-09
**Repository State**: Reset to commit 6383e06 (clean slate with basic CrossCorrelator)
**Phase**: Planning Complete, Ready for Implementation

## Completed Work

### Documentation
- ✅ **Requirements Document** (`requirements.md`) - Complete specification for core PIV engine
- ✅ **Task Plan** (`task_plan.md`) - 6-week implementation plan with detailed tasks
- ✅ **CLAUDE.md** - Updated with current project state and guidance
- ✅ **Q&A Session** - Comprehensive requirements gathering completed

### Key Decisions Made
- **Data Structure**: PIVResult with StructArray{PIVWindow} + metadata/auxiliary dicts
- **Processing**: Multi-stage with iterative deformation and outlier detection
- **Windowing**: Support for rectangular, Blackman, Hanning, Hamming functions
- **Interpolation**: Linear barycentric for outlier replacement, configurable methods
- **Validation**: Area-preserving affine transforms with determinant checking
- **Testing**: Start with synthetic data, realistic test cases later

## Next Session Action Plan

### Immediate Tasks (Phase 1: Foundation - Week 1)

**Start Here**: Task 1.1 - PIVWindow Structure
```julia
# Define in src/Hammerhead.jl
struct PIVWindow
    x::Float64          # Grid x-coordinate
    y::Float64          # Grid y-coordinate  
    u::Float64          # Displacement in x-direction
    v::Float64          # Displacement in y-direction
    status::Symbol      # :good, :interpolated, :bad, :secondary
    peak_ratio::Float64 # Primary/secondary peak ratio
    correlation_moment::Float64 # Correlation peak sharpness
end
```

**Then**: Task 1.4 - Add Dependencies
```toml
# Add to Project.toml [deps]
StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
Interpolations = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
ImageIO = "82e4d734-157c-48bb-816b-45c225c6df19"
```

**Next**: Task 1.2 - PIVResult with getproperty forwarding
```julia
struct PIVResult
    vectors::StructArray{PIVWindow}
    metadata::Dict{String, Any}
    auxiliary::Dict{String, Any}
end

# Property forwarding: pivdata.x → pivdata.vectors.x
Base.getproperty(r::PIVResult, s::Symbol) = s in (:vectors, :metadata, :auxiliary) ? 
    getfield(r, s) : getproperty(getfield(r, :vectors), s)
```

**Finally**: Task 1.3 - PIVStage Configuration
```julia
struct PIVStage
    window_size::Tuple{Int, Int}        # (height, width)
    overlap::Tuple{Float64, Float64}    # (vertical, horizontal)
    padding::Int
    deformation_iterations::Int
    window_function::Symbol             # :rectangular, :blackman, :hanning, :hamming
    interpolation_method::Symbol        # :nearest, :bilinear, :bicubic, :spline, :lanczos
end

# Helper constructor
PIVStages(n_stages::Int, final_size::Int, overlap::Float64) = # Implementation needed
```

### Success Criteria for Next Session
- [ ] All Phase 1 tasks completed and tested
- [ ] PIVWindow struct functional with tests
- [ ] PIVResult with working property forwarding
- [ ] PIVStage configuration with helper constructors
- [ ] Dependencies added and importing correctly
- [ ] Basic test suite passing

## Key Files to Modify

### Primary Implementation
- `src/Hammerhead.jl` - Add new structs and exports
- `Project.toml` - Add new dependencies
- `test/runtests.jl` - Add tests for new structs

### Reference Documents
- `requirements.md` - Detailed specification (READ FIRST)
- `task_plan.md` - Complete implementation roadmap
- `CLAUDE.md` - Development guidance

## Implementation Notes

### Critical Design Decisions
1. **StructArrays**: Efficient storage for large vector fields
2. **Property Forwarding**: Clean API (`result.x` instead of `result.vectors.x`)
3. **Missing Values**: Use `missing` for failed correlations, not NaN
4. **Status Flags**: Track vector quality (`:good`, `:interpolated`, `:bad`, `:secondary`)
5. **Area Preservation**: Validate affine transforms with `|det(A)| ≈ 1`

### Performance Targets
- 29 megapixel image pair in <30 seconds (single-stage)
- Memory efficient with large datasets (200-500 pairs)
- Type-stable implementations throughout

### Testing Strategy
- Unit tests for each struct and function
- Synthetic data with known displacements
- Integration tests for complete workflows
- Performance benchmarks on target hardware

## Potential Challenges

### Known Issues to Watch
1. **Memory Usage**: Large StructArrays with auxiliary data
2. **Type Stability**: getproperty forwarding complexity
3. **Interpolation Integration**: Scattered data handling
4. **Affine Transform Validation**: Tolerance tuning

### Debugging Resources
- `requirements.md` - Full specification for reference
- `task_plan.md` - Detailed implementation steps
- Existing `CrossCorrelator` - Working correlation example
- Test cases in `test/runtests.jl` - Validation patterns

## Communication Context

### Session History
- Extensive Q&A on requirements and design
- Consensus on StructArrays approach
- Agreement on multi-stage processing architecture
- Validation of synthetic testing strategy

### User Preferences
- Clean, well-documented code
- Modular design with multiple dispatch
- Performance-oriented implementation
- Comprehensive testing with synthetic data

## Quick Start Command

```julia
# Navigate to project
cd("/home/alex/.julia/dev/Hammerhead")

# Review requirements
# Open requirements.md and task_plan.md

# Start implementation
# Begin with Task 1.1 - PIVWindow struct in src/Hammerhead.jl
```

## Ready State Verification

Before starting implementation, verify:
- [ ] Repository at commit 6383e06
- [ ] Requirements document reviewed
- [ ] Task plan understood
- [ ] Development environment ready
- [ ] CLAUDE.md guidance internalized

**Status**: READY FOR IMPLEMENTATION 🚀