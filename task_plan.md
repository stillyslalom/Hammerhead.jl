# Core PIV Engine Implementation Task Plan

## Overview
This document outlines the detailed implementation plan for the core PIV engine based on the requirements document. Tasks are organized by priority and dependencies.

## Phase 1: Foundation Data Structures (COMPLETED ✅)

### Task 1.1: PIVVector Structure ✅
**Status**: COMPLETED
**Actual Time**: 2 hours

**Completed Implementation**:
- ✅ PIVVector struct with all required fields
- ✅ Type-stable constructors with validation
- ✅ Comprehensive documentation
- ✅ Complete unit tests

### Task 1.2: PIVResult Container ✅  
**Status**: COMPLETED
**Actual Time**: 4 hours

**Completed Implementation**:
- ✅ PIVResult struct with StructArray integration
- ✅ Property forwarding (`result.x` → `result.vectors.x`)
- ✅ Multiple constructor patterns
- ✅ Full test coverage with realistic data

### Task 1.3: PIVStage Configuration ✅
**Status**: COMPLETED
**Actual Time**: 6 hours (expanded scope)

**Completed Implementation**:
- ✅ PIVStage struct with comprehensive validation
- ✅ Symbol-to-type mapping for clean APIs
- ✅ PIVStages helper with flexible parameter handling
- ✅ Dispatch-based parameter conversion (scalars, vectors, tuples, matrices)
- ✅ Extensive documentation with examples

### Task 1.4: Add Dependencies ✅
**Status**: COMPLETED
**Actual Time**: 1 hour

**Completed Implementation**:
- ✅ All dependencies added (StructArrays, Interpolations, ImageIO, TimerOutputs, etc.)
- ✅ Clean import/export structure
- ✅ Version compatibility verified

## Phase 2: Core Processing Infrastructure (COMPLETED ✅)

### Task 2.1: Interrogation Window Grid Generation ✅
**Status**: COMPLETED
**Actual Time**: 3 hours

**Completed Implementation**:
- ✅ `generate_interrogation_grid()` function with boundary handling
- ✅ Non-uniform overlap ratio support
- ✅ Non-square window support
- ✅ Comprehensive edge case testing

### Task 2.2: Subimage Windowing Functions ✅
**Status**: COMPLETED
**Actual Time**: 4 hours

**Completed Implementation**:
- ✅ Complete windowing function system using DSP.jl
- ✅ Support for rectangular, Hanning, Hamming, Blackman, and parametric windows
- ✅ Type-stable dispatch system for performance
- ✅ Separable 2D windowing implementation
- ✅ Integration with PIVStage configuration

### Task 2.3: Basic run_piv Function Structure ✅
**Status**: COMPLETED
**Actual Time**: 6 hours

**Completed Implementation**:
- ✅ Complete `run_piv` function with single/multi-stage support
- ✅ Comprehensive input validation
- ✅ Integration with CrossCorrelator
- ✅ Robust error handling and graceful degradation
- ✅ Extensive integration tests with synthetic data

### Task 2.4: Complete Algorithm Features ✅
**Status**: COMPLETED
**Actual Time**: 6 hours

**Completed Implementation**:
- ✅ Peak ratio calculation with fast and robust methods
- ✅ Correlation moment calculation for peak sharpness
- ✅ Proper window padding with symmetric boundary conditions
- ✅ Comprehensive timing infrastructure with TimerOutputs.jl
- ✅ Input validation for window size vs image compatibility
- ✅ Quality metrics integration at stage level

## Phase 2.5: Performance Infrastructure (COMPLETED ✅)

### Task 2.5: Timing and Performance Measurement ✅
**Status**: COMPLETED
**Actual Time**: 8 hours

**Completed Implementation**:
- ✅ TimerOutputs.jl integration with local timers (thread-safe)
- ✅ Comprehensive timing instrumentation throughout PIV pipeline
- ✅ Performance overhead <3% with detailed insights
- ✅ Professional benchmarking infrastructure in `bench/` directory
- ✅ Fast vs robust algorithm comparison and optimization
- ✅ Performance regression testing framework

## Phase 3: Advanced Processing Features

### Task 3.1: Affine Transform Validation
**Priority**: Medium
**Estimated Time**: 3 hours
**Dependencies**: None

**Implementation Steps**:
1. Implement area-preserving validation function
2. Add determinant checking with configurable tolerance
3. Create transform rejection/acceptance logic
4. Add comprehensive tests for edge cases
5. Document validation criteria

**Acceptance Criteria**:
- Proper area-preservation checking
- Configurable tolerance levels
- Comprehensive edge case handling
- Well-documented validation logic

### Task 3.2: Linear Barycentric Interpolation
**Priority**: Medium
**Estimated Time**: 5 hours
**Dependencies**: PIVResult (1.2)

**Implementation Steps**:
1. Implement linear barycentric interpolation for scattered data
2. Add support for missing value replacement
3. Integrate with PIV processing loop
4. Add nearest neighbor fallback for isolated points
5. Create comprehensive tests with various data patterns

**Acceptance Criteria**:
- Robust interpolation for scattered data
- Proper handling of edge cases
- Integration with main processing loop
- Comprehensive test coverage

### Task 3.3: Iterative Deformation Implementation
**Priority**: Medium
**Estimated Time**: 8 hours
**Dependencies**: Affine Transform (3.1), Interpolation (3.2)

**Implementation Steps**:
1. Implement affine transform application to subimages
2. Add iterative deformation loop with convergence checking
3. Integrate with outlier detection and replacement
4. Add progress tracking and debugging output
5. Create tests with known deformation patterns

**Acceptance Criteria**:
- Robust iterative deformation algorithm
- Proper convergence detection
- Integration with outlier handling
- Validated against synthetic data

## Phase 4: Multi-Stage Processing Enhancement

### Task 4.1: Multi-Stage Processing Logic
**Priority**: Medium
**Estimated Time**: 6 hours
**Dependencies**: Current multi-stage foundation

**Implementation Steps**:
1. Enhance multi-stage processing with initial guess propagation
2. Add grid interpolation between stages
3. Implement proper stage-to-stage data flow
4. Add stage-specific configuration handling
5. Create comprehensive multi-stage tests

**Note**: Basic multi-stage framework exists but needs initial guess propagation

**Acceptance Criteria**:
- Seamless multi-stage processing with initial guess propagation
- Proper grid interpolation between stages
- Stage-specific configuration support
- Comprehensive test coverage

### Task 4.2: Outlier Detection Integration
**Priority**: Medium
**Estimated Time**: 4 hours
**Dependencies**: Multi-Stage Processing (4.1), Interpolation (3.2)

**Implementation Steps**:
1. Implement basic outlier detection algorithms
2. Integrate with per-iteration processing
3. Add configurable outlier detection methods
4. Implement status flag management
5. Create tests with synthetic outliers

**Acceptance Criteria**:
- Robust outlier detection
- Proper integration with processing loop
- Configurable detection methods
- Comprehensive outlier handling

## Phase 5: Test Suite Modernization and Robustness Enhancement

### Task 5.1: Physics & Algorithm Correctness Testing (HIGHEST PRIORITY)
**Priority**: CRITICAL
**Estimated Time**: 8 hours
**Dependencies**: Test cleanup recommendations analysis

**Implementation Steps**:
1. **Realistic Particle Field Generation**: Implement Poisson-distributed particles with diameter/intensity variation, Gaussian+Poisson noise, background gradients
2. **Large/Wrap-around Displacement Tests**: Test displacements approaching ±(window/2) for aliasing detection
3. **Multi-stage & Deformation Validation**: Synthetic flow tests (uniform translation + shear) comparing single vs multi-stage accuracy
4. **Robust Subpixel Testing**: Random 3×3 perturbations, flat-top peaks, closely spaced secondaries, low SNR cases
5. **Quality Metric Realism**: Generate realistic correlation planes with controlled twin peaks

**Acceptance Criteria**:
- Test multiple seeding densities (0.02, 0.05, 0.1 particles/pixel)
- Validate RMS displacement error <0.1 pixels for production quality
- Multi-stage error reduction verification
- Robust handling of near-ambiguous displacements
- Realistic correlation structure validation

### Task 5.2: Determinism & Reproducibility
**Priority**: High
**Estimated Time**: 3 hours
**Dependencies**: None

**Implementation Steps**:
1. **Consistent RNG Seeding**: Implement `@withseed` helper, seed every testset
2. **Performance Test Stabilization**: Add warm-up, environment gating, BenchmarkTools integration
3. **Elimination of Global State**: Ensure no RNG state leakage between tests

**Acceptance Criteria**:
- All tests perfectly reproducible across runs
- Performance tests stable and environment-gated
- No global state dependencies

### Task 5.3: Edge Case & Negative Input Coverage
**Priority**: High  
**Estimated Time**: 6 hours
**Dependencies**: Core infrastructure

**Implementation Steps**:
1. **NaN/Inf Handling**: Test masked regions, infinities with graceful handling
2. **Image Type Coverage**: UInt8, Gray{N0f8} input validation
3. **Non-square Windows**: Full pipeline tests with (48,32) windows, anisotropic overlap
4. **Border Interactions**: Particles outside image boundaries, padding logic validation

**Acceptance Criteria**:
- Graceful handling of degenerate inputs
- Support for standard image types
- Robust boundary condition handling
- Comprehensive error condition coverage

### Task 5.4: Performance & Type Stability Validation
**Priority**: High
**Estimated Time**: 4 hours
**Dependencies**: Core implementation

**Implementation Steps**:
1. **Allocation Checking**: Post-warm zero-allocation verification for Float32/64
2. **Type Inference**: `@inferred` assertions on critical functions
3. **Source Immutability**: Hash verification that input images unchanged
4. **Scaling Tests**: Optional 256², 512² tests with O(N log N) timing validation

**Acceptance Criteria**:
- Zero allocations in steady-state operation
- Type-stable critical path
- Input data immutability guaranteed
- Documented scaling behavior

### Task 5.5: Numerical Robustness & API Contracts
**Priority**: Medium
**Estimated Time**: 4 hours
**Dependencies**: Core algorithms

**Implementation Steps**:
1. **Physics-based Tolerances**: Replace `1e-10` with CRLB estimates or relative tolerances
2. **Degenerate Case Handling**: All-zero/uniform correlation plane behavior
3. **Interpolation Invariants**: Weight sum validation, linearity tests, fallback semantics
4. **Affine Transform Semantics**: Decide reflection policy, condition number bounds

**Acceptance Criteria**:
- Robust tolerance selection methodology
- Well-defined degenerate case behavior
- Mathematical invariant preservation
- Clear transform validation policy

### Task 5.6: Test Infrastructure & Organization
**Priority**: Medium
**Estimated Time**: 5 hours
**Dependencies**: Test analysis

**Implementation Steps**:
1. **Test File Modularization**: Split into thematic files (synthetic, correlator, subpixel, quality, etc.)
2. **Utility Functions**: Extract `generate_particle_field`, `nearest_vector`, `with_seed`, etc.
3. **Property-based Testing**: Random displacement/transform validation
4. **Coverage Integration**: Coverage.jl with CI threshold enforcement

**Acceptance Criteria**:
- Clean modular test organization
- Reusable test utilities
- Property-based fuzz testing
- >95% test coverage with CI enforcement

### Task 5.7: Integration Testing & Validation
**Priority**: High
**Estimated Time**: 6 hours
**Dependencies**: All previous test tasks

**Implementation Steps**:
1. End-to-end integration tests with enhanced synthetic data
2. Multi-stage processing validation with ground truth
3. Performance regression testing with baselines
4. Memory usage profiling and leak detection
5. Thread-safety validation (if applicable)

**Acceptance Criteria**:
- Complete end-to-end validation with realistic data
- Performance targets consistently met
- Memory usage within bounds with no leaks
- Thread-safety verified

### Task 5.8: Documentation and Examples
**Priority**: Medium
**Estimated Time**: 4 hours
**Dependencies**: Completed implementation

**Implementation Steps**:
1. Comprehensive API documentation with doctests
2. Usage examples for common scenarios
3. Tutorial notebook with realistic examples
4. Performance characteristics documentation
5. Troubleshooting guide with common issues

**Acceptance Criteria**:
- Complete API documentation with working examples
- User-friendly tutorials with real data
- Performance documentation with benchmarks
- Comprehensive troubleshooting resources

## Phase 6: Integration and Optimization

### Task 6.1: Performance Optimization
**Priority**: Medium
**Estimated Time**: 6 hours
**Dependencies**: Integration Testing (5.2)

**Implementation Steps**:
1. Profile code for performance bottlenecks
2. Optimize memory allocations
3. Improve algorithm efficiency where possible
4. Add parallel processing where appropriate
5. Validate performance improvements

**Acceptance Criteria**:
- Performance targets consistently met
- Efficient memory usage
- Scalable to large datasets
- Comprehensive performance testing

### Task 6.2: Error Handling and Robustness
**Priority**: Medium
**Estimated Time**: 4 hours
**Dependencies**: All previous tasks

**Implementation Steps**:
1. Implement comprehensive error handling
2. Add graceful degradation for edge cases
3. Improve logging and debugging output
4. Add input validation and sanitization
5. Create robustness tests

**Acceptance Criteria**:
- Robust error handling throughout
- Graceful degradation for edge cases
- Comprehensive logging system
- Validated robustness

## Timeline Summary

**Phase 1**: Foundation Data Structures ✅ (13 hours completed)
**Phase 2**: Core Processing Infrastructure ✅ (23 hours completed)
**Phase 2.5**: Performance Infrastructure ✅ (8 hours completed)
**Phase 3**: Advanced Processing Features (16 hours)
**Phase 4**: Multi-Stage Processing Enhancement (10 hours)
**Phase 5**: Test Suite Modernization and Robustness Enhancement (40 hours)
**Phase 6**: Integration and Optimization (10 hours)

**Total Estimated Effort**: 120 hours
**Progress**: 44/120 hours completed (37%)
**Next Priority**: Critical test suite modernization (Phase 5.1) - Physics & Algorithm Correctness

**Critical Path**: Test suite robustness is now the highest priority due to:
- Expert feedback identifying significant testing gaps
- Risk of undetected algorithm failures in production
- Need for physics-based validation methodology
- Foundation for reliable advanced feature development

## Risk Assessment (Updated Based on Expert Test Analysis)

**CRITICAL Risk** (Immediate Action Required):
- **Test suite inadequacy**: Current tests lack realistic particle fields, large displacements, noise handling
- **Physics validation gaps**: Missing multi-stage efficacy validation, wrap-around displacement detection  
- **Production readiness**: Insufficient edge case coverage, brittle numerical tolerances
- **Algorithm correctness**: Unvalidated quality metrics, unrealistic correlation structures

**High Risk**:
- Iterative deformation algorithm complexity (depends on robust testing first)
- Initial guess propagation between stages (needs validated foundation)
- Outlier detection algorithm selection (requires realistic test data)
- **Test reproducibility**: Inconsistent RNG seeding, global state leakage
- **Type stability**: Missing `@inferred` verification on critical paths

**Medium Risk**:
- Affine transform validation implementation (semantic clarity needed)
- Linear barycentric interpolation for scattered data (well-understood algorithm)
- Multi-stage processing with proper grid interpolation (testable incrementally)
- **Performance test stability**: Current benchmarks may be noisy

**Low Risk**:
- Basic data structure implementation ✅ (completed)
- Core PIV processing pipeline ✅ (completed, but needs robust testing)
- Windowing function implementation ✅ (completed, DSP.jl foundation solid)
- Performance infrastructure ✅ (completed, professional tooling in place)
- Quality metric calculations ✅ (implemented, but validation gaps identified)
- Window padding implementation ✅ (completed, boundary conditions working)

**Performance Note**: Core FFT and correlation operations are properly optimized with <3% timing overhead. Professional benchmarking infrastructure is in place for regression testing.

## Success Metrics (Enhanced Based on Expert Analysis)

**Functional Success**:
- **Physics Validation**: RMS displacement error <0.1 pixels on realistic synthetic data
- **Robustness**: Graceful handling of NaN/Inf inputs, edge cases, large displacements
- **Determinism**: Perfect reproducibility across platforms with proper RNG seeding
- **Performance**: Targets met with zero allocation steady-state operation
- **Type Stability**: All critical paths verified with `@inferred`
- **Coverage**: >95% test coverage with CI threshold enforcement

**Algorithm Correctness**:
- **Multi-stage Efficacy**: Demonstrated error reduction through processing stages
- **Quality Metrics**: Validated against realistic correlation structures  
- **Subpixel Accuracy**: Robust handling of noise, asymmetry, closely spaced peaks
- **Transform Validation**: Clear policies for area preservation, condition bounds
- **Interpolation**: Mathematical invariants preserved (linearity, weight conservation)

**Production Readiness**:
- **Realistic Test Data**: Poisson particle distributions, proper noise models
- **Edge Case Coverage**: Boundary conditions, degenerate inputs, wrap-around displacements
- **Scaling Validation**: O(N log N) behavior verified up to production image sizes
- **Memory Safety**: No leaks, bounded allocation patterns, immutable inputs

**Quality Success**:
- **Physics-based Tolerances**: CRLB-derived or relative tolerance methodology
- **Test Organization**: Modular structure with reusable utilities
- **Property-based Testing**: Fuzz testing for random inputs within valid ranges
- **Documentation**: Complete API docs with working doctests
- **CI Integration**: Automated regression testing with performance baselines

## Immediate Action Plan (Based on Expert Test Analysis)

### Phase 5.1 Implementation Priority (Next 8 Hours)

**Week 1: Critical Physics & Algorithm Validation**
1. **Day 1-2**: Realistic particle field generator with Poisson distribution, noise models
2. **Day 3**: Large displacement tests (±window/2), wrap-around detection  
3. **Day 4**: Multi-stage efficacy validation with synthetic flows
4. **Day 5**: Robust subpixel testing with perturbations, flat peaks, low SNR

**Week 2: Reproducibility & Edge Cases**  
1. **Day 6**: RNG seeding standardization, `@withseed` helper implementation
2. **Day 7**: NaN/Inf input handling, image type coverage expansion
3. **Day 8**: Performance test stabilization, allocation checking

**Expected Outcomes**:
- Production-quality physics validation (RMS error <0.1 pixels)
- Robust handling of realistic experimental conditions
- Deterministic, reproducible test suite
- Foundation for confident advanced feature development

**Risk Mitigation**: This test modernization is now the critical path - advanced features (iterative deformation, multi-stage enhancement) should not proceed until robust validation foundation is established.

**Resource Allocation**: The expanded 40-hour test suite investment will prevent much larger costs from undetected algorithmic failures in production deployments.