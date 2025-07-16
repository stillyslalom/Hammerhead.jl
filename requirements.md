# Core PIV Engine Requirements

## Overview
This document specifies the requirements for the core PIV engine `run_piv` function in Hammerhead.jl, designed to handle realistic image pairs with multi-stage processing, iterative deformation, and comprehensive outlier detection.

## Functional Requirements

### 1. Data Structures

#### 1.1 PIVVector
- **Purpose**: Individual vector data point containing position, displacement, and quality metrics
- **Fields**:
  - `x::Float64` - Grid x-coordinate
  - `y::Float64` - Grid y-coordinate  
  - `u::Float64` - Displacement in x-direction
  - `v::Float64` - Displacement in y-direction
  - `status::Symbol` - Vector status (`:good`, `:interpolated`, `:bad`, `:secondary`)
  - `peak_ratio::Float64` - Primary/secondary peak ratio
  - `correlation_moment::Float64` - Correlation peak sharpness metric

#### 1.2 PIVResult
- **Purpose**: Container for complete PIV analysis results
- **Structure**:
  - `vectors::StructArray{PIVVector}` - Vector field data
  - `metadata::Dict{String, Any}` - Processing parameters and run information
  - `auxiliary::Dict{String, Any}` - Additional data (correlation planes, secondary peaks, etc.)
- **Property Forwarding**: `pivdata.x` → `pivdata.vectors.x` via `getproperty`

#### 1.3 PIVStage
- **Purpose**: Configuration for individual processing stage
- **Fields**:
  - `window_size::Tuple{Int, Int}` - Window dimensions (height, width)
  - `overlap::Tuple{Float64, Float64}` - Overlap ratios (vertical, horizontal)
  - `padding::Int` - Window padding pixels
  - `deformation_iterations::Int` - Number of deformation iterations
  - `window_function::Symbol` - Windowing function (`:rectangular`, `:blackman`, `:hanning`, `:hamming`)
  - `interpolation_method::Symbol` - Interpolation method (`:nearest`, `:bilinear`, `:bicubic`, `:spline`, `:lanczos`)

### 2. Core Processing Functions

#### 2.1 run_piv Function
- **Basic Signature**: `run_piv(img1, img2; correlator=CrossCorrelator, kwargs...)`
- **Extended Signature**: `run_piv(img1, img2, stages::Vector{PIVStage}; correlator=CrossCorrelator, kwargs...)`
- **Input**: Two 2D arrays representing consecutive images
- **Output**: PIVResult or Vector{PIVResult} for multi-stage processing
- **Default Behavior**: Single-stage processing with sensible defaults

#### 2.2 Multi-Stage Processing
- **Sequential Refinement**: Each stage uses previous stage results as initial guess
- **Grid Interpolation**: Interpolate coarse grid to fine grid using configurable method
- **Progressive Window Sizing**: Support typical progressions (128→64→32)
- **Per-Stage Configuration**: Individual settings for each refinement stage

#### 2.3 Iterative Deformation
- **Affine Transforms**: Full 6-parameter affine transformation support
- **Area Preservation**: Validate and reject transforms with |det(A)| significantly ≠ 1
- **Convergence Criteria**: Configurable convergence thresholds
- **Maximum Iterations**: Prevent infinite loops with iteration limits

### 3. Windowing and Preprocessing

#### 3.1 Interrogation Window Grid
- **Automatic Generation**: Calculate grid based on image size, window size, and overlap
- **Flexible Overlap**: Support non-uniform overlap ratios
- **Boundary Handling**: Proper handling of edge windows
- **Non-Square Windows**: Support rectangular interrogation windows

#### 3.2 Subimage Windowing Functions
- **Rectangular**: Default, no modification
- **Blackman**: `0.42 - 0.5*cos(2π*n/N) + 0.08*cos(4π*n/N)`
- **Hanning**: `0.5*(1 - cos(2π*n/N))`
- **Hamming**: `0.54 - 0.46*cos(2π*n/N)`
- **Application**: Applied to each interrogation window before correlation

### 4. Outlier Detection and Replacement

#### 4.1 Outlier Detection
- **Per-Iteration**: Run after each deformation iteration
- **Between Stages**: Automatic application between multi-stage processing
- **Configurable Methods**: Support multiple outlier detection algorithms
- **Status Tracking**: Mark outliers with appropriate status flags

#### 4.2 Outlier Replacement
- **Linear Barycentric**: Primary interpolation method for scattered data
- **Configurable Methods**: Support for alternative interpolation algorithms
- **Missing Value Handling**: Fill `missing` values before deformation iterations
- **Quality Preservation**: Maintain quality metrics for interpolated vectors

### 5. Error Handling and Validation

#### 5.1 Correlation Failures
- **Missing Values**: Set displacement to `missing` for failed correlations
- **Status Flags**: Mark failed correlations as `:bad`
- **Graceful Degradation**: Continue processing despite individual window failures

#### 5.2 Input Validation
- **Image Compatibility**: Ensure images have compatible dimensions
- **Configuration Validation**: Validate stage configurations for consistency
- **Memory Bounds**: Check for reasonable memory usage with large images

## Non-Functional Requirements

### 6. Performance Requirements
- **Target Speed**: 29 megapixel image pair in <30 seconds (single-stage)
- **Memory Efficiency**: Minimize memory allocations during processing
- **Scalability**: Support for large image datasets (200-500 pairs)

### 7. Extensibility Requirements
- **Multiple Dispatch**: Support various input types via dispatch
- **Pluggable Correlators**: Accept different correlation algorithms
- **Configurable Processing**: Flexible configuration without code changes

### 8. Testing Requirements
- **Synthetic Data**: Generate test cases with known displacements
- **Validation**: Verify against analytical solutions where possible
- **Edge Cases**: Test boundary conditions and error scenarios

## Implementation Constraints

### 9. Dependencies
- **StructArrays.jl**: For efficient vector field storage
- **ImageIO.jl**: For flexible image loading
- **Interpolations.jl**: For grid interpolation and deformation
- **Existing Codebase**: Integrate with current CrossCorrelator implementation

### 10. Compatibility
- **Julia Version**: Support Julia 1.6+ (as per Project.toml)
- **Type Stability**: Maintain type-stable implementations for performance
- **API Consistency**: Follow established patterns in the codebase

## Success Criteria

### 11. Acceptance Criteria
- **Functional**: All core processing functions work with synthetic data
- **Performance**: Meet target processing speeds on typical hardware
- **Reliability**: Handle edge cases and errors gracefully
- **Maintainability**: Clean, well-documented code with comprehensive tests

### 12. Validation Metrics
- **Accuracy**: Sub-pixel displacement accuracy on synthetic data
- **Robustness**: Successful processing of challenging image pairs
- **Efficiency**: Memory usage and processing time within acceptable bounds