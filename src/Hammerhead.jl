module Hammerhead

using FFTW
using LinearAlgebra: mul!, ldiv!, inv
using ImageFiltering
using StructArrays
using Interpolations
using ImageIO

# Data structures
export PIVVector, PIVResult, PIVStage, PIVStages

# Core functionality  
export run_piv, CrossCorrelator, correlate!, analyze_correlation_plane

# Utilities
export subpixel_gauss3

# Define a Correlator type to encapsulate correlation methods and options
abstract type Correlator end

"""
    PIVVector

Individual vector data point containing position, displacement, and quality metrics.

# Fields
- `x::Float64` - Grid x-coordinate
- `y::Float64` - Grid y-coordinate  
- `u::Float64` - Displacement in x-direction
- `v::Float64` - Displacement in y-direction
- `status::Symbol` - Vector status (`:good`, `:interpolated`, `:bad`, `:secondary`)
- `peak_ratio::Float64` - Primary/secondary peak ratio
- `correlation_moment::Float64` - Correlation peak sharpness metric
"""
struct PIVVector
    x::Float64
    y::Float64
    u::Float64
    v::Float64
    status::Symbol
    peak_ratio::Float64
    correlation_moment::Float64
end

# Constructor with default values for quality metrics
function PIVVector(x::Real, y::Real, u::Real, v::Real, status::Symbol=:good)
    PIVVector(Float64(x), Float64(y), Float64(u), Float64(v), status, NaN, NaN)
end

# Constructor with all parameters
function PIVVector(x::Real, y::Real, u::Real, v::Real, status::Symbol, 
                  peak_ratio::Real, correlation_moment::Real)
    PIVVector(Float64(x), Float64(y), Float64(u), Float64(v), status, 
              Float64(peak_ratio), Float64(correlation_moment))
end

"""
    PIVResult

Container for complete PIV analysis results with property forwarding.

# Fields
- `vectors::StructArray{PIVVector}` - Vector field data
- `metadata::Dict{String, Any}` - Processing parameters and run information  
- `auxiliary::Dict{String, Any}` - Additional data (correlation planes, secondary peaks, etc.)

# Property Forwarding
Provides direct access to vector field properties:
- `result.x` → `result.vectors.x`
- `result.u` → `result.vectors.u`
- etc.
"""
struct PIVResult
    vectors::StructArray{PIVVector}
    metadata::Dict{String, Any}
    auxiliary::Dict{String, Any}
end

# Property forwarding: pivdata.x → pivdata.vectors.x
function Base.getproperty(r::PIVResult, s::Symbol)
    if s in (:vectors, :metadata, :auxiliary)
        return getfield(r, s)
    else
        return getproperty(getfield(r, :vectors), s)
    end
end

# Constructor from grid size
function PIVResult(grid_size::Tuple{Int, Int})
    n_vectors = prod(grid_size)
    vectors = StructArray{PIVVector}(undef, grid_size)
    metadata = Dict{String, Any}()
    auxiliary = Dict{String, Any}()
    PIVResult(vectors, metadata, auxiliary)
end

# Constructor with vector of PIVVector
function PIVResult(piv_vectors::AbstractArray{PIVVector})
    vectors = StructArray(piv_vectors)
    metadata = Dict{String, Any}()
    auxiliary = Dict{String, Any}()
    PIVResult(vectors, metadata, auxiliary)
end

# Window function types for type-stable dispatch (internal)
abstract type WindowFunction end
struct _Rectangular <: WindowFunction end
struct _Blackman <: WindowFunction end
struct _Hanning <: WindowFunction end
struct _Hamming <: WindowFunction end

# Interpolation method types for type-stable dispatch (internal)
abstract type InterpolationMethod end
struct _Nearest <: InterpolationMethod end
struct _Bilinear <: InterpolationMethod end
struct _Bicubic <: InterpolationMethod end
struct _Spline <: InterpolationMethod end
struct _Lanczos <: InterpolationMethod end

# Symbol to type mapping for window functions
function window_function_type(s::Symbol)
    if s == :rectangular
        return _Rectangular()
    elseif s == :blackman
        return _Blackman()
    elseif s == :hanning
        return _Hanning()
    elseif s == :hamming
        return _Hamming()
    else
        throw(ArgumentError("Unknown window function: $s. Supported: :rectangular, :blackman, :hanning, :hamming"))
    end
end

# Symbol to type mapping for interpolation methods
function interpolation_method_type(s::Symbol)
    if s == :nearest
        return _Nearest()
    elseif s == :bilinear
        return _Bilinear()
    elseif s == :bicubic
        return _Bicubic()
    elseif s == :spline
        return _Spline()
    elseif s == :lanczos
        return _Lanczos()
    else
        throw(ArgumentError("Unknown interpolation method: $s. Supported: :nearest, :bilinear, :bicubic, :spline, :lanczos"))
    end
end

"""
    PIVStage{W<:WindowFunction, I<:InterpolationMethod}

Configuration for individual processing stage in multi-stage PIV analysis.

# Fields
- `window_size::Tuple{Int, Int}` - Window dimensions (height, width)
- `overlap::Tuple{Float64, Float64}` - Overlap ratios (vertical, horizontal) ∈ [0, 1)
- `padding::Int` - Window padding pixels
- `deformation_iterations::Int` - Number of deformation iterations
- `window_function::W` - Windowing function type
- `interpolation_method::I` - Interpolation method type
"""
struct PIVStage{W<:WindowFunction, I<:InterpolationMethod}
    window_size::Tuple{Int, Int}
    overlap::Tuple{Float64, Float64}
    padding::Int
    deformation_iterations::Int
    window_function::W
    interpolation_method::I
end

# Constructor with defaults using symbols
function PIVStage(window_size::Tuple{Int, Int}; 
                 overlap::Tuple{Float64, Float64} = (0.5, 0.5),
                 padding::Int = 0,
                 deformation_iterations::Int = 3,
                 window_function::Symbol = :rectangular,
                 interpolation_method::Symbol = :bilinear)
    
    # Validate overlap ratios
    if any(x -> x < 0 || x >= 1, overlap)
        throw(ArgumentError("Overlap ratios must be in [0, 1)"))
    end
    
    # Validate window size
    if any(x -> x <= 0, window_size)
        throw(ArgumentError("Window size must be positive"))
    end
    
    # Convert symbols to types
    wf = window_function_type(window_function)
    im = interpolation_method_type(interpolation_method)
    
    PIVStage(window_size, overlap, padding, deformation_iterations, wf, im)
end

# Helper constructor for square windows
PIVStage(window_size::Int; kwargs...) = PIVStage((window_size, window_size); kwargs...)

"""
    PIVStages(n_stages::Int, final_size::Int; kwargs...) -> Vector{PIVStage}

Create a vector of PIVStage configurations for multi-stage PIV analysis with geometric window size progression.

# Arguments
- `n_stages::Int` - Number of processing stages
- `final_size::Int` - Final window size (smallest stage)

# Keyword Arguments
All PIVStage parameters are supported with either scalar or vector values:
- `overlap=0.5` - Overlap ratio(s). Can be scalar, tuple, or vector of scalars/tuples
- `padding=0` - Padding pixel(s). Can be scalar or vector
- `deformation_iterations=3` - Deformation iteration(s). Can be scalar or vector  
- `window_function=:rectangular` - Window function(s). Can be scalar symbol or vector of symbols
- `interpolation_method=:bilinear` - Interpolation method(s). Can be scalar symbol or vector of symbols

# Parameter Handling
- **Scalar values**: Applied to all stages
- **Vector values**: Must have length 1 (applied to all) or length `n_stages` (one per stage)
- **Overlap**: Scalar values create symmetric overlap `(val, val)`, tuples are used directly

# Window Size Progression
Stages use geometric progression: `final_size * 2^(n_stages - i)` for stage `i`, 
with minimum size constrained to `final_size`.

# Examples
```julia
# Basic usage with scalar parameters
stages = PIVStages(3, 32, overlap=0.5, window_function=:hanning)

# Mixed scalar and vector parameters  
stages = PIVStages(3, 32, overlap=[0.75, 0.5, 0.25], padding=5)

# Different window functions per stage
stages = PIVStages(2, 32, window_function=[:rectangular, :hanning])

# Asymmetric overlap
stages = PIVStages(2, 32, overlap=(0.6, 0.4))
```
"""
function PIVStages(n_stages::Int, final_size::Int; 
                   overlap=0.5, 
                   padding=0,
                   deformation_iterations=3,
                   window_function=:rectangular,
                   interpolation_method=:bilinear)
    
    if n_stages <= 0
        throw(ArgumentError("Number of stages must be positive"))
    end
    
    # Helper functions to get value for stage i (1-indexed) using dispatch
    
    # Handle scalar values (numbers, symbols, etc.)
    get_stage_value(param::Union{Number, Symbol}, i::Int, n_stages::Int) = param
    
    # Handle vectors
    function get_stage_value(param::AbstractVector, i::Int, n_stages::Int)
        if length(param) == 1
            return param[1]  # Single value for all stages
        elseif length(param) == n_stages
            return param[i]  # One value per stage
        else
            throw(ArgumentError("Parameter vector length ($(length(param))) must be 1 or equal to n_stages ($n_stages)"))
        end
    end
    
    # Handle tuples - convert to vector if appropriate size
    function get_stage_value(param::Tuple, i::Int, n_stages::Int)
        vec_param = collect(param)  # Convert tuple to vector
        return get_stage_value(vec_param, i, n_stages)  # Delegate to vector method
    end
    
    # Handle 1×n or n×1 matrices - convert to vector if appropriate
    function get_stage_value(param::AbstractMatrix, i::Int, n_stages::Int)
        if size(param, 1) == 1
            # 1×n matrix - convert to vector
            vec_param = vec(param)  # Flatten to vector
            return get_stage_value(vec_param, i, n_stages)
        elseif size(param, 2) == 1
            # n×1 matrix - convert to vector  
            vec_param = vec(param)  # Flatten to vector
            return get_stage_value(vec_param, i, n_stages)
        else
            throw(ArgumentError("Matrix parameters must be 1×n or n×1, got $(size(param))"))
        end
    end
    
    # Handle other iterables - convert to vector if possible
    function get_stage_value(param, i::Int, n_stages::Int)
        if hasmethod(iterate, (typeof(param),)) && hasmethod(length, (typeof(param),))
            # It's an iterable with known length - convert to vector
            try
                vec_param = collect(param)
                return get_stage_value(vec_param, i, n_stages)
            catch e
                throw(ArgumentError("Cannot convert parameter of type $(typeof(param)) to vector: $e"))
            end
        else
            throw(ArgumentError("Unsupported parameter type: $(typeof(param)). Use scalar values, vectors, tuples, or 1×n/n×1 matrices."))
        end
    end
    
    stages = PIVStage[]
    
    for i in 1:n_stages
        # Geometric progression: start with larger windows, end with final_size
        size_factor = 2.0^(n_stages - i)
        current_size = round(Int, final_size * size_factor)
        
        # Ensure minimum window size
        current_size = max(current_size, final_size)
        
        # Get stage-specific parameters
        stage_overlap = get_stage_value(overlap, i, n_stages)
        stage_padding = get_stage_value(padding, i, n_stages)
        stage_deformation_iterations = get_stage_value(deformation_iterations, i, n_stages)
        stage_window_function = get_stage_value(window_function, i, n_stages)
        stage_interpolation_method = get_stage_value(interpolation_method, i, n_stages)
        
        # Convert scalar overlap to tuple if needed
        if isa(stage_overlap, Real)
            stage_overlap_tuple = (Float64(stage_overlap), Float64(stage_overlap))
        elseif isa(stage_overlap, Tuple) && length(stage_overlap) == 2
            stage_overlap_tuple = (Float64(stage_overlap[1]), Float64(stage_overlap[2]))
        else
            stage_overlap_tuple = stage_overlap
        end
        
        stage = PIVStage(current_size, 
                        overlap=stage_overlap_tuple,
                        padding=Int(stage_padding),
                        deformation_iterations=Int(stage_deformation_iterations),
                        window_function=stage_window_function,
                        interpolation_method=stage_interpolation_method)
        push!(stages, stage)
    end
    
    return stages
end

struct CrossCorrelator{T, FP, IP} <: Correlator
    C1::Matrix{Complex{T}}
    C2::Matrix{Complex{T}}
    fp::FP # FFTW plan type for forward FFT (long & messy, so use a parametric type)
    ip::IP # FFTW plan type for inverse FFT (likewise)
    function CrossCorrelator{T}(image_size::Tuple{Int, Int}) where T
        C1 = zeros(Complex{T}, image_size)
        C2 = zeros(Complex{T}, image_size)
        # Create FFTW plans for forward and inverse FFT
        fp = FFTW.plan_fft!(C1)
        ip = inv(fp)
        FP = typeof(fp)
        IP = typeof(ip)
        new{T, FP, IP}(C1, C2, fp, ip)
    end
end

Base.show(io::IO, c::CrossCorrelator) = print(io, "CrossCorrelator{$(eltype(c.C1))}[$(size(c.C1, 1)) x $(size(c.C1, 2))]")

CrossCorrelator(image_size::Tuple{Int, Int}) = CrossCorrelator{Float32}(image_size)

"""
    calculate_quality_metrics(correlation_plane, peak_location, peak_value) -> (peak_ratio, correlation_moment)

Calculate quality metrics for PIV correlation analysis.

# Arguments
- `correlation_plane` - Complex correlation result matrix
- `peak_location` - CartesianIndex of primary peak location
- `peak_value` - Magnitude of primary peak

# Returns
- `peak_ratio::Float64` - Ratio of primary peak to secondary peak (higher is better)
- `correlation_moment::Float64` - Normalized correlation moment (measure of peak sharpness)
"""
function calculate_quality_metrics(correlation_plane::AbstractMatrix, peak_location::CartesianIndex, peak_value::Real)
    # Convert correlation plane to real magnitudes for analysis
    corr_mag = abs.(correlation_plane)
    
    # Find secondary peak (exclude region around primary peak)
    secondary_peak = find_secondary_peak(corr_mag, peak_location, peak_value)
    
    # Calculate peak ratio (primary/secondary)
    peak_ratio = secondary_peak > 0 ? peak_value / secondary_peak : Inf
    
    # Calculate correlation moment (normalized second moment of peak)
    correlation_moment = calculate_correlation_moment(corr_mag, peak_location)
    
    return Float64(peak_ratio), Float64(correlation_moment)
end

"""
    find_secondary_peak(correlation_magnitudes, primary_location, primary_value) -> secondary_value

Find the secondary peak in correlation plane, excluding region around primary peak.
"""
function find_secondary_peak(corr_mag::AbstractMatrix, primary_loc::CartesianIndex, primary_val::Real)
    # Exclude circular region around primary peak (radius = 3 pixels)
    exclusion_radius = 3
    secondary_val = zero(eltype(corr_mag))
    
    for idx in CartesianIndices(corr_mag)
        # Skip if within exclusion radius of primary peak
        dx = idx.I[1] - primary_loc.I[1]
        dy = idx.I[2] - primary_loc.I[2]
        if sqrt(dx^2 + dy^2) <= exclusion_radius
            continue
        end
        
        val = corr_mag[idx]
        if val > secondary_val
            secondary_val = val
        end
    end
    
    return secondary_val
end

"""
    calculate_correlation_moment(correlation_magnitudes, peak_location) -> moment

Calculate normalized second moment of correlation peak as sharpness measure.
"""
function calculate_correlation_moment(corr_mag::AbstractMatrix, peak_loc::CartesianIndex)
    # Calculate second moment in 5x5 region around peak
    window_radius = 2
    i_center, j_center = peak_loc.I[1], peak_loc.I[2]
    rows, cols = size(corr_mag)
    
    # Bounds checking
    i_start = max(1, i_center - window_radius)
    i_end = min(rows, i_center + window_radius)
    j_start = max(1, j_center - window_radius)
    j_end = min(cols, j_center + window_radius)
    
    # Calculate weighted second moment
    total_weight = 0.0
    moment_sum = 0.0
    
    for i in i_start:i_end, j in j_start:j_end
        weight = corr_mag[i, j]
        dx = i - i_center
        dy = j - j_center
        distance_sq = dx^2 + dy^2
        
        total_weight += weight
        moment_sum += weight * distance_sq
    end
    
    # Normalized moment (lower values indicate sharper peaks)
    return total_weight > 0 ? moment_sum / total_weight : NaN
end

function correlate!(c::CrossCorrelator, subimgA::AbstractArray, subimgB::AbstractArray)
    c.C1 .= subimgA
    c.C2 .= subimgB
    # Perform inplace FFT on both sub-images using pre-computed plan `fp`
    mul!(c.C1, c.fp, c.C1)
    mul!(c.C2, c.fp, c.C2)

    # Compute the cross-correlation matrix (conj(FFT(A)) * FFT(B))
    for i in eachindex(c.C1)
        c.C1[i] = conj(c.C1[i]) * c.C2[i]
    end
    # Inverse FFT and shift zero-lag to center
    ldiv!(c.C1, c.ip, c.C1)
    fftshift!(c.C2, c.C1)

    # Return view to correlation plane (c.C2) - no allocation!
    return c.C2
end

"""
    analyze_correlation_plane(correlation_plane) -> (displacement, peak_ratio, correlation_moment)

Analyze correlation plane to extract displacement and quality metrics at PIV stage level.
This allows the same analysis to be applied to results from different correlation algorithms.
"""
function analyze_correlation_plane(correlation_plane::AbstractMatrix)
    # Find the peak in the correlation result
    peakloc = CartesianIndex(0, 0)
    maxval = zero(real(eltype(correlation_plane)))
    for i in CartesianIndices(correlation_plane)
        absval = abs(correlation_plane[i])
        if absval > maxval
            maxval = absval
            peakloc = i
        end
    end

    # Calculate quality metrics
    peak_ratio, correlation_moment = calculate_quality_metrics(correlation_plane, peakloc, maxval)

    # Perform subpixel refinement and compute displacement relative to center
    center = size(correlation_plane) .÷ 2 .+ 1
    refined_peakloc = subpixel_gauss3(correlation_plane, peakloc.I)
    disp = center .- refined_peakloc

    # Return displacement and quality metrics
    return (disp[1], disp[2], peak_ratio, correlation_moment)
end

function subpixel_gauss3(correlation_matrix::Matrix{T}, peak_coords::Tuple{Int, Int}) where T
    nrows, ncols = size(correlation_matrix)
    i, j = peak_coords
    I0 = abs(correlation_matrix[i, j])
    zT = zero(real(T)) # avoid type instability
    # compute dx
    if 1 < i < nrows
        Im = abs(correlation_matrix[i-1, j])
        Ip = abs(correlation_matrix[i+1, j])
        denom_x = log(Im) - 2*log(I0) + log(Ip)
        dx = denom_x != 0 ? (log(Im) - log(Ip)) / (2 * denom_x) : zT
    else
        dx = zT
    end
    # compute dy
    if 1 < j < ncols
        Im = abs(correlation_matrix[i, j-1])
        Ip = abs(correlation_matrix[i, j+1])
        denom_y = log(Im) - 2*log(I0) + log(Ip)
        dy = denom_y != 0 ? (log(Im) - log(Ip)) / (2 * denom_y) : zT
    else
        dy = zT
    end
    return (i + dx, j + dy)
end

"""
    run_piv(img1, img2; correlator=CrossCorrelator, kwargs...) -> PIVResult
    run_piv(img1, img2, stages::Vector{PIVStage}; correlator=CrossCorrelator, kwargs...) -> Vector{PIVResult}

Perform PIV analysis on image pair with single-stage or multi-stage processing.

# Arguments
- `img1::AbstractArray{<:Real,2}` - First image  
- `img2::AbstractArray{<:Real,2}` - Second image
- `stages::Vector{PIVStage}` - Optional multi-stage configuration

# Keyword Arguments  
- `correlator` - Correlation method (default: CrossCorrelator)
- `window_size::Tuple{Int,Int}` - Window size for single-stage (default: (64,64))
- `overlap::Tuple{Float64,Float64}` - Overlap ratios (default: (0.5,0.5))

# Returns
- `PIVResult` for single-stage processing
- `Vector{PIVResult}` for multi-stage processing
"""
function run_piv(img1::AbstractArray{<:Real,2}, img2::AbstractArray{<:Real,2}; 
                 correlator=CrossCorrelator, window_size::Tuple{Int,Int}=(64,64),
                 overlap::Tuple{Float64,Float64}=(0.5,0.5), kwargs...)
    
    # Input validation
    if size(img1) != size(img2)
        throw(ArgumentError("Image sizes must match: $(size(img1)) vs $(size(img2))"))
    end
    
    # Create default single-stage configuration
    stage = PIVStage(window_size, overlap=overlap)
    
    # Perform single-stage PIV analysis
    return run_piv_stage(img1, img2, stage, correlator)
end

# Multi-stage version
function run_piv(img1::AbstractArray{<:Real,2}, img2::AbstractArray{<:Real,2}, 
                 stages::Vector{<:PIVStage}; correlator=CrossCorrelator, kwargs...)
    
    # Input validation
    if size(img1) != size(img2)
        throw(ArgumentError("Image sizes must match: $(size(img1)) vs $(size(img2))"))
    end
    
    if isempty(stages)
        throw(ArgumentError("At least one PIV stage must be provided"))
    end
    
    # Perform multi-stage PIV analysis
    results = PIVResult[]
    
    for (i, stage) in enumerate(stages)
        # For now, just perform independent analysis at each stage
        # TODO: Add initial guess propagation from previous stage
        result = run_piv_stage(img1, img2, stage, correlator)
        
        # Add stage information to metadata
        result.metadata["stage"] = i
        result.metadata["total_stages"] = length(stages)
        result.metadata["window_size"] = stage.window_size
        result.metadata["overlap"] = stage.overlap
        
        push!(results, result)
    end
    
    return results
end

"""
    run_piv_stage(img1, img2, stage, correlator) -> PIVResult

Perform PIV analysis for a single stage with given configuration.
Quality metrics and subpixel refinement are handled at this level for modularity.
"""
function run_piv_stage(img1::AbstractArray{<:Real,2}, img2::AbstractArray{<:Real,2}, 
                       stage::PIVStage, correlator_type)
    
    # Generate interrogation window grid
    grid_x, grid_y = generate_interrogation_grid(size(img1), stage.window_size, stage.overlap)
    n_windows = length(grid_x)
    
    # Initialize correlator
    correlator = correlator_type(stage.window_size)
    
    # Initialize result arrays
    positions_x = Float64[]
    positions_y = Float64[]
    displacements_u = Float64[]
    displacements_v = Float64[]
    status_flags = Symbol[]
    peak_ratios = Float64[]
    correlation_moments = Float64[]
    
    # Process each interrogation window
    for i in 1:n_windows
        # Extract window position
        window_x = grid_x[i]
        window_y = grid_y[i]
        
        # Calculate window bounds with boundary checking
        x_start = max(1, round(Int, window_x - stage.window_size[1]//2))
        x_end = min(size(img1, 1), x_start + stage.window_size[1] - 1)
        y_start = max(1, round(Int, window_y - stage.window_size[2]//2))
        y_end = min(size(img1, 2), y_start + stage.window_size[2] - 1)
        
        # Extract subimages
        try
            subimg1 = img1[x_start:x_end, y_start:y_end]
            subimg2 = img2[x_start:x_end, y_start:y_end]
            
            # Pad if necessary to match correlator size
            if size(subimg1) != stage.window_size
                # For now, skip windows that don't fit exactly
                # TODO: Implement proper padding
                push!(positions_x, window_x)
                push!(positions_y, window_y) 
                push!(displacements_u, NaN)
                push!(displacements_v, NaN)
                push!(status_flags, :bad)
                push!(peak_ratios, NaN)
                push!(correlation_moments, NaN)
                continue
            end
            
            # Perform correlation to get correlation plane
            correlation_plane = correlate!(correlator, subimg1, subimg2)
            
            # Analyze correlation plane for displacement and quality metrics
            disp_u, disp_v, peak_ratio, corr_moment = analyze_correlation_plane(correlation_plane)
            
            # Store results
            push!(positions_x, window_x)
            push!(positions_y, window_y)
            push!(displacements_u, disp_u)
            push!(displacements_v, disp_v)
            push!(status_flags, :good)
            push!(peak_ratios, peak_ratio)
            push!(correlation_moments, corr_moment)
            
        catch e
            # Handle correlation failures gracefully
            push!(positions_x, window_x)
            push!(positions_y, window_y)
            push!(displacements_u, NaN)
            push!(displacements_v, NaN)
            push!(status_flags, :bad)
            push!(peak_ratios, NaN)
            push!(correlation_moments, NaN)
        end
    end
    
    # Create PIVVector array
    piv_vectors = [PIVVector(positions_x[i], positions_y[i], displacements_u[i], displacements_v[i],
                            status_flags[i], peak_ratios[i], correlation_moments[i]) 
                   for i in 1:n_windows]
    
    # Create result with metadata
    result = PIVResult(piv_vectors)
    result.metadata["image_size"] = size(img1)
    result.metadata["window_size"] = stage.window_size
    result.metadata["overlap"] = stage.overlap
    result.metadata["n_windows"] = n_windows
    result.metadata["processing_time"] = time()  # TODO: Measure actual processing time
    
    return result
end

"""
    generate_interrogation_grid(image_size, window_size, overlap) -> (grid_x, grid_y)

Generate grid of interrogation window center positions.
"""
function generate_interrogation_grid(image_size::Tuple{Int,Int}, window_size::Tuple{Int,Int}, 
                                   overlap::Tuple{Float64,Float64})
    
    # Calculate step sizes based on overlap
    step_x = round(Int, window_size[1] * (1 - overlap[1]))
    step_y = round(Int, window_size[2] * (1 - overlap[2]))
    
    # Ensure minimum step size of 1
    step_x = max(1, step_x)
    step_y = max(1, step_y)
    
    # Calculate grid bounds
    half_window_x = window_size[1] ÷ 2
    half_window_y = window_size[2] ÷ 2
    
    start_x = half_window_x + 1
    end_x = image_size[1] - half_window_x
    start_y = half_window_y + 1  
    end_y = image_size[2] - half_window_y
    
    # Generate grid coordinates
    grid_x = Float64[]
    grid_y = Float64[]
    
    for x in start_x:step_x:end_x
        for y in start_y:step_y:end_y
            push!(grid_x, Float64(x))
            push!(grid_y, Float64(y))
        end
    end
    
    return grid_x, grid_y
end


end # module Hammerhead
