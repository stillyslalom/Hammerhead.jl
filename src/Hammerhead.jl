module Hammerhead

using FFTW
using LinearAlgebra: mul!, ldiv!, inv
using ImageFiltering
using StructArrays
using Interpolations
using ImageIO
using DSP

# Data structures
export PIVVector, PIVResult, PIVStage, PIVStages

# Core functionality  
export run_piv, run_piv_stage, CrossCorrelator, correlate!, analyze_correlation_plane

# Quality assessment
export find_secondary_peak, find_secondary_peak_robust, find_local_maxima

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
# Use DSP.jl implementations for mathematically correct and well-tested functions
abstract type WindowFunction end

# Simple window functions (no parameters)
struct SimpleWindow{F} <: WindowFunction
    func::F
end

# Parametric window functions (with parameters)
struct ParametricWindow{F,P} <: WindowFunction
    func::F
    params::P
end

# Interpolation method types for type-stable dispatch (internal)
abstract type InterpolationMethod end
struct _Nearest <: InterpolationMethod end
struct _Bilinear <: InterpolationMethod end
struct _Bicubic <: InterpolationMethod end
struct _Spline <: InterpolationMethod end
struct _Lanczos <: InterpolationMethod end

# Symbol to type mapping for window functions
function window_function_type(s::Symbol)
    # Map symbols to DSP.jl functions for simple windows
    simple_windows = Dict(
        :rectangular => DSP.rect,
        :rect => DSP.rect,
        :hanning => DSP.hanning,
        :hamming => DSP.hamming, 
        :blackman => DSP.blackman,
        :bartlett => DSP.bartlett,
        :cosine => DSP.cosine,
        :lanczos => DSP.lanczos,
        :triang => DSP.triang
    )
    
    if haskey(simple_windows, s)
        return SimpleWindow(simple_windows[s])
    else
        throw(ArgumentError("Unknown window function: $s. Supported: $(keys(simple_windows))"))
    end
end

# Tuple mapping for parametric window functions
function window_function_type(spec::Tuple{Symbol, Vararg{Real}})
    window_type, params... = spec
    
    # Map symbols to DSP.jl parametric functions
    parametric_windows = Dict(
        :kaiser => DSP.kaiser,
        :tukey => DSP.tukey,
        :gaussian => DSP.gaussian
    )
    
    if haskey(parametric_windows, window_type)
        return ParametricWindow(parametric_windows[window_type], params)
    else
        throw(ArgumentError("Unknown parametric window function: $window_type. Supported: $(keys(parametric_windows))"))
    end
end

"""
    apply_window_function(subimage, window_function) -> windowed_subimage

Apply windowing function to subimage to reduce spectral leakage in correlation analysis.
Uses DSP.jl implementations for mathematically correct and well-tested window functions.
"""
function apply_window_function(subimg::AbstractArray{T,2}, window_function::WindowFunction) where T
    rows, cols = size(subimg)
    
    # Generate separable window using DSP.jl
    window_1d_row = generate_window_1d(window_function, rows)
    window_1d_col = generate_window_1d(window_function, cols)
    
    # Apply separable 2D windowing
    windowed = similar(subimg)
    for i in 1:rows, j in 1:cols
        windowed[i, j] = subimg[i, j] * window_1d_row[i] * window_1d_col[j]
    end
    
    return windowed
end

"""
    generate_window_1d(window_function, length) -> window_vector

Generate 1D window function using DSP.jl implementations for accuracy and performance.
Supports both simple and parametric window functions.
"""
function generate_window_1d(w::SimpleWindow, n::Int)
    return w.func(n)
end

function generate_window_1d(w::ParametricWindow, n::Int)
    return w.func(n, w.params...)
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

# Constructor with defaults using symbols or tuples for parametric windows
function PIVStage(window_size::Tuple{Int, Int}; 
                 overlap::Tuple{Float64, Float64} = (0.5, 0.5),
                 padding::Int = 0,
                 deformation_iterations::Int = 3,
                 window_function::Union{Symbol, Tuple{Symbol, Vararg{Real}}} = :rectangular,
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
- `window_function=:rectangular` - Window function(s). Can be symbol, tuple (symbol, params...), or vector thereof
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

# Parametric window functions
stages = PIVStages(3, 32, window_function=[(:kaiser, 5.0), :hanning, (:tukey, 0.3)])

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
    calculate_quality_metrics(correlation_plane, peak_location, peak_value; robust=false) -> (peak_ratio, correlation_moment)

Calculate quality metrics for PIV correlation analysis.

# Arguments
- `correlation_plane` - Complex correlation result matrix
- `peak_location` - CartesianIndex of primary peak location
- `peak_value` - Magnitude of primary peak
- `robust=false` - Use robust local maxima method for secondary peak detection

# Returns
- `peak_ratio::Float64` - Ratio of primary peak to secondary peak (higher is better)
- `correlation_moment::Float64` - Normalized correlation moment (measure of peak sharpness)
"""
function calculate_quality_metrics(correlation_plane::AbstractMatrix, peak_location::CartesianIndex, peak_value::Real; robust::Bool=false)
    # Convert correlation plane to real magnitudes for analysis
    corr_mag = abs.(correlation_plane)
    
    # Find secondary peak using chosen method
    secondary_peak = if robust
        find_secondary_peak_robust(corr_mag, peak_location, peak_value)
    else
        find_secondary_peak(corr_mag, peak_location, peak_value)
    end
    
    # Calculate peak ratio (primary/secondary)
    peak_ratio = secondary_peak > 0 ? peak_value / secondary_peak : Inf
    
    # Calculate correlation moment (normalized second moment of peak)
    correlation_moment = calculate_correlation_moment(corr_mag, peak_location)
    
    return Float64(peak_ratio), Float64(correlation_moment)
end

"""
    find_secondary_peak(correlation_magnitudes, primary_location, primary_value) -> secondary_value

Find the secondary peak in correlation plane, excluding region around primary peak.
Uses exclusion radius approach for speed.
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
    find_secondary_peak_robust(correlation_magnitudes, primary_location, primary_value) -> secondary_value

Find the secondary peak using local maxima detection for robustness in high shear flows.
More computationally expensive but handles closely spaced peaks better.
"""
function find_secondary_peak_robust(corr_mag::AbstractMatrix, primary_loc::CartesianIndex, primary_val::Real)
    # Find all local maxima
    local_maxima = find_local_maxima(corr_mag)
    
    # If no local maxima found, return zero
    if isempty(local_maxima)
        return zero(eltype(corr_mag))
    end
    
    # Sort maxima by value in descending order
    sort!(local_maxima, by=x -> corr_mag[x], rev=true)
    
    # Find the secondary peak (largest local maximum that's not the primary)
    for loc in local_maxima
        if loc != primary_loc
            return corr_mag[loc]
        end
    end
    
    # If all local maxima are at primary location (shouldn't happen), return zero
    return zero(eltype(corr_mag))
end

"""
    find_local_maxima(correlation_magnitudes) -> Vector{CartesianIndex}

Find all local maxima in correlation plane using robust peak detection.
Identifies peaks by finding points that are local maxima in their immediate neighborhood,
handling both sharp and broad peaks correctly.
"""
function find_local_maxima(corr_mag::AbstractMatrix)
    rows, cols = size(corr_mag)
    
    # Find all potential maxima by checking if each point is greater than immediate neighbors
    candidates = CartesianIndex[]
    
    # Check all interior points
    for i in 2:(rows-1), j in 2:(cols-1)
        center_val = corr_mag[i, j]
        is_local_max = true
        
        # Check immediate neighbors (3x3 neighborhood)
        for di in -1:1, dj in -1:1
            if di == 0 && dj == 0
                continue  # Skip center point
            end
            
            if corr_mag[i + di, j + dj] > center_val
                is_local_max = false
                break
            end
        end
        
        if is_local_max && center_val > 0  # Only consider positive values
            push!(candidates, CartesianIndex(i, j))
        end
    end
    
    # For broad peaks, we might have multiple adjacent points that are all "local maxima"
    # Sort candidates by value and keep the significant ones
    if isempty(candidates)
        return candidates
    end
    
    # Sort by correlation value (descending)
    sort!(candidates, by=idx -> corr_mag[idx], rev=true)
    
    # Filter out candidates that are too close to higher-valued ones (merge broad peaks)
    merged_maxima = CartesianIndex[]
    min_separation = 2  # Minimum pixels between distinct peaks
    
    for candidate in candidates
        is_distinct = true
        
        # Check if this candidate is too close to an already accepted maximum
        for accepted in merged_maxima
            distance = sqrt((candidate.I[1] - accepted.I[1])^2 + (candidate.I[2] - accepted.I[2])^2)
            if distance < min_separation
                is_distinct = false
                break
            end
        end
        
        if is_distinct
            push!(merged_maxima, candidate)
        end
    end
    
    return merged_maxima
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
    pad_to_size(subimg, target_size) -> padded_array

Pad a subimage to match target window size using symmetric padding.
Uses the 'symmetric' boundary condition which reflects values across the boundary.
"""
function pad_to_size(subimg::AbstractArray{T,2}, target_size::Tuple{Int,Int}) where T
    current_size = size(subimg)
    
    # If already correct size, return as-is
    if current_size == target_size
        return subimg
    end
    
    # Calculate padding needed
    pad_h = target_size[1] - current_size[1]
    pad_w = target_size[2] - current_size[2]
    
    # Ensure we're only padding (not cropping)
    if pad_h < 0 || pad_w < 0
        throw(ArgumentError("Cannot pad to smaller size: $current_size -> $target_size"))
    end
    
    # Calculate padding on each side (distribute evenly, with extra on bottom/right if odd)
    pad_top = pad_h ÷ 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w ÷ 2
    pad_right = pad_w - pad_left
    
    # Create padded array using symmetric boundary condition
    # This reflects values across the boundary, avoiding discontinuities
    padded = zeros(T, target_size)
    
    # Copy original data to center
    padded[pad_top+1:pad_top+current_size[1], pad_left+1:pad_left+current_size[2]] = subimg
    
    # Fill padding regions using symmetric reflection
    fill_symmetric_padding!(padded, subimg, pad_top, pad_left, current_size)
    
    return padded
end

"""
    fill_symmetric_padding!(padded, original, pad_top, pad_left, original_size)

Fill padding regions using symmetric reflection to avoid discontinuities.
"""
function fill_symmetric_padding!(padded::AbstractArray{T,2}, original::AbstractArray{T,2}, 
                                 pad_top::Int, pad_left::Int, original_size::Tuple{Int,Int}) where T
    orig_h, orig_w = original_size
    
    # Copy original to center region
    center_h_range = (pad_top+1):(pad_top+orig_h)
    center_w_range = (pad_left+1):(pad_left+orig_w)
    
    # Top padding (reflect vertically)
    if pad_top > 0
        for i in 1:pad_top
            src_row = min(orig_h, pad_top - i + 1)  # Reflect from first row
            padded[i, center_w_range] = original[src_row, :]
        end
    end
    
    # Bottom padding (reflect vertically)
    bottom_start = pad_top + orig_h + 1
    if bottom_start <= size(padded, 1)
        for i in bottom_start:size(padded, 1)
            offset = i - bottom_start
            src_row = max(1, orig_h - offset)  # Reflect from last row
            padded[i, center_w_range] = original[src_row, :]
        end
    end
    
    # Left padding (reflect horizontally, including top/bottom padding)
    if pad_left > 0
        for j in 1:pad_left
            src_col = min(orig_w, pad_left - j + 1)  # Reflect from first column
            padded[:, j] = padded[:, pad_left + src_col]
        end
    end
    
    # Right padding (reflect horizontally, including top/bottom padding) 
    right_start = pad_left + orig_w + 1
    if right_start <= size(padded, 2)
        for j in right_start:size(padded, 2)
            offset = j - right_start
            src_col = max(1, pad_left + orig_w - offset)  # Reflect from last column
            padded[:, j] = padded[:, src_col]
        end
    end
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
                subimg1 = pad_to_size(subimg1, stage.window_size)
                subimg2 = pad_to_size(subimg2, stage.window_size)
            end
            
            # Apply windowing function to reduce spectral leakage
            windowed_subimg1 = apply_window_function(subimg1, stage.window_function)
            windowed_subimg2 = apply_window_function(subimg2, stage.window_function)
            
            # Perform correlation to get correlation plane
            correlation_plane = correlate!(correlator, windowed_subimg1, windowed_subimg2)
            
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
