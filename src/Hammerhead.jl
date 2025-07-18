module Hammerhead

using FFTW
using LinearAlgebra: mul!, ldiv!, inv, det, cond, eigvals, dot, I
using ImageFiltering
using StructArrays
using Interpolations
using ImageIO
using DSP
using TimerOutputs
using Statistics: mean

# Data structures
export PIVVector, PIVResult, PIVStage, PIVStages

# Core functionality  
export run_piv, run_piv_stage, CrossCorrelator, correlate!, analyze_correlation_plane

# Quality assessment
export find_secondary_peak, find_secondary_peak_robust, find_local_maxima

# Transform validation
export validate_affine_transform, is_area_preserving

# Interpolation
export linear_barycentric_interpolation, interpolate_vectors

# Iterative deformation
export apply_image_deformation, compute_local_affine_transform, run_iterative_deformation, compute_convergence_metric, bilinear_interpolate

# Utilities
export subpixel_gauss3

# Timing
export get_timer

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

"""
    get_timer(result::PIVResult) -> TimerOutput

Get the timer data from a PIV result for analysis or integration with other timing systems.
Returns a new TimerOutput if no timing data is available.
"""
function get_timer(result::PIVResult)
    timer_data = get(result.metadata, "timer", nothing)
    return timer_data !== nothing ? timer_data : TimerOutput()
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
Uses super-fast approach: find global maximum, then exclude if it's near primary.
"""
function find_secondary_peak(corr_mag::AbstractMatrix, primary_loc::CartesianIndex, primary_val::Real)
    # Strategy: Find global max first, check if it's far enough from primary
    # This is O(n) instead of O(n) with exclusion checks
    
    exclusion_radius = 3
    rows, cols = size(corr_mag)
    pi, pj = primary_loc.I
    
    # Find global maximum and its location
    max_val = zero(eltype(corr_mag))
    max_loc = CartesianIndex(1, 1)
    
    for i in 1:rows, j in 1:cols
        val = corr_mag[i, j]
        if val > max_val
            max_val = val
            max_loc = CartesianIndex(i, j)
        end
    end
    
    # Check if global max is far enough from primary
    mi, mj = max_loc.I
    if abs(mi - pi) > exclusion_radius || abs(mj - pj) > exclusion_radius
        return max_val
    end
    
    # If global max is too close to primary, find next best outside exclusion zone
    secondary_val = zero(eltype(corr_mag))
    min_i = max(1, pi - exclusion_radius)
    max_i = min(rows, pi + exclusion_radius) 
    min_j = max(1, pj - exclusion_radius)
    max_j = min(cols, pj + exclusion_radius)
    
    for i in 1:rows, j in 1:cols
        # Skip exclusion rectangle
        if min_i <= i <= max_i && min_j <= j <= max_j
            continue
        end
        
        val = corr_mag[i, j]
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

# Backward compatibility method for correlate! without timer
function correlate!(c::CrossCorrelator, subimgA::AbstractArray, subimgB::AbstractArray)
    # Create a temporary timer for non-timed calls
    temp_timer = TimerOutput()
    return correlate!(c, subimgA, subimgB, temp_timer)
end

function correlate!(c::CrossCorrelator, subimgA::AbstractArray, subimgB::AbstractArray, timer::TimerOutput)
    @timeit timer "FFT Setup" begin
        c.C1 .= subimgA
        c.C2 .= subimgB
    end
    
    @timeit timer "Forward FFT" begin
        # Perform inplace FFT on both sub-images using pre-computed plan `fp`
        mul!(c.C1, c.fp, c.C1)
        mul!(c.C2, c.fp, c.C2)
    end

    @timeit timer "Cross-correlation" begin
        # Compute the cross-correlation matrix (conj(FFT(A)) * FFT(B))
        for i in eachindex(c.C1)
            c.C1[i] = conj(c.C1[i]) * c.C2[i]
        end
    end
    
    @timeit timer "Inverse FFT" begin
        # Inverse FFT and shift zero-lag to center
        ldiv!(c.C1, c.ip, c.C1)
        fftshift!(c.C2, c.C1)
    end

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
    
    # Create local timer for this PIV run
    timer = TimerOutput()
    
    @timeit timer "Single-Stage PIV" begin
        # Input validation
        if size(img1) != size(img2)
            throw(ArgumentError("Image sizes must match: $(size(img1)) vs $(size(img2))"))
        end
        
        # Create default single-stage configuration
        stage = PIVStage(window_size, overlap=overlap)
        
        # Perform single-stage PIV analysis
        result = run_piv_stage(img1, img2, stage, correlator, timer)
        
        # Store timer in result metadata
        result.metadata["timer"] = timer
        
        return result
    end
end

# Multi-stage version
function run_piv(img1::AbstractArray{<:Real,2}, img2::AbstractArray{<:Real,2}, 
                 stages::Vector{<:PIVStage}; correlator=CrossCorrelator, kwargs...)
    
    # Create local timer for this PIV run
    timer = TimerOutput()
    
    @timeit timer "Multi-Stage PIV" begin
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
            @timeit timer "Stage $i" begin
                # For now, just perform independent analysis at each stage
                # TODO: Add initial guess propagation from previous stage
                result = run_piv_stage(img1, img2, stage, correlator, timer)
                
                # Add stage information to metadata
                result.metadata["stage"] = i
                result.metadata["total_stages"] = length(stages)
                result.metadata["window_size"] = stage.window_size
                result.metadata["overlap"] = stage.overlap
                
                push!(results, result)
            end
        end
        
        # Store timer in the first result's metadata for overall timing
        if !isempty(results)
            results[1].metadata["timer"] = timer
        end
        
        return results
    end
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
                       stage::PIVStage, correlator_type, timer::TimerOutput)
    
    @timeit timer "PIV Stage" begin
        # Generate interrogation window grid
        grid_x, grid_y = @timeit timer "Grid Generation" generate_interrogation_grid(size(img1), stage.window_size, stage.overlap)
        n_windows = length(grid_x)
        
        # Initialize correlator
        correlator = @timeit timer "Correlator Setup" correlator_type(stage.window_size)
    
        # Initialize result arrays
        positions_x = Float64[]
        positions_y = Float64[]
        displacements_u = Float64[]
        displacements_v = Float64[]
        status_flags = Symbol[]
        peak_ratios = Float64[]
        correlation_moments = Float64[]
        
        # Process each interrogation window
        @timeit timer "Window Processing" for i in 1:n_windows
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
            windowed_subimg1 = @timeit timer "Windowing" apply_window_function(subimg1, stage.window_function)
            windowed_subimg2 = @timeit timer "Windowing" apply_window_function(subimg2, stage.window_function)
            
            # Perform correlation to get correlation plane
            correlation_plane = @timeit timer "Correlation" correlate!(correlator, windowed_subimg1, windowed_subimg2, timer)
            
            # Analyze correlation plane for displacement and quality metrics
            disp_u, disp_v, peak_ratio, corr_moment = @timeit timer "Peak Analysis" analyze_correlation_plane(correlation_plane)
            
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
        piv_vectors = @timeit timer "Result Assembly" [PIVVector(positions_x[i], positions_y[i], displacements_u[i], displacements_v[i],
                                status_flags[i], peak_ratios[i], correlation_moments[i]) 
                       for i in 1:n_windows]
        
        # Create initial result with metadata
        result = PIVResult(piv_vectors)
        result.metadata["image_size"] = size(img1)
        result.metadata["window_size"] = stage.window_size
        result.metadata["overlap"] = stage.overlap
        result.metadata["n_windows"] = n_windows
        
        # Apply iterative deformation if requested
        if stage.deformation_iterations > 1
            result = @timeit timer "Iterative Deformation" run_iterative_deformation(
                img1, img2, stage, result, timer
            )
        end
        
        return result
    end  # @timeit timer "PIV Stage"
end

# Backward compatibility method for run_piv_stage without timer
function run_piv_stage(img1::AbstractArray{<:Real,2}, img2::AbstractArray{<:Real,2}, 
                       stage::PIVStage, correlator_type)
    # Create a temporary timer for non-timed calls
    temp_timer = TimerOutput()
    return run_piv_stage(img1, img2, stage, correlator_type, temp_timer)
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

"""
    validate_affine_transform(transform_matrix; tolerance=0.1) -> Bool

Validate an affine transformation matrix for use in iterative deformation.
Checks area preservation and other physical constraints.

# Arguments
- `transform_matrix::AbstractMatrix` - 2×2 or 3×3 affine transformation matrix
- `tolerance::Float64` - Tolerance for area preservation check (default: 0.1)

# Returns
- `Bool` - True if transform is valid, false otherwise

# Validation Criteria
- Area preservation: |det(A)| ≈ 1 within tolerance
- No excessive shear or rotation (condition number check)
- Finite and real values only

# Examples
```julia
# Valid identity transform
A = [1.0 0.0; 0.0 1.0]
validate_affine_transform(A)  # true

# Invalid transform (not area-preserving)
A = [2.0 0.0; 0.0 0.5]  # det = 1.0, but stretches by 2x and compresses by 0.5x
validate_affine_transform(A)  # false

# Valid small deformation
A = [1.05 0.02; -0.01 0.98]  # Small rotation and stretch
validate_affine_transform(A)  # likely true depending on tolerance
```
"""
function validate_affine_transform(transform_matrix::AbstractMatrix; tolerance::Float64=0.1)
    # Handle both 2×2 and 3×3 matrices (extract 2×2 linear part for 3×3)
    if size(transform_matrix) == (3, 3)
        A = transform_matrix[1:2, 1:2]
    elseif size(transform_matrix) == (2, 2)
        A = transform_matrix
    else
        throw(ArgumentError("Transform matrix must be 2×2 or 3×3, got $(size(transform_matrix))"))
    end
    
    # Check for finite and real values
    if !all(isfinite.(A)) || !all(isreal.(A))
        return false
    end
    
    # Check area preservation using determinant
    if !is_area_preserving(A, tolerance)
        return false
    end
    
    # Check condition number to avoid excessive distortion
    # High condition number indicates near-singular matrix or extreme aspect ratio changes
    max_condition = 10.0  # Allow up to 10:1 aspect ratio changes
    if cond(A) > max_condition
        return false
    end
    
    # Check for reasonable eigenvalues (no excessive stretching)
    eigenvals = eigvals(A)
    max_stretch = 3.0  # Allow up to 3x stretch in any direction
    min_compress = 1.0 / max_stretch  # And corresponding compression
    
    # Special case: check if this is a pure rotation (complex eigenvalues with |λ| = 1)
    if length(eigenvals) == 2 && !all(isreal.(eigenvals))
        # For 2D rotation matrices, eigenvalues are complex conjugates with |λ| = 1
        λ1, λ2 = eigenvals
        if abs(abs(λ1) - 1) < 1e-10 && abs(abs(λ2) - 1) < 1e-10 && 
           abs(λ1 - conj(λ2)) < 1e-10
            # This is a valid rotation matrix
            return true
        else
            return false  # Complex eigenvalues but not a proper rotation
        end
    end
    
    # For real eigenvalues, check stretching bounds
    for λ in eigenvals
        if !isreal(λ)
            return false  # Unexpected complex eigenvalues
        end
        # Check absolute value to handle reflections (negative eigenvalues)
        abs_λ = abs(real(λ))
        if abs_λ > max_stretch || abs_λ < min_compress
            return false
        end
    end
    
    return true
end

"""
    linear_barycentric_interpolation(points, values, query_points; fallback_method=:nearest)

Perform linear barycentric interpolation for scattered 2D data.

For each query point, finds the triangle containing it and computes the interpolated value
using barycentric coordinates. If a point is outside the convex hull, uses fallback method.

# Arguments
- `points::AbstractMatrix`: Nx2 matrix of point coordinates [x y]
- `values::AbstractVector`: N-element vector of values at each point
- `query_points::AbstractMatrix`: Mx2 matrix of query point coordinates
- `fallback_method::Symbol`: Method for points outside convex hull (`:nearest`, `:zero`, `:nan`)

# Returns
- `Vector`: Interpolated values at query points
"""
function linear_barycentric_interpolation(points::AbstractMatrix, values::AbstractVector, 
                                         query_points::AbstractMatrix; 
                                         fallback_method::Symbol=:nearest)
    if size(points, 2) != 2
        throw(ArgumentError("Points must be Nx2 matrix, got $(size(points))"))
    end
    if size(query_points, 2) != 2
        throw(ArgumentError("Query points must be Mx2 matrix, got $(size(query_points))"))
    end
    if size(points, 1) != length(values)
        throw(ArgumentError("Number of points ($(size(points, 1))) must match number of values ($(length(values)))"))
    end
    if !(fallback_method in (:nearest, :zero, :nan))
        throw(ArgumentError("Fallback method must be :nearest, :zero, or :nan, got $fallback_method"))
    end
    
    n_query = size(query_points, 1)
    n_points = size(points, 1)
    result = Vector{Float64}(undef, n_query)
    
    # Handle edge cases
    if n_points == 0
        fill!(result, fallback_method == :zero ? 0.0 : NaN)
        return result
    elseif n_points == 1
        # Single point - use its value for all queries
        fill!(result, values[1])
        return result
    elseif n_points == 2
        # Two points - linear interpolation along line
        return linear_interpolation_2points(points, values, query_points, fallback_method)
    end
    
    # For 3+ points, use triangulation-based barycentric interpolation
    # Simple approach: for each query point, find best triangle and interpolate
    for i in 1:n_query
        query_x, query_y = query_points[i, 1], query_points[i, 2]
        
        # Find the best triangle containing this point
        best_triangle, best_coords = find_containing_triangle(points, query_x, query_y)
        
        if best_triangle !== nothing
            # Interpolate using barycentric coordinates
            idx1, idx2, idx3 = best_triangle
            λ1, λ2, λ3 = best_coords
            result[i] = λ1 * values[idx1] + λ2 * values[idx2] + λ3 * values[idx3]
        else
            # Point outside convex hull - use fallback
            if fallback_method == :nearest
                # Find nearest point
                min_dist = Inf
                nearest_idx = 1
                for j in 1:n_points
                    dist = (points[j, 1] - query_x)^2 + (points[j, 2] - query_y)^2
                    if dist < min_dist
                        min_dist = dist
                        nearest_idx = j
                    end
                end
                result[i] = values[nearest_idx]
            elseif fallback_method == :zero
                result[i] = 0.0
            else  # :nan
                result[i] = NaN
            end
        end
    end
    
    return result
end

"""
    linear_interpolation_2points(points, values, query_points, fallback_method)

Linear interpolation between two points.
"""
function linear_interpolation_2points(points::AbstractMatrix, values::AbstractVector,
                                     query_points::AbstractMatrix, fallback_method::Symbol)
    n_query = size(query_points, 1)
    result = Vector{Float64}(undef, n_query)
    
    # Two points define a line
    p1, p2 = points[1, :], points[2, :]
    v1, v2 = values[1], values[2]
    
    # Line direction
    line_dir = p2 - p1
    line_length_sq = sum(line_dir.^2)
    
    for i in 1:n_query
        query = query_points[i, :]
        
        # Project query point onto line
        to_query = query - p1
        t = dot(to_query, line_dir) / line_length_sq
        
        if 0 <= t <= 1
            # Point projects onto line segment
            result[i] = (1 - t) * v1 + t * v2
        else
            # Outside line segment - use fallback
            if fallback_method == :nearest
                # Choose nearest endpoint
                dist1 = sum((query - p1).^2)
                dist2 = sum((query - p2).^2)
                result[i] = dist1 <= dist2 ? v1 : v2
            elseif fallback_method == :zero
                result[i] = 0.0
            else  # :nan
                result[i] = NaN
            end
        end
    end
    
    return result
end

"""
    find_containing_triangle(points, query_x, query_y)

Find triangle containing the query point and return barycentric coordinates.

Returns (triangle_indices, barycentric_coords) or (nothing, nothing) if outside convex hull.
"""
function find_containing_triangle(points::AbstractMatrix, query_x::Real, query_y::Real)
    n_points = size(points, 1)
    
    # Try all possible triangles
    for i in 1:n_points-2
        for j in i+1:n_points-1
            for k in j+1:n_points
                # Get triangle vertices
                p1 = points[i, :]
                p2 = points[j, :]
                p3 = points[k, :]
                
                # Compute barycentric coordinates
                coords = barycentric_coordinates(p1, p2, p3, [query_x, query_y])
                
                # Check if point is inside triangle (all coordinates >= 0)
                if all(coords .>= -1e-10)  # Small tolerance for numerical errors
                    return (i, j, k), coords
                end
            end
        end
    end
    
    return nothing, nothing
end

"""
    barycentric_coordinates(p1, p2, p3, query)

Compute barycentric coordinates of query point with respect to triangle p1-p2-p3.

Returns (λ1, λ2, λ3) such that query = λ1*p1 + λ2*p2 + λ3*p3 and λ1 + λ2 + λ3 = 1.
"""
function barycentric_coordinates(p1::AbstractVector, p2::AbstractVector, 
                               p3::AbstractVector, query::AbstractVector)
    # Using the standard formula for barycentric coordinates
    x1, y1 = p1[1], p1[2]
    x2, y2 = p2[1], p2[2]
    x3, y3 = p3[1], p3[2]
    x, y = query[1], query[2]
    
    # Area of the full triangle (twice the actual area)
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    
    # Handle degenerate triangles
    if abs(denom) < 1e-12
        return [NaN, NaN, NaN]
    end
    
    # Barycentric coordinates
    λ1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denom
    λ2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denom
    λ3 = 1.0 - λ1 - λ2
    
    return [λ1, λ2, λ3]
end

"""
    interpolate_vectors(result::PIVResult; method=:linear_barycentric, 
                       target_status=:interpolated, source_status=:good)

Interpolate vectors in PIV result to replace missing or bad vectors.

# Arguments
- `result::PIVResult`: PIV result to interpolate
- `method::Symbol`: Interpolation method (`:linear_barycentric`, `:nearest`)
- `target_status::Symbol`: Status to assign to interpolated vectors
- `source_status::Symbol`: Only use vectors with this status for interpolation

# Returns
- Modified PIVResult with interpolated vectors
"""
function interpolate_vectors(result::PIVResult; 
                           method::Symbol=:linear_barycentric,
                           target_status::Symbol=:interpolated,
                           source_status::Symbol=:good)
    if !(method in (:linear_barycentric, :nearest))
        throw(ArgumentError("Method must be :linear_barycentric or :nearest, got $method"))
    end
    
    # Find good vectors to use for interpolation
    good_mask = result.status .== source_status
    n_good = sum(good_mask)
    
    if n_good == 0
        @warn "No vectors with status $source_status found for interpolation"
        return result
    end
    
    # Extract coordinates and values of good vectors
    good_x = result.x[good_mask]
    good_y = result.y[good_mask]
    good_u = result.u[good_mask]
    good_v = result.v[good_mask]
    
    # Find vectors that need interpolation (not good and not NaN coordinates)
    bad_mask = (result.status .!= source_status) .& 
               .!isnan.(result.x) .& .!isnan.(result.y)
    
    if sum(bad_mask) == 0
        return result  # Nothing to interpolate
    end
    
    # Query points (bad vectors)
    query_x = result.x[bad_mask]
    query_y = result.y[bad_mask]
    
    # Prepare data for interpolation
    points = hcat(collect(good_x), collect(good_y))
    query_points = hcat(collect(query_x), collect(query_y))
    
    # Interpolate u and v components
    if method == :linear_barycentric
        interp_u = linear_barycentric_interpolation(points, collect(good_u), query_points)
        interp_v = linear_barycentric_interpolation(points, collect(good_v), query_points)
    else  # :nearest
        interp_u = nearest_neighbor_interpolation(points, collect(good_u), query_points)
        interp_v = nearest_neighbor_interpolation(points, collect(good_v), query_points)
    end
    
    # Create new vectors with interpolated values
    new_vectors = copy(result.vectors)
    bad_indices = findall(bad_mask)
    
    for (i, idx) in enumerate(bad_indices)
        if !isnan(interp_u[i]) && !isnan(interp_v[i])
            new_vectors[idx] = PIVVector(
                result.x[idx], result.y[idx],
                interp_u[i], interp_v[i],
                target_status, NaN, NaN
            )
        end
    end
    
    # Create new result
    new_result = PIVResult(new_vectors, 
                          copy(result.metadata), 
                          copy(result.auxiliary))
    new_result.metadata["interpolation_method"] = string(method)
    new_result.metadata["interpolated_count"] = sum(.!isnan.(interp_u))
    
    return new_result
end

"""
    nearest_neighbor_interpolation(points, values, query_points)

Simple nearest neighbor interpolation.
"""
function nearest_neighbor_interpolation(points::AbstractMatrix, values::AbstractVector,
                                      query_points::AbstractMatrix)
    n_query = size(query_points, 1)
    n_points = size(points, 1)
    result = Vector{Float64}(undef, n_query)
    
    for i in 1:n_query
        query_x, query_y = query_points[i, 1], query_points[i, 2]
        
        # Find nearest point
        min_dist = Inf
        nearest_val = NaN
        
        for j in 1:n_points
            dist = (points[j, 1] - query_x)^2 + (points[j, 2] - query_y)^2
            if dist < min_dist
                min_dist = dist
                nearest_val = values[j]
            end
        end
        
        result[i] = nearest_val
    end
    
    return result
end

"""
    is_area_preserving(matrix, tolerance=0.1) -> Bool

Check if a 2×2 matrix preserves area within tolerance.
Area preservation means |det(A)| ≈ 1.

# Arguments
- `matrix::AbstractMatrix` - 2×2 transformation matrix
- `tolerance::Float64` - Relative tolerance for determinant check

# Returns
- `Bool` - True if area-preserving within tolerance

# Examples
```julia
is_area_preserving([1.0 0.1; -0.1 1.0])  # true (rotation + small shear)
is_area_preserving([2.0 0.0; 0.0 0.5])   # false (scales area by factor of 1.0 but violates uniformity)
is_area_preserving([1.05 0.0; 0.0 0.95]) # true (small uniform scaling, det ≈ 1.0)
```
"""
function is_area_preserving(matrix::AbstractMatrix, tolerance::Float64=0.1)
    if size(matrix) != (2, 2)
        throw(ArgumentError("Matrix must be 2×2 for area preservation check"))
    end
    
    det_val = det(matrix)
    
    # Check if determinant magnitude is close to 1 (area-preserving)
    # Use relative tolerance: |det - 1| / 1 < tolerance
    return abs(abs(det_val) - 1.0) <= tolerance
end

"""
    run_iterative_deformation(img1, img2, stage, initial_result, timer)

Perform iterative deformation refinement of PIV analysis.

Uses initial displacement estimates to deform the second image iteratively,
improving accuracy by reducing bias from large displacements.

# Arguments
- `img1::AbstractArray{<:Real,2}` - First image 
- `img2::AbstractArray{<:Real,2}` - Second image
- `stage::PIVStage` - Processing stage configuration
- `initial_result::PIVResult` - Initial PIV analysis result
- `timer::TimerOutput` - Performance timing object

# Returns
- `PIVResult` - Refined result after iterative deformation

# Algorithm
1. Start with initial displacement estimates
2. For each iteration:
   - Compute local affine transforms from displacement field
   - Validate transforms for area preservation and stability
   - Apply deformation to second image
   - Re-run PIV analysis on deformed images
   - Update displacement estimates
   - Check for convergence
3. Return final refined results with convergence metadata
"""
function run_iterative_deformation(img1::AbstractArray{<:Real,2}, 
                                  img2::AbstractArray{<:Real,2},
                                  stage::PIVStage,
                                  initial_result::PIVResult,
                                  timer::TimerOutput)
    
    current_result = initial_result
    img2_deformed = copy(img2)
    convergence_history = Float64[]
    
    # Store initial displacements for total displacement tracking
    initial_u = copy(current_result.u)
    initial_v = copy(current_result.v)
    total_u = copy(initial_u)
    total_v = copy(initial_v)
    
    @timeit timer "Deformation Iterations" for iteration in 2:stage.deformation_iterations
        @timeit timer "Iteration $iteration" begin
            
            # Check for valid displacements
            valid_mask = .!isnan.(current_result.u) .& .!isnan.(current_result.v) .& 
                        (current_result.status .== :good)
            
            if sum(valid_mask) < 3
                @warn "Insufficient valid vectors for iteration $iteration, stopping deformation"
                break
            end
            
            # Apply deformation to second image
            img2_deformed = @timeit timer "Image Deformation" apply_image_deformation(
                img2_deformed, current_result, stage.window_size
            )
            
            # Re-run PIV analysis on deformed images
            # Create a temporary stage with single iteration to avoid infinite recursion
            temp_stage = PIVStage(
                stage.window_size, stage.overlap, stage.padding,
                1,  # Single iteration
                stage.window_function, stage.interpolation_method
            )
            
            new_result = @timeit timer "PIV Reanalysis" run_piv_stage(
                img1, img2_deformed, temp_stage, CrossCorrelator, timer
            )
            
            # Compute convergence metric
            convergence = @timeit timer "Convergence Check" compute_convergence_metric(
                current_result, new_result
            )
            push!(convergence_history, convergence)
            
            # Update total displacements
            @timeit timer "Displacement Update" begin
                for i in eachindex(total_u)
                    if !isnan(new_result.u[i]) && !isnan(new_result.v[i])
                        total_u[i] += new_result.u[i]
                        total_v[i] += new_result.v[i]
                    end
                end
                
                # Update current result with total displacements
                updated_vectors = similar(current_result.vectors)
                for i in eachindex(updated_vectors)
                    updated_vectors[i] = PIVVector(
                        current_result.x[i], current_result.y[i],
                        total_u[i], total_v[i],
                        new_result.status[i], new_result.peak_ratio[i], new_result.correlation_moment[i]
                    )
                end
                current_result = PIVResult(updated_vectors, copy(current_result.metadata), copy(current_result.auxiliary))
            end
            
            # Check convergence criteria
            if convergence < 0.1  # Convergence threshold in pixels
                @timeit timer "Early Convergence" begin
                    current_result.metadata["converged_at_iteration"] = iteration
                    current_result.metadata["convergence_reason"] = "displacement_threshold"
                    break
                end
            end
        end
    end
    
    # Add deformation metadata
    current_result.metadata["deformation_iterations_completed"] = min(stage.deformation_iterations, length(convergence_history) + 1)
    current_result.metadata["convergence_history"] = convergence_history
    current_result.metadata["final_convergence"] = isempty(convergence_history) ? NaN : last(convergence_history)
    
    return current_result
end

"""
    apply_image_deformation(img, displacement_field, window_size) -> deformed_image

Apply smooth deformation to an image based on a PIV displacement field.

Uses local affine transforms computed from displacement field to deform the image.
Each pixel's new location is computed by interpolating local transform parameters.

# Arguments
- `img::AbstractArray{<:Real,2}` - Input image to deform
- `displacement_field::PIVResult` - PIV result containing displacement vectors  
- `window_size::Tuple{Int,Int}` - Window size for local transform computation

# Returns
- `AbstractArray{<:Real,2}` - Deformed image with same size as input
"""
function apply_image_deformation(img::AbstractArray{<:Real,2}, 
                                displacement_field::PIVResult,
                                window_size::Tuple{Int,Int})
    
    deformed_img = zeros(eltype(img), size(img))
    
    # Filter valid displacement vectors
    valid_mask = .!isnan.(displacement_field.u) .& .!isnan.(displacement_field.v) .& 
                (displacement_field.status .== :good)
    
    if sum(valid_mask) < 3
        @warn "Insufficient valid displacement vectors for deformation, returning original image"
        return copy(img)
    end
    
    # Extract valid vectors
    valid_x = displacement_field.x[valid_mask]
    valid_y = displacement_field.y[valid_mask] 
    valid_u = displacement_field.u[valid_mask]
    valid_v = displacement_field.v[valid_mask]
    
    # For each pixel in the output image, find its source location
    for j in 1:size(img, 2), i in 1:size(img, 1)
        # Current pixel position
        px, py = Float64(i), Float64(j)
        
        # Compute local affine transform at this location
        transform = compute_local_affine_transform(
            px, py, valid_x, valid_y, valid_u, valid_v, window_size
        )
        
        if transform !== nothing && validate_affine_transform(transform)
            # Apply inverse transform to find source pixel
            # transform maps from original to deformed, so we need inverse
            try
                inv_transform = inv(transform)
                source_pos = inv_transform * [px; py; 1.0]
                source_x, source_y = source_pos[1], source_pos[2]
                
                # Bilinear interpolation from source image
                if 1 <= source_x <= size(img, 1) && 1 <= source_y <= size(img, 2)
                    deformed_img[i, j] = bilinear_interpolate(img, source_x, source_y)
                end
            catch
                # If transform inversion fails, use nearest neighbor
                nearest_idx = argmin((valid_x .- px).^2 + (valid_y .- py).^2)
                source_x = px - valid_u[nearest_idx]
                source_y = py - valid_v[nearest_idx]
                
                if 1 <= source_x <= size(img, 1) && 1 <= source_y <= size(img, 2)
                    deformed_img[i, j] = bilinear_interpolate(img, source_x, source_y)
                end
            end
        else
            # If no valid transform, use nearest neighbor displacement
            if !isempty(valid_x)
                distances = (valid_x .- px).^2 + (valid_y .- py).^2
                nearest_idx = argmin(distances)
                source_x = px - valid_u[nearest_idx]
                source_y = py - valid_v[nearest_idx]
                
                if 1 <= source_x <= size(img, 1) && 1 <= source_y <= size(img, 2)
                    deformed_img[i, j] = bilinear_interpolate(img, source_x, source_y)
                end
            end
        end
    end
    
    return deformed_img
end

"""
    compute_local_affine_transform(px, py, x_coords, y_coords, u_disps, v_disps, window_size)

Compute local affine transformation matrix at a specific point using nearby displacement vectors.

Fits a 2D affine transform (6 parameters) to displacement data within a local window.
The transform maps from original coordinates to deformed coordinates.

# Arguments
- `px, py::Float64` - Point where transform is computed
- `x_coords, y_coords::AbstractVector` - Coordinates of displacement vectors
- `u_disps, v_disps::AbstractVector` - Displacement components
- `window_size::Tuple{Int,Int}` - Size of local window for fitting

# Returns
- `Matrix{Float64}` - 3×3 homogeneous transformation matrix, or `nothing` if insufficient data

# Transform Format
```
[a11 a12 tx]   [x]   [x + u]
[a21 a22 ty] * [y] = [y + v]  
[0   0   1 ]   [1]   [1    ]
```
"""
function compute_local_affine_transform(px::Float64, py::Float64,
                                      x_coords::AbstractVector, y_coords::AbstractVector,
                                      u_disps::AbstractVector, v_disps::AbstractVector,
                                      window_size::Tuple{Int,Int})
    
    # Define search radius based on window size
    search_radius = max(window_size[1], window_size[2]) * 1.5
    
    # Find vectors within search radius
    distances = sqrt.((x_coords .- px).^2 + (y_coords .- py).^2)
    local_mask = distances .<= search_radius
    
    if sum(local_mask) < 6  # Need at least 6 points for 6-parameter affine transform
        return nothing
    end
    
    # Extract local data
    local_x = x_coords[local_mask]
    local_y = y_coords[local_mask]
    local_u = u_disps[local_mask]
    local_v = v_disps[local_mask]
    n_points = length(local_x)
    
    # Weight points by inverse distance (with small regularization)
    weights = 1.0 ./ (distances[local_mask] .+ 1.0)
    
    # Set up weighted least squares system for affine transform
    # For each point: [x_new, y_new] = [a11 a12 tx; a21 a22 ty] * [x_old; y_old; 1]
    # This gives us: x_old + u = a11*x_old + a12*y_old + tx
    #               y_old + v = a21*x_old + a22*y_old + ty
    
    # Build design matrix A and target vector b
    A = zeros(2*n_points, 6)
    b = zeros(2*n_points)
    W = zeros(2*n_points, 2*n_points)  # Weight matrix
    
    for i in 1:n_points
        # Equation for u displacement: u = (a11-1)*x + a12*y + tx
        row_u = 2*(i-1) + 1
        A[row_u, 1] = local_x[i]    # a11 coefficient  
        A[row_u, 2] = local_y[i]    # a12 coefficient
        A[row_u, 3] = 1.0           # tx coefficient
        b[row_u] = local_u[i] + local_x[i]  # target: x + u
        W[row_u, row_u] = weights[i]
        
        # Equation for v displacement: v = a21*x + (a22-1)*y + ty  
        row_v = 2*i
        A[row_v, 4] = local_x[i]    # a21 coefficient
        A[row_v, 5] = local_y[i]    # a22 coefficient  
        A[row_v, 6] = 1.0           # ty coefficient
        b[row_v] = local_v[i] + local_y[i]  # target: y + v
        W[row_v, row_v] = weights[i]
    end
    
    # Solve weighted least squares: (A'*W*A)*params = A'*W*b
    try
        AWA = A' * W * A
        AWb = A' * W * b
        
        # Add small regularization for numerical stability
        λ = 1e-6
        AWA += λ * I
        
        params = AWA \ AWb
        
        # Construct 3x3 homogeneous transformation matrix
        transform = [
            params[1] params[2] params[3];
            params[4] params[5] params[6];
            0.0       0.0       1.0
        ]
        
        return transform
        
    catch e
        # Return identity transform if solve fails
        @warn "Local affine transform computation failed at ($px, $py): $e"
        return nothing
    end
end

"""
    bilinear_interpolate(img, x, y) -> pixel_value

Bilinear interpolation of image intensity at non-integer coordinates.

# Arguments  
- `img::AbstractArray{<:Real,2}` - Input image
- `x, y::Float64` - Coordinates to interpolate (1-indexed)

# Returns
- `Real` - Interpolated pixel value
"""
function bilinear_interpolate(img::AbstractArray{<:Real,2}, x::Float64, y::Float64)
    # Get integer parts and fractional parts
    x_floor = floor(Int, x)
    y_floor = floor(Int, y)
    x_frac = x - x_floor
    y_frac = y - y_floor
    
    # Handle boundary conditions more carefully
    if x_floor < 1
        x_floor, x_frac = 1, 0.0
    elseif x_floor >= size(img, 1)
        x_floor, x_frac = size(img, 1), 0.0
    end
    
    if y_floor < 1
        y_floor, y_frac = 1, 0.0
    elseif y_floor >= size(img, 2)
        y_floor, y_frac = size(img, 2), 0.0
    end
    
    x_ceil = min(x_floor + 1, size(img, 1))
    y_ceil = min(y_floor + 1, size(img, 2))
    
    # Get four corner values
    val_00 = img[x_floor, y_floor]
    val_10 = img[x_ceil, y_floor] 
    val_01 = img[x_floor, y_ceil]
    val_11 = img[x_ceil, y_ceil]
    
    # Bilinear interpolation
    val_0 = val_00 * (1 - x_frac) + val_10 * x_frac
    val_1 = val_01 * (1 - x_frac) + val_11 * x_frac
    
    return val_0 * (1 - y_frac) + val_1 * y_frac
end

"""
    compute_convergence_metric(result1, result2) -> convergence_value

Compute convergence metric between two PIV results.

# Arguments
- `result1, result2::PIVResult` - PIV results to compare

# Returns  
- `Float64` - RMS displacement difference in pixels
"""
function compute_convergence_metric(result1::PIVResult, result2::PIVResult)
    # Check that results have same number of vectors
    if length(result1.vectors) != length(result2.vectors)
        throw(ArgumentError("PIV results must have same number of vectors for convergence comparison"))
    end
    
    # Find vectors that are valid in both results
    valid_mask = .!isnan.(result1.u) .& .!isnan.(result1.v) .& 
                .!isnan.(result2.u) .& .!isnan.(result2.v) .&
                (result1.status .== :good) .& (result2.status .== :good)
    
    if sum(valid_mask) == 0
        return Inf  # No valid comparison possible
    end
    
    # Compute RMS difference in displacement
    du = result2.u[valid_mask] - result1.u[valid_mask]
    dv = result2.v[valid_mask] - result1.v[valid_mask]
    
    rms_diff = sqrt(mean(du.^2 + dv.^2))
    
    return rms_diff
end


end # module Hammerhead
