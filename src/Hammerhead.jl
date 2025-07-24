module Hammerhead

using FFTW
using LinearAlgebra: mul!, ldiv!, inv, det, cond, eigvals, dot
using Statistics: median!, std
using ImageFiltering
using StructArrays
using Interpolations
using ImageIO
using ImageCore: Gray
using DSP
using TimerOutputs
using CairoMakie
using ChunkSplitters

# Data structures
export PIVVector, PIVResult, PIVStage, PIVStages

# Core functionality  
export run_piv

# Visualization (optional, requires CairoMakie)
export plot_piv

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

function Base.show(io::IO, r::PIVResult)
    frac_good = sum(r.vectors.status .== :good) / length(r.vectors)
    print(io, "PIVResult with $(length(r.vectors)) vectors (",
          "$(round(frac_good * 100, digits=1))% good)", r.metadata["timer"])
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
    build_window(size, window_function) -> window

Build a 2D window function for the given size using the specified window function type.
"""
function build_window(dims, window_function::WindowFunction)
    rows, cols = dims
    
    # Generate separable window using DSP.jl
    window_1d_row = generate_window_1d(window_function, rows)
    window_1d_col = generate_window_1d(window_function, cols)
    return [window_1d_row[i] * window_1d_col[j] for i in 1:rows, j in 1:cols]
end


"""
    apply_window!(subimage, window) -> windowed_subimage

Apply windowing array to subimage to reduce spectral leakage in correlation analysis.
"""
function apply_window!(subimg::AbstractArray{T,2}, window::AbstractArray{T}) where T
    rows, cols = size(subimg)
    if size(window) != (rows, cols)
        throw(ArgumentError("Window size $(size(window)) does not match subimage size $(rows), $(cols)"))
    end
    
    # Element-wise multiplication
    @inbounds for i in 1:rows, j in 1:cols
        subimg[i, j] *= window[i, j]
    end
    
    return subimg
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
struct PIVStage{W<:WindowFunction, I<:InterpolationMethod, V<:Tuple}
    window_size::Tuple{Int, Int}
    overlap::Tuple{Float64, Float64}
    padding::Int
    deformation_iterations::Int
    window_function::W
    interpolation_method::I
    validation::V  # Validation pipeline as parameterized tuple type
end

# Constructor with defaults using symbols or tuples for parametric windows
function PIVStage(window_size::Tuple{Int, Int}; 
                 overlap::Tuple{Float64, Float64} = (0.5, 0.5),
                 padding::Int = 0,
                 deformation_iterations::Int = 3,
                 window_function::Union{Symbol, Tuple{Symbol, Vararg{Real}}} = :rectangular,
                 interpolation_method::Symbol = :bilinear,
                 validation::Tuple = ())
    
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
    
    PIVStage(window_size, overlap, padding, deformation_iterations, wf, im, validation)
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
                   interpolation_method=:bilinear,
                   validation=())
    
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
    
    # Handle tuples - DO NOT convert to vector for validation pipelines  
    function get_stage_value(param::Tuple, i::Int, n_stages::Int)
        # For validation tuples, return as-is (they should be validation pipelines)
        # This prevents ambiguity with multi-stage PIV vectors
        return param
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
        stage_validation = get_stage_value(validation, i, n_stages)
        
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
                        interpolation_method=stage_interpolation_method,
                        validation=stage_validation)
        push!(stages, stage)
    end
    
    return stages
end

"""
    PIVStageCache{T, C}

Per-thread computational cache for PIV stage processing. Contains pre-computed
resources to avoid repeated allocations and enable thread-safe parallel processing.

# Fields
- `window::Matrix{T}` - Pre-computed window function matrix
- `correlator::C` - Thread-local correlator with FFT plans
- `vectors::Vector{PIVVector}` - Pre-allocated storage for results
- `windowed_img1::Matrix{T}` - Pre-allocated buffer for windowed subimage 1
- `windowed_img2::Matrix{T}` - Pre-allocated buffer for windowed subimage 2
"""
struct PIVStageCache{T, C <: Correlator}
    window::Matrix{T}  # Pre-computed window function matrix
    correlator::C      # Thread-local correlator
    vectors::Vector{PIVVector}  # Pre-allocated result storage
    windowed_img1::Matrix{T}   # Pre-allocated windowed image buffer 1
    windowed_img2::Matrix{T}   # Pre-allocated windowed image buffer 2
end

"""
    PIVStageCache(stage::PIVStage, ::Type{T}=Float32) -> PIVStageCache{T, CrossCorrelator{T}}

Create a per-thread computational cache from a PIV stage configuration.
Pre-computes the window function matrix and initializes thread-local correlator.
"""
function PIVStageCache(stage::PIVStage, ::Type{T}=Float32) where T
    # Pre-compute window function matrix
    window = Matrix{T}(build_window(stage.window_size, stage.window_function))
    
    # Create thread-local correlator
    correlator = CrossCorrelator{T}(stage.window_size)
    
    # Initialize result storage (will be resized as needed)
    vectors = PIVVector[]
    
    # Pre-allocate windowed image buffers
    windowed_img1 = Matrix{T}(undef, stage.window_size...)
    windowed_img2 = Matrix{T}(undef, stage.window_size...)
    
    return PIVStageCache{T, typeof(correlator)}(window, correlator, vectors, windowed_img1, windowed_img2)
end

"""
    apply_cached_window!(cache, subimg1, subimg2) -> (windowed1, windowed2)

Apply pre-computed window function to both subimages using pre-allocated buffers.
Returns references to the cached windowed image buffers for efficient memory usage.
"""
function apply_cached_window!(cache::PIVStageCache{T1},
                              subimg1::AbstractArray{T2}, 
                              subimg2::AbstractArray{T2}) where {T1, T2}
    # Copy subimages to pre-allocated buffers
    @. cache.windowed_img1 = T1(subimg1)
    @. cache.windowed_img2 = T1(subimg2)
    
    # Apply window function in-place
    apply_window!(cache.windowed_img1, cache.window)
    apply_window!(cache.windowed_img2, cache.window)
    
    return cache.windowed_img1, cache.windowed_img2
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
function calculate_quality_metrics(correlation_plane::AbstractMatrix{T}, peak_location::CartesianIndex, peak_value::Real; robust::Bool=false, work_buffer::AbstractMatrix{<:Real} = zeros(real(T), size(correlation_plane))) where {T}
    # Convert correlation plane to real magnitudes for analysis
    work_buffer .= abs.(correlation_plane)
    corr_mag = work_buffer
    
    # Find secondary peak using chosen method
    secondary_peak = if robust
        find_secondary_peak_robust(corr_mag, peak_location, peak_value)::real(T)
    else
        find_secondary_peak(corr_mag, peak_location, peak_value)::real(T)
    end
    
    # Calculate peak ratio (primary/secondary)
    peak_ratio = secondary_peak > 0 ? peak_value / secondary_peak : Inf
    
    # Calculate correlation moment (normalized second moment of peak)
    correlation_moment = calculate_correlation_moment(corr_mag, peak_location)
    
    return Float64(peak_ratio), Float64(correlation_moment)
end

# ============================================================================
# Vector Validation System
# ============================================================================

"""
Abstract base type for PIV vector validation algorithms.

Validators are applied sequentially to filter out unreliable displacement vectors.
Two main categories exist:
- `LocalValidator`: Tests individual vectors against thresholds (fast)
- `NeighborhoodValidator`: Tests vectors against local neighborhood statistics (slower)
"""
abstract type PIVValidator end

"""
Validators that examine individual vector properties without considering neighbors.
These can be efficiently batched together in a single loop.
"""
abstract type LocalValidator <: PIVValidator end

"""
Validators that examine vectors relative to their spatial neighbors.
These require knowledge of surrounding valid vectors and are applied separately.
"""
abstract type NeighborhoodValidator <: PIVValidator end

"""
    PeakRatioValidator(threshold::Float64)

Validates vectors based on correlation peak ratio (primary/secondary peak).
Higher ratios indicate more reliable correlations.

# Arguments
- `threshold`: Minimum acceptable peak ratio (typically 1.1-1.5)

# Example
```julia
validator = PeakRatioValidator(1.2)  # Require peak ratio ≥ 1.2
```
"""
struct PeakRatioValidator <: LocalValidator
    threshold::Float64
end

"""
    CorrelationMomentValidator(threshold::Float64)

Validates vectors based on correlation moment (peak sharpness).
Higher moments indicate sharper, more reliable peaks.

# Arguments  
- `threshold`: Minimum acceptable correlation moment (typically 0.1-0.5)

# Example
```julia
validator = CorrelationMomentValidator(0.2)  # Require moment ≥ 0.2
```
"""
struct CorrelationMomentValidator <: LocalValidator
    threshold::Float64
end

"""
    VelocityMagnitudeValidator(min::Float64, max::Float64)

Validates vectors based on displacement magnitude bounds.
Filters out physically unrealistic displacements.

# Arguments
- `min`: Minimum acceptable displacement magnitude
- `max`: Maximum acceptable displacement magnitude

# Example
```julia
validator = VelocityMagnitudeValidator(0.1, 50.0)  # Magnitude between 0.1-50 pixels
```
"""
struct VelocityMagnitudeValidator <: LocalValidator
    min::Float64
    max::Float64
end

"""
    LocalMedianValidator(window_size::Int, threshold::Float64)

Validates vectors against median of local neighborhood.
Vectors deviating more than `threshold` standard deviations from
local median are marked as outliers.

# Arguments
- `window_size`: Radius of neighborhood window (e.g., 3 = 7×7 window)
- `threshold`: Deviation threshold in standard deviations (typically 2-3)

# Example
```julia
validator = LocalMedianValidator(3, 2.0)  # 7×7 window, 2σ threshold
```
"""
struct LocalMedianValidator <: NeighborhoodValidator
    window_size::Int
    threshold::Float64
end

"""
    NormalizedResidualValidator(window_size::Int, threshold::Float64)

Validates vectors using normalized residual against local interpolated field.
More sophisticated than median test, accounts for local flow gradients.

# Arguments
- `window_size`: Radius of neighborhood for interpolation
- `threshold`: Normalized residual threshold (typically 2-4)

# Example
```julia
validator = NormalizedResidualValidator(5, 3.0)  # 11×11 window, 3σ threshold
```
"""
struct NormalizedResidualValidator <: NeighborhoodValidator
    window_size::Int
    threshold::Float64
end

"""
    parse_validator(spec) -> PIVValidator

Convert user specification to validator object.
Supports both object syntax and pair syntax.

# Examples
```julia
# Object syntax
parse_validator(PeakRatioValidator(1.2))

# Pair syntax
parse_validator(:peak_ratio => 1.2)
parse_validator(:local_median => (window_size=3, threshold=2.0))
```
"""
function parse_validator(validator::PIVValidator)
    return validator  # Already a validator object
end

function parse_validator(spec::Pair{Symbol, <:Real})
    symbol, threshold = spec
    return if symbol == :peak_ratio
        PeakRatioValidator(threshold)
    elseif symbol == :correlation_moment
        CorrelationMomentValidator(threshold)
    else
        error("Unknown validator: $symbol")
    end
end

function parse_validator(spec::Pair{Symbol, <:NamedTuple})
    symbol, config = spec
    return if symbol == :local_median
        LocalMedianValidator(config.window_size, config.threshold)
    elseif symbol == :normalized_residual
        NormalizedResidualValidator(config.window_size, config.threshold)
    elseif symbol == :velocity_magnitude
        VelocityMagnitudeValidator(config.min, config.max)
    else
        error("Unknown validator: $symbol")
    end
end

"""
    parse_validation_pipeline(pipeline::Tuple) -> Tuple of PIVValidators

Convert validation pipeline tuple to validator objects.
Uses tuple to distinguish from multi-stage PIV vector parameters.

# Examples
```julia
# Object syntax
pipeline = (PeakRatioValidator(1.2), LocalMedianValidator(3, 2.0))

# Pair syntax  
pipeline = (:peak_ratio => 1.2, :local_median => (window_size=3, threshold=2.0))
```
"""
function parse_validation_pipeline(pipeline::Tuple)
    return Tuple(parse_validator(spec) for spec in pipeline)
end

function parse_validation_pipeline(pipeline::Tuple{})
    return ()  # Empty pipeline
end

"""
    validate_vectors!(result::PIVResult, validation_pipeline::Tuple)

Apply validation pipeline to PIV result, optimizing for performance.
Local validators are batched together, neighborhood validators applied separately.
"""
function validate_vectors!(result::PIVResult, validation_pipeline::Tuple)
    if isempty(validation_pipeline)
        return  # No validation to perform
    end
    
    validators = parse_validation_pipeline(validation_pipeline)
    
    # Internal method to group adjacent local validators
    function group_batches(validators)
        batches = []
        i = 1
        
        while i <= length(validators)
            if validators[i] isa LocalValidator
                # Collect adjacent local validators
                local_batch = LocalValidator[]
                while i <= length(validators) && validators[i] isa LocalValidator
                    push!(local_batch, validators[i])
                    i += 1
                end
                # Return single validator or vector of validators
                if length(local_batch) == 1
                    push!(batches, local_batch[1])
                else
                    push!(batches, local_batch)
                end
            else
                # Single neighborhood validator
                push!(batches, validators[i])
                i += 1
            end
        end
        return batches
    end
    
    # Internal method dispatch for different validator batch types
    apply_batch!(validator::LocalValidator, valid_mask) = begin
        apply_local_validator!(result.vectors, validator)
        update_valid_mask!(valid_mask, result.vectors)
    end
    
    apply_batch!(validators::Vector{<:LocalValidator}, valid_mask) = begin
        apply_local_validators!(result.vectors, validators)
        update_valid_mask!(valid_mask, result.vectors)
    end
    
    apply_batch!(validator::NeighborhoodValidator, valid_mask) = begin
        apply_neighborhood_validator!(result.vectors, validator, valid_mask)
        update_valid_mask!(valid_mask, result.vectors)
    end
    
    # Group validators and process batches in sequence
    validator_batches = group_batches(validators)
    valid_mask = get_valid_mask(result.vectors)
    
    for batch in validator_batches
        apply_batch!(batch, valid_mask)
    end
end

"""
    apply_local_validator!(vectors::StructArray{PIVVector}, validator::LocalValidator)

Apply single local validator to vector field.
"""
function apply_local_validator!(vectors::StructArray{PIVVector}, validator::LocalValidator)
    for i in eachindex(vectors)
        vector = vectors[i]
        if vector.status == :good && !is_valid(vector, validator)
            vectors.status[i] = :bad
        end
    end
end

"""
    apply_local_validators!(vectors::StructArray{PIVVector}, validators::Vector{<:LocalValidator})

Batch local validators into single loop for efficiency.
"""
function apply_local_validators!(vectors::StructArray{PIVVector}, validators::Vector{<:LocalValidator})
    for i in eachindex(vectors)
        vector = vectors[i]
        if vector.status == :good  # Only validate good vectors
            for validator in validators
                if !is_valid(vector, validator)
                    vectors.status[i] = :bad
                    break  # No need to test further once marked bad
                end
            end
        end
    end
end

"""
    is_valid(vector::PIVVector, validator::LocalValidator) -> Bool

Test if individual vector passes validation criteria.
"""
is_valid(vector::PIVVector, v::PeakRatioValidator) = vector.peak_ratio >= v.threshold
is_valid(vector::PIVVector, v::CorrelationMomentValidator) = vector.correlation_moment >= v.threshold

function is_valid(vector::PIVVector, v::VelocityMagnitudeValidator)
    mag = sqrt(vector.u^2 + vector.v^2)
    return v.min <= mag <= v.max
end

"""
    get_valid_mask(vectors::StructArray{PIVVector}) -> BitMatrix

Create mask of currently valid vectors for neighborhood operations.
"""
function get_valid_mask(vectors::StructArray{PIVVector})
    rows, cols = size(vectors)
    mask = BitMatrix(undef, rows, cols)
    for i in 1:rows, j in 1:cols
        mask[i, j] = vectors[i, j].status == :good
    end
    return mask
end

"""
    update_valid_mask!(mask::BitMatrix, vectors::StructArray{PIVVector})

Update validity mask after validation step.
"""
function update_valid_mask!(mask::BitMatrix, vectors::StructArray{PIVVector})
    rows, cols = size(vectors)
    for i in 1:rows, j in 1:cols
        mask[i, j] = vectors[i, j].status == :good
    end
end

function apply_neighborhood_validator!(vectors::StructArray{PIVVector}, validator::LocalMedianValidator, valid_mask::BitMatrix)
    rows, cols = size(vectors)
    window_size = validator.window_size
    threshold = validator.threshold
    
    # Pre-allocate work buffers
    max_neighbors = (2 * window_size + 1)^2 - 1
    u_buffer = Vector{Float64}()
    v_buffer = Vector{Float64}()
    sizehint!(u_buffer, max_neighbors)
    sizehint!(v_buffer, max_neighbors)
    
    for i in 1:rows, j in 1:cols
        if vectors[i, j].status != :good
            continue  # Skip vectors that are already marked as bad
        end
        
        # Clear and collect neighborhood vectors
        empty!(u_buffer)
        empty!(v_buffer)
        
        # Define neighborhood bounds
        i_min = max(1, i - window_size)
        i_max = min(rows, i + window_size)
        j_min = max(1, j - window_size)
        j_max = min(cols, j + window_size)
        
        # Collect valid neighbors (excluding center point)
        for ni in i_min:i_max, nj in j_min:j_max
            if (ni != i || nj != j) && valid_mask[ni, nj]
                push!(u_buffer, vectors[ni, nj].u)
                push!(v_buffer, vectors[ni, nj].v)
            end
        end
        
        if length(u_buffer) < 3
            # Not enough neighbors for robust statistics
            continue
        end
        
        # Calculate median and standard deviation (using in-place median for efficiency)
        median_u = median!(u_buffer)
        median_v = median!(v_buffer)
        std_u = std(u_buffer)  # Note: u_buffer is now sorted from median!
        std_v = std(v_buffer)  # Note: v_buffer is now sorted from median!
        
        # Test current vector against neighborhood statistics
        current_u = vectors[i, j].u
        current_v = vectors[i, j].v
        
        # Normalized deviation from median
        dev_u = abs(current_u - median_u) / (std_u + 1e-10)  # Small epsilon to avoid division by zero
        dev_v = abs(current_v - median_v) / (std_v + 1e-10)
        
        # Mark as outlier if either component exceeds threshold
        if dev_u > threshold || dev_v > threshold
            vectors.status[i, j] = :bad
        end
    end
end

function apply_neighborhood_validator!(vectors::StructArray{PIVVector}, validator::NormalizedResidualValidator, valid_mask::BitMatrix)
    rows, cols = size(vectors)
    window_size = validator.window_size
    threshold = validator.threshold
    
    # Pre-allocate work buffers
    max_neighbors = (2 * window_size + 1)^2 - 1
    u_buffer = Vector{Float64}()
    v_buffer = Vector{Float64}()
    x_buffer = Vector{Float64}()
    y_buffer = Vector{Float64}()
    sizehint!(u_buffer, max_neighbors)
    sizehint!(v_buffer, max_neighbors)
    sizehint!(x_buffer, max_neighbors)
    sizehint!(y_buffer, max_neighbors)
    
    for i in 1:rows, j in 1:cols
        if vectors[i, j].status != :good
            continue  # Skip vectors that are already marked as bad
        end
        
        # Clear and collect neighborhood vectors with positions
        empty!(u_buffer)
        empty!(v_buffer) 
        empty!(x_buffer)
        empty!(y_buffer)
        
        # Define neighborhood bounds
        i_min = max(1, i - window_size)
        i_max = min(rows, i + window_size)
        j_min = max(1, j - window_size)
        j_max = min(cols, j + window_size)
        
        # Collect valid neighbors with their positions (excluding center point)
        for ni in i_min:i_max, nj in j_min:j_max
            if (ni != i || nj != j) && valid_mask[ni, nj]
                push!(u_buffer, vectors[ni, nj].u)
                push!(v_buffer, vectors[ni, nj].v)
                push!(x_buffer, vectors[ni, nj].x)
                push!(y_buffer, vectors[ni, nj].y)
            end
        end
        
        if length(u_buffer) < 4
            # Not enough neighbors for interpolation (need at least 4 points for bilinear)
            continue
        end
        
        # Current vector position
        current_x = vectors[i, j].x
        current_y = vectors[i, j].y
        current_u = vectors[i, j].u
        current_v = vectors[i, j].v
        
        # Interpolate expected velocity at current position using neighbors
        # Simple inverse distance weighting interpolation
        total_weight = 0.0
        interp_u = 0.0
        interp_v = 0.0
        
        for k in 1:length(x_buffer)
            # Distance to neighbor
            dx = current_x - x_buffer[k]
            dy = current_y - y_buffer[k]
            dist_sq = dx*dx + dy*dy
            
            if dist_sq > 1e-10  # Avoid division by zero
                weight = 1.0 / (dist_sq + 1e-6)  # Inverse distance weighting with small regularization
                interp_u += weight * u_buffer[k]
                interp_v += weight * v_buffer[k]
                total_weight += weight
            end
        end
        
        if total_weight > 1e-10
            interp_u /= total_weight
            interp_v /= total_weight
            
            # Calculate residuals
            residual_u = current_u - interp_u
            residual_v = current_v - interp_v
            
            # Estimate local variance from neighborhood
            variance_u = 0.0
            variance_v = 0.0
            for k in 1:length(u_buffer)
                diff_u = u_buffer[k] - interp_u
                diff_v = v_buffer[k] - interp_v
                variance_u += diff_u * diff_u
                variance_v += diff_v * diff_v
            end
            variance_u /= length(u_buffer)
            variance_v /= length(v_buffer)
            
            # Normalized residuals
            std_u = sqrt(variance_u + 1e-10)  # Small epsilon to avoid division by zero
            std_v = sqrt(variance_v + 1e-10)
            
            norm_residual_u = abs(residual_u) / std_u
            norm_residual_v = abs(residual_v) / std_v
            
            # Mark as outlier if either component exceeds threshold
            if norm_residual_u > threshold || norm_residual_v > threshold
                vectors.status[i, j] = :bad
            end
        end
    end
end

# ============================================================================
# Validators Sub-Module
# ============================================================================

"""
    Validators

Sub-module providing clean access to PIV vector validation types.

# Usage
```julia
using Hammerhead.Validators

# Clean validator names without "Validator" suffix
validation = (PeakRatio(1.2), LocalMedian(window_size=3, threshold=2.0))
```
"""
module Validators

# Import parent module types and functions
import ..PIVValidator, ..LocalValidator, ..NeighborhoodValidator
import ..PeakRatioValidator, ..CorrelationMomentValidator, ..VelocityMagnitudeValidator
import ..LocalMedianValidator, ..NormalizedResidualValidator

# Export clean names without "Validator" suffix
export PeakRatio, CorrelationMoment, VelocityMagnitude, LocalMedian, NormalizedResidual

# Clean type aliases
"""
    PeakRatio(threshold::Float64)

Validates vectors based on correlation peak ratio (primary/secondary peak).
Alias for `PeakRatioValidator`.
"""
const PeakRatio = PeakRatioValidator

"""
    CorrelationMoment(threshold::Float64)

Validates vectors based on correlation moment (peak sharpness).
Alias for `CorrelationMomentValidator`.
"""
const CorrelationMoment = CorrelationMomentValidator

"""
    VelocityMagnitude(min::Float64, max::Float64)

Validates vectors based on displacement magnitude bounds.
Alias for `VelocityMagnitudeValidator`.
"""
const VelocityMagnitude = VelocityMagnitudeValidator

"""
    LocalMedian(window_size::Int, threshold::Float64)

Validates vectors against median of local neighborhood.
Alias for `LocalMedianValidator`.
"""
const LocalMedian = LocalMedianValidator

"""
    NormalizedResidual(window_size::Int, threshold::Float64)

Validates vectors using normalized residual against local interpolated field.
Alias for `NormalizedResidualValidator`.
"""
const NormalizedResidual = NormalizedResidualValidator

end # module Validators

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

function correlate!(c::CrossCorrelator{T}, subimgA::AbstractArray, subimgB::AbstractArray, 
        timer = TimerOutput()) where {T}
    @timeit timer "FFT Setup" begin
        c.C1 .= T.(subimgA)
        c.C2 .= T.(subimgB)
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
function analyze_correlation_plane(correlation_plane::AbstractMatrix, work_buffer::AbstractMatrix{<:Real} = zeros(real(eltype(correlation_plane)), size(correlation_plane)))
    # Find the peak in the correlation result
    center = size(correlation_plane) .÷ 2 .+ 1
    peakloc = CartesianIndex(center...)  # Initialize to center instead of (0,0)
    maxval = zero(real(eltype(correlation_plane)))
    for i in CartesianIndices(correlation_plane)
        absval = abs(correlation_plane[i])
        if absval > maxval
            maxval = absval
            peakloc = i
        end
    end

    # Handle degenerate case where maxval is still zero
    if maxval == 0
        # Return NaN displacement for all-zero correlation plane
        nan_val = real(eltype(correlation_plane))(NaN)
        return (nan_val, nan_val, nan_val, nan_val)
    end

    # Calculate quality metrics
    peak_ratio, correlation_moment = calculate_quality_metrics(correlation_plane, peakloc, maxval, work_buffer=work_buffer)

    # Perform subpixel refinement and compute displacement relative to center
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
- `window_size::Tuple{Int,Int}` - Window size for single-stage (default: (64,64))
- `overlap::Tuple{Float64,Float64}` - Overlap ratios (default: (0.5,0.5))

# Returns
- `PIVResult` for single-stage processing
- `Vector{PIVResult}` for multi-stage processing
"""
function run_piv(img1::Matrix{T}, img2::Matrix{T}; 
                 window_size::Tuple{Int,Int}=(64,64),
                 overlap::Tuple{Float64,Float64}=(0.5,0.5), kwargs...)  where {T <: Union{Real, Gray}}
    
    result = run_piv(img1, img2, [PIVStage(window_size, overlap=overlap, kwargs...)])

    return only(result)
end

# Multi-stage version
function run_piv(img1_raw::Matrix{T}, img2_raw::Matrix{T}, stages::Vector{<:PIVStage};
                 kwargs...) where {T <: Union{Real, Gray}}
    
    # Create local timer for this PIV run
    timer = TimerOutput()

    # Helper function to promote images to Float32 (default) or Float64 if image type is Gray{Float64} or similar
    function promote_image_type(img::AbstractArray{T,2}) where T
        if (T <: Gray{Float64}) || (T <: Float64)
            return Float64.(img)  # Promote to Float64
        else
            return Float32.(img)  # Default to Float32 for other types
        end
    end

    # Promote images to Float32 or Float64 as needed
    
    img1 = @timeit timer "Image type promotion" promote_image_type(img1_raw)
    img2 = @timeit timer "Image type promotion" promote_image_type(img2_raw)
    
    @timeit timer "PIV processing" begin
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

                result = run_piv_stage(img1, img2, stage, eltype(img1), timer)
                
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
    process_chunk!(cache, chunk, grid_x, grid_y, img1, img2, stage, timer=nothing)

Process a chunk of interrogation windows using the provided cache.
Optionally uses timer for detailed performance tracking (thread 1 only).
"""
function process_chunk!(cache::PIVStageCache{T}, chunk, grid_x, grid_y, img1, img2, stage::PIVStage, timer=nothing) where T
    for i in chunk
        window_x = grid_x[i]
        window_y = grid_y[i]
        
        try
            # Calculate window bounds with boundary checking
            x_start = max(1, round(Int, window_x - stage.window_size[1]//2))
            x_end = min(size(img1, 1), x_start + stage.window_size[1] - 1)
            y_start = max(1, round(Int, window_y - stage.window_size[2]//2))
            y_end = min(size(img1, 2), y_start + stage.window_size[2] - 1)
            
            # Extract and convert subimages
            subimg1 = @view img1[x_start:x_end, y_start:y_end]
            subimg2 = @view img2[x_start:x_end, y_start:y_end]
            
            # Pad if necessary to match window size
            if size(subimg1) != stage.window_size
                subimg1 = pad_to_size(subimg1, stage.window_size)
                subimg2 = pad_to_size(subimg2, stage.window_size)
            end
            
            # Apply windowing using cached window function
            windowed_subimg1, windowed_subimg2 = if timer !== nothing
                @timeit timer "Applying window" apply_cached_window!(cache, subimg1, subimg2)
            else
                apply_cached_window!(cache, subimg1, subimg2)
            end
            
            # Perform correlation using cached correlator (with or without timing)
            correlation_plane = if timer !== nothing
                @timeit timer "Correlating" correlate!(cache.correlator, windowed_subimg1, windowed_subimg2, timer)
            else
                correlate!(cache.correlator, windowed_subimg1, windowed_subimg2)
            end
            
            # Analyze correlation plane for displacement and quality metrics
            disp_u, disp_v, peak_ratio, corr_moment = if timer !== nothing
                @timeit timer "Analyzing correlation plane" analyze_correlation_plane(correlation_plane, cache.windowed_img1)
            else
                analyze_correlation_plane(correlation_plane, cache.windowed_img1)
            end

            
            # Store result in cache
            vector = PIVVector(window_x, window_y, disp_u, disp_v, :good, peak_ratio, corr_moment)
            push!(cache.vectors, vector)
            
        catch e
            # Handle correlation failures gracefully
            vector = PIVVector(window_x, window_y, NaN, NaN, :bad, NaN, NaN)
            push!(cache.vectors, vector)
        end
    end
end

"""
    run_piv_stage(img1, img2, stage, ::Type{T}=Float32) -> PIVResult

Perform PIV analysis for a single stage with parallel processing using ChunkSplitters.jl.
Pre-allocates per-thread caches and merges results efficiently.
"""
function run_piv_stage(img1::AbstractArray{<:Real,2}, img2::AbstractArray{<:Real,2}, 
                                stage::PIVStage, ::Type{T}=Float32, timer = TimerOutput()) where T
    
    # Generate interrogation window grid
    grid_x, grid_y = generate_interrogation_grid(size(img1), stage.window_size, stage.overlap)
    n_windows = length(grid_x)
    
    # Pre-allocate per-thread caches
    n_threads = Threads.nthreads()
    caches = [PIVStageCache(stage, T) for _ in 1:n_threads]
    
    # Process windows in parallel using ChunkSplitters
    Threads.@threads for (thread_id, chunk) in collect(enumerate(chunks(1:n_windows; n=n_threads)))
        cache = caches[thread_id]
        
        # Clear previous results and resize for this chunk
        empty!(cache.vectors)
        sizehint!(cache.vectors, length(chunk))
        
        # Only do detailed timing on thread 1 to keep output clean
        if thread_id == 1
            process_chunk!(cache, chunk, grid_x, grid_y, img1, img2, stage, timer)
        else
            # Other threads process without detailed timing
            process_chunk!(cache, chunk, grid_x, grid_y, img1, img2, stage)
        end
    end
    
    # Merge results from all thread caches using reduce
    all_vectors = reduce(vcat, [cache.vectors for cache in caches])
    
    # Create result with metadata
    result = PIVResult(all_vectors)
    result.metadata["image_size"] = size(img1)
    result.metadata["window_size"] = stage.window_size
    result.metadata["overlap"] = stage.overlap
    result.metadata["n_windows"] = n_windows
    result.metadata["n_threads"] = n_threads
    
    # Apply validation pipeline if configured
    if !isempty(stage.validation)
        if timer !== nothing
            @timeit timer "Vector Validation" validate_vectors!(result, stage.validation)
        else
            validate_vectors!(result, stage.validation)
        end
    end
    
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
    plot_piv(background_image, piv_result; kwargs...) -> Figure

Create an interactive vector plot of PIV results overlaid on a background image.
Vectors are colored by their status and scaled for visibility.

# Arguments
- `background_image::AbstractMatrix` - Background image to display
- `piv_result::PIVResult` - PIV analysis results to visualize

# Keyword Arguments
- `scale::Float64 = 5.0` - Vector scaling factor for visibility
- `arrow_size::Float64 = 0.02` - Width of vector arrowheads (relative to plot size)
- `colormap_good = :viridis` - Colormap for good vectors (by magnitude)
- `show_bad::Bool = true` - Whether to show bad/interpolated vectors
- `title::String = "PIV Results"` - Plot title
- `vector_width::Float64 = 2.0` - Line width for vectors

# Vector Status Colors
- `:good` - Colored by velocity magnitude using specified colormap
- `:interpolated` - Orange
- `:bad` - Red
- `:secondary` - Yellow

# Returns
- `Figure` - CairoMakie figure object that can be displayed or saved

# Examples
```julia
# Basic usage
fig = plot_piv(img1, result)
display(fig)

# Customized visualization
fig = plot_piv(img1, result, 
                      scale=10.0, 
                      colormap_good=:plasma,
                      title="PIV Analysis - Flow Field")
```

# Requirements
Requires CairoMakie.jl to be loaded: `using CairoMakie`
"""
function plot_piv(background_image::AbstractMatrix, piv_result::PIVResult;
                          scale::Float64 = 5.0,
                          arrow_size::Float64 = 0.02,
                          colormap_good = :viridis,
                          show_bad::Bool = true,
                          title::String = "PIV Results",
                          vector_width::Float64 = 2.0)
    
    # Create figure
    fig = Main.CairoMakie.Figure(size = (800, 600))
    ax = Main.CairoMakie.Axis(fig[1, 1], 
                          title = title,
                          xlabel = "X (pixels)", 
                          ylabel = "Y (pixels)",
                          aspect = Main.CairoMakie.DataAspect())
    
    # Display background image (flip Y axis to match image coordinates)
    img_extent = (1, size(background_image, 2), 1, size(background_image, 1))
    Main.CairoMakie.image!(ax, background_image, colormap = :bone)
    
    # Separate vectors by status
    good_mask = piv_result.status .== :good
    interpolated_mask = piv_result.status .== :interpolated
    bad_mask = piv_result.status .== :bad
    secondary_mask = piv_result.status .== :secondary
    
    # Plot good vectors with magnitude-based coloring
    if any(good_mask)
        x_good = piv_result.x[good_mask]
        y_good = piv_result.y[good_mask]
        u_good = piv_result.u[good_mask] .* scale
        v_good = piv_result.v[good_mask] .* scale
        
        # Calculate velocity magnitude for coloring
        magnitude = sqrt.(u_good.^2 .+ v_good.^2)
        
        # Only plot if we have finite vectors
        finite_mask = isfinite.(u_good) .& isfinite.(v_good)
        if any(finite_mask)
            Main.CairoMakie.arrows!(ax, 
                               x_good[finite_mask], y_good[finite_mask],
                               u_good[finite_mask], v_good[finite_mask],
                               color = magnitude[finite_mask],
                               colormap = colormap_good,
                               tipwidth = arrow_size,
                               linewidth = vector_width,
                               label = "Good")
        end
    end
    
    # Plot other status vectors if requested
    if show_bad
        # Interpolated vectors (orange)
        if any(interpolated_mask)
            x_interp = piv_result.x[interpolated_mask]
            y_interp = piv_result.y[interpolated_mask]
            u_interp = piv_result.u[interpolated_mask] .* scale
            v_interp = piv_result.v[interpolated_mask] .* scale
            
            finite_mask = isfinite.(u_interp) .& isfinite.(v_interp)
            if any(finite_mask)
                Main.CairoMakie.arrows!(ax,
                                   x_interp[finite_mask], y_interp[finite_mask],
                                   u_interp[finite_mask], v_interp[finite_mask],
                                   color = :orange,
                                   tipwidth = arrow_size,
                                   linewidth = vector_width,
                                   label = "Interpolated")
            end
        end
        
        # Bad vectors (red)
        if any(bad_mask)
            x_bad = piv_result.x[bad_mask]
            y_bad = piv_result.y[bad_mask]
            u_bad = piv_result.u[bad_mask] .* scale
            v_bad = piv_result.v[bad_mask] .* scale
            
            finite_mask = isfinite.(u_bad) .& isfinite.(v_bad)
            if any(finite_mask)
                Main.CairoMakie.arrows!(ax,
                                   x_bad[finite_mask], y_bad[finite_mask],
                                   u_bad[finite_mask], v_bad[finite_mask],
                                   color = :red,
                                   tipwidth = arrow_size,
                                   linewidth = vector_width,
                                   label = "Bad")
            end
        end
        
        # Secondary vectors (yellow)
        if any(secondary_mask)
            x_sec = piv_result.x[secondary_mask]
            y_sec = piv_result.y[secondary_mask]
            u_sec = piv_result.u[secondary_mask] .* scale
            v_sec = piv_result.v[secondary_mask] .* scale
            
            finite_mask = isfinite.(u_sec) .& isfinite.(v_sec)
            if any(finite_mask)
                Main.CairoMakie.arrows!(ax,
                                   x_sec[finite_mask], y_sec[finite_mask],
                                   u_sec[finite_mask], v_sec[finite_mask],
                                   color = :yellow,
                                   tipwidth = arrow_size,
                                   linewidth = vector_width,
                                   label = "Secondary")
            end
        end
    end
    
    # Add colorbar for good vectors
    if any(good_mask)
        good_u = piv_result.u[good_mask] .* scale
        good_v = piv_result.v[good_mask] .* scale
        finite_mask = isfinite.(good_u) .& isfinite.(good_v)
        
        if any(finite_mask)
            magnitude = sqrt.(good_u[finite_mask].^2 .+ good_v[finite_mask].^2)
            if !isempty(magnitude) && any(isfinite.(magnitude))
                Main.CairoMakie.Colorbar(fig[1, 2], 
                                    limits = (minimum(magnitude), maximum(magnitude)),
                                    colormap = colormap_good,
                                    label = "Velocity Magnitude (scaled)")
            end
        end
    end
    
    # Flip Y axis to match image coordinates (Y increases downward)
    Main.CairoMakie.ylims!(ax, size(background_image, 1), 1)
    
    return fig
end


end # module Hammerhead
