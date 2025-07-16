module Hammerhead

using FFTW
using LinearAlgebra: mul!, ldiv!, inv
using ImageFiltering
using StructArrays
using Interpolations
using ImageIO

# Data structures
export PIVVector, PIVResult, PIVStage

# Core functionality  
export run_piv, CrossCorrelator, correlate

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

# Helper function to create multi-stage progression
function PIVStages(n_stages::Int, final_size::Int, overlap::Float64 = 0.5)
    if n_stages <= 0
        throw(ArgumentError("Number of stages must be positive"))
    end
    
    stages = PIVStage[]
    
    for i in 1:n_stages
        # Geometric progression: start with larger windows, end with final_size
        size_factor = 2.0^(n_stages - i)
        current_size = round(Int, final_size * size_factor)
        
        # Ensure minimum window size
        current_size = max(current_size, final_size)
        
        stage = PIVStage(current_size, overlap=(overlap, overlap))
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

function correlate(c::CrossCorrelator, subimgA::AbstractArray, subimgB::AbstractArray)
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

    # Find the peak in the correlation result
    peakloc = CartesianIndex(0, 0)
    maxval = zero(real(eltype(c.C2)))
    for i in CartesianIndices(c.C2)
        absval = abs(c.C2[i])
        if absval > maxval
            maxval = absval
            peakloc = i
        end
    end

    # Perform subpixel refinement and compute displacement relative to center
    center = size(c.C2) .÷ 2 .+ 1
    refined_peakloc = subpixel_gauss3(c.C2, peakloc.I)
    disp = center .- refined_peakloc

    # Return displacement as (x, y) = (row_offset, col_offset)
    return (disp[1], disp[2])
end

# Placeholder for preprocessing functionality
function preprocess_images(image_paths::Vector{String})
    println("Preprocessing images...")
    # Add image registration, distortion correction, etc.
end

# Core PIV correlation engine
function phase_correlation(image1::AbstractArray, image2::AbstractArray)
    println("Performing phase correlation...")
    # Placeholder for phase correlation logic
    # Compute FFT of both images, calculate cross-power spectrum, and find peak
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
    return run_piv_single_stage(img1, img2, stage, correlator)
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
        result = run_piv_single_stage(img1, img2, stage, correlator)
        
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
    run_piv_single_stage(img1, img2, stage, correlator) -> PIVResult

Perform single-stage PIV analysis with given configuration.
"""
function run_piv_single_stage(img1::AbstractArray{<:Real,2}, img2::AbstractArray{<:Real,2}, 
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
            
            # Perform correlation
            displacement = correlate(correlator, subimg1, subimg2)
            
            # Store results
            push!(positions_x, window_x)
            push!(positions_y, window_y)
            push!(displacements_u, displacement[1])
            push!(displacements_v, displacement[2])
            push!(status_flags, :good)
            push!(peak_ratios, NaN)  # TODO: Calculate actual peak ratio
            push!(correlation_moments, NaN)  # TODO: Calculate actual correlation moment
            
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

# Placeholder for postprocessing functionality
function postprocess_results(results)
    println("Postprocessing results...")
    # Add outlier detection, uncertainty quantification, etc.
end

end # module Hammerhead
