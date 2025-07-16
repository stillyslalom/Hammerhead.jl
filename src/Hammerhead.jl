module Hammerhead

using FFTW
using LinearAlgebra: mul!, ldiv!, inv
using ImageFiltering

export CrossCorrelator, correlate, subpixel_gauss3

# Define a Correlator type to encapsulate correlation methods and options
abstract type Correlator end

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

# Placeholder for the core PIV engine
function run_piv(image_pairs::Vector{Tuple{String, String}})
    println("Running PIV analysis...")
    # Add phase correlation, cross-correlation, etc.
end

# Placeholder for postprocessing functionality
function postprocess_results(results)
    println("Postprocessing results...")
    # Add outlier detection, uncertainty quantification, etc.
end

end # module Hammerhead
