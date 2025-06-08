using FFTW
using LsqFit

function cross_correlation(im1::AbstractArray, im2::AbstractArray, window_size::Tuple{Int, Int})
    # Ensure images are of the same size
    @assert size(im1) == size(im2) "Images must have the same dimensions"

    # Extract window dimensions
    win_x, win_y = window_size

    # Preallocate displacement fields
    u = zeros(Float64, size(im1))
    v = zeros(Float64, size(im1))

    # Iterate over the image in windows
    for i in 1:win_x:size(im1, 1)-win_x
        for j in 1:win_y:size(im1, 2)-win_y
            # Extract sub-images (windows)
            subimgA = im1[i:i+win_x-1, j:j+win_y-1]
            subimgB = im2[i:i+win_x-1, j:j+win_y-1]

            # Perform FFT-based cross-correlation
            fftA = fft(subimgA)
            fftB = fft(subimgB)
            cross_corr = ifft(fftA .* conj(fftB))

            # Find the peak in the cross-correlation
            max_index = argmax(abs.(cross_corr))
            peak_coords = Tuple(CartesianIndices(size(cross_corr))[max_index])

            # Subpixel refinement using least-squares Gaussian fit
            refined_coords = fit_gaussian_peak(cross_corr)

            # Compute displacements
            u[i, j] = refined_coords[1] - win_x / 2
            v[i, j] = refined_coords[2] - win_y / 2
        end
    end

    return u, v
end

function subpixel_peak(correlation_matrix, peak_coords)
    # Extract a 3x3 neighborhood around the peak
    dims = ndims(correlation_matrix)
    neighborhood = CartesianIndices((max(1, peak_coords[1]-1):min(size(correlation_matrix, 1), peak_coords[1]+1),
                                     max(1, peak_coords[2]-1):min(size(correlation_matrix, 2), peak_coords[2]+1)))

    # Flatten the correlation matrix
    data = vec(correlation_matrix)

    # Initial guess for parameters: amplitude, xo, yo, sigma_x, sigma_y, theta, offset
    p0 = [maximum(data), size_x / 2, size_y / 2, 1.0, 1.0, 0.0, minimum(data)]

    # Perform the fit
    fit = curve_fit(gaussian_2d, xy, data, p0)

    # Extract the fitted parameters
    amplitude, xo, yo, sigma_x, sigma_y, theta, offset = fit.param

    return (xo, yo)
end

function run_piv_pipeline(image1::AbstractArray, image2::AbstractArray, window_size::Tuple{Int, Int})
    println("Running PIV pipeline...")

    # Perform cross-correlation to compute displacement fields
    u, v = cross_correlation(image1, image2, window_size)

    println("PIV analysis complete.")
    return u, v
end