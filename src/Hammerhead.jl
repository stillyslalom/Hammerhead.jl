module Hammerhead

using FFTW
using LinearAlgebra: mul!, ldiv!, inv
using ImageFiltering
using Interpolations
using LsqFit

# Include submodules
include("Preprocessing.jl")
include("Postprocessing.jl")
include("Visualization.jl")
include("UserInterface.jl")

export CrossCorrelator, correlate, phase_correlate, AffineTransform, warp_image, correlate_deformable, subpixel_gauss2d
export Preprocessing # Export the submodule itself
export Postprocessing # Export the submodule itself
export Visualization # Export the submodule itself
export UserInterface # Export the submodule itself
# Re-export specific items from UserInterface for convenience
export PIVParameters, run_piv_from_gui, process_piv_pair, batch_process_piv

# Define a Correlator type to encapsulate correlation methods and options
abstract type Correlator end


# Make submodules accessible within Hammerhead module itself
using .Preprocessing
using .Postprocessing
using .Visualization
using .UserInterface
using Random # For dummy image data
using HDF5   # For saving results

# Main PIV Processing Pipeline
function process_piv_pair(
    image_path1::String,
    image_path2::String,
    params::PIVParameters;
    output_path::Union{String, Nothing}=nothing
)
    println("Starting PIV processing for '$image_path1' and '$image_path2'.")
    # show(stdout, params) # PIVParameters custom show method

    # 1. Load Images (Placeholder)
    println("Step 1: Loading images...")
    # Replace with actual image loading. For now, dummy 2D arrays.
    # Ensure images are Float based for processing if not loaded as such
    dummy_img_size = (params.window_size_y * 5, params.window_size_x * 5) # Ensure image is larger than window
    if dummy_img_size[1] < 128 || dummy_img_size[2] < 128
        dummy_img_size = (max(128, dummy_img_size[1]), max(128, dummy_img_size[2]))
    end
    imgA = rand(Float32, dummy_img_size...)
    imgB = rand(Float32, dummy_img_size...)
    img_height, img_width = size(imgA)
    println("Images loaded (dummy data): $(img_width)x$(img_height)")

    # 2. Determine Grid
    println("Step 2: Determining interrogation grid...")
    step_x = params.window_size_x - params.overlap_x
    step_y = params.window_size_y - params.overlap_y

    grid_centers_x = (params.window_size_x/2):step_x:(img_width - params.window_size_x/2 + 1 - (params.window_size_x % 2 == 0 ? 1 : 0))
    grid_centers_y = (params.window_size_y/2):step_y:(img_height - params.window_size_y/2 + 1 - (params.window_size_y % 2 == 0 ? 1 : 0))

    num_grid_x = length(grid_centers_x)
    num_grid_y = length(grid_centers_y)

    if num_grid_x == 0 || num_grid_y == 0
        error("Calculated PIV grid is empty. Check window size, overlap, and image dimensions. Image: ($img_width x $img_height), Window: ($(params.window_size_x) x $(params.window_size_y)), Step: ($step_x x $step_y)")
    end
    println("Grid determined: $(num_grid_x)x$(num_grid_y) interrogation windows.")

    # 3. Initialize PIV Data Storage
    U_field = zeros(Float64, num_grid_y, num_grid_x)
    V_field = zeros(Float64, num_grid_y, num_grid_x)
    PeakRatio_map = zeros(Float64, num_grid_y, num_grid_x)
    Moment_map = zeros(Float64, num_grid_y, num_grid_x)
    # CrossCorrelator object for 'correlate' method - only if used
    # cc = params.correlation_method == :cross_correlation ? CrossCorrelator((params.window_size_y, params.window_size_x)) : nothing

    println("Step 4: Iterating over grid and performing correlation...")
    for r_idx in 1:num_grid_y
        for c_idx in 1:num_grid_x
            center_x = grid_centers_x[c_idx]
            center_y = grid_centers_y[r_idx]

            # Define window boundaries (integer indices)
            # Ensure coordinates are 1-indexed for array access
            r_start = round(Int, center_y - params.window_size_y/2 + 0.5)
            r_end   = r_start + params.window_size_y - 1
            c_start = round(Int, center_x - params.window_size_x/2 + 0.5)
            c_end   = c_start + params.window_size_x - 1

            # Boundary checks for sub-images
            if r_start < 1 || r_end > img_height || c_start < 1 || c_end > img_width
                # This should not happen if grid calculation is correct and windows fit
                println("Warning: Window for ($center_x, $center_y) out of bounds. Skipping.")
                U_field[r_idx, c_idx] = NaN
                V_field[r_idx, c_idx] = NaN
                continue
            end

            subA = imgA[r_start:r_end, c_start:c_end]
            subB = imgB[r_start:r_end, c_start:c_end]

            # Perform Correlation
            # Initial displacement guess for deformable correlation
            initial_disp = (0.0, 0.0)
            dr, dc, peak_val, corr_matrix = (NaN, NaN, NaN, Matrix{Float32}(undef,0,0))

            if params.correlation_method == :phase_correlation
                if params.enable_deformation && params.deformation_iterations > 0
                    dr, dc = correlate_deformable(subA, subB, initial_disp, params.deformation_iterations, phase_correlate)
                    # Note: correlate_deformable doesn't currently return peak_val and corr_matrix from the *last* step.
                    # For now, we can't get quality metrics easily with deformation. This needs refinement.
                    # As a placeholder, run phase_correlate once more if quality needed.
                    # This is inefficient and not ideal.
                    if dr isa Real # Check if correlate_deformable returned valid displacement
                         _, _, peak_val, corr_matrix = phase_correlate(subA, subB) # Simplified for now
                    end
                else
                    dr, dc, peak_val, corr_matrix = phase_correlate(subA, subB)
                end
            elseif params.correlation_method == :cross_correlation
                # cc_instance = CrossCorrelator{Float32}((params.window_size_y, params.window_size_x))
                # dr, dc, peak_val, corr_matrix = correlate(cc_instance, subA, subB)
                # Using CrossCorrelator per window is inefficient. A single one should be passed or created.
                # For now, this path is a placeholder or simplified.
                println("Note: True cross-correlation with CrossCorrelator object per window is inefficient and simplified here.")
                # Simplified fallback for demonstration if :cross_correlation is chosen
                dr, dc, peak_val, corr_matrix = phase_correlate(subA, subB) # Fallback to phase_correlate
                # In a real scenario, you'd manage the CrossCorrelator object.
            else
                error("Unsupported correlation method: $(params.correlation_method)")
            end

            # Subpixel refinement with Gauss2D if selected (overrides default Gauss3 from phase_correlate)
            # This requires getting the integer peak from corr_matrix first.
            if params.subpixel_method == :gauss2d && !isempty(corr_matrix) && peak_val isa Real && !isnan(peak_val)
                # Find integer peak from corr_matrix (argmax)
                peak_loc_int = argmax(corr_matrix) # CartesianIndex
                # Convert CartesianIndex to tuple for subpixel_gauss2d
                dr_refined, dc_refined = Postprocessing.subpixel_gauss2d(corr_matrix, (peak_loc_int[1], peak_loc_int[2]))

                # The displacement is center - refined_peak. Original correlators return this directly.
                # Here, subpixel_gauss2d returns refined_peak_location.
                # We need to adjust dr, dc based on this.
                # This assumes corr_matrix is fftshifted. Center of corr_matrix is size(corr_matrix).÷2 .+ 1
                matrix_center = size(corr_matrix) .÷ 2 .+ 1
                # dr, dc were already subpixel from gauss3. We are replacing them.
                # The displacement is how much subB is shifted relative to subA.
                # If peak is at matrix_center, displacement is 0.
                # If peak is at matrix_center + (delta_r, delta_c), then displacement is -(delta_r, delta_c)
                # However, our correlators return (center - refined_peakloc)
                # subpixel_gauss2d returns (refined_row, refined_col) in matrix coords.
                # So, new displacement is (matrix_center[1] - dr_refined, matrix_center[2] - dc_refined)
                # BUT our phase_correlate and correlate already provide (center - refined_peakloc)
                # So dr = center[1] - refined_peak_row, dc = center[2] - refined_peak_col
                # If we use subpixel_gauss2d, it gives (refined_peak_row, refined_peak_col)
                # So new dr = matrix_center[1] - dr_refined ; new dc = matrix_center[2] - dc_refined
                # This is tricky because dr, dc from phase_correlate are already subpixel.
                # A cleaner way would be for correlators to optionally return integer peak + matrix
                # For now, let's assume this re-calculation of displacement is what we want if :gauss2d is picked.
                # This is a simplification and might not be perfectly accurate without refactoring correlators.
                dr = matrix_center[1] - dr_refined
                dc = matrix_center[2] - dc_refined
            end


            U_field[r_idx, c_idx] = dc # Standard PIV: U is displacement in X (columns)
            V_field[r_idx, c_idx] = dr # Standard PIV: V is displacement in Y (rows)

            # Calculate Quality Metrics (if corr_matrix is available)
            if !isempty(corr_matrix) && peak_val isa Real && !isnan(peak_val)
                peak_loc_int = argmax(corr_matrix) # CartesianIndex
                PeakRatio_map[r_idx, c_idx] = Postprocessing.calculate_peak_ratio(corr_matrix, (peak_loc_int[1], peak_loc_int[2]))

                # For correlation_moment, we need subpixel peak location.
                # dr, dc are displacements. peak_subpixel = center - displacement
                matrix_center = size(corr_matrix) .÷ 2 .+ 1
                peak_subpixel_loc_for_moment = (matrix_center[1] - dr, matrix_center[2] - dc)
                Moment_map[r_idx, c_idx] = Postprocessing.calculate_correlation_moment(corr_matrix, peak_subpixel_loc_for_moment)
            else
                PeakRatio_map[r_idx, c_idx] = NaN
                Moment_map[r_idx, c_idx] = NaN
            end
        end
    end
    println("Correlation complete.")

    # 5. Outlier Detection
    outlier_mask = falses(num_grid_y, num_grid_x) # Initialize before conditional check
    if params.postprocess_enable_uod
        println("Step 5: Performing Universal Outlier Detection...")
        outlier_mask = Postprocessing.universal_outlier_detection(
            U_field, V_field,
            params.uod_threshold,
            params.uod_neighborhood_size
        )
        # Placeholder for outlier replacement
        num_outliers = sum(outlier_mask)
        println("UOD complete. Found $num_outliers outliers.")
        # Example: simple replacement with NaN or a global median (not implemented here)
        # U_field[outlier_mask] .= NaN
        # V_field[outlier_mask] .= NaN
    end

    # 6. Prepare Results
    println("Step 6: Preparing results...")
    results = Dict(
        "image_path1" => image_path1,
        "image_path2" => image_path2,
        "parameters" => params, # copy? deepcopy? For now, just reference
        "grid_x_centers" => collect(grid_centers_x), # Ensure they are vectors
        "grid_y_centers" => collect(grid_centers_y),
        "U" => U_field,
        "V" => V_field,
        "peak_ratio" => PeakRatio_map,
        "correlation_moment" => Moment_map,
        "outlier_mask" => outlier_mask
    )
    println("Results prepared.")

    # 7. Save Results (Placeholder)
    if output_path !== nothing
        println("Step 7: Saving results to '$output_path'...")
        try
            h5open(output_path, "w") do file
                # Group for general attributes or scalar results
                attrs_group = g_create(file, "attributes")
                attrs_group["image_path1"] = results["image_path1"]
                attrs_group["image_path2"] = results["image_path2"]

                # Save PIVParameters
                params_group = g_create(file, "parameters")
                piv_params_instance = results["parameters"]
                for name in fieldnames(typeof(piv_params_instance))
                    value = getfield(piv_params_instance, name)
                    # Convert symbols to strings for HDF5 compatibility if necessary
                    if value isa Symbol
                        params_group[string(name)] = string(value)
                    else
                        params_group[string(name)] = value
                    end
                end

                grid_group = g_create(file, "grid")
                grid_group["x_centers"] = results["grid_x_centers"]
                grid_group["y_centers"] = results["grid_y_centers"]

                fields_group = g_create(file, "displacement_fields")
                fields_group["U"] = results["U"]
                fields_group["V"] = results["V"]

                quality_group = g_create(file, "quality_maps")
                quality_group["peak_ratio"] = results["peak_ratio"]
                quality_group["correlation_moment"] = results["correlation_moment"]

                if haskey(results, "outlier_mask") && results["outlier_mask"] isa AbstractArray
                    quality_group["outlier_mask"] = results["outlier_mask"]
                end
                println("Results successfully saved to '$output_path'.")
            end
        catch e
            println(stderr, "Error saving results to HDF5 file '$output_path': $e")
            # Optionally, rethrow(e) or handle more gracefully
        end
    end

    println("PIV processing finished successfully.")
    return results
end

"""
    batch_process_piv(image_pairs::Vector{Tuple{String, String}},
                      params::PIVParameters,
                      output_directory::String)

Processes multiple pairs of images using the same PIVParameters and saves each result
to a separate HDF5 file in the specified output directory.

- `image_pairs`: A vector of tuples, where each tuple contains paths to `(image_path1, image_path2)`.
- `params`: The `PIVParameters` object to be used for processing all pairs.
- `output_directory`: The directory where HDF5 result files will be saved.
"""
function batch_process_piv(
    image_pairs::Vector{Tuple{String, String}},
    params::PIVParameters,
    output_directory::String
)
    println("Starting batch PIV processing...")
    println("Total pairs to process: ", length(image_pairs))
    println("Output directory: ", output_directory)

    # 1. Check if output_directory exists, try to create it if not
    if !isdir(output_directory)
        println("Output directory '$output_directory' does not exist. Attempting to create it...")
        try
            mkpath(output_directory)
            println("Successfully created output directory '$output_directory'.")
        catch e
            error("Failed to create output directory '$output_directory': $e")
        end
    end

    total_pairs = length(image_pairs)
    for (idx, (img1_path, img2_path)) in enumerate(image_pairs)
        println("\nProcessing pair $idx/$total_pairs: '$img1_path' and '$img2_path'...")

        # a.i. Construct an output_filename
        img1_basename = first(splitext(basename(img1_path))) # Remove extension
        img2_basename = first(splitext(basename(img2_path)))
        # Sanitize basenames if they might contain problematic characters for filenames
        # For simplicity, this is omitted here but important in practice.

        # Ensure filename uniqueness if basenames are identical or very long
        # Using a prefixed index for guaranteed uniqueness and ordering.
        output_filename_base = "result_$(img1_basename)_vs_$(img2_basename)"
        # Truncate if too long, append index for safety
        max_len = 200 # Max base filename length before adding index and extension
        if length(output_filename_base) > max_len
            output_filename_base = output_filename_base[1:max_len]
        end
        output_filename = joinpath(output_directory, "$(output_filename_base)_pair$(lpad(idx, 3, '0')).h5")

        # a.iii. Call process_piv_pair
        try
            results = process_piv_pair(img1_path, img2_path, params, output_path=output_filename)
            if results !== nothing # process_piv_pair returns results dict
                println("Successfully processed and saved results for pair $idx to '$output_filename'.")
            else
                # This case should ideally not happen if process_piv_pair always returns dict or errors.
                println("Warning: processing_piv_pair returned 'nothing' for pair $idx.")
            end
        catch e
            println(stderr, "ERROR processing pair $idx ('$img1_path', '$img2_path'): $e")
            # Optionally, log error to a file, store failed pairs, etc.
            # For now, continue to the next pair.
            println(stderr, "Skipping to next pair.")
        end
    end

    println("\nBatch PIV processing complete.")
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

    # Return displacement as (x, y) = (row_offset, col_offset), max value, and correlation matrix
    return (disp[1], disp[2], maxval, c.C2)
end

function phase_correlate(subimgA::AbstractArray{T}, subimgB::AbstractArray{T}) where T
    # Get image size
    image_size = size(subimgA)

    # Allocate memory for FFTs
    C1 = Matrix{Complex{T}}(undef, image_size...)
    C2 = Matrix{Complex{T}}(undef, image_size...)
    C1 .= subimgA
    C2 .= subimgB

    # Create FFTW plans
    fp = FFTW.plan_fft!(C1)
    ip = inv(fp)

    # Perform inplace FFT on both sub-images
    mul!(C1, fp, C1)
    mul!(C2, fp, C2)

    # Compute the cross-power spectrum
    CPS = similar(C1) # Ensure CPS is of a similar type and size
    for i in eachindex(C1)
        CPS[i] = (C1[i] * conj(C2[i])) / (abs(C1[i] * conj(C2[i])) + eps(real(T))) # Add epsilon to avoid division by zero
    end

    # Inverse FFT of CPS
    ldiv!(CPS, ip, CPS)
    correlation_matrix = Matrix{real(T)}(undef, image_size...) # Ensure correct type for abs value
    fftshift!(correlation_matrix, abs.(CPS)) # Shift zero-lag to center and take absolute value


    # Find the peak in the correlation result
    peakloc = CartesianIndex(0, 0)
    maxval = zero(real(T))
    for i in CartesianIndices(correlation_matrix)
        val = correlation_matrix[i]
        if val > maxval
            maxval = val
            peakloc = i
        end
    end

    # Perform subpixel refinement and compute displacement relative to center
    center = size(correlation_matrix) .÷ 2 .+ 1
    refined_peakloc = subpixel_gauss3(correlation_matrix, peakloc.I)
    disp = refined_peakloc .- center # Corrected order for displacement calculation

    # Return displacement as (row_offset, col_offset), max value, and correlation matrix
    return (disp[1], disp[2], maxval, correlation_matrix)
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

# Affine Transformation
struct AffineTransform{T<:Real}
    A::Matrix{T}
    b::Vector{T}
end

# Default constructor for identity transform
AffineTransform() = AffineTransform(Matrix{Float32}(I, 2, 2), zeros(Float32, 2)) # Type T will be inferred

function warp_image(image::AbstractMatrix{T}, transform::AffineTransform{S}) where {T,S}
    # Create an interpolator for the image
    itp = interpolate(image, BSpline(Linear()), OnGrid())

    # Get image dimensions
    rows, cols = size(image)
    warped_image = similar(image)

    # Apply the inverse transform to the coordinates of the warped image
    inv_A = inv(transform.A)
    for r_out in 1:rows
        for c_out in 1:cols
            # Target coordinate in the output image
            x_out = [S(c_out), S(r_out)] # Represent as [col, row]

            # Corresponding coordinate in the original image using inverse transform: x_in = A_inv * (x_out - b)
            x_in = inv_A * (x_out - transform.b)

            # Bilinear interpolation
            # Ensure coordinates are within bounds for interpolation
            if 1 <= x_in[2] <= rows && 1 <= x_in[1] <= cols
                warped_image[r_out, c_out] = itp(x_in[2], x_in[1]) # itp expects (row, col)
            else
                warped_image[r_out, c_out] = zero(T) # Or some other boundary condition
            end
        end
    end
    return warped_image
end

function correlate_deformable(
    subA::AbstractMatrix{T},
    subB::AbstractMatrix{T},
    initial_displacement::NTuple{2, S},
    iterations::Int = 5,
    correlator_func::Function = phase_correlate
) where {T, S<:Real}

    current_displacement = S[initial_displacement[1], initial_displacement[2]]
    image_size = size(subA)

    # Pre-allocate for warped_subB if possible, though warp_image returns a new allocation
    # warped_subB = similar(subB)

    for iter in 1:iterations
        # 1. Create AffineTransform for current displacement (translation only)
        # The transform moves points in subB's coordinate system.
        # If (dx, dy) is the displacement of B relative to A, a point (x,y) in A
        # corresponds to (x-dx, y-dy) in B.
        # So, to warp subB to align with subA, we want to map (x_B, y_B) to (x_B+dx, y_B+dy)
        # However, warp_image applies x_new = A*x_old + b.
        # We are transforming subB. A point (x,y) in the *warped* subB should correspond to
        # a point (x',y') in the *original* subB.
        # The displacement we have is (current_row_offset, current_col_offset)
        # This means subA[r, c] corresponds to subB[r - current_row_offset, c - current_col_offset]
        # So, the transform for warp_image should map coordinates from the target grid (aligned with subA)
        # to the source grid (subB).
        # x_source = x_target - displacement
        # The 'b' vector in AffineTransform is added, so it should be the negative of our displacement.

        # Displacement is (row_offset, col_offset)
        # AffineTransform takes b as [x_translation, y_translation] which is [col_offset, row_offset]
        # current_displacement is [dr, dc]
        # We need transform.b = [dc, dr] for warp_image to sample subB at (rout-dr, cout-dc)
        transform = AffineTransform(Matrix{S}(I, 2, 2), S[current_displacement[2], current_displacement[1]])

        # 2. Warp subB
        warped_subB = warp_image(subB, transform)

        # 3. Correlate subA with warped_subB
        # The correlation result is the *additional* displacement required.
        # (delta_row, delta_col), peak_value, correlation_matrix_from_step
        delta_dr, delta_dc, _peak_val, _corr_matrix = correlator_func(subA, warped_subB)

        # 4. Update displacement estimate
        # Displacement from correlator_func is (row_offset, col_offset)
        current_displacement[1] += S(delta_dr)
        current_displacement[2] += S(delta_dc)

        # Convergence check (optional, for now fixed iterations)
        if abs(S(delta_dr)) < 1e-3 && abs(S(delta_dc)) < 1e-3
            break
        end
    end

    return (current_displacement[1], current_displacement[2])
end

# 2D Gaussian model for subpixel refinement
# G(x, y) = amplitude * exp(-((x-xo)^2/(2*sigma_x^2) + (y-yo)^2/(2*sigma_y^2))) + offset
@. gauss2d_model(xy, p) = p[1] * exp(-((xy[:,1]-p[2])^2/(2*p[3]^2) + (xy[:,2]-p[4])^2/(2*p[5]^2))) + p[6]
# p = [amplitude, xo, sigma_x, yo, sigma_y, offset] -> should be [amplitude, xo, sigma_x, yo, sigma_y, offset]
# LsqFit expects model (x, p) where x is a vector of x-coordinates.
# Here, xy is a matrix where each row is an (x,y) pair.
# So, xy[:,1] is x-coordinates, xy[:,2] is y-coordinates.
# p = [amplitude, x0, sigma_x, y0, sigma_y, offset]

function subpixel_gauss2d(correlation_matrix::AbstractMatrix{T}, peak_coords::Tuple{Int, Int}) where T
    nrows, ncols = size(correlation_matrix)
    pr, pc = peak_coords # peak row, peak col

    # Define neighborhood (e.g., 3x3)
    half_size = 1
    r_start = max(1, pr - half_size)
    r_end = min(nrows, pr + half_size)
    c_start = max(1, pc - half_size)
    c_end = min(ncols, pc + half_size)

    neighborhood = correlation_matrix[r_start:r_end, c_start:c_end]

    # If neighborhood is too small (e.g. peak at the very edge, resulting in 1xN or Nx1 or 1x1 neighborhood)
    # then least squares fitting might fail or be unreliable.
    # Fallback to integer peak or a simpler method like subpixel_gauss3 for such cases.
    if size(neighborhood,1) < 3 || size(neighborhood,2) < 3
        # Not enough points for a meaningful 2D Gaussian fit with 6 parameters.
        # Alternatively, could use a simpler model or return integer peak + 0.5 if it's exactly on edge.
        # For now, let's call subpixel_gauss3 which is already robust for edges.
        # subpixel_gauss3 expects (matrix, (peak_row_idx, peak_col_idx))
        # The peak_coords for subpixel_gauss3 should be relative to the input matrix.
        return subpixel_gauss3(correlation_matrix, peak_coords)
    end

    # Create coordinate grid for the neighborhood
    # These coordinates are relative to the top-left of the neighborhood
    y_coords = [r - r_start + 1 for r in r_start:r_end for _ in c_start:c_end]
    x_coords = [c - c_start + 1 for _ in r_start:r_end for c in c_start:c_end]
    xy_data = hcat(x_coords, y_coords) # N_points x 2 matrix, with (x,y) / (col,row)

    z_data = vec(neighborhood)

    # Initial parameter guesses
    # p = [amplitude, xo, sigma_x, yo, sigma_y, offset]
    amplitude_guess = T(maximum(neighborhood) - minimum(neighborhood))
    if amplitude_guess <= 0 # Handle flat neighborhood
        amplitude_guess = T(1.0)
    end
    # xo, yo are relative to the neighborhood grid (1-indexed)
    xo_guess = T(pc - c_start + 1)
    yo_guess = T(pr - r_start + 1)
    sigma_guess = T(1.0) # Default sigma
    offset_guess = T(minimum(neighborhood))

    p0 = [amplitude_guess, xo_guess, sigma_guess, yo_guess, sigma_guess, offset_guess]

    # Bounds for parameters (optional but good practice)
    # Lower bounds: [0, 0.5, 0.1, 0.5, 0.1, -Inf] (xo,yo relative to neighborhood, so 0.5 to size-0.5)
    # Upper bounds: [Inf, size_c+0.5, size_c, size_r+0.5, size_r, Inf]
    lower_bounds = [zero(T), T(0.5), T(0.1), T(0.5), T(0.1), T(-Inf)]
    upper_bounds = [T(Inf), T(size(neighborhood,2)+0.5), T(size(neighborhood,2)), T(size(neighborhood,1)+0.5), T(size(neighborhood,1)), T(Inf)]


    try
        fit = curve_fit(gauss2d_model, xy_data, z_data, p0, lower=lower_bounds, upper=upper_bounds)
        p_fit = fit.param

        # Fitted xo, yo are relative to the neighborhood's top-left corner (1-indexed)
        fit_xo_rel = p_fit[2] # This is relative column
        fit_yo_rel = p_fit[4] # This is relative row

        # Convert to absolute coordinates in the original correlation matrix
        # Absolute col = start_col_of_neighborhood + fit_col_relative - 1 (because fit is 1-indexed)
        # Absolute row = start_row_of_neighborhood + fit_row_relative - 1
        refined_pc = c_start + fit_xo_rel - 1
        refined_pr = r_start + fit_yo_rel - 1

        return (refined_pr, refined_pc)
    catch e
        # If fitting fails, fallback to a simpler method or integer peak
        # println("2D Gaussian fit failed: $e. Falling back to subpixel_gauss3.")
        return subpixel_gauss3(correlation_matrix, peak_coords)
    end
end

end # module Hammerhead
