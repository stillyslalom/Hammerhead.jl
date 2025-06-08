module Postprocessing

using Statistics # For median calculation

export calculate_peak_ratio, universal_outlier_detection, calculate_correlation_moment

"""
    calculate_peak_ratio(correlation_matrix::AbstractMatrix{T},
                         peak_location::Tuple{Int, Int},
                         window_size::Int=5) where T<:Real -> Float64

Calculates the ratio of the primary peak to the secondary peak in a correlation matrix.

- `correlation_matrix`: The 2D correlation map.
- `peak_location`: A tuple `(row, col)` of the primary peak's integer coordinates.
- `window_size`: An integer defining the size of a square window (e.g., 5 for a 5x5 window)
                 around the primary peak to exclude when searching for the secondary peak.
                 The window will have `window_size ÷ 2` pixels on each side of the peak.

Returns the ratio `P1 / P2`. If `P2` is zero or very close to zero, returns `Inf`
or a very large number if `P1` is also non-zero, or `NaN` or `0.0` if `P1` is zero.
"""
function calculate_peak_ratio(correlation_matrix::AbstractMatrix{T},
                               peak_location::Tuple{Int, Int},
                               window_size::Int=5) where T<:Real

    nrows, ncols = size(correlation_matrix)
    pr, pc = peak_location

    # 1. Find the value of the primary peak (P1)
    if !(1 <= pr <= nrows && 1 <= pc <= ncols)
        error("Peak location ($pr, $pc) is outside the matrix dimensions ($nrows, $ncols).")
    end
    P1 = correlation_matrix[pr, pc]

    # 2. Create a temporary copy of the correlation_matrix
    temp_matrix = copy(correlation_matrix)

    # 3. Set values within the window_size x window_size area around peak_location to a very low value.
    # Ensure window_size is odd for a symmetric window around the peak.
    # If window_size is 4, half_win is 2. Range pr-2:pr+2 is 5 elements.
    # If window_size is 5, half_win is 2. Range pr-2:pr+2 is 5 elements.
    # This seems correct for window_size being the total width.
    half_win = window_size ÷ 2

    r_start = max(1, pr - half_win)
    r_end = min(nrows, pr + half_win)
    c_start = max(1, pc - half_win)
    c_end = min(ncols, pc + half_win)

    # It's generally better to set to minimum possible value or a value known to be
    # lower than any potential true secondary peak, rather than modifying type (e.g. to -Inf for Real).
    # If the matrix can have negative values, setting to 0 might not be enough.
    # Using minimum of the matrix before modification could be an option, or just 0 if values are non-negative.
    # Assuming correlation values are typically non-negative (e.g. after abs() in phase_correlate).
    fill_value = zero(T)
    if eltype(temp_matrix) <: AbstractFloat # For float types, -Inf is a good choice
        fill_value = T(-Inf)
    else # For integer types, typemin(T)
        fill_value = typemin(T)
    end

    temp_matrix[r_start:r_end, c_start:c_end] .= fill_value

    # 4. Find the maximum value in the modified temporary matrix (P2)
    P2 = typemin(T) # Initialize with smallest possible value
    # Need to iterate to avoid the masked area if fill_value is not -Inf for all types.
    # However, if fill_value is indeed the smallest possible or -Inf, maximum() will work.
    P2 = maximum(temp_matrix)

    # If P2 is still the fill_value, it means no other peak was found (e.g., matrix was constant outside window)
    if P2 == fill_value
        # This could happen if the entire matrix outside the window was also 'fill_value' or less.
        # Or if the matrix is very small.
        # If P1 is positive, ratio is Inf. If P1 is zero/negative, it's more complex.
        return P1 > zero(T) ? Inf64 : (P1 == zero(T) ? NaN : -Inf64) # Or handle as error/specific case
    end

    # 5. Calculate the peak ratio
    if P2 == zero(T)
        return P1 > zero(T) ? Inf64 : (P1 == zero(T) ? NaN : -Inf64) # Or a very large float if P1 is non-zero
    end

    # Ensure floating point division
    return Float64(P1) / Float64(P2)
end

const UOD_EPSILON = 1e-6 # Small constant to prevent division by zero

"""
    universal_outlier_detection(u_field::AbstractMatrix{T},
                                v_field::AbstractMatrix{T},
                                threshold::Real,
                                neighborhood_size::Int=1) where T<:Real -> BitMatrix

Performs Universal Outlier Detection on a 2D vector field.

- `u_field`, `v_field`: 2D arrays of U and V components of the vector field.
- `threshold`: A value to determine outlier sensitivity. Higher values are less sensitive.
- `neighborhood_size`: Integer defining layers of neighbors (1 for 3x3, 2 for 5x5, etc.).

Returns a `BitMatrix` of the same dimensions as `u_field`, where `true` indicates an outlier.
"""
function universal_outlier_detection(u_field::AbstractMatrix{T},
                                     v_field::AbstractMatrix{T},
                                     threshold::Real,
                                     neighborhood_size::Int=1) where T<:Real

    if size(u_field) != size(v_field)
        error("U and V fields must have the same dimensions.")
    end
    if neighborhood_size < 0
        error("Neighborhood size must be non-negative.")
    end

    nrows, ncols = size(u_field)
    is_outlier = falses(nrows, ncols)

    # Temporary storage for neighbor values and their residuals
    # Max possible neighbors for a 3x3 (neighborhood_size=1) is 8. For 5x5 (size=2) is 24.
    # (2*neighborhood_size+1)^2 - 1
    max_neighbors = (2 * neighborhood_size + 1)^2 - 1
    if max_neighbors == 0 && neighborhood_size == 0 # Special case: 1x1 window means no neighbors
        # Or should this be an error / handled differently? UOD typically relies on neighbors.
        # If neighborhood_size is 0, the loop for neighbors won't run, medians will be based on empty lists (error).
        # Let's assume neighborhood_size >= 1 as per typical UOD. Or handle 0 explicitly.
        if neighborhood_size == 0
            # No neighbors, so can't compute medians. Or all are outliers? Or none?
            # This case is ill-defined for UOD. Let's prevent it or return all false.
            # For now, returning all false as no comparison is possible.
            # Alternatively, error("Neighborhood size must be at least 1 for UOD.")
             return is_outlier # Or error
        end
    end

    neighbor_u_values = Vector{T}(undef, max_neighbors)
    neighbor_v_values = Vector{T}(undef, max_neighbors)
    neighbor_u_residuals = Vector{T}(undef, max_neighbors)
    neighbor_v_residuals = Vector{T}(undef, max_neighbors)

    for r in 1:nrows
        for c in 1:ncols
            current_u = u_field[r,c]
            current_v = v_field[r,c]

            neighbor_count = 0
            # 3. Identify neighbors
            for nr in max(1, r-neighborhood_size):min(nrows, r+neighborhood_size)
                for nc in max(1, c-neighborhood_size):min(ncols, c+neighborhood_size)
                    if nr == r && nc == c
                        continue # Exclude central vector itself
                    end
                    neighbor_count += 1
                    neighbor_u_values[neighbor_count] = u_field[nr,nc]
                    neighbor_v_values[neighbor_count] = v_field[nr,nc]
                end
            end

            # If no neighbors (e.g. 1x1 image with neighborhood_size=1, or neighborhood_size=0 handled above)
            if neighbor_count == 0
                # This can happen if image is smaller than neighborhood, or neighborhood_size=0
                # Or if the point is in a corner of a very small array.
                # UOD is not well-defined. Mark as not an outlier or handle as error.
                continue # is_outlier[r,c] remains false
            end

            valid_neighbor_u = view(neighbor_u_values, 1:neighbor_count)
            valid_neighbor_v = view(neighbor_v_values, 1:neighbor_count)

            # 4. Calculate median of neighbors
            u_median_neighbors = median(valid_neighbor_u)
            v_median_neighbors = median(valid_neighbor_v)

            # 5. Calculate residuals for the central vector
            u_residual_central = current_u - u_median_neighbors
            v_residual_central = current_v - v_median_neighbors

            # 6. Calculate median of absolute residuals of neighbors from *their* medians (u_median_neighbors, v_median_neighbors)
            # These are residuals of neighbors relative to the median *of the neighbors*.
            for i in 1:neighbor_count
                neighbor_u_residuals[i] = abs(neighbor_u_values[i] - u_median_neighbors)
                neighbor_v_residuals[i] = abs(neighbor_v_values[i] - v_median_neighbors)
            end

            u_norm_median = median(view(neighbor_u_residuals, 1:neighbor_count))
            v_norm_median = median(view(neighbor_v_residuals, 1:neighbor_count))

            # 7. Normalize central vector's residuals
            u_normalized_residual = u_residual_central / (u_norm_median + UOD_EPSILON)
            v_normalized_residual = v_residual_central / (v_norm_median + UOD_EPSILON)

            # 8. Mark as outlier
            if abs(u_normalized_residual) > threshold || abs(v_normalized_residual) > threshold
                is_outlier[r,c] = true
            end
        end
    end

    return is_outlier
end

const CORR_MOMENT_EPSILON = 1e-9 # To avoid division by zero if sum of weights is tiny

"""
    calculate_correlation_moment(correlation_matrix::AbstractMatrix{T},
                                 peak_subpixel_location::Tuple{S, S},
                                 neighborhood_size::Int=3) where {T<:Real, S<:AbstractFloat} -> S

Calculates an uncertainty estimate based on the second moment of the correlation peak
in a specified neighborhood. This is akin to a weighted standard deviation of coordinates
around the subpixel peak, weighted by correlation values.

- `correlation_matrix`: The 2D correlation map.
- `peak_subpixel_location`: Tuple `(row_float, col_float)` of the subpixel-refined peak.
- `neighborhood_size`: Odd integer (e.g., 3 or 5) for the square neighborhood size.

Returns `sqrt(sigma_x_sq + sigma_y_sq)`. Returns `NaN` if the sum of correlation
values in the neighborhood is zero or too small.
"""
function calculate_correlation_moment(correlation_matrix::AbstractMatrix{T},
                                      peak_subpixel_location::Tuple{S, S},
                                      neighborhood_size::Int=3) where {T<:Real, S<:AbstractFloat}

    if neighborhood_size % 2 == 0 || neighborhood_size < 1
        error("Neighborhood size must be a positive odd integer.")
    end

    nrows, ncols = size(correlation_matrix)
    peak_row_float, peak_col_float = peak_subpixel_location

    # 1. Identify integer coordinates of the pixel containing/nearest to the peak_subpixel_location
    # Using round for the center of the neighborhood ensures it's centered on the closest integer pixel.
    center_r_int = round(Int, peak_row_float)
    center_c_int = round(Int, peak_col_float)

    # 2. Define neighborhood boundaries (truncating at edges)
    half_size = neighborhood_size ÷ 2

    r_start = max(1, center_r_int - half_size)
    r_end = min(nrows, center_r_int + half_size)
    c_start = max(1, center_c_int - half_size)
    c_end = min(ncols, center_c_int + half_size)

    # Check if the defined neighborhood is valid (e.g. peak is not too far out)
    if r_start > r_end || c_start > c_end
        # This could happen if center_r_int/center_c_int is outside matrix bounds,
        # or if matrix is smaller than neighborhood_size.
        return S(NaN) # Cannot compute moment for an empty or invalid neighborhood
    end

    sum_C = zero(S) # Accumulator for sum of correlation values, use type S for precision
    sum_weighted_dist_sq_x = zero(S)
    sum_weighted_dist_sq_y = zero(S)

    # 3. Iterate through the neighborhood
    for r_abs in r_start:r_end
        for c_abs in c_start:c_end
            C_rc = S(correlation_matrix[r_abs, c_abs]) # Convert to type S

            # Ensure weights are non-negative; negative correlation values can cause issues.
            # This method typically assumes positive correlation values forming a peak.
            if C_rc < zero(S)
                C_rc = zero(S)
            end

            sum_C += C_rc

            # Difference from the subpixel peak location
            dist_c = S(c_abs) - peak_col_float
            dist_r = S(r_abs) - peak_row_float

            sum_weighted_dist_sq_x += dist_c^2 * C_rc
            sum_weighted_dist_sq_y += dist_r^2 * C_rc
        end
    end

    # 4. Handle potential division by zero
    if sum_C < CORR_MOMENT_EPSILON
        return S(NaN) # Or S(Inf) depending on desired behavior for zero sum of weights
    end

    sigma_x_sq = sum_weighted_dist_sq_x / sum_C
    sigma_y_sq = sum_weighted_dist_sq_y / sum_C

    # 5. Return the combined moment
    return sqrt(sigma_x_sq + sigma_y_sq)
end

end # module Postprocessing
