module Preprocessing

using LinearAlgebra
using ..Hammerhead: AffineTransform # Access AffineTransform from the parent module

export calculate_manual_registration, transform_vector_field

"""
    calculate_manual_registration(points_image, points_reference) -> AffineTransform

Calculates an affine transformation that maps points from the image coordinate system
to the reference coordinate system.

`points_image` and `points_reference` are vectors of corresponding 2D points.
Each point can be a Tuple `(x, y)` or a Vector `[x, y]`.
Requires at least 3 point pairs for a unique solution. If more are provided,
a least-squares fit is performed.

The transformation is `x_ref = A * x_img + b`.
"""
function calculate_manual_registration(points_image::Vector{PT1}, points_reference::Vector{PT2}) where {T1, T2, PT1<:Union{Tuple{T1,T1}, AbstractVector{T1}}, PT2<:Union{Tuple{T2,T2}, AbstractVector{T2}}}
    N = length(points_image)
    if N != length(points_reference)
        error("Number of image points and reference points must be the same.")
    end
    if N < 3
        error("At least 3 point pairs are required to determine an affine transformation.")
    end

    # Construct the matrix M for the linear system M*p = R
    # Each point pair (xi, yi) -> (xi', yi') contributes two rows:
    # [xi, yi, 1,  0,  0, 0] [a11] = [xi']
    # [ 0,  0, 0, xi, yi, 1] [a12] = [yi']
    #                            [b1 ]
    #                            [a21]
    #                            [a22]
    #                            [b2 ]
    # The parameters vector p will be [a11, a12, b1, a21, a22, b2]'

    M = zeros(Float64, 2*N, 6)
    R = zeros(Float64, 2*N)

    for i in 1:N
        xi, yi = NTuple{2}(points_image[i])
        xi_p, yi_p = NTuple{2}(points_reference[i])

        M[2*i-1, :] = [xi, yi, 1,  0,  0, 0]
        M[2*i,   :] = [ 0,  0, 0, xi, yi, 1]

        R[2*i-1] = xi_p
        R[2*i]   = yi_p
    end

    # Solve for p using least squares (M'M)p = M'R, or simply M\R in Julia
    p = M \ R

    # Extract parameters and form the AffineTransform
    # p = [a11, a12, b1, a21, a22, b2]
    A = Matrix{Float64}([p[1] p[2]; p[4] p[5]])
    b = Vector{Float64}([p[3]; p[6]])

    # Note: AffineTransform{T} expects A::Matrix{T}, b::Vector{T}
    # The AffineTransform struct in Hammerhead.jl is defined as:
    # struct AffineTransform{T<:Real}
    # A::Matrix{T}
    # b::Vector{T}
    # end
    # So, this should be fine. We are returning AffineTransform{Float64}.
    return AffineTransform(A, b)
end


"""
    transform_vector_field(x_coords, y_coords, u_vectors, v_vectors, transform::AffineTransform)

Transforms the grid point locations and the vector components using a given AffineTransform.

- `x_coords`, `y_coords`: Matrices or vectors of original grid point locations.
- `u_vectors`, `v_vectors`: Matrices or vectors of velocity components at those grid points.
- `transform`: The `AffineTransform` object (x' = A*x + b).

Returns a tuple: `(new_x_coords, new_y_coords, new_u_vectors, new_v_vectors)`.
The shape of the input arrays (vector or matrix) is preserved.
"""
function transform_vector_field(
    x_coords::AbstractArray{T},
    y_coords::AbstractArray{T},
    u_vectors::AbstractArray{T},
    v_vectors::AbstractArray{T},
    transform::AffineTransform{S}
) where {T<:Real, S<:Real}

    if !(size(x_coords) == size(y_coords) == size(u_vectors) == size(v_vectors))
        error("All input arrays (coordinates and vectors) must have the same dimensions.")
    end

    new_x_coords = similar(x_coords, S) # Output type based on AffineTransform's type
    new_y_coords = similar(y_coords, S)
    new_u_vectors = similar(u_vectors, S)
    new_v_vectors = similar(v_vectors, S)

    A = transform.A
    b = transform.b

    for i in eachindex(x_coords)
        # Original point and vector
        orig_x = x_coords[i]
        orig_y = y_coords[i]
        orig_u = u_vectors[i]
        orig_v = v_vectors[i]

        # Transform point: new_coord = A * old_coord + b
        # A is 2x2, b is 2x1. Coords are [x; y]
        new_x = A[1,1]*orig_x + A[1,2]*orig_y + b[1]
        new_y = A[2,1]*orig_x + A[2,2]*orig_y + b[2]

        new_x_coords[i] = new_x
        new_y_coords[i] = new_y

        # Transform vector: new_vec = A * old_vec
        # A is 2x2. Vectors are [u; v]
        new_u = A[1,1]*orig_u + A[1,2]*orig_v
        new_v = A[2,1]*orig_u + A[2,2]*orig_v

        new_u_vectors[i] = new_u
        new_v_vectors[i] = new_v
    end

    return new_x_coords, new_y_coords, new_u_vectors, new_v_vectors
end

end # module Preprocessing
