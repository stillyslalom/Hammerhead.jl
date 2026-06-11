"""
    AffineTransform(A::AbstractMatrix, b::AbstractVector)
    AffineTransform()  # identity

2D affine transform `x' = A*x + b` acting on `[x, y]` coordinates (`x` along
columns, `y` along rows). `A` must be 2×2 and `b` of length 2.
"""
struct AffineTransform{T<:Real}
    A::Matrix{T}
    b::Vector{T}

    function AffineTransform{T}(A::AbstractMatrix, b::AbstractVector) where {T<:Real}
        size(A) == (2, 2) || throw(ArgumentError("A must be 2×2, got $(size(A))"))
        length(b) == 2 || throw(ArgumentError("b must have length 2, got $(length(b))"))
        new{T}(Matrix{T}(A), Vector{T}(b))
    end
end

AffineTransform(A::AbstractMatrix{TA}, b::AbstractVector{TB}) where {TA<:Real,TB<:Real} =
    AffineTransform{promote_type(TA, TB)}(A, b)
AffineTransform() = AffineTransform{Float64}(Matrix{Float64}(I, 2, 2), zeros(2))

Base.show(io::IO, t::AffineTransform{T}) where {T} =
    print(io, "AffineTransform{$T}(A=$(t.A), b=$(t.b))")

"""
    warp_image(image, tform::AffineTransform) -> Matrix

Apply `tform` to `image`: the output at `(row, col)` samples the input at the
inverse-transformed coordinate, using bilinear interpolation. Coordinates
outside the input are filled with zero.
"""
function warp_image(image::AbstractMatrix{T}, tform::AffineTransform{S}) where {T,S<:Real}
    itp = extrapolate(interpolate(image, BSpline(Linear())), zero(T))
    invA = inv(tform.A)
    bx, by = tform.b[1], tform.b[2]
    out = similar(image, T, size(image))
    @inbounds for c in axes(image, 2), r in axes(image, 1)
        # Source coordinate: A⁻¹ * ([c, r] - b), with x = col, y = row.
        x = invA[1, 1] * (c - bx) + invA[1, 2] * (r - by)
        y = invA[2, 1] * (c - bx) + invA[2, 2] * (r - by)
        out[r, c] = itp(y, x)
    end
    return out
end

"""
    calculate_manual_registration(points_image, points_reference) -> AffineTransform

Least-squares affine transform mapping image coordinates to reference
coordinates (`x_ref = A*x_img + b`). Both arguments are vectors of
corresponding 2D points (tuples or 2-vectors, as `(x, y)`); at least 3
non-collinear pairs are required.
"""
function calculate_manual_registration(points_image::AbstractVector, points_reference::AbstractVector)
    N = length(points_image)
    N == length(points_reference) ||
        throw(ArgumentError("point counts differ: $N image vs $(length(points_reference)) reference"))
    N >= 3 || throw(ArgumentError("at least 3 point pairs are required, got $N"))

    # Each pair contributes two rows to M * [a11, a12, b1, a21, a22, b2] = R.
    M = zeros(Float64, 2N, 6)
    R = zeros(Float64, 2N)
    for i in 1:N
        xi, yi = Float64.(NTuple{2}(points_image[i]))
        xr, yr = Float64.(NTuple{2}(points_reference[i]))
        M[2i-1, :] .= (xi, yi, 1.0, 0.0, 0.0, 0.0)
        M[2i, :] .= (0.0, 0.0, 0.0, xi, yi, 1.0)
        R[2i-1] = xr
        R[2i] = yr
    end
    p = M \ R
    return AffineTransform([p[1] p[2]; p[4] p[5]], [p[3], p[6]])
end

"""
    transform_vector_field(x, y, u, v, tform::AffineTransform)

Transform grid-point locations (`x' = A*x + b`) and vector components
(`u' = A*u`) by `tform`. All four arrays must share the same shape, which is
preserved. Returns `(new_x, new_y, new_u, new_v)`.
"""
function transform_vector_field(x::AbstractArray{<:Real}, y::AbstractArray{<:Real},
                                u::AbstractArray{<:Real}, v::AbstractArray{<:Real},
                                tform::AffineTransform{S}) where {S<:Real}
    size(x) == size(y) == size(u) == size(v) ||
        throw(ArgumentError("all coordinate and vector arrays must have the same dimensions"))
    A, b = tform.A, tform.b
    new_x = similar(x, S)
    new_y = similar(y, S)
    new_u = similar(u, S)
    new_v = similar(v, S)
    @inbounds for i in eachindex(x)
        new_x[i] = A[1, 1] * x[i] + A[1, 2] * y[i] + b[1]
        new_y[i] = A[2, 1] * x[i] + A[2, 2] * y[i] + b[2]
        new_u[i] = A[1, 1] * u[i] + A[1, 2] * v[i]
        new_v[i] = A[2, 1] * u[i] + A[2, 2] * v[i]
    end
    return new_x, new_y, new_u, new_v
end
