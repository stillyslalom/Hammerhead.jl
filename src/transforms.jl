"""
    AffineTransform(A::AbstractMatrix, b::AbstractVector)
    AffineTransform()  # identity

Map an image point `[x, y]` to `A * [x, y] + b`, with `x` along columns and
`y` along rows. Supply a 2×2 matrix `A` and a two-element offset `b`; the
zero-argument constructor is the identity transform.
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

Warp `image` into the coordinates specified by `tform`. Each output pixel
samples the input at its inverse-transformed location using bilinear
interpolation; locations outside the input are filled with zero. Returns a
new matrix with the same dimensions as `image`.
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

Fit an affine transform from image points to corresponding reference
points, both supplied as vectors of `(x, y)` tuples or two-element vectors.
At least three non-collinear pairs are required. The returned transform
maps `x_image` to `A * x_image + b` in reference coordinates.
Coordinates must be finite real numbers representable in Float64. Both
point sets must have numerical affine rank three, and the fitted map must
be finite and invertible. Rank checks center and scale each coordinate axis;
they do not impose a fit-residual acceptance threshold. The least-squares
fit itself retains the original Float64 coordinate convention.
"""
function calculate_manual_registration(points_image::AbstractVector, points_reference::AbstractVector)
    N = length(points_image)
    N == length(points_reference) ||
        throw(ArgumentError("point counts differ: $N image vs $(length(points_reference)) reference"))
    N >= 3 || throw(ArgumentError("at least 3 point pairs are required, got $N"))

    image = [_registration_point(p, "image", i) for (i, p) in enumerate(points_image)]
    reference = [_registration_point(p, "reference", i) for (i, p) in enumerate(points_reference)]
    _registration_full_rank(image) || throw(ArgumentError("image points must have affine rank three (non-collinear)"))
    _registration_full_rank(reference) || throw(ArgumentError("reference points must have affine rank three (non-collinear)"))

    # Each pair contributes two rows to M * [a11, a12, b1, a21, a22, b2] = R.
    M = zeros(Float64, 2N, 6)
    R = zeros(Float64, 2N)
    for i in 1:N
        xi, yi = image[i]
        xr, yr = reference[i]
        M[2i-1, :] .= (xi, yi, 1.0, 0.0, 0.0, 0.0)
        M[2i, :] .= (0.0, 0.0, 0.0, xi, yi, 1.0)
        R[2i-1] = xr
        R[2i] = yr
    end
    p = M \ R
    all(isfinite, p) || throw(ArgumentError("registration fit produced nonfinite coefficients"))
    # Exact determinant of the applied Float64 coefficients avoids determinant
    # underflow for a valid small or anisotropic coordinate conversion.
    q = Rational{BigInt}.(p)
    q[1] * q[5] != q[2] * q[4] || throw(ArgumentError("registration fit is singular"))
    return AffineTransform([p[1] p[2]; p[4] p[5]], [p[3], p[6]])
end

function _registration_point(point, role, index)
    (point isa Tuple || point isa AbstractVector) && length(point) == 2 ||
        throw(ArgumentError("$role point $index must contain exactly two coordinates"))
    converted = map(point) do value
        value isa Real && !(value isa Bool) && isfinite(value) ||
            throw(ArgumentError("$role point $index coordinates must be finite real numbers"))
        number = try
            Float64(value)
        catch
            throw(ArgumentError("$role point $index coordinates must be representable in Float64"))
        end
        isfinite(number) && !(value != 0 && number == 0) ||
            throw(ArgumentError("$role point $index coordinates must be representable in Float64"))
        number
    end
    x, y = converted
    return (x, y)
end

function _registration_full_rank(points)
    design = ones(Float64, length(points), 3)
    for axis in 1:2
        values = [p[axis] for p in points]
        magnitude = maximum(abs, values)
        magnitude == 0 && return false
        scaled = values ./ magnitude
        centered = scaled .- scaled[1]
        spread = maximum(abs, centered)
        spread == 0 && return false
        design[:, axis] .= centered ./ spread
    end
    return rank(design) == 3
end

"""
    transform_vector_field(x, y, u, v, tform::AffineTransform)

Apply `tform` to each grid-point location and displacement vector. The
offset affects positions (`x' = A*x + b`); only `A` affects vectors
(`u' = A*u`). The four input arrays must share a shape. Returns
`(new_x, new_y, new_u, new_v)` with that shape.
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
