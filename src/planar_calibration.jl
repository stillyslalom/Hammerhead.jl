"""
    PlanarTransform

Explicit affine transform from image `(x,y)` coordinates to planar physical
coordinates. `matrix` supports rotation, reflection, and anisotropic scales;
`offset` sets the physical origin.
"""
struct PlanarTransform{T<:AbstractFloat}
    matrix::SMatrix{2,2,T,4}
    offset::SVector{2,T}
end

PlanarTransform(matrix::AbstractMatrix{<:Real}, offset::AbstractVector{<:Real}) = begin
    size(matrix)==(2,2) || throw(DimensionMismatch("matrix must be 2x2"))
    length(offset)==2 || throw(DimensionMismatch("offset must have length 2"))
    T=float(promote_type(eltype(matrix),eltype(offset)))
    PlanarTransform{T}(SMatrix{2,2,T}(matrix), SVector{2,T}(offset))
end

"""
    planar_calibration(p1, p2, distance; origin=(0,0), reflection=false,
                       perpendicular_scale=nothing)

Build a two-point planar calibration. The physical x axis runs from image
point `p1` to `p2` and has the supplied physical `distance`; `origin` is the
physical coordinate assigned to `p1`. By default pixels are isotropic.
`perpendicular_scale` supplies a separate physical-units-per-pixel y scale,
and `reflection=true` reverses that physical y axis.
"""
function planar_calibration(p1::Tuple{<:Real,<:Real}, p2::Tuple{<:Real,<:Real},
                            distance::Real; origin::Tuple{<:Real,<:Real}=(0,0),
                            reflection::Bool=false,
                            perpendicular_scale::Union{Nothing,Real}=nothing)
    distance > 0 && isfinite(distance) || throw(ArgumentError("distance must be positive and finite"))
    dx,dy = p2[1]-p1[1],p2[2]-p1[2]; L=hypot(dx,dy)
    L > 0 || throw(ArgumentError("calibration points must be distinct"))
    sx=distance/L; sy=perpendicular_scale === nothing ? sx : perpendicular_scale
    sy > 0 && isfinite(sy) || throw(ArgumentError("perpendicular_scale must be positive and finite"))
    ex=(dx/L,dy/L); ey=(-dy/L,dx/L); reflection && (ey=(-ey[1],-ey[2]))
    A=[sx*ex[1] sx*ex[2]; sy*ey[1] sy*ey[2]]
    b=[origin[1],origin[2]] - A*[p1[1],p1[2]]
    PlanarTransform(A,b)
end

transform_point(t::PlanarTransform, p::Tuple{<:Real,<:Real}) = Tuple(t.matrix*SVector(p...) + t.offset)
transform_vector(t::PlanarTransform, v::Tuple{<:Real,<:Real}) = Tuple(t.matrix*SVector(v...))
(t::PlanarTransform)(p::Tuple{<:Real,<:Real}) = transform_point(t,p)
Base.inv(t::PlanarTransform{T}) where {T} = begin
    A=inv(t.matrix); PlanarTransform{T}(A, -(A*t.offset))
end
