# Derived quantities and extraction utilities for regular planar vector fields.

_field_valid(r::PIVResult) = .!r.mask .& .!r.outliers .& isfinite.(r.u) .& isfinite.(r.v)

function _axis_derivative(f::AbstractMatrix, axis::AbstractVector, dim::Int,
                          valid::AbstractMatrix{Bool})
    size(f) == size(valid) || throw(DimensionMismatch("field and validity mask must match"))
    n = size(f, dim)
    length(axis) == n || throw(DimensionMismatch("coordinate axis length does not match field"))
    n >= 2 || throw(ArgumentError("each differentiated grid dimension needs at least 2 points"))
    all(diff(axis) .!= 0) || throw(ArgumentError("grid coordinates must be distinct"))
    out = fill(NaN, size(f))
    nr, nc = size(f)
    @inbounds for j in 1:nc, i in 1:nr
        valid[i, j] || continue
        k = dim == 1 ? i : j
        left = k > 1 && (dim == 1 ? valid[i - 1, j] : valid[i, j - 1])
        right = k < n && (dim == 1 ? valid[i + 1, j] : valid[i, j + 1])
        if left && right
            fm = dim == 1 ? f[i - 1, j] : f[i, j - 1]
            fp = dim == 1 ? f[i + 1, j] : f[i, j + 1]
            out[i, j] = (fp - fm) / (axis[k + 1] - axis[k - 1])
        elseif right
            fp = dim == 1 ? f[i + 1, j] : f[i, j + 1]
            out[i, j] = (fp - f[i, j]) / (axis[k + 1] - axis[k])
        elseif left
            fm = dim == 1 ? f[i - 1, j] : f[i, j - 1]
            out[i, j] = (f[i, j] - fm) / (axis[k] - axis[k - 1])
        end
    end
    out
end

"""
    flow_derivatives(result; include_invalid=false)
    flow_derivatives(x, y, u, v; valid=...)

Compute the four entries of the planar velocity-gradient tensor on a regular
grid. Central differences are used where both immediate neighbours are valid,
one-sided differences at valid boundaries, and `NaN` where no valid local
stencil exists. Masked and flagged vectors are excluded by default.
"""
function flow_derivatives(x::AbstractVector, y::AbstractVector,
                          u::AbstractMatrix, v::AbstractMatrix;
                          valid::AbstractMatrix{Bool} = isfinite.(u) .& isfinite.(v))
    size(u) == size(v) == size(valid) || throw(DimensionMismatch("u, v, and valid must match"))
    length(x) == size(u, 2) && length(y) == size(u, 1) ||
        throw(DimensionMismatch("x/y axes do not match the vector field"))
    dudx = _axis_derivative(u, x, 2, valid)
    dudy = _axis_derivative(u, y, 1, valid)
    dvdx = _axis_derivative(v, x, 2, valid)
    dvdy = _axis_derivative(v, y, 1, valid)
    (; dudx, dudy, dvdx, dvdy, valid = copy(valid))
end
function flow_derivatives(r::PIVResult; include_invalid::Bool = false)
    valid = isfinite.(r.u) .& isfinite.(r.v) .& .!r.mask
    include_invalid || (valid .&= .!r.outliers)
    flow_derivatives(r.x, r.y, r.u, r.v; valid)
end

vorticity(d::NamedTuple) = d.dvdx .- d.dudy
divergence(d::NamedTuple) = d.dudx .+ d.dvdy
vorticity(args...; kwargs...) = vorticity(flow_derivatives(args...; kwargs...))
divergence(args...; kwargs...) = divergence(flow_derivatives(args...; kwargs...))

"""Return planar strain components `(xx, yy, xy, magnitude)` with `xy=(du/dy+dv/dx)/2`."""
function strain_rate(d::NamedTuple)
    xx, yy = d.dudx, d.dvdy
    xy = (d.dudy .+ d.dvdx) ./ 2
    magnitude = sqrt.(2 .* (xx.^2 .+ yy.^2 .+ 2 .* xy.^2))
    (; xx, yy, xy, magnitude)
end
strain_rate(args...; kwargs...) = strain_rate(flow_derivatives(args...; kwargs...))

"""
    swirling_strength(...)

Planar swirling strength: the magnitude of the imaginary part of the two
eigenvalues of the local 2x2 velocity-gradient tensor. It is zero where the
eigenvalues are real.
"""
function swirling_strength(d::NamedTuple)
    disc = d.dudx .* d.dvdy .- d.dudy .* d.dvdx .-
           ((d.dudx .+ d.dvdy) ./ 2).^2
    sqrt.(max.(disc, 0))
end
swirling_strength(args...; kwargs...) = swirling_strength(flow_derivatives(args...; kwargs...))

"""
    q_criterion(...)

The explicitly two-dimensional Q criterion,
`Q = (||Omega||^2 - ||S||^2)/2 = -tr(grad(u)^2)/2`. Positive values indicate
rotation dominating strain in the measured plane; no unmeasured gradients
are assumed.
"""
q_criterion(d::NamedTuple) = .-(d.dudx.^2 .+ 2 .* d.dudy .* d.dvdx .+ d.dvdy.^2) ./ 2
q_criterion(args...; kwargs...) = q_criterion(flow_derivatives(args...; kwargs...))

function _bilinear(x, y, f, qx, qy)
    (first(x) <= qx <= last(x) || last(x) <= qx <= first(x)) || return NaN
    (first(y) <= qy <= last(y) || last(y) <= qy <= first(y)) || return NaN
    ix = clamp(searchsortedlast(x, qx; rev=first(x)>last(x)), 1, length(x) - 1)
    iy = clamp(searchsortedlast(y, qy; rev=first(y)>last(y)), 1, length(y) - 1)
    x0, x1 = x[ix], x[ix + 1]; y0, y1 = y[iy], y[iy + 1]
    vals = (f[iy, ix], f[iy, ix + 1], f[iy + 1, ix], f[iy + 1, ix + 1])
    all(isfinite, vals) || return NaN
    tx = (qx - x0) / (x1 - x0); ty = (qy - y0) / (y1 - y0)
    (1-ty) * ((1-tx)*vals[1] + tx*vals[2]) + ty * ((1-tx)*vals[3] + tx*vals[4])
end

"""Sample a result along a polyline, returning arc length, coordinates, and interpolated components."""
function extract_profile(r::PIVResult, points::AbstractVector{<:Tuple}; n::Int = 100,
                         include_invalid::Bool = false)
    length(points) >= 2 || throw(ArgumentError("a profile needs at least two points"))
    n >= 2 || throw(ArgumentError("n must be at least 2"))
    seg = [hypot(points[k+1][1]-points[k][1], points[k+1][2]-points[k][2]) for k in 1:length(points)-1]
    total = sum(seg); total > 0 || throw(ArgumentError("profile length must be positive"))
    target = collect(range(0, total; length=n)); cumulative = cumsum(vcat(0.0, seg))
    qx = Float64[]; qy = Float64[]
    for s in target
        k = min(searchsortedlast(cumulative, s), length(seg)); t = (s-cumulative[k])/seg[k]
        push!(qx, (1-t)*points[k][1]+t*points[k+1][1]); push!(qy, (1-t)*points[k][2]+t*points[k+1][2])
    end
    valid = isfinite.(r.u) .& isfinite.(r.v) .& .!r.mask
    include_invalid || (valid .&= .!r.outliers)
    uf = ifelse.(valid, r.u, NaN); vf = ifelse.(valid, r.v, NaN)
    (; s=target, x=qx, y=qy, u=[_bilinear(r.x,r.y,uf,x,y) for (x,y) in zip(qx,qy)],
       v=[_bilinear(r.x,r.y,vf,x,y) for (x,y) in zip(qx,qy)])
end

"""Extract grid samples inside `(xmin,xmax,ymin,ymax)` or a polygon."""
function extract_region(r::PIVResult, region; include_invalid::Bool = false)
    inside = if region isa NTuple{4,Real}
        xmin,xmax,ymin,ymax = region
        [xmin <= x <= xmax && ymin <= y <= ymax for y in r.y, x in r.x]
    else
        verts = collect(region)
        length(verts) >= 3 || throw(ArgumentError("a polygon region needs at least 3 vertices"))
        [begin
            hit=false; k=length(verts)
            for q in eachindex(verts)
                xq,yq=verts[q]; xk,yk=verts[k]
                ((yq > y) != (yk > y)) &&
                    (x < (xk-xq)*(y-yq)/(yk-yq)+xq) && (hit = !hit)
                k=q
            end
            hit
        end for y in r.y, x in r.x]
    end
    valid = inside .& .!r.mask .& isfinite.(r.u) .& isfinite.(r.v)
    include_invalid || (valid .&= .!r.outliers)
    inds = findall(valid)
    (; indices=inds, x=[r.x[I[2]] for I in inds], y=[r.y[I[1]] for I in inds],
       u=r.u[inds], v=r.v[inds], mask=valid)
end

"""Compute circulation `integral(u dx + v dy)` around a supplied closed contour."""
function circulation(r::PIVResult, contour::AbstractVector{<:Tuple}; close::Bool = true,
                     include_invalid::Bool = false)
    pts = collect(contour)
    length(pts) >= 3 || throw(ArgumentError("a circulation contour needs at least three points"))
    close && pts[end] != pts[1] && push!(pts, pts[1])
    prof = extract_profile(r, pts; n=max(2, 20*(length(pts)-1)+1), include_invalid)
    all(isfinite, prof.u) && all(isfinite, prof.v) || return NaN
    sum((prof.u[k]+prof.u[k+1])*(prof.x[k+1]-prof.x[k])/2 +
        (prof.v[k]+prof.v[k+1])*(prof.y[k+1]-prof.y[k])/2 for k in 1:length(prof.x)-1)
end

function _clip_polygon(poly, dim::Int, bound, keep_greater::Bool)
    isempty(poly) && return poly
    inside(p) = keep_greater ? p[dim] >= bound : p[dim] <= bound
    result = Tuple{Float64,Float64}[]
    previous = poly[end]
    for current in poly
        a, b = inside(previous), inside(current)
        if a != b
            t = (bound - previous[dim]) / (current[dim] - previous[dim])
            push!(result, ((1-t)*previous[1]+t*current[1],
                           (1-t)*previous[2]+t*current[2]))
        end
        b && push!(result, current)
        previous = current
    end
    result
end

"""Area-form circulation over a rectangular or polygonal `region`, integrating vorticity over its overlap with the grid."""
function circulation(r::PIVResult; region, include_invalid::Bool=false)
    length(r.x)>=2 && length(r.y)>=2 || throw(ArgumentError("circulation needs at least a 2x2 grid"))
    poly = if region isa NTuple{4,Real}
        xmin, xmax, ymin, ymax = region
        xmin <= xmax && ymin <= ymax || throw(ArgumentError("rectangle bounds must be ordered"))
        [(Float64(xmin),Float64(ymin)), (Float64(xmax),Float64(ymin)),
         (Float64(xmax),Float64(ymax)), (Float64(xmin),Float64(ymax))]
    else
        points = [(Float64(p[1]),Float64(p[2])) for p in region]
        length(points) >= 3 || throw(ArgumentError("a polygon region needs at least 3 vertices"))
        points
    end
    signed_twice_area = sum(poly[k][1]*poly[mod1(k+1,length(poly))][2] -
                            poly[mod1(k+1,length(poly))][1]*poly[k][2]
                            for k in eachindex(poly))
    signed_twice_area != 0 || throw(ArgumentError("circulation region must have nonzero area"))
    orientation = sign(signed_twice_area)
    omega = vorticity(r; include_invalid)
    total = 0.0
    for j in 1:length(r.x)-1, i in 1:length(r.y)-1
        x0, x1 = r.x[j], r.x[j+1]
        y0, y1 = r.y[i], r.y[i+1]
        clipped = poly
        for (dim, bound, greater) in ((1,min(x0,x1),true), (1,max(x0,x1),false),
                                       (2,min(y0,y1),true), (2,max(y0,y1),false))
            clipped = _clip_polygon(clipped, dim, bound, greater)
            isempty(clipped) && break
        end
        length(clipped) >= 3 || continue
        corners = (omega[i,j],omega[i,j+1],omega[i+1,j],omega[i+1,j+1])
        all(isfinite,corners) || continue
        # Three-point triangle quadrature integrates the cell's bilinear
        # interpolant exactly, including its x*y term.
        a = clipped[1]
        for k in 2:length(clipped)-1
            b, c = clipped[k], clipped[k+1]
            area = orientation*((b[1]-a[1])*(c[2]-a[2])-
                                (c[1]-a[1])*(b[2]-a[2]))/2
            for (wa,wb,wc) in ((2/3,1/6,1/6),(1/6,2/3,1/6),(1/6,1/6,2/3))
                qx = wa*a[1]+wb*b[1]+wc*c[1]
                qy = wa*a[2]+wb*b[2]+wc*c[2]
                tx, ty = (qx-x0)/(x1-x0), (qy-y0)/(y1-y0)
                value = (1-ty)*((1-tx)*corners[1]+tx*corners[2]) +
                        ty*((1-tx)*corners[3]+tx*corners[4])
                total += area*value/3
            end
        end
    end
    total
end

"""
    result_spectrum(results, i, j; dt, component=:u, invalid=:error, window=:hann)

Spectrum of one grid point across a uniformly sampled result sequence. Supply
`dt` as the time between successive results. An attached `PhysicalScale.dt`
is the time between images within a pair and does not determine this interval,
especially for strided or overlapping pairs. Invalid samples can `:error`, be linearly `:interpolate`d, or
be replaced by the valid-sample `:mean`; invalid handling is always explicit.
"""
function result_spectrum(results::AbstractVector{<:PIVResult}, i::Int, j::Int;
                         component::Symbol=:u, invalid::Symbol=:error,
                         dt::Union{Nothing,Real}=nothing, window::Symbol=:hann)
    check_same_grid(results)
    dt === nothing && throw(ArgumentError("dt must be the sampling interval between successive results"))
    component in (:u,:v) || throw(ArgumentError("component must be :u or :v"))
    invalid in (:error,:interpolate,:mean) || throw(ArgumentError("invalid must be :error, :interpolate, or :mean"))
    vals = Float64[getproperty(r,component)[i,j] for r in results]
    good = BitVector([!r.mask[i,j] && !r.outliers[i,j] && isfinite(vals[k]) for (k,r) in enumerate(results)])
    if !all(good)
        invalid === :error && throw(ArgumentError("time series contains invalid samples; choose invalid=:interpolate or :mean"))
        any(good) || throw(ArgumentError("time series has no valid samples"))
        if invalid === :mean
            vals[.!good] .= sum(vals[good])/count(good)
        else
            gi = findall(good)
            for k in findall(.!good)
                l = findlast(<(k), gi); h = findfirst(>(k), gi)
                if l === nothing; vals[k]=vals[gi[h]]
                elseif h === nothing; vals[k]=vals[gi[l]]
                else
                    a,b=gi[l],gi[h]; vals[k]=vals[a]+(vals[b]-vals[a])*(k-a)/(b-a)
                end
            end
        end
    end
    power_spectrum(vals; dt, window)
end


# Familiar overloads alongside the named helper.
power_spectrum(results::AbstractVector{<:PIVResult}, i::Int, j::Int; kwargs...) =
    result_spectrum(results,i,j; kwargs...)
function power_spectrum(results::AbstractVector{<:PIVResult}; index::Tuple{Int,Int}, kwargs...)
    result_spectrum(results,index...; kwargs...)
end
