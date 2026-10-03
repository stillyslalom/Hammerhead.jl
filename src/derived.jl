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
    flow_derivatives(result::PIVResult; include_invalid=false)
    flow_derivatives(x, y, u, v; valid=isfinite.(u) .& isfinite.(v))

Return `(; dudx, dudy, dvdx, dvdy, valid)` on a regular planar grid. Each
derivative has the units of the supplied vector components divided by the
coordinate units. The result method uses stored arrays; call [`physical`](@ref)
first if physical velocity gradients are needed. Masked, nonfinite, and
flagged vectors are excluded by default; `include_invalid=true` admits
flagged vectors but still excludes masked and nonfinite ones. The array
method uses its explicit `valid` mask.

Central differences use two valid neighbors; boundaries or gaps use a
one-sided difference when possible. Values without a valid local stencil
are `NaN`. Each axis needs at least two distinct coordinates.
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

"""
    vorticity(derivatives)
    vorticity(result::PIVResult; include_invalid=false)

Return planar `∂v/∂x - ∂u/∂y` as a matrix. Pass the tuple from
[`flow_derivatives`](@ref), or a result to compute the derivatives first.
Units follow the supplied arrays; invalid stencils remain `NaN`.
"""
vorticity(d::NamedTuple) = d.dvdx .- d.dudy
"""
    divergence(derivatives)
    divergence(result::PIVResult; include_invalid=false)

Return planar `∂u/∂x + ∂v/∂y` as a matrix. Pass the tuple from
[`flow_derivatives`](@ref), or a result to compute the derivatives first.
Units follow the supplied arrays; invalid stencils remain `NaN`.
"""
divergence(d::NamedTuple) = d.dudx .+ d.dvdy
vorticity(args...; kwargs...) = vorticity(flow_derivatives(args...; kwargs...))
divergence(args...; kwargs...) = divergence(flow_derivatives(args...; kwargs...))

"""
    strain_rate(derivatives)
    strain_rate(result::PIVResult; include_invalid=false)

Return planar strain components `(; xx, yy, xy, magnitude)`, where
`xy = (∂u/∂y + ∂v/∂x)/2` and `magnitude = √(2(xx² + yy² + 2xy²))`.
The arrays retain the units and invalid-stencil `NaN`s of the derivatives.
"""
function strain_rate(d::NamedTuple)
    xx, yy = d.dudx, d.dvdy
    xy = (d.dudy .+ d.dvdx) ./ 2
    magnitude = sqrt.(2 .* (xx.^2 .+ yy.^2 .+ 2 .* xy.^2))
    (; xx, yy, xy, magnitude)
end
strain_rate(args...; kwargs...) = strain_rate(flow_derivatives(args...; kwargs...))

"""
    swirling_strength(derivatives)
    swirling_strength(result::PIVResult; include_invalid=false)

Planar swirling strength: the magnitude of the imaginary part of the two
eigenvalues of the local 2x2 velocity-gradient tensor. It is zero where the
eigenvalues are real. Returns a matrix with the same units as the gradient;
invalid stencils remain `NaN`.
"""
function swirling_strength(d::NamedTuple)
    disc = d.dudx .* d.dvdy .- d.dudy .* d.dvdx .-
           ((d.dudx .+ d.dvdy) ./ 2).^2
    sqrt.(max.(disc, 0))
end
swirling_strength(args...; kwargs...) = swirling_strength(flow_derivatives(args...; kwargs...))

"""
    q_criterion(derivatives)
    q_criterion(result::PIVResult; include_invalid=false)

The explicitly two-dimensional Q criterion,
`Q = (||Omega||^2 - ||S||^2)/2 = -tr(grad(u)^2)/2`. Positive values indicate
rotation dominating strain in the measured plane; no unmeasured gradients
are assumed. Returns a matrix in squared gradient units; invalid stencils
remain `NaN`.
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
    tx = (qx - x0) / (x1 - x0); ty = (qy - y0) / (y1 - y0)
    weights = ((1-ty)*(1-tx), (1-ty)*tx, ty*(1-tx), ty*tx)
    value = 0.0
    for k in eachindex(vals)
        weights[k] > 0 || continue
        isfinite(vals[k]) || return NaN
        value += weights[k] * vals[k]
    end
    value
end

"""
    extract_profile(result::PIVResult, points; n=100, include_invalid=false)

Sample the stored `u` and `v` arrays at `n` equally spaced arc-length
positions along a polyline of `(x, y)` points. Return
`(; s, x, y, u, v)` as vectors in the result's current units. At least two
points, positive total length, and `n ≥ 2` are required.

Samples outside the grid or requiring any nonfinite, masked, or flagged
corner with positive interpolation weight become `NaN`. An invalid corner
with zero weight does not affect an exact node or edge sample.
`include_invalid=true` admits flagged corners only. Use
[`physical`](@ref) first if physical positions and velocities are needed.
"""
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

"""
    extract_region(result::PIVResult, region; include_invalid=false)

Return `(; indices, x, y, u, v, mask, included)` for valid grid nodes inside a
rectangle `(xmin, xmax, ymin, ymax)` or a polygon of `(x, y)` vertices.
`included` is a grid-shaped inclusion mask: `true` means the node was returned.
`mask` is a backward-compatible alias for the same matrix. Both have the
opposite convention from `result.mask`, where `true` means excluded.
Masked and nonfinite vectors are always omitted; `include_invalid=true`
also includes flagged vectors. Values retain the result's stored units.
"""
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
       u=r.u[inds], v=r.v[inds], mask=valid, included=valid)
end

"""
    circulation(result::PIVResult, contour; close=true, include_invalid=false)

Estimate `∮(u dx + v dy)` by sampling along at least three contour points.
The sign follows the supplied vertex order. `close=true` joins the last
point to the first when needed; with `close=false`, only the supplied path
is integrated. Return `NaN` if any sampled component is invalid or outside
the grid. Units are stored component units multiplied by coordinate units.
"""
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

"""
    circulation(result::PIVResult; region, include_invalid=false,
                coverage=:error)

Integrate planar vorticity over a rectangle `(xmin, xmax, ymin, ymax)` or
polygonal `region`. The scalar result uses stored component and coordinate
units and does not depend on polygon vertex order. A grid cell contributes
only when all four vorticity corners are finite.

With the default `coverage=:error`, throw if any requested area lies outside
the grid or lacks valid vorticity. Use `coverage=:report` to receive
`(; value, valid_area, requested_area, coverage_fraction, complete)`. `value` integrates
only the valid area and is `NaN` when no cell contributes. `requested_area`
is the whole supplied region, including any portion outside the grid;
`coverage_fraction` is `valid_area / requested_area`. `complete` is the
authoritative coverage check; it remains `false` if a missing sliver is too
small for the ratio to differ from 1 in floating-point arithmetic.

`include_invalid=true` admits flagged vectors when deriving vorticity, but
masked and nonfinite vectors remain excluded.
"""
function circulation(r::PIVResult; region, include_invalid::Bool=false,
                     coverage::Symbol=:error)
    coverage in (:error, :report) ||
        throw(ArgumentError("coverage must be :error or :report, got :$coverage"))
    length(r.x)>=2 && length(r.y)>=2 || throw(ArgumentError("circulation needs at least a 2x2 grid"))
    for axis in (r.x, r.y)
        steps = diff(axis)
        (all(x -> x > 0, steps) || all(x -> x < 0, steps)) ||
            throw(ArgumentError("circulation grid axes must be strictly monotonic"))
    end
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
    all(p -> all(isfinite, p), poly) ||
        throw(ArgumentError("circulation region vertices must be finite"))
    # Translate before the shoelace sum to avoid subtracting large, nearly
    # equal products when a small region has large world coordinates.
    px, py = poly[1]
    signed_twice_area = sum((poly[k][1]-px)*(poly[mod1(k+1,length(poly))][2]-py) -
                            (poly[mod1(k+1,length(poly))][1]-px)*(poly[k][2]-py)
                            for k in eachindex(poly))
    isfinite(signed_twice_area) && signed_twice_area != 0 ||
        throw(ArgumentError("circulation region must have finite nonzero area"))
    orientation = sign(signed_twice_area)
    requested_area = abs(signed_twice_area) / 2
    xmin, xmax = extrema(r.x)
    ymin, ymax = extrema(r.y)
    outside_grid = any(p -> !(xmin <= p[1] <= xmax && ymin <= p[2] <= ymax), poly)
    omega = vorticity(r; include_invalid)
    total = 0.0
    valid_area = 0.0
    area_compensation = 0.0
    invalid_overlap = false
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
        a = clipped[1]
        cell_area = orientation * sum((clipped[k][1]-a[1])*(clipped[k+1][2]-a[2]) -
                                      (clipped[k+1][1]-a[1])*(clipped[k][2]-a[2])
                                      for k in 2:length(clipped)-1) / 2
        cell_area > 0 || continue
        corners = (omega[i,j],omega[i,j+1],omega[i+1,j],omega[i+1,j+1])
        if !all(isfinite,corners)
            invalid_overlap = true
            continue
        end
        # Compensated accumulation keeps the covered area stable on finer grids.
        area_step = cell_area - area_compensation
        next_area = valid_area + area_step
        area_compensation = (next_area - valid_area) - area_step
        valid_area = next_area
        # Three-point triangle quadrature integrates the cell's bilinear
        # interpolant exactly, including its x*y term.
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
    # A simple polygon wholly inside the rectangular grid is covered exactly
    # when every intersected cell is usable. Comparing summed cell areas with
    # polygon area is less reliable at large coordinate offsets.
    complete = valid_area > 0 && !invalid_overlap && !outside_grid
    complete && (valid_area = requested_area)
    if coverage === :report
        return (; value = valid_area > 0 ? total : NaN, valid_area,
                requested_area,
                coverage_fraction = complete ? 1.0 : valid_area / requested_area,
                complete)
    end
    complete || throw(ArgumentError("circulation covers $(valid_area / requested_area) of the requested area; use coverage=:report for the partial integral and coverage diagnostics"))
    return total
end

function _check_spectrum_results(results, i, j)
    r1=check_same_grid(results)
    all(isfinite,r1.x) && all(isfinite,r1.y) || throw(ArgumentError("spectrum grid coordinates must be finite"))
    dims=(length(r1.y),length(r1.x))
    1 <= i <= dims[1] && 1 <= j <= dims[2] || throw(ArgumentError("spectrum index is outside the interrogation grid"))
    signature=_statistics_scale_signature(r1.scale)
    for result in results
        all(a->size(a)==dims,(result.u,result.v,result.mask,result.outliers)) ||
            throw(ArgumentError("spectrum component and flag dimensions must match the interrogation grid"))
        _statistics_scale_signature(result.scale)==signature ||
            throw(ArgumentError("spectrum results must have identical scale factors and unit labels; explicitly convert compatible fields first"))
    end
    r1
end

"""
    result_spectrum(results, i, j; dt=nothing, sample_times=nothing, component=:u,
        invalid=:error, window=:hann, timing_atol=0, timing_rtol=0,
        time_unit=nothing, return_timing=false)

Return `(; frequencies, psd)` for component `:u` or `:v` at grid row `i`,
column `j` across uniformly sampled results. Frequencies are cycles per
time unit; `psd` is one-sided power spectral density. Supply `dt` between
successive results; an attached `PhysicalScale.dt` describes an image pair
and is not used here. The stored component values are analyzed without
automatic physical conversion. `window` is passed to [`power_spectrum`](@ref).
Explicit `sample_times` can replace `dt`; the exact uniformity/tolerance contract
is the same as `power_spectrum`, with no implicit timestamp discovery/resampling.
Grid coordinates, component/flag dimensions, scale factors and unit labels must
agree across results, including absent versus attached scales. Convert differing
pair-delay results explicitly with [`physical`](@ref) before combining compatible
velocity fields. Metadata agreement cannot establish stored-value representation.
The sampling `time_unit` label is independent of component velocity-unit labels.

`return_timing=true` adds detached timing provenance, component/index, original
invalid count, fill policy and attached-scale metadata. The default two-field
return is unchanged. No field/image/result payload is retained in that report.

Masked, flagged, or nonfinite samples cause an error by default. Set
`invalid=:mean` to replace them with the mean of valid samples, or
`invalid=:interpolate` to interpolate interior gaps and hold the nearest
valid value at either end. At least one valid sample is required.
"""
function result_spectrum(results::AbstractVector{<:PIVResult}, i::Int, j::Int;
                         component::Symbol=:u, invalid::Symbol=:error,
                         dt::Union{Nothing,Real}=nothing, sample_times=nothing,
                         timing_atol=0, timing_rtol=0, time_unit=nothing,
                         return_timing::Bool=false, window::Symbol=:hann)
    component in (:u,:v) || throw(ArgumentError("component must be :u or :v"))
    invalid in (:error,:interpolate,:mean) || throw(ArgumentError("invalid must be :error, :interpolate, or :mean"))
    r1=_check_spectrum_results(results,i,j)
    period,timing=_spectrum_sampling(length(results),dt,sample_times,timing_atol,timing_rtol,time_unit,return_timing;require_timing=true)
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
    # The timeline was checked before sample extraction/filling. Filling retains
    # the original regular-grid sample positions, including endpoint holds.
    if sample_times!==nothing
        # Match the inferred-spacing guards of the signal API before its FFT.
        period <= floatmax(Float64)/2 || throw(ArgumentError("inferred FFT PSD normalization is not representable in Float64"))
    end
    spectrum=power_spectrum(vals; dt=period, window)
    if return_timing
        timing["component"]=String(component);timing["index"]=[i,j]
        timing["invalid_count"]=length(good)-count(good);timing["fill_policy"]=String(invalid)
        timing["value_basis"]="stored_component_no_conversion"
        timing["attached_scale"]=_timing_scale(r1.scale)
        return (; spectrum.frequencies,spectrum.psd,timing)
    end
    spectrum
end


# Familiar overloads alongside the named helper.
power_spectrum(results::AbstractVector{<:PIVResult}, i::Int, j::Int; kwargs...) =
    result_spectrum(results,i,j; kwargs...)
function power_spectrum(results::AbstractVector{<:PIVResult}; index::Tuple{Int,Int}, kwargs...)
    result_spectrum(results,index...; kwargs...)
end
