# Calibrated point sampling onto an explicit common grid. These are detached
# sampled arrays, not new measurements or native PIVResult objects.

function _resampling_label(value,name;optional=false)
    optional && value===nothing && return nothing
    value isa AbstractString && !isempty(strip(value)) ||
        throw(ArgumentError("$name must be a nonempty string$(optional ? " or nothing" : "")"))
    String(value)
end

function _resampling_axis(axis,name)
    axis isa AbstractVector{<:Real} || throw(ArgumentError("$name must be a real vector or range"))
    Base.require_one_based_indexing(axis)
    isempty(axis) && throw(ArgumentError("$name must not be empty"))
    values=try Float64.(axis) catch
        throw(ArgumentError("$name must be representable in Float64 geometry"))
    end
    values=collect(values)
    all(isfinite,values) || throw(ArgumentError("$name must have finite Float64 coordinates"))
    any(i->!iszero(axis[i]) && iszero(values[i]),eachindex(values)) &&
        throw(ArgumentError("$name has nonzero coordinates below the Float64 geometry range"))
    if length(values)>1
        ascending=values[2]>values[1]
        all(i->ascending ? values[i]>values[i-1] : values[i]<values[i-1],2:length(values)) ||
            throw(ArgumentError("$name must be strictly monotone after Float64 conversion"))
    end
    values
end

function _resampling_type(types...)
    T=try float(promote_type(types...)) catch
        throw(ArgumentError("numeric inputs must support floating point promotion"))
    end
    isconcretetype(T) && T<:AbstractFloat || throw(ArgumentError("numeric inputs require a concrete floating point output type"))
    T
end

function _resampling_transform(transform)
    transform isa PlanarTransform || throw(ArgumentError("transform must be a PlanarTransform"))
    eltype(transform.matrix) in (Float32,Float64) ||
        throw(ArgumentError("calibration coefficients must use Float32 or Float64"))
    all(isfinite,transform.matrix) && all(isfinite,transform.offset) ||
        throw(ArgumentError("calibration must have finite coefficients"))
    # Float32 widens exactly; Float64 stays unchanged. Use this ONE applied map
    # for inverse queries and vector components, independently of output type.
    applied=PlanarTransform(Float64.(transform.matrix),Float64.(transform.offset))
    A=Matrix(applied.matrix)
    determinant=Rational{BigInt}(A[1,1])*Rational{BigInt}(A[2,2])-
        Rational{BigInt}(A[1,2])*Rational{BigInt}(A[2,1])
    iszero(determinant) && throw(ArgumentError("calibration must be nonsingular"))
    # Generic LU inversion avoids the underflowing 2x2 determinant formula.
    inverse=try inv(A) catch
        throw(ArgumentError("calibration inverse is unavailable in Float64 arithmetic"))
    end
    all(isfinite,inverse) || throw(ArgumentError("calibration inverse is outside the finite Float64 range"))
    applied,inverse
end

function _resampling_setup(source_x,source_y,x,y,transform,source_type;
                           length_unit,coordinate_frame,dt=nothing)
    sx=_resampling_axis(source_x,"source x");sy=_resampling_axis(source_y,"source y")
    tx=_resampling_axis(x,"target x");ty=_resampling_axis(y,"target y")
    applied,inverse=_resampling_transform(transform)
    lu=_resampling_label(length_unit,"length_unit")
    frame=_resampling_label(coordinate_frame,"coordinate_frame";optional=true)
    T=_resampling_type(source_type,eltype(transform.matrix),eltype(x),eltype(y),
        dt===nothing ? source_type : typeof(dt))
    target_x=try T.(tx) catch;throw(ArgumentError("target x cannot be represented in the output type"));end
    target_y=try T.(ty) catch;throw(ArgumentError("target y cannot be represented in the output type"));end
    # No coordinate aliasing; the output axes describe the represented queries.
    target_x=collect(target_x);target_y=collect(target_y)
    all(isfinite,target_x) && all(isfinite,target_y) || throw(ArgumentError("target coordinates exceed the output numeric range"))
    Float64.(target_x)==tx && Float64.(target_y)==ty ||
        throw(ArgumentError("applied target geometry cannot be represented exactly in the promoted output precision"))
    common=(method=:bilinear_point_sampling,coordinate_frame=frame,length_unit=lu,
        source_size=(length(sy),length(sx)),source_axes=(x=sx,y=sy),
        applied_transform=applied,geometry_type=Float64,output_type=T)
    (;sx,sy,tx,ty,applied,inverse,T,target_x,target_y,common)
end

function _resampling_buffers(setup;planar)
    dims=(length(setup.ty),length(setup.tx));T=setup.T
    available=falses(dims);outside=falses(dims);masked_support=falses(dims)
    nonfinite_support=falses(dims);arithmetic_failure=falses(dims)
    contributor_count=zeros(UInt8,dims)
    if planar
        (;u=fill(T(NaN),dims),v=fill(T(NaN),dims),available,outside,masked_support,
            outlier_support=falses(dims),nonfinite_support,arithmetic_failure,contributor_count)
    else
        (;values=fill(T(NaN),dims),available,outside,masked_support,nonfinite_support,
            arithmetic_failure,contributor_count)
    end
end

# Exact represented coverage only. A singleton has no spatial extent. Exact
# nodes use one contributor, avoiding artificial requirements on zero weights.
function _resampling_bracket(axis,q)
    lo,hi=minmax(first(axis),last(axis))
    lo<=q<=hi || return nothing
    n=length(axis)
    n==1 && return (indices=(1,0),weights=(1.0,0.0),count=1,failure=false)
    index=searchsortedlast(axis,q;rev=first(axis)>last(axis))
    axis[index]==q && return (indices=(index,0),weights=(1.0,0.0),count=1,failure=false)
    a,b=axis[index],axis[index+1]
    denominator=b-a
    if isfinite(denominator)
        left=(b-q)/denominator;right=(q-a)/denominator
    else
        denominator=b/2-a/2
        left=(b/2-q/2)/denominator;right=(q/2-a/2)/denominator
    end
    # Both contributors are geometrically positive, even when a quotient
    # underflows. Do not silently erase their masks/outlier/nonfinite flags.
    failure=!(isfinite(left) && isfinite(right) && left>0 && right>0)
    (indices=(index,index+1),weights=(left,right),count=2,failure=failure)
end

function _resampling_query(setup,x,y)
    A=setup.inverse;b=setup.applied.offset
    dx=x-b[1];dy=y-b[2]
    (A[1,1]*dx+A[1,2]*dy,A[2,1]*dx+A[2,2]*dy)
end

function _resampling_convert(::Type{T},value) where T
    converted=try T(value) catch;return nothing;end
    isfinite(converted) ? converted : nothing
end

function _resampling_planar!(out,r,setup,interval,include_invalid)
    W=promote_type(Float64,setup.T);A=setup.applied.matrix
    for j in eachindex(setup.tx),i in eachindex(setup.ty)
        qx,qy=_resampling_query(setup,setup.tx[j],setup.ty[i])
        if !(isfinite(qx) && isfinite(qy))
            out.arithmetic_failure[i,j]=true;continue
        end
        bx=_resampling_bracket(setup.sx,qx);by=_resampling_bracket(setup.sy,qy)
        if bx===nothing || by===nothing
            out.outside[i,j]=true;continue
        end
        out.contributor_count[i,j]=UInt8(bx.count*by.count)
        failure=bx.failure || by.failure
        # Inspect every positive per-axis contributor before multiplying weights.
        # Support flags overlap intentionally (a masked NaN has both flags).
        for cy in 1:by.count,cx in 1:bx.count
            row=by.indices[cy];col=bx.indices[cx]
            out.masked_support[i,j] |= r.mask[row,col]
            out.outlier_support[i,j] |= r.outliers[row,col]
            finite=isfinite(r.u[row,col]) && isfinite(r.v[row,col])
            out.nonfinite_support[i,j] |= !finite
            weight=by.weights[cy]*bx.weights[cx]
            good=isfinite(weight) && weight>0
            failure |= !good
        end
        out.arithmetic_failure[i,j]=failure
        (failure || out.masked_support[i,j] || out.nonfinite_support[i,j] ||
            (out.outlier_support[i,j] && !include_invalid)) && continue
        # Ineligible neighborhoods never perform value arithmetic. Eligibility
        # and weight admission are complete before this second, <=4-node pass.
        u=zero(W);v=zero(W)
        for cy in 1:by.count,cx in 1:bx.count
            row=by.indices[cy];col=bx.indices[cx]
            weight=by.weights[cy]*bx.weights[cx]
            su=_resampling_convert(W,r.u[row,col]);sv=_resampling_convert(W,r.v[row,col])
            if su===nothing || sv===nothing
                failure=true;break
            end
            u+=W(weight)*su;v+=W(weight)*sv
        end
        if failure || !(isfinite(u) && isfinite(v))
            out.arithmetic_failure[i,j]=true;continue
        end
        # Convert the sampled vector basis with the SAME applied coefficients
        # used above for geometry. Translation never affects the vector.
        mapped_u=(W(A[1,1])*u+W(A[1,2])*v)/W(interval)
        mapped_v=(W(A[2,1])*u+W(A[2,2])*v)/W(interval)
        result_u=_resampling_convert(setup.T,mapped_u);result_v=_resampling_convert(setup.T,mapped_v)
        if result_u===nothing || result_v===nothing
            out.arithmetic_failure[i,j]=true
        else
            out.u[i,j]=result_u;out.v[i,j]=result_v;out.available[i,j]=true
        end
    end
    out
end

function _resampling_image!(out,image,mask,setup)
    W=promote_type(Float64,setup.T)
    for j in eachindex(setup.tx),i in eachindex(setup.ty)
        qx,qy=_resampling_query(setup,setup.tx[j],setup.ty[i])
        if !(isfinite(qx) && isfinite(qy))
            out.arithmetic_failure[i,j]=true;continue
        end
        bx=_resampling_bracket(setup.sx,qx);by=_resampling_bracket(setup.sy,qy)
        if bx===nothing || by===nothing
            out.outside[i,j]=true;continue
        end
        out.contributor_count[i,j]=UInt8(bx.count*by.count)
        failure=bx.failure || by.failure
        for cy in 1:by.count,cx in 1:bx.count
            row=by.indices[cy];col=bx.indices[cx]
            mask===nothing || (out.masked_support[i,j] |= mask[row,col])
            finite=isfinite(image[row,col]);out.nonfinite_support[i,j] |= !finite
            weight=by.weights[cy]*bx.weights[cx]
            good=isfinite(weight) && weight>0;failure |= !good
        end
        out.arithmetic_failure[i,j]=failure
        (failure || out.masked_support[i,j] || out.nonfinite_support[i,j]) && continue
        value=zero(W)
        for cy in 1:by.count,cx in 1:bx.count
            row=by.indices[cy];col=bx.indices[cx]
            weight=by.weights[cy]*bx.weights[cx]
            sample=_resampling_convert(W,image[row,col])
            if sample===nothing
                failure=true;break
            end
            value+=W(weight)*sample
        end
        if failure || !isfinite(value)
            out.arithmetic_failure[i,j]=true;continue
        end
        result=_resampling_convert(setup.T,value)
        if result===nothing
            out.arithmetic_failure[i,j]=true
        else
            out.values[i,j]=result;out.available[i,j]=true
        end
    end
    out
end

"""
    resample_planar(result::PIVResult, x, y; transform, length_unit,
        dt=nothing, time_unit=nothing, coordinate_frame=nothing,
        include_invalid=false)

Sample a raw unscaled planar vector field onto the explicit common calibration
grid `x`, `y`. `transform::PlanarTransform` maps ORIGINAL image pixels to that
grid; ROI result axes already retain full-image coordinates. Queries are inverse
mapped, bilinearly sampled, and components mapped by the affine matrix only.
An explicit positive finite `dt` plus `time_unit` converts displacement to
velocity. Without it, components remain displacement per input-pair interval,
with the conventional `"frame"` label. Attached scales (including already
physical results) are refused to prevent double conversion.

Return a detached named tuple with copied common-grid `x`, `y`, matrices `u`,
`v`, `available`, `outside`, `masked_support`, `outlier_support`,
`nonfinite_support`, `arithmetic_failure`, UInt8 `contributor_count`, and
`metadata`. Availability is JOINT for both components; unavailable values are
NaN. Every positive-weight contributor must be unmasked and finite. Current
outlier flags are excluded unless `include_invalid=true`; their support flag
remains visible when admitted. Flags can overlap. Exact-node/edge samples ignore
zero-weight neighbors. Positive-weight underflow is an arithmetic failure,
never a reason to ignore an invalid contributor. Outside queries have no
contributors; arithmetic failures are distinct from geometrical exclusion.

Source/target axes must be nonempty, finite and strictly monotone after Float64
geometry conversion; descending/irregular axes are supported. A singleton
source axis supports only its exact represented coordinate, without tolerance
or invented area coverage. Calibration coefficients must use Float32/Float64.
One exact Float64 copy governs queries AND vector basis. Inverse coefficients
must be finite in Float64. Output precision promotes source, calibration,
target-axis and explicit delay types; all-Float32 inputs can retain Float32.
Returned target axes are the applied Float64 query geometry represented in the
promoted output type; lossy output-axis conversion and nonzero geometry
underflow are refused. Ineligible neighborhoods do not perform value arithmetic.

`metadata` records `method=:bilinear_point_sampling`, optional opaque
`coordinate_frame`, `length_unit`, `source_size`, detached `source_axes`,
`applied_transform`, `geometry_type=Float64`, `output_type`, `quantity`
(`:displacement`/`:velocity`), `dt`, `time_unit`, `component_unit` and
`include_invalid`. A shared label does not verify calibration or synchronization.
No result/source payload is retained. Memory is O(target nodes + axis lengths).
This is point sampling, without extrapolation, anti-alias filtering or
conservative averaging. No PIVResult, peak diagnostics, uncertainty or measurement
history is fabricated; numerical availability is not measurement accuracy.
"""
function resample_planar(r::PIVResult,x,y;transform,length_unit,dt=nothing,
                         time_unit=nothing,coordinate_frame=nothing,include_invalid::Bool=false)
    r.scale===nothing || throw(ArgumentError("resample_planar requires raw pixel data without an attached PhysicalScale; removing metadata does not undo conversion"))
    Base.require_one_based_indexing(r.x,r.y,r.u,r.v,r.mask,r.outliers)
    dims=(length(r.y),length(r.x))
    all(a->size(a)==dims,(r.u,r.v,r.mask,r.outliers)) || throw(DimensionMismatch("planar geometry, components, mask and outliers must match"))
    if dt===nothing
        time_unit===nothing || throw(ArgumentError("time_unit requires an explicit dt"))
        tu="frame"
    else
        dt isa Real && !(dt isa Bool) && isfinite(dt) && dt>0 || throw(ArgumentError("dt must be a positive finite numeric interval"))
        tu=_resampling_label(time_unit,"time_unit")
    end
    setup=_resampling_setup(r.x,r.y,x,y,transform,eltype(r.u);length_unit,coordinate_frame,dt)
    delay=dt===nothing ? nothing : _resampling_convert(setup.T,dt)
    dt!==nothing && (delay===nothing || delay<=0) && throw(ArgumentError("dt must remain positive and finite in the promoted output precision"))
    interval=delay===nothing ? one(setup.T) : delay
    out=_resampling_buffers(setup;planar=true)
    _resampling_planar!(out,r,setup,interval,include_invalid)
    metadata=merge(setup.common,(quantity=dt===nothing ? :displacement : :velocity,
        dt=delay,time_unit=tu,component_unit=setup.common.length_unit*"/"*tu,include_invalid=include_invalid))
    merge((x=setup.target_x,y=setup.target_y),out,(;metadata))
end

"""
    resample_image(image, x, y; transform, length_unit, mask=nothing,
        source_x=axes(image,2), source_y=axes(image,1),
        value_unit=nothing, coordinate_frame=nothing)

Sample a real scalar image onto the same explicit calibrated grid used by
[`resample_planar`](@ref), without changing scalar values' basis. `transform`
maps source pixels to common coordinates. Default source coordinates are
one-based full-image x=columns, y=rows. A crop must supply its original pixel
`source_x`, `source_y`, or use a calibration adjusted to the crop; registration
cannot infer a cropped image's origin. Source arrays require one-based indexing.
Convert color images with `load_image` first.

Return detached `x`, `y`, `values`, `available`, `outside`, `masked_support`,
`nonfinite_support`, `arithmetic_failure`, UInt8 `contributor_count`, and
`metadata`. `mask=true` excludes source pixels. All positive contributors must
be finite and unmasked. Outside/unavailable values are NaN, never zero fill.
Exact/irregular/descending/singleton support, arithmetic flags, Float64 applied
geometry and output promotion follow `resample_planar`. No image is retained.

Metadata shares the geometry fields documented there, with
`quantity=:scalar_sample` and optional opaque `value_unit` instead of vector
delay/component fields. Units and frame labels are caller assertions. Sampling
does not calibrate PLIF intensity into concentration, conserve pixel integrals,
filter aliasing or certify static alignment of a transient refractive flow.
Combine both outputs' `available` masks explicitly for joint analysis.
"""
function resample_image(image::AbstractMatrix{<:Real},x,y;transform,length_unit,mask=nothing,
                        source_x=axes(image,2),source_y=axes(image,1),value_unit=nothing,
                        coordinate_frame=nothing)
    Base.require_one_based_indexing(image)
    mask===nothing || (mask isa AbstractMatrix{Bool} && size(mask)==size(image)) ||
        throw(DimensionMismatch("image mask must be a Bool matrix matching the source image"))
    mask===nothing || Base.require_one_based_indexing(mask)
    length(source_x)==size(image,2) && length(source_y)==size(image,1) ||
        throw(DimensionMismatch("image source axes must match its columns and rows"))
    vu=_resampling_label(value_unit,"value_unit";optional=true)
    setup=_resampling_setup(source_x,source_y,x,y,transform,eltype(image);length_unit,coordinate_frame)
    out=_resampling_buffers(setup;planar=false)
    _resampling_image!(out,image,mask,setup)
    metadata=merge(setup.common,(quantity=:scalar_sample,value_unit=vu))
    merge((x=setup.target_x,y=setup.target_y),out,(;metadata))
end
