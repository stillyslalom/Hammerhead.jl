# Original-stencil information is a scientific convention, not mathematical
# support of the cardinal B-spline (whose prefilter has nonlocal influence).
# A code describes eligible ORIGINAL pixels in the clipped 4×4 raw stencil:
# 0 = none, 17 = at least two distinct values, 1:16 = first eligible offset
# in an exactly constant stencil. Values are compared in processing precision.
struct _OriginalStencilMap{T,A,M}
    image::A
    mask::M
    codes::Union{Nothing,Matrix{UInt8}} # nothing means every stencil varies
end

mutable struct _OriginalSupportContext{T,A,B,M,W}
    imageA::A
    imageB::B
    mask::M
    workspace::W
    prepared::Bool
    maps::Any
end

_original_support_context(A, B, mask, ::Type{T}, workspace) where {T} =
    _OriginalSupportContext{T,typeof(A),typeof(B),typeof(mask),typeof(workspace)}(
        A, B, mask, workspace, false, nothing)

function _original_stencil_map(image, mask, ::Type{T}, workspace, frame) where {T}
    nr, nc = size(image)
    codes = nothing
    cached = workspace === nothing ? nothing : get(workspace.engines, (:original_stencil, frame), nothing)
    for j in 1:(nc - 1), i in 1:(nr - 1)
        code = UInt8(0)
        reference = zero(T)
        for dc in 0:3
            for dr in 0:3
                r, c = i - 1 + dr, j - 1 + dc
                (1 <= r <= nr && 1 <= c <= nc) || continue
                mask !== nothing && mask[r, c] && continue
                value = T(image[r, c])
                if code == 0
                    code = UInt8(1 + dr + 4dc)
                    reference = value
                elseif value != reference
                    code = UInt8(17)
                    break
                end
            end
            code == 17 && break
        end
        if codes === nothing && code != 17
            if cached !== nothing && size(cached[1]) == (nr - 1, nc - 1)
                codes = cached[1]
                fill!(codes, UInt8(17))
            else
                codes = fill(UInt8(17), nr - 1, nc - 1)
                workspace === nothing ||
                    (workspace.engines[(:original_stencil, frame)] = Any[codes])
            end
        end
        codes === nothing || (codes[i, j] = code)
    end
    return _OriginalStencilMap{T,typeof(image),typeof(mask)}(image, mask, codes)
end

function _prepare_original_support!(context::_OriginalSupportContext{T}) where {T}
    context.prepared && return context.maps
    context.prepared = true
    # Do not turn a corrupted sample into a zero ensemble contribution. With
    # nonfinite source pixels, retain the existing correlation-plane rejection.
    if all(v -> isfinite(T(v)), context.imageA) &&
       all(v -> isfinite(T(v)), context.imageB)
        context.maps = (
            _original_stencil_map(context.imageA, context.mask, T, context.workspace, :A),
            _original_stencil_map(context.imageB, context.mask, T, context.workspace, :B))
    end
    return context.maps
end

@inline function _original_stencil_value(map::_OriginalStencilMap{T}, yy, xx) where {T}
    nr, nc = size(map.image)
    # Extrapolated zeros are numerical boundary handling, not source contrast.
    (1 <= yy <= nr && 1 <= xx <= nc) || return UInt8(0), zero(T)
    iy = min(floor(Int, yy), nr - 1)
    ix = min(floor(Int, xx), nc - 1)
    map.codes === nothing && return UInt8(17), zero(T)
    @inbounds code = map.codes[iy, ix]
    (code == 0 || code == 17) && return code, zero(T)
    offset = Int(code) - 1
    @inbounds value = T(map.image[iy - 1 + offset % 4, ix - 1 + offset ÷ 4])
    return code, value
end

# Same arithmetic as the portable deformation kernel. Also checking the CPU
# interpolation coordinates below makes the proof conservative at cell edges
# where the two evaluation orders can differ by a rounding unit.
@inline function _original_portable_displacement(predictor, r, c, ::Type{T}) where {T}
    ny, nx = length(predictor.y), length(predictor.x)
    ysp = ny > 1 ? T(predictor.y[2] - predictor.y[1]) : one(T)
    xsp = nx > 1 ? T(predictor.x[2] - predictor.x[1]) : one(T)
    iy, ly = _pk_lin_axis(T(first(predictor.y)), ysp, ny, T(r))
    ix, lx = _pk_lin_axis(T(first(predictor.x)), xsp, nx, T(c))
    iy2 = ny == 1 ? iy : iy + 1
    ix2 = nx == 1 ? ix : ix + 1
    @inbounds begin
        a = predictor.u[iy, ix] + lx * (predictor.u[iy, ix2] - predictor.u[iy, ix])
        b = predictor.u[iy2, ix] + lx * (predictor.u[iy2, ix2] - predictor.u[iy2, ix])
        du = a + ly * (b - a)
        a = predictor.v[iy, ix] + lx * (predictor.v[iy, ix2] - predictor.v[iy, ix])
        b = predictor.v[iy2, ix] + lx * (predictor.v[iy2, ix2] - predictor.v[iy2, ix])
        dv = a + ly * (b - a)
    end
    return du, dv
end

function _original_window_contrast(map::_OriginalStencilMap{T}, rs, cs, wr, wc,
                                   mask, predictor, itpu, itpv, sign) where {T}
    found = false
    reference = zero(T)
    @inbounds for j in cs:(cs + wc - 1), i in rs:(rs + wr - 1)
        mask !== nothing && mask[i, j] && continue
        # Union the sampled raw stencils of both supported deformation models;
        # any real contrast in that union is enough, with no magnitude cutoff.
        du, dv = _original_portable_displacement(predictor, i, j, T)
        cpu_y, cpu_x = i + sign * itpv(i, j) / 2, j + sign * itpu(i, j) / 2
        ka_y, ka_x = T(i) + sign * dv / 2, T(j) + sign * du / 2
        for (yy, xx) in ((cpu_y, cpu_x), (ka_y, ka_x))
            code, value = _original_stencil_value(map, yy, xx)
            code == 17 && return true
            code == 0 && continue
            if found
                value != reference && return true
            else
                reference = value
                found = true
            end
        end
    end
    return false
end

function _original_source_gate(context, predictor, grid, params, mask;
                               gate = nothing, threaded = false)
    (context === nothing || predictor === nothing) && return nothing
    maps = _prepare_original_support!(context)
    maps === nothing && return nothing
    shape = (length(grid.y), length(grid.x))
    # A fresh name: reassigning a captured variable would box it.
    out = gate === nothing ? fill(true, shape) : fill!(gate, true)
    itpu = predictor_interpolant(predictor.y, predictor.x, predictor.u)
    itpv = predictor_interpolant(predictor.y, predictor.x, predictor.v)
    wr, wc = params.window_size
    sr, sc = params.search_area_size
    mr, mc = div.(params.search_area_size .- params.window_size, 2)
    function check_jobs(jobs)
        for (gi, gj, rs, cs) in jobs
            out[gi, gj] =
                _original_window_contrast(maps[1], rs, cs, wr, wc, mask,
                                          predictor, itpu, itpv, -1) &&
                _original_window_contrast(maps[2], rs - mr, cs - mc, sr, sc, mask,
                                          predictor, itpu, itpv, 1)
        end
    end
    if threaded && Threads.nthreads() > 1 && length(grid.jobs) > 1
        chunk = cld(length(grid.jobs), Threads.nthreads())
        @sync for jobs in Iterators.partition(grid.jobs, chunk)
            Threads.@spawn check_jobs(jobs)
        end
    else
        check_jobs(grid.jobs)
    end
    return out
end

@inline _source_informative(gate, gi, gj) = gate === nothing || gate[gi, gj]
@inline _source_origin(gate, job) = _source_informative(gate, job[1], job[2]) ? job[3] : 0
