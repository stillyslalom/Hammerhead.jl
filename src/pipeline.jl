"""
    PIVWorkspace

Reusable scratch for [`run_piv`](@ref): the padded cubic-B-spline coefficient
buffers of the two image interpolants, the two image-deformation output
buffers, and a pool of window correlators (cached FFTW plans) keyed by window
configuration. Build one with [`piv_workspace`](@ref) and pass it as the
`workspace` keyword to reuse this scratch across many `run_piv` calls on
**equally sized** image pairs — [`run_piv_sequence`](@ref) and
[`run_piv_ensemble`](@ref) do this internally so a whole batch pays the
allocations once. The buffers resize automatically if a later call presents a
different image size or precision.

A workspace is stateful scratch: it must **not** be shared across `run_piv`
calls running concurrently. The drivers hold one workspace on their serial
pair loop, so `run_piv`'s own internal threading (which only reads the
interpolant coefficients and writes disjoint buffer regions) stays race-free.
"""
mutable struct PIVWorkspace
    imgsize::Union{Nothing,Dims{2}}
    T::Union{Nothing,DataType}
    itpA_coefs::Any                          # padded coefficient buffer, image A
    itpB_coefs::Any                          # padded coefficient buffer, image B
    warpA::Any                               # deformation output buffer, image A
    warpB::Any                               # deformation output buffer, image B
    correlators::Dict{Any,Vector{Correlator}}
end

"""
    piv_workspace() -> PIVWorkspace

Construct an empty [`PIVWorkspace`](@ref). Its buffers allocate lazily on the
first [`run_piv`](@ref) call and are reused (and resized on demand) thereafter
— pass one via the `workspace` keyword when running many equally sized pairs
yourself, to amortize the per-pair interpolant/deformation/correlator
allocations across the batch.
"""
piv_workspace() = PIVWorkspace(nothing, nothing, nothing, nothing, nothing, nothing,
                               Dict{Any,Vector{Correlator}}())

# Point the workspace at this call's image size/precision, discarding buffers
# from a differently shaped prior run. Deformation output buffers are ensured
# here; interpolant coefficient buffers fill in on first use (image_interpolant!)
# and the correlator pool grows on demand (piv_correlators).
function ws_prepare!(ws::PIVWorkspace, imgsize::Dims{2}, ::Type{T}) where {T}
    if ws.imgsize != imgsize || ws.T !== T
        ws.imgsize = imgsize
        ws.T = T
        ws.itpA_coefs = nothing
        ws.itpB_coefs = nothing
        ws.warpA = nothing
        ws.warpB = nothing
        empty!(ws.correlators)
    end
    return ws
end

# Cubic B-spline interpolant reusing a preallocated padded coefficient buffer.
# `coefs === nothing` allocates one (via `interpolate`, whose `copy_with_padding`
# yields the correctly offset-axed array) and returns it for the caller to
# stash; otherwise the buffer is refilled exactly as `copy_with_padding` would
# (zero, then copy the image into the interior) and prefiltered in place with
# `interpolate!`. The in-place prefilter runs the identical tridiagonal system
# on the identical padded array, so the coefficients — and every value the
# resulting interpolant produces — are bitwise identical to `image_interpolant`.
function image_interpolant!(coefs, img::AbstractMatrix, ::Type{T}) where {T}
    it = BSpline(Cubic(Line(OnGrid())))
    if coefs === nothing
        itp = interpolate(eltype(img) === T ? img : T.(img), it)
        return extrapolate(itp, zero(T)), itp.coefs
    end
    fill!(coefs, zero(T))
    ci = CartesianIndices(axes(img))
    copyto!(coefs, ci, img, ci)
    return extrapolate(interpolate!(coefs, it), zero(T)), coefs
end

# One correlator per chunk for this pass. With a workspace, correlators are
# drawn from (and lazily added to) a pool keyed by window configuration, so
# their FFTW plans are built once per configuration for the whole batch; the
# pool is grown only serially here, before any fan-out, and each chunk uses a
# distinct entry, so concurrent tasks never share a correlator. Without a
# workspace, fresh correlators are made per pass (the prior behavior).
function piv_correlators(workspace, params::PIVParameters, ::Type{T}, nchunks::Int) where {T}
    workspace === nothing && return Correlator[make_correlator(params, T) for _ in 1:nchunks]
    key = (T, params.correlation_method, params.window_size, params.padding, params.apodization)
    pool = get!(() -> Correlator[], workspace.correlators, key)
    while length(pool) < nchunks
        push!(pool, make_correlator(params, T))
    end
    return pool
end

"""
    run_piv(imgA, imgB, passes::AbstractVector{PIVParameters};
            threaded = Threads.nthreads() > 1,
            predictor_smoothing = true,
            mask = nothing, mask_threshold = 0.5,
            workspace = nothing) -> PIVResult
    run_piv(imgA, imgB, params::PIVParameters = PIVParameters(); kwargs...)

Run a (multi-pass) PIV analysis on an in-memory image pair. `imgA` and `imgB`
must be equally sized real-valued matrices (load and convert image files with
your preferred image package).

Each pass tiles the images into interrogation windows (`window_size`,
`overlap`), correlates each window pair (`correlation_method`, optionally
zero-padded and apodized), refines the peak to subpixel precision
(`subpixel_method`), and validates the resulting field (universal outlier
detection, peak-ratio threshold, and any additional `validation` pipeline —
see the [validation how-to](../howto/validation.md) for the full list of
validators and their pair-spec syntax) with local-median replacement of
invalid vectors.

From the second pass on, the previous pass's validated field is used as a
predictor: it is smoothed (`predictor_smoothing`), interpolated to pixel
resolution, and both images are symmetrically deformed by ±half the predictor
displacement (central-difference image deformation, cubic B-spline
resampling). The pass then measures only the small residual displacement, so
window sizes can shrink across passes — e.g. `multipass_parameters([64, 32,
16])` — without violating the quarter-window displacement limit. Repeating the
final window size adds convergence sweeps.

A pass whose parameters set `max_iterations > 1` additionally *iterates
in place*: its own validated field is fed back as the deformation predictor
and the windows are re-correlated until the bulk field stops changing
(`convergence_tol`) or the budget runs out, so a bad vector caught by
validation is re-measured within the stage instead of leaking its local-median
replacement into the next pass's predictor. Iterating the final pass to
convergence is also the predictor state the `uncertainty` estimator assumes —
`multipass_parameters([64, 32, 16]; final = (max_iterations = 3,))` is
equivalent to repeating 16-px passes until converged, but stops as soon as
the field settles.

With `uncertainty = true` in the final pass's parameters, a per-vector
measurement uncertainty is estimated from correlation statistics (Wieneke
2015) into `uncertainty_u`/`uncertainty_v` of the result — see
[`PIVParameters`](@ref) for its convergence requirements.

`mask` is an optional image-sized `Bool` matrix marking pixels to exclude
(`true` = excluded), e.g. model geometry or reflection regions — build one
with [`polygon_mask`](@ref), [`load_mask`](@ref), or any Bool array. The mask
describes static lab-frame geometry and is not warped between passes. Windows
whose masked-pixel fraction reaches `mask_threshold` produce no vector: they
are flagged in `result.mask`, hold `NaN`, are never counted as outliers, and
neither enter validation neighborhoods nor donate to replacement medians.
Windows below the threshold are correlated over their valid pixels only.

With `threaded = true` (the default on multithreaded sessions) the window grid
of each pass is split across tasks; results are identical to the serial path.

`workspace` optionally supplies a [`PIVWorkspace`](@ref) (from
[`piv_workspace`](@ref)) whose interpolant, deformation, and correlator scratch
is reused across calls — pass the same one to every `run_piv` in a hand-written
loop over equally sized pairs to amortize those allocations. Results are
bitwise identical to `workspace = nothing`. [`run_piv_sequence`](@ref) and
[`run_piv_ensemble`](@ref) manage a workspace for you.

The numeric precision of the analysis follows the images:
`T = float(promote_type(eltype(imgA), eltype(imgB)))` is used for the
correlators, deformation, and every field of the returned
[`PIVResult`](@ref)`{T}`. Feed `Float32` matrices (e.g.
`load_image(Float32, path)`) to run the whole pipeline in single precision.

Returns the [`PIVResult`](@ref) of the final pass.
"""
function run_piv(imgA::AbstractMatrix{<:Real}, imgB::AbstractMatrix{<:Real},
                 passes::AbstractVector{PIVParameters};
                 threaded::Bool = Threads.nthreads() > 1,
                 predictor_smoothing::Bool = true,
                 mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                 mask_threshold::Real = 0.5,
                 workspace::Union{Nothing,PIVWorkspace} = nothing)
    isempty(passes) && throw(ArgumentError("at least one pass is required"))
    size(imgA) == size(imgB) ||
        throw(DimensionMismatch("images must have the same size, got $(size(imgA)) and $(size(imgB))"))
    mask === nothing || size(mask) == size(imgA) ||
        throw(DimensionMismatch("mask must have the same size as the images, got $(size(mask))"))
    0 < mask_threshold <= 1 ||
        throw(ArgumentError("mask_threshold must be in (0, 1], got $mask_threshold"))

    # The cubic B-spline image interpolants used for predictor deformation
    # depend only on the (unchanging) source images, so their expensive
    # prefilter is paid once here and reused by every deforming pass. A
    # single-pass, non-iterating schedule never deforms, so it skips the
    # prefilter entirely.
    T = float(promote_type(eltype(imgA), eltype(imgB)))
    deforms = length(passes) > 1 || any(p -> p.max_iterations > 1, passes)
    workspace === nothing || ws_prepare!(workspace, size(imgA), T)
    if !deforms
        itpA = itpB = nothing
        warp_buffers = nothing
    elseif workspace === nothing
        itpA = image_interpolant(imgA, T)
        itpB = image_interpolant(imgB, T)
        # The deformed image pair is the same size every pass, and each pass
        # fully overwrites it before use, so allocate the buffers once here.
        warp_buffers = (Matrix{T}(undef, size(imgA)), Matrix{T}(undef, size(imgB)))
    else
        # Reuse the workspace's padded coefficient buffers across pairs (each is
        # refilled and prefiltered in place), stashing any freshly allocated one.
        itpA, workspace.itpA_coefs = image_interpolant!(workspace.itpA_coefs, imgA, T)
        itpB, workspace.itpB_coefs = image_interpolant!(workspace.itpB_coefs, imgB, T)
        workspace.warpA === nothing && (workspace.warpA = Matrix{T}(undef, size(imgA)))
        workspace.warpB === nothing && (workspace.warpB = Matrix{T}(undef, size(imgB)))
        warp_buffers = (workspace.warpA, workspace.warpB)
    end

    result = nothing
    for (k, params) in enumerate(passes)
        predictor = result === nothing ? nothing :
                    build_predictor(result, predictor_smoothing)
        # Intermediate passes always replace invalid vectors: the predictor
        # field must stay well behaved for the deformation to converge.
        result = piv_pass(imgA, imgB, params, predictor, itpA, itpB;
                          threaded, force_replace = k < length(passes),
                          predictor_smoothing, mask, mask_threshold,
                          warp_buffers, workspace)
    end
    return result
end

# Predictor for the next pass: masked cells (NaN) are filled from valid
# neighbors so interpolation and deformation stay finite everywhere, then the
# field is optionally smoothed.
function build_predictor(result::PIVResult, smoothing::Bool)
    pu, pv = result.u, result.v
    if any(result.mask)
        pu, pv = copy(pu), copy(pv)
        replace_vectors!(pu, pv, result.mask)
    end
    if smoothing
        pu, pv = smooth_field(pu), smooth_field(pv)
    end
    return (x = result.x, y = result.y, u = pu, v = pv)
end

run_piv(imgA::AbstractMatrix{<:Real}, imgB::AbstractMatrix{<:Real},
        params::PIVParameters = PIVParameters(); kwargs...) =
    run_piv(imgA, imgB, [params]; kwargs...)

"""
    multipass_parameters(window_sizes; overlap_fraction = 0.5, final = (;), kwargs...) -> Vector{PIVParameters}

Build a multi-pass schedule with one `PIVParameters` per entry of
`window_sizes` (integers or `(rows, cols)` tuples), each with `overlap =
floor(window_size * overlap_fraction)`. All remaining keyword arguments are
forwarded to every pass.

`final` is a `NamedTuple` of keyword overrides applied to the **last** entry
only (its fields override the shared `kwargs` for that pass); it applies even
to a length-1 schedule. Use it for settings you want only on the final pass —
e.g. saving correlation planes for inspection, or turning off outlier
replacement:

```julia
passes = multipass_parameters([64, 32, 16]; padding = true,
                              final = (n_peaks = 3, keep_correlation_planes = true))
```

A schedule is just a `Vector{PIVParameters}`, so arbitrary per-pass control is
always available by constructing the entries directly.

```julia
passes = multipass_parameters([64, 32, 16]; padding = true, apodization = :gauss)
result = run_piv(imgA, imgB, passes)
```
"""
function multipass_parameters(window_sizes::AbstractVector;
                              overlap_fraction::Real = 0.5,
                              final::NamedTuple = (;), kwargs...)
    0 <= overlap_fraction < 1 ||
        throw(ArgumentError("overlap_fraction must be in [0, 1), got $overlap_fraction"))
    isempty(window_sizes) && throw(ArgumentError("window_sizes must not be empty"))
    shared = values(kwargs)
    n = length(window_sizes)
    return [begin
                ws = w isa Integer ? (Int(w), Int(w)) : (Int(w[1]), Int(w[2]))
                extra = i == n ? merge(shared, final) : shared
                PIVParameters(; window_size = ws,
                              overlap = floor.(Int, ws .* overlap_fraction), extra...)
            end
            for (i, w) in enumerate(window_sizes)]
end

# Interrogation grid of one pass: T-typed window-center coordinates, the
# grid-level mask (windows whose masked-pixel fraction reaches mask_threshold
# produce no vector), and the job list of remaining windows.
function pass_grid(::Type{T}, imgsize::Dims{2}, params::PIVParameters,
                   mask::Union{Nothing,AbstractMatrix{Bool}},
                   mask_threshold::Real) where {T}
    nr, nc = imgsize
    wr, wc = params.window_size
    (wr <= nr && wc <= nc) ||
        throw(ArgumentError("window size $(params.window_size) exceeds image size $((nr, nc))"))
    row_starts = 1:(wr - params.overlap[1]):(nr - wr + 1)
    col_starts = 1:(wc - params.overlap[2]):(nc - wc + 1)
    x = T[cs + (wc - 1) / 2 for cs in col_starts]
    y = T[rs + (wr - 1) / 2 for rs in row_starts]
    grid_mask = falses(length(row_starts), length(col_starts))
    if mask !== nothing
        for (gj, cs) in enumerate(col_starts), (gi, rs) in enumerate(row_starts)
            frac = count(view(mask, rs:(rs + wr - 1), cs:(cs + wc - 1))) / (wr * wc)
            grid_mask[gi, gj] = frac >= mask_threshold
        end
    end
    jobs = [(gi, gj, rs, cs) for (gj, cs) in enumerate(col_starts)
                             for (gi, rs) in enumerate(row_starts)
                             if !grid_mask[gi, gj]]
    return (; x, y, grid_mask, jobs)
end

# Deform the image pair symmetrically by the predictor and evaluate the
# predictor displacement on the pass grid. `itpA`/`itpB` are the prebuilt cubic
# B-spline image interpolants (unused, hence permitted `nothing`, when there is
# no predictor to deform by).
function apply_predictor(imgA::AbstractMatrix, imgB::AbstractMatrix, itpA, itpB,
                         predictor, x::AbstractVector, y::AbstractVector, ::Type{T};
                         threaded::Bool = false,
                         warpA::Union{Nothing,Matrix{T}} = nothing,
                         warpB::Union{Nothing,Matrix{T}} = nothing) where {T}
    ny, nx = length(y), length(x)
    predictor === nothing && return imgA, imgB, zeros(T, ny, nx), zeros(T, ny, nx)
    itp_u = extrapolate(interpolate((predictor.y, predictor.x), predictor.u,
                                    Gridded(Linear())), Flat())
    itp_v = extrapolate(interpolate((predictor.y, predictor.x), predictor.v,
                                    Gridded(Linear())), Flat())
    warpA, warpB = deform_images(itpA, itpB, itp_u, itp_v, size(imgA), T;
                                 threaded, warpA, warpB)
    u = T[itp_u(yi, xj) for yi in y, xj in x]
    v = T[itp_v(yi, xj) for yi in y, xj in x]
    return warpA, warpB, u, v
end

# Shared validation + replacement tail of a pass (single-pair or ensemble).
# `alternatives` optionally carries the secondary/tertiary peak displacements
# as a pair of (ny, nx, n_peaks-1) arrays (NaN where absent) for peak
# substitution of flagged vectors.
function validate_and_replace!(result::PIVResult{T}, params::PIVParameters,
                               force_replace::Bool; alternatives = nothing) where {T}
    if params.uod_enable
        apply_validator!(result, UniversalOutlierValidator(params.uod_threshold;
            neighborhood_size = params.uod_neighborhood))
    end
    if params.min_peak_ratio > 1
        apply_validator!(result, PeakRatioValidator(params.min_peak_ratio))
    end
    validate_vectors!(result, params.validation)
    # Masked windows carry no measurement: they are dropped, not "bad".
    result.outliers .&= .!result.mask

    if alternatives !== nothing
        substitute_alternatives!(result, alternatives[1], alternatives[2], params)
    end

    if force_replace || params.replace_outliers
        # Masked cells hold NaN and must not donate to replacement medians;
        # flag them invalid for the fill, then restore their NaN.
        invalid = result.outliers .| result.mask
        if any(invalid)
            replace_vectors!(result.u, result.v, invalid)
            result.u[result.mask] .= T(NaN)
            result.v[result.mask] .= T(NaN)
        end
    end
    return result
end

# One interrogation stage: deform by the predictor, correlate the residual on
# the pass grid, add the predictor back, then validate and (maybe) replace.
# With `max_iterations > 1` the stage iterates: its validated (and always
# replaced) field becomes the next deformation predictor, the images are
# re-deformed, and the windows re-correlated, until the bulk field converges
# (field_change < convergence_tol between sweeps) or the budget is spent — so
# replacement artifacts relax toward measured data within the stage instead of
# cascading into later passes.
function piv_pass(imgA::AbstractMatrix, imgB::AbstractMatrix, params::PIVParameters,
                  predictor, itpA = nothing, itpB = nothing; threaded::Bool,
                  force_replace::Bool,
                  predictor_smoothing::Bool = true,
                  mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                  mask_threshold::Real = 0.5,
                  warp_buffers = nothing,
                  workspace::Union{Nothing,PIVWorkspace} = nothing)
    # Pipeline precision follows the images; every per-pass array shares it.
    T = float(promote_type(eltype(imgA), eltype(imgB)))
    grid = pass_grid(T, size(imgA), params, mask, mask_threshold)
    ny, nx = length(grid.y), length(grid.x)
    # Reuse the deformation output buffers across passes when supplied (each
    # sweep overwrites them fully before use); allocate fresh otherwise.
    bufA = warp_buffers === nothing ? nothing : warp_buffers[1]
    bufB = warp_buffers === nothing ? nothing : warp_buffers[2]

    maxiter = params.max_iterations
    # Per-sweep output arrays, allocated once and reused: process_windows!
    # rewrites every job cell each sweep and the masked cells are reset below,
    # so no stale values survive an iteration.
    residual_u = zeros(T, ny, nx)
    residual_v = zeros(T, ny, nx)
    peak_ratio = zeros(T, ny, nx)
    correlation_moment = zeros(T, ny, nx)
    n_alt = params.n_peaks - 1
    alt_u = n_alt > 0 ? fill(T(NaN), ny, nx, n_alt) : nothing
    alt_v = n_alt > 0 ? fill(T(NaN), ny, nx, n_alt) : nothing
    # Wieneke (2015) uncertainty assumes the correlation peak of the deformed
    # windows sits at ~zero residual, so it is only estimated on the final
    # pass (force_replace marks intermediate passes). NaN when disabled. An
    # iterating stage estimates it once after the loop settles instead of
    # every sweep — the estimator reads only the deformed windows, never the
    # correlation plane, so the post-loop sweep yields the exact values the
    # fused path would.
    unc = params.uncertainty && !force_replace
    uncertainty_u = fill(T(NaN), ny, nx)
    uncertainty_v = fill(T(NaN), ny, nx)
    fused_unc = unc && maxiter == 1
    # Opt-in full-plane storage: one cell per window (masked/skipped windows,
    # which are absent from grid.jobs, stay `nothing`). Each job writes only
    # its own cell, so the threaded path needs no locking.
    planes = params.keep_correlation_planes ?
             fill!(Matrix{Union{Nothing,Matrix{T}}}(undef, ny, nx), nothing) : nothing

    nchunks = threaded ? min(Threads.nthreads(), length(grid.jobs)) : 1
    # One correlator per chunk (pooled and reused across pairs when a workspace
    # is supplied); each chunk uses a distinct entry, so the fan-out is safe.
    correlators = piv_correlators(workspace, params, T, nchunks)
    chunk_size = cld(length(grid.jobs), max(nchunks, 1))

    # Sweeps always replace flagged vectors (the next predictor must be well
    # behaved). When the pass semantics don't want replacement, the measured
    # field is stashed each sweep and restored at still-flagged cells after
    # the loop — exactly the unreplaced output, since peak-substituted cells
    # hold measured data and are unflagged.
    keep_measured = maxiter > 1 && !(force_replace || params.replace_outliers)
    meas_u = meas_v = nothing
    prev_u = prev_v = nothing
    change_buf = Vector{T}()
    maxiter > 2 && params.convergence_tol > 0 && sizehint!(change_buf, ny * nx)
    local result, warpA, warpB
    for it in 1:maxiter
        warpA, warpB, u, v = apply_predictor(imgA, imgB, itpA, itpB, predictor,
                                             grid.x, grid.y, T; threaded,
                                             warpA = bufA, warpB = bufB)
        if it > 1
            fill!(residual_u, zero(T))
            fill!(residual_v, zero(T))
            alt_u === nothing || fill!(alt_u, T(NaN))
            alt_v === nothing || fill!(alt_v, T(NaN))
        end
        if nchunks == 1
            process_windows!(residual_u, residual_v, peak_ratio, correlation_moment,
                             alt_u, alt_v, fused_unc ? uncertainty_u : nothing,
                             fused_unc ? uncertainty_v : nothing, grid.jobs, warpA,
                             warpB, params, correlators[1], mask, planes)
        elseif nchunks > 1
            @sync for (ci, chunk) in enumerate(Iterators.partition(grid.jobs, chunk_size))
                Threads.@spawn process_windows!(residual_u, residual_v, peak_ratio,
                                                correlation_moment, alt_u, alt_v,
                                                fused_unc ? uncertainty_u : nothing,
                                                fused_unc ? uncertainty_v : nothing,
                                                chunk, warpA, warpB, params,
                                                correlators[ci], mask, planes)
            end
        end
        if alt_u !== nothing
            # Alternatives are residuals in the deformed frame; add the shared
            # predictor (u/v still hold it here) to make them total displacements.
            alt_u .+= u
            alt_v .+= v
        end
        u .+= residual_u
        v .+= residual_v
        if any(grid.grid_mask)
            for f in (u, v, peak_ratio, correlation_moment)
                f[grid.grid_mask] .= T(NaN)
            end
        end
        if keep_measured
            meas_u, meas_v = copy(u), copy(v)
        end
        result = PIVResult(grid.x, grid.y, u, v, peak_ratio, correlation_moment,
                           uncertainty_u, uncertainty_v,
                           falses(ny, nx), grid.grid_mask, params, planes)
        validate_and_replace!(result, params, force_replace || maxiter > 1;
                              alternatives = alt_u === nothing ? nothing : (alt_u, alt_v))
        it == maxiter && break
        # convergence_tol = 0 disables the early exit; skip the change
        # tracking entirely then.
        if params.convergence_tol > 0
            if prev_u === nothing
                prev_u, prev_v = copy(u), copy(v)
            else
                change = field_change(change_buf, u, v, prev_u, prev_v, grid.grid_mask)
                change < params.convergence_tol && break
                copyto!(prev_u, u)
                copyto!(prev_v, v)
            end
        end
        predictor = build_predictor(result, predictor_smoothing)
    end

    if unc && !fused_unc
        # warpA/warpB still hold the deformation of the last sweep — the
        # windows whose correlation produced the returned field.
        if nchunks == 1
            uncertainty_sweep!(uncertainty_u, uncertainty_v, grid.jobs,
                               warpA, warpB, params, correlators[1].apod, mask)
        elseif nchunks > 1
            @sync for (ci, chunk) in enumerate(Iterators.partition(grid.jobs, chunk_size))
                Threads.@spawn uncertainty_sweep!(uncertainty_u, uncertainty_v, chunk,
                                                  warpA, warpB, params,
                                                  correlators[ci].apod, mask)
            end
        end
    end
    if keep_measured && any(result.outliers)
        result.u[result.outliers] = meas_u[result.outliers]
        result.v[result.outliers] = meas_v[result.outliers]
    end
    return result
end

# The convergence norm quantile: an iterating stage is converged once this
# fraction of its unmasked vectors changed by less than `convergence_tol`
# since the previous sweep. A max-norm never converges in practice: a small
# minority of bistable low-signal windows flickers between correlation peaks
# indefinitely (measured on synthetic scenes: the max change stays ~1 px for
# arbitrarily many sweeps while the median falls below 1e-3 px), and those
# windows are validation's problem, not the iteration's. The 95th percentile
# tracks the bulk field and ignores that persistent minority.
const CONVERGENCE_QUANTILE = 0.95

# 95th-percentile displacement-component change of the unmasked vectors
# between two successive sweeps of an iterating stage. NaN-safe: cells that
# are NaN in both sweeps (e.g. unreplaceable) contribute nothing, while a
# cell flipping between NaN and finite counts as an infinite change. `buf` is
# reusable scratch (emptied here; consumed by the quantile's partial sort).
function field_change(buf::Vector{T}, u, v, prev_u, prev_v, mask) where {T}
    empty!(buf)
    @inbounds for i in eachindex(u, v, prev_u, prev_v)
        mask[i] && continue
        du = abs(u[i] - prev_u[i])
        dv = abs(v[i] - prev_v[i])
        if isnan(du) || isnan(dv)
            if isnan(u[i]) == isnan(prev_u[i]) && isnan(v[i]) == isnan(prev_v[i])
                continue
            end
            push!(buf, T(Inf))
            continue
        end
        push!(buf, max(du, dv))
    end
    isempty(buf) && return zero(T)
    return T(quantile!(buf, CONVERGENCE_QUANTILE))
end

# Estimate the Wieneke (2015) per-window uncertainty for `jobs` from the
# deformed image pair, exactly as the fused path inside process_windows!
# would (the estimator reads only the deformed windows, never the correlation
# plane). Iterating stages call this once after the loop settles, so the
# estimate is paid once instead of every sweep.
function uncertainty_sweep!(uncertainty_u, uncertainty_v, jobs,
                            imgA::AbstractMatrix, imgB::AbstractMatrix,
                            params::PIVParameters, apod::AbstractMatrix,
                            mask::Union{Nothing,AbstractMatrix{Bool}} = nothing)
    wr, wc = params.window_size
    T = float(promote_type(eltype(imgA), eltype(imgB)))
    uscratch = uncertainty_scratch(T, params.window_size)
    ustats = zeros(2, UQ_NSTATS)
    for (gi, gj, rs, cs) in jobs
        subA = @view imgA[rs:(rs + wr - 1), cs:(cs + wc - 1)]
        subB = @view imgB[rs:(rs + wr - 1), cs:(cs + wc - 1)]
        submask = mask === nothing ? nothing :
                  view(mask, rs:(rs + wr - 1), cs:(cs + wc - 1))
        # Fully clean windows take the unmasked fast path.
        submask !== nothing && !any(submask) && (submask = nothing)
        fill!(ustats, 0.0)
        accumulate_uncertainty!(ustats, uscratch, subA, subB, submask, apod)
        uncertainty_u[gi, gj] = finalize_uncertainty(T, view(ustats, 1, :))
        uncertainty_v[gi, gj] = finalize_uncertainty(T, view(ustats, 2, :))
    end
    return nothing
end

"""
    image_interpolant(img, ::Type{T}) -> extrapolation

Build the cubic B-spline resampler (`T`-typed, zero-extrapolated) that
[`deform_images`](@ref) samples. The prefilter is the expensive part, and it
depends only on the source image, so multipass [`run_piv`](@ref) builds this
once per image and reuses it across every deforming pass.
"""
function image_interpolant(img::AbstractMatrix, ::Type{T}) where {T}
    # `interpolate` copies its input into the (padded) coefficient array during
    # prefiltering, so it never mutates the source — pass `img` straight through
    # when it is already `T`, skipping a full-image copy per pass.
    extrapolate(interpolate(eltype(img) === T ? img : T.(img),
                            BSpline(Cubic(Line(OnGrid())))), zero(T))
end

"""
    deform_images(itpA, itpB, itp_u, itp_v, imgsize, ::Type{T}; threaded = false) -> (warpedA, warpedB)

Symmetric (central-difference) image deformation: each `imgsize` output is
resampled from its prebuilt cubic B-spline image interpolant
([`image_interpolant`](@ref)) — `itpA` shifted by −d/2 and `itpB` by +d/2,
where `d = (itp_u, itp_v)` is the displacement field evaluated at each pixel.
After deformation, content displaced by exactly `d` is aligned in both
outputs, so correlating them measures the residual displacement. Out-of-image
samples are zero-filled (a property of the passed interpolants).

Interpolant evaluation is a pure read, so with `threaded = true` the output
columns are filled concurrently.

`warpA`/`warpB` optionally supply the output buffers (each `imgsize`,
element type `T`); when `nothing` they are freshly allocated. Multipass
[`run_piv`](@ref) allocates them once and reuses them across every deforming
pass, since each pass fully overwrites them before use.
"""
function deform_images(itpA, itpB, itp_u, itp_v, imgsize::Dims{2}, ::Type{T};
                       threaded::Bool = false,
                       warpA::Union{Nothing,Matrix{T}} = nothing,
                       warpB::Union{Nothing,Matrix{T}} = nothing) where {T}
    nr, nc = imgsize
    warpA = warpA === nothing ? Matrix{T}(undef, nr, nc) : warpA
    warpB = warpB === nothing ? Matrix{T}(undef, nr, nc) : warpB
    if threaded && Threads.nthreads() > 1 && nc > 1
        nchunks = min(Threads.nthreads(), nc)
        chunk_size = cld(nc, nchunks)
        @sync for cols in Iterators.partition(1:nc, chunk_size)
            Threads.@spawn deform_columns!(warpA, warpB, itpA, itpB, itp_u, itp_v, cols, nr)
        end
    else
        deform_columns!(warpA, warpB, itpA, itpB, itp_u, itp_v, 1:nc, nr)
    end
    return warpA, warpB
end

# Fill a column range of the deformed pair. Each column is written by exactly
# one task, so the threaded fan-out over disjoint column ranges is race-free.
function deform_columns!(warpA, warpB, itpA, itpB, itp_u, itp_v, cols, nr)
    @inbounds for c in cols, r in 1:nr
        du2 = itp_u(r, c) / 2
        dv2 = itp_v(r, c) / 2
        warpA[r, c] = itpA(r - dv2, c - du2)
        warpB[r, c] = itpB(r + dv2, c + du2)
    end
    return nothing
end

make_correlator(params::PIVParameters, ::Type{T}) where {T} =
    params.correlation_method === :cross ?
        CrossCorrelator{T}(params.window_size; params.padding, params.apodization) :
        PhaseCorrelator{T}(params.window_size; params.padding, params.apodization)

# The caller supplies this chunk's correlator (one per chunk), plus per-call
# peak scratch, so chunks can run on concurrent tasks. Every (gi, gj) is written
# by exactly one job and the per-window math doesn't depend on chunking or on
# which correlator instance runs it, so threaded results match serial results
# exactly.
function process_windows!(u, v, peak_ratio, correlation_moment, alt_u, alt_v,
                          uncertainty_u, uncertainty_v, jobs,
                          imgA::AbstractMatrix, imgB::AbstractMatrix, params::PIVParameters,
                          correlator::Correlator,
                          mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                          planes = nothing)
    wr, wc = params.window_size
    T = float(promote_type(eltype(imgA), eltype(imgB)))
    k = max(params.n_peaks, 2)
    vals = Vector{T}(undef, k)
    locs = Vector{NTuple{2,Int}}(undef, k)
    uscratch = uncertainty_u === nothing ? nothing : uncertainty_scratch(T, params.window_size)
    ustats = uncertainty_u === nothing ? nothing : zeros(2, UQ_NSTATS)
    for (gi, gj, rs, cs) in jobs
        subA = @view imgA[rs:(rs + wr - 1), cs:(cs + wc - 1)]
        subB = @view imgB[rs:(rs + wr - 1), cs:(cs + wc - 1)]
        submask = mask === nothing ? nothing :
                  view(mask, rs:(rs + wr - 1), cs:(cs + wc - 1))
        # Fully clean windows take the unmasked fast path.
        submask !== nothing && !any(submask) && (submask = nothing)
        R = correlation_plane!(correlator, subA, subB, submask)
        # R is an aliased internal buffer, overwritten by the next window —
        # copy before retaining it.
        planes === nothing || (planes[gi, gj] = copy(R))
        res = analyze_plane!(vals, locs, R, params)
        u[gi, gj] = res.du
        v[gi, gj] = res.dv
        peak_ratio[gi, gj] = res.ratio
        correlation_moment[gi, gj] = res.moment
        if alt_u !== nothing
            # Alternatives use the cheap 3-point fit regardless of the
            # primary's subpixel method — they are fallback candidates.
            for m in 2:min(res.found, params.n_peaks)
                aref = subpixel_gauss3(R, locs[m])
                alt_u[gi, gj, m - 1] = aref[2] - res.center[2]
                alt_v[gi, gj, m - 1] = aref[1] - res.center[1]
            end
        end
        if uncertainty_u !== nothing
            fill!(ustats, 0.0)
            accumulate_uncertainty!(ustats, uscratch, subA, subB, submask,
                                    correlator.apod)
            uncertainty_u[gi, gj] = finalize_uncertainty(T, view(ustats, 1, :))
            uncertainty_v[gi, gj] = finalize_uncertainty(T, view(ustats, 2, :))
        end
    end
    return nothing
end
