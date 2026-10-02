# Stereoscopic 3C reconstruction (Phase 5, slice 3): the 2D engine runs per
# camera on images dewarped to the common world plane, then the two 2C fields
# combine into (u, v, w) by geometric least squares.
#
# The per-point reconstruction (like the calibration it rests on) works in
# Float64 and converts on store: it is O(vector grid) per pair, far from the
# per-pixel hot path that the precision-follows-images rule protects.

"""
    StereoPIVResult{T<:AbstractFloat}

Three-component displacement field from [`run_piv_stereo`](@ref). `T`
follows the input image precision. Coordinates and displacements use the
[`DewarpGrid`](@ref)'s world length unit until [`physical`](@ref) converts
displacements to velocity.

# Fields
- `x`, `y`: world coordinates of the vector grid (`x` along the dewarp grid's
  `x` range, `y` along its `y` range, in the grid's world units, e.g. mm).
- `z`: out-of-plane position of the measurement plane (the `DewarpGrid`'s `z`).
- `u`, `v`, `w`: displacement components on the `(length(y), length(x))`
  grid, in world units per frame interval: `u` along world X, `v` along
  world Y (signs follow the dewarp grid's ranges), `w` along world +Z.
- `uncertainty_u`, `uncertainty_v`, `uncertainty_w`: per-vector measurement
  uncertainty (one standard deviation, world units) propagated from both
  camera estimates. They are `NaN` unless uncertainty was enabled in the
  PIV parameters.
- `outliers`: union of the two cameras' outlier flags. Flagged vectors were
  reconstructed from at least one flagged 2C vector, which may have been
  replaced. An alternative peak accepted by validation is not flagged.
- `mask`: windows dropped because they overlap either camera's out-of-view
  region or the user mask (`NaN` fields, never outliers).
- `cam1`, `cam2`: the per-camera 2C [`PIVResult`](@ref)s on the dewarped
  images (displacements in dewarped pixels), retained for diagnostics.
- `parameters`: the `PIVParameters` of the (final) pass.
- `scale`: [`PhysicalScale`](@ref) metadata attached with `scale` or
  [`with_scale`](@ref); `nothing` when none was attached. For stereo, set
  `dt` and unit labels, leaving `pixel_size = 1` because arrays are already
  in world units. [`physical`](@ref) converts the stereo arrays; `cam1`
  and `cam2` remain in dewarped pixels.
"""
struct StereoPIVResult{T<:AbstractFloat}
    x::Vector{T}
    y::Vector{T}
    z::Float64
    u::Matrix{T}
    v::Matrix{T}
    w::Matrix{T}
    uncertainty_u::Matrix{T}
    uncertainty_v::Matrix{T}
    uncertainty_w::Matrix{T}
    outliers::BitMatrix
    mask::BitMatrix
    cam1::PIVResult{T}
    cam2::PIVResult{T}
    parameters::PIVParameters
    scale::Union{Nothing,PhysicalScale}
end

# Backward-compatible constructors: no physical scale stored. Keep the
# positional 14-argument call sites (reconstruct_stereo history, GUI test
# fixture) valid, in both the inferred and explicit `{T}` forms.
StereoPIVResult(x, y, z, u, v, w, uncertainty_u, uncertainty_v, uncertainty_w,
                outliers, mask, cam1::PIVResult{T}, cam2::PIVResult{T},
                parameters::PIVParameters) where {T} =
    StereoPIVResult{T}(x, y, z, u, v, w, uncertainty_u, uncertainty_v, uncertainty_w,
                       outliers, mask, cam1, cam2, parameters, nothing)

StereoPIVResult{T}(x, y, z, u, v, w, uncertainty_u, uncertainty_v, uncertainty_w,
                   outliers, mask, cam1, cam2, parameters) where {T} =
    StereoPIVResult{T}(x, y, z, u, v, w, uncertainty_u, uncertainty_v, uncertainty_w,
                       outliers, mask, cam1, cam2, parameters, nothing)

function Base.show(io::IO, r::StereoPIVResult{T}) where {T}
    ny, nx = size(r.u)
    print(io, "StereoPIVResult{$T}($(nx)×$(ny) grid, $(sum(r.outliers)) outliers",
          any(r.mask) ? ", $(sum(r.mask)) masked)" : ")")
end

# d(X, Y)/dZ along the viewing ray through the world point (X, Y, z): how far
# the point seen at a fixed pixel drifts in-plane per unit Z. Solved from the
# forward map's Jacobian (central differences with step δ): at a fixed pixel,
# J_XY * [dX/dZ, dY/dZ] = -dp/dZ. Returns NaNs for degenerate (edge-on)
# viewing geometry.
function ray_slopes(cam::CameraCalibration, X::Real, Y::Real, z::Real, δ::Real)
    dpX = (world_to_pixel(cam, (X + δ, Y, z)) - world_to_pixel(cam, (X - δ, Y, z))) / (2δ)
    dpY = (world_to_pixel(cam, (X, Y + δ, z)) - world_to_pixel(cam, (X, Y - δ, z))) / (2δ)
    dpZ = (world_to_pixel(cam, (X, Y, z + δ)) - world_to_pixel(cam, (X, Y, z - δ))) / (2δ)
    J = @SMatrix [dpX[1] dpY[1]; dpX[2] dpY[2]]
    abs(det(J)) > 1e-12 || return SVector(NaN, NaN)
    return -(J \ dpZ)
end

"""
    run_piv_stereo(A1, B1, A2, B2, dw1, dw2,
                   params = PIVParameters(); mask = nothing, kwargs...)
        -> StereoPIVResult
    run_piv_stereo(A1, B1, A2, B2, dw1, dw2;
                   effort = :low/:medium/:high, mask = nothing, kwargs...)

Reconstruct three displacement components from two synchronized camera
pairs. `A1`/`B1` belong to camera 1, `A2`/`B2` to camera 2. Their
[`ImageDewarper`](@ref)s must share a [`DewarpGrid`](@ref). The driver
dewarps both pairs, runs planar PIV on each, then combines the two fields.
Pass a `PIVParameters` value or pass schedule; alternatively, use an
`effort` preset. The returned `(u, v, w)` are in world length units per
image-pair interval, with no time scaling applied.

The four measured in-plane components constrain three world displacements
through the calibrated viewing rays. Reconstruction uses unweighted least
squares; degenerate camera geometry yields `NaN` components. With
uncertainty enabled in the PIV parameters, per-camera correlation estimates
propagate into `uncertainty_u`, `uncertainty_v`, and `uncertainty_w`, assuming independent
camera measurement errors. Geometry and image quality affect how much
uncertainty appears in each component.

`mask` is an optional grid-sized `Bool` matrix of world-plane pixels to
exclude (`true` = excluded); it is combined with the dewarpers' out-of-view
masks (`dw1.mask .| dw2.mask`), so only the stereo overlap region is
analyzed. `scale` attaches a [`PhysicalScale`](@ref) to the stereo result
(not to the per-camera results): set `dt` and the unit labels, leaving
`pixel_size = 1` because the stereo fields are already in world units.
[`physical`](@ref) only needs to divide by the frame interval. Remaining
keyword arguments (`threaded`, `predictor_smoothing`, `backend`,
`mask_threshold`) are forwarded to [`run_piv`](@ref). A GPU backend accelerates
the two per-camera PIV analyses; dewarping and reconstruction remain on the
CPU (see [Run PIV on a GPU](@ref)).

The returned [`StereoPIVResult`](@ref) retains both per-camera fields and
combines their mask and outlier flags. Inspect those fields when one camera
has weak seeding or a larger uncertainty estimate.

Matrix inputs contain no exposure timestamps: the caller must establish
synchronization. To validate timestamped [`FrameRef`](@ref)s before loading
images, use [`run_piv_stereo_sequence`](@ref).
"""
function run_piv_stereo(A1::AbstractMatrix{<:Real}, B1::AbstractMatrix{<:Real},
                        A2::AbstractMatrix{<:Real}, B2::AbstractMatrix{<:Real},
                        dw1::ImageDewarper, dw2::ImageDewarper,
                        params::Union{PIVParameters,AbstractVector{PIVParameters}};
                        effort::Union{Nothing,Symbol} = nothing,
                        backend::Symbol = :cpu,
                        mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                        scale::Union{Nothing,PhysicalScale} = nothing,
                        kwargs...)
    effort === nothing ||
        throw(ArgumentError("effort cannot be combined with explicit PIVParameters or pass schedules"))
    # Check the backend's option scope before the (relatively expensive)
    # dewarps; the per-camera run_piv calls would reject it later anyway.
    _check_backend_params(_resolve_backend(backend),
                          params isa PIVParameters ? [params] : params)
    dw1.grid == dw2.grid ||
        throw(ArgumentError("the two dewarpers must share the same DewarpGrid, " *
                            "got $(dw1.grid) and $(dw2.grid)"))
    grid = dw1.grid
    node_mask = dw1.mask .| dw2.mask
    if mask !== nothing
        size(mask) == size(grid) ||
            throw(DimensionMismatch("mask size $(size(mask)) does not match the " *
                                    "dewarp grid size $(size(grid))"))
        node_mask .|= mask
    end

    T = float(promote_type(eltype(A1), eltype(B1), eltype(A2), eltype(B2)))
    a = Matrix{T}(undef, size(grid))
    b = Matrix{T}(undef, size(grid))
    result = _run_piv_stereo!(a, b, A1, B1, A2, B2, dw1, dw2, params,
                              node_mask; backend, kwargs...)
    return scale === nothing ? result : with_scale(result, scale)
end

function run_piv_stereo(A1::AbstractMatrix{<:Real}, B1::AbstractMatrix{<:Real},
                        A2::AbstractMatrix{<:Real}, B2::AbstractMatrix{<:Real},
                        dw1::ImageDewarper, dw2::ImageDewarper;
                        effort::Union{Nothing,Symbol} = nothing,
                        backend::Symbol = :cpu,
                        mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                        kwargs...)
    if effort === nothing
        return run_piv_stereo(A1, B1, A2, B2, dw1, dw2, PIVParameters();
                              backend, mask, kwargs...)
    end
    dw1.grid == dw2.grid ||
        throw(ArgumentError("the two dewarpers must share the same DewarpGrid, " *
                            "got $(dw1.grid) and $(dw2.grid)"))
    piv_kwargs, driver_kwargs = split_effort_kwargs(kwargs)
    passes = effort_schedule(effort; image_size = size(dw1.grid), piv_kwargs...)
    return run_piv_stereo(A1, B1, A2, B2, dw1, dw2, passes;
                          backend, mask, driver_kwargs...)
end

# Allocation-controllable core used by the sequence driver. One dewarped pair
# buffer and one stateful PIV workspace are sufficient because the two camera
# analyses are deliberately serial; both are reused across cameras and pairs.
function _run_piv_stereo!(a, b, A1, B1, A2, B2, dw1, dw2, params, node_mask;
                          backend::Symbol, workspace = nothing, kwargs...)
    dewarp!(a, dw1, A1)
    dewarp!(b, dw1, B1)
    r1 = run_piv(a, b, params; backend, workspace, mask = node_mask, kwargs...)
    dewarp!(a, dw2, A2)
    dewarp!(b, dw2, B2)
    r2 = run_piv(a, b, params; backend, workspace, mask = node_mask, kwargs...)
    return reconstruct_stereo(r1, r2, dw1.cam, dw2.cam, dw1.grid)
end

function _stereo_node_mask(dw1, dw2, mask)
    dw1.grid == dw2.grid ||
        throw(ArgumentError("the two dewarpers must share the same DewarpGrid, " *
                            "got $(dw1.grid) and $(dw2.grid)"))
    node_mask = dw1.mask .| dw2.mask
    if mask !== nothing
        size(mask) == size(dw1.grid) ||
            throw(DimensionMismatch("mask size $(size(mask)) does not match the " *
                                    "dewarp grid size $(size(dw1.grid))"))
        node_mask .|= mask
    end
    return node_mask
end


"""
    run_piv_stereo_sequence(pairs1, pairs2, dw1, dw2, params = PIVParameters(); kwargs...)
    run_piv_stereo_sequence(acquisitions, dw1, dw2, params = PIVParameters(); kwargs...)

Process synchronized stereo image pairs. `pairs1` and `pairs2` must have
the same number of entries, with each camera's pair in the format accepted
by [`run_piv_sequence`](@ref). Alternatively, pass `acquisitions` as
4-tuples `(A1, B1, A2, B2)`. Dewarping buffers and a PIV workspace are
reused across acquisitions.

`preprocess` may be one function shared by both cameras or a tuple
`(preprocess1, preprocess2)`; each hook runs after loading and before
dewarping. `output` is either one incrementally written JLD2 path or a
function `(i, acquisition) -> path` for per-acquisition files. Four source
paths are recorded when all four inputs are paths. `progress` and
`on_result` have the same callback contracts as [`run_piv_sequence`](@ref)
(the latter receives each [`StereoPIVResult`](@ref) as it completes).
`cancel` may be a zero-argument predicate; when it becomes true, processing
stops between acquisitions and the completed prefix is returned (and remains
persisted).
Set `collect_results = false` to return `nothing` instead of retaining the
completed results in memory, including on cancellation. `on_result`, `output`,
and `progress` still run for each completed acquisition in the same order.
Timestamped [`FramePair`](@ref)s attach their actual pair-specific `dt` to
each result when `scale` is supplied; the two cameras' intervals must agree.

Before loading any images or opening `output`, matching A exposures and
matching B exposures are checked using [`FrameRef`](@ref) source timestamps.
Both camera clocks must use the same origin and time unit. The tolerance is
`sync_atol + sync_rtol * max(dt1, dt2)`, using the available positive pair
delays (zero if neither is available), never the absolute clock epoch.
`sync_atol = 0.0` and `sync_rtol = 0.0` require exact agreement; both must be
finite and nonnegative. The same tolerance applies to the two pair delays.
Available timestamps must be finite real numbers, and available delays must
be finite and positive. Equal delays alone do not establish synchronization.
When exposure timestamps and `FramePair.dt` are both supplied, the observed
delay takes precedence for checking synchronization and must agree with the
declared `dt` within the same tolerance.

With `missing_timestamps = :allow` (default), missing exposure timestamps
(`nothing` or `missing`) skip only their matching-exposure comparison; all
available comparisons still run. This preserves matrix/path workflows but
does not verify synchronization when metadata is absent. Use
`missing_timestamps = :error` to require all four timestamps in every
acquisition. This policy also applies to the 4-tuple overload.

Pass explicit `params`, or omit them and use `effort = :low`, `:medium`, or
`:high`. Other keywords are forwarded to [`run_piv_stereo`](@ref).
"""
function run_piv_stereo_sequence(pairs1::AbstractVector, pairs2::AbstractVector,
                                 dw1::ImageDewarper, dw2::ImageDewarper,
                                 params::Union{PIVParameters,AbstractVector{PIVParameters}};
                                 effort::Union{Nothing,Symbol} = nothing,
                                 sync_atol::Real = 0.0, sync_rtol::Real = 0.0,
                                 missing_timestamps::Symbol = :allow, kwargs...)
    effort === nothing ||
        throw(ArgumentError("effort cannot be combined with explicit PIVParameters or pass schedules"))
    length(pairs1) == length(pairs2) ||
        throw(DimensionMismatch("camera pair sequences must have equal length, got " *
                                "$(length(pairs1)) and $(length(pairs2))"))
    _check_stereo_pair_times(pairs1, pairs2; sync_atol, sync_rtol, missing_timestamps)
    acquisitions = [(p1[1], p1[2], p2[1], p2[2]) for (p1, p2) in zip(pairs1, pairs2)]
    return _run_piv_stereo_sequence(acquisitions, dw1, dw2, params;
                                    scale_pairs = pairs1, sync_atol, sync_rtol,
                                    missing_timestamps, kwargs...)
end

function run_piv_stereo_sequence(pairs1::AbstractVector, pairs2::AbstractVector,
                                 dw1::ImageDewarper, dw2::ImageDewarper;
                                 effort::Union{Nothing,Symbol} = nothing,
                                 sync_atol::Real = 0.0, sync_rtol::Real = 0.0,
                                 missing_timestamps::Symbol = :allow, kwargs...)
    length(pairs1) == length(pairs2) ||
        throw(DimensionMismatch("camera pair sequences must have equal length, got " *
                                "$(length(pairs1)) and $(length(pairs2))"))
    _check_stereo_pair_times(pairs1, pairs2; sync_atol, sync_rtol, missing_timestamps)
    acquisitions = [(p1[1], p1[2], p2[1], p2[2]) for (p1, p2) in zip(pairs1, pairs2)]
    if effort === nothing
        return _run_piv_stereo_sequence(acquisitions, dw1, dw2, PIVParameters();
                                        scale_pairs = pairs1, sync_atol, sync_rtol,
                                        missing_timestamps, kwargs...)
    end
    piv_kwargs, driver_kwargs = split_effort_kwargs(kwargs)
    passes = effort_schedule(effort; image_size = size(dw1.grid), piv_kwargs...)
    return _run_piv_stereo_sequence(acquisitions, dw1, dw2, passes;
                                    scale_pairs = pairs1, sync_atol, sync_rtol,
                                    missing_timestamps, driver_kwargs...)
end

# FrameRef is included later, so resolve its type only when this helper runs.
function _stereo_frame_time(frame)
    t = frame isa FrameRef ? frame_timestamp(frame.source, frame.index) : nothing
    return ismissing(t) ? nothing : t
end

function _stereo_pair_delay(pair, ta, tb, i, camera)
    declared = hasproperty(pair, :dt) ? getproperty(pair, :dt) : nothing
    ismissing(declared) && (declared = nothing)
    observed = nothing
    if ta !== nothing && tb !== nothing
        tb > ta || throw(ArgumentError("camera $camera pair $i must have increasing exposure timestamps"))
        observed = tb - ta
    end
    for dt in (declared, observed)
        dt === nothing && continue
        dt isa Real && isfinite(dt) && dt > 0 ||
            throw(ArgumentError("camera $camera pair $i must have a finite positive frame interval, got $dt"))
    end
    return observed === nothing ? declared : observed, declared
end

function _check_stereo_pair_times(pairs1, pairs2;
                                   sync_atol::Real = 0.0, sync_rtol::Real = 0.0,
                                   missing_timestamps::Symbol = :allow)
    isfinite(sync_atol) && sync_atol >= 0 ||
        throw(ArgumentError("sync_atol must be finite and nonnegative"))
    isfinite(sync_rtol) && sync_rtol >= 0 ||
        throw(ArgumentError("sync_rtol must be finite and nonnegative"))
    missing_timestamps in (:allow, :error) ||
        throw(ArgumentError("missing_timestamps must be :allow or :error"))
    for (i, (p1, p2)) in enumerate(zip(pairs1, pairs2))
        times = (_stereo_frame_time(p1[1]), _stereo_frame_time(p1[2]),
                 _stereo_frame_time(p2[1]), _stereo_frame_time(p2[2]))
        for t in times
            t === nothing && continue
            t isa Real && isfinite(t) ||
                throw(ArgumentError("camera pair $i has a nonfinite or non-real exposure timestamp: $t"))
        end
        missing_timestamps === :error && any(isnothing, times) &&
            throw(ArgumentError("camera pair $i is missing exposure timestamps; synchronization cannot be verified"))
        dt1, declared1 = _stereo_pair_delay(p1, times[1], times[2], i, 1)
        dt2, declared2 = _stereo_pair_delay(p2, times[3], times[4], i, 2)
        delay = max(dt1 === nothing ? 0 : dt1, dt2 === nothing ? 0 : dt2)
        tol = sync_atol + sync_rtol * delay
        isfinite(tol) || throw(ArgumentError("camera pair $i synchronization tolerance is nonfinite"))
        for (camera, dt, declared) in ((1, dt1, declared1), (2, dt2, declared2))
            if dt !== nothing && declared !== nothing && abs(dt - declared) > tol
                throw(ArgumentError("camera $camera pair $i has a declared frame interval $declared " *
                                    "inconsistent with exposure timestamps (observed $dt, tolerance $tol)"))
            end
        end
        if dt1 !== nothing && dt2 !== nothing && abs(dt1 - dt2) > tol
            throw(ArgumentError("camera pair $i has inconsistent frame intervals: " *
                                "$dt1 and $dt2"))
        end
        for (k, exposure) in ((1, "A"), (2, "B"))
            t1, t2 = times[k], times[k + 2]
            if t1 !== nothing && t2 !== nothing && abs(t1 - t2) > tol
                throw(ArgumentError("camera pair $i has unsynchronized $exposure exposures: " *
                                    "$t1 and $t2 (tolerance $tol)"))
            end
        end
    end
    return nothing
end

function run_piv_stereo_sequence(acquisitions::AbstractVector,
                                 dw1::ImageDewarper, dw2::ImageDewarper,
                                 params::Union{PIVParameters,AbstractVector{PIVParameters}};
                                 effort::Union{Nothing,Symbol} = nothing, kwargs...)
    effort === nothing ||
        throw(ArgumentError("effort cannot be combined with explicit PIVParameters or pass schedules"))
    return _run_piv_stereo_sequence(acquisitions, dw1, dw2, params; kwargs...)
end

function run_piv_stereo_sequence(acquisitions::AbstractVector,
                                 dw1::ImageDewarper, dw2::ImageDewarper;
                                 effort::Union{Nothing,Symbol} = nothing, kwargs...)
    if effort === nothing
        return _run_piv_stereo_sequence(acquisitions, dw1, dw2, PIVParameters(); kwargs...)
    end
    piv_kwargs, driver_kwargs = split_effort_kwargs(kwargs)
    passes = effort_schedule(effort; image_size = size(dw1.grid), piv_kwargs...)
    return _run_piv_stereo_sequence(acquisitions, dw1, dw2, passes; driver_kwargs...)
end

function _run_piv_stereo_sequence(acquisitions, dw1, dw2, params;
                                  backend::Symbol = :cpu,
                                  preprocess = nothing,
                                  output::Union{Nothing,AbstractString,Function} = nothing,
                                  progress::Union{Bool,Function} = true,
                                  on_result::Union{Nothing,Function} = nothing,
                                  collect_results::Bool = true,
                                  cancel::Union{Nothing,Function} = nothing,
                                  image_type::Type{<:AbstractFloat} = Float64,
                                  mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                                  scale::Union{Nothing,PhysicalScale} = nothing,
                                  scale_pairs = nothing,
                                  sync_atol::Real = 0.0, sync_rtol::Real = 0.0,
                                  missing_timestamps::Symbol = :allow,
                                  kwargs...)
    isempty(acquisitions) && throw(ArgumentError("acquisitions must not be empty"))
    all(a -> a isa Tuple && length(a) == 4, acquisitions) ||
        throw(ArgumentError("each stereo acquisition must be a 4-tuple (A1, B1, A2, B2)"))
    _check_stereo_pair_times(((a[1], a[2]) for a in acquisitions),
                             ((a[3], a[4]) for a in acquisitions);
                             sync_atol, sync_rtol, missing_timestamps)
    node_mask = _stereo_node_mask(dw1, dw2, mask)
    pre1, pre2 = preprocess isa Tuple && length(preprocess) == 2 ? preprocess :
                 (preprocess, preprocess)
    workspace = piv_workspace(; backend)
    results = collect_results ? StereoPIVResult[] : nothing
    a = b = nothing
    file = output isa AbstractString ? jldopen(output, "w") : nothing
    load_acquisition(acq) = Threads.@spawn begin
        frames = ntuple(4) do k
            img = load_frame(acq[k], image_type)
            pre = k <= 2 ? pre1 : pre2
            pre === nothing ? img : pre(img)
        end
        T = float(promote_type(map(eltype, frames)...))
        return frames, T
    end
    pending = nothing
    failed = false
    cancelled = false
    try
        file === nothing || (file["format_version"] = RESULTS_FORMAT_VERSION)
        meter = Progress(length(acquisitions); desc = "Stereo PIV sequence: ",
                         enabled = progress === true)
        pending = load_acquisition(first(acquisitions))
        for (i, acq) in enumerate(acquisitions)
            if cancel !== nothing && cancel()
                cancelled = true
                break
            end
            try
                frames, T = fetch_frames(pending)
                i < length(acquisitions) && (pending = load_acquisition(acquisitions[i + 1]))
                if a === nothing || eltype(a) !== T
                    a = Matrix{T}(undef, size(dw1.grid))
                    b = similar(a)
                end
                result = _run_piv_stereo!(a, b, frames..., dw1, dw2, params,
                                          node_mask; backend, workspace, kwargs...)
                result_scale = scale_pairs === nothing ? scale : pair_scale(scale, scale_pairs[i])
                result = result_scale === nothing ? result : with_scale(result, result_scale)
                collect_results && push!(results, result)
            catch
                @error "Stereo PIV sequence failed on acquisition $i of $(length(acquisitions))"
                rethrow()
            end
            # Live-consumer hook, mirroring run_piv_sequence's on_result:
            # called on the caller's task, in acquisition order, before the
            # incremental write and the progress callback.
            on_result === nothing || on_result(i, result)
            if file !== nothing
                file[result_key(i)] = result
                all(x -> x isa AbstractString, acq) &&
                    (file[source_key(i)] = String[String(x) for x in acq])
            elseif output isa Function
                _write_stereo_pair_file(String(output(i, acq)), result, acq)
            end
            progress isa Function ? progress(i, length(acquisitions)) : next!(meter)
            result = nothing
        end
    catch
        failed = true
        rethrow()
    finally
        try
            if pending !== nothing
                try
                    fetch_frames(pending)
                catch
                    (failed || cancelled) || rethrow()
                end
            end
        finally
            if file !== nothing
                try
                    close(file)
                catch
                    failed || rethrow()
                end
            end
        end
    end
    return results
end

function _write_stereo_pair_file(path, result, acquisition)
    dir = dirname(path)
    isempty(dir) || mkpath(dir)
    jldopen(path, "w") do f
        f["format_version"] = RESULTS_FORMAT_VERSION
        f[result_key(1)] = result
        all(x -> x isa AbstractString, acquisition) &&
            (f[source_key(1)] = String[String(x) for x in acquisition])
    end
    return path
end


# Lazy source adapter used by the stereo ensemble driver. It lets the planar
# ensemble engine reload and dewarp each raw frame once per pass without
# retaining the entire dewarped recording in memory.
struct _DewarpedFrame{S,D,F,T<:AbstractFloat}
    source::S
    dewarper::D
    preprocess::F
    image_type::Type{T}
end

function load_frame(src::_DewarpedFrame, ::Type)
    img = load_frame(src.source, src.image_type)
    src.preprocess === nothing || (img = src.preprocess(img))
    return dewarp(src.dewarper, img)
end

"""
    run_piv_stereo_ensemble(pairs1, pairs2, dw1, dw2,
                            params = PIVParameters(); kwargs...) -> StereoPIVResult

Estimate one stereo field by summing correlations over synchronized pairs
for each camera, then reconstructing three components. Use this for weak
single-pair peaks in a statistically stationary interval. The two camera
pair lists must have the same nonzero length. Frames are loaded and
dewarped as needed on each ensemble pass. `preprocess` may be one function
or a two-function tuple for separate cameras. The shared dewarp overlap and
optional world-grid `mask` apply to both cameras; other keywords follow
[`run_piv_ensemble`](@ref), including `effort`, `backend`, and `image_type`.

Exposure timestamps and pair delays are checked before loading either
camera using the `sync_atol`, `sync_rtol`, and `missing_timestamps` options
of [`run_piv_stereo_sequence`](@ref), with the same defaults and policy.
Missing metadata allowed by the default does not establish synchronization.

The result is a peak estimate from the pooled correlations, which can differ
from the arithmetic mean of individual vectors when their displacements
vary widely. Its propagated uncertainty describes correlation noise in that
pooled estimate; it does not measure pair-to-pair flow fluctuations. Use
[`field_statistics`](@ref) on a stereo sequence for those fluctuations.
"""
function run_piv_stereo_ensemble(pairs1::AbstractVector, pairs2::AbstractVector,
                                 dw1::ImageDewarper, dw2::ImageDewarper,
                                 params::Union{PIVParameters,AbstractVector{PIVParameters}};
                                 effort::Union{Nothing,Symbol} = nothing,
                                 backend::Symbol = :cpu, preprocess = nothing,
                                 image_type::Type{<:AbstractFloat} = Float64,
                                 progress::Bool = true,
                                 mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                                 scale::Union{Nothing,PhysicalScale} = nothing,
                                 sync_atol::Real = 0.0, sync_rtol::Real = 0.0,
                                 missing_timestamps::Symbol = :allow,
                                 kwargs...)
    effort === nothing ||
        throw(ArgumentError("effort cannot be combined with explicit PIVParameters or pass schedules"))
    length(pairs1) == length(pairs2) ||
        throw(DimensionMismatch("camera pair sequences must have equal length, got " *
                                "$(length(pairs1)) and $(length(pairs2))"))
    isempty(pairs1) && throw(ArgumentError("camera pair sequences must not be empty"))
    _check_stereo_pair_times(pairs1, pairs2; sync_atol, sync_rtol, missing_timestamps)
    node_mask = _stereo_node_mask(dw1, dw2, mask)
    pre1, pre2 = preprocess isa Tuple && length(preprocess) == 2 ? preprocess :
                 (preprocess, preprocess)
    wrap(pairs, dw, pre) = [(_DewarpedFrame(p[1], dw, pre, image_type),
                             _DewarpedFrame(p[2], dw, pre, image_type)) for p in pairs]
    r1 = run_piv_ensemble(wrap(pairs1, dw1, pre1), params; backend, mask = node_mask,
                          image_type, progress, kwargs...)
    r2 = run_piv_ensemble(wrap(pairs2, dw2, pre2), params; backend, mask = node_mask,
                          image_type, progress, kwargs...)
    result = reconstruct_stereo(r1, r2, dw1.cam, dw2.cam, dw1.grid)
    return scale === nothing ? result : with_scale(result, scale)
end

function run_piv_stereo_ensemble(pairs1::AbstractVector, pairs2::AbstractVector,
                                 dw1::ImageDewarper, dw2::ImageDewarper;
                                 effort::Union{Nothing,Symbol} = nothing, kwargs...)
    if effort === nothing
        return run_piv_stereo_ensemble(pairs1, pairs2, dw1, dw2, PIVParameters(); kwargs...)
    end
    # Let the planar ensemble effort method split parameter and driver keys.
    piv_kwargs, driver_kwargs = split_effort_kwargs(kwargs)
    passes = effort_schedule(effort; ensemble = true, image_size = size(dw1.grid), piv_kwargs...)
    return run_piv_stereo_ensemble(pairs1, pairs2, dw1, dw2, passes; driver_kwargs...)
end

# Combine two per-camera 2C results (on the same dewarped grid) into the 3C
# field. Displacements convert to world units as u·step(x), v·step(y) (signs
# included), then each point's 4×3 least-squares system is solved; the
# pseudoinverse also propagates the per-camera uncertainties.
function reconstruct_stereo(r1::PIVResult{T}, r2::PIVResult{T},
                            cam1::CameraCalibration, cam2::CameraCalibration,
                            grid::DewarpGrid) where {T}
    sx, sy = step(grid.x), step(grid.y)
    δ = max(abs(sx), abs(sy))
    ny, nx = size(r1.u)
    # World coordinates of the vector grid (window centers in dewarped px).
    X = [first(grid.x) + (Float64(xi) - 1) * sx for xi in r1.x]
    Y = [first(grid.y) + (Float64(yi) - 1) * sy for yi in r1.y]

    u = Matrix{T}(undef, ny, nx)
    v = Matrix{T}(undef, ny, nx)
    w = Matrix{T}(undef, ny, nx)
    uu = Matrix{T}(undef, ny, nx)
    uv = Matrix{T}(undef, ny, nx)
    uw = Matrix{T}(undef, ny, nx)
    for j in 1:nx, i in 1:ny
        u[i, j] = v[i, j] = w[i, j] = uu[i, j] = uv[i, j] = uw[i, j] = T(NaN)
        r1.mask[i, j] && continue
        t1 = ray_slopes(cam1, X[j], Y[i], grid.z, δ)
        t2 = ray_slopes(cam2, X[j], Y[i], grid.z, δ)
        A = @SMatrix [1.0 0.0 -t1[1];
                      0.0 1.0 -t1[2];
                      1.0 0.0 -t2[1];
                      0.0 1.0 -t2[2]]
        N = A' * A
        # Degenerate geometry (parallel viewing rays) has no 3C solution.
        (all(isfinite, A) && abs(det(N)) > 1e-10) || continue
        G = inv(N) * A'  # 3×4 least-squares operator
        m = SVector(Float64(r1.u[i, j]) * sx, Float64(r1.v[i, j]) * sy,
                    Float64(r2.u[i, j]) * sx, Float64(r2.v[i, j]) * sy)
        d = G * m
        u[i, j] = T(d[1])
        v[i, j] = T(d[2])
        w[i, j] = T(d[3])
        σ² = SVector(abs2(Float64(r1.uncertainty_u[i, j]) * sx),
                     abs2(Float64(r1.uncertainty_v[i, j]) * sy),
                     abs2(Float64(r2.uncertainty_u[i, j]) * sx),
                     abs2(Float64(r2.uncertainty_v[i, j]) * sy))
        σd = sqrt.((G .^ 2) * σ²)
        uu[i, j] = T(σd[1])
        uv[i, j] = T(σd[2])
        uw[i, j] = T(σd[3])
    end
    return StereoPIVResult{T}(T.(X), T.(Y), grid.z, u, v, w, uu, uv, uw,
                              r1.outliers .| r2.outliers, r1.mask .| r2.mask,
                              r1, r2, r1.parameters)
end
