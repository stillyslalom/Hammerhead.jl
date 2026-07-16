# Stereoscopic 3C reconstruction (Phase 5, slice 3): the 2D engine runs per
# camera on images dewarped to the common world plane, then the two 2C fields
# combine into (u, v, w) by geometric least squares.
#
# The per-point reconstruction (like the calibration it rests on) works in
# Float64 and converts on store: it is O(vector grid) per pair, far from the
# per-pixel hot path that the precision-follows-images rule protects.

"""
    StereoPIVResult{T<:AbstractFloat}

Result of [`run_piv_stereo`](@ref). The numeric precision `T` follows the
input images, like [`PIVResult`](@ref).

# Fields
- `x`, `y`: world coordinates of the vector grid (`x` along the dewarp grid's
  `x` range, `y` along its `y` range, in the grid's world units, e.g. mm).
- `z`: out-of-plane position of the measurement plane (the `DewarpGrid`'s `z`).
- `u`, `v`, `w`: displacement components on the `(length(y), length(x))`
  grid, in world units per frame interval: `u` along world X, `v` along
  world Y (signs follow the dewarp grid's ranges), `w` along world +Z.
- `uncertainty_u`, `uncertainty_v`, `uncertainty_w`: per-vector measurement
  uncertainty (one standard deviation, world units) propagated from the two
  cameras' correlation-statistics estimates through the reconstruction —
  `NaN` unless the `uncertainty` parameter was enabled.
- `outliers`: union of the two cameras' outlier flags. Flagged vectors were
  reconstructed from at least one replaced/substituted 2C vector.
- `mask`: windows dropped because they overlap either camera's out-of-view
  region or the user mask (`NaN` fields, never outliers).
- `cam1`, `cam2`: the per-camera 2C [`PIVResult`](@ref)s on the dewarped
  images (displacements in dewarped pixels), retained for diagnostics.
- `parameters`: the `PIVParameters` of the (final) pass.
- `scale`: the [`PhysicalScale`](@ref) attached via the `scale` keyword of
  [`run_piv_stereo`](@ref) or [`with_scale`](@ref); `nothing` when none was
  attached. For stereo, set `dt` (and the unit labels) only and leave
  `pixel_size = 1` — the arrays are already in world units. Metadata only
  until [`physical`](@ref) converts them; `cam1`/`cam2` always stay raw.
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

Stereoscopic (2D3C) PIV on one frame pair: `A1`/`B1` are camera 1's raw
frames and `A2`/`B2` camera 2's, `dw1`/`dw2` the cameras'
[`ImageDewarper`](@ref)s (which must share one [`DewarpGrid`](@ref)). Each
camera's pair is dewarped onto the common world plane, analyzed with the 2D
engine ([`run_piv`](@ref) with `params`, which may be a multi-pass schedule,
or with `effort = :low`, `:medium`, or `:high`), and the two 2C fields are
combined per vector into the three-component displacement `(u, v, w)` in world
units per frame interval (no time scaling is applied).

Reconstruction: at each grid point, camera `i`'s dewarped in-plane
displacement measures `uᵢ = dx - dz·tXᵢ`, `vᵢ = dy - dz·tYᵢ`, where
`(tXᵢ, tYᵢ)` is the in-plane drift per unit Z along that camera's viewing
ray (evaluated from the calibration). The four equations are solved for
`(dx, dy, dz)` by unweighted least squares; points with degenerate geometry
(parallel viewing rays, e.g. identical cameras) come out `NaN`. With the
`uncertainty` parameter enabled, the per-camera Wieneke (2015) estimates are
propagated through the same least-squares operator into
`uncertainty_u`/`v`/`w` (world units, assuming independent per-camera
errors).

`mask` is an optional grid-sized `Bool` matrix of world-plane pixels to
exclude (`true` = excluded); it is combined with the dewarpers' out-of-view
masks (`dw1.mask .| dw2.mask`), so only the stereo overlap region is
analyzed. `scale` attaches a [`PhysicalScale`](@ref) to the stereo result
(not to the per-camera results): set `dt` and the unit labels only, leaving
`pixel_size = 1` — the stereo fields are already in world units, so
[`physical`](@ref) only needs to divide by the frame interval. Remaining
keyword arguments (`threaded`, `predictor_smoothing`, `backend`,
`mask_threshold`) are forwarded to [`run_piv`](@ref). A GPU backend accelerates
the two per-camera PIV analyses; dewarping and reconstruction remain on the
CPU (see [Run PIV on a GPU](@ref)).

Both cameras are analyzed with the same parameters and mask, so their vector
grids, masks, and (via the union) outlier maps are directly compatible; the
per-camera results are retained in the returned [`StereoPIVResult`](@ref).
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

Process a synchronized stereo recording. `pairs1` and `pairs2` are equal-length
camera pair lists in the same format accepted by [`run_piv_sequence`](@ref).
As a convenience, `acquisitions` may instead contain 4-tuples
`(A1, B1, A2, B2)`. The supplied [`ImageDewarper`](@ref)s, a dewarped-image
buffer pair, and one [`PIVWorkspace`](@ref) are reused for the whole sequence.

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
The next synchronized acquisition is prefetched while the current one runs.
Timestamped [`FramePair`](@ref)s attach their actual pair-specific `dt` to
each result when `scale` is supplied; the two cameras' intervals must agree.

Pass explicit `params`, or omit them and use `effort = :low`, `:medium`, or
`:high`. Other keywords are forwarded to [`run_piv_stereo`](@ref).
"""
function run_piv_stereo_sequence(pairs1::AbstractVector, pairs2::AbstractVector,
                                 dw1::ImageDewarper, dw2::ImageDewarper,
                                 params::Union{PIVParameters,AbstractVector{PIVParameters}};
                                 effort::Union{Nothing,Symbol} = nothing, kwargs...)
    effort === nothing ||
        throw(ArgumentError("effort cannot be combined with explicit PIVParameters or pass schedules"))
    length(pairs1) == length(pairs2) ||
        throw(DimensionMismatch("camera pair sequences must have equal length, got " *
                                "$(length(pairs1)) and $(length(pairs2))"))
    _check_stereo_pair_times(pairs1, pairs2)
    acquisitions = [(p1[1], p1[2], p2[1], p2[2]) for (p1, p2) in zip(pairs1, pairs2)]
    return _run_piv_stereo_sequence(acquisitions, dw1, dw2, params;
                                    scale_pairs = pairs1, kwargs...)
end

function run_piv_stereo_sequence(pairs1::AbstractVector, pairs2::AbstractVector,
                                 dw1::ImageDewarper, dw2::ImageDewarper;
                                 effort::Union{Nothing,Symbol} = nothing, kwargs...)
    length(pairs1) == length(pairs2) ||
        throw(DimensionMismatch("camera pair sequences must have equal length, got " *
                                "$(length(pairs1)) and $(length(pairs2))"))
    _check_stereo_pair_times(pairs1, pairs2)
    acquisitions = [(p1[1], p1[2], p2[1], p2[2]) for (p1, p2) in zip(pairs1, pairs2)]
    if effort === nothing
        return _run_piv_stereo_sequence(acquisitions, dw1, dw2, PIVParameters();
                                        scale_pairs = pairs1, kwargs...)
    end
    piv_kwargs, driver_kwargs = split_effort_kwargs(kwargs)
    passes = effort_schedule(effort; image_size = size(dw1.grid), piv_kwargs...)
    return _run_piv_stereo_sequence(acquisitions, dw1, dw2, passes;
                                    scale_pairs = pairs1, driver_kwargs...)
end

function _check_stereo_pair_times(pairs1, pairs2)
    for (i, (p1, p2)) in enumerate(zip(pairs1, pairs2))
        dt1 = hasproperty(p1, :dt) ? getproperty(p1, :dt) : nothing
        dt2 = hasproperty(p2, :dt) ? getproperty(p2, :dt) : nothing
        if dt1 !== nothing && dt2 !== nothing && dt1 != dt2
            throw(ArgumentError("camera pair $i has inconsistent frame intervals: " *
                                "$dt1 and $dt2"))
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
                                  cancel::Union{Nothing,Function} = nothing,
                                  image_type::Type{<:AbstractFloat} = Float64,
                                  mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                                  scale::Union{Nothing,PhysicalScale} = nothing,
                                  scale_pairs = nothing,
                                  kwargs...)
    isempty(acquisitions) && throw(ArgumentError("acquisitions must not be empty"))
    all(a -> a isa Tuple && length(a) == 4, acquisitions) ||
        throw(ArgumentError("each stereo acquisition must be a 4-tuple (A1, B1, A2, B2)"))
    node_mask = _stereo_node_mask(dw1, dw2, mask)
    pre1, pre2 = preprocess isa Tuple && length(preprocess) == 2 ? preprocess :
                 (preprocess, preprocess)
    workspace = piv_workspace(; backend)
    results = StereoPIVResult[]
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
    try
        file === nothing || (file["format_version"] = RESULTS_FORMAT_VERSION)
        meter = Progress(length(acquisitions); desc = "Stereo PIV sequence: ",
                         enabled = progress === true)
        pending = load_acquisition(first(acquisitions))
        for (i, acq) in enumerate(acquisitions)
            if cancel !== nothing && cancel()
                # Ensure a prefetched preprocessor is no longer active when
                # this call returns to its caller.
                try
                    fetch(pending)
                catch
                end
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
                push!(results, result_scale === nothing ? result : with_scale(result, result_scale))
            catch
                @error "Stereo PIV sequence failed on acquisition $i of $(length(acquisitions))"
                rethrow()
            end
            # Live-consumer hook, mirroring run_piv_sequence's on_result:
            # called on the caller's task, in acquisition order, before the
            # incremental write and the progress callback.
            on_result === nothing || on_result(i, results[end])
            if file !== nothing
                file[result_key(i)] = results[end]
                all(x -> x isa AbstractString, acq) &&
                    (file[source_key(i)] = String[String(x) for x in acq])
            elseif output isa Function
                _write_stereo_pair_file(String(output(i, acq)), results[end], acq)
            end
            progress isa Function ? progress(i, length(acquisitions)) : next!(meter)
        end
    finally
        file === nothing || close(file)
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

Low-SNR stereoscopic PIV by composing two synchronized
[`run_piv_ensemble`](@ref) analyses followed by the same calibrated 3C
reconstruction used by [`run_piv_stereo`](@ref). The two camera pair lists
must have equal nonzero length. Frames are loaded, optionally preprocessed,
and dewarped lazily on every ensemble pass; the whole dewarped recording is
never retained in memory. `preprocess` may be shared or a camera-specific
2-tuple. The dewarper overlap and optional world-grid `mask` are applied to
both cameras. Other keywords follow [`run_piv_ensemble`](@ref), including
`effort`, `backend`, `image_type`, and `progress`.

The result estimates one stationary ensemble-mean 3C field. Its propagated
uncertainty is the noise-driven uncertainty of that mean, not physical
pair-to-pair turbulence; use [`field_statistics`](@ref) on a stereo sequence
for the latter.
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
                                 kwargs...)
    effort === nothing ||
        throw(ArgumentError("effort cannot be combined with explicit PIVParameters or pass schedules"))
    length(pairs1) == length(pairs2) ||
        throw(DimensionMismatch("camera pair sequences must have equal length, got " *
                                "$(length(pairs1)) and $(length(pairs2))"))
    isempty(pairs1) && throw(ArgumentError("camera pair sequences must not be empty"))
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
