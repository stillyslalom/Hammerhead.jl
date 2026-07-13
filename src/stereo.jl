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
    dewarp!(a, dw1, A1)
    dewarp!(b, dw1, B1)
    r1 = run_piv(a, b, params; backend, mask = node_mask, kwargs...)
    dewarp!(a, dw2, A2)
    dewarp!(b, dw2, B2)
    r2 = run_piv(a, b, params; backend, mask = node_mask, kwargs...)
    result = reconstruct_stereo(r1, r2, dw1.cam, dw2.cam, grid)
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
