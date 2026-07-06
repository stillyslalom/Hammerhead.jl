# Camera calibration models for stereo PIV (Phase 5).
#
# Everything in this file is a deliberate Float64 island: calibration is an
# offline, once-per-experiment fit on a few hundred points, far from the
# image-correlation hot path that the precision-follows-images rule protects.
#
# Coordinate conventions: pixel coordinates are `(x, y)` with `x` along
# columns and `y` along rows (package-wide sign convention). World
# coordinates are `(X, Y, Z)` in physical units (e.g. mm) with `X`/`Y` in the
# calibration-target plane and `Z` out of plane (positive toward the cameras
# when `X` is right and `Y` is up in the image — right-handed).

"""
    CameraCalibration

Abstract supertype for camera mapping models ([`PinholeCamera`](@ref),
[`SoloffCamera`](@ref)). Subtypes implement [`world_to_pixel`](@ref) and
[`pixel_to_world`](@ref); fit either with [`calibrate_camera`](@ref).
"""
abstract type CameraCalibration end

"""
    PinholeCamera(P::AbstractMatrix)
    PinholeCamera(K, R, t)

Pinhole (projective / DLT) camera: world point `(X, Y, Z)` maps to the pixel
`(x, y) = (h₁/h₃, h₂/h₃)` where `h = P * [X, Y, Z, 1]` and `P` is the 3×4
projection matrix. The second form builds `P = K * [R t]` from a 3×3
intrinsic matrix `K`, a 3×3 rotation `R`, and a translation 3-vector `t`.

`P` is stored normalized so that its third row's rotational part is a unit
vector (making `h₃` the point's depth along the optical axis). Fit from
point correspondences with `calibrate_camera(...; model = :pinhole)`; the fit
requires non-coplanar world points, i.e. a calibration target imaged at two
or more `Z` positions.
"""
struct PinholeCamera <: CameraCalibration
    P::SMatrix{3,4,Float64,12}

    function PinholeCamera(P::AbstractMatrix{<:Real})
        size(P) == (3, 4) ||
            throw(ArgumentError("projection matrix must be 3×4, got $(size(P))"))
        n = hypot(P[3, 1], P[3, 2], P[3, 3])
        n > 0 ||
            throw(ArgumentError("degenerate projection matrix: third-row rotation part is zero"))
        new(SMatrix{3,4,Float64}(P) / n)
    end
end

function PinholeCamera(K::AbstractMatrix{<:Real}, R::AbstractMatrix{<:Real},
                       t::AbstractVector{<:Real})
    size(K) == (3, 3) || throw(ArgumentError("K must be 3×3, got $(size(K))"))
    size(R) == (3, 3) || throw(ArgumentError("R must be 3×3, got $(size(R))"))
    length(t) == 3 || throw(ArgumentError("t must have length 3, got $(length(t))"))
    return PinholeCamera([K * R K * t])
end

Base.show(io::IO, cam::PinholeCamera) =
    print(io, "PinholeCamera(P = ", round.(cam.P; sigdigits = 5), ")")

# Soloff (1997) polynomial camera: cubic in X and Y, quadratic in Z — the
# standard 19-term basis. Terms are evaluated on normalized coordinates
# (centered and scaled to ~[-1, 1]) to keep the Vandermonde fit conditioned.
@inline soloff_terms(X, Y, Z) =
    SVector(1.0, X, Y, Z, X^2, X * Y, Y^2, X * Z, Y * Z, Z^2,
            X^3, X^2 * Y, X * Y^2, Y^3, X^2 * Z, X * Y * Z, Y^2 * Z, X * Z^2, Y * Z^2)

@inline soloff_terms_dX(X, Y, Z) =
    SVector(0.0, 1.0, 0.0, 0.0, 2X, Y, 0.0, Z, 0.0, 0.0,
            3X^2, 2X * Y, Y^2, 0.0, 2X * Z, Y * Z, 0.0, Z^2, 0.0)

@inline soloff_terms_dY(X, Y, Z) =
    SVector(0.0, 0.0, 1.0, 0.0, 0.0, X, 2Y, 0.0, Z, 0.0,
            0.0, X^2, 2X * Y, 3Y^2, 0.0, X * Z, 2Y * Z, 0.0, Z^2)

"""
    SoloffCamera

Soloff polynomial camera (Soloff, Adrian & Liu 1997): each pixel coordinate
is a 19-term polynomial in the world coordinates — cubic in `X` and `Y`,
quadratic in `Z` — evaluated on internally normalized coordinates. The
polynomial absorbs lens distortion and refraction that a pinhole model
cannot, at the cost of having no closed-form inverse
([`pixel_to_world`](@ref) uses Newton iteration).

Fit from point correspondences with `calibrate_camera(...; model = :soloff)`;
the fit requires calibration points at three or more distinct `Z` positions.
"""
struct SoloffCamera <: CameraCalibration
    ax::SVector{19,Float64}
    ay::SVector{19,Float64}
    center::SVector{3,Float64}  # world normalization: X̂ = (X - center) ./ scale
    scale::SVector{3,Float64}
end

Base.show(io::IO, cam::SoloffCamera) =
    print(io, "SoloffCamera(19-term Soloff polynomial, world center = ",
          round.(cam.center; sigdigits = 4), ", scale = ",
          round.(cam.scale; sigdigits = 4), ")")

"""
    world_to_pixel(cam::CameraCalibration, world) -> SVector{2,Float64}

Map a world point `(X, Y, Z)` (any 3-element indexable) to its pixel
coordinates `(x, y)` (`x` along columns, `y` along rows).
"""
function world_to_pixel(cam::PinholeCamera, world)
    h = cam.P * SVector(Float64(world[1]), Float64(world[2]), Float64(world[3]), 1.0)
    return SVector(h[1] / h[3], h[2] / h[3])
end

function world_to_pixel(cam::SoloffCamera, world)
    w = SVector(Float64(world[1]), Float64(world[2]), Float64(world[3]))
    n = (w - cam.center) ./ cam.scale
    t = soloff_terms(n[1], n[2], n[3])
    return SVector(dot(cam.ax, t), dot(cam.ay, t))
end

"""
    pixel_to_world(cam::CameraCalibration, pixel, z) -> SVector{3,Float64}

Map pixel coordinates `(x, y)` back to the world point `(X, Y, z)` on the
plane at out-of-plane position `z`. For a [`PinholeCamera`](@ref) this is an
exact linear solve (the ray–plane intersection); for a
[`SoloffCamera`](@ref) or [`TransformedCamera`](@ref) it is a 2D Newton
iteration and returns a NaN-filled point if the iteration fails to converge
(e.g. for pixels far outside the calibrated domain).
"""
function pixel_to_world(cam::PinholeCamera, pixel, z::Real)
    x, y = Float64(pixel[1]), Float64(pixel[2])
    P = cam.P
    A = @SMatrix [P[1, 1]-x*P[3, 1] P[1, 2]-x*P[3, 2];
                  P[2, 1]-y*P[3, 1] P[2, 2]-y*P[3, 2]]
    b = @SVector [x * (P[3, 3] * z + P[3, 4]) - (P[1, 3] * z + P[1, 4]),
                  y * (P[3, 3] * z + P[3, 4]) - (P[2, 3] * z + P[2, 4])]
    XY = A \ b
    return SVector(XY[1], XY[2], Float64(z))
end

function pixel_to_world(cam::SoloffCamera, pixel, z::Real; maxiter::Int = 50)
    target = SVector(Float64(pixel[1]), Float64(pixel[2]))
    Zn = (Float64(z) - cam.center[3]) / cam.scale[3]
    tol = 1e-8 * (1 + norm(target))

    # Initial guess from the linear terms of the polynomial.
    A0 = @SMatrix [cam.ax[2] cam.ax[3]; cam.ay[2] cam.ay[3]]
    b0 = target - SVector(cam.ax[1] + cam.ax[4] * Zn, cam.ay[1] + cam.ay[4] * Zn)
    XY = abs(det(A0)) > 1e-12 ? A0 \ b0 : zero(SVector{2,Float64})

    for _ in 1:maxiter
        t = soloff_terms(XY[1], XY[2], Zn)
        r = SVector(dot(cam.ax, t), dot(cam.ay, t)) - target
        if norm(r) <= tol
            return SVector(XY[1] * cam.scale[1] + cam.center[1],
                           XY[2] * cam.scale[2] + cam.center[2], Float64(z))
        end
        tx = soloff_terms_dX(XY[1], XY[2], Zn)
        ty = soloff_terms_dY(XY[1], XY[2], Zn)
        J = @SMatrix [dot(cam.ax, tx) dot(cam.ax, ty); dot(cam.ay, tx) dot(cam.ay, ty)]
        abs(det(J)) > 1e-14 || break
        XY -= J \ r
    end
    return SVector(NaN, NaN, Float64(z))
end

# --- Rigid world-frame transforms ---------------------------------------------

"""
    TransformedCamera(cam::CameraCalibration, R, t)

Camera model pre-composed with a rigid world-coordinate transform: a point
`w` in the transformed (new) world frame maps to the pixel
`world_to_pixel(cam, R * w + t)`, where the 3×3 rotation `R` and 3-vector
`t` express the new frame in the original calibration's frame. Wrapping a
`TransformedCamera` collapses into a single composed transform.

Produced by [`self_calibrate`](@ref) when correcting camera models whose
functional form cannot absorb a world rotation exactly (e.g.
[`SoloffCamera`](@ref) — a [`PinholeCamera`](@ref) correction is instead
baked directly into its projection matrix). Behaves like any other
[`CameraCalibration`](@ref): `world_to_pixel`, `pixel_to_world`, and
[`ImageDewarper`](@ref) all apply.
"""
struct TransformedCamera{C<:CameraCalibration} <: CameraCalibration
    cam::C
    R::SMatrix{3,3,Float64,9}
    t::SVector{3,Float64}

    function TransformedCamera(cam::C, R::AbstractMatrix{<:Real},
                               t::AbstractVector{<:Real}) where {C<:CameraCalibration}
        Rs, ts = _check_rigid(R, t)
        return new{C}(cam, Rs, ts)
    end
end

TransformedCamera(tc::TransformedCamera, R::AbstractMatrix{<:Real},
                  t::AbstractVector{<:Real}) =
    TransformedCamera(tc.cam, tc.R * SMatrix{3,3,Float64}(R),
                      tc.R * SVector{3,Float64}(t) + tc.t)

function _check_rigid(R::AbstractMatrix{<:Real}, t::AbstractVector{<:Real})
    size(R) == (3, 3) || throw(ArgumentError("R must be 3×3, got $(size(R))"))
    length(t) == 3 || throw(ArgumentError("t must have length 3, got $(length(t))"))
    Rs = SMatrix{3,3,Float64}(R)
    norm(Rs' * Rs - I) < 1e-6 && det(Rs) > 0 ||
        throw(ArgumentError("R must be a proper rotation matrix"))
    return Rs, SVector{3,Float64}(t)
end

Base.show(io::IO, tc::TransformedCamera) =
    print(io, "TransformedCamera(", tc.cam, ", t = ", round.(tc.t; sigdigits = 4), ")")

function world_to_pixel(cam::TransformedCamera, world)
    w = SVector(Float64(world[1]), Float64(world[2]), Float64(world[3]))
    return world_to_pixel(cam.cam, cam.R * w + cam.t)
end

function pixel_to_world(cam::TransformedCamera, pixel, z::Real; maxiter::Int = 30)
    target = SVector(Float64(pixel[1]), Float64(pixel[2]))
    zf = Float64(z)
    # Initial guess: invert the inner camera at the depth where the
    # transformed plane sits, then pull the point back into the new frame.
    z0 = (cam.R * SVector(0.0, 0.0, zf) + cam.t)[3]
    w0 = cam.R' * (pixel_to_world(cam.cam, pixel, z0) - cam.t)
    XY = all(isfinite, w0) ? SVector(w0[1], w0[2]) : zero(SVector{2,Float64})
    tol = 1e-8 * (1 + norm(target))
    for _ in 1:maxiter
        r = world_to_pixel(cam, SVector(XY[1], XY[2], zf)) - target
        norm(r) <= tol && return SVector(XY[1], XY[2], zf)
        δ = 1e-4 * (1 + max(abs(XY[1]), abs(XY[2])))
        dpx = (world_to_pixel(cam, SVector(XY[1] + δ, XY[2], zf)) -
               world_to_pixel(cam, SVector(XY[1] - δ, XY[2], zf))) / (2δ)
        dpy = (world_to_pixel(cam, SVector(XY[1], XY[2] + δ, zf)) -
               world_to_pixel(cam, SVector(XY[1], XY[2] - δ, zf))) / (2δ)
        J = @SMatrix [dpx[1] dpy[1]; dpx[2] dpy[2]]
        (all(isfinite, J) && abs(det(J)) > 1e-14) || break
        XY -= J \ r
    end
    return SVector(NaN, NaN, zf)
end

"""
    apply_world_transform(cam::CameraCalibration, R, t) -> CameraCalibration

Pre-compose `cam` with a rigid world transform: the returned camera maps a
point `w` of the new world frame as `cam` maps `R * w + t`. A
[`PinholeCamera`](@ref) absorbs the transform exactly into its projection
matrix; other models are wrapped in a [`TransformedCamera`](@ref) (nested
wrappers collapse into one).
"""
apply_world_transform(cam::CameraCalibration, R::AbstractMatrix{<:Real},
                      t::AbstractVector{<:Real}) = TransformedCamera(cam, R, t)

function apply_world_transform(cam::PinholeCamera, R::AbstractMatrix{<:Real},
                               t::AbstractVector{<:Real})
    Rs, ts = _check_rigid(R, t)
    M = vcat(hcat(Rs, ts), SMatrix{1,4,Float64}(0.0, 0.0, 0.0, 1.0))
    return PinholeCamera(cam.P * M)
end

# --- Fitting -----------------------------------------------------------------

# Hartley-style similarity normalization: translate the points to zero
# centroid and scale so the mean norm is √(dimension). Returns the (d+1)×(d+1)
# homogeneous normalization matrix.
function _normalization_matrix(points::AbstractVector{<:SVector{N,Float64}}) where {N}
    c = sum(points) / length(points)
    meannorm = sum(p -> norm(p - c), points) / length(points)
    s = meannorm > 0 ? sqrt(N) / meannorm : 1.0
    T = Matrix{Float64}(I, N + 1, N + 1)
    T[1:N, 1:N] .*= s
    T[1:N, N+1] .= -s .* c
    return T
end

function fit_pinhole(pixels::Vector{SVector{2,Float64}}, world::Vector{SVector{3,Float64}})
    N = length(pixels)
    N >= 6 || throw(ArgumentError("pinhole calibration needs at least 6 point pairs, got $N"))

    # (Near-)coplanar world points leave the DLT underdetermined.
    c = sum(world) / N
    M = reduce(hcat, (w - c for w in world))
    sv = svdvals(M)
    sv[3] > 1e-6 * sv[1] ||
        throw(ArgumentError("world points are (nearly) coplanar; a pinhole fit needs " *
                            "calibration points at two or more Z positions — add planes " *
                            "or use model = :soloff with three or more planes"))

    Tpx = _normalization_matrix(pixels)
    Twd = _normalization_matrix(world)

    A = zeros(2N, 12)
    for i in 1:N
        x = Tpx[1, 1] * pixels[i][1] + Tpx[1, 3]
        y = Tpx[2, 2] * pixels[i][2] + Tpx[2, 3]
        X = Twd[1, 1] * world[i][1] + Twd[1, 4]
        Y = Twd[2, 2] * world[i][2] + Twd[2, 4]
        Z = Twd[3, 3] * world[i][3] + Twd[3, 4]
        A[2i-1, :] .= (X, Y, Z, 1.0, 0.0, 0.0, 0.0, 0.0, -x * X, -x * Y, -x * Z, -x)
        A[2i, :] .= (0.0, 0.0, 0.0, 0.0, X, Y, Z, 1.0, -y * X, -y * Y, -y * Z, -y)
    end
    p = svd(A).V[:, end]
    Pn = transpose(reshape(p, 4, 3))
    P = Tpx \ (Pn * Twd)

    # Fix the projective sign so point depths come out positive.
    depths = [P[3, 1] * w[1] + P[3, 2] * w[2] + P[3, 3] * w[3] + P[3, 4] for w in world]
    median(depths) < 0 && (P = -P)
    return PinholeCamera(P)
end

# Count distinct Z planes among the world points (tolerance scaled to the span).
function _n_z_planes(world::Vector{SVector{3,Float64}})
    zs = sort!([w[3] for w in world])
    gap = 1e-9 + 1e-6 * (zs[end] - zs[1])
    return 1 + count(i -> zs[i+1] - zs[i] > gap, 1:length(zs)-1)
end

function fit_soloff(pixels::Vector{SVector{2,Float64}}, world::Vector{SVector{3,Float64}})
    N = length(pixels)
    N >= 19 || throw(ArgumentError("Soloff calibration needs at least 19 point pairs, got $N"))
    _n_z_planes(world) >= 3 ||
        throw(ArgumentError("Soloff calibration is quadratic in Z and needs points at " *
                            "three or more distinct Z positions"))

    lo = SVector{3}(minimum(w[i] for w in world) for i in 1:3)
    hi = SVector{3}(maximum(w[i] for w in world) for i in 1:3)
    center = (lo + hi) / 2
    scale = SVector{3}(max((hi[i] - lo[i]) / 2, 1e-12) for i in 1:3)

    A = zeros(N, 19)
    for i in 1:N
        n = (world[i] - center) ./ scale
        A[i, :] .= soloff_terms(n[1], n[2], n[3])
    end
    F = qr(A)
    # A rank-deficient design matrix means degenerate point geometry (e.g. all
    # points on one line per plane).
    κ = cond(A)
    κ < 1e10 ||
        throw(ArgumentError("degenerate calibration-point geometry (design matrix " *
                            "condition number $(round(κ, sigdigits=3))); spread the " *
                            "points over both in-plane directions and several Z planes"))
    ax = F \ [p[1] for p in pixels]
    ay = F \ [p[2] for p in pixels]
    return SoloffCamera(SVector{19,Float64}(ax), SVector{19,Float64}(ay), center, scale)
end

"""
    calibrate_camera(pixel_points, world_points; model = :soloff) -> CameraCalibration

Fit a camera model to corresponding calibration points. `pixel_points` are
2-element indexables `(x, y)` (pixels, `x` along columns) and `world_points`
3-element indexables `(X, Y, Z)` (physical units); typically these come from
a dot-grid target via [`detect_calibration_grid`](@ref) and
[`calibration_points`](@ref), imaged at several `Z` positions.

- `model = :soloff` (default): [`SoloffCamera`](@ref), 19-term polynomial;
  needs ≥ 19 points on ≥ 3 distinct Z planes.
- `model = :pinhole`: [`PinholeCamera`](@ref) via the normalized DLT; needs
  ≥ 6 non-coplanar points (≥ 2 Z planes).

Check the fit with [`calibration_quality`](@ref).
"""
function calibrate_camera(pixel_points::AbstractVector, world_points::AbstractVector;
                          model::Symbol = :soloff)
    length(pixel_points) == length(world_points) ||
        throw(ArgumentError("point counts differ: $(length(pixel_points)) pixel vs " *
                            "$(length(world_points)) world"))
    px = [SVector(Float64(p[1]), Float64(p[2])) for p in pixel_points]
    wd = [SVector(Float64(w[1]), Float64(w[2]), Float64(w[3])) for w in world_points]
    model === :pinhole && return fit_pinhole(px, wd)
    model === :soloff && return fit_soloff(px, wd)
    throw(ArgumentError("model must be :pinhole or :soloff, got :$model"))
end

"""
    reprojection_errors(cam::CameraCalibration, pixel_points, world_points) -> Vector{Float64}

Per-point Euclidean distance (pixels) between each measured pixel position
and the projection of its world point through `cam`.
"""
reprojection_errors(cam::CameraCalibration, pixel_points::AbstractVector,
                    world_points::AbstractVector) =
    [norm(world_to_pixel(cam, world_points[i]) -
          SVector(Float64(pixel_points[i][1]), Float64(pixel_points[i][2])))
     for i in eachindex(pixel_points, world_points)]

"""
    calibration_quality(cam::CameraCalibration, pixel_points, world_points)
        -> (rms, max, n)

Summary of the reprojection error of `cam` over the given correspondences:
root-mean-square and maximum error in pixels, and the number of points.
"""
function calibration_quality(cam::CameraCalibration, pixel_points::AbstractVector,
                             world_points::AbstractVector)
    e = reprojection_errors(cam, pixel_points, world_points)
    return (rms = sqrt(sum(abs2, e) / length(e)), max = maximum(e), n = length(e))
end
