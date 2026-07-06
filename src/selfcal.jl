# Disparity self-calibration (Wieneke 2005; Phase 5, slice 4): correct the
# residual misregistration between the calibration-target plane and the
# actual laser sheet. The two cameras' dewarped views of the *same* instant
# should coincide exactly; any residual disparity field means the sheet is
# offset or tilted from the assumed plane. The disparity vectors are
# triangulated to world points on the true sheet, a plane is fitted through
# them, and both camera models are transformed so that plane becomes the
# measurement plane — repeated until the disparity vanishes.
#
# Like the calibration it corrects, this is an offline Float64 island: it
# runs once per experiment, far from the image-correlation hot path.

"""
    SelfCalPass

Diagnostics of one measurement pass of [`self_calibrate`](@ref) (one entry
per disparity-map computation, in `report.passes`):

- `disparity_rms`, `disparity_median`: RMS and median magnitude of the
  disparity vectors (dewarped pixels) over valid, non-outlier grid points —
  measured *before* any correction of this pass. Wieneke (2005) reports
  well-corrected setups below 0.1 px.
- `n_vectors`: number of disparity vectors entering the statistics.
- `triangulation_rms`: RMS of the per-vector 4-equation triangulation
  residual (dewarped pixels) over the accepted vectors; `NaN` when the pass
  applied no correction. Large values with a small plane-fit residual
  indicate calibration errors rather than sheet misalignment.
- `plane`: the fitted sheet plane `z = a + b·X + c·Y` (world units, in the
  frame current at that pass), or `nothing` when the pass only measured
  (converged, or the trailing verification pass).
"""
struct SelfCalPass
    disparity_rms::Float64
    disparity_median::Float64
    n_vectors::Int
    triangulation_rms::Float64
    plane::Union{Nothing,NamedTuple{(:a, :b, :c),NTuple{3,Float64}}}
end

"""
    SelfCalibrationReport

Diagnostics returned by [`self_calibrate`](@ref).

# Fields
- `passes`: per-measurement diagnostics ([`SelfCalPass`](@ref)); entries with
  `plane === nothing` measured without correcting (the final verification).
- `converged`: whether the residual disparity RMS reached `tol`.
- `tol`: the convergence tolerance used (dewarped pixels).
- `R`, `t`: the cumulative rigid world transform — a point `w` of the
  corrected frame sits at `R * w + t` in the original calibration frame (the
  fitted sheet plane maps onto the corrected frame's `z = grid.z` plane).
- `disparity_maps`: the disparity [`PIVResult`](@ref) of every pass when
  `keep_disparity_maps = true` (camera-1 → camera-2 displacements in
  dewarped pixels); empty otherwise.
"""
struct SelfCalibrationReport
    passes::Vector{SelfCalPass}
    converged::Bool
    tol::Float64
    R::SMatrix{3,3,Float64,9}
    t::SVector{3,Float64}
    disparity_maps::Vector{PIVResult}
end

function Base.show(io::IO, r::SelfCalibrationReport)
    n = count(p -> p.plane !== nothing, r.passes)
    print(io, "SelfCalibrationReport(", n, n == 1 ? " correction" : " corrections",
          ", disparity RMS ", round(r.passes[1].disparity_rms; sigdigits = 3), " → ",
          round(r.passes[end].disparity_rms; sigdigits = 3), " px, ",
          r.converged ? "converged" : "not converged", ")")
end

# Paper recipe: a single pass with large windows and 50% overlap — the outer
# correction loop supplies the iteration. The window adapts down from the
# 128-px default when the dewarp grid is small.
function default_selfcal_parameters((ny, nx)::Tuple{Int,Int})
    w = min(128, prevpow(2, max(min(ny, nx) ÷ 2, 8)))
    return PIVParameters(window_size = w, overlap = w ÷ 2,
                         padding = true, apodization = :gauss)
end

# Triangulate the accepted disparity vectors to world points on the true
# measurement plane. The pattern correlated at a grid node is seen at
# `node ∓ d/2` by the two cameras (symmetric attribution, second-order exact;
# the outer iteration absorbs the rest), and camera i's viewing ray drifts
# in-plane by (tXᵢ, tYᵢ) per unit Z, so each vector yields four ray equations
# `X - ζ·tXᵢ = posᵢ` for the point `(X, Y, grid.z + ζ)`. Vectors whose
# residual exceeds `max_error` (dewarped px) are rejected as false
# correlations (Wieneke 2005 suggests 0.5 px).
function triangulate_disparities(disp::PIVResult, sel::AbstractMatrix{Bool},
                                 cam1::CameraCalibration, cam2::CameraCalibration,
                                 grid::DewarpGrid, max_error::Real)
    sx, sy = step(grid.x), step(grid.y)
    δ = max(abs(sx), abs(sy))
    pts = SVector{3,Float64}[]
    sse = 0.0
    for j in axes(disp.u, 2), i in axes(disp.u, 1)
        sel[i, j] || continue
        X0 = first(grid.x) + (Float64(disp.x[j]) - 1) * sx
        Y0 = first(grid.y) + (Float64(disp.y[i]) - 1) * sy
        t1 = ray_slopes(cam1, X0, Y0, grid.z, δ)
        t2 = ray_slopes(cam2, X0, Y0, grid.z, δ)
        (all(isfinite, t1) && all(isfinite, t2)) || continue
        dxw = Float64(disp.u[i, j]) * sx
        dyw = Float64(disp.v[i, j]) * sy
        A = @SMatrix [1.0 0.0 -t1[1];
                      0.0 1.0 -t1[2];
                      1.0 0.0 -t2[1];
                      0.0 1.0 -t2[2]]
        b = SVector(X0 - dxw / 2, Y0 - dyw / 2, X0 + dxw / 2, Y0 + dyw / 2)
        N = A' * A
        abs(det(N)) > 1e-10 || continue   # parallel rays carry no depth
        p = N \ (A' * b)
        r = A * p - b
        err = sqrt((abs2(r[1] / sx) + abs2(r[2] / sy) +
                    abs2(r[3] / sx) + abs2(r[4] / sy)) / 4)
        err <= max_error || continue
        push!(pts, SVector(p[1], p[2], grid.z + p[3]))
        sse += err^2
    end
    terr = isempty(pts) ? NaN : sqrt(sse / length(pts))
    return pts, terr
end

# Least-squares plane z = a + b·X + c·Y through the triangulated sheet points.
function fit_sheet_plane(pts::Vector{SVector{3,Float64}})
    n = length(pts)
    m = sum(pts) / n
    Sxx = Sxy = Syy = Sxz = Syz = 0.0
    for p in pts
        dx, dy, dz = p[1] - m[1], p[2] - m[2], p[3] - m[3]
        Sxx += dx * dx; Sxy += dx * dy; Syy += dy * dy
        Sxz += dx * dz; Syz += dy * dz
    end
    det2 = Sxx * Syy - Sxy^2
    det2 > 1e-12 * (Sxx + Syy)^2 ||
        throw(ArgumentError("triangulated sheet points do not span a plane — the " *
                            "disparity field is too sparse or degenerate to fit the " *
                            "light-sheet position"))
    b = (Sxz * Syy - Syz * Sxy) / det2
    c = (Syz * Sxx - Sxz * Sxy) / det2
    a = m[3] - b * m[1] - c * m[2]
    return a, b, c
end

# Rigid world transform (R, t) placing the fitted sheet plane z = a + b·X + c·Y
# at Z = grid.z of the corrected frame: w_current = R * w_corrected + t.
# Anchoring follows the paper: +Z along the sheet normal, +X the previous X
# axis projected onto the sheet, and the point camera 1 sees at the previous
# (0, 0, grid.z) stays fixed, so camera 1's dewarped image barely moves.
function plane_transform(a::Float64, b::Float64, c::Float64,
                         cam1::CameraCalibration, grid::DewarpGrid)
    n = SVector(-b, -c, 1.0) / sqrt(b^2 + c^2 + 1)
    ex = normalize(SVector(1.0, 0.0, 0.0) - n[1] * n)
    ey = cross(n, ex)
    R = hcat(ex, ey, n)
    gz = grid.z
    t1 = ray_slopes(cam1, 0.0, 0.0, gz, max(abs(step(grid.x)), abs(step(grid.y))))
    den = 1.0 - b * t1[1] - c * t1[2]
    q0 = if all(isfinite, t1) && abs(den) > 1e-6
        Z = (a - gz * (b * t1[1] + c * t1[2])) / den
        SVector((Z - gz) * t1[1], (Z - gz) * t1[2], Z)
    else
        SVector(0.0, 0.0, a)   # degenerate ray: keep the sheet point above the origin
    end
    return R, q0 - gz * n
end

"""
    self_calibrate(frames1, frames2, dw1, dw2; kwargs...)
        -> (dw1′, dw2′, report)

Disparity self-calibration (Wieneke 2005): correct residual misalignment
between the calibration-target plane and the actual laser sheet. The two
cameras' dewarped images of the *same* instant should coincide; the residual
disparity field (camera 1 vs camera 2, ensemble sum-of-correlation over all
instants) is triangulated to world points on the true sheet, a plane is
fitted through them, and both camera models are rigidly transformed so the
fitted plane becomes the measurement plane `z = grid.z` — repeated until the
disparity RMS drops below `tol` or `iterations` corrections have been
applied, then verified with a final disparity measurement.

`frames1` and `frames2` are vectors of raw same-instant frames of cameras 1
and 2 (file paths and/or matrices, as in [`run_piv_sequence`](@ref)); entry
`k` of both must show the same instant. Wieneke recommends 5–50 instants
for a well-shaped ensemble correlation peak; a single dense image per camera
also works and may be passed directly as `self_calibrate(A1, A2, dw1, dw2)`.
`dw1`/`dw2` are the cameras' [`ImageDewarper`](@ref)s sharing one
[`DewarpGrid`](@ref).

Returns replacement dewarpers built from the corrected cameras on the same
grid (drop-in for [`run_piv_stereo`](@ref); the corrected models are
`dw1′.cam`/`dw2′.cam` — a corrected [`PinholeCamera`](@ref) stays a
`PinholeCamera`, other models come back wrapped in a
[`TransformedCamera`](@ref)) and a [`SelfCalibrationReport`](@ref) with
per-pass diagnostics and the cumulative world transform. When the first
measurement is already below `tol`, the input dewarpers are returned as-is.

# Keyword arguments
- `params = nothing`: `PIVParameters` (or multi-pass schedule) for the
  disparity correlation; the default is the paper's recipe — a single pass
  of large windows (128 px or less on small grids) with 50% overlap,
  padded and Gaussian-apodized.
- `iterations = 3`: maximum number of plane-fit corrections.
- `tol = 0.05`: convergence threshold on the disparity RMS (dewarped px).
- `max_triangulation_error = 0.5`: reject disparity vectors whose
  triangulation residual exceeds this (dewarped px) as false correlations.
- `mask = nothing`: grid-sized `Bool` matrix of world-plane pixels to
  exclude (`true` = excluded), combined with the dewarpers' out-of-view
  masks as in [`run_piv_stereo`](@ref).
- `keep_disparity_maps = false`: retain every pass's disparity
  [`PIVResult`](@ref) in the report (for troubleshooting; off for
  production runs — the scalar diagnostics are always recorded).
- `preprocess = nothing`: function applied to each raw frame before
  dewarping (e.g. background subtraction).
- `image_type = Float64`: precision for frames loaded from file paths.
- `progress = false`: show the ensemble-correlation progress meter.
- Remaining keywords (`threaded`, ...) are forwarded to
  [`run_piv_ensemble`](@ref).
"""
function self_calibrate(frames1::AbstractVector, frames2::AbstractVector,
                        dw1::ImageDewarper, dw2::ImageDewarper;
                        params::Union{Nothing,PIVParameters,AbstractVector{PIVParameters}} = nothing,
                        iterations::Integer = 3,
                        tol::Real = 0.05,
                        max_triangulation_error::Real = 0.5,
                        mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                        keep_disparity_maps::Bool = false,
                        preprocess = nothing,
                        image_type::Type{<:AbstractFloat} = Float64,
                        progress::Bool = false,
                        kwargs...)
    dw1.grid == dw2.grid ||
        throw(ArgumentError("the two dewarpers must share the same DewarpGrid, " *
                            "got $(dw1.grid) and $(dw2.grid)"))
    length(frames1) == length(frames2) ||
        throw(ArgumentError("frames1 and frames2 must pair up instant by instant, " *
                            "got $(length(frames1)) and $(length(frames2)) frames"))
    isempty(frames1) && throw(ArgumentError("at least one frame per camera is required"))
    iterations >= 1 || throw(ArgumentError("iterations must be at least 1, got $iterations"))
    tol >= 0 || throw(ArgumentError("tol must be non-negative, got $tol"))
    max_triangulation_error > 0 ||
        throw(ArgumentError("max_triangulation_error must be positive, " *
                            "got $max_triangulation_error"))
    grid = dw1.grid
    mask === nothing || size(mask) == size(grid) ||
        throw(DimensionMismatch("mask size $(size(mask)) does not match the dewarp " *
                                "grid size $(size(grid))"))

    imgs1 = [load_frame(f, image_type) for f in frames1]
    imgs2 = [load_frame(f, image_type) for f in frames2]
    if preprocess !== nothing
        imgs1 = map(preprocess, imgs1)
        imgs2 = map(preprocess, imgs2)
    end
    for (imgs, dw, name) in ((imgs1, dw1, "camera 1"), (imgs2, dw2, "camera 2"))
        all(img -> size(img) == dw.image_size, imgs) ||
            throw(DimensionMismatch("$name frames must match the dewarper's image size " *
                                    "$(dw.image_size)"))
    end
    T = float(promote_type(mapreduce(eltype, promote_type, imgs1),
                           mapreduce(eltype, promote_type, imgs2)))
    p = params === nothing ? default_selfcal_parameters(size(grid)) : params

    d1, d2 = dw1, dw2
    Rcum = SMatrix{3,3,Float64}(I)
    tcum = zero(SVector{3,Float64})
    passes = SelfCalPass[]
    maps = PIVResult[]
    converged = false

    for it in 1:(Int(iterations) + 1)
        node_mask = d1.mask .| d2.mask
        mask === nothing || (node_mask .|= mask)
        pairs = [(dewarp!(Matrix{T}(undef, size(grid)), d1, imgs1[k]),
                  dewarp!(Matrix{T}(undef, size(grid)), d2, imgs2[k]))
                 for k in eachindex(imgs1)]
        disp = run_piv_ensemble(pairs, p; mask = node_mask, progress, kwargs...)
        keep_disparity_maps && push!(maps, disp)

        sel = .!disp.mask .& .!disp.outliers .& isfinite.(disp.u) .& isfinite.(disp.v)
        n = count(sel)
        n > 0 || throw(ArgumentError("no valid disparity vectors — check the stereo " *
                                     "overlap region, seeding density, and window size"))
        mag² = abs2.(disp.u[sel]) .+ abs2.(disp.v[sel])
        rms = sqrt(sum(mag²) / n)
        med = median(sqrt.(mag²))

        if rms <= tol || it > iterations
            push!(passes, SelfCalPass(rms, med, n, NaN, nothing))
            converged = rms <= tol
            break
        end

        pts, terr = triangulate_disparities(disp, sel, d1.cam, d2.cam, grid,
                                            max_triangulation_error)
        length(pts) >= 3 ||
            throw(ArgumentError("only $(length(pts)) disparity vectors survived " *
                                "triangulation (need ≥ 3) — the disparity field is too " *
                                "noisy; add instants or increase the window size"))
        a, b, c = fit_sheet_plane(pts)
        Rp, tp = plane_transform(a, b, c, d1.cam, grid)
        d1 = ImageDewarper(apply_world_transform(d1.cam, Rp, tp), grid, d1.image_size)
        d2 = ImageDewarper(apply_world_transform(d2.cam, Rp, tp), grid, d2.image_size)
        tcum = Rcum * tp + tcum
        Rcum = Rcum * Rp
        push!(passes, SelfCalPass(rms, med, n, terr, (a = a, b = b, c = c)))
    end

    report = SelfCalibrationReport(passes, converged, Float64(tol), Rcum, tcum, maps)
    return d1, d2, report
end

self_calibrate(A1::AbstractMatrix{<:Real}, A2::AbstractMatrix{<:Real},
               dw1::ImageDewarper, dw2::ImageDewarper; kwargs...) =
    self_calibrate([A1], [A2], dw1, dw2; kwargs...)
