# Calibration-target (dot grid) detection: turns plate images into the
# (pixel, world) point pairs that calibrate_camera consumes. Like
# calibration.jl, this is an offline Float64 island.
#
# Supported targets: rectilinear dot grids (e.g. LaVision plates), single- or
# two-level. On a two-level plate the back-level dots sit at half-spacing
# diagonal offsets from the front-level dots, so the combined set forms one
# 45°-rotated square lattice with spacing `spacing/√2`; indexing works on that
# combined lattice and recovers the level from index parity. Optional fiducial
# markers: a filled square (origin anchor) and a filled upward-pointing
# triangle (reported for diagnostics).
#
# World axes are anchored to the image: +X is the lattice direction closest to
# image-right and +Y the one closest to image-up (decreasing row). This is
# consistent across cameras as long as they are mounted roughly upright (no
# large roll), which covers standard stereo rigs; rolled cameras would need an
# explicit orientation convention on top.

"""
    CalibrationGrid

Detected calibration-target dot grid (see [`detect_calibration_grid`](@ref)).

# Fields
- `pixels`: dot centroids in pixel coordinates (`x` along columns, `y` along
  rows), intensity-weighted, subpixel.
- `indices`: integer dot positions in units of **half** the dot spacing along
  the world X/Y axes; the origin dot is `(0, 0)`. Front-level dots have two
  even indices, back-level dots two odd indices.
- `level`: `0` for dots on the reference ("front") level, `1` for the other
  level (all zero for single-level targets).
- `spacing`: dot spacing on one level, world units.
- `level_separation`: Z offset of level 1 behind level 0, world units.
- `square`, `triangle`: fiducial marker centroids in pixel coordinates, or
  `nothing` when not found.
"""
struct CalibrationGrid
    pixels::Vector{SVector{2,Float64}}
    indices::Vector{NTuple{2,Int}}
    level::Vector{Int8}
    spacing::Float64
    level_separation::Float64
    square::Union{Nothing,SVector{2,Float64}}
    triangle::Union{Nothing,SVector{2,Float64}}
end

function Base.show(io::IO, g::CalibrationGrid)
    print(io, "CalibrationGrid($(length(g.pixels)) dots",
          any(!iszero, g.level) ? ", two-level" : "",
          g.square === nothing ? "" : ", square marker",
          g.triangle === nothing ? "" : ", triangle marker", ")")
end

# --- Thresholding and blob extraction ---------------------------------------

# Otsu's method on a 256-bin histogram; returns a threshold in intensity units.
function _otsu_threshold(A::AbstractMatrix{<:Real})
    lo, hi = extrema(A)
    hi > lo || return Float64(lo)
    nbins = 256
    counts = zeros(Int, nbins)
    scale = nbins / (Float64(hi) - Float64(lo))
    for a in A
        bin = clamp(floor(Int, (Float64(a) - lo) * scale) + 1, 1, nbins)
        counts[bin] += 1
    end
    total = length(A)
    sum_all = sum(counts[b] * b for b in 1:nbins)
    w0, sum0 = 0, 0.0
    best, best_bin = -1.0, 1
    for b in 1:nbins-1
        w0 += counts[b]
        (w0 == 0 || w0 == total) && continue
        sum0 += counts[b] * b
        μ0 = sum0 / w0
        μ1 = (sum_all - sum0) / (total - w0)
        between = Float64(w0) * (total - w0) * (μ0 - μ1)^2
        between > best && ((best, best_bin) = (between, b))
    end
    return Float64(lo) + best_bin / scale
end

struct _Blob
    centroid::SVector{2,Float64}  # (x, y), intensity-weighted
    area::Int
    bbox::NTuple{4,Int}           # (rmin, rmax, cmin, cmax)
    fill::Float64                 # area / bounding-box area
    offset::Float64               # |centroid - bbox center| / bbox extent
end

# Label 8-connected foreground components (BFS flood fill) and compute
# intensity-weighted blob statistics, weighting by the intensity above the
# threshold so the background pedestal does not bias centroids.
function _detect_blobs(A::AbstractMatrix{Float64}, thresh::Float64)
    rows, cols = size(A)
    labels = zeros(Int32, rows, cols)
    blobs = _Blob[]
    queue = Vector{Tuple{Int,Int}}()
    for c0 in 1:cols, r0 in 1:rows
        (A[r0, c0] > thresh && labels[r0, c0] == 0) || continue
        label = Int32(length(blobs) + 1)
        labels[r0, c0] = label
        empty!(queue)
        push!(queue, (r0, c0))
        area = 0
        wsum = wx = wy = 0.0
        rmin, rmax, cmin, cmax = r0, r0, c0, c0
        while !isempty(queue)
            r, c = pop!(queue)
            area += 1
            w = A[r, c] - thresh
            wsum += w
            wx += w * c
            wy += w * r
            rmin = min(rmin, r); rmax = max(rmax, r)
            cmin = min(cmin, c); cmax = max(cmax, c)
            for dc in -1:1, dr in -1:1
                rn, cn = r + dr, c + dc
                (1 <= rn <= rows && 1 <= cn <= cols) || continue
                (A[rn, cn] > thresh && labels[rn, cn] == 0) || continue
                labels[rn, cn] = label
                push!(queue, (rn, cn))
            end
        end
        h, w_ = rmax - rmin + 1, cmax - cmin + 1
        centroid = SVector(wx / wsum, wy / wsum)
        bbcenter = SVector((cmin + cmax) / 2, (rmin + rmax) / 2)
        offset = norm(centroid - bbcenter) / max(h, w_)
        push!(blobs, _Blob(centroid, area, (rmin, rmax, cmin, cmax),
                           area / (h * w_), offset))
    end
    return blobs
end

# Split blobs into grid dots and fiducial markers. Discriminators (stable
# under moderate perspective): a filled circle fills ~π/4 ≈ 0.79 of its
# bounding box with a centered centroid; a grid-aligned filled square fills
# ~1.0; a filled upward triangle fills ~0.5 with its centroid offset from the
# bounding-box center by ~1/6 of the extent. Blobs touching the image border
# (clipped dots) and small or misshapen blobs (damaged dots, specks) are
# dropped.
function _classify_blobs(blobs::Vector{_Blob}, imgsize::Tuple{Int,Int}, min_area::Int)
    rows, cols = imgsize
    interior = [b for b in blobs if b.bbox[1] > 1 && b.bbox[2] < rows &&
                                    b.bbox[3] > 1 && b.bbox[4] < cols &&
                                    b.area >= min_area]
    isempty(interior) && throw(ArgumentError("no calibration dots detected — check " *
                                             "`threshold`/`invert` and the image"))
    typical = median([b.area for b in interior])
    big = [b for b in interior if b.area >= 0.15 * typical]

    squares = [b for b in big if b.fill >= 0.88 && b.offset <= 0.06]
    square = isempty(squares) ? nothing : argmax(b -> b.fill, squares)
    triangles = [b for b in big if b.fill <= 0.70 && b.offset >= 0.10 && b !== square]
    triangle = isempty(triangles) ? nothing : argmax(b -> b.offset, triangles)

    dots = [b.centroid for b in big
            if b !== square && b !== triangle && 0.55 <= b.fill && b.offset <= 0.15]
    return dots, (square === nothing ? nothing : square.centroid),
                 (triangle === nothing ? nothing : triangle.centroid)
end

# --- Lattice indexing --------------------------------------------------------

_happly(H::SMatrix{3,3,Float64}, p) =
    (h = H * SVector(Float64(p[1]), Float64(p[2]), 1.0); SVector(h[1] / h[3], h[2] / h[3]))

# Least-squares 2D homography src → dst (normalized DLT); `nothing` when the
# source points are degenerate (too few or collinear).
function _fit_homography(src::Vector{SVector{2,Float64}}, dst::Vector{SVector{2,Float64}})
    N = length(src)
    N >= 4 || return nothing
    (length(unique(p[1] for p in src)) >= 2 && length(unique(p[2] for p in src)) >= 2) ||
        return nothing
    Ts = _normalization_matrix(src)
    Td = _normalization_matrix(dst)
    A = zeros(2N, 9)
    for i in 1:N
        x = Ts[1, 1] * src[i][1] + Ts[1, 3]
        y = Ts[2, 2] * src[i][2] + Ts[2, 3]
        u = Td[1, 1] * dst[i][1] + Td[1, 3]
        v = Td[2, 2] * dst[i][2] + Td[2, 3]
        A[2i-1, :] .= (x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u)
        A[2i, :] .= (0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v)
    end
    h = svd(A).V[:, end]
    Hn = transpose(reshape(h, 3, 3))
    return SMatrix{3,3,Float64}(Td \ (Hn * Ts))
end

# Assign integer lattice indices to dot centroids: establish a local basis at
# a central seed dot, then grow outward matching dots near predicted
# positions, refitting a global homography as coverage builds so perspective
# and smooth distortion are tracked. Predictions are anchored on the measured
# position of the already-indexed dot plus a model step, so only the local
# step needs to be accurate (on two-level plates this also absorbs the small
# alternating parallax between the front- and back-level sublattices).
function _index_dots(pts::Vector{SVector{2,Float64}})
    n = length(pts)
    n >= 6 || throw(ArgumentError("too few grid dots detected ($n) to index a lattice"))
    ctr = sum(pts) / n
    seed = argmin(i -> norm(pts[i] - ctr), 1:n)
    dnn = median([minimum(norm(pts[i] - pts[j]) for j in 1:n if j != i) for i in 1:n])

    neigh = sort!([j for j in 1:n if j != seed && norm(pts[j] - pts[seed]) < 1.7 * dnn];
                  by = j -> norm(pts[j] - pts[seed]))
    isempty(neigh) && throw(ArgumentError("could not establish a grid basis: seed dot has no neighbors"))
    e1 = pts[neigh[1]] - pts[seed]
    e2 = nothing
    for j in neigh[2:end]
        v = pts[j] - pts[seed]
        if abs(e1[1] * v[2] - e1[2] * v[1]) / (norm(e1) * norm(v)) > 0.5
            e2 = v
            break
        end
    end
    e2 === nothing &&
        throw(ArgumentError("could not establish a second grid axis around the seed dot"))

    affine = AffineMap(hcat(e1, e2), pts[seed])  # index (i, j) → pixel
    H = nothing
    predict(q) = H === nothing ? affine(SVector(Float64(q[1]), Float64(q[2]))) :
                                 _happly(H, q)

    index_of = Dict{Int,NTuple{2,Int}}(seed => (0, 0))
    dot_at = Dict{NTuple{2,Int},Int}((0, 0) => seed)
    for _ in 1:8
        added_this_round = false
        while true  # sweep to a fixpoint under the current predictor
            added = false
            for (id, q) in collect(index_of), δ in ((1, 0), (-1, 0), (0, 1), (0, -1))
                qn = (q[1] + δ[1], q[2] + δ[2])
                haskey(dot_at, qn) && continue
                step = predict(qn) - predict(q)
                pp = pts[id] + step
                tol = 0.35 * norm(step)
                best, bestd = 0, tol
                for j in 1:n
                    haskey(index_of, j) && continue
                    d = norm(pts[j] - pp)
                    d < bestd && ((best, bestd) = (j, d))
                end
                best == 0 && continue
                index_of[best] = qn
                dot_at[qn] = best
                added = added_this_round = true
            end
            added || break
        end
        had_H = H !== nothing
        ids = collect(keys(index_of))
        Hfit = length(ids) >= 8 ?
            _fit_homography([SVector(Float64(index_of[i][1]), Float64(index_of[i][2])) for i in ids],
                            [pts[i] for i in ids]) : nothing
        Hfit !== nothing && (H = Hfit)
        (added_this_round || !had_H && H !== nothing) || break
    end
    return index_of, predict
end

# --- Grid detection ----------------------------------------------------------

"""
    detect_calibration_grid(img; spacing, kwargs...) -> CalibrationGrid

Detect a rectilinear calibration dot grid in `img` (grayscale, any real
element type) and index it: dots are found as bright blobs (intensity-weighted
subpixel centroids), assigned integer lattice positions, and anchored to a
world coordinate frame. World +X is the lattice direction closest to
image-right, +Y the one closest to image-up; the origin dot is chosen from
the square fiducial marker when present (see `origin_offset`), otherwise the
dot nearest the image center. Dots clipped by the image border are excluded.

# Keywords
- `spacing` (required): dot spacing on one level, world units (e.g. mm).
- `two_level = false`: two-level plate (back-level dots at half-spacing
  diagonal offsets, e.g. LaVision type 21/31 plates). Requires
  `level_separation`.
- `level_separation = 0.0`: Z distance by which level 1 sits behind the
  reference level (level 1 world Z = `z - level_separation` in
  [`calibration_points`](@ref); may be negative if the plate faces the other
  way).
- `origin_level = :front`: which level the origin dot belongs to; that level
  becomes level 0, the Z reference.
- `origin_offset = nothing`: world-units `(ΔX, ΔY)` from the square marker
  centroid to the origin dot (e.g. `(30.0, 7.5)` for PIV Challenge case 4E:
  "the dot 30 mm right of and 7.5 mm above the square marker"). When
  `nothing`, the dot nearest the square marker (or the image center when no
  marker is found) becomes the origin. A consistent origin across cameras
  requires the marker.
- `invert = false`: set for dark dots on a bright background.
- `threshold = :otsu`: foreground threshold — `:otsu` or an absolute value in
  image intensity units.
- `min_area = 9`: minimum blob area in pixels; blobs smaller than 15% of the
  median blob area are dropped as well.
"""
function detect_calibration_grid(img::AbstractMatrix{<:Real};
                                 spacing::Real,
                                 two_level::Bool = false,
                                 level_separation::Real = 0.0,
                                 origin_level::Symbol = :front,
                                 origin_offset::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
                                 invert::Bool = false,
                                 threshold = :otsu,
                                 min_area::Int = 9)
    spacing > 0 || throw(ArgumentError("spacing must be positive, got $spacing"))
    origin_level in (:front, :back) ||
        throw(ArgumentError("origin_level must be :front or :back, got :$origin_level"))
    two_level && iszero(level_separation) &&
        throw(ArgumentError("a two-level target needs a nonzero level_separation"))

    A = Float64.(img)
    invert && (A .= maximum(A) .- A)
    t = threshold === :otsu ? _otsu_threshold(A) :
        threshold isa Real ? Float64(threshold) :
        throw(ArgumentError("threshold must be :otsu or a number, got $threshold"))

    blobs = _detect_blobs(A, t)
    dots, square, triangle = _classify_blobs(blobs, size(A), min_area)
    index_of, predict = _index_dots(dots)

    # Orient: express indices in half-spacing units along image-anchored axes.
    ids = collect(keys(index_of))
    qs = [index_of[i] for i in ids]
    q0 = (round(Int, sum(q[1] for q in qs) / length(qs)),
          round(Int, sum(q[2] for q in qs) / length(qs)))
    candidates = two_level ? ((1, 1), (1, -1), (-1, -1), (-1, 1)) :
                             ((1, 0), (0, 1), (-1, 0), (0, -1))
    pdir(c) = normalize(predict(q0 .+ c) - predict(q0))
    c_x = argmax(c -> pdir(c)[1], candidates)                       # image-right
    c_y = argmax(c -> -pdir(c)[2],
                 filter(c -> c != c_x && c != (-c_x[1], -c_x[2]), collect(candidates)))  # image-up
    half = two_level ? 1 : 2
    tocoord(qi) = (half * (qi[1] * c_x[1] + qi[2] * c_x[2]),
                   half * (qi[1] * c_y[1] + qi[2] * c_y[2]))
    coords = Dict(i => tocoord(index_of[i]) for i in ids)

    # Anchor the origin dot.
    origin_id = 0
    if square !== nothing && origin_offset !== nothing
        # Fractional lattice coordinates of the marker via the inverse predictor.
        qm = _invert_predictor(predict, square, index_of, dots)
        am = tocoord(qm)
        target = (am[1] + 2 * Float64(origin_offset[1]) / spacing,
                  am[2] + 2 * Float64(origin_offset[2]) / spacing)
        best, bestd = 0, 0.45
        for i in ids
            d = hypot(coords[i][1] - target[1], coords[i][2] - target[2])
            d < bestd && ((best, bestd) = (i, d))
        end
        best != 0 || throw(ArgumentError("no dot found at origin_offset = $origin_offset " *
                                         "from the square marker"))
        origin_id = best
    elseif square !== nothing
        origin_id = argmin(i -> norm(dots[i] - square), ids)
    else
        origin_offset === nothing ||
            throw(ArgumentError("origin_offset given but no square marker was detected"))
        imgctr = SVector(size(A, 2) / 2, size(A, 1) / 2)
        origin_id = argmin(i -> norm(dots[i] - imgctr), ids)
    end

    a0 = coords[origin_id]
    pixels = SVector{2,Float64}[]
    indices = NTuple{2,Int}[]
    level = Int8[]
    other = origin_level === :front ? Int8(1) : Int8(0)
    for i in ids
        a = (round(Int, coords[i][1] - a0[1]), round(Int, coords[i][2] - a0[2]))
        push!(pixels, dots[i])
        push!(indices, a)
        push!(level, two_level && isodd(a[1]) ? other : Int8(1) - other)
    end
    two_level || fill!(level, 0)
    return CalibrationGrid(pixels, indices, level, Float64(spacing),
                           Float64(level_separation), square, triangle)
end

# Fractional lattice position of an arbitrary pixel point: invert the final
# predictor numerically by Newton iteration seeded from the nearest indexed dot.
function _invert_predictor(predict, p::SVector{2,Float64},
                           index_of::Dict{Int,NTuple{2,Int}}, dots)
    nearest = argmin(i -> norm(dots[i] - p), collect(keys(index_of)))
    q = SVector(Float64(index_of[nearest][1]), Float64(index_of[nearest][2]))
    for _ in 1:30
        r = predict(q) - p
        norm(r) < 1e-9 && break
        # Numerical Jacobian of the predictor (columns: ∂pixel/∂i, ∂pixel/∂j).
        h = 1e-3
        Ji = (predict(q + SVector(h, 0.0)) - predict(q - SVector(h, 0.0))) / 2h
        Jj = (predict(q + SVector(0.0, h)) - predict(q - SVector(0.0, h))) / 2h
        J = hcat(Ji, Jj)
        abs(det(J)) > 1e-12 || break
        q -= J \ r
    end
    return (q[1], q[2])
end

"""
    calibration_points(grid::CalibrationGrid, z) -> (pixel_points, world_points)

Convert a detected grid into calibration point pairs: pixel centroids and
world `(X, Y, Z)` positions with the reference level of the plate at
out-of-plane position `z` (level-1 dots sit at `z - level_separation`).
Concatenate the pairs from several plate positions and feed them to
[`calibrate_camera`](@ref), or use the `calibrate_camera(grids, zs)`
convenience method.
"""
function calibration_points(grid::CalibrationGrid, z::Real)
    world = [SVector(grid.indices[i][1] * grid.spacing / 2,
                     grid.indices[i][2] * grid.spacing / 2,
                     Float64(z) - grid.level[i] * grid.level_separation)
             for i in eachindex(grid.indices)]
    return grid.pixels, world
end

"""
    calibrate_camera(grids::AbstractVector{CalibrationGrid}, zs; model = :soloff)

Calibrate a camera from dot grids detected at several plate positions `zs`
(reference-level Z of each grid, world units).
"""
function calibrate_camera(grids::AbstractVector{CalibrationGrid},
                          zs::AbstractVector{<:Real}; model::Symbol = :soloff)
    length(grids) == length(zs) ||
        throw(ArgumentError("got $(length(grids)) grids but $(length(zs)) z positions"))
    pixels = SVector{2,Float64}[]
    world = SVector{3,Float64}[]
    for (g, z) in zip(grids, zs)
        px, wd = calibration_points(g, z)
        append!(pixels, px)
        append!(world, wd)
    end
    return calibrate_camera(pixels, world; model)
end

# --- Synthetic target rendering (test fixture for the stereo rig) ------------

# Rasterize a world-plane shape into the image through `cam`: for every pixel
# near the shape's projection, supersampled pixel points are mapped back to
# the plane at `z` and tested against `inside(X, Y)`.
function _render_shape!(img::Matrix{Float64}, cam::CameraCalibration, z::Real,
                        inside, center::SVector{2,Float64}, radius::Real,
                        intensity::Real, ss::Int)
    c3 = SVector(center[1], center[2], Float64(z))
    p0 = world_to_pixel(cam, c3)
    rpx = max(norm(world_to_pixel(cam, c3 + SVector(radius, 0.0, 0.0)) - p0),
              norm(world_to_pixel(cam, c3 + SVector(0.0, radius, 0.0)) - p0))
    isfinite(rpx) || return img
    cmin = max(1, floor(Int, p0[1] - rpx - 2)); cmax = min(size(img, 2), ceil(Int, p0[1] + rpx + 2))
    rmin = max(1, floor(Int, p0[2] - rpx - 2)); rmax = min(size(img, 1), ceil(Int, p0[2] + rpx + 2))
    for c in cmin:cmax, r in rmin:rmax
        hits = 0
        for kc in 1:ss, kr in 1:ss
            x = c - 0.5 + (kc - 0.5) / ss
            y = r - 0.5 + (kr - 0.5) / ss
            w = pixel_to_world(cam, SVector(x, y), z)
            (isfinite(w[1]) && inside(w[1], w[2])) && (hits += 1)
        end
        img[r, c] += intensity * hits / ss^2
    end
    return img
end

"""
    render_calibration_target(cam::CameraCalibration, image_size; spacing, kwargs...)
        -> Matrix{Float64}

Render a synthetic calibration-plate image as seen through `cam`: the
ground-truth fixture for testing target detection and calibration fitting
(and, downstream, the stereo reconstruction chain). Dots are anti-aliased by
supersampled back-projection onto the plate plane(s), so each rendered dot's
centroid is the exact projection of its world-lattice position.

# Keywords
- `spacing` (required): dot spacing on one level, world units.
- `z = 0.0`: world Z of the front (reference) level.
- `dot_diameter = 0.21 * spacing`: dot diameter, world units.
- `two_level = false`, `level_separation = 0.0`: back level at
  `z - level_separation`, dots offset by half a spacing in X and Y.
- `dot_intensity = 0.85`, `back_intensity = 0.65`, `background = 0.05`.
- `marker_square`, `marker_triangle`: world `(X, Y)` centers of fiducial
  markers on the front level, or `nothing`.
- `marker_size = 1.1 * dot_diameter`: marker side length, world units.
- `supersample = 3`: antialiasing subsamples per pixel axis.
"""
function render_calibration_target(cam::CameraCalibration, image_size::Tuple{Int,Int};
                                   spacing::Real,
                                   z::Real = 0.0,
                                   dot_diameter::Real = 0.21 * spacing,
                                   two_level::Bool = false,
                                   level_separation::Real = 0.0,
                                   dot_intensity::Real = 0.85,
                                   back_intensity::Real = 0.65,
                                   background::Real = 0.05,
                                   marker_square::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
                                   marker_triangle::Union{Nothing,Tuple{<:Real,<:Real}} = nothing,
                                   marker_size::Real = 1.1 * dot_diameter,
                                   supersample::Int = 3)
    spacing > 0 || throw(ArgumentError("spacing must be positive, got $spacing"))
    two_level && iszero(level_separation) &&
        throw(ArgumentError("a two-level target needs a nonzero level_separation"))
    img = fill(Float64(background), image_size)
    r = dot_diameter / 2

    levels = [(Float64(z), Float64(dot_intensity), 0.0)]
    two_level &&
        push!(levels, (Float64(z) - level_separation, Float64(back_intensity), spacing / 2))

    for (zl, intensity, off) in levels
        # World extent of the visible plane: back-project a border sample grid.
        xs = Float64[]; ys = Float64[]
        for fx in (0.0, 0.5, 1.0), fy in (0.0, 0.5, 1.0)
            w = pixel_to_world(cam, SVector(0.5 + fx * image_size[2], 0.5 + fy * image_size[1]), zl)
            isfinite(w[1]) && (push!(xs, w[1]); push!(ys, w[2]))
        end
        length(xs) >= 4 ||
            throw(ArgumentError("could not back-project the image extent onto the target plane"))
        irange = ceil(Int, (minimum(xs) - off - r) / spacing):floor(Int, (maximum(xs) - off + r) / spacing)
        jrange = ceil(Int, (minimum(ys) - off - r) / spacing):floor(Int, (maximum(ys) - off + r) / spacing)
        for j in jrange, i in irange
            cx, cy = i * spacing + off, j * spacing + off
            _render_shape!(img, cam, zl, (X, Y) -> (X - cx)^2 + (Y - cy)^2 <= r^2,
                           SVector(cx, cy), r, intensity, supersample)
        end
    end

    s = Float64(marker_size)
    if marker_square !== nothing
        cx, cy = Float64.(marker_square)
        _render_shape!(img, cam, z, (X, Y) -> max(abs(X - cx), abs(Y - cy)) <= s / 2,
                       SVector(cx, cy), s, dot_intensity, supersample)
    end
    if marker_triangle !== nothing
        cx, cy = Float64.(marker_triangle)
        _render_shape!(img, cam, z,
                       (X, Y) -> Y >= cy - s / 2 && 2 * abs(X - cx) <= (cy + s / 2 - Y),
                       SVector(cx, cy), s, dot_intensity, supersample)
    end
    return clamp!(img, 0.0, 1.0)
end
