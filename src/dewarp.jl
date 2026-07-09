# Image dewarping / back-projection onto a common world-plane grid (Phase 5,
# slice 2). Each camera's image is resampled onto a regular grid of world
# coordinates so the 2D correlation engine can run per camera on
# geometrically aligned images.
#
# The mapper's coordinate map is a deliberate Float64 island: it is built
# once per camera from the (offline, Float64) calibration and reused across
# every frame. The resampling itself follows the images' precision.

"""
    DewarpGrid(; x, y, z = 0.0)

Regular world-plane pixel grid for image dewarping: `x` and `y` are ranges of
world coordinates (physical units, e.g. mm) spanning the measurement region,
and `z` is the out-of-plane position of the measurement plane. The grid's
resolution is `step(x)` / `step(y)` world units per dewarped pixel; choose it
finer than the target vector spacing, since correlation windows live on the
dewarped images.

A dewarped image is indexed `[row, col]` with `row` running along `y` and
`col` along `x` **in the order given**: `out[r, c]` shows the world point
`(x[c], y[r], z)`. With an ascending `y`, world +Y therefore points down the
image; pass a descending `y` range for a display-oriented (+Y up) image.
Displacements measured on dewarped images convert to world units as
`du * step(x)`, `dv * step(y)` — signs included.

One `DewarpGrid` is shared by all cameras of a stereo rig (build one
[`ImageDewarper`](@ref) per camera on it), and the stereo vector grid is
later derived from it.
"""
struct DewarpGrid
    x::LinRange{Float64,Int}
    y::LinRange{Float64,Int}
    z::Float64

    function DewarpGrid(; x::AbstractRange{<:Real}, y::AbstractRange{<:Real},
                        z::Real = 0.0)
        for (name, r) in ((:x, x), (:y, y))
            length(r) >= 2 ||
                throw(ArgumentError("grid range $name needs at least 2 nodes, got $(length(r))"))
            isfinite(first(r)) && isfinite(last(r)) ||
                throw(ArgumentError("grid range $name has non-finite endpoints"))
            first(r) != last(r) ||
                throw(ArgumentError("grid range $name has zero extent"))
        end
        isfinite(z) || throw(ArgumentError("grid z must be finite, got $z"))
        return new(LinRange{Float64,Int}(first(x), last(x), length(x)),
                   LinRange{Float64,Int}(first(y), last(y), length(y)), Float64(z))
    end
end

"""
    size(grid::DewarpGrid) -> (ny, nx)

Size of the dewarped image produced on `grid`: `(length(grid.y), length(grid.x))`.
"""
Base.size(grid::DewarpGrid) = (length(grid.y), length(grid.x))

Base.show(io::IO, g::DewarpGrid) =
    print(io, "DewarpGrid(x = ", first(g.x), ":", round(step(g.x); sigdigits = 6), ":",
          last(g.x), ", y = ", first(g.y), ":", round(step(g.y); sigdigits = 6), ":",
          last(g.y), ", z = ", g.z, ")")

"""
    common_dewarp_grid(cameras, image_size, z = 0.0;
                       spacing = :auto, margin = 0.0,
                       coverage = :intersection) -> DewarpGrid

Construct a [`DewarpGrid`](@ref) covering the world-plane region that a set of
cameras jointly image at out-of-plane position `z` — the grid-construction
step of a stereo (or multi-camera) analysis, without hand-tuning extents.

`cameras` is an iterable of [`CameraCalibration`](@ref)s; `image_size` is the
pixel size `(rows, cols)` used for every camera, or a vector of per-camera
sizes. Each camera's image border is projected to the `z` plane
([`pixel_to_world`](@ref), ~50 samples per edge) and reduced to an
axis-aligned world bounding box.

- `coverage = :intersection` (default — the stereo case) keeps only the
  region **all** cameras see; `:union` keeps everything **any** camera sees
  (out-of-view nodes are still flagged per camera in [`ImageDewarper`](@ref)'s
  `mask`). An empty intersection throws.
- `margin` (world units) shrinks the region on all sides when positive, grows
  it when negative.
- `spacing = :auto` estimates each camera's median world-units-per-pixel over
  its footprint and uses the **coarsest** camera's value, so no camera is
  resampled below its native resolution; pass a `Real` to set it directly.

The `y` range is built **descending** so the dewarped image displays upright
(world +Y up); PIV downstream is orientation-agnostic (`dv * step(y)` carries
the sign — see [`DewarpGrid`](@ref)).
"""
function common_dewarp_grid(cameras, image_size, z::Real = 0.0;
                            spacing::Union{Symbol,Real} = :auto,
                            margin::Real = 0.0,
                            coverage::Symbol = :intersection)
    cams = collect(cameras)
    isempty(cams) && throw(ArgumentError("at least one camera is required"))
    coverage in (:intersection, :union) ||
        throw(ArgumentError("coverage must be :intersection or :union, got :$coverage"))
    sizes = image_size isa Tuple{Integer,Integer} ?
            [(Int(image_size[1]), Int(image_size[2])) for _ in cams] :
            [(Int(s[1]), Int(s[2])) for s in image_size]
    length(sizes) == length(cams) ||
        throw(ArgumentError("image_size must be one (rows, cols) or one per camera, " *
                            "got $(length(sizes)) for $(length(cams)) cameras"))

    boxes = [_camera_world_bbox(cam, sz, Float64(z)) for (cam, sz) in zip(cams, sizes)]
    if coverage === :intersection
        xlo = maximum(b[1][1] for b in boxes); xhi = minimum(b[1][2] for b in boxes)
        ylo = maximum(b[2][1] for b in boxes); yhi = minimum(b[2][2] for b in boxes)
    else
        xlo = minimum(b[1][1] for b in boxes); xhi = maximum(b[1][2] for b in boxes)
        ylo = minimum(b[2][1] for b in boxes); yhi = maximum(b[2][2] for b in boxes)
    end
    xlo += margin; xhi -= margin; ylo += margin; yhi -= margin
    (xlo < xhi && ylo < yhi) ||
        throw(ArgumentError("empty dewarp region (coverage = :$coverage, margin = $margin): " *
                            "x = [$xlo, $xhi], y = [$ylo, $yhi]"))

    st = spacing === :auto ? _auto_dewarp_spacing(cams, sizes, Float64(z)) :
         spacing isa Real ? Float64(spacing) :
         throw(ArgumentError("spacing must be :auto or a positive real, got :$spacing"))
    st > 0 || throw(ArgumentError("spacing must be positive, got $st"))

    nx = max(2, round(Int, (xhi - xlo) / st) + 1)
    ny = max(2, round(Int, (yhi - ylo) / st) + 1)
    # Descending y → world +Y up in the displayed dewarped image.
    return DewarpGrid(x = range(xlo, xhi; length = nx),
                      y = range(yhi, ylo; length = ny), z = z)
end

# Axis-aligned world bounding box of one camera's image border at plane z.
function _camera_world_bbox(cam::CameraCalibration, image_size::Tuple{Int,Int}, z::Float64)
    nr, nc = image_size
    npts = 50
    xs = Float64[]; ys = Float64[]
    project!(pxl) = begin
        w = pixel_to_world(cam, pxl, z)
        (isfinite(w[1]) && isfinite(w[2])) && (push!(xs, w[1]); push!(ys, w[2]))
    end
    for c in range(1.0, Float64(nc); length = npts)
        project!((c, 1.0)); project!((c, Float64(nr)))
    end
    for r in range(1.0, Float64(nr); length = npts)
        project!((1.0, r)); project!((Float64(nc), r))
    end
    isempty(xs) &&
        throw(ArgumentError("camera footprint at z = $z is empty (no finite border projections)"))
    return (extrema(xs), extrema(ys))
end

# Median world-units-per-pixel over the coarsest camera's footprint (central
# differences of the back-projection at a coarse interior sample grid).
function _auto_dewarp_spacing(cams, sizes, z::Float64)
    scales = Float64[]
    for (cam, (nr, nc)) in zip(cams, sizes)
        per_cam = Float64[]
        for c in range(2.0, Float64(nc - 1); length = 5), r in range(2.0, Float64(nr - 1); length = 5)
            dc = pixel_to_world(cam, (c + 1.0, r), z) .- pixel_to_world(cam, (c - 1.0, r), z)
            dr = pixel_to_world(cam, (c, r + 1.0), z) .- pixel_to_world(cam, (c, r - 1.0), z)
            sx = hypot(dc[1], dc[2]) / 2
            sy = hypot(dr[1], dr[2]) / 2
            (isfinite(sx) && isfinite(sy)) && (push!(per_cam, sx); push!(per_cam, sy))
        end
        isempty(per_cam) || push!(scales, median(per_cam))
    end
    isempty(scales) &&
        throw(ArgumentError("could not estimate an automatic spacing (no finite footprint samples)"))
    return maximum(scales)
end

"""
    ImageDewarper(cam::CameraCalibration, grid::DewarpGrid, image_size)

Precomputed dewarping map for one camera: for every node of `grid` the source
pixel coordinate `world_to_pixel(cam, (X, Y, grid.z))` is evaluated once and
stored, so applying the map to a frame ([`dewarp!`](@ref)) is a pure
resampling — build the dewarper once per camera and reuse it across a whole
sequence. `image_size` is the camera's image size `(rows, cols)`.

Grid nodes whose source coordinate falls outside the camera's image (or is
non-finite) are recorded in `mask`, a grid-sized `BitMatrix` in the
package mask convention (`true` = excluded, static lab-frame geometry) —
pass it to `run_piv(...; mask)` on the dewarped images, and combine the
cameras' masks with `.|` for the stereo overlap region.
"""
struct ImageDewarper{C<:CameraCalibration}
    cam::C
    grid::DewarpGrid
    image_size::Tuple{Int,Int}
    rows::Matrix{Float64}   # source row coordinate per output pixel
    cols::Matrix{Float64}   # source column coordinate per output pixel
    mask::BitMatrix         # true = out of view (zero-filled in the output)
end

function ImageDewarper(cam::CameraCalibration, grid::DewarpGrid,
                       image_size::Tuple{Integer,Integer})
    nr, nc = image_size
    (nr >= 1 && nc >= 1) ||
        throw(ArgumentError("image_size must be positive, got $image_size"))
    ny, nx = size(grid)
    rows = Matrix{Float64}(undef, ny, nx)
    cols = Matrix{Float64}(undef, ny, nx)
    mask = falses(ny, nx)
    for j in 1:nx, i in 1:ny
        p = world_to_pixel(cam, (grid.x[j], grid.y[i], grid.z))
        if isfinite(p[1]) && isfinite(p[2]) && 1 <= p[2] <= nr && 1 <= p[1] <= nc
            rows[i, j] = p[2]
            cols[i, j] = p[1]
        else
            rows[i, j] = 0.0    # outside the interpolation domain → zero fill
            cols[i, j] = 0.0
            mask[i, j] = true
        end
    end
    return ImageDewarper(cam, grid, (Int(nr), Int(nc)), rows, cols, mask)
end

Base.show(io::IO, dw::ImageDewarper) =
    print(io, "ImageDewarper(", nameof(typeof(dw.cam)), ", ", dw.grid, ", ",
          count(dw.mask), " of ", length(dw.mask), " nodes out of view)")

"""
    dewarp!(out, dw::ImageDewarper, img) -> out

Resample the camera frame `img` onto `dw`'s world grid, writing into `out`
(size `size(dw.grid)`). Sampling uses cubic B-spline interpolation at the
precomputed source coordinates; values are computed in the image's precision
(`float(eltype(img))`). Out-of-view grid nodes (see `dw.mask`) are filled
with zero.
"""
function dewarp!(out::AbstractMatrix{<:AbstractFloat}, dw::ImageDewarper,
                 img::AbstractMatrix{<:Real})
    size(img) == dw.image_size ||
        throw(DimensionMismatch("image size $(size(img)) does not match the dewarper's " *
                                "$(dw.image_size)"))
    size(out) == size(dw.grid) ||
        throw(DimensionMismatch("output size $(size(out)) does not match the grid size " *
                                "$(size(dw.grid))"))
    T = float(eltype(img))
    itp = extrapolate(interpolate(T.(img), BSpline(Cubic(Line(OnGrid())))), zero(T))
    @inbounds for j in axes(out, 2), i in axes(out, 1)
        out[i, j] = dw.mask[i, j] ? zero(T) : itp(T(dw.rows[i, j]), T(dw.cols[i, j]))
    end
    return out
end

"""
    dewarp(dw::ImageDewarper, img) -> Matrix

Allocating form of [`dewarp!`](@ref): returns the dewarped image as a
`Matrix{float(eltype(img))}` of size `size(dw.grid)`.
"""
dewarp(dw::ImageDewarper, img::AbstractMatrix{<:Real}) =
    dewarp!(Matrix{float(eltype(img))}(undef, size(dw.grid)), dw, img)
