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
    ImageDewarper(cam::CameraCalibration, grid::DewarpGrid, image_size)

Precomputed dewarping map for one camera: for every node of `grid` the source
pixel coordinate `world_to_pixel(cam, (X, Y, grid.z))` is evaluated once and
stored, so applying the map to a frame ([`dewarp!`](@ref)) is a pure
resampling — build the dewarper once per camera and reuse it across a whole
sequence. `image_size` is the camera's image size `(rows, cols)`.

Grid nodes whose source coordinate falls outside the camera's image (or is
non-finite) are recorded in `mask`, an image-sized `BitMatrix` in the
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
