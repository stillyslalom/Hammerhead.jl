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

Set the world coordinates sampled by each dewarped image. `x` and `y` are
regular ranges in your chosen length unit, and `z` is the measurement plane.
Their steps give world length per dewarped pixel; select spacing fine enough
for your interrogation windows without assuming it adds source-image detail.

A dewarped image is indexed `[row, col]` with `row` running along `y` and
`col` along `x` **in the order given**: `out[r, c]` shows the world point
`(x[c], y[r], z)`. With an ascending `y`, world +Y therefore points down the
image; pass a descending `y` range for a display-oriented (+Y up) image.
Displacements measured on dewarped images convert to world units as
`du * step(x)`, `dv * step(y)` — signs included.

Use the same `DewarpGrid` for every camera in a stereo analysis, with one
[`ImageDewarper`](@ref) per camera. The stereo vector grid is derived from it.
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

Build a [`DewarpGrid`](@ref) from the cameras' projected image footprints at
world plane `z`. The default covers their common bounding-box region;
out-of-view corners remain masked by each [`ImageDewarper`](@ref).

`cameras` is an iterable of [`CameraCalibration`](@ref)s. Pass one image
size `(rows, cols)` if every camera matches, or one size per camera. The
function projects samples on each image border to plane `z` and forms
axis-aligned world bounding boxes.

- `coverage = :intersection` (default) intersects camera bounding boxes;
  `:union` covers their union. Either choice can include out-of-view nodes
  inside a bounding box; each dewarper flags those nodes in its `mask`.
  An empty intersection throws.
- `margin` (world units) shrinks the region on all sides when positive, grows
  it when negative.
- `spacing = :auto` estimates world length per source pixel for each camera
  and uses the coarsest estimate. Pass a positive number in world length
  units to choose a spacing yourself.

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

Map one camera's source pixels to a shared [`DewarpGrid`](@ref). The
constructor projects each grid node through `cam` once; reuse the result
for frames from that camera. `image_size` is the raw frame size `(rows, cols)`.

`dw.mask` marks grid nodes outside the camera view (`true` = excluded).
[`dewarp`](@ref) fills them with zero. Pass this mask to planar PIV on a
dewarped pair; stereo drivers combine both camera masks automatically.
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

Resample `img` onto `dw.grid` using cubic B-spline interpolation. `img`
must match `dw.image_size`, and the floating-point `out` must have size
`size(dw.grid)`. The interpolation uses `float(eltype(img))` internally;
values are written into `out` and masked nodes are filled with zero.
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

Allocate and return a `Matrix{float(eltype(img))}` of size `size(dw.grid)`.
See [`dewarp!`](@ref) to reuse an output buffer across frames.
"""
dewarp(dw::ImageDewarper, img::AbstractMatrix{<:Real}) =
    dewarp!(Matrix{float(eltype(img))}(undef, size(dw.grid)), dw, img)

# --- saved calibrations ------------------------------------------------------
# A calibration file stores each camera model, its raw image size, and the
# shared grid as plain JLD2 dictionaries (like recipes); the dewarpers'
# coordinate maps are rebuilt on load, which reproduces them bitwise.
const CALIBRATION_FORMAT_VERSION = 1

_camera_data(cam::PinholeCamera) = Dict{String,Any}("model" => "pinhole", "P" => Matrix(cam.P))
_camera_data(cam::SoloffCamera) =
    Dict{String,Any}("model" => "soloff", "ax" => collect(cam.ax), "ay" => collect(cam.ay),
                     "center" => collect(cam.center), "scale" => collect(cam.scale))
_camera_data(cam::TransformedCamera) =
    Dict{String,Any}("model" => "transformed", "camera" => _camera_data(cam.cam),
                     "R" => Matrix(cam.R), "t" => collect(cam.t))
_camera_data(cam::CameraCalibration) =
    throw(ArgumentError("a $(nameof(typeof(cam))) camera cannot be saved in a calibration file"))

function _camera_from_data(d)
    model = d["model"]
    model == "pinhole" && return PinholeCamera(SMatrix{3,4,Float64}(d["P"]), Val(:normalized))
    model == "soloff" && return SoloffCamera(SVector{19,Float64}(d["ax"]), SVector{19,Float64}(d["ay"]),
                                             SVector{3,Float64}(d["center"]), SVector{3,Float64}(d["scale"]))
    model == "transformed" && return TransformedCamera(_camera_from_data(d["camera"]), d["R"], d["t"])
    throw(ArgumentError("unknown camera model \"$model\" in calibration"))
end

_range_data(r::LinRange) = Dict{String,Any}("first" => first(r), "last" => last(r), "length" => length(r))
_range_from_data(d) = LinRange{Float64,Int}(d["first"], d["last"], d["length"])

function _calibration_data(dewarpers)
    isempty(dewarpers) && throw(ArgumentError("a calibration needs at least one dewarper"))
    grid = first(dewarpers).grid
    all(dw -> dw.grid == grid, dewarpers) ||
        throw(ArgumentError("the dewarpers must share one DewarpGrid"))
    Dict{String,Any}(
        "grid" => Dict{String,Any}("x" => _range_data(grid.x), "y" => _range_data(grid.y),
                                   "z" => grid.z),
        "cameras" => [merge(_camera_data(dw.cam),
                            Dict{String,Any}("image_size" => collect(dw.image_size)))
                      for dw in dewarpers])
end

function _write_calibration(file, dewarpers)
    file["calibration_format_version"] = CALIBRATION_FORMAT_VERSION
    file["calibration"] = _calibration_data(dewarpers)
    nothing
end

"""
    save_calibration(path, dw1, dw2, ...) -> path

Save a camera rig to a JLD2 file: each [`ImageDewarper`](@ref)'s camera model
and raw image size, and the [`DewarpGrid`](@ref) they share. A
self-calibration applied with [`self_calibrate`](@ref) is part of the
cameras, so it is saved too. [`load_calibration`](@ref) rebuilds the
dewarpers.

Pinhole, Soloff, and [`TransformedCamera`](@ref) models can be saved.
"""
function save_calibration(path::AbstractString, dewarpers::ImageDewarper...)
    data = _calibration_data(dewarpers)   # validate before creating the file
    jldopen(path, "w") do file
        file["calibration_format_version"] = CALIBRATION_FORMAT_VERSION
        file["calibration"] = data
    end
    path
end

"""
    load_calibration(path) -> Tuple of ImageDewarper

Load the dewarpers saved with [`save_calibration`](@ref), in the order they
were saved, or the calibration a stereo [`apply_recipe`](@ref) stored with
its results.
"""
function load_calibration(path::AbstractString)
    jldopen(path, "r") do file
        haskey(file, "calibration") ||
            throw(ArgumentError("$path contains no Hammerhead calibration"))
        version = file["calibration_format_version"]
        version == CALIBRATION_FORMAT_VERSION ||
            throw(ArgumentError("$path has unsupported calibration_format_version $version"))
        d = file["calibration"]
        g = d["grid"]
        grid = DewarpGrid(; x = _range_from_data(g["x"]), y = _range_from_data(g["y"]), z = g["z"])
        Tuple(ImageDewarper(_camera_from_data(c), grid, Tuple(c["image_size"]))
              for c in d["cameras"])
    end
end
