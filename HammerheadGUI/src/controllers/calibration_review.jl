# Calibration-review controller: grid detection over a set of plate images,
# camera fitting, and per-dot reprojection errors — plus the self-calibration
# report summary. Framework-free.

"""
    CalibrationReview(images, zs; model = :soloff, detect_kwargs...)

Detect dot grids in calibration images and fit a camera for reprojection
review. `images` contains matrices and/or image paths; `zs` gives one plate
position per image in world units. Keywords such as `spacing`, `two_level`,
and `origin_offset` are passed to `detect_calibration_grid`.

Observables: `plane` (selected plane index), `model` (`:soloff` /
`:pinhole`; changing it refits), `camera` (the fitted `CameraCalibration`,
or `nothing` when the fit lacks enough distinct world points or otherwise fails),
and `fit_message` (the failure reason, empty on success).
"""
struct CalibrationReview
    images::Vector{Matrix{Float64}}
    zs::Vector{Float64}
    grids::Vector{CalibrationGrid}
    model::Observable{Symbol}
    plane::Observable{Int}
    camera::Observable{Any}
    fit_message::Observable{String}
end

function CalibrationReview(images::AbstractVector, zs::AbstractVector{<:Real};
                           model::Symbol = :soloff, detect_kwargs...)
    isempty(images) && throw(ArgumentError("no calibration images"))
    length(images) == length(zs) ||
        throw(ArgumentError("need one z per image, got $(length(images)) images and $(length(zs)) zs"))
    imgs = [img isa AbstractString ? load_image(img) : Matrix{Float64}(img)
            for img in images]
    grids = CalibrationGrid[]
    for (i, img) in enumerate(imgs)
        try
            push!(grids, detect_calibration_grid(img; detect_kwargs...))
        catch err
            throw(ArgumentError("grid detection failed on plane $i (z = $(zs[i])): $(_errmsg(err))"))
        end
    end
    cr = CalibrationReview(imgs, collect(Float64, zs), grids,
                           Observable(model), Observable(1),
                           Observable{Any}(nothing), Observable(""))
    on(_ -> refit!(cr), cr.model)
    refit!(cr)
    return cr
end

function Base.show(io::IO, cr::CalibrationReview)
    print(io, "CalibrationReview($(length(cr.grids)) planes, :$(cr.model[])",
          cr.camera[] === nothing ? ", no fit)" : ")")
end

"""
    nplanes(cr::CalibrationReview) -> Int

Number of calibration planes.
"""
nplanes(cr::CalibrationReview) = length(cr.grids)

"""
    set_plane!(cr::CalibrationReview, i::Integer)

Select plane `i`, clamped to `1:nplanes(cr)`.
"""
set_plane!(cr::CalibrationReview, i::Integer) = cr.plane[] = clamp(i, 1, nplanes(cr))

"""
    refit!(cr::CalibrationReview)

Refit the camera for the current `model`; on failure `camera` becomes
`nothing` and `fit_message` carries the reason. Runs automatically when
`model` changes.
"""
function refit!(cr::CalibrationReview)
    try
        cr.camera[] = calibrate_camera(cr.grids, cr.zs; model = cr.model[])
        cr.fit_message[] = ""
    catch err
        cr.camera[] = nothing
        cr.fit_message[] = _errmsg(err)
    end
    return cr
end

"""
    plane_errors(cr::CalibrationReview, i = cr.plane[])

Return `(; pixels, errors)` for plane `i`: detected dot locations and their
Euclidean reprojection errors in pixels. Return `nothing` if no camera is fitted.
"""
function plane_errors(cr::CalibrationReview, i::Integer = cr.plane[])
    cam = cr.camera[]
    cam === nothing && return nothing
    px, wd = calibration_points(cr.grids[i], cr.zs[i])
    return (; pixels = px, errors = reprojection_errors(cam, px, wd))
end

"""
    plane_residuals(cr::CalibrationReview, i = cr.plane[])

Return `(; pixels, residuals)` for plane `i`: detected dot locations and
their reprojection residual vectors in pixels (`world_to_pixel` of the dot's
world point minus its detected location), as `(dx, dy)` tuples. Return
`nothing` if no camera is fitted.
"""
function plane_residuals(cr::CalibrationReview, i::Integer = cr.plane[])
    cam = cr.camera[]
    cam === nothing && return nothing
    px, wd = calibration_points(cr.grids[i], cr.zs[i])
    residuals = map(px, wd) do p, w
        q = world_to_pixel(cam, w)
        (Float64(q[1] - p[1]), Float64(q[2] - p[2]))
    end
    return (; pixels = px, residuals)
end

"""
    plane_summary(cr::CalibrationReview, i = cr.plane[]) -> String

One-line summary of plane `i`: z, dot count, markers, reprojection errors.
"""
function plane_summary(cr::CalibrationReview, i::Integer = cr.plane[])
    g = cr.grids[i]
    marks = String[]
    g.square === nothing || push!(marks, "square")
    g.triangle === nothing || push!(marks, "triangle")
    s = "z = $(_fmt(cr.zs[i])): $(length(g.pixels)) dots" *
        (isempty(marks) ? "" : " (" * join(marks, " + ") * ")")
    pe = plane_errors(cr, i)
    pe === nothing && return s
    rms = sqrt(sum(abs2, pe.errors) / length(pe.errors))
    return s * "\nreprojection rms $(_fmt(rms)) px, max $(_fmt(maximum(pe.errors))) px"
end

"""
    fit_summary(cr::CalibrationReview) -> String

Overall fit summary (`calibration_quality` over all planes), or the fit
failure message.
"""
function fit_summary(cr::CalibrationReview)
    cr.camera[] === nothing && return "no fit: $(cr.fit_message[])"
    q = calibration_quality(cr.camera[], cr.grids, cr.zs)
    return "$(cr.model[]) fit over $(nplanes(cr)) planes\n" *
           "rms $(_fmt(q.rms)) px, max $(_fmt(q.max)) px ($(q.n) dots)"
end

"""
    selfcal_summary(report::SelfCalibrationReport) -> String

Summarize each pass's disparity magnitudes and RMS, fitted planes,
convergence, and cumulative rigid correction. When the residual RMS remains
above tolerance, inspect the spatial disparity maps for patterns.
"""
function selfcal_summary(report::SelfCalibrationReport)
    lines = String[]
    for (k, p) in enumerate(report.passes)
        s = "pass $k: disparity median magnitude $(_fmt(p.disparity_median)) px, " *
            "RMS $(_fmt(p.disparity_rms)) px ($(p.n_vectors) vectors)"
        if p.plane === nothing
            s *= ", no correction"
        else
            s *= "\n  plane a = $(_fmt(p.plane.a)), b = $(_fmt(p.plane.b)), " *
                 "c = $(_fmt(p.plane.c)); triangulation rms $(_fmt(p.triangulation_rms)) px"
        end
        push!(lines, s)
    end
    if report.converged
        push!(lines, "converged (RMS tolerance $(_fmt(report.tol)) px)")
    else
        advice = isempty(report.disparity_maps) ?
                 "rerun with keep_disparity_maps = true to inspect spatial residuals" :
                 "inspect the final disparity map for spatial residuals"
        push!(lines, "not converged (RMS above $(_fmt(report.tol)) px); $advice")
    end
    angle = acosd(clamp((LinearAlgebra.tr(report.R) - 1) / 2, -1.0, 1.0))
    push!(lines, "correction: rotation $(_fmt(angle))°, shift $(_fmt(LinearAlgebra.norm(report.t))) (world units)")
    return join(lines, "\n")
end

"""
    build_dewarpers(cr1::CalibrationReview, cr2::CalibrationReview;
                    z = 0.0, spacing = :auto, coverage = :intersection,
                    margin = 0.0) -> (dw1, dw2)

Build a pair of [`ImageDewarper`](@ref)s on one world-coordinate grid from
two fitted reviews, for example to pass to `stereo_window(; dewarpers)` or
`run_piv_stereo`. `z` selects the world plane; `spacing` and `margin` use
the calibrations' world length unit. `coverage = :intersection` or `:union`
combines the cameras' projected boundary boxes, not their exact visible
regions; out-of-view samples remain masked per camera. The grid uses each
review's first plate image size. Throws if either camera has no fit.
"""
function build_dewarpers(cr1::CalibrationReview, cr2::CalibrationReview;
                         z::Real = 0.0, spacing = :auto,
                         coverage::Symbol = :intersection, margin::Real = 0.0)
    cam1, cam2 = cr1.camera[], cr2.camera[]
    cam1 === nothing && throw(ArgumentError("camera 1 has no fitted calibration: $(cr1.fit_message[])"))
    cam2 === nothing && throw(ArgumentError("camera 2 has no fitted calibration: $(cr2.fit_message[])"))
    return _dewarper_pair((cam1, cam2), (size(cr1.images[1]), size(cr2.images[1])), z;
                          spacing, coverage, margin)
end

# Two cameras' dewarpers on their common grid (shared with the stereo
# workflow's Calibration step, which captures cameras and sizes first so the
# build can run on a worker).
function _dewarper_pair(cams, sizes, z::Real; spacing, coverage::Symbol, margin::Real)
    grid = common_dewarp_grid(collect(cams), collect(sizes), z; spacing, coverage, margin)
    return (ImageDewarper(cams[1], grid, sizes[1]), ImageDewarper(cams[2], grid, sizes[2]))
end
