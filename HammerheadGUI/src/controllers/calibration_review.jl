# Calibration-review controller: grid detection over a set of plate images,
# camera fitting, and per-dot reprojection errors — plus the self-calibration
# report summary. Framework-free.

"""
    CalibrationReview(images, zs; model = :soloff, detect_kwargs...)

Controller for the calibration diagnostics: detects the dot grid on every
plate image (`detect_calibration_grid`; `detect_kwargs` — `spacing`,
`two_level`, `origin_offset`, … — are forwarded), then fits a camera and
exposes per-plane reprojection errors. `images` are matrices and/or image
paths; `zs` are the plate positions (world units, one per image).

Observables: `plane` (selected plane index), `model` (`:soloff` /
`:pinhole`; changing it refits), `camera` (the fitted `CameraCalibration`,
or `nothing` when the fit fails — e.g. `:soloff` with fewer than 3 planes),
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

Detected dot pixels and their reprojection errors on plane `i`, as
`(; pixels, errors)` — or `nothing` when no camera is fitted.
"""
function plane_errors(cr::CalibrationReview, i::Integer = cr.plane[])
    cam = cr.camera[]
    cam === nothing && return nothing
    px, wd = calibration_points(cr.grids[i], cr.zs[i])
    return (; pixels = px, errors = reprojection_errors(cam, px, wd))
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

Multi-line summary of a self-calibration run: per-pass disparity and
triangulation statistics, the fitted planes, convergence, and the size of
the cumulative rigid correction.
"""
function selfcal_summary(report::SelfCalibrationReport)
    lines = String[]
    for (k, p) in enumerate(report.passes)
        s = "pass $k: disparity median $(_fmt(p.disparity_median)) px, " *
            "rms $(_fmt(p.disparity_rms)) px ($(p.n_vectors) vectors)"
        if p.plane === nothing
            s *= " — no correction"
        else
            s *= "\n  plane a = $(_fmt(p.plane.a)), b = $(_fmt(p.plane.b)), " *
                 "c = $(_fmt(p.plane.c)); triangulation rms $(_fmt(p.triangulation_rms)) px"
        end
        push!(lines, s)
    end
    push!(lines, report.converged ? "converged (tol $(_fmt(report.tol)) px)" :
                 "not converged at tol $(_fmt(report.tol)) px — judge by the signed median disparity")
    angle = acosd(clamp((LinearAlgebra.tr(report.R) - 1) / 2, -1.0, 1.0))
    push!(lines, "correction: rotation $(_fmt(angle))°, shift $(_fmt(LinearAlgebra.norm(report.t))) (world units)")
    return join(lines, "\n")
end
