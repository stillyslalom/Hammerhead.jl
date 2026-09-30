# Physical units: results carry an optional PhysicalScale (types.jl) as
# metadata while their arrays always stay in measured units — pixels for
# planar PIV/PTV/tracking, world units for stereo. `physical` is the single
# conversion point; everything upstream (validation thresholds, peak locking,
# correlation diagnostics) is pixel-native and must run before it.

"""
    with_scale(result, scale::Union{Nothing,PhysicalScale})

Return the same result type with `scale` replaced. The stored arrays are
shared, not copied or converted. Works on [`PIVResult`](@ref),
[`StereoPIVResult`](@ref), [`PTVResult`](@ref), and [`TrackingResult`](@ref).
Pass `nothing` to remove a scale from a raw result, for example before
plotting it on a source image in pixel coordinates. Removing the scale
does not undo a previous [`physical`](@ref) conversion.
"""
with_scale(r::PIVResult{T}, scale::Union{Nothing,PhysicalScale}) where {T} =
    PIVResult{T}(r.x, r.y, r.u, r.v, r.peak_ratio, r.correlation_moment,
                 r.uncertainty_u, r.uncertainty_v, r.outliers, r.mask,
                 r.parameters, r.correlation_planes, scale)

with_scale(r::StereoPIVResult{T}, scale::Union{Nothing,PhysicalScale}) where {T} =
    StereoPIVResult{T}(r.x, r.y, r.z, r.u, r.v, r.w,
                       r.uncertainty_u, r.uncertainty_v, r.uncertainty_w,
                       r.outliers, r.mask, r.cam1, r.cam2, r.parameters, scale)

with_scale(r::PTVResult{T}, scale::Union{Nothing,PhysicalScale}) where {T} =
    PTVResult{T}(r.x, r.y, r.u, r.v, r.match_residual, r.outliers,
                 r.index_a, r.index_b, r.particles_a, r.particles_b,
                 r.parameters, scale)

with_scale(r::TrackingResult{T}, scale::Union{Nothing,PhysicalScale}) where {T} =
    TrackingResult{T}(r.trajectories, r.n_frames, r.parameters, scale)

"""
    physical(result) -> same-type result in physical units
    physical(result, scale::PhysicalScale)

Convert a result using its attached [`PhysicalScale`](@ref). The two-argument
form replaces any attached scale before conversion. Positions (`x`, `y`,
`z`, trajectory points) and PTV `match_residual` values multiply by
`pixel_size`. Displacement components (`u`, `v`, `w`) and their uncertainty
estimates multiply by `pixel_size / dt`, becoming velocities. Stereo arrays
already use world coordinates, so their scale normally has `pixel_size = 1`.
Without a scale, or with an identity scale, return the original result.

For PIV, stereo PIV, and PTV, the converted result carries an identity
scale with the original unit labels. A second call to `physical` therefore
does not convert the arrays again. Pixel-native diagnostics stay unchanged:
`peak_ratio`, `correlation_moment`, `correlation_planes`, embedded PTV
`particles_a`/`particles_b`, and stereo `cam1`/`cam2` results. Run validators,
[`peak_locking`](@ref), and other pixel-calibrated tools before conversion.

A [`TrackingResult`](@ref) stores positions rather than displacement
components. Its converted scale retains `dt` so
[`trajectory_velocities`](@ref) can derive velocities from either raw or
converted trajectories using the frame interval.
"""
physical(r::Union{PIVResult,StereoPIVResult,PTVResult,TrackingResult},
         scale::PhysicalScale) = physical(with_scale(r, scale))

function physical(r::PIVResult{T}) where {T}
    s = r.scale
    (s === nothing || is_identity(s)) && return r
    fp = T(s.pixel_size)
    fv = T(s.pixel_size / s.dt)
    return PIVResult{T}(r.x .* fp, r.y .* fp, r.u .* fv, r.v .* fv,
                        r.peak_ratio, r.correlation_moment,
                        r.uncertainty_u .* fv, r.uncertainty_v .* fv,
                        r.outliers, r.mask, r.parameters, r.correlation_planes,
                        PhysicalScale(1.0, 1.0, s.length_unit, s.time_unit))
end

function physical(r::StereoPIVResult{T}) where {T}
    s = r.scale
    (s === nothing || is_identity(s)) && return r
    fp = T(s.pixel_size)
    fv = T(s.pixel_size / s.dt)
    return StereoPIVResult{T}(r.x .* fp, r.y .* fp, r.z * s.pixel_size,
                              r.u .* fv, r.v .* fv, r.w .* fv,
                              r.uncertainty_u .* fv, r.uncertainty_v .* fv,
                              r.uncertainty_w .* fv, r.outliers, r.mask,
                              r.cam1, r.cam2, r.parameters,
                              PhysicalScale(1.0, 1.0, s.length_unit, s.time_unit))
end

function physical(r::PTVResult{T}) where {T}
    s = r.scale
    (s === nothing || is_identity(s)) && return r
    fp = T(s.pixel_size)
    fv = T(s.pixel_size / s.dt)
    return PTVResult{T}(r.x .* fp, r.y .* fp, r.u .* fv, r.v .* fv,
                        r.match_residual .* fp, r.outliers, r.index_a, r.index_b,
                        r.particles_a, r.particles_b, r.parameters,
                        PhysicalScale(1.0, 1.0, s.length_unit, s.time_unit))
end

function physical(r::TrackingResult{T}) where {T}
    s = r.scale
    # pixel_size == 1 leaves nothing to apply: positions are unchanged and dt
    # must survive anyway (velocities are derived by differencing).
    (s === nothing || s.pixel_size == 1.0) && return r
    fp = T(s.pixel_size)
    trajectories = [Trajectory{T}(t.start_frame, t.x .* fp, t.y .* fp, t.frames)
                    for t in r.trajectories]
    return TrackingResult{T}(trajectories, r.n_frames, r.parameters,
                             PhysicalScale(1.0, s.dt, s.length_unit, s.time_unit))
end

# Axis labels for plotting a result's positions: "x (px)"/"y (px)" without a
# scale, the scale's length unit with one. The Makie extension routes results
# through `physical` first, whose identity-with-labels scale keeps these
# consistent with the plotted arrays. Core-side (Makie-free) so it is
# testable without a plotting backend, like `arrow_lengthscale`.
plot_axis_labels(::Nothing) = ("x (px)", "y (px)")
plot_axis_labels(s::PhysicalScale) = ("x ($(s.length_unit))", "y ($(s.length_unit))")
