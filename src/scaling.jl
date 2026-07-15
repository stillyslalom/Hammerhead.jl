# Physical units: results carry an optional PhysicalScale (types.jl) as
# metadata while their arrays always stay in measured units — pixels for
# planar PIV/PTV/tracking, world units for stereo. `physical` is the single
# conversion point; everything upstream (validation thresholds, peak locking,
# correlation diagnostics) is pixel-native and must run before it.

"""
    with_scale(result, scale::Union{Nothing,PhysicalScale})

Return a result identical to `result` (all arrays shared, not copied) with its
`scale` field replaced. Works on [`PIVResult`](@ref), [`StereoPIVResult`](@ref),
[`PTVResult`](@ref), and [`TrackingResult`](@ref); `nothing` strips an
attached scale (e.g. to overlay vectors on the source image in pixel
coordinates). Attaching is pure metadata — the stored arrays keep their
measured units until [`physical`](@ref) converts them.
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

Convert `result` to physical units using its attached [`PhysicalScale`](@ref)
(the two-argument form attaches `scale` first, replacing any existing one).
Positions (`x`, `y`, `z`, trajectory points) are multiplied by `pixel_size`
and displacements together with their uncertainties (`u`, `v`, `w`,
`uncertainty_*`, and a [`PTVResult`](@ref)'s `match_residual` by `pixel_size`
alone) by `pixel_size / dt`, turning displacements per frame interval into
velocities. Returns `result` itself when there is nothing to convert
(`scale === nothing` or an identity scale).

The returned result carries an *identity* scale with the same unit labels, so
`physical` is idempotent and plot labels always match the arrays. Pixel-native
diagnostics are never converted: `peak_ratio`, `correlation_moment`,
`correlation_planes`, the embedded `particles_a`/`particles_b`, and a
[`StereoPIVResult`](@ref)'s per-camera `cam1`/`cam2` results are shared
untouched. Convert last — validators, [`peak_locking`](@ref), and the other
pixel-calibrated tools must run on the raw result.

One deviation from the identity rule: a [`TrackingResult`](@ref) stores only
positions (velocities are *derived* by differencing), so its converted scale
keeps `dt` — pass it to [`trajectory_velocities`](@ref) to get physical
velocities from either the raw or the converted result.
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
