# Multi-frame trajectory linking (Phase 8): chain per-transition PTV matches
# into Lagrangian tracks. Each frame is detected once; transitions are linked
# frame-to-frame with the same greedy matcher as run_ptv, predicting each track
# head by constant velocity (heads with ≥ 2 points) or by a field predictor
# (fresh 1-point heads). No gap bridging in v1 (deferred).
#
# `load_frame`/`frame_label` live in io.jl, which is included after this file;
# they are only called at runtime here, so the forward reference is fine.

"""
    Trajectory{T<:AbstractFloat}

A single particle track: `start_frame` (1-based) is the frame of the first
point, and `x`/`y` hold one subpixel position per consecutive frame (no gaps in
v1). `length(t)` is the number of points. See [`track_particles`](@ref).
"""
struct Trajectory{T<:AbstractFloat}
    start_frame::Int
    x::Vector{T}
    y::Vector{T}
end

Base.length(t::Trajectory) = length(t.x)

Base.show(io::IO, t::Trajectory{T}) where {T} =
    print(io, "Trajectory{$T}($(length(t)) points from frame $(t.start_frame))")

"""
    TrackingResult{T<:AbstractFloat}

Result of [`track_particles`](@ref): the linked `trajectories`, the number of
frames `n_frames`, the `parameters` used, and the optional [`PhysicalScale`](@ref)
`scale` (`nothing` when none was attached — positions stay in pixels until
[`physical`](@ref) converts them).
"""
struct TrackingResult{T<:AbstractFloat}
    trajectories::Vector{Trajectory{T}}
    n_frames::Int
    parameters::PTVParameters
    scale::Union{Nothing,PhysicalScale}
end

# Backward-compatible constructors: no physical scale stored.
TrackingResult(trajectories::Vector{Trajectory{T}}, n_frames::Int,
               parameters::PTVParameters) where {T} =
    TrackingResult{T}(trajectories, n_frames, parameters, nothing)

TrackingResult{T}(trajectories, n_frames, parameters) where {T} =
    TrackingResult{T}(trajectories, n_frames, parameters, nothing)

Base.show(io::IO, r::TrackingResult{T}) where {T} =
    print(io, "TrackingResult{$T}($(length(r.trajectories)) tracks over $(r.n_frames) frames)")

# Mutable working track, extended in place during linking.
mutable struct _Track{T<:AbstractFloat}
    start_frame::Int
    x::Vector{T}
    y::Vector{T}
end

make_field_interp(::Nothing) = nothing
function make_field_interp(field)
    itp_u = extrapolate(interpolate((field.y, field.x), field.u, Gridded(Linear())), Flat())
    itp_v = extrapolate(interpolate((field.y, field.x), field.v, Gridded(Linear())), Flat())
    return (itp_u, itp_v)
end

# Field predictor for a transition's fresh 1-point heads, built from the
# previous transition's accepted matches (binned + smoothed). `nothing` when
# there is nothing to bin, so those heads fall back to zero displacement.
function grid_predictor(mx, my, mu, mv, image_size::Tuple{Int,Int}, ::Type{T}) where {T}
    isempty(mx) && return nothing
    gridres = bin_to_grid(T, mx, my, mu, mv, image_size,
                          PIVParameters(; window_size = (32, 32), overlap = (16, 16)), 3)
    all(gridres.mask) && return nothing
    return build_predictor(gridres, true)
end

"""
    track_particles(frames, params = PTVParameters();
                    predictor = :piv, piv_passes = multipass_parameters([64, 32]),
                    min_track_length = 3, mask = nothing, scale = nothing,
                    image_type = Float64, progress = true) -> TrackingResult

Link particles across a sequence of `frames` (≥ 2 file paths or real-valued
matrices) into trajectories. Each frame is detected once; for every transition
`k → k+1`, track heads are predicted — constant-velocity extrapolation for
heads with ≥ 2 points, and a field predictor for fresh 1-point heads (the
`predictor` option on the first transition, exactly as in [`run_ptv`](@ref);
the binned previous transition afterwards) — then matched with the greedy PTV
matcher. Scattered-UOD-flagged links are rejected (never linked, so they cannot
poison the constant-velocity predictor). Unmatched heads terminate their track
(no gap bridging in v1); unmatched frame-`(k+1)` particles seed new tracks.

Only tracks with `length ≥ min_track_length` (which must be ≥ 2) are returned,
sorted by `start_frame` then first position. `scale` attaches a
[`PhysicalScale`](@ref) to the result (positions stay in pixels; see
[`physical`](@ref) and [`trajectory_velocities`](@ref)). `image_type` is the
element type frames load as (`Float32` runs in single precision); `progress`
is a `Bool` meter or an `(i, n)` callback ticked per transition, as in
[`run_piv_sequence`](@ref).
"""
function track_particles(frames::AbstractVector, params::PTVParameters = PTVParameters();
                         predictor = :piv,
                         piv_passes = multipass_parameters([64, 32]),
                         min_track_length::Int = 3,
                         mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                         scale::Union{Nothing,PhysicalScale} = nothing,
                         image_type::Type{<:AbstractFloat} = Float64,
                         progress::Union{Bool,Function} = true)
    n_frames = length(frames)
    n_frames >= 2 || throw(ArgumentError("track_particles needs at least 2 frames, got $n_frames"))
    min_track_length >= 2 ||
        throw(ArgumentError("min_track_length must be at least 2, got $min_track_length"))
    T = float(image_type)

    # Load every frame once (needed for detection, and frames 1–2 for the
    # initial :piv predictor) and detect particles in each.
    imgs = [load_frame(f, image_type) for f in frames]
    all(size(im) == size(imgs[1]) for im in imgs) ||
        throw(DimensionMismatch("all frames must have the same size"))
    image_size = size(imgs[1])
    particles = [detect_particles(im, params; mask) for im in imgs]

    all_tracks = _Track{T}[]
    active = _Track{T}[]
    p1 = particles[1]
    for i in 1:length(p1)
        tr = _Track{T}(1, T[p1.x[i]], T[p1.y[i]])
        push!(all_tracks, tr)
        push!(active, tr)
    end

    prev_matches = nothing   # (mx, my, mu, mv) of the last transition's accepted links
    n_trans = n_frames - 1
    meter = Progress(n_trans; desc = "Tracking: ", enabled = progress === true)
    for k in 1:n_trans
        pb = particles[k + 1]
        field = k == 1 ? resolve_predictor(predictor, imgs[1], imgs[2], piv_passes, mask, T) :
                (prev_matches === nothing ? nothing :
                 grid_predictor(prev_matches..., image_size, T))
        interp = make_field_interp(field)

        M = length(active)
        pred_x = Vector{T}(undef, M)
        pred_y = Vector{T}(undef, M)
        for (i, tr) in enumerate(active)
            hx = tr.x[end]; hy = tr.y[end]
            if length(tr.x) >= 2
                pred_x[i] = hx + (tr.x[end] - tr.x[end - 1])
                pred_y[i] = hy + (tr.y[end] - tr.y[end - 1])
            elseif interp === nothing
                pred_x[i] = hx
                pred_y[i] = hy
            else
                pred_x[i] = hx + T(interp[1](hy, hx))
                pred_y[i] = hy + T(interp[2](hy, hx))
            end
        end

        cl_b = build_cell_list(pb.x, pb.y, params.search_radius)
        index_a, index_b, _ = greedy_match(pred_x, pred_y, cl_b, params.search_radius)

        # Scattered UOD on this transition's matches, at the frame-k head
        # positions; flagged links are rejected below.
        nm = length(index_a)
        hx = T[active[index_a[m]].x[end] for m in 1:nm]
        hy = T[active[index_a[m]].y[end] for m in 1:nm]
        mu = T[pb.x[index_b[m]] - hx[m] for m in 1:nm]
        mv = T[pb.y[index_b[m]] - hy[m] for m in 1:nm]
        flags = params.uod_enable ? scattered_uod(hx, hy, mu, mv, params) : falses(nm)

        extended = falses(M)
        accepted_b = falses(length(pb))
        amx = T[]; amy = T[]; amu = T[]; amv = T[]
        for m in 1:nm
            flags[m] && continue
            ia = index_a[m]; ib = index_b[m]
            push!(active[ia].x, pb.x[ib])
            push!(active[ia].y, pb.y[ib])
            extended[ia] = true
            accepted_b[ib] = true
            push!(amx, hx[m]); push!(amy, hy[m]); push!(amu, mu[m]); push!(amv, mv[m])
        end

        newactive = _Track{T}[]
        for (i, tr) in enumerate(active)
            extended[i] && push!(newactive, tr)   # unextended heads terminate
        end
        for j in 1:length(pb)
            accepted_b[j] && continue
            tr = _Track{T}(k + 1, T[pb.x[j]], T[pb.y[j]])
            push!(all_tracks, tr)
            push!(newactive, tr)
        end
        active = newactive
        prev_matches = (amx, amy, amu, amv)

        progress isa Function ? progress(k, n_trans) : next!(meter)
    end

    kept = [tr for tr in all_tracks if length(tr.x) >= min_track_length]
    sort!(kept; by = tr -> (tr.start_frame, tr.x[1], tr.y[1]))
    trajectories = [Trajectory{T}(tr.start_frame, tr.x, tr.y) for tr in kept]
    return TrackingResult{T}(trajectories, n_frames, params, scale)
end

"""
    trajectory_velocities(t::Trajectory, scale = nothing) -> (u, v)

Per-point displacement estimates along a trajectory (px per frame interval):
central differences in the interior and one-sided differences at the ends.
`u` is the column (x) component, `v` the row (y). Requires `length(t) ≥ 2`.

With a [`PhysicalScale`](@ref) the differences are multiplied by
`pixel_size / dt` into physical velocities. Pass the owning result's `scale`
field — this yields the same velocities whether `t` comes from a raw or a
[`physical`](@ref)-converted [`TrackingResult`](@ref), because the converted
result's scale keeps `dt` (its positions are already lengths).
"""
function trajectory_velocities(t::Trajectory{T},
                               scale::Union{Nothing,PhysicalScale} = nothing) where {T}
    n = length(t)
    n >= 2 || throw(ArgumentError("trajectory_velocities needs at least 2 points, got $n"))
    u = Vector{T}(undef, n)
    v = Vector{T}(undef, n)
    u[1] = t.x[2] - t.x[1]
    v[1] = t.y[2] - t.y[1]
    u[n] = t.x[n] - t.x[n - 1]
    v[n] = t.y[n] - t.y[n - 1]
    for i in 2:(n - 1)
        u[i] = (t.x[i + 1] - t.x[i - 1]) / 2
        v[i] = (t.y[i + 1] - t.y[i - 1]) / 2
    end
    if scale !== nothing && !(scale.pixel_size == 1.0 && scale.dt == 1.0)
        f = T(scale.pixel_size / scale.dt)
        u .*= f
        v .*= f
    end
    return u, v
end
