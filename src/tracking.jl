# Multi-frame trajectory linking (Phase 8): chain per-transition PTV matches
# into Lagrangian tracks. Each frame is detected once; transitions are linked
# with the same greedy matcher as run_ptv, predicting each track head by
# constant velocity (heads with ≥ 2 points) or by a field predictor (fresh
# 1-point heads). Unmatched heads can bridge up to max_gap missed frames.
#
# `load_frame`/`frame_label` live in io.jl, which is included after this file;
# they are only called at runtime here, so the forward reference is fine.

"""
    Trajectory{T<:AbstractFloat}

One linked particle path from [`track_particles`](@ref). `x` and `y` hold
observed subpixel positions in image columns and rows; `frames` gives each
observation's frame number, including gaps after a reacquisition.
`start_frame` is the first frame number and `length(t)` counts observations.
"""
struct Trajectory{T<:AbstractFloat}
    start_frame::Int
    x::Vector{T}
    y::Vector{T}
    frames::Vector{Int}
end

Trajectory{T}(start_frame::Int, x::Vector{T}, y::Vector{T}) where {T} =
    Trajectory{T}(start_frame, x, y, collect(start_frame:start_frame+length(x)-1))
Trajectory(start_frame::Int, x::Vector{T}, y::Vector{T}) where {T<:AbstractFloat} =
    Trajectory{T}(start_frame, x, y)

Base.length(t::Trajectory) = length(t.x)

Base.show(io::IO, t::Trajectory{T}) where {T} =
    print(io, "Trajectory{$T}($(length(t)) points from frame $(t.start_frame))")

"""
    TrackingResult{T<:AbstractFloat}

Collection of linked `trajectories` from [`track_particles`](@ref), with
input frame count `n_frames` and the `parameters` used. `scale` is optional
[`PhysicalScale`](@ref) metadata; positions remain in pixels until
[`physical`](@ref) converts them.
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
    frames::Vector{Int}
    intensity::T
    diameter::T
    missed::Int
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
                    min_track_length = 3, max_gap = 0, mask = nothing, scale = nothing,
                    image_type = Float64, progress = true) -> TrackingResult

Link detections across at least two frames, supplied as file paths or
real-valued matrices. Each frame is detected once. Tracks with two or more
points use constant-velocity prediction; newer tracks use a field predictor.
The first transition uses `predictor` and `piv_passes` as in
[`run_ptv`](@ref); later transitions use binned matches from the previous
pair. Scattered-UOD-flagged links are rejected. `max_gap` permits that many
missed frames before a track ends and records any reacquisition gap in
[`Trajectory`](@ref)`s `frames`; its default of zero ends a track at the first
miss. Unmatched detections can start new tracks.

Returns tracks with at least `min_track_length` observations (minimum 2),
sorted by starting frame and first position. `scale` attaches physical-unit
metadata without converting stored pixel positions. Use [`physical`](@ref)
and [`trajectory_velocities`](@ref) for velocities. `image_type` selects the
precision used when loading file paths; in-memory matrix types are promoted.
`progress` is a Boolean meter setting or an `(i, n)` callback after each
transition. Frames are loaded and detected one at a time. For lazy
[`FrameRef`](@ref) sources whose later element types cannot be inspected
without loading, the first frame sets result precision and later detections
are converted to it.
"""
function track_particles(frames::AbstractVector, params::PTVParameters = PTVParameters();
                         predictor = :piv,
                         piv_passes = multipass_parameters([64, 32]),
                         min_track_length::Int = 3,
                         max_gap::Int = 0,
                         mask::Union{Nothing,AbstractMatrix{Bool}} = nothing,
                         scale::Union{Nothing,PhysicalScale} = nothing,
                         image_type::Type{<:AbstractFloat} = Float64,
                         progress::Union{Bool,Function} = true)
    n_frames = length(frames)
    n_frames >= 2 || throw(ArgumentError("track_particles needs at least 2 frames, got $n_frames"))
    min_track_length >= 2 ||
        throw(ArgumentError("min_track_length must be at least 2, got $min_track_length"))
    max_gap >= 0 || throw(ArgumentError("max_gap must be nonnegative, got $max_gap"))
    img_first = load_frame(frames[1], image_type)
    image_size = size(img_first)
    # In-memory matrices retain their element type. Inspect their metadata
    # for a common precision without materializing any later lazy frames.
    T = float(foldl(promote_type,
                    (f isa AbstractMatrix ? eltype(f) :
                     f isa AbstractString ? image_type : eltype(img_first) for f in frames);
                    init=eltype(img_first)))
    p1 = convert_particles(T, detect_particles(img_first, params; mask))

    all_tracks = _Track{T}[]
    active = _Track{T}[]
    for i in 1:length(p1)
        tr = _Track{T}(1, T[p1.x[i]], T[p1.y[i]], Int[1],
                       T(p1.intensity[i]), T(p1.diameter[i]), 0)
        push!(all_tracks, tr)
        push!(active, tr)
    end

    prev_matches = nothing   # (mx, my, mu, mv) of the last transition's accepted links
    n_trans = n_frames - 1
    meter = Progress(n_trans; desc = "Tracking: ", enabled = progress === true)
    for k in 1:n_trans
        img_next = load_frame(frames[k + 1], image_type)
        size(img_next) == image_size ||
            throw(DimensionMismatch("all frames must have the same size"))
        pb = convert_particles(T, detect_particles(img_next, params; mask))
        field = k == 1 ? resolve_predictor(predictor, img_first, img_next, piv_passes, mask, T) :
                (prev_matches === nothing ? nothing :
                 grid_predictor(prev_matches..., image_size, T))
        k == 1 && (img_first = nothing)
        interp = make_field_interp(field)

        M = length(active)
        pred_x = Vector{T}(undef, M)
        pred_y = Vector{T}(undef, M)
        for (i, tr) in enumerate(active)
            hx = tr.x[end]; hy = tr.y[end]
            elapsed = (k + 1) - tr.frames[end]
            if length(tr.x) >= 2
                dtlast = tr.frames[end] - tr.frames[end - 1]
                pred_x[i] = hx + (tr.x[end] - tr.x[end - 1]) * elapsed / dtlast
                pred_y[i] = hy + (tr.y[end] - tr.y[end - 1]) * elapsed / dtlast
            elseif interp === nothing
                pred_x[i] = hx
                pred_y[i] = hy
            else
                pred_x[i] = hx + T(interp[1](hy, hx)) * elapsed
                pred_y[i] = hy + T(interp[2](hy, hx)) * elapsed
            end
        end

        cl_b = build_cell_list(pb.x, pb.y, params.search_radius)
        index_a, index_b, _ = greedy_match(pred_x, pred_y, cl_b, params.search_radius;
            intensity_a=T[tr.intensity for tr in active], intensity_b=pb.intensity,
            diameter_a=T[tr.diameter for tr in active], diameter_b=pb.diameter,
            intensity_weight=params.intensity_weight, diameter_weight=params.diameter_weight)

        # Scattered UOD on this transition's matches, at the frame-k head
        # positions; flagged links are rejected below.
        nm = length(index_a)
        hx = T[active[index_a[m]].x[end] for m in 1:nm]
        hy = T[active[index_a[m]].y[end] for m in 1:nm]
        gaps = Int[(k + 1) - active[index_a[m]].frames[end] for m in 1:nm]
        mu = T[(pb.x[index_b[m]] - hx[m]) / gaps[m] for m in 1:nm]
        mv = T[(pb.y[index_b[m]] - hy[m]) / gaps[m] for m in 1:nm]
        flags = params.uod_enable ? scattered_uod(hx, hy, mu, mv, params) : falses(nm)

        extended = falses(M)
        accepted_b = falses(length(pb))
        amx = T[]; amy = T[]; amu = T[]; amv = T[]
        for m in 1:nm
            flags[m] && continue
            ia = index_a[m]; ib = index_b[m]
            push!(active[ia].x, pb.x[ib])
            push!(active[ia].y, pb.y[ib])
            push!(active[ia].frames, k + 1)
            active[ia].intensity = pb.intensity[ib]
            active[ia].diameter = pb.diameter[ib]
            active[ia].missed = 0
            extended[ia] = true
            accepted_b[ib] = true
            push!(amx, hx[m]); push!(amy, hy[m]); push!(amu, mu[m]); push!(amv, mv[m])
        end

        newactive = _Track{T}[]
        for (i, tr) in enumerate(active)
            if extended[i]
                push!(newactive, tr)
            else
                tr.missed += 1
                tr.missed <= max_gap && push!(newactive, tr)
            end
        end
        for j in 1:length(pb)
            accepted_b[j] && continue
            tr = _Track{T}(k + 1, T[pb.x[j]], T[pb.y[j]], Int[k + 1],
                           T(pb.intensity[j]), T(pb.diameter[j]), 0)
            push!(all_tracks, tr)
            push!(newactive, tr)
        end
        active = newactive
        prev_matches = (amx, amy, amu, amv)

        progress isa Function ? progress(k, n_trans) : next!(meter)
    end

    kept = [tr for tr in all_tracks if length(tr.x) >= min_track_length]
    sort!(kept; by = tr -> (tr.start_frame, tr.x[1], tr.y[1]))
    trajectories = [Trajectory{T}(tr.start_frame, tr.x, tr.y, tr.frames) for tr in kept]
    return TrackingResult{T}(trajectories, n_frames, params, scale)
end

"""
    trajectory_velocities(t::Trajectory, scale = nothing) -> (u, v)

Estimate one velocity at each observed trajectory point. The endpoints use
one-sided differences and interior points use central differences, each
divided by the actual difference in frame indices so gaps are accounted for.
Without a scale, `u` and `v` are pixels per frame interval, along columns
and rows respectively. At least two observed points are required.

Pass the owning result's [`PhysicalScale`](@ref) as `scale` for physical
velocity units. A raw result uses `pixel_size / dt`; a result converted by
[`physical`](@ref) already has length coordinates and retains `dt`, so the
same call also works on its trajectories.
"""
function trajectory_velocities(t::Trajectory{T},
                               scale::Union{Nothing,PhysicalScale} = nothing) where {T}
    n = length(t)
    n >= 2 || throw(ArgumentError("trajectory_velocities needs at least 2 points, got $n"))
    u = Vector{T}(undef, n)
    v = Vector{T}(undef, n)
    u[1] = (t.x[2] - t.x[1]) / (t.frames[2] - t.frames[1])
    v[1] = (t.y[2] - t.y[1]) / (t.frames[2] - t.frames[1])
    u[n] = (t.x[n] - t.x[n - 1]) / (t.frames[n] - t.frames[n - 1])
    v[n] = (t.y[n] - t.y[n - 1]) / (t.frames[n] - t.frames[n - 1])
    for i in 2:(n - 1)
        u[i] = (t.x[i + 1] - t.x[i - 1]) / (t.frames[i + 1] - t.frames[i - 1])
        v[i] = (t.y[i + 1] - t.y[i - 1]) / (t.frames[i + 1] - t.frames[i - 1])
    end
    if scale !== nothing && !(scale.pixel_size == 1.0 && scale.dt == 1.0)
        f = T(scale.pixel_size / scale.dt)
        u .*= f
        v .*= f
    end
    return u, v
end
