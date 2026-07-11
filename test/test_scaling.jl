using Hammerhead
using Test
using Statistics
using Unitful: @u_str, unit

# Physical units: PhysicalScale metadata, with_scale/physical conversion,
# driver plumbing, and the Unitful extension. Uses the particle_pair helper
# from runtests.jl.

@testset "Physical units" begin

@testset "PhysicalScale" begin
    s0 = PhysicalScale()
    @test s0.pixel_size == 1.0 && s0.dt == 1.0
    @test s0.length_unit == "px" && s0.time_unit == "frame"
    @test Hammerhead.is_identity(s0)
    @test Hammerhead.velocity_unit(s0) == "px/frame"

    s = PhysicalScale(pixel_size = 0.02, dt = 1e-3, length_unit = "mm", time_unit = "s")
    @test !Hammerhead.is_identity(s)
    @test Hammerhead.velocity_unit(s) == "mm/s"
    @test occursin("mm", string(s)) && occursin("0.02", string(s))
    @test occursin("identity", string(s0))

    @test_throws ArgumentError PhysicalScale(pixel_size = 0.0)
    @test_throws ArgumentError PhysicalScale(pixel_size = -1.0)
    @test_throws ArgumentError PhysicalScale(dt = 0.0)
    @test_throws ArgumentError PhysicalScale(dt = NaN)
    @test_throws ArgumentError PhysicalScale(pixel_size = Inf)
end

# Hand-built fixtures shared by the conversion testsets.
function scaled_piv_fixture(::Type{T} = Float64; scale = nothing) where {T}
    x = collect(T, 16:16:64)
    y = collect(T, 16:16:48)
    ny, nx = length(y), length(x)
    u = T[i + j for i in 1:ny, j in 1:nx]
    v = -T[i * j for i in 1:ny, j in 1:nx]
    pr = fill(T(2), ny, nx)
    cm = fill(T(0.3), ny, nx)
    uu = fill(T(0.05), ny, nx)
    uv = fill(T(0.07), ny, nx)
    mask = falses(ny, nx)
    mask[1, 1] = true
    for f in (u, v, pr, cm, uu, uv)
        f[1, 1] = T(NaN)
    end
    r = PIVResult(x, y, u, v, pr, cm, uu, uv, falses(ny, nx), mask, PIVParameters())
    return scale === nothing ? r : with_scale(r, scale)
end

@testset "back-compat constructors leave scale === nothing" begin
    r = scaled_piv_fixture()
    @test r.scale === nothing && r.correlation_planes === nothing
    r12 = PIVResult(r.x, r.y, r.u, r.v, r.peak_ratio, r.correlation_moment,
                    r.uncertainty_u, r.uncertainty_v, r.outliers, r.mask,
                    r.parameters, nothing)
    @test r12.scale === nothing
    rT = PIVResult{Float64}(r.x, r.y, r.u, r.v, r.peak_ratio, r.correlation_moment,
                            r.uncertainty_u, r.uncertainty_v, r.outliers, r.mask,
                            r.parameters)
    @test rT.scale === nothing

    st = StereoPIVResult(r.x, r.y, 0.0, r.u, r.v, r.u, r.uncertainty_u,
                         r.uncertainty_v, r.uncertainty_u, r.outliers, r.mask,
                         r, r, r.parameters)
    @test st isa StereoPIVResult{Float64} && st.scale === nothing
    stT = StereoPIVResult{Float64}(r.x, r.y, 0.0, r.u, r.v, r.u, r.uncertainty_u,
                                   r.uncertainty_v, r.uncertainty_u, r.outliers,
                                   r.mask, r, r, r.parameters)
    @test stT.scale === nothing
end

@testset "with_scale attaches/strips metadata only" begin
    s = PhysicalScale(pixel_size = 0.02, dt = 1e-3, length_unit = "mm", time_unit = "s")
    r = scaled_piv_fixture()
    rs = with_scale(r, s)
    @test rs.scale === s
    @test rs.u === r.u && rs.v === r.v && rs.x === r.x && rs.y === r.y &&
          rs.peak_ratio === r.peak_ratio && rs.uncertainty_u === r.uncertainty_u &&
          rs.outliers === r.outliers && rs.mask === r.mask
    @test with_scale(rs, nothing).scale === nothing
    @test with_scale(rs, nothing).u === r.u
end

@testset "physical(::PIVResult)" begin
    s = PhysicalScale(pixel_size = 0.02, dt = 1e-3, length_unit = "mm", time_unit = "s")
    fp, fv = 0.02, 0.02 / 1e-3
    r = scaled_piv_fixture()
    @test physical(r) === r                      # no scale attached: identity
    rs = with_scale(r, s)
    p = physical(rs)
    @test p !== rs
    @test p.x ≈ r.x .* fp && p.y ≈ r.y .* fp
    @test isequal(p.u, r.u .* fv) && isequal(p.v, r.v .* fv)
    @test isequal(p.uncertainty_u, r.uncertainty_u .* fv)
    @test isequal(p.uncertainty_v, r.uncertainty_v .* fv)
    # px-native diagnostics/flags are shared, not converted.
    @test p.peak_ratio === r.peak_ratio && p.correlation_moment === r.correlation_moment
    @test p.outliers === r.outliers && p.mask === r.mask
    @test p.correlation_planes === nothing
    # Masked cells stay NaN.
    @test isnan(p.u[1, 1]) && isnan(p.uncertainty_u[1, 1])
    # Identity scale with the labels preserved; idempotent.
    @test Hammerhead.is_identity(p.scale)
    @test p.scale.length_unit == "mm" && p.scale.time_unit == "s"
    @test physical(p) === p
    # Two-arg form attaches then converts.
    p2 = physical(r, s)
    @test isequal(p2.u, p.u) && p2.scale == p.scale
    # Attaching an identity scale is a pure no-op for physical().
    ri = with_scale(r, PhysicalScale())
    @test physical(ri) === ri

    # Float32 results stay Float32.
    r32 = scaled_piv_fixture(Float32; scale = s)
    p32 = physical(r32)
    @test p32 isa PIVResult{Float32}
    @test eltype(p32.u) == Float32 && eltype(p32.x) == Float32
    @test p32.u[2, 2] ≈ Float32(r32.u[2, 2]) * Float32(fv)
end

@testset "physical(::StereoPIVResult)" begin
    cam = scaled_piv_fixture()
    ny, nx = size(cam.u)
    s = PhysicalScale(dt = 5e-4, length_unit = "mm", time_unit = "s")
    st = StereoPIVResult(cam.x, cam.y, 1.5, cam.u, cam.v, copy(cam.u),
                         cam.uncertainty_u, cam.uncertainty_v, copy(cam.uncertainty_u),
                         cam.outliers, cam.mask, cam, cam, cam.parameters)
    sts = with_scale(st, s)
    @test sts.u === st.u                          # metadata only
    p = physical(sts)
    # dt-only scale: positions untouched, velocities divided by dt (the
    # factor is written as it is computed internally, for bitwise equality).
    fv = 1.0 / 5e-4
    @test p.x == st.x && p.y == st.y && p.z == st.z
    @test isequal(p.u, st.u .* fv) && isequal(p.w, st.w .* fv)
    @test isequal(p.uncertainty_v, st.uncertainty_v .* fv)
    # Per-camera results are never converted.
    @test p.cam1 === st.cam1 && p.cam2 === st.cam2 && p.cam1.scale === nothing
    @test Hammerhead.is_identity(p.scale) && p.scale.length_unit == "mm"
    @test physical(p) === p
    # A non-1 pixel_size acts as a world-length unit conversion (mm → µm).
    pµ = physical(st, PhysicalScale(pixel_size = 1000.0, dt = 5e-4,
                                    length_unit = "µm", time_unit = "s"))
    @test pµ.x ≈ st.x .* 1000
    @test pµ.z ≈ st.z * 1000
    @test pµ.u[2, 2] ≈ st.u[2, 2] * 1000 / 5e-4
end

@testset "run_piv scale plumbing" begin
    image_size = (64, 64)
    imgA, imgB = particle_pair(image_size, [(20.0, 20.0), (20.0, 44.0),
                                            (44.0, 20.0), (44.0, 44.0)], 1.5, 2.5)
    s = PhysicalScale(pixel_size = 0.02, dt = 1e-3, length_unit = "mm", time_unit = "s")
    params = PIVParameters(window_size = 32, overlap = 16)
    r_raw = run_piv(imgA, imgB, params; threaded = false)
    r_sc = run_piv(imgA, imgB, params; threaded = false, scale = s)
    @test r_sc.scale === s
    @test r_raw.scale === nothing
    @test isequal(r_sc.u, r_raw.u) && isequal(r_sc.v, r_raw.v)   # metadata only
    p = physical(r_sc)
    @test isequal(p.u, r_sc.u .* (0.02 / 1e-3))

    # Effort path: scale is a driver kwarg, split away from PIV-parameter kwargs.
    r_eff = run_piv(imgA, imgB; effort = :low, threaded = false, scale = s)
    @test r_eff.scale === s

    # Sequence driver forwards scale to every pair.
    seq = run_piv_sequence([(imgA, imgB), (imgA, imgB)], params;
                           progress = false, scale = s)
    @test all(r.scale === s for r in seq)
end

@testset "run_piv_ensemble scale plumbing" begin
    image_size = (64, 64)
    imgA, imgB = particle_pair(image_size, [(20.0, 20.0), (20.0, 44.0),
                                            (44.0, 20.0), (44.0, 44.0)], 1.0, 2.0)
    s = PhysicalScale(pixel_size = 0.02, dt = 1e-3, length_unit = "mm", time_unit = "s")
    pairs = [(imgA, imgB), (imgA, imgB)]
    ens = run_piv_ensemble(pairs, PIVParameters(window_size = 32, overlap = 16);
                           progress = false, scale = s)
    @test ens.scale === s
    ens_eff = run_piv_ensemble(pairs; effort = :low, progress = false, scale = s)
    @test ens_eff.scale === s
    # Unknown kwargs are still rejected.
    @test_throws ArgumentError run_piv_ensemble(pairs; bogus = 1, progress = false)
end

@testset "PTV scale plumbing and conversion" begin
    image_size = (96, 96)
    positions = [(y, x) for y in 16.0:16.0:80.0 for x in 16.0:16.0:80.0]
    imgA, imgB = particle_pair(image_size, positions, 0.8, 1.2)
    s = PhysicalScale(pixel_size = 0.05, dt = 2e-3, length_unit = "mm", time_unit = "s")
    fp, fv = 0.05, 0.05 / 2e-3

    r = run_ptv(imgA, imgB; predictor = nothing, scale = s)
    @test r.scale === s
    @test !isempty(r.x)
    p = physical(r)
    @test p.x ≈ r.x .* fp && p.y ≈ r.y .* fp
    @test p.u ≈ r.u .* fv && p.v ≈ r.v .* fv
    @test p.match_residual ≈ r.match_residual .* fp   # a frame-A distance
    @test p.particles_a === r.particles_a && p.index_a === r.index_a
    @test Hammerhead.is_identity(p.scale) && physical(p) === p

    # Empty-frame short-circuit still carries the scale.
    empty = run_ptv(zeros(64, 64), zeros(64, 64); predictor = nothing, scale = s)
    @test isempty(empty.x) && empty.scale === s

    # ptv_to_grid propagates the scale onto the binned grid.
    grid = ptv_to_grid(r, image_size; window_size = (48, 48), overlap = (24, 24),
                       min_count = 1)
    @test grid.scale === s
    pg = physical(grid)
    finite = .!grid.mask
    @test pg.u[finite] ≈ grid.u[finite] .* fv
end

@testset "tracking scale plumbing and trajectory velocities" begin
    n = 96
    base = [(y, x) for y in 20.0:18.0:80.0 for x in 20.0:18.0:80.0]
    dv, du = 0.7, 1.1
    frames = [zeros(n, n) for _ in 1:3]
    for (k, img) in enumerate(frames), pos in base
        add_particle!(img, (pos[1] + (k - 1) * dv, pos[2] + (k - 1) * du), 3.0)
    end
    s = PhysicalScale(pixel_size = 0.05, dt = 2e-3, length_unit = "mm", time_unit = "s")
    fp, fv = 0.05, 0.05 / 2e-3

    tr = track_particles(frames; predictor = nothing, progress = false, scale = s)
    @test tr.scale === s
    @test !isempty(tr.trajectories)
    t = tr.trajectories[1]
    u_px, v_px = trajectory_velocities(t)
    @test median(u_px) ≈ du atol = 0.2               # unchanged px/frame default
    u_ph, v_ph = trajectory_velocities(t, tr.scale)
    @test u_ph ≈ u_px .* fv && v_ph ≈ v_px .* fv

    # physical(): positions convert, dt survives in the returned scale so
    # trajectory velocities stay consistent on raw and converted results.
    ptr = physical(tr)
    @test ptr.scale.pixel_size == 1.0 && ptr.scale.dt == s.dt
    pt = ptr.trajectories[1]
    @test pt.x ≈ t.x .* fp && pt.y ≈ t.y .* fp
    @test pt.start_frame == t.start_frame
    u_ph2, _ = trajectory_velocities(pt, ptr.scale)
    @test u_ph2 ≈ u_ph
    @test physical(ptr) === ptr
end

@testset "JLD2 round-trip with scale" begin
    s = PhysicalScale(pixel_size = 0.02, dt = 1e-3, length_unit = "mm", time_unit = "s")
    r = scaled_piv_fixture(; scale = s)
    tmp = tempname() * ".jld2"
    save_results(tmp, r)
    loaded = load_results(tmp)[1]
    @test loaded.scale == s
    @test isequal(loaded.u, r.u)
    Base.rm(tmp; force = true)
end

@testset "Unitful extension" begin
    s = PhysicalScale(0.02u"mm", 0.001u"s")
    @test s.pixel_size == 0.02 && s.dt == 0.001
    @test s.length_unit == "mm" && s.time_unit == "s"
    @test s == PhysicalScale(pixel_size = 0.02, dt = 0.001,
                             length_unit = "mm", time_unit = "s")
    # Compare against Unitful's own unit printing (µ normalization varies).
    @test Hammerhead.velocity_unit(PhysicalScale(20.0u"µm", 0.5u"ms")) ==
          string(unit(20.0u"µm"), "/", unit(0.5u"ms"))

    imgA, imgB = particle_pair((64, 64), [(32.0, 32.0)], 1.0, 2.0)
    r = run_piv(imgA, imgB, PIVParameters(window_size = 32, overlap = 16);
                threaded = false, scale = s)
    @test r.scale === s
end

@testset "plot_axis_labels" begin
    @test Hammerhead.plot_axis_labels(nothing) == ("x (px)", "y (px)")
    s = PhysicalScale(pixel_size = 0.02, dt = 1e-3, length_unit = "mm", time_unit = "s")
    @test Hammerhead.plot_axis_labels(s) == ("x (mm)", "y (mm)")
end

end # Physical units
