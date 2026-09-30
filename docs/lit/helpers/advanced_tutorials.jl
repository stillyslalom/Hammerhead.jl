# Scene construction and reference matching for the advanced tutorials.
# Kept separate so the tutorial pages can focus on the analysis steps.

function tutorial_camera(yaw_deg; f = 2600.0, cx = 192.0, cy = 192.0, dist = 500.0)
    th = deg2rad(yaw_deg)
    R = [cos(th) 0.0 -sin(th); 0.0 1.0 0.0; sin(th) 0.0 cos(th)]
    C = R' * [0.0, 0.0, -dist]
    K = [f 0.0 cx; 0.0 -f cy; 0.0 0.0 1.0]
    PinholeCamera(K, R, -R * C)
end

function tutorial_sheet_image(cam, points, image_size, sheet;
                              displacement = (0.0, 0.0, 0.0))
    img = zeros(image_size)
    for (X, Y) in points
        Z = sheet.a + sheet.b * X + sheet.c * Y
        p = world_to_pixel(cam, (X + displacement[1], Y + displacement[2],
                                 Z + displacement[3]))
        Hammerhead.SyntheticData.generate_gaussian_particle!(img, (p[1], p[2]), 6.0)
    end
    img
end

function tutorial_ptv_diagnostics(ptv, truthA, truthB; tolerance = 0.5)
    function nearest_truth(p, truth)
        map = fill(0, length(p))
        for i in 1:length(p)
            best, index = Inf, 0
            for j in eachindex(truth.x)
                d2 = (p.x[i] - truth.x[j])^2 + (p.y[i] - truth.y[j])^2
                d2 < best && ((best, index) = (d2, j))
            end
            best < tolerance^2 && (map[i] = index)
        end
        map
    end
    mA = nearest_truth(ptv.particles_a, truthA)
    mB = nearest_truth(ptv.particles_b, truthB)
    errors = Float64[]
    identifiable = 0
    correct = 0
    switched = 0
    for k in eachindex(ptv.index_a)
        ta, tb = mA[ptv.index_a[k]], mB[ptv.index_b[k]]
        (ta > 0 && tb > 0) || continue
        identifiable += 1
        if ta == tb
            correct += 1
            push!(errors, hypot(ptv.u[k] - (truthB.x[tb] - truthA.x[ta]),
                                ptv.v[k] - (truthB.y[tb] - truthA.y[ta])))
        else
            switched += 1
        end
    end
    (detections_a = length(ptv.particles_a), truth_a = length(truthA.x),
     linked = length(ptv.index_a), identifiable = identifiable,
     unmatched_a = length(ptv.particles_a) - length(unique(ptv.index_a)),
     identity_switches = switched,
     correct_fraction = identifiable == 0 ? NaN : correct / identifiable,
     median_error_px = isempty(errors) ? NaN : median(errors))
end

function tutorial_gui_pair()
    center, rc, circulation = (128.0, 128.0), 40.0, 1200.0
    function flow(x, y, z, t)
        dx, dy = x - center[1], y - center[2]
        r2 = dx^2 + dy^2
        k = r2 < 1e-9 ? circulation / (2π * rc^2) :
            circulation / (2π * r2) * (1 - exp(-r2 / rc^2))
        (-k * dy, k * dx, 0.0)
    end
    a, b, _, _ = Hammerhead.SyntheticData.generate_synthetic_piv_pair(
        flow, (256, 256), 1.0; particle_density = 0.05,
        background_noise = 0.03, z_range = (-1.0, 1.0),
        rng = Random.MersenneTwister(42))
    # A fixed bright reflection obscures particle images in both exposures.
    a[12:60, 12:60] .= 1.0
    b[12:60, 12:60] .= 1.0
    (a, b)
end

function tutorial_gui_scale_plate()
    plate = zeros(256, 256)
    p1, p2 = (64.0, 128.0), (128.0, 128.0)
    Hammerhead.SyntheticData.generate_gaussian_particle!(plate, p1, 6.0)
    Hammerhead.SyntheticData.generate_gaussian_particle!(plate, p2, 6.0)
    (plate, p1, p2)
end
