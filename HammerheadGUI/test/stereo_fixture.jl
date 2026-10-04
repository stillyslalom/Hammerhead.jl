# Synthetic stereo rig shared by the stereo workflow tests, the stereo canvas
# tests, and the stereo Qt window script.
#
# Two pinhole cameras at ±20° yaw (the core make_test_camera recipe),
# dot-plate images at z = -3/0/3 mm, and particle pairs on a light sheet
# offset to z = 0.8 mm, displaced by (1.0, 0.5) mm. Particle positions come
# from an inline 64-bit LCG (not an RNG stream, so the scenes are identical
# on every Julia version; a low-discrepancy sequence is too lattice-like for
# single-window correlation).

"""
    stereo_rig(; acquisitions = 3, particles = 1500)

`(; cams, zs, plates, files1, files2, sheet, disp)`: the rig's cameras,
plate z positions, plate images per camera, each camera's frames
(`2 × acquisitions` entries, A/B per acquisition), the sheet offset, and the
displacement.
"""
function stereo_rig(; acquisitions::Integer = 3, particles::Integer = 1500)
    gauss! = Hammerhead.SyntheticData.generate_gaussian_particle!
    function rig_camera(θdeg)
        θ = deg2rad(θdeg)
        R = [cos(θ) 0.0 -sin(θ); 0.0 1.0 0.0; sin(θ) 0.0 cos(θ)]
        camC = R' * [0.0, 0.0, -500.0]
        K = [3500.0 0.0 256.0; 0.0 -3500.0 256.0; 0.0 0.0 1.0]
        return PinholeCamera(K, R, -R * camC)
    end
    cams = (rig_camera(20.0), rig_camera(-20.0))
    zs = [-3.0, 0.0, 3.0]
    plates = map(cams) do cam
        [render_calibration_target(cam, (512, 512); spacing = 15.0, z = z,
                                   marker_square = (-30.0, -7.5), marker_triangle = (-15.0, -7.5))
         for z in zs]
    end
    sheet, disp = 0.8, (1.0, 0.5)
    # uniform points in ±35 mm from Knuth's MMIX LCG
    lcg = Ref(0x2545f4914f6cdd1d)
    uniform() = (lcg[] = lcg[] * 0x5851f42d4c957f2d + 0x14057b7ef767814f; (lcg[] >> 11) / 2.0^53)
    points(n) = [(70 * uniform() - 35, 70 * uniform() - 35) for _ in 1:n]
    function stereo_pair(pts)
        map(cams) do cam
            A, B = zeros(512, 512), zeros(512, 512)
            for (X, Y) in pts
                pa = world_to_pixel(cam, (X, Y, sheet))
                pb = world_to_pixel(cam, (X + disp[1], Y + disp[2], sheet))
                gauss!(A, (pa[1], pa[2]), 5.0)
                gauss!(B, (pb[1], pb[2]), 5.0)
            end
            (A, B)
        end
    end
    acqs = [stereo_pair(points(particles)) for _ in 1:acquisitions]
    files1 = Any[f for a in acqs for f in a[1]]
    files2 = Any[f for a in acqs for f in a[2]]
    return (; cams, zs, plates, files1, files2, sheet, disp)
end

# The fixture's plates and detection settings in a StereoCalibration.
function add_fixture_plates!(cal, fx)
    for k in 1:2, (img, z) in zip(fx.plates[k], fx.zs)
        add_plate!(cal, k, img, z)
    end
    set_calibration_option!(cal, :spacing, "15")
    set_calibration_option!(cal, :origin_offset, "30, 7.5")
    set_calibration_option!(cal, :grid_spacing, 0.3)
    return cal
end
