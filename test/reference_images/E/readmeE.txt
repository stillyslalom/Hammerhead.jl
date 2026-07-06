Subset of case E of the 4th International PIV Challenge (2014):
time-resolved stereo PIV of a vortex ring (Re ~ 2300), recorded by
R. R. La Foy in the AEThER laboratory at Virginia Tech. Full data set
and instructions: https://www.pivchallenge.org/ ; challenge summary:
Kaehler et al., "Main results of the 4th International PIV Challenge",
Exp Fluids 57:97 (2016).

Acquisition: two Photron FASTCAM cameras, 1024 x 1024 px, 105 mm
lenses. Camera 1 is normal to the light sheet; camera 3 is tilted
about 25 degrees about the horizontal axis, without a Scheimpflug
adapter (aperture stopped down instead). Dual-cavity 527 nm Nd:YLF
laser, both heads pulsed simultaneously at 1000 Hz, so consecutive
frames are 1 ms apart. Seeding: 27 um fluorescent polystyrene
spheres.

Calibration: LaVision Type #21 two-level dot plate (15 mm dot
spacing per level, 3 mm level separation, 3.2 mm dots), traversed
through seven Z positions at 1 mm spacing. The world origin is the
dot 30 mm to the right of and 7.5 mm above the filled square marker
(origin_offset = (30.0, 7.5)).

Files here (losslessly re-encoded from the distributed uncompressed
16-bit TIFFs to 16-bit PNG; pixel values are unchanged):

  E_camera_{1,3}_z_{1,4,7}.png         calibration plate at
                                       z = -3, 0, +3 mm (planes 1, 4, 7
                                       of the seven; enough for a
                                       Soloff fit)
  E_camera_{1,3}_frame_000{50,51}.png  consecutive particle frames,
                                       t = 0.049 s and 0.050 s

This subset backs the real-data stereo tutorial in the documentation
and the stereo reference test; the full 100-frame sequence is
available from pivchallenge.org.
