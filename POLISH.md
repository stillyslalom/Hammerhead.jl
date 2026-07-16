# Hammerhead polish backlog

Hammerhead's numerical core and the baseline planar, stereo, PTV, batch,
masking, analysis, export, and calibration workflows are implemented. This
file tracks only the remaining work, grouped into suggested delivery slices
rather than historical priority buckets. The step numbers express a useful
order, not a global dependency chain: Steps 2–6 can proceed independently,
subject to the within-step dependencies called out below.

Tomographic PIV, volumetric reconstruction, acquisition hardware control,
Shake-the-Box, pressure-from-PIV, and modal-analysis suites remain out of scope
unless the package scope changes.

## Step 0: register the core package

Core registration does not freeze a pre-1.0 API, and making installation
routine is the best way to get the real-user feedback needed to stabilize the
remaining interfaces.

- [ ] Complete General-registry registration for the core package.

## Step 1: stabilize advanced correlation semantics

Build the search-area abstraction first because it defines window geometry,
output-grid coordinates, masking behavior, deformation, retained planes, and
CPU/GPU backend contracts. Evaluate other correlation changes only after those
semantics are stable.

- [ ] Support independent interrogation and search-area sizes for large
  first-pass displacement without increasing the particle-sampling window.
- [ ] Evaluate variance-normalized cross-correlation for strong local intensity
  and contrast changes; document when phase correlation or CLAHE is preferable.
- [ ] Consider adaptive/nonuniform interrogation after the search-area API and
  output-grid semantics are stable.
- [ ] Revisit ensemble pass iteration (`max_iterations` is currently ignored)
  if low-SNR accuracy cases demonstrate a practical benefit.

## Step 2: add PTV/trajectory GUI support and particle uncertainty

Basic browsing does not depend on an uncertainty estimator and should ship
first. Uncertainty overlays should follow only after particle-fit and match
uncertainty have defensible semantics. The GUI should consume the existing
package-native persistence rather than inventing a GUI-only representation.

- [x] Load and visualize persisted `PTVResult` and `TrackingResult` values in
  the GUI, including attached-scale axis labels and trajectory gaps.
- [ ] Add per-particle position and displacement uncertainty.
- [ ] Add particle/displacement uncertainty overlays to the GUI.

## Step 3: finish interactive planar workflow ergonomics

Reuse the existing `ROI` and `PlanarTransform` core APIs so GUI selections and
exports share exactly the same coordinate semantics.

- [ ] Make ROI selection editable in the GUI.
- [ ] Provide a GUI calibration-line tool and make the resulting planar
  coordinate transform explicit in table and VTK exports.

## Step 4: harden rotated-target calibration

The indexing algorithm is rotation-invariant via
`orientation = :fiducials`, which requires both square and triangle markers;
the default `:image` convention remains image-oriented. The remaining work is
regression coverage across rendering conditions and real optics.

- [ ] Add rolled-camera synthetic fixtures covering perspective, noise, marker
  visibility, and two-level targets.
- [ ] Add at least one real rotated-target regression case while preserving the
  current world-axis and marker-origin conventions.

## Step 5: mature ingestion, archival formats, and real-data examples

Keep camera/video formats behind adapters or weak dependencies. Stabilize the
existing table and VTK contracts with real users before adding another archival
schema.

- [ ] Add optional video ingestion through a weak dependency or frame-source
  adapter.
- [ ] Consider HDF5 or netCDF after the table and VTK schemas have stabilized.
- [ ] Add end-to-end real sequence examples once download and caching fit the
  documentation CI budget.

## Step 6: add validation-dependent analysis features

These items need explicit physical/statistical assumptions or a contributed
validated dataset before an API should be committed.

- [ ] Propagate measurement uncertainty into derived quantities after defining
  spatial error-correlation assumptions.
- [ ] Add optional morphological phase separation for two-phase images when a
  validated use case is contributed.
- [ ] Estimate light-sheet thickness/overlap from disparity-correlation peak
  widths once the Wieneke 2005 §5 model has a validated stereo fixture.

## Step 7: package and distribute the GUI

Do this after the remaining public APIs and desktop workflow are sufficiently
stable to avoid publishing short-lived installation contracts.

- [ ] Complete General-registry registration for the GUI package.
- [ ] Evaluate a PackageCompiler desktop bundle for non-Julia GUI users.

## Reference expectations

- [OpenPIV settings and API](https://openpiv.readthedocs.io/en/latest/src/openpiv.html)
  expose ROI selection, dynamic masks, normalized correlation, flexible batch
  settings, validation, scaling, and conventional vector output.
- [OpenPIV masking](https://openpiv.readthedocs.io/en/stable/src/masking.html)
  distinguishes static and pair-dependent dynamic masks.
- [PIVlab](https://github.com/Shrediquette/PIVlab) combines acquisition,
  preprocessing, analysis, calibration, exploration, and export in one GUI.
- [PIVlab 3 workflow changes](https://pivlab.blogspot.com/2024/04/i-have-just-released-pivlab-3.html)
  emphasize editable and automatic masks, ROI/calibration ergonomics,
  line/area extraction, and image/video export.
- [LaVision 2D and Stereo PIV](https://www.lavision.de/en/products/flowmaster/2d-stereo-piv/)
  provides the commercial reference for stereo self-calibration,
  correlation-statistics uncertainty, and GPU processing.
