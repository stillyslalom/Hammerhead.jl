# Hammerhead polish backlog

Hammerhead's numerical core is already competitive with established PIV
toolchains: multi-pass WIDIM with symmetric deformation, robust validation,
ensemble correlation, correlation-statistics uncertainty, stereo calibration
and self-calibration, GPU execution, and baseline 2D PTV are implemented.

The remaining practitioner-facing gaps are primarily workflow completeness and
interoperability rather than correlation accuracy. This backlog compares the
current package with common expectations established by OpenPIV, PIVlab, and
commercial 2D/stereo PIV workflows.

Tomographic PIV, volumetric reconstruction, acquisition hardware control,
Shake-the-Box, pressure-from-PIV, and modal-analysis suites are not baseline
targets here. They remain out of scope unless the package scope changes.

## P0: complete the advertised result workflows

### Stereo sequences and statistics

Stereo sequence, ensemble, temporal-validation, and statistics APIs now carry
the full 2D3C result through the same workflow as a planar result.

- [x] Add `run_piv_stereo_sequence` with synchronized camera sources,
  incremental output, preprocessing hooks, progress/cancellation, and reuse of
  dewarpers and PIV workspaces.
- [x] Generalize `field_statistics` to `StereoPIVResult`, including mean/RMS
  for all three components and all six Reynolds-stress terms.
- [x] Generalize temporal validation to stereo results.
- [x] Decide whether low-SNR stereo needs a first-class stereo ensemble driver
  or a documented composition of per-camera ensemble results followed by 3C
  reconstruction.
- [x] Add sequence coverage to the stereo GUI workflow.

### Interoperable result export

JLD2 is appropriate for lossless Julia round trips but is not a sufficient
exchange format for MATLAB, Python, ParaView, Tecplot, CFD comparison, or data
archival workflows.

- [x] Add a long-form table/CSV exporter for planar, stereo, and PTV results.
  Include coordinates, components, mask/outlier flags, quality metrics,
  uncertainty, frame/source identifiers, and units.
- [x] Add VTK structured-grid export for planar and stereo fields.
- [x] Define and document a stable, language-neutral column/schema contract.
- [x] Add `TrackingResult` to package-native persistence.
- [ ] Consider HDF5 or netCDF only after the table and VTK schemas stabilize.

### Dynamic masks

The current mask is one static lab-frame `Bool` matrix for a whole run. This is
insufficient for moving bodies, free surfaces, flexible structures, translating
models, and frame-dependent reflections.

- [x] Let sequence drivers accept a static mask, a mask sequence, or a callback
  such as `(i, frameA, frameB) -> mask`.
- [x] Define pair-mask semantics when the geometry differs between frames A
  and B (normally the union of both frame masks).
- [x] Add basic intensity-, contrast-, and edge-derived automatic masks.
- [x] Extend the GUI editor with holes and grow/shrink operations without
  weakening the existing `true = excluded` convention.

## P1: remove common ingestion and analysis friction

### Frame sources and pairing

`load_image` deliberately accepts only a single 2D image, and `image_pairs`
only constructs adjacent paired or chained sequences. Real recordings commonly
arrive as stacks, videos, camera containers, and multi-delay sequences.

- [x] Add multi-page TIFF stack support.
- [x] Introduce a small frame-source interface so image lists, stacks, videos,
  and user-defined camera readers can feed the same sequence drivers lazily.
- [ ] Add optional video ingestion through a weak dependency or adapter.
- [x] Add arbitrary frame stride/offset and multi-`delta t` pair generation.
- [x] Preserve per-frame timestamps and allow variable `dt` where the source
  supplies it.
- [x] Document the adapter pattern for proprietary CINE/MRAW/SEQ/vendor files;
  do not add hard dependencies on every camera format.

### Derived flow quantities

The built-in analysis currently stops at planar means, RMS components,
`mean(u'v')`, valid counts, and a scalar PSD. Common PIV exploration begins
with spatial derivatives and profile extraction.

- [x] Add mask-aware vorticity, divergence, and strain-rate calculations.
- [x] Add swirling strength and Q criterion with explicit 2D semantics.
- [x] Add line/profile and region/area extraction utilities.
- [x] Add circulation over a user-supplied contour or region.
- [x] Add a results-vector spectrum helper that obtains `dt` from attached
  scale metadata and handles invalid samples explicitly.
- [ ] Propagate measurement uncertainty into derived quantities once spatial
  error-correlation assumptions are defined.
- [x] Keep streamlines/pathlines and higher-level turbulence analysis as a
  second slice rather than coupling them to the initial derivative API.

### PTV trajectory robustness

The current tracker terminates a trajectory after one missed match. Real PTV
data routinely contains one- or two-frame detection dropout.

- [x] Add bounded gap bridging and track reacquisition.
- [x] Add optional intensity and diameter consistency to the match cost.
- [ ] Add per-particle position and displacement uncertainty.
- [ ] Persist and visualize `TrackingResult` and `PTVResult` in the GUI.
- [x] Treat stereo PTV as a separate later milestone; do not overload the 2D
  tracker design prematurely.

### Rotation-invariant calibration-target detection

The camera models accept manually supplied correspondences, but automatic
target detection assumes a roughly upright target/camera orientation. Camera
roll is common in constrained stereo arrangements.

- [x] Make lattice-axis and fiducial indexing invariant to arbitrary in-plane
  rotation.
- [ ] Add rolled-camera synthetic fixtures and at least one real rotated-target
  regression case.
- [x] Preserve the current world-axis and marker-origin conventions after
  rotation is resolved.

## P2: close smaller core-API gaps

### First-class region of interest

Users can crop matrices manually, but then must restore coordinate offsets and
keep masks, calibration, physical scaling, and exports consistent.

- [x] Add an ROI/offset abstraction to single-pair and sequence drivers.
- [x] Return coordinates in the original image frame by default.
- [ ] Make ROI selection editable in the GUI.

### Correlation options used in reference toolchains

- [ ] Support independent interrogation and search-area sizes for large first-
  pass displacement without increasing the particle-sampling window.
- [ ] Evaluate variance-normalized cross-correlation for strong local intensity
  and contrast changes; document when phase correlation or CLAHE is preferable.
- [ ] Consider adaptive/nonuniform interrogation only after the search-area API
  and output-grid semantics are stable.
- [ ] Revisit ensemble pass iteration (`max_iterations` is currently ignored)
  if low-SNR accuracy cases demonstrate a practical benefit.

### Planar calibration ergonomics

`PhysicalScale` supplies one pixel-size factor and one constant `dt`. Add a
convenient planar workflow without turning result arrays into unitful storage.

- [x] Add calibration from two image points plus a known physical distance.
- [x] Support axis rotation/reflection and anisotropic pixel scales where
  appropriate.
- [ ] Provide a GUI calibration-line tool and make the resulting coordinate
  transform explicit in exports.

### Preprocessing breadth

- [x] Add percentile contrast stretching and image inversion.
- [x] Evaluate local variance normalization against current high-pass/CLAHE
  workflows.
- [ ] Add optional morphological phase separation for two-phase images if a
  validated use case is contributed.
- [x] Keep all additions in-place-first and avoid duplicating maintained
  JuliaImages functionality.

## Release and adoption polish

- [ ] Complete General-registry registration for the core and GUI packages.
- [ ] Evaluate a PackageCompiler desktop bundle for non-Julia GUI users.
- [x] Publish a compact feature matrix covering CPU/KA/CUDA/AMDGPU and planar,
  ensemble, stereo, PTV, uncertainty, and retained-plane support.
- [ ] Add end-to-end real sequence examples once data download/caching can fit
  the documentation CI budget.
- [x] Add a public compatibility policy for saved results and external export
  schemas before the first stable release.

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
  correlation-statistics uncertainty, and GPU processing. Hammerhead already
  covers these core numerical capabilities; the backlog above focuses on the
  surrounding workflow.
