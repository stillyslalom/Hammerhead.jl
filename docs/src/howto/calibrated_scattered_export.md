# Export calibrated particle and trajectory tables

Use `export_calibrated_table` to apply a `PlanarTransform` to raw `PTVResult`,
`TrackingResult` or `TimedTrackingResult` data. It creates a CSV and a versioned
TOML companion. Ordinary `export_table` behavior stays unchanged. Keep both files:
the companion records the affine matrix and offset, coordinate conventions,
units, timing policy, diagnostic availability, CSV schema, hash and row count.

```@example calibrated_scattered_export
using Hammerhead
using Hammerhead.SyntheticData: generate_gaussian_particle!

images = [zeros(96, 96) for _ in 1:2]
for (i, image) in enumerate(images), (x, y) in ((20., 20.), (30., 45.), (50., 65.))
    generate_gaussian_particle!(image, (x + 2(i-1), y), 3., 1.)
end
initial = (x=[1., 96.], y=[1., 96.], u=fill(2., 2, 2), v=zeros(2, 2))
particles = run_ptv(images[1], images[2],
    PTVParameters(search_radius=0.7, uod_enable=false); predictor=initial)
transform = PlanarTransform([0.1 0.; 0. -0.2], [-1., 10.])

directory = mktempdir()
files = export_calibrated_table(joinpath(directory, "particles.csv"), particles;
    transform, length_unit="mm", coordinate_frame="laboratory_xy",
    dt=0.01, time_unit="s")
report = load_calibrated_table_metadata(files.metadata_path)
metadata = calibrated_table_data(report)
@assert metadata["row_count"] == length(particles.x)
@assert metadata["time"]["quantity"] == "pair_mean_velocity"
@assert metadata["verification"]["csv_structure_and_hash_at_load"]
report
```

Positions use `A*p+b`, with source x along image columns and y along image rows.
Vectors use `A*d`, with no offset. The example reverses the image y direction and
uses different calibration factors for each axis. `coordinate_frame` is a
user-provided opaque label, not proof that two exports share a physical frame.
Unit labels describe the supplied transform; they do not perform conversions.

The result must contain raw pixels and have no attached `PhysicalScale`. A timed
wrapper must also retain its pixel position basis and pass its binding check.
Removing a scale cannot undo an earlier numerical conversion. Exporting already
converted coordinates as raw pixels would misapply the transform.

## Choose the time basis

| Input | Without `dt` | With `dt` and `time_unit` |
|:--|:--|:--|
| PTV | Pair displacement in `length_unit` | Pair mean velocity in `length_unit/time_unit` |
| Ordinal tracking | Observation secants in `length_unit/frame` | Secants over a uniform selected-frame interval |
| Timed tracking | Secants over actual sample durations | Rejected; actual sample times determine durations |

For ordinal tracking, `dt` denotes the interval between adjacent **selected input
frames**. Gaps multiply that interval by the selected-frame index difference.
No acquisition timestamp is invented. For PTV displacement, the CSV's
`velocity_unit` is blank; `vector_quantity` and `vector_unit` identify the exported
quantity explicitly.

Timed tracking keeps exact sample and elapsed numerator/denominator columns,
source indices/IDs/labels and acquisition clock/unit labels. Endpoint velocities
use the adjacent observation pair; interior velocities use the outer secant.
These are interval-average slopes, not instantaneous irregular-grid derivatives.
The vector transform acts on original coordinate differences before division by
the exact duration, avoiding subtraction of rounded transformed large-origin
positions. `PhysicalScale.dt` is not used.

```julia
files = export_calibrated_table("tracks.csv", timed_tracks;
    transform, length_unit="mm", coordinate_frame="laboratory_xy")
```

A known chosen-timeline unit cannot be overridden. If it is unknown, the vector
unit remains unknown unless `time_unit` is supplied explicitly. That label is
recorded as an `explicit_export_time_unit_assumption`; it never rewrites original
acquisition metadata or rescales timestamps. A single-observation track has blank
velocity components and `velocity_valid=false`.

## Interpret diagnostics and precision

PTV outlier decisions are retained from original pixel-space validation. Tracking
does not record equivalent per-observation outlier decisions. Finite-position and
finite-vector flags describe the exported arithmetic, not scientific acceptance.
Invalid original nonfinite samples remain invalid rows; finite arithmetic that
overflows or loses a nonzero value to underflow is rejected before publication.

PTV retains only a scalar pixel match residual, so its direction cannot be
transformed under an anisotropic calibration. The calibrated `match_residual`
column is blank for every transform; `match_residual_pixel` preserves the original
diagnostic with unit `px`. Tracking match residuals and uncertainty are not
recorded. No transformed uncertainty or new validator decision is invented.

The applied affine coefficients and output numbers use Float64. Metadata records
the original transform coefficient type and each coefficient's precision, plus
the applied matrix and offset. Nonfinite coefficients, finite-to-infinite or
nonzero-to-zero conversion, and singular applied matrices are rejected. Exact
determinant checks preserve nonsingular matrices whose floating determinant would
underflow. These checks do not certify that a poorly conditioned calibration is
scientifically useful.

## Publish and verify the pair

By default, both destinations must be fresh. `overwrite=true` permits replacing
unrelated files. Known input aliases, including aliases between the two outputs,
are always refused. Prospective paths resolve existing parent links; Windows
checks conservatively fold case and reject trailing-dot/space path components.
The writer rechecks output identity after publishing the CSV, before publishing
the companion. Bare results do not know every input path: supply such paths
through `protected_paths`. The selected `ResultFile` overload protects its native
source automatically:

```julia
index = ResultFile("raw-results.jld2")
files = export_calibrated_table("selected.csv", index, 1;
    transform, length_unit="mm", coordinate_frame="laboratory_xy")
```

Both artifacts are fully prepared and closed before sequential publication.
Publication of two files is **not atomic**. An I/O failure may leave an incomplete
or mismatched pair; retain the previous pair separately if replacement recovery
is required. Concurrent writers and external source mutation are unsupported.

The default reader checks the metadata schema and checksum, then streams the CSV
to check its exact header, quoting, column counts, row count and SHA-256. Missing
or mismatched CSVs are rejected. It does not validate numeric values against a
result or authenticate acquisition inputs. Checksums are integrity checks, not
signatures. Empty tables still carry the full geometry and timing policy.

```julia
metadata_only = load_calibrated_table_metadata(files.metadata_path; verify_csv=false)
# A relocated byte-identical CSV can be supplied explicitly:
verified = load_calibrated_table_metadata(files.metadata_path; csv_path="moved.csv")
data = calibrated_table_data(verified)
```

`verify_csv=false` checks only the companion and does not read the CSV.
Verification flags describe checks performed at load time; retrieving copied
metadata does not reopen files. `protected_locators` retains recorded provenance
verbatim, including foreign Windows/POSIX absolute locators. The accessor's
computed `local_protected_paths` includes locally interpretable original sources
and the actual local companion/associated CSV paths, which can be supplied to
later exports:

```julia
export_calibrated_table("another.csv", raw_tracks;
    transform, length_unit="mm", coordinate_frame="laboratory_xy",
    protected_paths=data["local_protected_paths"])
```

The report retains
no result arrays or CSV rows. Export and verification stream rows; timed metadata
also retains O(selected frames) scalar acquisition context.

Original source locators are validated lexically, rather than with the receiving
host's absolute-path rules. Foreign locators are never resolved relative to the
current directory or treated as verified local inputs. No image/source relocation
is inferred. Supply relocated local sources explicitly in `protected_paths`;
foreign-looking explicit writer/reader arguments are refused before path
resolution. Existing version-1 metadata and integrity encoding remain unchanged.
On POSIX, a recorded backslash-separated relative CSV locator needs an explicit
receiving-host `csv_path`; the reader does not translate or guess separators,
including with `verify_csv=false`. Forward-slash relative associations work on
both platforms. Leading `//server/share` is conservatively classified as Windows
UNC because version 1 has no dialect tag; use a single leading slash for an
unambiguous POSIX absolute locator. Drive-relative/current-drive locators and
malformed UNC roots are refused.

Portability regression tests simulate foreign serialized locators on the current
Windows runtime, preserving scientific bindings and native alias guards. They do
not establish execution on Linux/macOS or an actual UNC share.

See the [calibrated table reference](../reference/calibrated_table.md) for API
signatures and the [actual-time tracking guide](tracking_timing.md) for time
selection and secant conventions.
