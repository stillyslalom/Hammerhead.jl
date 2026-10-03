# Stereo experiment records

Stereo recipes use an independent primitive version-1 schema. Camera matrices,
polynomial coefficients, normalization, rigid transform, signed grid, raw image
sizes, processing settings and ordered input/timing descriptors are explicit.
Typed camera objects are reconstructed through validation rather than persisted
as opaque JLD2 camera payloads. Fitted Pinhole coefficients retain their original
Float64 bytes without a second normalization; the public constructor is unchanged.

Settings identity excludes descriptive calibration notes/supplied self-calibration
summaries and environment-dependent reference-map digests. The full snapshot hash
protects those fields. Scientific input identity binds ordered camera roles,
frame byte hashes/dimensions, provided timestamps/delays/unit/clock labels and
pair-list versus four-tuple scaling mode. Paths and opaque IDs remain protected
record metadata but do not alter that location-independent identity.

Native results, execution diagnostics and timing companions retain their existing
formats. A separate versioned run sibling binds this record and completed raw
measurement fields. Metadata integrity is distinct from streaming result checks,
current input-byte checks, calibration accuracy and source authenticity. On failed
runs the verifier checks only completed acquisitions and explicitly reports any
single trailing unverified entry.

```@autodocs
Modules = [Hammerhead]
Pages = ["src/stereo_experiments.jl"]
Public = true
```
