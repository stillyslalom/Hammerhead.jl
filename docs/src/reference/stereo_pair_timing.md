# [Stereo pair timing](@id stereo-pair-timing-reference)

```@autodocs
Modules = [Hammerhead]
Pages = ["src/stereo_pair_timing.jl"]
Public = true
```

The independent version-1 native sibling group is
`stereo_pair_timing/<result suffix>`, with root marker
`stereo_pair_timing_format_version=1`. An entry contains `result_key`, `timing`
and `timing_sha256`. Native result format 1, registered result layouts and the
planar pair-timing schema remain unchanged. The reader checks exact key linkage;
an unknown marker or a group without its marker is refused.

The payload contains:

| Field | Meaning |
|:--|:--|
| `input_sequence_index` | Acquisition ordinal in this invocation; distinct from source frame indices |
| `cameras` | Exactly two ordered camera roles, each with A/B frame descriptors, exact timestamp/delay encodings, separate midpoint and missing/partial unit/clock status |
| `delay_tolerance` | Exact-encoding agreement tolerances, separately recorded from synchronization tolerances |
| `synchronization_policy`, `synchronization` | Missing timestamp policy, delay-scaled bound, exact camera2-minus-camera1 exposure offsets, comparison arithmetic and metadata availability; no certification |
| `scale`, `effective_delay`, `effective_delay_provenance` | Reconstructed scale and legacy applied-delay/metadata distinction; retained camera fields remain raw dewarped pixels |
| `reconstructed_time_reference` | Camera 1's provided-timestamp midpoint, or absent; never a synthesized common midpoint |
| `geometry` | Provided common world-grid axes/plane, signed spacing and measurement shape; not a calibration identity |
| `raw_result_sha256`, `binding_basis` | Reconstructed and camera measurement-field binding, excluding parameters and correlation planes |

Each source frame descriptor preserves kind, label, original source index, opaque
source/frame IDs, original numeric timestamp encoding and optional time-unit/clock
labels. Standard signed/unsigned integers (including BigInt), Float16/32/64 and
standard-integer rationals are supported. Bool, BigFloat, custom Real and
nonfinite values are refused. Derived values use reduced exact rationals.

Schema validation independently recomputes timing fields and checks geometry.
Metadata-only reading verifies neither raw fields nor the accuracy of provided
source metadata. A supplied/loaded raw-result check additionally validates array
dimensions, camera axes, reconstructed coordinates and measurement hashes. A
recomputed metadata checksum is not evidence of source or calibration authenticity.

`pair_timing_data` returns copied metadata plus runtime-only `verification`:
`captured_measurement_fields`, `metadata_only`,
`loaded_measurement_fields_verified` or `supplied_measurement_fields_verified`.
The supplied-result accessor leaves the packet unchanged and retains no result.
These labels describe the check at capture/read/inspection, not continuing validity
of a caller's mutable result arrays. Integrity checks and supplied-result verification
are repeated on each public access. File size/mtime checks detect some changes;
concurrent writer support and cross-file authentication are outside this schema.

See [the sequence how-to](../howto/stereo_pair_timing.md) for synchronization
tolerance arithmetic, legacy scaling, callback order and bounded lifetime.
