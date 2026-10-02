```@meta
CurrentModule = Hammerhead
```

# Representative-pair comparisons

`pair_comparison_format_version=1` is independent of experiment, native result,
checkpoint and run-quality formats. It records a controlled fresh rerun of
two built-in planar recipes on one verified ordered pair, not a comparison of
arbitrary saved fields or an accuracy validation. See
[Compare recipes on a representative pair](../howto/pair_comparison.md).

| Root field | Meaning |
|:--|:--|
| `generator`, `actual_environment` | Julia/core identity, common execution environment and its signature. |
| `provenance` | Verification scope, ordered selected-pair content identity, explicit environment override, selected indices, recipe/full-record IDs, creation environment IDs and selected file descriptors/locators. |
| `settings_changes` | Deterministic recipe-change paths and detached before/after values. |
| `basis` | Pixel displacement or strictly compatible physical velocity, conversion factor, original attached scales, unit and after-minus-before direction. |
| `native` | Separate complete-grid extents and stored-field quality populations for each recipe. |
| `common` | Exact original-pixel coordinate intersections, population partitions and paired velocity/UQ summaries. |
| `unavailable` | Accuracy, coverage, normalized uncertainty differences and absent measurement-history diagnostics. |
| `protected_locators` | Known input/record/result dependencies protected when saving. |

Null settings/provenance use explicit tagged mappings because TOML has no null.
Settings distinguish `kind="missing"` from `kind="nothing"`; symbols, strings,
integers, floats, booleans, tuples and named mappings preserve their kinds.
Embedded arrays are `array_summary` values containing shape, element type and
canonical content SHA-256. Recipe settings may include constructor-supported
infinity; available numerical metrics must always be finite. Nullable package
versions/tree hashes and Project/Manifest provenance use string/nothing tags.
The environment validator reconstructs their original signature for identity
checking. The report is not a complete replay recipe and does not embed arrays.

## Populations and moments

Native quality counts/fractions follow [`quality_report`](@ref) on each result
separately. Their denominators are each recipe's native grid. Common axes are
intersected by exact numerical equality; the Cartesian product defines `nodes`.
Finite, unique, strictly increasing axes and all field/flag dimensions are
validated before comparison. There is no tolerance, interpolation, extrapolation
or nearest-node fallback.

`masked_both`, `masked_before_only`, `masked_after_only`, and `unmasked_both`
partition common nodes. Within `unmasked_both`, `valid_both`,
`valid_before_only`, `valid_after_only`, and `invalid_both` partition the
current validity states. Validity requires both stored components finite and
the current outlier flag false. Finite filled outputs can remain flagged;
cleared alternative-peak flags cannot recover their earlier history.
The comparison reruns do not request the separate measurement-history
companion; their `not_persisted` reasons describe that missing observation.
Use [measurement history](measurement_history.md) when individual output origin
is needed. These version-1 comparisons do not aggregate companion events.

Velocity moments use precisely `valid_both`. For each component, with
`d = after - before`, `mean` is the population mean of d and `rms` is
`sqrt(mean(d^2))`. `vector_rms_difference = hypot(rms_u, rms_v)`.
Stable running means and scaled sums of squares avoid unnecessary overflow
from squaring large finite differences. These are output differences, not bias
or RMS accuracy errors.

`uq_counts_on_joint_valid` further partitions the jointly valid velocity nodes
into both/before-only/after-only/neither having both uncertainty components
finite and nonnegative. Stored uncertainty summaries use `available_both`
consistently for all six component summaries: `before_u/v`, `after_u/v`, and
`difference_u/v`. Before/after summaries are stored values; difference summaries
are after minus before. Each includes mean and RMS. Numerical availability is
independent of calibrated coverage or known error correlation. Native counters
retain whether the final uncertainty option was requested.

Metric groups contain `available` and their exact `count`. Available groups add
finite component summaries. Unavailable groups omit values and provide
`reason_code`: `no_common_nodes`, `no_joint_valid_nodes`, `no_joint_uq_nodes`,
or `nonfinite_arithmetic`. Arithmetic unavailability preserves the raw population
count, including finite raw values that overflow physical conversion or
subtraction; it does not silently select a smaller comparison population.

## Provenance, persistence and scope

Selected ordered pair identity includes SHA-256 bytes, byte sizes and decoded
dimensions; paths are locators. Whole-record IDs may differ because only the
selected pair is numerically compared. Record metadata and all recipe settings
are validated, while unrelated input files are not opened. Selected files are
verified before computation, before/after decoded loading and after comparison.
Strict creation-environment checks are default; an explicit override reruns
both recipes under the current environment. Environment/source identities are
checked again after computation. This is not a concurrent-mutation guarantee,
bitwise portability claim, or automatic representative-frame selection.

The comparison object retains a detached primitive snapshot and metadata
locators, with no result/image/recipe arrays or open handles. Native result
payloads and workspaces are transient during computation; retained correlation
planes can dominate that cost. Locator metadata may grow with recording/run
history. Returned data is copied, and mutation of a snapshot's internal data
is detected before access or saving.

Loading validates format, typed identities, population partitions, denominators,
moment availability, scale/unit compatibility and unavailable reasons. It does
not authenticate the report or reverify sources. Saving validates/serializes
before opening output, rejects normalized/filesystem same-file aliases of
known dependencies, and preserves these protections after loading. Additional
dependencies can be supplied through `protected_paths`. Ordinary report files
may be overwritten; writes have no atomic-publication/resume guarantee.

```@index
Pages = ["pair_comparison.md"]
```

```@autodocs
Modules = [Hammerhead]
Pages = ["pair_comparison.jl"]
Private = false
```
