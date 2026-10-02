```@meta
CurrentModule = Hammerhead
```

# Experiment checkpoints

Checkpoint version 1 is a separate append-only protocol for file-based planar
PIV. It uses the complete core experiment recipe and ordered input identities,
with immutable native singleton results in an explicitly selected output
directory. Metadata and result directories must be disjoint and new/empty at
creation. Existing data are never deleted or replaced to initialize a store.

Supported recipes use built-in preprocessing, an original full-image static
mask/ROI, scalar `PhysicalScale`, CPU/KA, and Float32/Float64. External scripts
and custom preprocessors are refused: arbitrary callback state cannot safely
be skipped or reconstructed on restart. Software identity must match recipe
creation and all checkpoint attempts, including core source, resolved package
versions/hashes, Julia, platform, and Julia/FFTW thread counts. There is no
environment-change override that mixes checkpoint segments in version 1.

| Artifact | Meaning |
|:---------|:--------|
| `header.jld2` | Checkpoint format version, UUID, recipe/input identities, pair count, result locator, saved-experiment digest, environment and protected record paths. |
| `experiment.jld2` | An intact primitive version-1 experiment snapshot, checked against its header digest. |
| `attempts/*.begin.jld2` | Attempt UUID/ordinal, starting committed count, actual environment, input locators and start time. |
| `attempts/*.end.jld2` | Completed/cancelled/failed/interrupted status, authoritative committed count, optional finish time and failure summary. |
| `commits/pair-*.jld2` | One descriptor per absolute ordered pair index: checkpoint/recipe/input identities, pair identity, attempt UUID, native result basename/digest and commit time. |
| Output-directory `pair-*.jld2` | Immutable ordinary native singleton `PIVResult` files. |
| `*.partial` | Uncommitted staging; never automatically adopted or deleted. |
| `writer-lock/` | Exclusive writer ownership marker; other process PIDs are provenance, not proof of liveness. |
| `recovered-writer-lock-*/` | Preserved recognized old lock state after explicit interrupted-writer recovery. |

Each metadata artifact contains `checkpoint_format_version`, `checkpoint_kind`
and one primitive `checkpoint_data` mapping. Unknown versions/kinds and malformed
fields are refused. No functions or application-defined types are persisted in
metadata. Result payloads retain the existing native result format. Checkpoint
and experiment versions are independent.

## Commit and recovery rules

For each result, the writer creates a unique staging file in the result
directory, closes it, validates its native singleton/type/precision/pass/scale,
and computes its SHA-256. It publishes an immutable result basename, then
creates/closes/validates a commit descriptor and publishes that descriptor.
Both publications use same-directory native rename calls and surface errors;
they do not use copy fallback or force-replace an existing destination. An
exclusive writer and existing-destination checks protect generated names.
Concurrent external mutation/recovery is unsupported.

Only a final descriptor with a validated matching payload counts as committed.
A closed or published result without a descriptor remains an orphan and is
recomputed on resume. Staging/orphans remain available for inspection; there is
no automatic cleanup or adoption. Descriptors must form the exact prefix `1:K`.
Missing, duplicate, noncontiguous, corrupt or changed committed entries cause
refusal before mutation rather than silently trimming the prefix. Resume starts
at absolute index `K+1`, preserving earlier committed bytes and original order.

Inputs and all committed payloads are checked before an attempt starts. A
relocated `ExperimentRecord` is accepted only when recipe and ordered input
identities match exactly; paths remain locators. An explicitly relocated result
directory must pass full committed-payload verification and remain disjoint
from metadata. Original images need not be present merely to inspect a store.

`checkpoint_state` distinguishes `data_complete` from attempt status. An
unmatched begin is `:unfinished`, not an inferred failure or cancellation. A
process can stop after the final commit but before terminal metadata, leaving
all data committed with an unfinished attempt. Explicit recovery records
`:interrupted` with no guessed finish time; a subsequent attempt can finalize
complete data without recomputing it. A completed store can be resumed
idempotently without adding another attempt.

Handled failures record `:failed` where possible and rethrow the original
exception. Committed counts come from descriptor verification, including an
exception after publication but before an in-memory counter update. Secondary
failure recording/cleanup errors are logged. Cancellation is checked before
computation and between committed pairs, acknowledges after the pair in flight,
and records `:cancelled`. Cancellation after the final pair is completion.
Progress receives absolute `(committed,total)` counts after descriptor
publication. A progress/cancel callback exception is a failure.

Known live writers in the current process are refused. An unfinished attempt or
remaining lock requires `recover_interrupted=true`, an explicit caller assertion
that the former writer has stopped. Foreign PIDs are not used for automatic
stale-lock inference. Recognized empty/partial-owner/complete-owner lock states
are archived by rename without deletion; unknown files/symlinks are refused.
Do not recover while another writer is active.

These are local-filesystem publication/recovery rules, not a power-loss
durability guarantee. Closing files and rename publication do not establish
directory/data `fsync` guarantees here. Network filesystems and concurrent
writer mutation are unsupported. Process termination is tested separately from
handled failures and cancellation; no power-failure recovery is claimed.

## Results and native export

`checkpoint_results` verifies the current prefix and returns a fixed lazy
`CheckpointResults` index. It retains O(number of pairs) path/digest metadata
and loads one raw result per access, without a payload cache or open handle.
Verification reads all committed bytes and one payload at a time; it is not a
constant-time status poll. The index does not follow later commits. Numerical
processing also retains current/prefetched images and the workspace; this API
does not establish production-size host/GPU memory measurements.

`save_checkpoint_results` streams complete checkpoint data into a fresh ordinary
native file, one payload at a time. Source labels retain the actual input
locators for each owning attempt, including mixed original/relocated resumes.
It adds the checkpoint/recipe/input IDs. Existing destinations and paths inside either
owned directory are refused. Aggregate export is a derivative operation: a
failed/terminated export leaves authoritative checkpoint data untouched and can
be retried to another fresh path. An exclusive per-destination export lock
prevents concurrent library exporters from replacing each other's output.
After termination, a remaining export lock requires another fresh destination;
there is no automatic export-lock recovery. Concurrent outside writers are
unsupported. Generic `save_results` and quality-report writers also refuse
aliases of payloads retained by a checkpoint index or its views.
`ResultFile` and the GUI's lazy result explorer
can browse the completed aggregate normally. The first slice adds no GUI
checkpoint controls or live checkpoint reader.

```@index
Pages = ["checkpoints.md"]
```

```@autodocs
Modules = [Hammerhead]
Pages = ["experiment_checkpoint.jl"]
Private = false
```
