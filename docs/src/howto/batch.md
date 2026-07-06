# Batch processing and result files

**Goal:** process a whole recording to a results file, survive crashes
mid-batch, and read everything back later.

## Build the pair list

[`image_pairs`](@ref) groups an ordered frame list (file paths and/or
in-memory matrices) into correlation pairs:

```julia
using Hammerhead

files = sort(readdir("run_042"; join = true))
pairs = image_pairs(files)                   # double-frame: (1,2), (3,4), ...
pairs = image_pairs(files; mode = :chained)  # time-resolved: (1,2), (2,3), ...
```

## Run the sequence with incremental output

```julia
passes = multipass_parameters([64, 32, 16]; padding = true, apodization = :gauss)
results = run_piv_sequence(pairs, passes;
    preprocess = img -> intensity_cap!(img),
    output = "run_042_piv.jld2",
    mask = mask,                  # optional, shared by all pairs
)
```

With `output` set, each result is written to the JLD2 file **as it
completes** — a crashed batch keeps its finished pairs. For file-path pairs
the source image paths are stored alongside the results. Threading applies
within each pair (the window grid is split across tasks); results are
bitwise identical to serial processing.

To halve memory traffic on large recordings, load frames in single
precision with `image_type = Float32` (see the
[precision policy](../explanation/precision.md)).

## Read results back

```julia
results = load_results("run_042_piv.jld2")   # Vector of results, in order
```

[`load_results`](@ref) returns [`PIVResult`](@ref) and/or
[`StereoPIVResult`](@ref) entries in sequence order. The stored source
paths, when present, are retrievable directly with JLD2:

```julia
using JLD2
sources = JLD2.load("run_042_piv.jld2", "sources/000001")
```

Standalone results (e.g. from interactive work) round-trip with
[`save_results`](@ref) / [`load_results`](@ref):

```julia
save_results("snapshot.jld2", result)        # single result or a vector
```

## Post-process the sequence

Sequences of same-grid results feed the statistics utilities directly:

```julia
validate_temporal!(results)                  # flag temporal outliers in place
stats = field_statistics(results)            # mean, RMS, Reynolds stress, counts
f, psd = power_spectrum([r.u[12, 8] for r in results]; dt = 1/10_000)
```

## A note on failure

If a pair fails (unreadable file, size mismatch), `run_piv_sequence` logs
which pair and rethrows — the incremental output file retains everything
processed up to that point, so you can fix the offending frame and resume
from a trimmed pair list.
