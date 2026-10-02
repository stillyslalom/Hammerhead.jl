```@meta
CurrentModule = Hammerhead
```

# Interrupt and resume a planar experiment

**Goal:** process a file-based planar recording in committed pairs, stop at a
pair boundary, and resume without recomputing or replacing committed results.
Start with a complete [experiment recipe](experiments.md). Checkpoints support
built-in preprocessing and a matching software environment; custom script
state and mixed-environment restart are outside version 1.

Select separate new or empty directories for checkpoint metadata and per-pair
native output. This executable example uses the same committed fixture pair
twice to make the interruption/resume visible without a large recording:

```@example checkpoints
using Hammerhead

directory = joinpath(pkgdir(Hammerhead), "test", "reference_images", "A")
files = sort(filter(path -> endswith(lowercase(path), ".tif"),
                    readdir(directory; join=true)))
pair = (files[1], files[2])
recipe = PIVRecipe(multipass_parameters([32, 16]; padding=true);
                   roi=ROI(1:64, 1:64), image_type=Float32)
experiment = ExperimentRecord([pair, pair], recipe)

mktempdir() do work
    checkpoint = create_checkpoint(joinpath(work, "checkpoint"), experiment;
                                    output_dir=joinpath(work, "pair-results"))
    stop = Ref(false)
    first_attempt = resume_checkpoint!(checkpoint;
        progress=(committed, total) -> (stop[] = committed == 1),
        cancel=() -> stop[])
    prefix = checkpoint_results(checkpoint)

    reopened = load_checkpoint(checkpoint.path)
    resumed = resume_checkpoint!(reopened)
    output = save_checkpoint_results(joinpath(work, "vectors.jld2"), reopened)
    (first_status=first_attempt.status, prefix_pairs=length(prefix),
     resume_started_after=resumed.start_committed,
     final_pairs=length(load_results(output; lazy=true)))
end
```

Cancellation acknowledges after the pair in flight has committed. Progress
reports absolute counts after each published descriptor. Callback errors become
failed attempts; completed entries remain available. `checkpoint_results`
returns a fixed lazy prefix index, so an index opened before resume keeps its
original length. Use `checkpoint_state` or reopen the index to inspect later
progress. Checking a store verifies committed bytes and is more expensive than
reading a counter.

## Recover after process termination

Handled failures and cancellation close the writer and release its lock.
After a terminated process, a lock or unmatched attempt begin can remain.
Stop the old writer before explicitly asserting recovery:

```julia
checkpoint = load_checkpoint("checkpoint")
state = checkpoint_state(checkpoint)
attempt = resume_checkpoint!(checkpoint; recover_interrupted=true)
```

The flag is your assertion that the former writer has stopped. The library
refuses known live writers in this process, but does not automatically establish
whether another PID is alive. Concurrent recovery is unsupported. Recognized
old lock states are archived, including a crash during lock-directory or owner
publication. Unknown files are refused and no existing data are deleted.

Recovery verifies the entire committed prefix and resumes at the next absolute
pair. Staging files and payloads without final descriptors are ignored rather
than guessed complete. Corrupt or missing committed results cause refusal;
the reader does not silently discard recorded work. If termination follows the
last commit but precedes terminal metadata, data can be complete while attempt
status is unfinished. Explicit recovery records interruption without inventing
a finish time, then finalizes the data without recomputing it.

Keep input and result files unchanged during processing/recovery. Relocation
requires an explicitly supplied equivalent `ExperimentRecord` or output
directory and matching content identities. Changed recipe settings, ordered
inputs or software are refused before another attempt writes data. These rules
cover tested local-file publication and process termination; they do not promise
power-loss durability or network-filesystem behavior.

## Browse or export completed results

Each committed output is an ordinary native singleton file. The lazy core
checkpoint index loads one raw result at a time. For the existing GUI, export
complete data with `save_checkpoint_results` and open the aggregate through
`ResultExplorer(output; lazy=true)`.

Choose a fresh aggregate path outside both owned directories. Export streams
one result at a time and cannot replace an input, record, committed result or
existing destination. An exclusive destination lock protects concurrent library
exports. A terminated export can leave that lock; select another fresh path
rather than deleting unfamiliar lock contents. Outside writers must not mutate
the destination concurrently. Actual source locators from each attempt are
retained when earlier and resumed pairs use different input directories.
A failed export leaves the checkpoint intact. Per-pair
data and an aggregate use additional disk space; export is optional and does
not become the authoritative restart store. See the
[checkpoint reference](../reference/checkpoints.md) for schema, commit semantics,
attempt statuses and scope limits.
