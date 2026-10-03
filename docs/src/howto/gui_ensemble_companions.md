# Inspect ensemble pooling and report a native file

Open a completed native results file lazily:

```julia
using HammerheadGUI
explorer = ResultExplorer("ensemble-results.jld2"; lazy=true)
figure = result_explorer(explorer)
```

Enable **recorded processing details**. A recorded ensemble packet is checked
against the selected raw measurement fields and geometry before conversion to
display units. The inspector also checks a separate digest of the physical
display fields; editing those arrays requires disabling and reenabling details
to reload the source. Binding does not verify source images or scientific accuracy.

The paged drawer distinguishes ensemble pooling from ordinary pair iterations.
Each ensemble pass performs one pooled sweep. Requested iteration counts and
tolerance are ignored. Window/pair opportunities partition into masked,
source-gated and accumulated contributions. Accumulated correlation planes have
separate zero, flat nonzero, nonflat and nonfinite counts. Contributor summaries
describe eligible nodes with zero, some or all finite nonzero planes, not
independent sample size or valid vectors. Primary residuals remain processing
pixels before predictor addition, even when the plot shows speed in physical
units. Pooled uncertainty counts precede validation and availability cleanup;
they do not establish applicability or coverage for a displayed vector.

A selected node shows ordinary displayed values and an explicit explanation
that per-node ensemble contributor counts, replacement history and uncertainty
attribution are unavailable. A bare planar result with no packet is not labelled
an ensemble. Mixed native files can contain ordinary pairs, pools and other
result kinds; incompatible companion families on the same entry are refused.
Failed navigation or enabling details retains the previous frame, selection,
mode and packet. Recover by choosing a readable entry or disabling details.

The explorer retains one display payload and current packets, including an
array-free ensemble packet. It releases prior frame payloads on navigation;
caller-retained values still cost memory. Integrity checks hash current display
arrays, so refresh cost is proportional to the current measurement size.

Click **whole-file quality report** to open a separate window. All options start
unchecked. **Recorded ensemble pooling observations** opts into report format 4;
ordinary execution and final-sweep history can be requested independently. A file
containing ensemble metadata needs the ensemble option when requesting ordinary
execution observations; legacy format 3 refuses those pools. Click
**generate report** or **generate / save TOML**. The report scans the entire raw
native file, including frames not visited in the plot. Selected frame, display
conversion and the details toggle do not change this population.

```julia
report = explorer_quality_report(explorer;
    include_ensemble_execution_diagnostics=true)
save_explorer_quality_report("quality.toml", explorer;
    include_execution_diagnostics=true,
    include_ensemble_execution_diagnostics=true)
```

Format 4 classifies planar entries into recorded ensembles, recorded ordinary
iteration entries and entries without execution metadata. Missing metadata does
not mean a missing ensemble packet. Existing current-field metrics retain their
node-weighted denominators; ensemble counters use their explicitly labelled
window/pair and final-node populations. Default generation remains format 1.

The window checks the complete sorted native key mapping and detaches its index
on opening and before each request. Omitted, duplicated or reordered index keys
are refused; reopen the explorer rather than editing its key list. This adds
O(number of entries) key metadata, without retaining those result payloads.
The window captures options before
observers or a save dialog run. A failed scan, save or cancelled dialog retains
the previous report with its own source path and SHA256. Text pages contain the
full status, identities and summary. Changing the explorer frame does not relabel
that report. These explorer reports are unassociated with a saved recipe or run.
Use the [saved ensemble workflow](gui_ensemble_experiments.md) for a report tied
to a verified ensemble run. Generation verifies the unchanged
whole file before and after scanning. Loading a saved report does not reverify
the source.

Scanning and hashing run synchronously and can pause rendering. Eager/bare,
timed-tracking and checkpoint explorers cannot generate this native-file report.
Known source/dependency aliases are protected before opening the output, but
filesystem failure may leave partial TOML; publication is not atomic. There is
no live-writer, checkpoint, stationarity, convergence, accuracy or uncertainty
coverage claim.
