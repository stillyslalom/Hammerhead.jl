# Evaluate particle tracking on VSJ301

Use this command to compare Hammerhead's detector, pair matcher and trajectory
linker with particle positions from an independently generated synthetic image
sequence. It processes the publisher's fixed frames **000–007**. The source
lists particle positions and persistent IDs; visibility is recorded as unknown.

The [VSJ homepage](https://www.vsj.jp/~pivstd/) describes download and analysis
and requests citation. The [301 page](https://www.vsj.jp/~pivstd/image3d/image301.html)
and [particle example](https://www.vsj.jp/~pivstd/image3d/ptc.html) explain the
world positions, projected X/Y and peak intensities. Cite Okamoto, Nishio,
Kobayashi, Saga and Takehara (2000), *Evaluation of the 3D-PIV standard images
(PIV-STD project)*, [DOI:10.1007/BF03182404](https://link.springer.com/article/10.1007/BF03182404).
The article publisher lists pages 115–123; the dataset site lists 115–124.
Keep downloaded images, annotations and derived association ledgers in the
private, ignored cache. Redistribution requires separate permission.

## Acquire and pin a private cache

The acquisition tool uses Python 3.11+ and its standard library:

```bash
python bench/prepare_vsj301.py --cache=bench/profile-output/vsj301-cache
python bench/prepare_vsj301.py --offline --cache=bench/profile-output/vsj301-cache
```

The first command downloads
[301_raw.zip](https://www.vsj.jp/~pivstd/image3d/301_raw.zip) (6,292,551 bytes) and
[301_ptc.zip](https://www.vsj.jp/~pivstd/image3d/301_ptc.zip) (11,100,328 bytes),
plus bounded copies of the usage and format pages. The second audits the cache
offline. Choose a fresh directory for acquisition; failed downloads leave their
partial files available for inspection.

The audit checks response sizes, permission text, archive structure and member
identities. Archives must contain exactly the 145 numbered image/particle members.
The tool bounds decompressed sizes and rejects duplicate names, links, encryption,
unsafe paths and unsupported compression. It extracts frames 000–007.

`cache.toml` (`hammerhead-vsj301-cache-1`) records URLs, response metadata,
archive SHA256, selected-member CRC/SHA256, attribution, acquisition UTC,
Python version and acquisition-script identity. These hashes pin the bytes
from your first download. Review and retain that manifest: subsequent runs
verify files against it. Each Julia cache load also invokes the offline Python
ZIP audit, then checks the pinned files and processing pixels.

## Process once, compare three coordinate conventions

```bash
julia --project=. --threads=1 bench/validation_vsj301.jl --cache=bench/profile-output/vsj301-cache --output=bench/profile-output/validation-vsj301
```

Choose a fresh output directory. Use `--python=/local/path/to/python` to select
an interpreter; the default searches `python`, then `python3`.

The RAW frames are 256×256 UInt8 arrays, with x varying fastest and the top row
first. Processing converts them to Float64 and divides by 255. The study uses
complete `PTVParameters()` settings, production `:piv` prediction with
`multipass_parameters([64,32])`, minimum retained track length 2 and max-gap
policies 0/1/2. Predictions come from the images and previously linked motion.
Annotations enter the scoring stage.

The run makes 8 standalone detector calls, 7 pair calls and 3 tracking calls:
46 detector invocations, 28 matcher transitions and at most 10 full PIV predictors.
Later tracking predictors use previous links and constant-velocity heads.
Source nominal spacing is 0.005 s; exports retain ordinal frame indices.
Internal PIV follows its `Threads.nthreads() > 1` default, which records `false`
for the single-thread command above.

The source defines X right and Y down. Its pixel-center origin needs interpretation,
so scoring compares three offsets on the same production detections and tracks:

| Added to both source X/Y | Interpretation under test |
|---:|---|
| 0 | Source coordinates used directly as one-based processing coordinates. |
| +0.5 | Source origin at the upper/left pixel-area boundary; first center at 0.5. |
| +1 | Source coordinates are zero-based pixel-center indices; first center at 0. |

Read the three columns together to assess coordinate sensitivity. Source
centroids have 0.01 px precision and world positions 0.001 cm precision.
Scoring retains all finite listed contributors, including faint particles and
boundary positions. Peak intensity retains the source's intensity units.

## Read the scoring populations

The sparse schema (`hammerhead-sparse-particle-clip-2`) gives every listed ID
a row in every selected frame. Available rows retain all seven source fields;
missing positions and intensities are marked `"unknown"`. Visibility is
`"unknown"` throughout. This makes annotation gaps visible in each denominator.

Localization associates detections with listed positions inside a 0.75 px gate.
An independent Hungarian assignment maximizes the number of associations, then
minimizes their total squared distance. It solves disconnected bipartite
components and retains isolated detections and annotations. Resource limits
are 10,000 annotation nodes, 20,000 prediction nodes, 200,000 edges, 512 vertices
per component and 1,000,000 augmented matrix cells per component. Evaluation
stops if a limit is exceeded. Returned trajectory coordinates must identify
exactly one cached detection row.

Each metric describes this gated association under one coordinate hypothesis:

- **Detection:** compare localized listed positions with both the number of
  predictions and the number of listed positions. Unassigned predictions and
  unlocalized annotations remain separate populations with unknown visibility.
- **Correspondence:** every raw pair, accepted pair and returned track edge is
  classified as same listed ID, different listed IDs, ambiguous or unannotated.
  The operational precision interval is `same/N` to
  `(same+ambiguous+unscorable)/N`, using all predictions in `N`.
- **Adjacent recall:** use IDs listed in both neighboring frames. Track recall
  credits exact returned edges between these adjacent listed endpoints.
- **Identity continuity:** inspect same-track listed-ID changes, extra output
  tracks per listed ID, and output-ID changes within annotated runs separately.
  Unknown or ambiguous samples start a new continuity interval.
- **Reappearance:** inspect endpoint relinking counts, localized endpoints and
  ambiguity together. The intervening frames have unknown annotation status.

Merged particle images and unlisted contributors make some associations
ambiguous. Read those counts alongside the matched-ID results. Track filtering
also matters: detector observations missing from retained tracks combine linking
and minimum-length losses. Displacement-error summaries describe the correctly
associated endpoint subset.

## Inspect and protect the artifacts

The fixed eight-frame Windows study retained 5,055 listed identities across
33,250 provided positions and 7,190 unknown positions. Each origin hypothesis
scored the same 12,123 detections and 9,415 accepted pair correspondences:

| Source-coordinate offset | Localized listed positions | Accepted same-listed-ID associations |
|---:|---:|---:|
| 0 px | 1,993 | 857 |
| +0.5 px | 12,097 | 8,262 |
| +1 px | 1,954 | 872 |

The +0.5 px interpretation associates substantially more listed positions.
This is a useful coordinate-sensitivity check to resolve before interpreting
matching performance. The adjacent-listed-ID denominator is 28,193. Across
all nine origin/gap-policy combinations, neither of the two annotated
reappearance events had an exact returned endpoint link.

Start with `vsj301.md` for localization, pair-error, identity-continuity and
endpoint-relinking tables. Use `vsj301.toml` to inspect the complete recipe,
environment, three coordinate hypotheses, hashes and population counts.

The ordinary `native-ptv-pair-*.csv` and `native-tracks-gap-*.csv` files use
[`export_table`](@ref). Trajectory IDs identify returned tracks; the separate
association tables connect their observations to source particle IDs.
Scoring CSVs include every detection, returned observation and edge, identity
summary and endpoint event. `sparse-annotation-ledger.csv` includes unknown rows.

For a repeatable comparison, finish source edits, start a fresh Julia process,
and write to a new output directory. The command checks consumed inputs and
source/environment identities before publishing its report. If a write fails,
inspect the partial files and retry in another directory.
