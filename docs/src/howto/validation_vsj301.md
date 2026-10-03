# Evaluate the independently generated VSJ301 particles

This bench command evaluates the publisher's fixed frames **000-007** using
Hammerhead's production detector, pair matcher and trajectory linker. VSJ301 is
an independent **synthetic** source. It provides neither real-recording evidence
nor complete visibility annotations.

The [VSJ homepage](https://www.vsj.jp/~pivstd/) permits download and analysis and
requests citation. The [301 page](https://www.vsj.jp/~pivstd/image3d/image301.html)
and [particle example](https://www.vsj.jp/~pivstd/image3d/ptc.html) describe persistent
particle IDs, world positions, projected X/Y and peak intensity. Cite Okamoto,
Nishio, Kobayashi, Saga and Takehara (2000), *Evaluation of the 3D-PIV standard
images (PIV-STD project)*, [DOI:10.1007/BF03182404](https://link.springer.com/article/10.1007/BF03182404).
The article publisher lists pages 115-123; the dataset site lists 115-124.
No explicit redistribution license was found. Keep the cache, source images and
raw annotation files private and ignored; do not add them to Git. Generated
annotation/association ledgers also contain source annotation data and should
remain private until redistribution permission is established.

## Acquire and pin a private cache

Python 3.11+ is required only by this bench workflow. Its acquisition/audit tool
uses the Python standard library; no production dependencies are added.

```bash
python bench/prepare_vsj301.py --cache=bench/profile-output/vsj301-cache
python bench/prepare_vsj301.py --offline --cache=bench/profile-output/vsj301-cache
```

Acquisition uses the fixed primary HTTPS endpoints
[301_raw.zip](https://www.vsj.jp/~pivstd/image3d/301_raw.zip) (6,292,551 bytes) and
[301_ptc.zip](https://www.vsj.jp/~pivstd/image3d/301_ptc.zip) (11,100,328 bytes).
It also saves bounded copies of the usage, dataset and particle-format pages.
Unknown response sizes, changed permission text, malformed archives and unexpected
entries refuse completion. Existing cache directories are never replaced.
Failed acquisitions leave partial data for inspection; retry in a fresh directory.

`cache.toml`, version `hammerhead-vsj301-cache-1`, records URLs, response metadata,
full archive SHA256, selected-member CRC/SHA256, attribution, acquisition UTC,
Python version and acquisition-script identity. No publisher checksum was found:
these first-use hashes identify the downloaded bytes, **not** authenticated
publisher content. Retain and review the completed cache manifest; later offline
audits check these pinned identities instead of silently downloading replacements.

The ZIP audit requires exactly the 145 numbered image/particle members, rejects
duplicate names, links, encryption, unsafe paths and unsupported compression,
and bounds decompressed sizes. Only source frames 000-007 are extracted. A fresh
offline Python audit verifies the selected bytes against their ZIP members on
every Julia cache load; Julia then verifies pinned files and processing pixels.
Julia itself does not implement ZIP decompression or claim independent ZIP checks.
Coordinated malicious editing of both data and its hash descriptor is not authenticated.

## Process once, score three fixed coordinate hypotheses

```bash
julia --project=. --threads=1 bench/validation_vsj301.jl --cache=bench/profile-output/vsj301-cache --output=bench/profile-output/validation-vsj301
```

Use a **fresh** output directory. To select a particular local Python interpreter,
add `--python=/local/path/to/python`. The default searches `python`, then `python3`.

RAW frames contain256×256 UInt8 pixels, x fastest, top row first. Processing uses
Float64 values divided by 255, with no contrast normalization, cropping or
preprocessing. The first source RAW was independently compared with correctly
palette-decoded, bottom-up BMP content during research; it was identical.
The study uses RAW directly and does not depend on a BMP decoder.

The default uses complete `PTVParameters()` settings, production `:piv` prediction
with `multipass_parameters([64,32])`, minimum retained track length 2 and max-gap
policies 0/1/2. No annotations enter predictors. It makes 8 standalone detector calls,
7 pair calls and 3 tracking calls: 46 detector invocations, 28 matcher transitions and
at most 10 full PIV predictors. Later tracking predictors use production previous
links/constant-velocity heads. Source nominal spacing is 0.005 s; processing/export
retains ordinal frame indices without invented absolute timestamps or physical scales.
Internal PIV follows its `Threads.nthreads() > 1` threading default; the specified
single-thread command records `false`, without overriding production settings.

The primary pages establish X right/Y down, but not pixel-center origin. All three
predeclared hypotheses score **the same** production objects:

| Added to both source X/Y | Interpretation under test |
|---:|---|
| 0 | No shift: source coordinates interpreted directly in one-based processing coordinates. |
| +0.5 | Source origin at the upper/left pixel-area boundary; first center at 0.5. |
| +1 | Source coordinates are zero-based pixel-center indices; first center at 0. |

No hypothesis is selected as a winner or called verified registration. Source
centroids are reported to 0.01 px, world coordinates to 0.001 cm. Preserve all finite
listed contributors, including faint particles and boundary positions; do not
discard them because one hypothesis places their centers outside the pixel-center
domain. Particle peak intensity remains in source units, distinct from normalized
processing-pixel intensity and actual detectability.

## Sparse annotation and conditional scoring

The sparse bench schema is `hammerhead-sparse-particle-clip-2`. Every union ID has
a row for every selected frame. Listed rows retain all seven source fields;
missing rows contain literal `"unknown"` positions/intensity. Visibility is
`"unknown"` for **all** rows. No hidden position, boundary exit, laser-sheet loss
or true absence is invented. The complete-annotation v1 importer/scorer remains
unchanged.

The independent localization association maximizes cardinality, then minimizes
sum squared distance within the fixed 0.75 px gate. It solves disconnected
bipartite components with the existing independent Hungarian objective, retaining
every isolated prediction and annotation. It does not use the production greedy
matcher or its cell list. Deterministic tiny tests enumerate all partial injections.
Limits are 10,000 annotation nodes, 20,000 prediction nodes, 200,000 edges,
512 vertices per connected component and 1,000,000 augmented matrix cells per
component. Exceeding a limit refuses evaluation, without truncating populations.
Returned trajectory coordinates must identify exactly one cached detection row;
duplicate refined coordinates are refused rather than inventing a producer index.

All metrics are **conditional on the operational gated annotation association and
coordinate hypothesis**. A unique association does not prove which particle
generated merged intensity, especially with unlisted faint contributors.

- Detection reports localized listed rows / all predictions / all listed rows.
  Unassigned predictions are not proven false positives; unlocalized listed rows
  are not proven misses of visibly detectable particles.
- Every raw/accepted pair and returned track edge enters associated-same-listed-ID,
  associated-different-listed-IDs, ambiguous or unscorable-unannotated accounting.
  Operational precision bounds retain the all-prediction denominator: same/N to
  (same+ambiguous+unscorable)/N. They are not unconditional truth guarantees or
  confidence intervals.
- Pair recall uses IDs listed in both **adjacent** frames. Track adjacent-edge
  recall credits only exact returned edges for these adjacent listed endpoints.
  No recall crosses an unknown annotation interval.
- Same-track listed-ID changes, extra output track IDs per listed identity and
  annotated-run output-ID changes/restart episodes are separate. Unknown and
  ambiguous samples reset transition memory; their sample counts remain visible.
  Extra track IDs across unknown intervals can reflect legitimate interruption.
- Reappearing IDs receive an **endpoint relinking diagnostic**, including full
  event counts, localized endpoints and ambiguity. Unknown middle frames prevent
  interpreting this fraction as true-absence gap-recovery recall.

Returned-track filtering hides singleton/internal candidate histories. Detector
observations absent from returned tracks therefore combine linking and retention
loss. Correctly associated endpoint displacement errors are descriptive subset
errors; they do not replace precision, annotation recall or unknown populations.

## Inspect and protect the artifacts

The fixed eight-frame Windows study retained 5,055 listed identities across
33,250 provided positions and 7,190 unknown positions. The same 12,123 detected
positions and 9,415 accepted pair correspondences were scored under each origin
hypothesis:

| Source-coordinate offset | Localized listed positions | Accepted same-listed-ID associations |
|---:|---:|---:|
| 0 px | 1,993 | 857 |
| +0.5 px | 12,097 | 8,262 |
| +1 px | 1,954 | 872 |

These contrasts show sensitivity to the coordinate convention; they do not
verify an origin or identify a physical contributor. The full adjacent-listed-ID
denominator is 28,193. All nine hypothesis/gap-policy combinations found zero
exact returned links across the two annotated reappearance events; unknown
middle samples prevent a true-absence recovery conclusion. Independent audits
checked source identities, native/scoring CSV populations and conditional error
moments. The private artifacts retain the complete recipe and environment;
this observation is not a universal acceptance threshold.

`vsj301.toml` stores the complete recipe, all three hypotheses, source/environment
identities, cache/processing hashes, association resource counts and population
denominators. `vsj301.md` displays localization, pair errors and classification,
tracking identity/fragmentation and endpoint-relinking contrasts side by side.

Ordinary `native-ptv-pair-*.csv` and `native-tracks-gap-*.csv` use the existing
`export_table` schema. Output trajectory IDs are not source particle IDs. Separate
scoring CSVs retain every detection, returned observation/edge, identity summary
and endpoint event. `sparse-annotation-ledger.csv` includes unknown rows explicitly.

Output uses exclusive fresh-directory acquisition, protects consumed cache paths
and preserves existing files on preflight refusal. Concurrent library creators
cannot both acquire the same directory. Partial write failures remain inspectable;
outside concurrent writers and power-loss durability are unsupported. Changed
inputs or on-disk source/environment identities refuse publication. Run final
evidence in a fresh process after source freeze; this study makes no timing claim.
