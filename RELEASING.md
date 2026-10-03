# Releasing Hammerhead and HammerheadGUI

The core and GUI are separate registered packages in one repository. Release
the core first when the GUI uses new core APIs, then require that core version
in the GUI compatibility bounds before registering the GUI. A local path-based
development environment can hide a missing dependency lower bound.

## Prepare the candidate

Update [CHANGELOG.md](CHANGELOG.md) with user-visible API, numerical-behavior,
and file-format changes. Assign the candidate package versions only when
preparing an actual release. Apply the
[compatibility policy](docs/src/explanation/compatibility.md): describe any
migration needed for native records, result constructors, and table readers;
bump the relevant format/schema version if existing meanings become incompatible.
Keep the root and GUI package versions independent.

Record the candidate commit, Julia and package versions, operating system,
thread count, and the exact commands used. First couple both development
environments to the candidate source, as CI does. From the repository root,
start `julia --project=HammerheadGUI` and run:

```julia
using Pkg
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
Pkg.develop(PackageSpec(path = pwd()))
Pkg.instantiate()
```

Exit that session, start `julia --project=docs` from the same root, and run:

```julia
using Pkg
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
Pkg.develop([PackageSpec(path = pwd()),
             PackageSpec(path = joinpath(pwd(), "HammerheadGUI"))])
Pkg.instantiate()
```

This explicit setup also works on Julia 1.10, which does not use the GUI's
`[sources]` override. The docs environment otherwise may resolve registered
packages instead of the candidate. Precompilation is deferred until a GL
context is available. Then run from the candidate checkout:

```sh
julia --project=. -t 4 -e 'using Pkg; Pkg.test()'
julia --project=HammerheadGUI -e 'using Pkg; Pkg.test()'
julia --project=docs docs/make.jl
julia --project=. -t 4 bench/validation_scorecard.jl
julia --project=. -t 4 bench/validation_uncertainty.jl
```

The GUI and docs require a GL context. Linux CI supplies one through Xvfb;
see the actual setup in [.github/workflows/CI.yml](.github/workflows/CI.yml).
The docs build executes the seven tutorials and checks public docstrings.
Expected failure-propagation logs in tests and a local skipped-deployment
warning are not failed assertions. Record other warnings and their implications.

Check the CI matrix at the candidate commit rather than assuming a local pass
covers every supported configuration. It currently exercises single-threaded
core tests on LTS, stable, and prerelease Julia; multithreaded stable Julia on
Linux and Windows; and GUI tests on LTS/stable Linux. That matrix does not
establish macOS GUI support or device-backend correctness.

## Record measurement and workflow evidence

Keep the evidence attached to the candidate release or its review record, with
links to logs and artifacts. Record what was not exercised as well as what passed.

| Evidence | Record |
|---|---|
| Synthetic accuracy | Seeds, generator/settings, reference convention, bias/RMS error, validity and uncertainty checks; identify tests changed with numerical behavior. |
| Committed real data | Challenge A and 4E fixture identities and test/tutorial outcomes. These are real-data smoke checks, not ground-truth accuracy measurements. |
| Larger or external data | Dataset identity/checksum, processing recipe, reference provenance, selection rules, and applicable metrics. If unavailable, state that limitation. |
| CPU performance | Named hardware, thread/FFT configuration, precision, warmup/repeats, representative image size and sequence length; link the [benchmark procedure](bench/README.md). |
| CUDA/AMDGPU | Device, driver/runtime/package versions, backend validation command, supported paths, accuracy comparison and memory/performance measurements. KA CPU tests do not replace hardware execution. |
| GUI workflow | OS/display configuration, setup/preprocessing/ROI/mask/scale, batch completion/cancellation, saved-result reopen, and export checks. Include a lab-user walkthrough when available. |
| Persistence | Current round trips, unknown-version rejection, promised historical fixtures, cancellation/failure behavior, and any documented recovery limits. |
| Planar experiment replay | Recipe/input hashes, creation and run environments, native-result parity, changed-input/unknown-version refusal, and completed/failed run records. Record any explicit environment override. |
| Ordinary GUI replay | Captured requests, ordered completed-pair progress, cancellation before start/at pair boundaries, final-pair completion, observer failures and loader cleanup. Distinguish a native prefix from resumable checkpoints and offscreen checks from desktop responsiveness. |
| Spectrum timing | Legacy interval parity, exact sample-time validation, interval/global drift bounds, numerical range failures and result grid/shape/scale compatibility. Record any explicit timing tolerance and invalid-sample filling. |
| Artifact portability | Foreign locator preservation, explicit local relocation, consumed-file protection and prospective aliases. Lexical foreign-path fixtures on one OS do not constitute runtime validation on other operating systems. |
| Checkpoint recovery | Filesystem/OS, exact identities, handled failure, cancellation, hard process termination at publication boundaries, unchanged committed prefix, and explicit interrupted-writer recovery. Same-directory rename evidence is not power-loss durability. |
| Quality reports | Counter/denominator correctness, unavailable diagnostics, verified output association, protected destinations, and shared script/GUI report behavior. |
| Associated stereo reports | Completed-run association, frozen geometry, ordered sources and raw fields before/after streaming, optional input-byte checks, explicit relocation, protected inputs/history/results and unchanged v1/v3 schemas. Refuse failed runs and unsupported stereo history. |
| Saved stereo GUI workflow | Lossless batch/imported recipe handling, captured requests and selected historical run IDs, cancellation through cleanup, verified lazy physical display, prior report identity after failed actions, protected destinations, compact layout and actual hidden-control mouse routing. |
| Ensemble execution diagnostics | CPU/KA Float32/64 numerical parity, one pooled sweep and ignored iteration settings, contribution partitions, source-gated versus admitted numerical planes, pre-addition residuals, pooled UQ support, native measurement binding and callback/alias guards. Do not infer independent samples or stationarity; vendor capture and execution-report aggregation remain unsupported. |
| History-aware reports/GUI | Default-v1 compatibility, opt-in-v2 recorded/missing/unsupported populations, event/origin counts, packet/run association, raw-to-physical verification, failed-navigation state and bounded payload retention. |
| Execution diagnostics | Diagnostics-on/off numerical parity, actual stopping/check semantics, primary-residual population, callback failure cleanup, optional companion schema and replay association. |
| Stereo execution diagnostics | Per-camera pixel basis and common-grid geometry, distinct execution/camera identities, measurement-field binding and its exclusions, callback mutation refusal, optional native companions and bounded noncollecting lifetime. Metadata-only checks do not verify reconstructed fields or calibration. |
| Measurement history | Numerical parity, actual alternative/fill/restoration events, first-observed rejection semantics, callback mutation guards, result-content binding and bounded sequence lifetime. Keep final-sweep evidence distinct from a complete processing trace. |
| Pair timing | Exact epoch/delta/midpoint preservation, delay/unit/clock semantics, all-pair rejection before loading/output, frozen selections and metadata, optional native binding, callback-only mutation guards and unsupported-driver refusal. |
| Stereo pair timing | Ordered camera metadata, separate midpoints and synchronization observations, camera-1 versus supplied scaling delays, frozen selections, raw stereo/camera binding, native corruption refusal and callback/loader lifetime. No calibration or hardware synchronization certification. |
| Frozen-camera stereo replay | Exact fitted-camera round trips, rebuilt-map identity, ordered input bytes/timing and scale modes, CPU/KA direct parity, complete preflight, companion association, verified raw fields/source order and failed-prefix behavior. Keep supplied calibration provenance distinct from calibration fitting or self-calibration execution. |
| Derivative support | Independent legacy-quotient parity, mask/boundary stencils, descending/irregular axes, input eligibility versus structural support versus finite outputs, native overflow/subnormal behavior and physical units. |
| GUI derivative inspection | Consistent stencil policy across fields and area circulation, discrete legends including excluded nodes, selected contributor coordinates and units, current flags versus recorded history, mutation refusal, transactional navigation and bounded current-frame retention. Inspect the paged drawer and policy controls. |
| Compact saved workflow | Bounds and rendered text at 1100×800 and 900×600, reachable full paths/errors/identities, persistent cancellation/progress, retained options and actual mouse routing after section changes. |
| Actual-time tracking | Irregular predictions, fresh/gapped tracks and duration-normalized validation, exact source metadata, secant time support, spatial-scale-only velocities, dedicated artifact rejection by generic readers and timed CSV round trips. |
| Timed tracking GUI | Dedicated artifact loading, native-vector compatibility, timing-preserving physical display, observation-mean speeds, unavailable tracks, gap/selection semantics and inspected layouts. |
| Calibrated scattered tables | Affine position/vector bases, timing and unit assumptions, unavailable transformed scalar diagnostics, empty metadata, CSV/metadata verification and failure before destination replacement. |
| GUI recipe comparison | Explicit pair choices, captured full recipes, core provenance/value-basis checks, previous-report identity after failure, protected saves and inspected paged layouts. |
| Synthetic uncertainty | Seed/input/source identities, controlled primary-only output recipe, component populations, zero/nonfinite uncertainty handling, coverage and signed normalized errors. Synthetic observations do not establish experimental coverage or Gaussian calibration. |
| Conditional noise diagnostics | Fixed clean-image identities, independent unclipped noise streams, original full-error metrics, complete-case losses, disjoint-pair populations and independent covariance-weighting algebra. Keep this evidence separate from experimental validation. |
| Rendering/interpolation diagnostics | Unchanged baseline hashes/scientific rows, fixed particle placements and shifts, exact support/sampling policies, numerical integration checks, full-error/common-primary populations and separate known-shift image-warp comparisons. Deterministic sensitivity does not calibrate random uncertainty. |
| Spatial response | Prescribed particle displacement and analytic midpoint truth, matched output spacing, complete schedule/input identities, guarded harmonic fits, exact common-primary selections independent of sigma, and original full-error/yield/UQ populations. A finite-amplitude schedule comparison is not a universal resolution limit. |
| Annotated PTV/tracking study | Independent localization-assignment oracle, complete target/nuisance/visibility ledger, conservative identity ambiguity, conserved predictions and full/conditional recall, explicit identity/fragmentation/gap fixtures, protected manifests and separate truth/output IDs. Keep controlled synthetic evidence distinct from external or real annotated recordings. |
| Independent VSJ301 study | Publisher/archive/member identities and usage provenance, RAW orientation/scaling, complete provided/unknown annotation ledger, resource-bounded assignment oracle, identical processing outputs across fixed registration hypotheses, conditional association scores and audited native/scoring tables. Unknown middle samples cannot establish genuine invisibility or true gap-recovery recall. |
| Execution-aware reports and GUI | Unchanged default schemas, entry coverage versus pass/support counts, raw stereo verification before physical conversion, strict persisted counters, source protection and report-generation-time wording. Inspect missing/wrong-kind companions, transactional navigation, current display integrity and captured report options. |
| Qt saved-experiment prototype | Complete saved-recipe capture, verified lazy inspection with truthful physical units and retained-display identity, cancellation/cleanup, disposable viewport/shell subscriptions and a clean software-child exit. Keep native GL, desktop input and cross-platform adoption gates separate. |
| Recipe comparisons | Selected input identities, exact common-grid populations, scale/unit compatibility, arithmetic availability, stored UQ semantics, detached report round trips and alias guards. Differences between recipes are not ground-truth errors. |
| GUI checkpoints | Full-recipe capture, pair-boundary progress/cancellation, explicit recovery, fixed lazy prefixes, fresh aggregate export, and inspected default-size layouts. Distinguish offscreen checks from native input/responsiveness evidence. |

Do not present unavailable GPU runs, large-recording benchmarks, known-motion
experiments, or an untested GUI/stereo saved-experiment workflow as validated. The
remaining work belongs in [ROADMAP.md](ROADMAP.md).

## Register in dependency order

1. Finalize the core candidate and its release notes. After the core version
   change and validation pass, request core registration with the repository's
   Registrator workflow (`@JuliaRegistrator register`). Confirm registry
   availability and the resulting core tag before proceeding with a dependent
   GUI release.
2. Set the GUI's Hammerhead compatibility lower bound to the first registered
   core release supplying the APIs it now calls. The current development
   `[sources]` entry points to `..`; validate the release against the registered
   core in an isolated environment without that path override. Confirm the
   resolved version and rerun GUI tests. Do not publish a GUI release that only
   works against an unregistered checkout.
3. Finalize GUI notes/version and request subdirectory registration with
   `@JuliaRegistrator register subdir=HammerheadGUI`. Verify the GUI tag from the
   subdirectory TagBot job in
   [.github/workflows/TagBot.yml](.github/workflows/TagBot.yml).
4. Verify both packages install from the registry in a fresh environment and
   that published documentation matches the released API. Move the applicable
   Unreleased entries into dated, package-versioned sections; retain entries for
   changes that have not shipped.

The release record should link the final commits, registry entries, validation
evidence, and any known limitations.
