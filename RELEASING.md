# Releasing Hammerhead and HammerheadGUI

The core and GUI are separate registered packages in one repository. Release
the core first when the GUI uses new core APIs, then require that core version
in the GUI compatibility bounds before registering the GUI. A local path-based
development environment can hide a missing dependency lower bound.

## Prepare the candidate

Update [CHANGELOG.md](CHANGELOG.md) with user-visible API, numerical-behavior,
and file-format changes. Apply the
[compatibility policy](docs/src/explanation/compatibility.md): describe any
migration needed for results files, recipe files, result constructors, and
table readers, and bump the relevant format/schema version if existing
meanings change. Keep the root and GUI package versions independent.

Record the candidate commit, Julia and package versions, operating system,
thread count, and the commands used. First couple both development
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
`[sources]` override; without it the docs environment may resolve registered
packages instead of the candidate. Then run from the candidate checkout:

```sh
julia --project=. -t 4 -e 'using Pkg; Pkg.test()'
julia --project=HammerheadGUI -e 'using Pkg; Pkg.test()'
julia --project=docs docs/make.jl
```

The GUI tests and docs need a GL context; Linux CI supplies one through Xvfb
(see [.github/workflows/CI.yml](.github/workflows/CI.yml)). The docs build
executes the seven tutorials and checks public docstrings. Failure-propagation
logs in the tests and a local skipped-deployment warning are expected.

Check the CI matrix at the candidate commit: single-threaded core tests on
LTS, stable, and prerelease Julia; four-threaded stable Julia on Linux and
Windows; and GUI tests on LTS/stable Linux. macOS GUI support and GPU backends
need separate runs.

## Record evidence

Attach the evidence to the release record, with links to logs.

| Evidence | Record |
|---|---|
| Synthetic accuracy | Seeds, settings, reference convention, bias/RMS error, validity and uncertainty checks; name any tests changed with numerical behavior. |
| Committed real data | Challenge A and 4E fixture test/tutorial outcomes (smoke checks without ground truth). |
| CPU performance | Hardware, threads, precision, image size, and sequence length, following the [benchmark procedure](bench/README.md); `test/test_performance.jl` must pass. |
| CUDA/AMDGPU | Device, driver/runtime/package versions, `bench/gpu_validate.jl` results, and memory/performance measurements. |
| GUI workflow | OS/display, preprocessing/mask/ROI/scale, batch run and cancellation, settings save/open, reopening saved results, and export. |
| Persistence | Results and recipe round trips, unknown-version rejection, and loading files written by the previous release. |

## Register in dependency order

1. Finalize the core version and release notes, then request registration with
   `@JuliaRegistrator register`. Confirm the registry entry and core tag before
   releasing a GUI that depends on it.
2. Set the GUI's Hammerhead compat lower bound to the first core release with
   the APIs it calls. Validate against the registered core in an isolated
   environment without the `[sources]` path override, and rerun the GUI tests.
3. Finalize the GUI version and notes, and request
   `@JuliaRegistrator register subdir=HammerheadGUI`. Verify the
   `HammerheadGUI-v*` tag from the subdirectory TagBot job in
   [.github/workflows/TagBot.yml](.github/workflows/TagBot.yml).
4. Install both packages from the registry in a fresh environment and check
   that the published documentation matches the released API. Move the
   shipped Unreleased entries into dated, versioned sections.
