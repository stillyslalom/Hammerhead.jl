# Development and validation

This section is for contributors and readers investigating how Hammerhead is
tested. The analysis workflow is covered by
[the tutorials](../tutorials/first_vector_field.md).

## Test a scientific change

[How accurate are the measurements?](../explanation/validation_results.md)
summarizes the synthetic accuracy, uncertainty, spatial-response and particle
tracking studies, with the numbers a change should reproduce or improve. The
test suite (`julia --project=. -t 4 -e 'using Pkg; Pkg.test()'`) checks the
displacement tolerances and conventions behind them; `bench/README.md` lists
the performance benchmarks.

## Change the software

The [GUI architecture](../explanation/gui.md) separates application logic from
its widgets. The [compatibility policy](../explanation/compatibility.md) covers public APIs
and saved data when preparing a release.

The repository's `ROADMAP.md` is the development backlog. Learning pages should
teach a useful task; keep test inventories and delivery status in the
development and reference material.
