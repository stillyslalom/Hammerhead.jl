# Saved ensemble experiment GUI

The framework-free controller owns complete ensemble recipes, metadata history,
captured replay/cancellation state and verified result/report requests. GLMakie
views own forms, paging and window launch. See the
[workflow guide](../howto/gui_ensemble_experiments.md).

`progress` counts joined contributions, `location` identifies `(pass,pair)`, and
`pool_progress` counts `(completed_pools,published_results)`. A completed budget
does not imply publication. A history failure after a known completed/cancelled
run uses `:history_save_failed` or `:history_refresh_failed` while retaining the
actual `last_run.status`, counts and verified-output eligibility. These are GUI
states, not additions to the persisted run schema.

Imported recipes preserve requested ignored pass settings. Inspection/reporting
verify output association at the time of the request; metadata inspection does
not load source images. Input-byte verification is explicit. Lazy exploration
retains one display payload and current scalar companion. A separate displayed
window or saved report keeps its captured identity after workflow selection changes.

```@meta
CurrentModule = HammerheadGUI
```

```@autodocs
Modules = [HammerheadGUI.Controllers]
Pages = ["controllers/ensemble_experiments.jl"]
Private = false
```

```@autodocs
Modules = [HammerheadGUI]
Pages = ["views/ensemble_experiments.jl"]
Private = false
```
