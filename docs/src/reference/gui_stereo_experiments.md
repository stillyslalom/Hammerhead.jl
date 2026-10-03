# Saved fitted-stereo GUI workflow

`StereoExperimentController` holds a detached full `StereoExperimentRecord`.
It does not coerce stereo settings into `ExperimentController` or rebuild
imported settings through `StereoBatchRunner` form fields.

Observable request choices include `output_path`, `run_record_path`,
`allow_environment_change`, `record_diagnostics` and `record_pair_timing`.
Observable state includes `running`, `state`, `status`, `error`, `progress`,
`last_run`, `selected_run_id` and `active_request`. The active request is a
detached scalar tuple of captured identities, destinations and options, published
before running notifications and cleared after cleanup. Progress is
`(written,total)` acquisitions.
`last_run` is the latest recorded attempt; selection independently identifies
historical inspection/reporting. Without a persisted history destination, core
failed-run metadata is unavailable. The GUI cancellation state corresponds to
a failed version-1 core run when failure history can be saved.

Result verification scans the completed native artifact before lazy opening.
It verifies bindings/settings and does not refit cameras, rerun PIV, certify
calibration accuracy or authenticate external sources. Displayed physical
arrays use the existing explorer conversion and integrity contracts. Reports
call the associated stereo core overload directly and preserve its default
format-1 / opt-in format-3 schema and generation-time scope. Known source and
destination aliases are protected before publication; two-file result/history
publication is not atomic and concurrent mutation is unsupported.

```@autodocs
Modules = [HammerheadGUI.Controllers, HammerheadGUI]
Pages = ["controllers/stereo_experiment_controller.jl", "views/stereo_experiment_workflow.jl"]
Private = false
```
