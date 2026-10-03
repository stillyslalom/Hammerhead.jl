# Opt-in actual FileDialog evidence; no replay or numerical processing is started.
const file_dialog_smoke_command = Observable("")
const file_dialog_smoke_url = Observable(String(QML.toString(FilePaths.file_url(fixture.path))))
const file_dialog_smoke_path = Observable(String(fixture.path))
const dialog_probe_directory = mktempdir(joinpath(@__DIR__, "artifacts"); prefix="dialog-data-", cleanup=false)
const dialog_paths = Dict(
    "record" => joinpath(dialog_probe_directory, "saved recipe λ # %.jld2"),
    "result" => joinpath(dialog_probe_directory, "native vectors 测量 # %.jld2"),
    "output" => joinpath(dialog_probe_directory, "fresh output λ # %.jld2"),
    "history" => joinpath(dialog_probe_directory, "history destination λ.jld2"))
Base.cp(fixture.path, dialog_paths["record"])
Hammerhead.save_results(dialog_paths["result"], [Prototype.dense_result(8) for _ in 1:3])
write(dialog_paths["history"], "Existing destination bytes must survive choosing SaveFile")
Prototype.open_saved_experiment(state, fixture.path) || error(state.experiment.error[])
Prototype.configure_saved_experiment(state, fixture.output, fixture.history, false) || error(state.experiment.error[])
state.explorer = HammerheadGUI.ResultExplorer(dialog_paths["result"]; lazy=true)
state.dataset[] = :native
state.displayed[] = "Displayed native fixture: three deterministic fields (no replay)"
state.count[] = 3
const dialog_input_digests = Dict(path => open(io -> bytes2hex(sha256(io)), path, "r")
    for path in (fixture.path, dialog_paths["record"], dialog_paths["result"], dialog_paths["history"]))
const dialog_snapshot = Ref{Any}(nothing)
const dialog_step = Ref(0)
const dialog_ack = Ref("")
const dialog_probe_queue = NamedTuple[]
const dialog_report = Dict{String,Any}("fixture_scope" => "Three deterministic native fields and an unprocessed saved planar recipe; no replay/PIV launched",
    "automation" => "offscreen Qt software, DontUseNativeDialog, Popup.Item, production SaveFile DontConfirmOverwrite, synthetic QtTest keys",
    "choices" => Any[], "processing_started" => false)
const dialog_steps = [
    "open-record", "select-record", "accept-record",
    "open-result", "select-result", "accept-result",
    "open-output", "select-output", "accept-output",
    "open-history", "select-history", "accept-history",
    "reject-result", "escape-result", "modal-left-output", "modal-right-output", "modal-close-output", "modal-run-output",
    "stale-result", "busy-output", "opening-failure-record",
    "capture-small", "capture-large", "shutdown-history"]

function dialog_stage(name; kwargs...)
    isdefined(Main, :shell_journal) && LifecycleEvidence.stage!(shell_journal, name; kwargs...)
end
function dialog_preserved(; shutdown=false)
    snapshot = dialog_snapshot[]
    snapshot === nothing && return
    ec = state.experiment.controller
    state.explorer === snapshot.explorer || error("dialog replaced the displayed explorer")
    state.frame[] == snapshot.frame && state.explorer.frame[] == snapshot.frame || error("dialog navigated the retained frame")
    state.explorer.selection[] == snapshot.selection && state.selection[] == snapshot.description || error("dialog changed selection")
    state.displayed[] == snapshot.displayed || error("dialog relabelled the display")
    ec.record[] === snapshot.record && ec.output_path[] == snapshot.output && ec.run_record_path[] == snapshot.history || error("dialog changed controller destinations or recipe")
    ec.progress[] == snapshot.progress && ec.state[] == snapshot.state && ec.status[] == snapshot.status || error("dialog changed replay state")
    !Prototype.busy(state) && state.experiment.job === nothing && state.experiment.request === nothing || error("dialog started processing")
    shutdown || (state.batch.cancel[] == snapshot.batch_cancel || error("Escape cancelled processing behind the picker"))
    viewports.generation == snapshot.generation && transition_open[] == snapshot.open || error("dialog closed/reopened the viewport")
    for (path, digest) in dialog_input_digests
        isfile(path) && open(io -> bytes2hex(sha256(io)), path, "r") == digest || error("dialog changed existing bytes: $path")
    end
    !ispath(dialog_paths["output"]) && !ispath(fixture.output) && !ispath(fixture.history) || error("SaveFile choice created a file")
    nothing
end

function file_dialog_probe(stage, purpose, token, visible, text)
    label, target, value = String(stage), String(purpose), String(text)
    captured_token, shown = Int(token), Bool(visible)
    # Callback scope is primitive capture only. The deliberate busy guard must
    # become effective before the same Qt callback attempts acceptance.
    if label == "busy-prepare"
        pending_work.val = true
    else
        push!(dialog_probe_queue, (; label, target, value, captured_token, shown))
    end
    nothing
end

function process_dialog_probe(probe)
  try
    (; label, target, value, captured_token, shown) = probe
    if startswith(label, "capture-") || label == "dialog-capture"
        fields = split(value, ',')
        length(fields) == 3 || error("invalid compact layout probe")
        w, h, reachable = parse.(Float64, fields)
        reachable == 1 || error("Browse controls do not fit the compact layout")
        suffix = label == "dialog-capture" ? "dialog" : split(label, '-')[end]
        path = joinpath(@__DIR__, "artifacts", "file_dialog-" * suffix * ".png")
        isfile(path) || error("missing actual Qt capture")
        dialog_report[label] = Dict("width"=>w, "height"=>h, "browse_controls_fit"=>true)
        dialog_stage("dialog_" * replace(label, '-' => '_'); width=w, height=h, browse_controls_fit=true, dialog_visible=shown)
        if label == "dialog-capture"
            shown && file_dialog_state.active || error("captured FileDialog was not open")
            dialog_report["dialogs_forced_non_native"] = true
        end
        label == "dialog-capture" && return
    elseif label == "shutdown-history"
        state.shutdown || error("root close did not request shutdown")
        !file_dialog_state.active && file_dialog_state.closed || error("shutdown retained a picker token")
        dialog_preserved(; shutdown=true)
        dialog_report["shutdown_preserved"] = true
        dialog_stage("dialog_shutdown"; accepted_after_close=false, controller_preserved=true)
        lifecycle_done[] = true
        captured[] = true
    else
        label == "busy-output" && (pending_work.val = false)
        command = dialog_steps[dialog_step[]]
        label == command || error("crossed/stale dialog smoke acknowledgement")
        action = first(split(label, '-'))
        if action == "open"
            shown && file_dialog_state.active && captured_token == file_dialog_state.generation || error("FileDialog did not actually open")
        elseif action == "select"
            shown && file_dialog_state.active || error("selection closed the dialog")
        elseif action == "accept"
            !shown && !file_dialog_state.active || error("accepted dialog remained active")
            normpath(value) == normpath(dialog_paths[target]) && basename(value) == basename(dialog_paths[target]) ||
                error("accepted URL lost file-name data: observed=$(repr(value)), expected=$(repr(dialog_paths[target])), picker_error=$(repr(file_dialog_error[]))")
            push!(dialog_report["choices"], Dict("purpose"=>target, "path"=>value, "accepted"=>true, "controller_preserved"=>true))
        elseif action in ("reject", "escape", "modal", "stale", "busy", "opening")
            !shown && !file_dialog_state.active || error("rejected/invalid picker retained its token")
            dialog_report[label] = true
        else
            error("unknown dialog probe: $label")
        end
        dialog_preserved()
        dialog_stage("dialog_" * replace(label, '-' => '_'); purpose=target, token=captured_token, dialog_visible=shown, controller_preserved=true)
    end
    dialog_ack[] = label
    nothing
  catch err
    file_dialog_smoke_command[] = ""
    dialog_stage("dialog_probe_failed"; error=sprint(showerror, err, catch_backtrace()), command=probe.label)
    rethrow()
  end
end

function file_dialog_smoke_tick()
  try
    while !isempty(dialog_probe_queue)
        process_dialog_probe(popfirst!(dialog_probe_queue))
    end
    dialog_snapshot[] === nothing || dialog_preserved(; shutdown=state.shutdown)
    if dialog_step[] == 0
        viewports.active === nothing && return
        Prototype.navigate(state, 2) || error(state.open_error[])
        result = HammerheadGUI.current_result(state.explorer)
        Prototype.pick(state, first(result.x), first(result.y))
        ec = state.experiment.controller
        dialog_snapshot[] = (explorer=state.explorer, frame=state.frame[], selection=state.explorer.selection[],
            description=state.selection[], displayed=state.displayed[], record=ec.record[], output=ec.output_path[],
            history=ec.run_record_path[], progress=ec.progress[], state=ec.state[], status=ec.status[],
            generation=viewports.generation, open=transition_open[], batch_cancel=state.batch.cancel[])
        dialog_report["retained_frame"] = 2
        dialog_report["input_sha256"] = copy(dialog_input_digests)
        dialog_report["fresh_paths"] = [dialog_paths["output"], fixture.output, fixture.history]
        dialog_stage("dialog_fixture_ready"; frame=2, input_files=length(dialog_input_digests), processing_started=false)
    elseif dialog_ack[] != dialog_steps[dialog_step[]]
        return
    elseif dialog_step[] == length(dialog_steps)
        return
    end
    dialog_step[] += 1
    command = dialog_steps[dialog_step[]]
    purpose = split(command, '-')[end]
    file_dialog_smoke_url[] = String(QML.toString(FilePaths.file_url(dialog_paths[purpose in FilePaths.PURPOSES ? purpose : "record"])))
    file_dialog_smoke_path[] = dialog_paths[purpose in FilePaths.PURPOSES ? purpose : "record"]
    dialog_ack[] = ""
    file_dialog_smoke_command[] = command
    nothing
  catch err
    file_dialog_smoke_command[] = ""
    dialog_stage("dialog_tick_failed"; error=sprint(showerror, err, catch_backtrace()))
    rethrow()
  end
end
