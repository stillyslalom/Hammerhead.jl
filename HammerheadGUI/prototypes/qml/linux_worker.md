# Linux replay-worker ownership

This is the ownership backend for the isolated Qt prototype's saved-planar
replay. It does not add Qt dependencies to Hammerhead or establish Linux GUI
rendering, desktop input, packaging or macOS support.

## Process lifetime

The owner starts a small Julia guardian, which starts the core-only scientific
worker in a private process group. The guardian never imports Hammerhead or Qt.
Its exclusive socket endpoint detects owner exit independently of the worker's
numerical loop. A healthy guardian terminates the bound group when the owner
disappears, reaps the root, and checks adopted descendants before exiting.

Guardian and worker each open their own PID file descriptor while alive and
transfer it using `SCM_RIGHTS`. The owner retains those kernel references;
it does not reopen a numeric PID or process-group ID after asynchronous startup.
Enrollment is nonblocking: `pid(job) == 0` means that the worker identity has not
arrived. `job.process` refers to the guardian on Linux and must not be presented
as the scientific worker. The bootstrap cannot enter a worker script or import
the scientific package until owner enrollment.

The group operation requires `PIDFD_SIGNAL_PROCESS_GROUP`, introduced in
Linux 6.9. A kernel version string alone is insufficient: the bootstrap and
guardian exercise the required operation before scientific enrollment.
[Linux signal interface](https://man7.org/linux/man-pages/man2/pidfd_send_signal.2.html).
The guardian uses `PR_SET_CHILD_SUBREAPER` so orphaned descendants are adopted
and can be waited for.
[Linux subreaper interface](https://man7.org/linux/man-pages/man2/PR_SET_CHILD_SUBREAPER.2const.html).

The current implementation selects x86_64 Linux and uses libc's exported
`pidfd_open` and `pidfd_send_signal` functions plus the checked 64-bit message
header layout. Local execution uses Ubuntu 24.04.2 under WSL, kernel
6.18.33.2-microsoft-standard-WSL2 and Julia 1.11.6. Other libc implementations,
architectures and hosted runner images require their own capability evidence.
Unsupported host selection refuses before spawning; missing Linux kernel/libc
capabilities can fail in the guardian/bootstrap before scientific enrollment.

## Completion and failure

Successful ownership cleanup requires all of the following: the root was
waited for, the held group reference reports no members, the subreaper reports
`ECHILD`, a strictly validated proof matches the request and enrolled identities,
and the guardian exits normally and is reaped by the owner. Scientific success
separately requires valid terminal metadata, matching worker OS status and all
required progress acknowledgements. Forced termination can prove OS cleanup
without proving scientific completion.

A failed enrollment requests termination without admitting scientific work.
The cached `ownership_error(job)` diagnostic allows the shell to explain why
cleanup is unconfirmed. It must retain the job, captured request and busy guard,
refuse a new replay and stop progress delivery/acknowledgements. It must not
manufacture a terminal outcome or discard the original failure.

If a descendant escapes the private group, group emptiness alone is insufficient:
the guardian remains alive while that adopted child remains. Abnormal guardian
death destroys its descendant-cleanup guarantee. The owner attempts termination
through the retained worker-group reference but still refuses to certify cleanup
or launch again. Independent fixture rescue demonstrates test cleanup only;
it cannot restore the production guardian's lost proof.

The guardian is an additional Julia process. Its startup and memory cost need
measurement in platform adoption work; the small injected CPU-loop fixture is
not a scientific throughput or desktop responsiveness benchmark.

## Integration evidence

The final integrated Linux sequence passed 295 assertions: 123 ownership checks
(transport 24, enrollment 17, startup failure 13, abrupt owner loss 18, guardian
loss 17, escaped descendants 14 and malformed proof/enrollment 20), 41 shared
client checks, 90 real-replay checks and 41 lifecycle-owner checks. All ten
commands and the outer driver exited zero. The actual owner-loss fixture holds
independently transferred owner, guardian, worker and descendant descriptors;
it observes them alive before killing the client and verifies worker/descendant
exit, normal guardian exit and complete reaping afterward.

Logs and source/environment snapshots are retained in
`bench/profile-output/linux-worker-draft/main-integration-20261003-105112`.
Before/after snapshots match, including 69 prototype files, core/GUI source and
extension files, core Project/Manifest files and the isolated Linux core
environment. These runs use the integrated main prototype paths rather than the
earlier ignored development copy. The Linux lifecycle self-test also rejects a
child that publishes its completion stage and then dies by SIGKILL; an exit-code
field of zero cannot override its termination signal.

The source-frozen Windows integration checks passed 113 adapter assertions,
12 cached-error-page checks and 38 lifecycle-owner checks, with an authoritative
zero exit for the complete chain. The adapter checks include a real deferred
write acknowledgement whose worker reaches its failure terminal before the
next GUI poll: shutdown adopts that terminal without writing a late ACK. A
separate adapter fixture verifies visible ownership errors, retained display and
request identity, refused progress/acknowledgements, and refusal to dispose a
still-busy model. These adapter fixtures do not establish Linux kernel behavior.

The retained local log is
`bench/profile-output/qml-linux-integration-adapter-focused.log`, with owner
self-test evidence in `artifacts/lifecycle-PVI9Fj`. The parent snapshot records
69 prototype files, 47 core source files and 31 GUI source files in
`bench/profile-output/linux-integration-main-source.json`.

The final Windows worker regressions also passed: 48 client/ownership assertions
in 85.79 seconds and 90 real-replay assertions in 269.35 seconds, each with a
zero process exit. Retained logs are
`artifacts/windows-linux-merge-worker_client_tests.log` and
`artifacts/windows-linux-merge-worker_replay_tests.log`. These runs overlapped
other focused checks, so their elapsed times are test-duration evidence rather
than startup or performance benchmarks. The manual replay lane allows 600 seconds.

After those numerical runs finished, the final Windows Qt/GLFW child in
`artifacts/worker-lifecycle-LLuQcD` exited zero with no termination signal or
timeout in 100.42 seconds. All 20 stages passed: control/render servicing,
picking/pan/zoom, close/reopen, three actual native writes, compact layouts and
joined worker/subscription/window disposal. Four viewport generations were
released. Independent audits matched the 69/47/31 source maps and all five PNG
digests; active, small, large and scientific captures were visually inspected.

For the recorded three-pair 64-square fixture, the maximum owner service gap was
1.031 seconds over the whole active interval and 0.036 seconds after the first
write acknowledgement. The latter excludes imports and the first pair; injected
control-barrier work is distinguished from numerical processing in
`worker_report.toml`. These scoped observations do not establish desktop latency
or Linux graphics support. Native Qt context-release verification remains false;
the scientific screen uses the separately owned GLFW path.

## Reproduction

Run each ownership fixture in a separate process. A failed fixture stops the
sequence until its retained process evidence establishes cleanup. The manual
workflow enforces a conservative refusal of subsequent launches after any failed
child command. Generated requests, logs and proofs remain under `artifacts/`.

```sh
julia --startup-file=no --threads=1 --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/linux_transport_tests.jl
julia --startup-file=no --threads=1 --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/linux_enrollment_tests.jl
julia --startup-file=no --threads=1 --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/linux_startup_failure_test.jl
julia --startup-file=no --threads=1 --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/linux_poll_fault_tests.jl
julia --startup-file=no --threads=1 --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/linux_worker_owner_loss_tests.jl
julia --startup-file=no --threads=1 --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/linux_client_ownership_tests.jl guardian_loss
julia --startup-file=no --threads=1 --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/linux_client_ownership_tests.jl escaped
julia --startup-file=no --threads=1 --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/worker_client_tests.jl
julia --startup-file=no --threads=1 --project=HammerheadGUI/prototypes/qml HammerheadGUI/prototypes/qml/worker_replay_tests.jl
```

These tests import the core package and standard libraries, so a separate
environment that develops the same Hammerhead checkout can exercise them without
Qt/GL dependencies. Record the actual Project/Manifest and source identities;
that environment does not validate the GUI's resolved dependency set.
