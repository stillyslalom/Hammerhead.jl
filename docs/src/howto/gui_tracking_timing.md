# Explore trajectories with actual sample times

Open a dedicated actual-time tracking artifact explicitly:

```julia
using HammerheadGUI
explorer = ResultExplorer("timed-tracks.jld2"; format=:timed_tracking)
figure = result_explorer(explorer)
# Equivalently: result_explorer("timed-tracks.jld2"; format=:timed_tracking)
```

The ordinary path constructor continues to read native result sequences. It
does not guess a timed format, unwrap a `TimedTrackingResult`, or substitute
ordinal frame intervals. You can also pass `ResultExplorer(timed)` directly.
Dedicated artifacts contain one **complete trajectory bundle**; the bundle
counter is one even when many input samples contributed. `lazy=true`, native
`ResultFile` browsing, mixed timed/native lists, and appending to this explorer
are unsupported. The complete bundle must fit memory.

Trajectories are colored by the **arithmetic observation mean of secant
magnitudes**. Each endpoint uses the adjacent observation; each interior row
uses the outer observations on either side. On irregular sampling these are
time-window mean slopes, not instantaneous velocities at that row. Averaging
their magnitudes does not produce an elapsed-time-weighted mean or path length
divided by elapsed time. No accuracy or uncertainty applicability is inferred.

Gray tracks have unavailable speed. Empty/singleton tracks, any nonfinite
position, secant or magnitude, and arithmetic overflow/underflow make the
whole track's speed unavailable. Bad observations are never silently omitted
and singleton speeds are never assigned zero. Finite singleton observations
remain visible and selectable. The plot title counts unavailable track speeds.

Click a trajectory to inspect the **whole track**: observation count, selected
input ordinal range, gap count, exact actual-time range/elapsed span, speed and
time-unit provenance. **Previous** and **next** page through the bounded
selection panel, including every digit of large timestamps and opaque clock
labels. A marker denotes the track's first finite observation; clicking a
vertex does not select a particular observation's instantaneous velocity.

Line breaks mean missing **selected-input observations**: consecutive stored
ordinals differ by more than one. A long actual interval between adjacent
selected samples does not itself create a detection gap. Original source frame
indices can differ from selected-input ordinals; the panel does not label these
ordinals as acquisition frame numbers. No gap observations are invented.

Scaled results pass through the timing-aware `physical` method: spatial
coordinates are converted once, the wrapper is retained and rebound, and
nominal `PhysicalScale.dt` never divides actual-time speeds. Known time units
are shown with positions' length unit (for example `mm/s` or `px/s`). Unknown
time units remain unknown. If a scale supplies a same-unit assumption for
unlabeled samples, the panel explicitly distinguishes that assumption from a
known acquisition unit. Clock labels are provenance, not synchronization proof.

The bulk core helper provides the same detached scalar summaries for scripts:

```@example gui_tracking_timing
using Hammerhead
using Hammerhead.SyntheticData: generate_gaussian_particle!

times = [0, 1, 4, 5]
images = [zeros(96, 96) for _ in times]
for (image, time) in zip(images, times)
    generate_gaussian_particle!(image, (20. + 2time, 30.), 3., 1.)
end
initial = (x=[1., 96.], y=[1., 96.], u=fill(2., 2, 2), v=zeros(2, 2))
timed = track_particles(images, PTVParameters(search_radius=.7, uod_enable=false);
    sample_times=times, time_unit="s", predictor=initial,
    min_track_length=4, progress=false)
summary = tracking_speed_summary(timed)
@assert length(summary.speeds) == 1 && summary.available[1]
@assert isapprox(summary.speeds[1], 2.; atol=1e-6)
(speed=summary.speeds[1], units=summary.length_unit * "/" * summary.time_unit,
    convention=summary.mean_convention, observations=summary.tracks[1].observations)
```

`tracking_speed_summary` verifies the whole timing/result binding once and
decodes the timeline once, then processes each trajectory without retaining
all trajectories' velocity arrays. Cost is O(all observations + selected input
samples); output is O(number of trajectories), with temporary arrays for one
trajectory. GUI refreshes and inspection also validate the complete current
bundle; they do not perform whole-result validation separately for every track.
No trusted mutable context, second raw-result cache, or unbounded track cache is
retained. Caller-retained raw/physical copies have their own memory cost.
Detached summaries describe the captured values; they do not continuously
monitor later mutations. A subsequent refresh/access detects broken binding
and reports an error instead of showing ordinal velocities.

The explorer has no new export buttons. Its public result retains timing:

```julia
displayed = current_result(explorer) # TimedTrackingResult, not its legacy .result
save_timed_tracking("displayed-timed-tracks.jld2", displayed)
export_table("displayed-timed-tracks.csv", displayed)
```

Dedicated persistence and CSV export check binding and protect known source
aliases. `save_results(path, explorer.results)` refuses the timed container;
use `save_timed_tracking`. Native eager/lazy explorers keep their existing
save dispatch. Planar history/execution inspection and planar profile/circulation
tools remain unsupported for timed trajectory bundles. See the
[actual-time tracking guide](tracking_timing.md) for linking, secant support,
precise timestamp persistence and export semantics.
