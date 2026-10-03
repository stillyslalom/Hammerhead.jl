# Explore trajectories with actual sample times

Open a dedicated actual-time tracking artifact explicitly:

```julia
using HammerheadGUI
explorer = ResultExplorer("timed-tracks.jld2"; format=:timed_tracking)
figure = result_explorer(explorer)
# Equivalently: result_explorer("timed-tracks.jld2"; format=:timed_tracking)
```

Use `format=:timed_tracking` for a dedicated artifact, or pass
`ResultExplorer(timed)` directly. Each artifact opens one complete trajectory
bundle, even when it contains many samples. Choose a bundle that fits memory.

Trajectories are colored by the **arithmetic observation mean of secant
magnitudes**. Endpoints use the adjacent observation; interior rows use the
observations on either side. Each secant measures average motion over its time
interval. The track color averages those magnitudes with equal observation weights.

Gray tracks have unavailable speed. Empty/singleton tracks, a nonfinite position
or secant, and arithmetic range failures make the whole summary unavailable.
Finite singleton observations remain visible and selectable. The plot title
counts unavailable tracks.

Click a trajectory to inspect the **whole track**: observation count, selected
input ordinal range, gaps, actual-time span, speed, and unit provenance.
**Previous** and **next** expose full timestamps and clock labels. The marker
identifies the first finite observation; selection reports the whole track.

Line breaks mark missing **selected-input observations**, where stored ordinals
differ by more than one. The time display separately shows elapsed intervals
between samples. Source frame indices and selected-input ordinals can differ;
the panel names its ordinal basis explicitly.

Spatial scaling converts positions once while retaining the timing wrapper.
Speeds use actual sample-time differences. Known time units appear alongside
the length unit, such as `mm/s` or `px/s`; unlabeled times stay unknown unless
an explicit scale-unit assumption is supplied. The panel distinguishes that
assumption from recorded acquisition units and shows clock provenance.

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

`tracking_speed_summary` checks the current timing/result binding and returns
detached per-track summaries. Refresh/access repeats the binding check. If you
edit a bound result and a check fails, reopen the original artifact.

Export the current timed result through its dedicated persistence or CSV APIs:

```julia
displayed = current_result(explorer) # TimedTrackingResult, not its legacy .result
save_timed_tracking("displayed-timed-tracks.jld2", displayed)
export_table("displayed-timed-tracks.csv", displayed)
```

Dedicated persistence and CSV export check binding and protect known source
aliases. Use `save_timed_tracking` for the timed container. The
[actual-time tracking guide](tracking_timing.md) describes linking, secant support,
timestamp persistence, and export columns.