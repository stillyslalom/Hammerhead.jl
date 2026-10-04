# Analyze an image pair in the GUI

Start with the frames of a recording. In one window you will check the
images, mask what should not be measured, choose the passes, test one pair,
run the recording, and inspect the result. For a complete session with sample
images, follow [Your first PIV session in the GUI](../tutorials/gui_tour.md).

## Open the window

Install the GUI package with `pkg> add HammerheadGUI`. Start Julia with
several threads, so tests, runs and previews leave the window responsive:

```julia
# julia -t auto
using HammerheadGUI
wf = planar_window()
```

The call returns when you close the window. It returns the window's
`PlanarWorkflow`, with every setting you made. Pass frames or saved settings
to start further along: `planar_window(files = paths, settings = "settings.jld2")`.

The steps on the left run in order: **Images → Prepare → Passes → Test pair →
Run → Results**. Each step shows a one-line summary, and a dot marks it as
to do, done, needing attention, or busy. The viewer on the right follows the
step: frames and overlays while you prepare, window outlines on **Passes**,
vectors on **Test pair** and **Run**, and the result field on **Results**.
Drag a box in the viewer to zoom, right-drag to pan, and scroll to zoom.
**Pop out viewer** in the toolbar moves the viewer into its own window, for a second
screen. Closing that window docks the viewer again.

## Add the frames

On **Images**, click **Add frames…** and select the frames. Then choose
**Pairing**: **Paired** for separate A/B exposures (1–2, 3–4), **Chained** for a
uniformly sampled sequence (1–2, 2–3). The bar at the bottom of the window selects the
*representative pair* and switches between frame A and frame B. Previews,
probes, window outlines and the test use this pair. Choose one with the
flow features you care about; dim or fast regions are good checks.

![The Images step with four frames in two pairs.](../assets/gui_window/images.png)

Frames load in the background. The step rail shows "loading…" until a new
pair arrives, and the viewer keeps the previous one meanwhile.

## Prepare the images

**Prepare** has four pages. Clicks on the viewer go to the open page.

### Preprocess and probe the correlation

Add steps with **Add step**; they run in the listed order on every frame
before correlation. The arrows reorder a step, **✕** removes it, and each
option is edited in place. **Estimate background** takes the pixel-wise
minimum of the first frames and subtracts it as the first step. It runs in
the background; a status line reports progress.

Switch the viewer between **Raw** and **Processed**, then click the image to
place the **correlation probe**. The probe correlates one window of the
processed pair there and reports the displacement and the peak ratio. A ratio
well above 1.5 means a clear match. Probe a few places, including weak
regions. The preview runs exactly the preprocessing the batch runs, so what
you see is what will be correlated.
[Build a preprocessing chain](preprocessing.md) explains when each operation
helps.

![Prepare, Preprocess page: a highpass filter, the processed frame, and the probe window.](../assets/gui_window/prepare_preprocess.png)

### Mask walls and reflections

On **Mask**, click the image to add polygon vertices and right-click to
close the polygon. The shaded area is excluded from the analysis.

- **Backspace** undoes the last vertex; **Escape** cancels the polygon.
- Click inside a finished polygon to select it; **Delete** removes it.
- **New hole** makes the next polygon restore an area inside an excluded
  region.
- **Grow** and **Shrink** widen or narrow the excluded area by a margin. The
  polygons then become a fixed raster mask, which you can still draw on.
- **Open mask image…** reads a mask file (white = excluded), and **Save mask
  image…** writes one.

![Prepare, Mask page: a polygon excludes the reflection.](../assets/gui_window/prepare_mask.png)

The mask is static: it stays in the lab frame for every pair. For automatic
masks and the masking model, see [Mask reflections and geometry](masking.md).

### Analyze part of the frame

On **Region**, click two opposite corners on the image or type inclusive row
and column bounds, then **Apply bounds**. **Full image** removes the region.
Results keep the original image coordinates, and the pass presets size
themselves to the region.

### Attach a physical scale

On **Scale**, type the pixel size with its length unit and the time between
the paired exposures with its time unit. For example, 0.02 mm per pixel and
0.001 s. To measure the pixel size, click two points of known separation on
the image, such as ruler marks, and type their **Distance**. Without a scale,
results stay in pixels and frames. Vectors are always computed in pixels; the
scale converts what the results show (see
[Scale results to physical units](scaling.md)).

![Prepare, Scale page: pixel size and frame interval typed in.](../assets/gui_window/prepare_scale.png)

## Choose the passes

On **Passes**, click a preset: **Low**, **Medium** or **High**. The preset fills
the pass table for the frame or region size; edit any cell to make the
schedule your own. The first window should be at least four times the largest
displacement. The viewer outlines each window size against the particles
to help you judge this. [Choose an effort level](effort.md) explains the
trade-off.

Below the table:

- **Correlation:** method, subpixel fit, **Padding and Gaussian weighting** (the
  most accurate setting), and per-vector **Uncertainty on final pass**.
- **Evaluation:** **Per pair** gives a time series; **Ensemble** sums the
  correlation over all pairs into one mean field
  (see [Measure one field from many pairs](ensemble.md)). The window tests an
  ensemble on the first ten pairs; its **Run** step processes per-pair
  sequences, so run a full ensemble from saved settings with `apply_recipe`.
- **Precision:** Float32 halves the memory of Float64.

![Passes: the medium preset and its window sizes outlined on the particles.](../assets/gui_window/passes.png)

## Test one pair

On **Test pair**, click **Test pair** (or **Test ensemble**, which uses the
first ten pairs). The test runs the same call the batch will run. It reports
the valid and flagged fractions, the median peak ratio, the largest
displacement and how long the test took. The viewer shows valid vectors in blue and
flagged vectors in red. When you change a setting or the representative pair,
the summary says it is out of date until you test again.

![Test pair: the summary and the vectors of the representative pair.](../assets/gui_window/test_pair.png)

Many red vectors usually mean the first window is too small for the
displacement, or that a region needs a mask or preprocessing.
[Tune validation](validation.md) covers the outlier test.

## Run the recording

On **Run**, choose an output file with **Browse…** (leave it empty to keep
results in memory), then click **Run**. Results are written as each pair
finishes, together with the settings that produced them. The viewer shows
the latest finished pair, with elapsed time and an estimate of the time left.
**Cancel** stops after the pair in progress and keeps the finished ones.
Closing the window also cancels a run in this way.

## Inspect the results

When a run finishes, **Results** holds its output. **Open results…** browses
another results file; entries load one at a time.

- **Pair** steps through the results, **Field** chooses what is coloured
  (components, magnitude, diagnostics, and derived fields such as vorticity),
  and **Colour range** switches between a robust 2–98 % range and the full range.
- **Inspect:** click a vector to read its position, components and status.
- **Profile:** click two points to sample u, v and |V| along a line. A panel
  under the field plots them.
- **Circulation:** click contour vertices and right-click to close the contour.
  The summary gives Γ from the line integral and from the enclosed vorticity,
  and the covered fraction when masked or flagged cells leave gaps.
- **Clear** (or Escape on the viewer) removes the line or contour.

![Results with the Profile tool: a line across the vortex and the velocity along it.](../assets/gui_window/results_profile.png)

With a scale attached, axes, colour bars and summaries use its units.
Profile and circulation need a planar PIV result. Check flagged vectors and the mask
before you interpret derived quantities, because derivatives amplify local
errors.

## Save and reuse the settings

**Save settings…** writes the passes, preprocessing, mask, region and scale to
a recipe file. **Open settings…** reads a recipe file, or the settings stored
in any results file. **Use these settings** on **Results** does the same for the
open results file. A dot after the title in the window bar marks unsaved
changes. Settings files are core recipes, so a script can run them with
`apply_recipe`. See [Save settings and reuse them](recipes.md).

## Browse a results file without the workflow

The standalone result explorer opens a completed file and loads one entry at
a time:

```julia
using HammerheadGUI
result_explorer("vectors.jld2"; lazy = true)
```

It browses planar and stereo PIV, PTV particles and trajectories. The
[result explorer reference](../reference/gui_results.md) lists its fields
and tools.

## Work with two cameras

The planar window handles one camera. For stereo, start with
[Calibrate a real stereo rig](stereo_rig.md).
`stereo_calibration(cr1, cr2; batch)` reviews both cameras' calibrations
and installs the dewarpers into a `stereo_batch_runner` form, which runs the
synchronized frames.

## Script the window's steps

Every button calls a function on the workflow's controllers in
`HammerheadGUI.Controllers`, so the same steps run without a window. The
[GUI tour](../tutorials/gui_tour.md) shows the code next to each click, and the
[GUI reference](../reference/gui.md) lists the functions.
