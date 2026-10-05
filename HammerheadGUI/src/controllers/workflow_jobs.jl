# Background jobs of a workflow window: testing the representative pair and
# running the batch. Work runs on a worker thread (`spawn = true`); every
# observable update is handed to `deliver`, which a GUI shell points at its
# event-loop queue so observers only ever run on the GUI thread. With
# `spawn = false` and the default `deliver`, jobs run inline (tests, scripts).

_run_job(job, spawn::Bool) = spawn ? errormonitor(Threads.@spawn job()) : job()

"""
    PairTest()

The latest test of the current settings on the representative pair:
`result` (a `PIVResult`, a `StereoPIVResult` in a stereo workflow, or a
`PTVResult`/`TrackingResult` in the particle modes), the
`recipe` and `pair` it was computed with, `seconds` taken, the `previous`
test's summary for comparison, `running`/`status`, and `show_vectors`
(whether the Passes viewer draws the result; a new test turns it on). `inputs[]` holds the
`apply_recipe` inputs after the recipe (pairs; for stereo also the
dewarpers) of the last successful test, and `options[]` its keyword inputs
(per-pair `masks`).
"""
const TestResult = Union{Nothing,PIVResult,StereoPIVResult,PTVResult,TrackingResult}

struct PairTest
    result::Observable{TestResult}
    recipe::Observable{Union{Nothing,PIVRecipe}}
    pair::Observable{Int}
    seconds::Observable{Float64}
    previous::Observable{Union{Nothing,NamedTuple}}
    running::Observable{Bool}
    status::Observable{String}
    show_vectors::Observable{Bool}
    inputs::Base.RefValue{Any}
    options::Base.RefValue{Any}
end

PairTest() = PairTest(Observable{TestResult}(nothing),
                      Observable{Union{Nothing,PIVRecipe}}(nothing), Observable(0),
                      Observable(0.0), Observable{Union{Nothing,NamedTuple}}(nothing),
                      Observable(false), Observable(""), Observable(true), Ref{Any}(nothing),
                      Ref{Any}((;)))

"""
    start_test!(pt::PairTest, recipe, pairs, label; deliver = f -> f(), spawn = true, options = (;))
    start_test!(pt::PairTest, recipe, inputs::Tuple, label; deliver, spawn, options)

Run `apply_recipe(recipe, pairs; options...)` (one pair, or several for an
ensemble recipe) and store the result. `label` is the representative pair
index. The tuple form runs `apply_recipe(recipe, inputs...)`, e.g. the stereo
`(pairs1, pairs2, dw1, dw2)`; `options` are further keyword inputs such as
per-pair `masks`.
"""
start_test!(pt::PairTest, recipe::PIVRecipe, pairs::AbstractVector, label::Integer; kwargs...) =
    start_test!(pt, recipe, (pairs,), label; kwargs...)

function start_test!(pt::PairTest, recipe::PIVRecipe, inputs::Tuple, label::Integer;
                     deliver = f -> f(), spawn::Bool = true, options::NamedTuple = (;))
    pt.running[] && return pt
    pt.running[] = true
    pt.status[] = recipe.mode === :ensemble ?
        "testing ensemble of $(length(first(inputs))) pairs…" :
        recipe.mode === :tracking ? "tracking through $(length(first(inputs))) frames…" :
        "testing pair $label…"
    job = function ()
        t0 = time()
        outcome = try
            r = apply_recipe(recipe, inputs...; progress = false, options...)
            (; result = r isa AbstractVector ? first(r) : r, seconds = time() - t0, err = nothing)
        catch err
            (; result = nothing, seconds = time() - t0, err)
        end
        deliver(() -> _finish_test!(pt, recipe, Int(label), outcome, inputs, options))
    end
    _run_job(job, spawn)
    return pt
end

function _finish_test!(pt::PairTest, recipe, label, outcome, inputs = nothing, options = (;))
    if outcome.err === nothing
        pt.result[] === nothing || (pt.previous[] = test_summary(pt.result[], pt.recipe[], pt.seconds[]))
        pt.inputs[] = inputs
        pt.options[] = options
        pt.recipe[] = recipe
        pt.pair[] = label
        pt.seconds[] = outcome.seconds
        pt.show_vectors[] || (pt.show_vectors[] = true)
        pt.result[] = outcome.result
        pt.status[] = ""
    else
        pt.status[] = "test failed: " * _errmsg(outcome.err)
    end
    pt.running[] = false
    return pt
end

"""
    test_summary(result, recipe, seconds) -> NamedTuple
    test_summary(pt::PairTest) -> Union{Nothing,NamedTuple}

Quality figures for a test: vector counts (`vectors`, `valid`, `flagged`,
`masked`), `valid_fraction`, median `peak_ratio`, median `sigma` (`NaN`
without uncertainty) in `sigma_unit`, the largest valid displacement
`max_displacement` (px) and `quarter_window` (¼ of the final window;
displacements beyond it are hard to correlate reliably), and `seconds`.

For a `StereoPIVResult` the peak ratio is the weaker camera's, `sigma` is
the 3C uncertainty in world units (named by the recipe scale's length unit
when there is one), and `max_displacement` is the larger camera
displacement in dewarped pixels.
"""
function test_summary(r::PIVResult, recipe::PIVRecipe, seconds::Real)
    good = .!(r.mask .| r.outliers) .& isfinite.(r.u) .& isfinite.(r.v)
    sig = hypot.(r.uncertainty_u[good], r.uncertainty_v[good])
    disp = hypot.(r.u[good], r.v[good])
    return _test_summary(r, good, r.peak_ratio[good], sig, disp, "px", recipe, seconds)
end

function test_summary(r::StereoPIVResult, recipe::PIVRecipe, seconds::Real)
    good = .!(r.mask .| r.outliers) .& isfinite.(r.u) .& isfinite.(r.v) .& isfinite.(r.w)
    c1, c2 = r.cam1, r.cam2
    peak = min.(c1.peak_ratio[good], c2.peak_ratio[good])
    sig = sqrt.(abs2.(r.uncertainty_u[good]) .+ abs2.(r.uncertainty_v[good]) .+
                abs2.(r.uncertainty_w[good]))
    disp = max.(hypot.(c1.u[good], c1.v[good]), hypot.(c2.u[good], c2.v[good]))
    unit = recipe.scale === nothing ? "world units" : recipe.scale.length_unit
    return _test_summary(r, good, peak, sig, disp, unit, recipe, seconds)
end

function _test_summary(r, good, peak, sig, disp, sigma_unit, recipe, seconds)
    n = length(r.u)
    masked = count(r.mask)
    flagged = count(r.outliers .& .!r.mask)
    valid = count(good)
    med(xs) = isempty(xs) ? NaN : _median(xs)
    final = last(recipe.passes)
    return (; kind = :piv, vectors = n, valid, flagged, masked,
            valid_fraction = n - masked == 0 ? NaN : valid / (n - masked),
            peak_ratio = med(filter(isfinite, Float64.(peak))),
            sigma = med(filter(isfinite, Float64.(sig))), sigma_unit = String(sigma_unit),
            max_displacement = valid == 0 ? NaN : maximum(Float64.(disp)),
            quarter_window = minimum(final.window_size) / 4, seconds = Float64(seconds))
end

"""
    test_summary(r::PTVResult, recipe, seconds)

For a PTV test: `particles_a`/`particles_b` detected, `matches`, `valid`
and `flagged` matches, `valid_fraction` (unflagged share of the matches),
`match_fraction` (matches per frame-A particle), median `displacement` and
`residual` (px; the distance from the predicted position), `search_radius`,
`predictor` (`:piv`/`:none`), and `seconds`.
"""
function test_summary(r::PTVResult, recipe::PIVRecipe, seconds::Real)
    n = length(r.u)
    good = .!r.outliers .& isfinite.(r.u) .& isfinite.(r.v)
    valid = count(good)
    med(xs) = isempty(xs) ? NaN : _median(Float64.(xs))
    na = length(r.particles_a)
    return (; kind = :ptv, particles_a = na, particles_b = length(r.particles_b), matches = n,
            valid, flagged = n - valid, valid_fraction = n == 0 ? NaN : valid / n,
            match_fraction = na == 0 ? NaN : n / na,
            displacement = med(hypot.(r.u[good], r.v[good])),
            residual = med(filter(isfinite, r.match_residual[good])),
            search_radius = recipe.ptv.search_radius, predictor = recipe.ptv_predictor,
            seconds = Float64(seconds))
end

"""
    test_summary(r::TrackingResult, recipe, seconds)

For a tracking test: `tracks`, `frames` followed, mean and longest track
length (observations), `gaps` bridged, `min_track_length`, and `seconds`.
"""
function test_summary(r::TrackingResult, recipe::PIVRecipe, seconds::Real)
    lens = [length(t.x) for t in r.trajectories]
    gaps = sum((count(>(1), diff(t.frames)) for t in r.trajectories); init = 0)
    return (; kind = :tracking, tracks = length(lens), frames = r.n_frames,
            mean_length = isempty(lens) ? NaN : sum(lens) / length(lens),
            longest = isempty(lens) ? 0 : maximum(lens), gaps,
            min_track_length = recipe.min_track_length, valid_fraction = NaN,
            seconds = Float64(seconds))
end

"""
    test_brief(summary) -> String

A test summary in a few words, for the step rail.
"""
function test_brief(s::NamedTuple)
    k = get(s, :kind, :piv)
    k === :tracking && return @sprintf("%d tracks through %d frames · %.2f s", s.tracks, s.frames, s.seconds)
    k === :ptv && return @sprintf("%d matches, %.0f %% valid · %.2f s", s.matches,
                                  100 * (isfinite(s.valid_fraction) ? s.valid_fraction : 0.0), s.seconds)
    return @sprintf("%.0f %% valid · %.2f s", 100 * s.valid_fraction, s.seconds)
end

test_summary(pt::PairTest) =
    pt.result[] === nothing ? nothing : test_summary(pt.result[], pt.recipe[], pt.seconds[])

function _median(xs::Vector{Float64})
    s = sort(xs)
    n = length(s)
    return isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2
end

"""
    summary_lines(summary; previous = nothing) -> Vector{String}

Readable lines for a `test_summary`, with changes from `previous`.
"""
function summary_lines(s::NamedTuple; previous = nothing)
    pct(x) = isfinite(x) ? @sprintf("%.1f %%", 100x) : "–"
    num(x) = isfinite(x) ? @sprintf("%.2f", x) : "–"
    # compare only with a previous test of the same kind
    kind = get(s, :kind, :piv)
    previous !== nothing && get(previous, :kind, :piv) !== kind && (previous = nothing)
    delta(f, fmt) = previous === nothing || !isfinite(previous[f]) || !isfinite(s[f]) ? "" :
                    " (" * fmt(s[f] - previous[f]) * ")"
    signed_pct(d) = @sprintf("%+.1f pts", 100d)
    signed(d) = @sprintf("%+.2f", d)
    if kind === :ptv
        lines = ["Particles: $(s.particles_a) in frame A, $(s.particles_b) in frame B",
                 "Matches: $(s.matches) ($(pct(s.match_fraction)) of frame A)" *
                     delta(:match_fraction, signed_pct),
                 "Valid matches: $(s.valid) ($(pct(s.valid_fraction))) · flagged: $(s.flagged)" *
                     delta(:valid_fraction, signed_pct),
                 "Median displacement: $(num(s.displacement)) px · median match residual: $(num(s.residual)) px"]
        # without a predictor the search is centered on the particle itself; with
        # one, matches far from the prediction point to a poor predictor
        if s.predictor === :none && isfinite(s.displacement) && s.displacement > 0.8 * s.search_radius
            push!(lines, "Displacements approach the search radius ($(num(s.search_radius)) px): " *
                         "use the PIV predictor or a larger radius")
        elseif s.predictor === :piv && isfinite(s.residual) && s.residual > 0.5 * s.search_radius
            push!(lines, "Matches sit far from the PIV prediction (search radius " *
                         "$(num(s.search_radius)) px): check the predictor passes or widen the search")
        end
        push!(lines, @sprintf("Time: %.2f s", s.seconds))
        return lines
    elseif kind === :tracking
        return ["Tracks: $(s.tracks) through $(s.frames) frames (at least $(s.min_track_length) observations each)" *
                    delta(:tracks, d -> @sprintf("%+d", d)),
                "Mean length: $(num(s.mean_length)) observations · longest: $(s.longest)",
                "Gaps bridged: $(s.gaps)",
                @sprintf("Time: %.2f s", s.seconds)]
    end
    lines = ["Valid vectors: $(s.valid) of $(s.vectors - s.masked) ($(pct(s.valid_fraction)))" *
                 delta(:valid_fraction, signed_pct),
             "Flagged: $(s.flagged) · masked: $(s.masked)",
             "Median peak ratio: $(num(s.peak_ratio))" * delta(:peak_ratio, signed)]
    isfinite(s.sigma) &&
        push!(lines, "Median uncertainty: $(num(s.sigma)) $(get(s, :sigma_unit, "px"))" * delta(:sigma, signed))
    if isfinite(s.max_displacement)
        line = "Largest displacement: $(num(s.max_displacement)) px"
        s.max_displacement > s.quarter_window &&
            (line *= "; above ¼ of the final window ($(num(s.quarter_window)) px); use a larger first window")
        push!(lines, line)
    end
    push!(lines, @sprintf("Time: %.2f s", s.seconds))
    return lines
end

"""
    RunState(; output_path = "")

Batch-run state: `output_path` (empty keeps results in memory), `running`,
`progress` `(done, total)`, `status`, `completed` (finished results, in
order), and `started` (`time()` at start). `mode` is the running (or last)
batch's recipe mode, `pairs` its pair count (frame count for tracking), and
`cameras` 1 (planar) or 2 (stereo). A `:sequence` or `:ptv` batch counts
`progress` in pairs; an `:ensemble` batch counts pair correlations
accumulated over every pass (and both cameras for stereo) and finishes with
one result; a `:tracking` batch counts frame steps and finishes with one
`TrackingResult`. Canceling a sequence keeps finished pairs, in memory and
in the output file; canceling an ensemble or tracking run stops at the next
pair or frame step and keeps no result.
"""
struct RunState
    output_path::Observable{String}
    running::Observable{Bool}
    progress::Observable{Tuple{Int,Int}}
    status::Observable{String}
    completed::Observable{Vector{Any}}
    started::Observable{Float64}
    finished_output::Observable{Union{Nothing,String}}
    mode::Observable{Symbol}
    pairs::Observable{Int}
    cameras::Observable{Int}
    cancel::Threads.Atomic{Bool}
    recipe::Base.RefValue{Any}               # the last run's recipe and inputs
    inputs::Base.RefValue{Any}
    options::Base.RefValue{Any}              # ... and its keyword inputs
end

RunState(; output_path::AbstractString = "") =
    RunState(Observable(String(output_path)), Observable(false), Observable((0, 0)),
             Observable(""), Observable(Any[]), Observable(0.0),
             Observable{Union{Nothing,String}}(nothing), Observable(:sequence), Observable(0),
             Observable(1), Threads.Atomic{Bool}(false), Ref{Any}(nothing), Ref{Any}(nothing),
             Ref{Any}((;)))

"""
    start_run!(rs::RunState, recipe, pairs; deliver = f -> f(), spawn = true, options = (;))
    start_run!(rs::RunState, recipe, inputs::Tuple; deliver, spawn, options)

Run `apply_recipe(recipe, pairs)` as a batch, writing to `output_path` when
set (the file also stores the recipe). A `:sequence` recipe appends each
finished pair to `completed` as it arrives; an `:ensemble` recipe appends its
one pooled result when the run finishes. The tuple form runs
`apply_recipe(recipe, inputs...)`, e.g. the stereo `(pairs1, pairs2, dw1, dw2)`;
`options` are further keyword inputs such as per-pair `masks`.
"""
start_run!(rs::RunState, recipe::PIVRecipe, pairs::AbstractVector; kwargs...) =
    start_run!(rs, recipe, (pairs,); kwargs...)

function start_run!(rs::RunState, recipe::PIVRecipe, inputs::Tuple;
                    deliver = f -> f(), spawn::Bool = true, options::NamedTuple = (;))
    rs.running[] && return rs
    pairs = first(inputs)
    isempty(pairs) && (rs.status[] = "no pairs to process"; return rs)
    output = isempty(rs.output_path[]) ? nothing : rs.output_path[]
    ensemble = recipe.mode === :ensemble
    pooled = ensemble || recipe.mode === :tracking      # one result at the end
    cameras = length(inputs) > 1 ? 2 : 1
    rs.cancel[] = false
    rs.completed[] = Any[]
    rs.recipe[] = recipe
    rs.inputs[] = inputs
    rs.options[] = options
    rs.mode[] = recipe.mode
    rs.pairs[] = length(pairs)
    rs.cameras[] = cameras
    rs.progress[] = (0, ensemble ? length(recipe.passes) * length(pairs) * cameras :
                        recipe.mode === :tracking ? length(pairs) - 1 : length(pairs))
    rs.started[] = time()
    rs.finished_output[] = nothing
    rs.status[] = "running…"
    rs.running[] = true
    job = function ()
        progress = (i, n) -> begin
            deliver(() -> (rs.progress[] = (i, n)))
            rs.cancel[] && throw(BatchCancelled())
        end
        keep = r -> deliver(() -> (push!(rs.completed[], r); notify(rs.completed)))
        outcome = try
            if pooled
                keep(apply_recipe(recipe, inputs...; output, progress, options...))
            else
                apply_recipe(recipe, inputs...; output, progress, on_result = (i, r) -> keep(r),
                             options...)
            end
            :done
        catch err
            err isa BatchCancelled ? :canceled : err
        end
        deliver(() -> _finish_run!(rs, output, outcome))
    end
    _run_job(job, spawn)
    return rs
end

function _finish_run!(rs::RunState, output, outcome)
    done, total = rs.progress[]
    n = length(rs.completed[])
    to = output === nothing ? "" : " → $(basename(output))"
    rs.status[] = if outcome === :done
        rs.mode[] === :ensemble ? "done: ensemble of $(rs.pairs[]) pairs" * to :
        rs.mode[] === :tracking ? "done: $(_ntracks(rs)) tracks through $(rs.pairs[]) frames" * to :
        "done: $n pairs" * to
    elseif outcome === :canceled
        rs.mode[] === :ensemble ? "canceled; an ensemble keeps no partial result" :
        rs.mode[] === :tracking ? "canceled; tracking keeps no partial result" :
                                  "canceled after $n of $total pairs"
    else
        "failed: " * _errmsg(outcome)
    end
    rs.finished_output[] = output === nothing || n == 0 ? nothing : output
    rs.running[] = false
    return rs
end

_ntracks(rs::RunState) =
    isempty(rs.completed[]) ? 0 : length(last(rs.completed[]).trajectories)

"""
    run_progress(rs::RunState) -> String

The batch's progress in words: `"3 of 10 pairs"` for a sequence;
`"ensemble of 10 pairs · pass 2 of 3 · 4 of 10 pairs"` for an ensemble (with
the camera for stereo).
"""
function run_progress(rs::RunState)
    done, total = rs.progress[]
    rs.mode[] === :tracking && return "tracking: frame step $done of $total"
    rs.mode[] === :ensemble || return "$done of $total pairs"
    n = rs.pairs[]
    (n > 0 && total >= n) || return "ensemble of $n pairs"
    rounds = total ÷ n
    r = min(done ÷ n + 1, rounds)
    passes = max(rounds ÷ rs.cameras[], 1)
    txt = "ensemble of $n pairs · "
    rs.cameras[] > 1 && (txt *= "camera $((r - 1) ÷ passes + 1) · ")
    return txt * "pass $((r - 1) % passes + 1) of $passes · $(done - (r - 1) * n) of $n pairs"
end

"""
    cancel_run!(rs::RunState)

Stop after the pair in flight. A sequence keeps its finished pairs; an
ensemble keeps no result.
"""
cancel_run!(rs::RunState) = (rs.cancel[] = true; rs)

"""
    run_eta(rs::RunState) -> String

Elapsed time and estimated time remaining while running.
"""
function run_eta(rs::RunState)
    done, total = rs.progress[]
    (rs.running[] && done > 0) || return ""
    el = time() - rs.started[]
    rem = el / done * (total - done)
    fmt(t) = t < 90 ? @sprintf("%.0f s", t) : t < 5400 ? @sprintf("%.0f min", t / 60) : @sprintf("%.1f h", t / 3600)
    return "$(fmt(el)) elapsed · about $(fmt(rem)) left"
end
