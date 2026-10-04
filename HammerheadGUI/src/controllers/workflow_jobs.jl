# Background jobs of a workflow window: testing the representative pair and
# running the batch. Work runs on a worker thread (`spawn = true`); every
# observable update is handed to `deliver`, which a GUI shell points at its
# event-loop queue so observers only ever run on the GUI thread. With
# `spawn = false` and the default `deliver`, jobs run inline (tests, scripts).

_run_job(job, spawn::Bool) = spawn ? errormonitor(Threads.@spawn job()) : job()

"""
    PairTest()

The latest test of the current settings on the representative pair:
`result` (a `PIVResult`), the `recipe` and `pair` it was computed with,
`seconds` taken, the `previous` test's summary for comparison, and
`running`/`status`.
"""
struct PairTest
    result::Observable{Union{Nothing,PIVResult}}
    recipe::Observable{Union{Nothing,PIVRecipe}}
    pair::Observable{Int}
    seconds::Observable{Float64}
    previous::Observable{Union{Nothing,NamedTuple}}
    running::Observable{Bool}
    status::Observable{String}
end

PairTest() = PairTest(Observable{Union{Nothing,PIVResult}}(nothing),
                      Observable{Union{Nothing,PIVRecipe}}(nothing), Observable(0),
                      Observable(0.0), Observable{Union{Nothing,NamedTuple}}(nothing),
                      Observable(false), Observable(""))

"""
    start_test!(pt::PairTest, recipe, pairs, label; deliver = f -> f(), spawn = true)

Run `apply_recipe(recipe, pairs)` (one pair, or several for an ensemble
recipe) and store the result. `label` is the representative pair index.
"""
function start_test!(pt::PairTest, recipe::PIVRecipe, pairs::AbstractVector, label::Integer;
                     deliver = f -> f(), spawn::Bool = true)
    pt.running[] && return pt
    pt.running[] = true
    pt.status[] = recipe.mode === :ensemble ?
        "testing ensemble of $(length(pairs)) pairs…" : "testing pair $label…"
    job = function ()
        t0 = time()
        outcome = try
            r = apply_recipe(recipe, pairs; progress = false)
            (; result = r isa AbstractVector ? first(r) : r, seconds = time() - t0, err = nothing)
        catch err
            (; result = nothing, seconds = time() - t0, err)
        end
        deliver(() -> _finish_test!(pt, recipe, Int(label), outcome))
    end
    _run_job(job, spawn)
    return pt
end

function _finish_test!(pt::PairTest, recipe, label, outcome)
    if outcome.err === nothing
        pt.result[] === nothing || (pt.previous[] = test_summary(pt.result[], pt.recipe[], pt.seconds[]))
        pt.recipe[] = recipe
        pt.pair[] = label
        pt.seconds[] = outcome.seconds
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
`masked`), `valid_fraction`, median `peak_ratio`, median `sigma` (px, `NaN`
without uncertainty), the largest valid displacement `max_displacement` and
`quarter_window` (¼ of the final window; displacements beyond it are hard to
correlate reliably), and `seconds`.
"""
function test_summary(r::PIVResult, recipe::PIVRecipe, seconds::Real)
    n = length(r.u)
    masked = count(r.mask)
    flagged = count(r.outliers .& .!r.mask)
    good = .!(r.mask .| r.outliers) .& isfinite.(r.u) .& isfinite.(r.v)
    valid = count(good)
    med(xs) = isempty(xs) ? NaN : _median(xs)
    peak = med(filter(isfinite, Float64.(r.peak_ratio[good])))
    sig = med(filter(isfinite, Float64.(hypot.(r.uncertainty_u[good], r.uncertainty_v[good]))))
    dmax = valid == 0 ? NaN : maximum(Float64.(hypot.(r.u[good], r.v[good])))
    final = last(recipe.passes)
    return (; vectors = n, valid, flagged, masked,
            valid_fraction = n - masked == 0 ? NaN : valid / (n - masked),
            peak_ratio = peak, sigma = sig, max_displacement = dmax,
            quarter_window = minimum(final.window_size) / 4, seconds = Float64(seconds))
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
    delta(f, fmt) = previous === nothing || !isfinite(previous[f]) || !isfinite(s[f]) ? "" :
                    " (" * fmt(s[f] - previous[f]) * ")"
    signed_pct(d) = @sprintf("%+.1f pts", 100d)
    signed(d) = @sprintf("%+.2f", d)
    lines = ["Valid vectors: $(s.valid) of $(s.vectors - s.masked) ($(pct(s.valid_fraction)))" *
                 delta(:valid_fraction, signed_pct),
             "Flagged: $(s.flagged) · masked: $(s.masked)",
             "Median peak ratio: $(num(s.peak_ratio))" * delta(:peak_ratio, signed)]
    isfinite(s.sigma) && push!(lines, "Median uncertainty: $(num(s.sigma)) px" * delta(:sigma, signed))
    if isfinite(s.max_displacement)
        line = "Largest displacement: $(num(s.max_displacement)) px"
        s.max_displacement > s.quarter_window &&
            (line *= " — above ¼ of the final window ($(num(s.quarter_window)) px); use a larger first window")
        push!(lines, line)
    end
    push!(lines, @sprintf("Time: %.2f s", s.seconds))
    return lines
end

"""
    RunState(; output_path = "")

Batch-run state: `output_path` (empty keeps results in memory), `running`,
`progress` `(done, total)`, `status`, `completed` (finished results, in
order), and `started` (`time()` at start). Cancelling keeps finished pairs,
in memory and in the output file.
"""
struct RunState
    output_path::Observable{String}
    running::Observable{Bool}
    progress::Observable{Tuple{Int,Int}}
    status::Observable{String}
    completed::Observable{Vector{Any}}
    started::Observable{Float64}
    finished_output::Observable{Union{Nothing,String}}
    cancel::Threads.Atomic{Bool}
end

RunState(; output_path::AbstractString = "") =
    RunState(Observable(String(output_path)), Observable(false), Observable((0, 0)),
             Observable(""), Observable(Any[]), Observable(0.0),
             Observable{Union{Nothing,String}}(nothing), Threads.Atomic{Bool}(false))

"""
    start_run!(rs::RunState, recipe, pairs; deliver = f -> f(), spawn = true)

Run `apply_recipe(recipe, pairs)` as a batch, writing to `output_path` when
set. Each finished pair is appended to `completed` as it arrives.
"""
function start_run!(rs::RunState, recipe::PIVRecipe, pairs::AbstractVector;
                    deliver = f -> f(), spawn::Bool = true)
    rs.running[] && return rs
    recipe.mode === :sequence ||
        (rs.status[] = "this window runs per-pair sequences; ensemble runs are not available yet"; return rs)
    isempty(pairs) && (rs.status[] = "no pairs to process"; return rs)
    output = isempty(rs.output_path[]) ? nothing : rs.output_path[]
    rs.cancel[] = false
    rs.completed[] = Any[]
    rs.progress[] = (0, length(pairs))
    rs.started[] = time()
    rs.finished_output[] = nothing
    rs.status[] = "running…"
    rs.running[] = true
    job = function ()
        progress = (i, n) -> begin
            deliver(() -> (rs.progress[] = (i, n)))
            rs.cancel[] && throw(BatchCancelled())
        end
        on_result = (i, r) -> deliver(() -> (push!(rs.completed[], r); notify(rs.completed)))
        outcome = try
            apply_recipe(recipe, pairs; output, progress, on_result)
            :done
        catch err
            err isa BatchCancelled ? :cancelled : err
        end
        deliver(() -> _finish_run!(rs, output, outcome))
    end
    _run_job(job, spawn)
    return rs
end

function _finish_run!(rs::RunState, output, outcome)
    done, total = rs.progress[]
    n = length(rs.completed[])
    rs.status[] = outcome === :done ? "done: $n pairs" * (output === nothing ? "" : " → $(basename(output))") :
                  outcome === :cancelled ? "cancelled after $n of $total pairs" :
                  "failed: " * _errmsg(outcome)
    rs.finished_output[] = output === nothing || n == 0 ? nothing : output
    rs.running[] = false
    return rs
end

"""
    cancel_run!(rs::RunState)

Stop after the pair in flight; finished pairs are kept.
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
