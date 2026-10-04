# Frame list, pairing, and the representative pair the workflow steps show.
# Images load on demand; only the current pair stays cached. In a window
# (`spawn[] == true`) the pair loads on a worker task: queries never read a
# file on the GUI thread, they request the pair and report it as loading
# until it is delivered.

"""
    FrameSet(; files = Any[], pair_mode = :paired, spawn = Ref(false), deliver = Ref{Any}(f -> f()))

Frames to analyze, in acquisition order (file paths and/or in-memory
matrices), how they form pairs, and the representative pair the workflow
shows. `pair` is the index of that pair; `shown` is `:a` or `:b`.

With `spawn[] == true`, the representative pair is read on a worker task
and handed to `deliver[]` (a window's GUI-thread queue): until it arrives,
[`pair_images`](@ref) returns `nothing` and `loading` is `true`; `loaded`
counts delivered pairs and `load_error` holds a read failure. Otherwise
images load synchronously when first needed.
"""
struct FrameSet
    files::Observable{Vector{Any}}
    pair_mode::Observable{Symbol}
    pair::Observable{Int}
    shown::Observable{Symbol}
    cache::Dict{Any,Matrix{Float32}}
    loading::Observable{Bool}
    loaded::Observable{Int}
    load_error::Observable{String}
    spawn::Base.RefValue{Bool}
    deliver::Base.RefValue{Any}
    request::Base.RefValue{Any}           # entries of the pending or failed request
    generation::Base.RefValue{Int}
end

function FrameSet(; files = Any[], pair_mode::Symbol = :paired,
                  spawn::Base.RefValue{Bool} = Ref(false),
                  deliver::Base.RefValue{Any} = Ref{Any}(f -> f()))
    pair_mode in (:paired, :chained) ||
        throw(ArgumentError("pair_mode must be :paired or :chained, got :$pair_mode"))
    fs = FrameSet(Observable{Vector{Any}}(collect(Any, files)), Observable(pair_mode),
                  Observable(1), Observable(:a), Dict{Any,Matrix{Float32}}(),
                  Observable(false), Observable(0), Observable(""), spawn, deliver,
                  Ref{Any}(nothing), Ref(0))
    onany((_...) -> _clamp_pair!(fs), fs.files, fs.pair_mode)
    onany((_...) -> _pair_changed!(fs), fs.files, fs.pair_mode, fs.pair)
    return fs
end

# A new representative pair: forget the previous request (a pending load
# for another pair is dropped on arrival) and, in a window, start loading.
function _pair_changed!(fs::FrameSet)
    pr, req = current_pair(fs), fs.request[]
    pr !== nothing && req !== nothing && _same_entry(pr[1], req[1]) &&
        _same_entry(pr[2], req[2]) && return fs     # still the requested pair
    fs.generation[] += 1
    fs.request[] = nothing
    isempty(fs.load_error[]) || (fs.load_error[] = "")
    fs.loading[] && (fs.loading[] = false)
    fs.spawn[] && pair_images(fs)
    return fs
end

function Base.show(io::IO, fs::FrameSet)
    print(io, "FrameSet($(length(fs.files[])) frames, :$(fs.pair_mode[]), pair $(fs.pair[]))")
end

"""
    add_files!(fs::FrameSet, entries)

Append frames (paths or matrices) in acquisition order.
"""
add_files!(fs::FrameSet, entries) = (append!(fs.files[], entries); notify(fs.files); fs)

"""
    clear_files!(fs::FrameSet)

Remove all frames.
"""
clear_files!(fs::FrameSet) = (empty!(fs.files[]); empty!(fs.cache); notify(fs.files); fs)

"""
    set_pair_mode!(fs::FrameSet, mode)

Pair frames `1-2, 3-4, …` (`:paired`) or `1-2, 2-3, …` (`:chained`).
"""
function set_pair_mode!(fs::FrameSet, mode::Symbol)
    mode in (:paired, :chained) ||
        throw(ArgumentError("pair_mode must be :paired or :chained, got :$mode"))
    fs.pair_mode[] == mode || (fs.pair_mode[] = mode)
    return fs
end

"""
    frame_pairs(fs::FrameSet) -> Vector

All pairs for the current frames and pairing rule (`Hammerhead.image_pairs`;
throws on an odd frame count in `:paired` mode).
"""
frame_pairs(fs::FrameSet) = image_pairs(fs.files[]; mode = fs.pair_mode[])

"""
    npairs(fs::FrameSet) -> Int

Number of pairs, or 0 when the frames do not form pairs.
"""
npairs(fs::FrameSet) = (prs = _pairs_or_nothing(fs); prs === nothing ? 0 : length(prs))

_pairs_or_nothing(fs) = try
    frame_pairs(fs)
catch
    nothing
end

function _clamp_pair!(fs::FrameSet)
    n = npairs(fs)
    i = clamp(fs.pair[], 1, max(n, 1))
    i == fs.pair[] || (fs.pair[] = i)
    return fs
end

"""
    select_pair!(fs::FrameSet, i)

Make pair `i` (clamped to the available pairs) the representative pair.
"""
function select_pair!(fs::FrameSet, i::Integer)
    i = clamp(Int(i), 1, max(npairs(fs), 1))
    i == fs.pair[] || (fs.pair[] = i)
    return fs
end

"""
    show_frame!(fs::FrameSet, which)

Show frame `:a` or `:b` of the representative pair.
"""
function show_frame!(fs::FrameSet, which::Symbol)
    which in (:a, :b) || throw(ArgumentError("which must be :a or :b, got :$which"))
    fs.shown[] == which || (fs.shown[] = which)
    return fs
end

"""
    current_pair(fs::FrameSet) -> Union{Nothing,Tuple}

The frame entries of the representative pair, or `nothing` without pairs.
"""
function current_pair(fs::FrameSet)
    prs = _pairs_or_nothing(fs)
    (prs === nothing || isempty(prs)) && return nothing
    return prs[clamp(fs.pair[], 1, length(prs))]
end

_read_frame(entry) = entry isa AbstractMatrix ? Matrix{Float32}(entry) : load_image(Float32, entry)

_same_entry(x, y) = x === y || (x isa AbstractString && y isa AbstractString && x == y)

function _keep_only!(fs::FrameSet, a, b)
    for k in collect(keys(fs.cache))
        (_same_entry(k, a) || _same_entry(k, b)) || delete!(fs.cache, k)
    end
    return fs
end

"""
    pair_images(fs::FrameSet) -> Union{Nothing,Tuple{Matrix{Float32},Matrix{Float32}}}

Frames A and B of the representative pair as `Float32` images (cached;
other pairs' images are released), or `nothing` without pairs. With
`spawn[]`, a pair that is not cached yet is requested from a worker task
and `nothing` is returned meanwhile (see `loading`).
"""
function pair_images(fs::FrameSet)
    pr = current_pair(fs)
    pr === nothing && return nothing
    a, b = pr[1], pr[2]
    if haskey(fs.cache, a) && haskey(fs.cache, b)
        return (fs.cache[a], fs.cache[b])
    end
    fs.spawn[] && return (_request_pair!(fs, a, b); nothing)
    _keep_only!(fs, a, b)
    imga = get!(() -> _read_frame(a), fs.cache, a)
    imgb = get!(() -> _read_frame(b), fs.cache, b)
    return (imga, imgb)
end

# Load a pair on a worker; the GUI thread stores it when it is delivered.
function _request_pair!(fs::FrameSet, a, b)
    req = fs.request[]
    req !== nothing && _same_entry(req[1], a) && _same_entry(req[2], b) && return fs
    g = (fs.generation[] += 1)
    fs.request[] = (a, b)
    fs.loading[] || (fs.loading[] = true)
    deliver = fs.deliver[]
    job = function ()
        out = _try_job(() -> (_read_frame(a), _read_frame(b)))
        deliver(() -> _finish_pair!(fs, g, a, b, out))
    end
    _run_job(job, true)
    return fs
end

function _finish_pair!(fs::FrameSet, g::Int, a, b, out)
    g == fs.generation[] || return fs        # the pair changed meanwhile
    if out.err === nothing
        _keep_only!(fs, a, b)
        fs.cache[a], fs.cache[b] = out.value
        fs.request[] = nothing
        fs.loading[] = false
        fs.loaded[] += 1
    else
        # keep `request`, so the failed pair is not requested again
        fs.load_error[] = "cannot read pair $(fs.pair[]): " * _errmsg(out.err)
        fs.loading[] = false
    end
    return fs
end

"""
    pair_loading(fs::FrameSet) -> Bool

Whether the representative pair is being read on a worker task.
"""
pair_loading(fs::FrameSet) = fs.loading[]

"""
    shown_image(fs::FrameSet) -> Union{Nothing,Matrix{Float32}}

The frame of the representative pair selected by `shown`.
"""
function shown_image(fs::FrameSet)
    imgs = pair_images(fs)
    imgs === nothing && return nothing
    return fs.shown[] === :a ? imgs[1] : imgs[2]
end

"""
    frames_problem(fs::FrameSet) -> Union{Nothing,String}

`nothing` when the frames form pairs of equally sized images, otherwise a
message describing the first problem (with `spawn[]`, also while the
representative pair loads).
"""
function frames_problem(fs::FrameSet)
    isempty(fs.files[]) && return "add frames to analyze"
    prs = try
        frame_pairs(fs)
    catch err
        return _errmsg(err)
    end
    isempty(prs) && return "these frames form no pairs"
    imgs = try
        pair_images(fs)
    catch err
        return "cannot read pair $(fs.pair[]): " * _errmsg(err)
    end
    if imgs === nothing                      # requested from a worker
        return isempty(fs.load_error[]) ? "loading pair $(fs.pair[])…" : fs.load_error[]
    end
    size(imgs[1]) == size(imgs[2]) ||
        return "pair $(fs.pair[]) has frames of different sizes $(size(imgs[1])) and $(size(imgs[2]))"
    return nothing
end

"""
    frames_summary(fs::FrameSet) -> String

One line: frame count, pair count, and image size.
"""
function frames_summary(fs::FrameSet)
    n = length(fs.files[])
    n == 0 && return "no frames"
    np = npairs(fs)
    txt = "$n frame" * (n == 1 ? "" : "s") * " · $np pair" * (np == 1 ? "" : "s")
    imgs = try
        pair_images(fs)
    catch
        nothing
    end
    imgs === nothing || (txt *= " · $(size(imgs[1], 2))×$(size(imgs[1], 1)) px")
    return txt
end

"""
    frame_size(fs::FrameSet) -> Union{Nothing,Dims{2}}

Size `(rows, cols)` of the representative frames, or `nothing` (also while
they load on a worker task).
"""
function frame_size(fs::FrameSet)
    imgs = try
        pair_images(fs)
    catch
        nothing
    end
    return imgs === nothing ? nothing : size(imgs[1])
end
