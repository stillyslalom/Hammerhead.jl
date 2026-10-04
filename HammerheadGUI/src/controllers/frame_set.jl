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

`pattern_dir` and `pattern` describe frames by folder and file-name pattern
(see [`set_frame_pattern!`](@ref)); `pattern_matches` lists the matching
files and [`add_matching!`](@ref) appends them.
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
    pattern_dir::Observable{String}
    pattern::Observable{String}
    pattern_matches::Observable{Vector{String}}
    pattern_error::Observable{String}
end

function FrameSet(; files = Any[], pair_mode::Symbol = :paired,
                  spawn::Base.RefValue{Bool} = Ref(false),
                  deliver::Base.RefValue{Any} = Ref{Any}(f -> f()))
    pair_mode in (:paired, :chained) ||
        throw(ArgumentError("pair_mode must be :paired or :chained, got :$pair_mode"))
    fs = FrameSet(Observable{Vector{Any}}(collect(Any, files)), Observable(pair_mode),
                  Observable(1), Observable(:a), Dict{Any,Matrix{Float32}}(),
                  Observable(false), Observable(0), Observable(""), spawn, deliver,
                  Ref{Any}(nothing), Ref(0), Observable(""), Observable("*.tif"),
                  Observable(String[]), Observable(""))
    onany((_...) -> _update_matches!(fs), fs.pattern_dir, fs.pattern)
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
    matching_files(dir, pattern) -> Vector{String}

The files in `dir` whose names match the glob `pattern` (`*` matches any
run of characters, `?` one character; case-insensitive on Windows), in
natural order: numbers in the names compare by value, so `frame_2` precedes
`frame_10`.
"""
function matching_files(dir::AbstractString, pattern::AbstractString)
    isdir(dir) || throw(ArgumentError("there is no folder \"$dir\""))
    rx = _glob_regex(pattern)
    names = filter(n -> occursin(rx, n) && isfile(joinpath(dir, n)), readdir(dir))
    return [joinpath(dir, n) for n in sort!(names; by = _natural_key)]
end

function _glob_regex(pattern::AbstractString)
    io = IOBuffer()
    print(io, '^')
    for c in pattern
        c == '*' ? print(io, ".*") : c == '?' ? print(io, '.') :
        c in ".^\$+()[]{}|\\" ? print(io, '\\', c) : print(io, c)
    end
    print(io, '$')
    return Regex(String(take!(io)), Sys.iswindows() ? "i" : "")
end

# Natural sort key: digit runs compare by value, other text case-insensitively.
_natural_key(name::AbstractString) =
    [isdigit(m.match[1]) ? (0, something(tryparse(Int, m.match), typemax(Int)), "") :
                           (1, 0, lowercase(m.match)) for m in eachmatch(r"\d+|\D+", name)]

"""
    set_frame_pattern!(fs::FrameSet, dir, pattern)

Describe frames by folder and glob pattern; `fs.pattern_matches` then lists
the matching files (`fs.pattern_error` says why there are none).
"""
function set_frame_pattern!(fs::FrameSet, dir::AbstractString, pattern::AbstractString)
    d, p = String(strip(dir)), String(strip(pattern))
    fs.pattern_dir[] == d || (fs.pattern_dir[] = d)
    fs.pattern[] == p || (fs.pattern[] = p)
    return fs
end

function _update_matches!(fs::FrameSet)
    d, p = fs.pattern_dir[], fs.pattern[]
    files, err = if isempty(d)
        String[], ""
    elseif isempty(p)
        String[], "enter a file-name pattern"
    else
        try
            m = matching_files(d, p)
            m, isempty(m) ? "no files in the folder match \"$p\"" : ""
        catch e
            String[], _errmsg(e)
        end
    end
    fs.pattern_matches[] = files
    fs.pattern_error[] == err || (fs.pattern_error[] = err)
    return fs
end

"""
    add_matching!(fs::FrameSet) -> Int

Append the files matching the folder and pattern (in natural order);
returns how many were added.
"""
function add_matching!(fs::FrameSet)
    _update_matches!(fs)                       # the folder may have changed
    files = fs.pattern_matches[]
    isempty(files) && throw(ArgumentError(isempty(fs.pattern_error[]) ? "choose a folder" :
                                          fs.pattern_error[]))
    add_files!(fs, files)
    return length(files)
end

"""
    infer_pattern(path_a, path_b) -> (dir, pattern)

A folder and glob pattern for a recording from two of its frames (for
example the frames of the first pair): digit runs that differ between the
two names, or that have at least three digits (frame counters), become `*`;
differing text keeps its common beginning and end around a `*`; the rest is
kept. `A001_1.tif`/`A001_2.tif` gives `A*_*.tif`,
`cam1_00001.tif`/`cam1_00002.tif` gives `cam1_*.tif`, and
`run_0001_a.tif`/`run_0001_b.tif` gives `run_*_*.tif`.
"""
function infer_pattern(a::AbstractString, b::AbstractString)
    na, nb = basename(a), basename(b)
    ra = [m.match for m in eachmatch(r"\d+|\D+", na)]
    rb = [m.match for m in eachmatch(r"\d+|\D+", nb)]
    pattern = if length(ra) == length(rb) &&
                 all(((x, y),) -> isdigit(x[1]) == isdigit(y[1]), zip(ra, rb))
        join(isdigit(x[1]) ? (x != y || length(x) >= 3 ? "*" : x) : _common_ends(x, y)
             for (x, y) in zip(ra, rb))
    else                                       # different structure
        _common_ends(na, nb)
    end
    return (dirname(a), replace(pattern, r"\*+" => "*"))
end

# `x` when equal to `y`, otherwise their common beginning and end around `*`.
function _common_ends(x::AbstractString, y::AbstractString)
    x == y && return String(x)
    cx, cy = collect(x), collect(y)
    m = min(length(cx), length(cy))
    p = 0
    while p < m && cx[p+1] == cy[p+1]
        p += 1
    end
    s = 0
    while s < m - p && cx[end-s] == cy[end-s]
        s += 1
    end
    return String(cx[1:p]) * "*" * String(cx[end-s+1:end])
end

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
