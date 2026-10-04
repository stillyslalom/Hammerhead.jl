# Frame list, pairing, and the representative pair the workflow steps show.
# Images load on demand; only the current pair stays cached.

"""
    FrameSet(; files = Any[], pair_mode = :paired)

Frames to analyze, in acquisition order (file paths and/or in-memory
matrices), how they form pairs, and the representative pair the workflow
shows. `pair` is the index of that pair; `shown` is `:a` or `:b`.
"""
struct FrameSet
    files::Observable{Vector{Any}}
    pair_mode::Observable{Symbol}
    pair::Observable{Int}
    shown::Observable{Symbol}
    cache::Dict{Any,Matrix{Float32}}
end

function FrameSet(; files = Any[], pair_mode::Symbol = :paired)
    pair_mode in (:paired, :chained) ||
        throw(ArgumentError("pair_mode must be :paired or :chained, got :$pair_mode"))
    fs = FrameSet(Observable{Vector{Any}}(collect(Any, files)), Observable(pair_mode),
                  Observable(1), Observable(:a), Dict{Any,Matrix{Float32}}())
    onany((_...) -> _clamp_pair!(fs), fs.files, fs.pair_mode)
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

function _frame_image(fs::FrameSet, entry)
    get!(fs.cache, entry) do
        entry isa AbstractMatrix ? Matrix{Float32}(entry) : load_image(Float32, entry)
    end
end

"""
    pair_images(fs::FrameSet) -> Union{Nothing,Tuple{Matrix{Float32},Matrix{Float32}}}

Frames A and B of the representative pair as `Float32` images (cached;
other pairs' images are released).
"""
function pair_images(fs::FrameSet)
    pr = current_pair(fs)
    pr === nothing && return nothing
    a, b = pr[1], pr[2]
    for k in collect(keys(fs.cache))
        (k === a || k === b || isequal(k, a) || isequal(k, b)) || delete!(fs.cache, k)
    end
    return (_frame_image(fs, a), _frame_image(fs, b))
end

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
message describing the first problem.
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

Size `(rows, cols)` of the representative frames, or `nothing`.
"""
function frame_size(fs::FrameSet)
    imgs = try
        pair_images(fs)
    catch
        nothing
    end
    return imgs === nothing ? nothing : size(imgs[1])
end
