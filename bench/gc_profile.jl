# Allocation/GC profile for the Case E sequence workload.
#
# Run from the package root, for example:
#   julia --project=. --threads=4 bench/gc_profile.jl --pairs=5
#
# The default workload mirrors the common smoke profile:
#   frames = sort(readdir(Case E Camera_01, join=true))
#   pairs = image_pairs(frames; mode=:chained)
#   passes = multipass_parameters([128, 64, 32, 16])
#   run_piv_sequence(pairs[1:5], passes; progress=false)
#
# Outputs land in bench/profile-output/ by default and are gitignored.

using Dates
using Profile
using Serialization

if "--help" in ARGS
    println("""
    Usage:
      julia --project=. --threads=4 bench/gc_profile.jl [options]

    Options:
      --camera-dir=PATH       Case E Camera_01 directory (default: cases/.../Camera_01)
      --pairs=N               Number of chained pairs to profile (default: 5)
      --warmup-pairs=N        Number of pairs for warmup before profiling (default: 1)
      --windows=LIST          Comma-separated window sizes (default: 128,64,32,16)
      --sample-rate=X         Profile.Allocs sample rate, 0 < X <= 1 (default: 0.1)
      --image-type=Float64    Float64 or Float32 frame loading (default: Float64)
      --progress=true|false   Include ProgressMeter updates in the profile (default: false)
      --output-dir=PATH       Directory for reports (default: bench/profile-output)
      --raw=true|false        Serialize raw Profile.Allocs data as .jls (default: true)
      --stdlib-reports=true|false
                              Write Profile.Allocs flat/tree reports (default: true)
      --help                  Show this message
    """)
    exit(0)
end

using Hammerhead

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
const DEFAULT_CAMERA_DIR = joinpath(PROJECT_ROOT, "cases", "4th_PIV-Challenge_Case_E",
                                    "E_Particle_Images", "Camera_01")
const DEFAULT_OUTPUT_DIR = joinpath(@__DIR__, "profile-output")

function usage()
    println("""
    Usage:
      julia --project=. --threads=4 bench/gc_profile.jl [options]

    Options:
      --camera-dir=PATH       Case E Camera_01 directory (default: cases/.../Camera_01)
      --pairs=N               Number of chained pairs to profile (default: 5)
      --warmup-pairs=N        Number of pairs for warmup before profiling (default: 1)
      --windows=LIST          Comma-separated window sizes (default: 128,64,32,16)
      --sample-rate=X         Profile.Allocs sample rate, 0 < X <= 1 (default: 0.1)
      --image-type=Float64    Float64 or Float32 frame loading (default: Float64)
      --progress=true|false   Include ProgressMeter updates in the profile (default: false)
      --output-dir=PATH       Directory for reports (default: bench/profile-output)
      --raw=true|false        Serialize raw Profile.Allocs data as .jls (default: true)
      --stdlib-reports=true|false
                              Write Profile.Allocs flat/tree reports (default: true)
      --help                  Show this message
    """)
end

function parse_bool(s::AbstractString)
    v = lowercase(strip(s))
    v in ("true", "yes", "1", "on") && return true
    v in ("false", "no", "0", "off") && return false
    throw(ArgumentError("expected true/false, got $s"))
end

function parse_image_type(s::AbstractString)
    s == "Float64" && return Float64
    s == "Float32" && return Float32
    throw(ArgumentError("--image-type must be Float64 or Float32, got $s"))
end

function parse_windows(s::AbstractString)
    windows = parse.(Int, split(s, ","))
    isempty(windows) && throw(ArgumentError("--windows must not be empty"))
    all(>(0), windows) || throw(ArgumentError("--windows entries must be positive"))
    return windows
end

function parse_args(args)
    opts = Dict{String,Any}(
        "camera-dir" => DEFAULT_CAMERA_DIR,
        "pairs" => 5,
        "warmup-pairs" => 1,
        "windows" => [128, 64, 32, 16],
        "sample-rate" => 0.1,
        "image-type" => Float64,
        "progress" => false,
        "output-dir" => DEFAULT_OUTPUT_DIR,
        "raw" => true,
        "stdlib-reports" => true,
    )
    for arg in args
        arg == "--help" && (usage(); exit(0))
        startswith(arg, "--") || throw(ArgumentError("unexpected argument $arg"))
        keyval = split(arg[3:end], "=", limit = 2)
        length(keyval) == 2 || throw(ArgumentError("expected --key=value, got $arg"))
        key, val = keyval
        haskey(opts, key) || throw(ArgumentError("unknown option --$key"))
        if key in ("pairs", "warmup-pairs")
            opts[key] = parse(Int, val)
        elseif key == "windows"
            opts[key] = parse_windows(val)
        elseif key == "sample-rate"
            opts[key] = parse(Float64, val)
        elseif key == "image-type"
            opts[key] = parse_image_type(val)
        elseif key in ("progress", "raw", "stdlib-reports")
            opts[key] = parse_bool(val)
        else
            opts[key] = val
        end
    end
    opts["pairs"] >= 1 || throw(ArgumentError("--pairs must be positive"))
    opts["warmup-pairs"] >= 0 || throw(ArgumentError("--warmup-pairs must be nonnegative"))
    0 < opts["sample-rate"] <= 1 ||
        throw(ArgumentError("--sample-rate must satisfy 0 < sample-rate <= 1"))
    return opts
end

function format_bytes(n::Real)
    n = Float64(n)
    for (suffix, scale) in (("GiB", 2.0^30), ("MiB", 2.0^20), ("KiB", 2.0^10))
        n >= scale && return string(round(n / scale, digits = 2), " ", suffix)
    end
    return string(round(Int, n), " B")
end

function display_frame(frame)
    file = String(frame.file)
    shown = if startswith(file, "@Hammerhead")
        file
    elseif isabspath(file)
        try
            relpath(normpath(file), PROJECT_ROOT)
        catch
            file
        end
    else
        file
    end
    return "$(shown):$(frame.line); $(frame.func)"
end

pathcase(path::AbstractString) = Sys.iswindows() ? lowercase(path) : String(path)

function project_frame(frame)
    raw = String(frame.file)
    startswith(raw, "@Hammerhead") && return true
    file = try
        pathcase(normpath(raw))
    catch
        return false
    end
    root = pathcase(PROJECT_ROOT)
    return startswith(file, root) &&
           !endswith(file, pathcase(joinpath("bench", "gc_profile.jl")))
end

function first_project_frame(stacktrace)
    for frame in stacktrace
        project_frame(frame) && return display_frame(frame)
    end
    return "[outside Hammerhead frames]"
end

function add_sample!(dict, key, bytes)
    old = get(dict, key, (bytes = 0, count = 0))
    dict[key] = (bytes = old.bytes + bytes, count = old.count + 1)
    return dict
end

function print_aggregate(io, title, dict; limit = 25)
    println(io)
    println(io, title)
    println(io, repeat("-", length(title)))
    rows = sort!(collect(dict), by = row -> row[2].bytes, rev = true)
    println(io, lpad("sampled bytes", 14), "  ", lpad("samples", 8), "  site/type")
    println(io, lpad("=============", 14), "  ", lpad("=======", 8), "  =========")
    for (key, stats) in Iterators.take(rows, limit)
        println(io, lpad(format_bytes(stats.bytes), 14), "  ",
                lpad(string(stats.count), 8), "  ", key)
    end
end

function write_summary(io, opts, timed, allocs)
    sample_rate = opts["sample-rate"]
    sampled_bytes = sum(a.size for a in allocs.allocs)
    by_site = Dict{String,NamedTuple{(:bytes, :count),Tuple{Int,Int}}}()
    by_type = Dict{String,NamedTuple{(:bytes, :count),Tuple{Int,Int}}}()
    for alloc in allocs.allocs
        add_sample!(by_site, first_project_frame(alloc.stacktrace), alloc.size)
        add_sample!(by_type, string(alloc.type), alloc.size)
    end

    println(io, "Hammerhead Case E allocation/GC profile")
    println(io, "Generated:        ", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
    println(io, "Julia:            ", VERSION)
    println(io, "Threads:          ", Threads.nthreads())
    println(io, "Camera directory: ", opts["camera-dir"])
    println(io, "Pairs:            ", opts["pairs"])
    println(io, "Warmup pairs:     ", opts["warmup-pairs"])
    println(io, "Windows:          ", join(opts["windows"], ", "))
    println(io, "Image type:       ", opts["image-type"])
    println(io, "Progress meter:   ", opts["progress"])
    println(io, "Sample rate:      ", sample_rate)
    println(io)
    println(io, "Run counters")
    println(io, "------------")
    println(io, "Elapsed:          ", round(timed.time, digits = 4), " s")
    println(io, "Allocated:        ", format_bytes(timed.bytes))
    println(io, "GC time:          ", round(timed.gctime, digits = 4), " s (",
            round(100 * timed.gctime / max(timed.time, eps()), digits = 2), "%)")
    println(io, "Lock conflicts:   ", timed.lock_conflicts)
    println(io, "Compile time:     ", round(timed.compile_time, digits = 4), " s")
    println(io, "Recompile time:   ", round(timed.recompile_time, digits = 4), " s")
    println(io, "GC allocd:        ", format_bytes(timed.gcstats.allocd))
    println(io, "GC pool allocs:   ", timed.gcstats.poolalloc)
    println(io, "GC big allocs:    ", timed.gcstats.bigalloc)
    println(io, "GC full sweeps:   ", timed.gcstats.full_sweep)
    println(io, "GC pause:         ", round(timed.gcstats.pause / 1e9, digits = 4), " s")
    println(io)
    println(io, "Allocation profiler")
    println(io, "--------------------")
    println(io, "Samples:          ", length(allocs.allocs))
    println(io, "Sampled bytes:    ", format_bytes(sampled_bytes))
    println(io, "Estimated bytes:  ", format_bytes(sampled_bytes / sample_rate))
    println(io, "Note: @timed allocated bytes are exact; sampled/estimated bytes come from Profile.Allocs.")

    print_aggregate(io, "Top sampled allocation sites", by_site)
    print_aggregate(io, "Top sampled allocation types", by_type)
end

function run_profile(opts)
    camera_dir = String(opts["camera-dir"])
    isdir(camera_dir) || throw(ArgumentError("camera directory does not exist: $camera_dir"))
    frames = sort(readdir(camera_dir; join = true))
    length(frames) >= opts["pairs"] + 1 ||
        throw(ArgumentError("need at least $(opts["pairs"] + 1) frames, found $(length(frames))"))

    pairs = image_pairs(frames; mode = :chained)
    selected = pairs[1:opts["pairs"]]
    passes = multipass_parameters(opts["windows"])
    image_type = opts["image-type"]
    progress = opts["progress"]

    if opts["warmup-pairs"] > 0
        nwarm = min(opts["warmup-pairs"], length(selected))
        println("Warmup: profiling excluded, $nwarm pair(s)")
        run_piv_sequence(selected[1:nwarm], passes; progress = false, image_type)
    end

    GC.gc()
    Profile.Allocs.clear()
    println("Profiling: $(length(selected)) pair(s), sample_rate=$(opts["sample-rate"])")
    timed = @timed begin
        Profile.Allocs.@profile sample_rate=opts["sample-rate"] begin
            run_piv_sequence(selected, passes; progress, image_type)
        end
        nothing
    end
    allocs = Profile.Allocs.fetch()

    outdir = String(opts["output-dir"])
    mkpath(outdir)
    stamp = Dates.format(now(), dateformat"yyyymmdd_HHMMSS")
    summary_path = joinpath(outdir, "gc_profile_$stamp.txt")
    flat_path = joinpath(outdir, "allocs_flat_$stamp.txt")
    tree_path = joinpath(outdir, "allocs_tree_$stamp.txt")
    raw_path = joinpath(outdir, "allocs_$stamp.jls")

    open(summary_path, "w") do io
        write_summary(io, opts, timed, allocs)
    end
    if opts["stdlib-reports"]
        open(flat_path, "w") do io
            Profile.Allocs.print(io, allocs; format = :flat, sortedby = :count)
        end
        open(tree_path, "w") do io
            Profile.Allocs.print(io, allocs; format = :tree, sortedby = :count,
                                 maxdepth = 40)
        end
    end
    if opts["raw"]
        open(raw_path, "w") do io
            serialize(io, allocs)
        end
    end

    println("Wrote summary: ", summary_path)
    if opts["stdlib-reports"]
        println("Wrote flat allocation profile: ", flat_path)
        println("Wrote tree allocation profile: ", tree_path)
    end
    opts["raw"] && println("Wrote raw allocation data: ", raw_path)
    println("Allocated: ", format_bytes(timed.bytes),
            ", GC: ", round(100 * timed.gctime / max(timed.time, eps()), digits = 2),
            "%, lock conflicts: ", timed.lock_conflicts)
end

try
    run_profile(parse_args(ARGS))
catch err
    err isa InterruptException && rethrow()
    println(stderr, "ERROR: ", sprint(showerror, err))
    println(stderr)
    usage()
    exit(1)
end
