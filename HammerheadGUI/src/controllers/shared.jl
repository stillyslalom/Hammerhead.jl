# Small helpers shared by the controllers: error messages, text-field
# parsing, and the batch-cancellation signal.

"""
    BatchCancelled()

Thrown from a batch driver's `progress` callback to stop between frame
pairs. Finished pairs stay in memory and in the output file.
"""
struct BatchCancelled <: Exception end

# First line of an exception's message, for status lines.
_errmsg(err) = first(split(sprint(showerror, err), '\n'))

# Run a background computation, capturing its outcome as `(; value, err)`.
_try_job(job) = try
    (; value = job(), err = nothing)
catch err
    (; value = nothing, err)
end

# Parse a positive-number text field entry.
function _parse_positive(str::AbstractString, what::AbstractString)
    v = tryparse(Float64, strip(str))
    (v === nothing || !(isfinite(v) && v > 0)) &&
        throw(ArgumentError("$what must be a positive number, got \"$str\""))
    return v
end

"""
    parse_schedule(str) -> Vector{Int}

Parse a window-schedule entry: positive integers separated by commas and/or
spaces, e.g. `"64, 32, 32"`. Throws `ArgumentError` on anything else.
"""
function parse_schedule(str::AbstractString)
    tokens = split(str, r"[,\s]+"; keepempty = false)
    isempty(tokens) && throw(ArgumentError("empty window schedule"))
    sizes = Int[]
    for t in tokens
        n = tryparse(Int, t)
        (n === nothing || n <= 0) &&
            throw(ArgumentError("window sizes must be positive integers, got \"$t\""))
        push!(sizes, n)
    end
    return sizes
end

"""
Effort levels of the stereo batch form: `:custom` (manual schedule) or a
core effort preset.
"""
const EFFORT_LEVELS = (:custom, :low, :medium, :high)
