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

# A number as text-field text: as printed when that is short, otherwise
# rounded to 6 significant digits (the stored value stays exact until the
# field is edited).
function display_number(x::Real)
    x isa Integer && return string(x)
    s = string(x)
    return length(s) <= 8 ? s : @sprintf("%.6g", x)
end

# Parse a positive-number text field entry.
function _parse_positive(str::AbstractString, what::AbstractString)
    v = tryparse(Float64, strip(str))
    (v === nothing || !(isfinite(v) && v > 0)) &&
        throw(ArgumentError("$what must be a positive number, got \"$str\""))
    return v
end
