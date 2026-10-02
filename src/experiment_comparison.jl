# Scientific recipe comparison. Locators, inputs, environments, and executions
# deliberately remain separate from recipe settings.

"""
    RecipeArraySummary

Compact embedded mask/background content in a [`RecipeChange`](@ref):
`size` (dimension tuple), `element_type` (type name), and `sha256` (full
canonical content digest). The digest includes precision, shape, and
column-major values, using the experiment identity encoding. No source
array or pixel values are retained.
"""
struct RecipeArraySummary
    size::Tuple{Vararg{Int}}
    element_type::String
    sha256::String
end

"""
    RecipeChange

One scientific setting change from [`recipe_diff`](@ref), with `path::String`,
`before`, and `after`. Paths use field names and 1-based sequence indices,
for example `"passes[2].window_size"`. Added/removed items use `missing`,
distinct from an explicitly disabled optional setting (`nothing`). Small
settings are scalars/tuples; nested item snapshots are named tuples; embedded
mask/background arrays become [`RecipeArraySummary`](@ref) values.
"""
struct RecipeChange
    path::String
    before::Any
    after::Any
end

"""
    RecipeDiff

Structured recipe revision comparison: verified `before_id` and `after_id`,
and deterministic ordered `changes::Vector{RecipeChange}`. Supports
`length`, `isempty`, indexing, and iteration over changes. A text/plain
display lists readable paths and compact before/after values. No recipe,
mask, or background payload is retained.
"""
struct RecipeDiff
    before_id::String
    after_id::String
    changes::Vector{RecipeChange}
end

Base.length(diff::RecipeDiff) = length(diff.changes)
Base.isempty(diff::RecipeDiff) = isempty(diff.changes)
Base.getindex(diff::RecipeDiff, i::Integer) = diff.changes[i]
Base.firstindex(diff::RecipeDiff) = firstindex(diff.changes)
Base.lastindex(diff::RecipeDiff) = lastindex(diff.changes)
Base.iterate(diff::RecipeDiff, state...) = iterate(diff.changes, state...)
Base.eltype(::Type{RecipeDiff}) = RecipeChange

Base.:(==)(a::RecipeArraySummary, b::RecipeArraySummary) =
    a.size == b.size && a.element_type == b.element_type && a.sha256 == b.sha256
Base.:(==)(a::RecipeChange, b::RecipeChange) =
    a.path == b.path && isequal(a.before, b.before) && isequal(a.after, b.after)
Base.:(==)(a::RecipeDiff, b::RecipeDiff) =
    a.before_id == b.before_id && a.after_id == b.after_id && a.changes == b.changes
Base.hash(value::RecipeArraySummary, h::UInt) = hash((value.size, value.element_type, value.sha256), h)
Base.hash(value::RecipeChange, h::UInt) = hash((value.path, value.before, value.after), h)
Base.hash(value::RecipeDiff, h::UInt) = hash((value.before_id, value.after_id, Tuple(value.changes)), h)

function Base.show(io::IO, summary::RecipeArraySummary)
    print(io, summary.element_type, " array ", summary.size,
          " sha256=", summary.sha256[1:12], "…")
end
Base.show(io::IO, ::MIME"text/plain", summary::RecipeArraySummary) = show(io, summary)

function Base.show(io::IO, change::RecipeChange)
    print(io, change.path, ": ")
    show(io, change.before)
    print(io, " -> ")
    show(io, change.after)
end
Base.show(io::IO, ::MIME"text/plain", change::RecipeChange) = show(io, change)
Base.show(io::IO, diff::RecipeDiff) = print(io, "RecipeDiff(", length(diff), " change", length(diff) == 1 ? "" : "s", ")")
function Base.show(io::IO, ::MIME"text/plain", diff::RecipeDiff)
    if isempty(diff)
        print(io, "No scientific recipe changes (", diff.before_id[1:12], "…)")
    else
        show(io, diff)
        print(io, " [", diff.before_id[1:12], "… -> ", diff.after_id[1:12], "…]")
        for change in diff
            print(io, '\n', "  ")
            show(io, change)
        end
    end
end

# Stream the existing canonical encoding into SHA in small chunks instead of
# constructing a pixel-size report value or a full canonical array byte buffer.
struct _RecipeDigestIO <: IO
    context::SHA.SHA2_256_CTX
    buffer::IOBuffer
end
function Base.flush(io::_RecipeDigestIO)
    position(io.buffer) == 0 || SHA.update!(io.context, take!(io.buffer))
    nothing
end
function Base.write(io::_RecipeDigestIO, byte::UInt8)
    count = write(io.buffer, byte)
    position(io.buffer) >= 8192 && flush(io)
    count
end
function Base.unsafe_write(io::_RecipeDigestIO, pointer::Ptr{UInt8}, count::UInt)
    written = Base.unsafe_write(io.buffer, pointer, count)
    position(io.buffer) >= 8192 && flush(io)
    written
end
function _recipe_array_summary(array::AbstractArray)
    io = _RecipeDigestIO(SHA.SHA2_256_CTX(), IOBuffer(sizehint = 8192))
    _experiment_canonical(io, array)
    flush(io)
    RecipeArraySummary(size(array), string(eltype(array)), bytes2hex(SHA.digest!(io.context)))
end

function _recipe_comparison_data(recipe::PIVRecipe)
    external = recipe.external_preprocess
    Dict{String,Any}(
        "passes" => [Dict{String,Any}(String(key) =>
            key === :validation ? [_experiment_validator_data(v) for v in pass.validation] :
            getfield(pass, key) for key in fieldnames(PIVParameters)) for pass in recipe.passes],
        "preprocessing" => [Dict{String,Any}("operation" => step.operation,
                            "options" => step.options) for step in recipe.preprocessing],
        "external_preprocess" => external === nothing ? nothing :
            Dict{String,Any}("sha256" => external.sha256, "entrypoint" => external.entrypoint),
        "mask" => recipe.mask,
        "roi" => recipe.roi === nothing ? nothing :
            Dict{String,Any}("rows" => (first(recipe.roi.rows), last(recipe.roi.rows)),
                             "cols" => (first(recipe.roi.cols), last(recipe.roi.cols))),
        "scale" => recipe.scale === nothing ? nothing :
            Dict{String,Any}(String(k) => getfield(recipe.scale, k) for k in fieldnames(PhysicalScale)),
        "backend" => recipe.backend, "image_type" => string(recipe.image_type),
        "threaded" => recipe.threaded, "predictor_smoothing" => recipe.predictor_smoothing,
        "mask_threshold" => recipe.mask_threshold, "uncertainty_backend" => recipe.uncertainty_backend)
end

_recipe_payload_path(path) = path == "mask" || endswith(path, ".background")
_recipe_child_path(path, key) = isempty(path) ? key : path * "." * key

# Only changed values are snapshotted. Recursive snapshots are immutable and
# replace pixel payloads even when an entire preprocessing step is added.
function _recipe_change_value(value, path)
    if value isa AbstractArray && _recipe_payload_path(path)
        return _recipe_array_summary(value)
    elseif value isa AbstractDict
        keys = sort!(collect(Base.keys(value)))
        return NamedTuple{Tuple(Symbol.(keys))}(Tuple(
            _recipe_change_value(value[key], _recipe_child_path(path, key)) for key in keys))
    elseif value isa AbstractVector
        return Tuple(_recipe_change_value(v, path * "[$i]") for (i, v) in enumerate(value))
    else
        return value
    end
end

function _recipe_compare!(changes, before, after, path)
    if before isa AbstractArray && after isa AbstractArray && _recipe_payload_path(path)
        # Element type/shape matter even when numerically equal.
        if eltype(before) !== eltype(after) || size(before) != size(after) || !isequal(before, after)
            push!(changes, RecipeChange(path, _recipe_array_summary(before), _recipe_array_summary(after)))
        end
    elseif before isa AbstractDict && after isa AbstractDict
        for key in sort!(union(collect(keys(before)), collect(keys(after))))
            _recipe_compare!(changes, get(before, key, missing), get(after, key, missing),
                             _recipe_child_path(path, key))
        end
    elseif before isa AbstractVector && after isa AbstractVector && !_recipe_payload_path(path)
        # Small numeric form values such as CLAHE tiles remain readable tuples.
        if all(v -> v isa Number, before) && all(v -> v isa Number, after)
            _recipe_compare!(changes, Tuple(before), Tuple(after), path)
        else
            for i in 1:max(length(before), length(after))
                _recipe_compare!(changes, i <= length(before) ? before[i] : missing,
                                 i <= length(after) ? after[i] : missing, path * "[$i]")
            end
        end
    elseif typeof(before) !== typeof(after) || !isequal(before, after)
        push!(changes, RecipeChange(path, _recipe_change_value(before, path), _recipe_change_value(after, path)))
    end
    nothing
end

"""
    recipe_diff(before::PIVRecipe, after::PIVRecipe) -> RecipeDiff

Compare two valid scientific recipe snapshots without running PIV, accessing
input/script files, or writing files. Both [`recipe_identity`](@ref) values
are checked first; mutated snapshots are rejected even when comparing a
recipe with itself.

Changes have deterministic field paths: dictionary fields are sorted and
pass/preprocessing/validator sequences compared by 1-based index. Reordering
is visible as changes at the affected positions; this is not an edit-distance
or move detector. Small tuples (window sizes, ROI bounds, CLAHE tiles) remain
readable. Embedded masks/backgrounds use content summaries including shape,
precision, and canonical SHA-256; reports retain no array payloads.

Script content digest and entrypoint matter; script locators do not. Equal
recipes with relocated scripts compare equal, and script files need not be
available during comparison. Input identities, run outputs, and software
environments are separate from recipe settings: compare `input_id` explicitly
when comparing experiments. This reports configuration changes, not their
numerical effects on representative image pairs.
"""
function recipe_diff(before::PIVRecipe, after::PIVRecipe)
    before_id = recipe_identity(before)
    after_id = recipe_identity(after)
    changes = RecipeChange[]
    if before_id != after_id
        _recipe_compare!(changes, _recipe_comparison_data(before), _recipe_comparison_data(after), "")
    end
    RecipeDiff(before_id, after_id, changes)
end
