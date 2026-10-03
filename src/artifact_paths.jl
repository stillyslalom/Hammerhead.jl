# Serialized locators are provenance. Classify them without touching the host
# filesystem or interpreting another platform's roots relative to the cwd.
function _artifact_locator_style(path::AbstractString)
    p=String(path)
    (isempty(p) || occursin('\0',p)) && return :invalid
    if startswith(p,"\\\\?\\")
        tail=p[5:end]
        occursin(r"^[A-Za-z]:[\\/]",tail) && return :windows
        startswith(uppercase(tail),"UNC\\") || return :invalid
        tail=tail[5:end]
        parts=split(tail,r"[\\/]";keepempty=true)
        return length(parts)>=2 && !isempty(parts[1]) && !isempty(parts[2]) ? :windows : :invalid
    end
    occursin(r"^[A-Za-z]:[\\/]",p) && return :windows
    if startswith(p,"\\\\") || startswith(p,"//")
        parts=split(p[3:end],r"[\\/]";keepempty=true)
        return length(parts)>=2 && !isempty(parts[1]) && !isempty(parts[2]) && !(parts[1] in ("?",".")) ? :windows : :invalid
    end
    startswith(p,"/") && return :posix
    :relative
end
_artifact_absolute_locator(path) = path isa AbstractString && _artifact_locator_style(path) in (:windows,:posix)
# Leading // is conservatively UNC, even on POSIX: v1 has no dialect tag.
# A receiving-host POSIX path can be made unambiguous with a single leading /.
_artifact_local_locator(path) = _artifact_absolute_locator(path) &&
    _artifact_locator_style(path)==(Sys.iswindows() ? :windows : :posix)

# Explicit writer/reader arguments refer to local files. Relative paths are
# allowed, but foreign absolute, drive-relative and current-drive roots are not.
function _artifact_local_path(path::AbstractString)
    p=String(path);style=_artifact_locator_style(p)
    style===:invalid && throw(ArgumentError("invalid local artifact path"))
    if style in (:windows,:posix)
        _artifact_local_locator(p) || throw(ArgumentError("foreign artifact locator requires an explicit receiving-host path"))
    else
        (occursin(r"^[A-Za-z]:",p) || startswith(p,"\\")) &&
            throw(ArgumentError("drive-relative/current-drive artifact paths are ambiguous"))
    end
    abspath(p)
end
function _artifact_relative_locator(path)
    path isa AbstractString && _artifact_locator_style(path)===:relative &&
        !occursin(r"^[A-Za-z]:",path) && !startswith(path,"\\")
end
function _artifact_resolve_relative(parent,path)
    _artifact_relative_locator(path) || throw(ArgumentError("artifact association must be relative"))
    !Sys.iswindows() && occursin('\\',path) &&
        throw(ArgumentError("foreign CSV separator dialect requires explicit csv_path"))
    _artifact_local_path(joinpath(parent,path))
end
_artifact_local_protected_paths(locators) = sort!(unique(String[p for p in locators if _artifact_local_locator(p)]))

# Existing and prospective local aliases: resolve the nearest existing ancestor
# so a symlinked parent cannot hide a not-yet-created destination. Windows
# comparison is conservatively case-insensitive even in case-sensitive folders.
function _artifact_prospective_path(path;allow_unavailable_root::Bool=false)
    ancestor=normpath(_artifact_local_path(path))
    Sys.iswindows() && any(p->endswith(p,".") || endswith(p," "),splitpath(ancestor)) &&
        throw(ArgumentError("Windows trailing-dot/space path components are ambiguous"))
    original=ancestor;remaining=String[]
    while !ispath(ancestor) && !islink(ancestor)
        parent=dirname(ancestor)
        if parent==ancestor
            allow_unavailable_root && return Sys.iswindows() ? lowercase(original) : original
            throw(ArgumentError("cannot resolve artifact output ancestor"))
        end
        push!(remaining,basename(ancestor));ancestor=parent
    end
    resolved=try realpath(ancestor) catch;throw(ArgumentError("cannot resolve artifact path ancestor"));end
    canonical=normpath(joinpath(resolved,reverse(remaining)...))
    Sys.iswindows() ? lowercase(canonical) : canonical
end
function _artifact_alias(a,b;allow_unavailable_other::Bool=false)
    left,right=_artifact_local_path(a),_artifact_local_path(b)
    # Validate prospective spellings before a shortcut, including trailing dots.
    ca=_artifact_prospective_path(left)
    cb=_artifact_prospective_path(right;allow_unavailable_root=allow_unavailable_other)
    ca==cb || (ispath(left) && ispath(right) && Base.samefile(left,right))
end
