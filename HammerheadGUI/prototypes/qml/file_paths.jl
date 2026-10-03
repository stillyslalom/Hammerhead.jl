# Qt values are consumed on the owner thread, before any queued action.
module FilePaths
using QML

const PURPOSES = ("record", "result", "output", "history")

function file_url(path::AbstractString)
    localpath = String(path)
    isempty(localpath) && throw(ArgumentError("choose a local file path"))
    occursin('\0', localpath) && throw(ArgumentError("file path contains a NUL character"))
    isabspath(localpath) || throw(ArgumentError("choose an absolute local file path"))
    Sys.iswindows() && isempty(first(splitdrive(localpath))) &&
        throw(ArgumentError("choose a file path with a drive or UNC share"))
    QML.QUrlFromLocalFile(localpath)
end

function local_path(url)
    applicable(QML.toLocalFile, url) || throw(ArgumentError("choose a Qt local file URL"))
    path = String(QML.toLocalFile(url))
    canonical = file_url(path)
    # Qt has no exposed scheme/query/fragment accessors in this binding.
    # Its own round trip rejects URL metadata and non-file schemes without
    # decoding reserved characters ourselves or losing a UNC hostname.
    String(QML.toString(canonical)) == String(QML.toString(url)) ||
        throw(ArgumentError("choose a local file URL without query or fragment"))
    path
end

function initial_folder(draft::AbstractString)
    path = String(draft)
    local_absolute = isabspath(path) && (!Sys.iswindows() || !isempty(first(splitdrive(path))))
    folder = !isempty(path) && !occursin('\0', path) && local_absolute ? dirname(path) : pwd()
    file_url(folder)
end

mutable struct DialogState
    generation::Int
    purpose::String
    active::Bool
    closed::Bool
end
DialogState() = DialogState(0, "", false, false)

function begin!(state::DialogState, purpose; allowed=true)
    label = String(purpose)
    label in PURPOSES || throw(ArgumentError("unknown file picker purpose"))
    (!allowed || state.closed || state.active) && return 0
    state.generation = Base.checked_add(state.generation, 1)
    state.purpose = label
    state.active = true
    state.generation
end

matches(state, token, purpose) = state.active && !state.closed &&
    token == state.generation && String(purpose) == state.purpose

function reject!(state::DialogState, token)
    state.active && token == state.generation || return false
    state.active = false
    true
end

function accept!(state::DialogState, token, purpose, url; allowed=true)
    matches(state, token, purpose) || return nothing
    try
        allowed || return nothing
        local_path(url)
    finally
        reject!(state, token)
    end
end

function close!(state::DialogState)
    state.closed = true
    state.active = false
    nothing
end
end
