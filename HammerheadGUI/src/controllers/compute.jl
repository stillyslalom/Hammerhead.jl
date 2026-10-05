# Where the analysis runs: the CPU, or a GPU through the core's device
# backends. A device package (CUDA, AMDGPU) is loaded on demand, the first time
# the GPU is switched on; the core's extension then registers its backend.

"""
GPU backends the windows can use, in order of preference, with the device
package each needs: `(:cuda, "CUDA")`, `(:amdgpu, "AMDGPU")`.
"""
const GPU_BACKENDS = ((:cuda, "CUDA"), (:amdgpu, "AMDGPU"))

"""
    gpu_packages() -> Vector{Symbol}

The GPU backends whose device package is installed (found in the active
environments), loaded or not.
"""
gpu_packages() = [b for (b, name) in GPU_BACKENDS if Base.find_package(name) !== nothing]

function _gpu_package(b::Symbol)
    for (k, name) in GPU_BACKENDS
        k === b && return name
    end
    return String(b)
end

"""
    use_gpu!(wf::AbstractWorkflow, on::Bool; spawn = wf.spawn[]) -> wf

Run tests and batches on a GPU (`true`) or the CPU (`false`). Switching on
loads the first installed device package (CUDA, then AMDGPU) that reports a
working device, on a worker task with `spawn`; `passes.gpu_loading` is `true`
meanwhile and `passes.gpu_status` reports the outcome. The GPU covers PIV
modes; particle analysis stays on the CPU.
"""
function use_gpu!(wf::AbstractWorkflow, on::Bool; spawn::Bool = wf.spawn[])
    pe = wf.passes
    if !on
        set_backend!(pe, :cpu)
        pe.gpu_status[] = ""
        return wf
    end
    pe.gpu_loading[] && return wf
    candidates = gpu_packages()
    if isempty(candidates)
        pe.gpu_status[] = "no GPU package is installed: add CUDA or AMDGPU to the environment"
        return wf
    end
    pe.gpu_loading[] = true
    pe.gpu_status[] = "loading " * join(map(_gpu_package, candidates), " or ") * "…"
    deliver = wf.deliver[]
    job = function ()
        out = _try_job(() -> _first_working_gpu(candidates))
        deliver(() -> _finish_gpu!(pe, out, candidates))
    end
    _run_job(job, spawn)
    return wf
end

function _first_working_gpu(candidates)
    for b in candidates
        try
            Base.require(Main, Symbol(_gpu_package(b)))
        catch
            continue                       # a broken install: try the next package
        end
        Base.invokelatest(backend_available, b) && return b
    end
    return nothing
end

function _finish_gpu!(pe, out, candidates)
    pe.gpu_loading[] = false
    if out.err !== nothing
        pe.gpu_status[] = "cannot load the GPU package: " * _errmsg(out.err)
    elseif out.value === nothing
        pe.gpu_status[] = join(map(_gpu_package, candidates), " and ") *
                          (length(candidates) == 1 ? " is" : " are") *
                          " installed but found no working GPU"
    else
        set_backend!(pe, out.value)
        pe.gpu_status[] = "running on the GPU (" * _gpu_package(out.value) * ")"
    end
    return pe
end

"""
    set_backend!(pe::PassesEditor, backend::Symbol)

Run on `backend` (`:cpu`, or a backend the core has registered, such as
`:cuda`, `:amdgpu`, or the hardware-free `:ka`) without loading anything;
[`use_gpu!`](@ref) loads a device package first.
"""
function set_backend!(pe::PassesEditor, backend::Symbol)
    backend === :cpu || Base.invokelatest(backend_available, backend) ||
        throw(ArgumentError("backend :$backend is not available in this session"))
    pe.backend[] == backend || (pe.backend[] = backend)
    return pe
end

# The backend keyword of apply_recipe: PIV modes only (particle analysis runs
# on the CPU).
function _backend_kw(wf::AbstractWorkflow)
    b = wf.passes.backend[]
    (b === :cpu || _particle_mode(wf.passes.mode[])) && return (;)
    return (; backend = b)
end

"""
    gpu_problem(wf::AbstractWorkflow) -> Union{Nothing,String}

Why the current pass schedule cannot run on a GPU, in the Passes page's terms
(e.g. "the 2-D Gaussian subpixel fit is CPU-only"), or `nothing`. Before a
device package is loaded the hardware-free `:ka` backend answers: the GPU
backends share its kernels and option scope.
"""
function gpu_problem(wf::AbstractWorkflow)
    b = wf.passes.backend[]
    passes = wf.passes.passes[]
    msg = Base.invokelatest(backend_problem, b === :cpu ? :ka : b, passes)
    msg === nothing && return nothing
    for p in passes
        setting = _cpu_only_setting(p)
        setting === nothing || return setting * " is CPU-only"
    end
    return replace(msg, r";? ?use backend = :cpu" => "")
end

# The Passes-page name of a setting the GPU backends do not implement.
function _cpu_only_setting(p::PIVParameters)
    p.search_area_size == p.window_size || return "a search area larger than the window"
    p.subpixel_method in (:gauss3, :gauss9) || return "the 2-D Gaussian subpixel fit"
    p.image_interpolation === :cubic || return "linear image interpolation"
    p.predictor_interpolation === :linear || return "cubic predictor interpolation"
    return nothing
end

# Why the pass schedule cannot run on the selected GPU backend, or nothing.
function _backend_problem(wf::AbstractWorkflow)
    isempty(_backend_kw(wf)) && return nothing
    msg = gpu_problem(wf)
    return msg === nothing ? nothing : "on the GPU: " * msg * "; switch the GPU off"
end
