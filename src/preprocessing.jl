# Image preprocessing for PIV: background removal, intensity conditioning, and
# contrast enhancement. The mutating versions (`subtract_background!`,
# `intensity_cap!`, `highpass_filter!`, `clahe!`) are the core implementations
# and operate on floating-point buffers (Float32 or Float64) so chained
# pipelines don't allocate at each step; the non-mutating names are thin
# copying wrappers accepting any real-valued matrix and returning a matrix of
# `float(eltype(img))` (integers promote to Float64), ready for run_piv.

# Copy `img` into a fresh floating-point buffer the mutating functions can own.
float_copy(img::AbstractMatrix{<:Real}) = float(eltype(img)).(img)

"""
    compute_background(images; method=:min) -> Matrix{Float64}

Estimate a static background from an iterable of equally sized images:
`:min` (pixel-wise minimum — robust for sparse particles over a static
background) or `:mean`.
"""
function compute_background(images; method::Symbol = :min)
    method in (:min, :mean) ||
        throw(ArgumentError("method must be :min or :mean, got :$method"))
    state = iterate(images)
    state === nothing && throw(ArgumentError("at least one image is required"))
    first_img, rest = state
    bg = Float64.(first_img)
    n = 1
    while (state = iterate(images, rest)) !== nothing
        img, rest = state
        size(img) == size(bg) ||
            throw(DimensionMismatch("all images must have the same size"))
        method === :min ? (bg .= min.(bg, img)) : (bg .+= img)
        n += 1
    end
    method === :mean && (bg ./= n)
    return bg
end

"""
    subtract_background(img, background) -> Matrix{Float64}
    subtract_background!(img, background) -> img

Subtract a background image (see [`compute_background`](@ref)), clamping the
result at zero. The mutating version overwrites the floating-point `img`
without allocating.
"""
function subtract_background!(img::AbstractMatrix{<:AbstractFloat}, background::AbstractMatrix{<:Real})
    size(img) == size(background) ||
        throw(DimensionMismatch("image and background must have the same size"))
    img .= max.(img .- background, zero(eltype(img)))
    return img
end

"""
    subtract_background(img, background) -> Matrix

Allocating form of [`subtract_background!`](@ref): returns a new
floating-point matrix and leaves `img` untouched.
"""
subtract_background(img::AbstractMatrix{<:Real}, background::AbstractMatrix{<:Real}) =
    subtract_background!(float_copy(img), background)

"""
    intensity_cap(img; n_sigma=2) -> Matrix{Float64}
    intensity_cap!(img; n_sigma=2) -> img

Cap pixel intensities at `median + n_sigma * std` (Shavit et al. 2007),
limiting the influence of overexposed particles and reflections on the
correlation. The mutating version overwrites the floating-point `img`.
"""
function intensity_cap!(img::AbstractMatrix{<:AbstractFloat}; n_sigma::Real = 2)
    n_sigma > 0 || throw(ArgumentError("n_sigma must be positive, got $n_sigma"))
    v = vec(img)
    cap = median(v) + n_sigma * std(v)
    img .= min.(img, cap)
    return img
end

"""
    intensity_cap(img; n_sigma=2) -> Matrix

Allocating form of [`intensity_cap!`](@ref): returns a new floating-point
matrix and leaves `img` untouched.
"""
intensity_cap(img::AbstractMatrix{<:Real}; n_sigma::Real = 2) =
    intensity_cap!(float_copy(img); n_sigma)

# Separable Gaussian blur with replicated edges (kernel radius 3σ), in the
# image's own precision.
function gaussian_blur(img::AbstractMatrix{<:AbstractFloat}, sigma::Real)
    sigma > 0 || throw(ArgumentError("sigma must be positive, got $sigma"))
    l = 2 * max(1, ceil(Int, 3sigma)) + 1
    return imfilter(eltype(img), img, KernelFactors.gaussian((sigma, sigma), (l, l)), "replicate")
end

"""
    highpass_filter(img; sigma=3) -> Matrix{Float64}
    highpass_filter!(img; sigma=3) -> img

Remove low-frequency background (sheet inhomogeneity, glare) by subtracting a
Gaussian blur of scale `sigma` pixels, clamping at zero. `sigma` should be a
few times the particle image diameter so particles survive the filter. The
mutating version overwrites the floating-point `img`; the blur itself requires
one internal buffer.
"""
function highpass_filter!(img::AbstractMatrix{<:AbstractFloat}; sigma::Real = 3)
    blur = gaussian_blur(img, sigma)
    img .= max.(img .- blur, zero(eltype(img)))
    return img
end

"""
    highpass_filter(img; sigma=3) -> Matrix

Allocating form of [`highpass_filter!`](@ref): returns a new floating-point
matrix and leaves `img` untouched.
"""
highpass_filter(img::AbstractMatrix{<:Real}; sigma::Real = 3) =
    highpass_filter!(float_copy(img); sigma)

# Kept in-house rather than delegating to ImageContrastAdjustment's
# AdaptiveEqualization: that implementation silently `imresize`s images whose
# dimensions don't divide evenly into blocks, and the interpolation round-trip
# perturbs subpixel particle intensity distributions.
"""
    clahe(img; tiles=(8, 8), clip_limit=2.0, nbins=256) -> Matrix{Float64}
    clahe!(img; tiles=(8, 8), clip_limit=2.0, nbins=256) -> img

Contrast-limited adaptive histogram equalization. The image is divided into
`tiles`, each tile's histogram is clipped at `clip_limit` times the uniform
bin count (excess redistributed) and converted to a CDF mapping; pixel values
are remapped with bilinear interpolation between the four surrounding tile
mappings. Output is in `[0, 1]`. Standard preprocessing for unevenly
illuminated PIV recordings. The mutating version remaps the floating-point
`img` in place.
"""
function clahe!(img::AbstractMatrix{<:AbstractFloat};
                tiles::Tuple{Int,Int} = (8, 8), clip_limit::Real = 2.0, nbins::Int = 256)
    all(>=(1), tiles) || throw(ArgumentError("tiles must be positive, got $tiles"))
    clip_limit >= 1 || throw(ArgumentError("clip_limit must be at least 1, got $clip_limit"))
    nbins >= 2 || throw(ArgumentError("nbins must be at least 2, got $nbins"))
    nr, nc = size(img)
    tr, tc = min(tiles[1], nr), min(tiles[2], nc)
    lo, hi = extrema(img)
    hi > lo || return fill!(img, 0.5)
    scale = (nbins - 1) / (hi - lo)
    binof(x) = 1 + round(Int, (x - lo) * scale)

    redges = round.(Int, range(0, nr; length = tr + 1))
    cedges = round.(Int, range(0, nc; length = tc + 1))
    maps = Array{Float64}(undef, nbins, tr, tc)
    hist = zeros(Float64, nbins)
    for tj in 1:tc, ti in 1:tr
        fill!(hist, 0.0)
        rows = (redges[ti] + 1):redges[ti + 1]
        cols = (cedges[tj] + 1):cedges[tj + 1]
        npx = length(rows) * length(cols)
        for c in cols, r in rows
            hist[binof(img[r, c])] += 1
        end
        limit = clip_limit * npx / nbins
        excess = 0.0
        for k in 1:nbins
            if hist[k] > limit
                excess += hist[k] - limit
                hist[k] = limit
            end
        end
        hist .+= excess / nbins
        cum = 0.0
        for k in 1:nbins
            cum += hist[k]
            maps[k, ti, tj] = cum / npx
        end
    end

    rcent = [(redges[i] + redges[i + 1] + 1) / 2 for i in 1:tr]
    ccent = [(cedges[j] + cedges[j + 1] + 1) / 2 for j in 1:tc]
    # Each output pixel depends only on the input value at the same position
    # (the tile mappings are already built), so the remap is safely in-place.
    @inbounds for c in 1:nc
        tj1 = clamp(searchsortedfirst(ccent, c), 1, tc)
        tj0 = max(tj1 - 1, 1)
        wc = ccent[tj1] == ccent[tj0] ? 0.0 :
             clamp((c - ccent[tj0]) / (ccent[tj1] - ccent[tj0]), 0.0, 1.0)
        for r in 1:nr
            ti1 = clamp(searchsortedfirst(rcent, r), 1, tr)
            ti0 = max(ti1 - 1, 1)
            wr = rcent[ti1] == rcent[ti0] ? 0.0 :
                 clamp((r - rcent[ti0]) / (rcent[ti1] - rcent[ti0]), 0.0, 1.0)
            k = binof(img[r, c])
            img[r, c] = (1 - wr) * ((1 - wc) * maps[k, ti0, tj0] + wc * maps[k, ti0, tj1]) +
                        wr * ((1 - wc) * maps[k, ti1, tj0] + wc * maps[k, ti1, tj1])
        end
    end
    return img
end

"""
    clahe(img; tiles=(8, 8), clip_limit=2.0, nbins=256) -> Matrix

Allocating form of [`clahe!`](@ref): returns a new floating-point matrix and
leaves `img` untouched.
"""
clahe(img::AbstractMatrix{<:Real}; kwargs...) = clahe!(float_copy(img); kwargs...)

"""Stretch the `low` and `high` intensity percentiles to `[0,1]`, clipping outside."""
function percentile_stretch!(img::AbstractMatrix{<:AbstractFloat}; low::Real=1, high::Real=99)
    0 <= low < high <= 100 || throw(ArgumentError("percentiles must satisfy 0 <= low < high <= 100"))
    lo,hi = quantile(vec(img), [low/100,high/100])
    if !(hi > lo); return fill!(img, eltype(img)(0.5)); end
    img .= clamp.((img .- lo) ./ (hi-lo), zero(eltype(img)), one(eltype(img)))
    img
end
percentile_stretch(img::AbstractMatrix{<:Real}; kwargs...) = percentile_stretch!(float_copy(img); kwargs...)

"""Invert intensities about their finite range (`lo + hi - value`)."""
function invert_image!(img::AbstractMatrix{<:AbstractFloat})
    lo,hi=extrema(img); img .= lo+hi .- img; img
end
invert_image(img::AbstractMatrix{<:Real}) = invert_image!(float_copy(img))

"""
    local_variance_normalize!(img; sigma=3, epsilon=1e-3)

Subtract a Gaussian local mean and divide by the local standard deviation.
This is useful for multiplicative illumination/contrast variation; unlike
CLAHE it preserves a linear standardized response, but it can amplify noise
in flat areas, controlled by `epsilon`.
"""
function local_variance_normalize!(img::AbstractMatrix{<:AbstractFloat}; sigma::Real=3,
                                   epsilon::Real=1e-3)
    sigma > 0 || throw(ArgumentError("sigma must be positive")); epsilon > 0 || throw(ArgumentError("epsilon must be positive"))
    mu=gaussian_blur(img,sigma); second=gaussian_blur(img.^2,sigma)
    img .= (img .- mu) ./ sqrt.(max.(second .- mu.^2, zero(eltype(img))) .+ epsilon^2)
    img
end
local_variance_normalize(img::AbstractMatrix{<:Real}; kwargs...) = local_variance_normalize!(float_copy(img); kwargs...)
