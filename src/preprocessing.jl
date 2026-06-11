# Image preprocessing for PIV: background removal, intensity conditioning, and
# contrast enhancement. All functions take real-valued matrices and return
# Matrix{Float64}, ready for run_piv.

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

Subtract a background image (see [`compute_background`](@ref)), clamping the
result at zero.
"""
function subtract_background(img::AbstractMatrix{<:Real}, background::AbstractMatrix{<:Real})
    size(img) == size(background) ||
        throw(DimensionMismatch("image and background must have the same size"))
    return max.(Float64.(img) .- background, 0.0)
end

"""
    intensity_cap(img; n_sigma=2) -> Matrix{Float64}

Cap pixel intensities at `median + n_sigma * std` (Shavit et al. 2007),
limiting the influence of overexposed particles and reflections on the
correlation.
"""
function intensity_cap(img::AbstractMatrix{<:Real}; n_sigma::Real = 2)
    n_sigma > 0 || throw(ArgumentError("n_sigma must be positive, got $n_sigma"))
    v = vec(Float64.(img))
    cap = median(v) + n_sigma * std(v)
    return min.(Float64.(img), cap)
end

# Separable Gaussian blur with replicated edges (kernel radius 3σ).
function gaussian_blur(img::AbstractMatrix{<:Real}, sigma::Real)
    sigma > 0 || throw(ArgumentError("sigma must be positive, got $sigma"))
    radius = max(1, ceil(Int, 3sigma))
    k = [exp(-x^2 / (2sigma^2)) for x in -radius:radius]
    k ./= sum(k)
    nr, nc = size(img)
    tmp = Matrix{Float64}(undef, nr, nc)
    out = Matrix{Float64}(undef, nr, nc)
    @inbounds for c in 1:nc, r in 1:nr
        s = 0.0
        for (i, x) in enumerate(-radius:radius)
            s += k[i] * img[r, clamp(c + x, 1, nc)]
        end
        tmp[r, c] = s
    end
    @inbounds for c in 1:nc, r in 1:nr
        s = 0.0
        for (i, x) in enumerate(-radius:radius)
            s += k[i] * tmp[clamp(r + x, 1, nr), c]
        end
        out[r, c] = s
    end
    return out
end

"""
    highpass_filter(img; sigma=3) -> Matrix{Float64}

Remove low-frequency background (sheet inhomogeneity, glare) by subtracting a
Gaussian blur of scale `sigma` pixels, clamping at zero. `sigma` should be a
few times the particle image diameter so particles survive the filter.
"""
highpass_filter(img::AbstractMatrix{<:Real}; sigma::Real = 3) =
    max.(Float64.(img) .- gaussian_blur(img, sigma), 0.0)

"""
    clahe(img; tiles=(8, 8), clip_limit=2.0, nbins=256) -> Matrix{Float64}

Contrast-limited adaptive histogram equalization. The image is divided into
`tiles`, each tile's histogram is clipped at `clip_limit` times the uniform
bin count (excess redistributed) and converted to a CDF mapping; pixel values
are remapped with bilinear interpolation between the four surrounding tile
mappings. Output is in `[0, 1]`. Standard preprocessing for unevenly
illuminated PIV recordings.
"""
function clahe(img::AbstractMatrix{<:Real};
               tiles::Tuple{Int,Int} = (8, 8), clip_limit::Real = 2.0, nbins::Int = 256)
    all(>=(1), tiles) || throw(ArgumentError("tiles must be positive, got $tiles"))
    clip_limit >= 1 || throw(ArgumentError("clip_limit must be at least 1, got $clip_limit"))
    nbins >= 2 || throw(ArgumentError("nbins must be at least 2, got $nbins"))
    nr, nc = size(img)
    tr, tc = min(tiles[1], nr), min(tiles[2], nc)
    lo, hi = extrema(img)
    hi > lo || return fill(0.5, nr, nc)
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
    out = Matrix{Float64}(undef, nr, nc)
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
            out[r, c] = (1 - wr) * ((1 - wc) * maps[k, ti0, tj0] + wc * maps[k, ti0, tj1]) +
                        wr * ((1 - wc) * maps[k, ti1, tj0] + wc * maps[k, ti1, tj1])
        end
    end
    return out
end
