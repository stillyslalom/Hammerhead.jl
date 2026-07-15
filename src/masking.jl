# Mask construction utilities. Analysis masks are plain Bool matrices the
# size of the images, `true` marking excluded pixels — see `mask` in run_piv.

"""
    polygon_mask(size, vertices) -> BitMatrix

Rasterize a polygon into a `(rows, cols)` mask, `true` inside the polygon.
`vertices` is a vector of `(x, y)` corners in pixel coordinates (`x` along
columns, `y` along rows), in order, without repeating the first vertex.
Combine multiple exclusion regions with `.|` and pass the result as `mask`
to [`run_piv`](@ref).

```julia
mask = polygon_mask(size(img), [(120, 40), (480, 90), (450, 300), (100, 260)])
result = run_piv(imgA, imgB, passes; mask)
```
"""
function polygon_mask(sz::Dims{2}, vertices::AbstractVector)
    length(vertices) >= 3 ||
        throw(ArgumentError("a polygon needs at least 3 vertices, got $(length(vertices))"))
    nr, nc = sz
    xs = [Float64(v[1]) for v in vertices]
    ys = [Float64(v[2]) for v in vertices]
    mask = falses(nr, nc)
    n = length(vertices)
    crossings = Float64[]
    for r in 1:nr
        y = Float64(r)
        empty!(crossings)
        for k in 1:n
            x1, y1 = xs[k], ys[k]
            x2, y2 = xs[mod1(k + 1, n)], ys[mod1(k + 1, n)]
            # Half-open interval so a scanline through a vertex counts once.
            ((y1 <= y < y2) || (y2 <= y < y1)) || continue
            push!(crossings, x1 + (y - y1) / (y2 - y1) * (x2 - x1))
        end
        sort!(crossings)
        for k in 1:2:(length(crossings) - 1)
            c1 = max(1, ceil(Int, crossings[k]))
            c2 = min(nc, floor(Int, crossings[k + 1]))
            c1 <= c2 && (mask[r, c1:c2] .= true)
        end
    end
    return mask
end

"""
    grow_mask(mask, radius=1) -> BitMatrix
    shrink_mask(mask, radius=1) -> BitMatrix

Grow or shrink an exclusion mask by a circular pixel radius. These helpers
preserve Hammerhead's `true = excluded` convention. A zero radius copies the
input.
"""
function grow_mask(mask::AbstractMatrix{Bool}, radius::Integer = 1)
    radius >= 0 || throw(ArgumentError("radius must be nonnegative, got $radius"))
    out = falses(size(mask))
    offsets = [(dr, dc) for dr in -radius:radius, dc in -radius:radius
               if dr * dr + dc * dc <= radius * radius]
    nr, nc = size(mask)
    for c in 1:nc, r in 1:nr
        mask[r, c] || continue
        for (dr, dc) in offsets
            rr, cc = r + dr, c + dc
            1 <= rr <= nr && 1 <= cc <= nc && (out[rr, cc] = true)
        end
    end
    return out
end

function shrink_mask(mask::AbstractMatrix{Bool}, radius::Integer = 1)
    radius >= 0 || throw(ArgumentError("radius must be nonnegative, got $radius"))
    radius == 0 && return BitMatrix(mask)
    padded = falses(size(mask) .+ 2radius)
    padded[radius+1:end-radius, radius+1:end-radius] .= mask
    eroded = .!grow_mask(.!padded, radius)
    return BitMatrix(eroded[radius+1:end-radius, radius+1:end-radius])
end

function _local_variance(A::AbstractMatrix{<:Real}, radius::Int)
    radius >= 1 || throw(ArgumentError("radius must be at least 1, got $radius"))
    nr, nc = size(A)
    S = zeros(Float64, nr + 1, nc + 1)
    S2 = similar(S)
    for c in 1:nc, r in 1:nr
        x = Float64(A[r, c])
        S[r + 1, c + 1] = x + S[r, c + 1] + S[r + 1, c] - S[r, c]
        S2[r + 1, c + 1] = x * x + S2[r, c + 1] + S2[r + 1, c] - S2[r, c]
    end
    variance = zeros(Float64, nr, nc)
    for c in 1:nc, r in 1:nr
        r1, r2 = max(1, r - radius), min(nr, r + radius)
        c1, c2 = max(1, c - radius), min(nc, c + radius)
        n = (r2 - r1 + 1) * (c2 - c1 + 1)
        s = S[r2 + 1, c2 + 1] - S[r1, c2 + 1] - S[r2 + 1, c1] + S[r1, c1]
        s2 = S2[r2 + 1, c2 + 1] - S2[r1, c2 + 1] - S2[r2 + 1, c1] + S2[r1, c1]
        variance[r, c] = max(0.0, s2 / n - (s / n)^2)
    end
    return variance
end

"""
    automatic_mask(image; method=:intensity, threshold=:auto, side=:high,
                   radius=5, grow=0) -> BitMatrix
    automatic_mask(imageA, imageB; kwargs...) -> BitMatrix

Construct an image-derived exclusion mask. `method` is `:intensity`,
`:contrast` (local standard deviation), or `:edge` (gradient magnitude).
`threshold=:auto` selects a robust tail quantile; a numeric threshold uses
the image's intensity units. The pair method returns the union of both frame
masks, the normal convention for moving geometry.
"""
function automatic_mask(image::AbstractMatrix{<:Real};
                        method::Symbol = :intensity, threshold = :auto,
                        side::Symbol = method === :contrast ? :low : :high,
                        radius::Integer = 5, grow::Integer = 0)
    method in (:intensity, :contrast, :edge) ||
        throw(ArgumentError("method must be :intensity, :contrast, or :edge, got :$method"))
    side in (:low, :high) || throw(ArgumentError("side must be :low or :high, got :$side"))
    A = Float64.(image)
    score = if method === :intensity
        A
    elseif method === :contrast
        sqrt.(_local_variance(A, Int(radius)))
    else
        nr, nc = size(A)
        G = zeros(Float64, nr, nc)
        for c in 1:nc, r in 1:nr
            gx = (A[r, min(nc, c + 1)] - A[r, max(1, c - 1)]) / 2
            gy = (A[min(nr, r + 1), c] - A[max(1, r - 1), c]) / 2
            G[r, c] = hypot(gx, gy)
        end
        G
    end
    t = if threshold === :auto
        q = method === :contrast && side === :low ? 0.05 :
            side === :low ? 0.01 : method === :contrast ? 0.95 : 0.99
        quantile(vec(score), q)
    elseif threshold isa Real
        Float64(threshold)
    else
        throw(ArgumentError("threshold must be :auto or a number, got $threshold"))
    end
    mask = side === :high ? score .>= t : score .<= t
    return grow == 0 ? BitMatrix(mask) : grow_mask(mask, grow)
end

function automatic_mask(imageA::AbstractMatrix{<:Real}, imageB::AbstractMatrix{<:Real}; kwargs...)
    size(imageA) == size(imageB) || throw(DimensionMismatch("pair images must have equal size"))
    return automatic_mask(imageA; kwargs...) .| automatic_mask(imageB; kwargs...)
end
