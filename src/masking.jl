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
