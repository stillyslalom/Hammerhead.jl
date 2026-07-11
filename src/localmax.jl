# Shared local-maximum predicate for regional peak finding. Plateau ties are
# deterministic: the last point in column-major order wins.
@inline function is_local_maximum(A::AbstractMatrix, r::Int, c::Int)
    nr, nc = size(A)
    I0 = A[r, c]
    isnan(I0) && return false
    lin0 = r + (c - 1) * nr
    @inbounds for jj in max(1, c - 1):min(nc, c + 1),
                  ii in max(1, r - 1):min(nr, r + 1)
        (ii == r && jj == c) && continue
        In = A[ii, jj]
        if ii + (jj - 1) * nr < lin0
            In > I0 && return false
        else
            In >= I0 && return false
        end
    end
    return true
end
