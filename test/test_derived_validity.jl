using Hammerhead
using Test

@testset "Derived sampling and region inclusion" begin
    x = [0.0, 1.0]
    y = [0.0, 1.0]
    u = [1.0 NaN; 3.0 4.0]

    # Invalid corners only affect samples to which they contribute.
    sample = Hammerhead._bilinear
    @test sample(x, y, u, 0.0, 0.0) == 1.0
    @test sample(x, y, u, 0.0, 0.5) == 2.0
    @test sample(x, y, u, 1.0, 1.0) == 4.0
    @test sample(x, y, u, 0.5, 1.0) == 3.5
    @test isnan(sample(x, y, u, 0.5, 0.0))
    @test isnan(sample(x, y, u, 0.5, 0.5))
    @test isnan(sample(x, y, u, -0.1, 0.5))

    xd, yd = reverse(x), reverse(y)
    ud = reverse(reverse(u; dims = 1); dims = 2)
    @test sample(xd, yd, ud, 0.0, 0.0) == 1.0
    @test sample(xd, yd, ud, 0.0, 0.5) == 2.0
    @test isnan(sample(xd, yd, ud, 0.5, 0.5))
    @test isnan(sample(xd, yd, ud, 0.5, 0.0))

    function field(xg, yg, uf; mask = falses(size(uf)),
                   outliers = falses(size(uf)))
        z = zeros(size(uf))
        return PIVResult(xg, yg, uf, z, ones(size(uf)), z,
                         fill(NaN, size(uf)), fill(NaN, size(uf)),
                         outliers, mask, PIVParameters())
    end

    profile = extract_profile(field(x, y, u), [(0.0, 0.0), (0.0, 1.0)]; n = 3)
    @test profile.u == [1.0, 2.0, 3.0]
    @test isnan(extract_profile(field(x, y, u), [(0.0, 0.0), (1.0, 0.0)]; n = 3).u[2])

    finite_u = [1.0 2.0; 3.0 4.0]
    corner = falses(2, 2); corner[1, 2] = true
    masked = field(x, y, finite_u; mask = corner)
    @test extract_profile(masked, [(0.0, 0.0), (0.0, 1.0)]; n = 3).u ==
          [1.0, 2.0, 3.0]  # masked corner has zero weight
    @test isnan(extract_profile(masked, [(0.0, 0.0), (1.0, 0.0)]; n = 3).u[2])
    @test isnan(extract_profile(masked, [(0.0, 0.0), (1.0, 0.0)];
                                n = 3, include_invalid = true).u[2])
    flagged = field(x, y, finite_u; outliers = corner)
    @test extract_profile(flagged, [(0.0, 0.0), (0.0, 1.0)]; n = 3).u ==
          [1.0, 2.0, 3.0]
    @test isnan(extract_profile(flagged, [(0.0, 0.0), (1.0, 0.0)]; n = 3).u[2])
    @test extract_profile(flagged, [(0.0, 0.0), (1.0, 0.0)];
                          n = 3, include_invalid = true).u == [1.0, 1.5, 2.0]

    grid = reshape(collect(1.0:9.0), 3, 3)
    exclusion = falses(3, 3); exclusion[2, 2] = true
    flags = falses(3, 3); flags[1, 1] = true
    grid[3, 3] = NaN
    r = field([0.0, 1.0, 2.0], [0.0, 1.0, 2.0], grid;
              mask = exclusion, outliers = flags)
    region = extract_region(r, (0.0, 2.0, 0.0, 2.0))
    expected = trues(3, 3)
    expected[1, 1] = expected[2, 2] = expected[3, 3] = false
    @test propertynames(region) == (:indices, :x, :y, :u, :v, :mask, :included)
    @test region.included === region.mask
    @test region.included == expected
    @test region.indices == findall(expected)
    @test region.u == r.u[region.indices]
    @test region[6] === region.mask  # existing positional access

    with_flags = extract_region(r, (0.0, 2.0, 0.0, 2.0); include_invalid = true)
    @test with_flags.included[1, 1]
    @test !with_flags.included[2, 2] && !with_flags.included[3, 3]
    @test !extract_region(r, (0.0, 0.0, 0.0, 0.0)).included[1, 1]
    @test !any(extract_region(r, (3.0, 4.0, 3.0, 4.0)).included)
end
