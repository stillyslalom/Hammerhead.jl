using Hammerhead
using Test
using Statistics
using TiffImages

# PIV Challenge (2001) case A: tip vortex behind a transport aircraft model,
# with loss of seeding in the vortex core — see reference_images/A/readmeA.txt.
@testset "Reference Images (PIV Challenge A)" begin
    dir = joinpath(@__DIR__, "reference_images", "A")
    imgA = Float64.(TiffImages.load(joinpath(dir, "A001_1.tif")))
    imgB = Float64.(TiffImages.load(joinpath(dir, "A001_2.tif")))
    @test size(imgA) == size(imgB)

    passes = multipass_parameters([64, 32]; padding = true, apodization = :gauss)
    result = run_piv(imgA, imgB, passes)

    @test all(isfinite, result.u)
    @test all(isfinite, result.v)
    # The reference analysis uses 32 px windows with no offset, so in-plane
    # displacements stay within the quarter-window rule.
    valid_u = result.u[.!result.outliers]
    valid_v = result.v[.!result.outliers]
    @test maximum(hypot.(valid_u, valid_v)) < 16
    # Seeding loss is confined to the vortex core; most of the field validates.
    @test count(result.outliers) / length(result.outliers) < 0.25
end
