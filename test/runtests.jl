using Hammerhead
using Test

function generate_gaussian_particle!(array::AbstractArray, centroid::Tuple{Float64, Float64}, diameter::Float64)
    sigma = diameter / 2.0  # Approximation: diameter covers ~1 standard deviation
    for i in axes(array, 1), j in axes(array, 2)
        x, y = i - centroid[1], j - centroid[2]
        v = exp(-0.5 * (x^2 + y^2) / sigma^2)
        v < 1e-10 && continue  # Avoid numerical issues
        array[i, j] += exp(-0.5 * (x^2 + y^2) / sigma^2)
    end
end

@testset "Hammerhead.jl" begin
    # Write your tests here.
end

@testset "CrossCorrelator Tests with Gaussian particle" begin
    # Define test image size
    image_size = (64, 64)
    centroid1 = image_size ./ 2
    displacement = (2.2, 1.3)
    diameter = 2.6

    # Create test images with a Gaussian particle
    img1 = zeros(Float64, image_size)
    img2 = zeros(Float64, image_size)

    # Add a Gaussian particle in the center of img1
    generate_gaussian_particle!(img1, centroid1, diameter)

    # Shift the particle to create img2
    generate_gaussian_particle!(img2, centroid1 .+ displacement, diameter)

    # Initialize CrossCorrelator
    correlator = CrossCorrelator(image_size)
    
    # Perform correlation
    displacement = correlate(correlator, img1, img2)
    @test all(displacement .≈ (2.2, 1.3))
end
