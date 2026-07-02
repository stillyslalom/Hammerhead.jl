using Hammerhead
using Test
using Statistics

@testset "Validation Pipeline" begin
    # Create a synthetic result with some outliers
    x = collect(1.0:32.0)
    y = collect(1.0:32.0)
    u = ones(32, 32)
    v = zeros(32, 32)
    peak_ratio = fill(2.0, 32, 32)
    moment = fill(0.2, 32, 32)
    outliers = falses(32, 32)
    params = PIVParameters()
    
    # Add some outliers
    u[5, 5] = 10.0 # Velocity magnitude outlier
    peak_ratio[10, 10] = 1.05 # Peak ratio outlier
    moment[15, 15] = 1.0 # Correlation moment outlier
    u[20, 20] = 5.0 # UOD outlier (local deviation)
    
    result = PIVResult(x, y, u, v, peak_ratio, moment, fill(NaN, 32, 32),
                       fill(NaN, 32, 32), outliers, falses(32, 32), params)
    
    @testset "PeakRatioValidator" begin
        r = deepcopy(result)
        validate_vectors!(r, (:peak_ratio => 1.2,))
        @test r.outliers[10, 10] == true
        @test count(r.outliers) == 1
    end
    
    @testset "CorrelationMomentValidator" begin
        r = deepcopy(result)
        validate_vectors!(r, (:correlation_moment => 0.5,))
        @test r.outliers[15, 15] == true
        @test count(r.outliers) == 1
    end
    
    @testset "VelocityMagnitudeValidator" begin
        r = deepcopy(result)
        validate_vectors!(r, (:velocity_magnitude => (min=0.0, max=5.0),))
        @test r.outliers[5, 5] == true
        @test count(r.outliers) == 1
    end
    
    @testset "UniversalOutlierValidator" begin
        r = deepcopy(result)
        validate_vectors!(r, (:uod => (threshold=2.0, neighborhood_size=1),))
        @test r.outliers[20, 20] == true
        @test r.outliers[5, 5] == true # Also a local deviation
        @test count(r.outliers) == 2
    end
    
    @testset "Combined Pipeline" begin
        r = deepcopy(result)
        pipeline = (
            :peak_ratio => 1.2,
            :velocity_magnitude => (min=0.0, max=8.0),
            :uod => (threshold=2.0, neighborhood_size=1)
        )
        validate_vectors!(r, pipeline)
        @test r.outliers[10, 10] == true
        @test r.outliers[5, 5] == true
        @test r.outliers[20, 20] == true
        @test r.outliers[15, 15] == false # Not in pipeline
        @test count(r.outliers) == 3
    end
end
