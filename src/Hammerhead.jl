module Hammerhead

using FFTW
using LinearAlgebra
using Interpolations
using LsqFit: curve_fit
using Statistics: median

export PIVParameters, PIVResult, run_piv
export Correlator, CrossCorrelator, PhaseCorrelator, correlate, correlate_deformable
export AffineTransform, warp_image, calculate_manual_registration, transform_vector_field
export calculate_peak_ratio, calculate_correlation_moment, universal_outlier_detection

include("types.jl")
include("correlators.jl")
include("transforms.jl")
include("quality.jl")
include("pipeline.jl")

end # module Hammerhead
