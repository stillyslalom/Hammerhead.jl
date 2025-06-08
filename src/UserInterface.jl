module UserInterface

using ..Visualization # To access plot_vector_field
# Potentially using ..Hammerhead for other components if the PIV pipeline was called here.

export PIVParameters, run_piv_from_gui

"""
    PIVParameters

A mutable struct to hold common PIV settings.
"""
mutable struct PIVParameters
    # Interrogation window settings
    window_size_x::Int
    window_size_y::Int
    overlap_x::Int
    overlap_y::Int

    # Correlation settings
    correlation_method::Symbol  # :cross_correlation, :phase_correlation
    subpixel_method::Symbol     # :gauss3, :gauss2d, :none

    # Iterative deformation settings
    enable_deformation::Bool
    deformation_iterations::Int

    # Preprocessing settings (examples, might be more detailed)
    preprocess_enable_registration::Bool
    # registration_points_img::Vector{Tuple{Float64,Float64}}
    # registration_points_ref::Vector{Tuple{Float64,Float64}}

    # Postprocessing settings
    postprocess_enable_uod::Bool
    uod_threshold::Float64
    uod_neighborhood_size::Int

    # Constructor with default values
    function PIVParameters(;
        window_size_x::Int = 32, window_size_y::Int = 32,
        overlap_x::Int = 16, overlap_y::Int = 16, # 50% overlap
        correlation_method::Symbol = :phase_correlation,
        subpixel_method::Symbol = :gauss3,
        enable_deformation::Bool = false,
        deformation_iterations::Int = 3,
        preprocess_enable_registration::Bool = false,
        postprocess_enable_uod::Bool = true,
        uod_threshold::Float64 = 2.0,
        uod_neighborhood_size::Int = 1
    )
        new(
            window_size_x, window_size_y,
            overlap_x, overlap_y,
            correlation_method,
            subpixel_method,
            enable_deformation,
            deformation_iterations,
            preprocess_enable_registration,
            postprocess_enable_uod,
            uod_threshold,
            uod_neighborhood_size
        )
    end
end

# Custom show method for PIVParameters for better printing
function Base.show(io::IO, params::PIVParameters)
    println(io, "PIVParameters:")
    for name in fieldnames(typeof(params))
        println(io, "  ", name, ": ", getfield(params, name))
    end
end

"""
    run_piv_from_gui(image_path1::String, image_path2::String, params::PIVParameters)

Simulates a PIV analysis run triggered from a GUI, including plotting results.
"""
function run_piv_from_gui(image_path1::String, image_path2::String, params::PIVParameters)
    println("Running PIV with parameters for images: '$image_path1' and '$image_path2'")
    show(stdout, params) # Print parameters using the custom show method
    println("-"^40)

    # 1. (Simulate) Load images - In a real scenario, this would use an image loading library
    println("Simulating image loading...")
    # Dummy image dimensions (replace with actual if images were loaded)
    img_height = 256
    img_width = 256

    # 2. (Simulate) Main PIV processing pipeline
    println("Simulating PIV processing...")

    # Calculate step size from window size and overlap
    step_x = params.window_size_x - params.overlap_x
    step_y = params.window_size_y - params.overlap_y

    # Generate dummy grid points based on PIV parameters
    # Grid centers start half a window in, then step
    x_grid_centers = (params.window_size_x/2):step_x:(img_width - params.window_size_x/2 + 1)
    y_grid_centers = (params.window_size_y/2):step_y:(img_height - params.window_size_y/2 + 1)

    if isempty(x_grid_centers) || isempty(y_grid_centers)
        println("Warning: Calculated grid is empty. Check window size, overlap, and image dimensions.")
        # Create a minimal grid to avoid errors in plotting
        x_grid_centers = [Float64(img_width/2)]
        y_grid_centers = [Float64(img_height/2)]
    end

    num_x_points = length(x_grid_centers)
    num_y_points = length(y_grid_centers)

    # Generate dummy vector field data (e.g., a simple shear flow)
    u_dummy = zeros(Float64, num_y_points, num_x_points)
    v_dummy = zeros(Float64, num_y_points, num_x_points)

    for i in 1:num_y_points
        for j in 1:num_x_points
            # Example: u = y_coord (scaled), v = 0
            u_dummy[i,j] = (y_grid_centers[i] / img_height - 0.5) * 50.0
            v_dummy[i,j] = 0.0
        end
    end

    println("Dummy PIV processing complete. Generated $(num_x_points)x$(num_y_points) vector field.")

    # 3. Call plot_vector_field with dummy data
    println("Displaying results...")
    try
        # x_coords and y_coords for plot_vector_field should be 1D vectors if u/v are grids
        Visualization.plot_vector_field(
            collect(x_grid_centers),
            collect(y_grid_centers),
            u_dummy,
            v_dummy;
            axis_kwargs=Dict(:title => "Simulated PIV Result for $image_path1 & $image_path2")
        )
        println("Plot displayed.")
    catch e
        println(stderr, "Error during plotting: $e")
        println(stderr, "Ensure GLMakie and a suitable backend are working correctly.")
    end

    println("-"^40)
    println("PIV analysis and visualization simulation complete.")
    println("-"^40)
    # In a real GUI, might return status or results object
end

end # module UserInterface
