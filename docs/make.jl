using Hammerhead
using HammerheadGUI
using Documenter
using DocumenterCitations
using Literate

DocMeta.setdocmeta!(Hammerhead, :DocTestSetup, :(using Hammerhead); recursive=true)

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:authoryear)

# Tutorials are Literate.jl scripts, converted to Documenter markdown with
# executable @example blocks — the docs build runs them, so they double as
# integration tests.
const TUTORIALS = ["first_vector_field.jl", "real_data.jl", "sequence_statistics.jl", "stereo.jl", "stereo_real.jl", "ptv.jl", "gui_tour.jl"]
for tutorial in TUTORIALS
    Literate.markdown(joinpath(@__DIR__, "lit", tutorial),
                      joinpath(@__DIR__, "src", "tutorials"); documenter=true)
end

makedocs(;
    modules=[Hammerhead, HammerheadGUI],
    authors="Alex Ames <alexander.m.ames@gmail.com> and contributors",
    sitename="Hammerhead.jl",
    plugins=[bib],
    format=Documenter.HTML(;
        canonical="https://stillyslalom.github.io/Hammerhead.jl",
        edit_link="main",
        assets=String["assets/citations.css"],
    ),
    pages=[
        "Home" => "index.md",
        "Tutorials" => [
            "Your first vector field" => "tutorials/first_vector_field.md",
            "A real recording: tip vortex" => "tutorials/real_data.md",
            "From image pairs to flow statistics" => "tutorials/sequence_statistics.md",
            "Stereo PIV end to end" => "tutorials/stereo.md",
            "Stereo on a real recording: vortex ring" => "tutorials/stereo_real.md",
            "Particle tracking (PTV)" => "tutorials/ptv.md",
            "A tour of the GUI" => "tutorials/gui_tour.md",
        ],
        "How-to guides" => [
            "Inspect image quality" => "howto/image_quality.md",
            "Mask reflections and geometry" => "howto/masking.md",
            "Build a preprocessing chain" => "howto/preprocessing.md",
            "Choose an effort level" => "howto/effort.md",
            "Run PIV on a GPU" => "howto/gpu.md",
            "Scale results to physical units" => "howto/scaling.md",
            "Tune validation" => "howto/validation.md",
            "Run a validation scorecard" => "howto/validation_scorecard.md",
            "Evaluate synthetic uncertainty" => "howto/validation_uncertainty.md",
            "Diagnose uncertainty estimates" => "howto/diagnostic_uncertainty.md",
            "Measure conditional noise variability" => "howto/conditional_uncertainty.md",
            "Ensemble correlation for low SNR" => "howto/ensemble.md",
            "Validate sampling times for spectra" => "howto/spectrum_timing.md",
            "Batch processing and result files" => "howto/batch.md",
            "Save and replay an experiment" => "howto/experiments.md",
            "Compare recipes on a representative pair" => "howto/pair_comparison.md",
            "Inspect actual pass execution" => "howto/execution_diagnostics.md",
            "Trace final-vector measurements" => "howto/measurement_history.md",
            "Preserve frame-pair timing" => "howto/pair_timing.md",
            "Track particles with actual sample times" => "howto/tracking_timing.md",
            "Export calibrated particles and trajectories" => "howto/calibrated_scattered_export.md",
            "Save a run-quality report" => "howto/run_quality.md",
            "Checkpoint and resume an experiment" => "howto/checkpoints.md",
            "Calibrate a real stereo rig" => "howto/stereo_rig.md",
            "Work interactively with the GUI" => "howto/gui.md",
            "Save and replay GUI experiments" => "howto/gui_experiments.md",
            "Monitor and cancel GUI replay" => "howto/gui_experiment_replay.md",
            "Resume experiments in the GUI" => "howto/gui_checkpoints.md",
            "Inspect recorded GUI companions" => "howto/gui_companions.md",
            "Compare saved GUI recipes" => "howto/gui_comparison.md",
            "Inspect timed trajectories in the GUI" => "howto/gui_tracking_timing.md",
        ],
        "Explanation" => [
            "Coordinates, signs, and units" => "explanation/conventions.md",
            "Correlation accuracy" => "explanation/correlation.md",
            "Non-informative windows" => "explanation/noninformative_windows.md",
            "Multi-pass interrogation and image deformation" => "explanation/multipass.md",
            "The masking model" => "explanation/masking.md",
            "Uncertainty quantification" => "explanation/uncertainty.md",
            "Stereo geometry and self-calibration" => "explanation/stereo.md",
            "Numeric precision policy" => "explanation/precision.md",
            "The GUI's controller–view split" => "explanation/gui.md",
            "GUI framework evaluation" => "explanation/gui_framework.md",
            "Compatibility policy" => "explanation/compatibility.md",
        ],
        "Reference" => [
            "Core pipeline and parameters" => "reference/pipeline.md",
            "Preprocessing" => "reference/preprocessing.md",
            "Derived flow analysis" => "reference/derived.md",
            "Validation and quality" => "reference/validation.md",
            "Calibration, dewarping, and stereo" => "reference/stereo.md",
            "I/O and batch processing" => "reference/io.md",
            "Experiment records" => "reference/experiments.md",
            "Representative-pair comparisons" => "reference/pair_comparison.md",
            "Execution diagnostics" => "reference/execution_diagnostics.md",
            "Measurement history" => "reference/measurement_history.md",
            "Pair timing" => "reference/pair_timing.md",
            "Tracking with actual sample times" => "reference/tracking_timing.md",
            "Calibrated particle and trajectory tables" => "reference/calibrated_table.md",
            "Run-quality reports" => "reference/run_quality.md",
            "Experiment checkpoints" => "reference/checkpoints.md",
            "Ensemble and statistics" => "reference/ensemble.md",
            "Synthetic data" => "reference/synthetic.md",
            "PTV (particle tracking)" => "reference/ptv.md",
            "GUI (HammerheadGUI)" => "reference/gui.md",
            "GUI result explorer" => "reference/gui_results.md",
            "GUI experiment workflows" => "reference/gui_experiments.md",
            "GUI recipe comparison" => "reference/gui_comparison.md",
            "GUI checkpoint workflows" => "reference/gui_checkpoints.md",
            "KernelAbstractions and GPU backends" => "reference/backends.md",
            "Feature matrix" => "reference/feature_matrix.md",
            "Internals" => "reference/internals.md",
        ],
        "Bibliography" => "references.md",
    ],
)

deploydocs(;
    repo="github.com/stillyslalom/Hammerhead.jl",
    devbranch="main",
)
