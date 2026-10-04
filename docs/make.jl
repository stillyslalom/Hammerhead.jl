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
    # Keep lessons and reader tasks visible; detailed pages remain linked,
    # searchable and built at their existing URLs beneath each topic.
    pages=[
        "Start here" => "index.md",
        "Learn by doing" => [
            "Your first vector field" => "tutorials/first_vector_field.md",
            "Find a vortex in a real recording" => "tutorials/real_data.md",
            "From image pairs to flow statistics" => "tutorials/sequence_statistics.md",
            "Stereo PIV end to end" => "tutorials/stereo.md",
            "Stereo on a real recording: vortex ring" => "tutorials/stereo_real.md",
            "Particle tracking (PTV)" => "tutorials/ptv.md",
            "Your first PIV session in the GUI" => "tutorials/gui_tour.md",
        ],
        "Work with your recording" => [
            hide("Prepare images" => "guides/images.md", [
                "Inspect image quality" => "howto/image_quality.md",
                "Mask reflections and geometry" => "howto/masking.md",
                "Build a preprocessing chain" => "howto/preprocessing.md",
            ]),
            hide("Choose a processing method" => "guides/processing.md", [
                "Choose an effort level" => "howto/effort.md",
                "Tune validation" => "howto/validation.md",
                "Measure one field from many pairs" => "howto/ensemble.md",
                "Run PIV on a GPU" => "howto/gpu.md",
                "Calibrate a real stereo rig" => "howto/stereo_rig.md",
            ]),
            hide("Analyze and export results" => "guides/results.md", [
                "Scale results to physical units" => "howto/scaling.md",
                "Register PIV and PLIF on a shared grid" => "howto/calibrated_resampling.md",
            ]),
            hide("Process a recording again" => "guides/repeat.md", [
                "Batch processing and result files" => "howto/batch.md",
                "Save settings and reuse them" => "howto/recipes.md",
            ]),
            "Work in the GUI" => [
                "Analyze an image pair in the GUI" => "howto/gui.md",
                "Run stereo PIV in the GUI" => "howto/gui_stereo.md",
            ],
        ],
        hide("Understand the measurements" => "explanation/index.md", [
            "Coordinates, signs, and units" => "explanation/conventions.md",
            "Correlation accuracy" => "explanation/correlation.md",
            "Non-informative windows" => "explanation/noninformative_windows.md",
            "Multi-pass interrogation and image deformation" => "explanation/multipass.md",
            "The masking model" => "explanation/masking.md",
            "Uncertainty quantification" => "explanation/uncertainty.md",
            "How accurate are the measurements?" => "explanation/validation_results.md",
            "Stereo geometry and self-calibration" => "explanation/stereo.md",
            "Numeric precision policy" => "explanation/precision.md",
        ]),
        hide("API reference" => "reference/index.md", [
            "Core pipeline and parameters" => "reference/pipeline.md",
            "Preprocessing" => "reference/preprocessing.md",
            "Derived flow analysis" => "reference/derived.md",
            "Validation and quality" => "reference/validation.md",
            "Calibration, dewarping, and stereo" => "reference/stereo.md",
            "I/O and batch processing" => "reference/io.md",
            "Saved settings (recipes)" => "reference/recipes.md",
            "Calibrated image and planar-field resampling" => "reference/calibrated_resampling.md",
            "Ensemble and statistics" => "reference/ensemble.md",
            "Synthetic data" => "reference/synthetic.md",
            "PTV (particle tracking)" => "reference/ptv.md",
            "GUI (HammerheadGUI)" => "reference/gui.md",
            "GUI stereo window" => "reference/gui_stereo.md",
            "GUI result explorer" => "reference/gui_results.md",
            "KernelAbstractions and GPU backends" => "reference/backends.md",
            "Feature matrix" => "reference/feature_matrix.md",
            "Internals" => "reference/internals.md",
        ]),
        hide("Development and validation" => "development/index.md", [
            "The GUI's controller–view split" => "explanation/gui.md",
            "Compatibility policy" => "explanation/compatibility.md",
        ]),
        "Bibliography" => "references.md",
    ],
)

deploydocs(;
    repo="github.com/stillyslalom/Hammerhead.jl",
    devbranch="main",
)
