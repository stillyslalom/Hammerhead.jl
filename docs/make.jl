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
const TUTORIALS = ["first_vector_field.jl", "real_data.jl", "stereo.jl", "stereo_real.jl", "ptv.jl", "gui_tour.jl"]
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
            "Stereo PIV end to end" => "tutorials/stereo.md",
            "Stereo on a real recording: vortex ring" => "tutorials/stereo_real.md",
            "Particle tracking (PTV)" => "tutorials/ptv.md",
            "A tour of the GUI" => "tutorials/gui_tour.md",
        ],
        "How-to guides" => [
            "Mask reflections and geometry" => "howto/masking.md",
            "Build a preprocessing chain" => "howto/preprocessing.md",
            "Tune validation" => "howto/validation.md",
            "Ensemble correlation for low SNR" => "howto/ensemble.md",
            "Batch processing and result files" => "howto/batch.md",
            "Calibrate a real stereo rig" => "howto/stereo_rig.md",
            "Work interactively with the GUI" => "howto/gui.md",
        ],
        "Explanation" => [
            "Coordinates, signs, and units" => "explanation/conventions.md",
            "Correlation accuracy" => "explanation/correlation.md",
            "Multi-pass interrogation and image deformation" => "explanation/multipass.md",
            "The masking model" => "explanation/masking.md",
            "Uncertainty quantification" => "explanation/uncertainty.md",
            "Stereo geometry and self-calibration" => "explanation/stereo.md",
            "Numeric precision policy" => "explanation/precision.md",
            "The GUI's controller–view split" => "explanation/gui.md",
        ],
        "Reference" => [
            "Core pipeline and parameters" => "reference/pipeline.md",
            "Preprocessing" => "reference/preprocessing.md",
            "Validation and quality" => "reference/validation.md",
            "Calibration, dewarping, and stereo" => "reference/stereo.md",
            "I/O and batch processing" => "reference/io.md",
            "Ensemble and statistics" => "reference/ensemble.md",
            "Synthetic data" => "reference/synthetic.md",
            "PTV (particle tracking)" => "reference/ptv.md",
            "GUI (HammerheadGUI)" => "reference/gui.md",
            "Internals" => "reference/internals.md",
        ],
        "Bibliography" => "references.md",
    ],
)

deploydocs(;
    repo="github.com/stillyslalom/Hammerhead.jl",
    devbranch="main",
)
