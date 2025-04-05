using Hammerhead
using Documenter

DocMeta.setdocmeta!(Hammerhead, :DocTestSetup, :(using Hammerhead); recursive=true)

makedocs(;
    modules=[Hammerhead],
    authors="Alex Ames <alexander.m.ames@gmail.com> and contributors",
    sitename="Hammerhead.jl",
    format=Documenter.HTML(;
        canonical="https://stillyslalom.github.io/Hammerhead.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/stillyslalom/Hammerhead.jl",
    devbranch="main",
)
