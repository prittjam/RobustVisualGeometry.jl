using Documenter
using RobustVisualGeometry

makedocs(;
    modules=[RobustVisualGeometry],
    authors="James Pritts <jbpritts@gmail.com>",
    sitename="RobustVisualGeometry.jl",
    remotes=nothing,
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://prittjam.github.io/RobustVisualGeometry.jl",
        edit_link="master",
        assets=String[],
    ),
    warnonly=true,
    pages=[
        "Home" => "index.md",
        "Getting Started" => "guide/getting-started.md",
        "User Guide" => [
            "Architecture" => "guide/architecture.md",
            "Scoring Functions" => "guide/scoring.md",
            "RANSAC" => "guide/ransac.md",
            "Fitting Algorithms" => "guide/fitting.md",
            "Extending" => "guide/extending.md",
        ],
        "API Reference" => [
            "Interface" => "api/interface.md",
            "IRLS" => "api/irls.md",
            "GNC" => "api/gnc.md",
            "Scoring" => "api/scoring.md",
            "RANSAC" => "api/ransac.md",
            "Conic Fitting" => "api/conic.md",
            "Line Fitting" => "api/line.md",
            "Homography" => "api/homography.md",
            "Fundamental Matrix" => "api/fundmat.md",
            "P3P" => "api/p3p.md",
        ],
    ],
)
