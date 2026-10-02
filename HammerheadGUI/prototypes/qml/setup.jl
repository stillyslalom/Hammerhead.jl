# Isolated candidate environment. Never activates or updates production projects.
using Pkg
ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"
Pkg.activate(@__DIR__)
project = joinpath(@__DIR__, "Project.toml")
original = read(project, String)
try
    cd(@__DIR__) do
        Pkg.develop([PackageSpec(path = "../../.."), PackageSpec(path = "../..")])
    end
    Pkg.instantiate()
finally
    # Pkg.develop canonicalizes [sources] to this machine's absolute paths.
    # Keep the checked-in candidate project portable; the ignored manifest may
    # retain its normal local paths.
    write(project, original)
end
