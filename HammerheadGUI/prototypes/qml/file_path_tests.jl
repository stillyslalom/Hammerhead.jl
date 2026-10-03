# No dialog/window is created by these Qt URL and request-boundary checks.
using Test, QML
include("file_paths.jl")
const FP = FilePaths

@testset "Local Qt URL conversion preserves file-name data" begin
    mktempdir() do directory
        for filename in ("plain.jld2", "space name.jld2", "résultat λ 粒子.jld2", "literal # and %23.jld2")
            path = joinpath(directory, filename)
            write(path, "unchanged")
            url = FP.file_url(path)
            actual = FP.local_path(url)
            @test normpath(actual) == normpath(path)
            @test read(actual, String) == "unchanged"
            @test String(QML.toString(FP.file_url(actual))) == String(QML.toString(url))
        end
        future = joinpath(directory, "new output # λ.jld2")
        @test normpath(FP.local_path(FP.file_url(future))) == normpath(future)
        @test !ispath(future) # choosing SaveFile data never creates the destination
        @test normpath(FP.local_path(FP.initial_folder(future))) == normpath(directory)
        @test normpath(FP.local_path(FP.initial_folder("relative.jld2"))) == normpath(pwd())
        base = String(QML.toString(FP.file_url(joinpath(directory, "plain.jld2"))))
        for bad in (base * "?query=value", base * "#fragment", "https://example.test/image.jld2", "file:relative.jld2", "")
            @test_throws ArgumentError FP.local_path(QML.QUrl(bad))
        end
        for bad in ("", "relative.jld2", "bad\0path")
            @test_throws ArgumentError FP.file_url(bad)
        end
        if Sys.iswindows()
            @test_throws ArgumentError FP.file_url("/arbitrary/manual.jld2")
            @test normpath(FP.local_path(FP.initial_folder("/arbitrary/manual.jld2"))) == normpath(pwd())
            # URL conversion only: no share connection or UNC file access.
            unc = raw"\\server\share\space λ # percent%.jld2"
            @test replace(FP.local_path(FP.file_url(unc)), '\\' => '/') == replace(unc, '\\' => '/')
            @test startswith(FP.local_path(FP.file_url(unc)), "//server/share/")
        end
    end
end

@testset "Picker requests produce accepted-only drafts" begin
    mktempdir() do directory
        destination = joinpath(directory, "future.jld2")
        url = FP.file_url(destination)
        state = FP.DialogState()
        drafts = Dict(purpose => "old-$purpose" for purpose in FP.PURPOSES)
        for purpose in FP.PURPOSES
            before = copy(drafts)
            token = FP.begin!(state, purpose)
            @test token > 0 && state.active
            @test FP.begin!(state, purpose) == 0
            @test FP.accept!(state, token - 1, purpose, url) === nothing
            @test FP.accept!(state, token, "wrong-purpose", url) === nothing
            @test drafts == before && state.active
            accepted = FP.accept!(state, token, purpose, url)
            accepted === nothing || (drafts[purpose] = accepted)
            @test normpath(drafts[purpose]) == normpath(destination)
            @test !state.active && !ispath(destination)
            @test FP.accept!(state, token, purpose, url) === nothing
        end
        token = FP.begin!(state, "output")
        @test !FP.reject!(state, token - 1) && state.active
        @test FP.reject!(state, token) && !state.active
        @test FP.accept!(state, token, "output", url) === nothing
        next = FP.begin!(state, "history")
        @test next > token
        @test FP.accept!(state, token, "history", url) === nothing && state.active
        @test FP.accept!(state, next, "history", url; allowed=false) === nothing
        @test !state.active && !ispath(destination)
        @test FP.begin!(state, "output"; allowed=false) == 0 && !state.active
        invalid = FP.begin!(state, "record")
        @test_throws ArgumentError FP.accept!(state, invalid, "record", QML.QUrl("https://example.test/no"))
        @test !state.active
        last = FP.begin!(state, "result")
        FP.close!(state)
        @test FP.accept!(state, last, "result", url) === nothing
        @test FP.begin!(state, "record") == 0 && !state.active
        @test !ispath(destination)
        @test_throws ArgumentError FP.begin!(FP.DialogState(), "unknown")
        exhausted = FP.DialogState(typemax(Int), "", false, false)
        @test_throws OverflowError FP.begin!(exhausted, "output")
        @test !exhausted.active && exhausted.generation == typemax(Int)
    end
end
