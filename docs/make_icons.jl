# Local only: regenerate the Hammerhead icon files from one drawing.
#
#     julia docs/make_icons.jl
#
# Writes the SVG (docs sidebar logo, GUI header) and a multi-size ICO (docs
# favicon, Windows title bar and taskbar icon of the GUI windows) to
# docs/src/assets/ and HammerheadGUI/src/qml/icons/. The drawing: a
# hammerhead seen from above whose shaft is a displacement vector, with the
# arrowhead set off ahead of the head, eye dots at the hammer ends and fins
# halfway down the shaft, in Julia's colors, swimming toward the upper right.
using Pkg
Pkg.activate(; temp = true)
Pkg.add("Librsvg_jll"; io = devnull)
using Librsvg_jll

const BLUE, RED, GREEN, PURPLE = "#4063D8", "#CB3C33", "#389826", "#9558B2"

const SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">\
<rect width="64" height="64" rx="14" fill="#F6F8FA"/>\
<g transform="rotate(45 32 32) translate(32 32) scale(0.9) translate(-32 -32)">\
<line x1="32" y1="55" x2="32" y2="27" stroke="$BLUE" stroke-width="7" stroke-linecap="round"/>\
<path d="M15.5,28 Q32,18 48.5,28" fill="none" stroke="$BLUE" stroke-width="8" stroke-linecap="round"/>\
<path d="M30,38 L23.5,45.5 L30,43.5 Z M34,38 L40.5,45.5 L34,43.5 Z" fill="$BLUE"/>\
<path d="M32,4.5 L23.5,15.5 L40.5,15.5 Z" fill="$PURPLE" stroke="$PURPLE" stroke-width="2" stroke-linejoin="round"/>\
<circle cx="12" cy="29.5" r="6" fill="$RED"/><circle cx="52" cy="29.5" r="6" fill="$GREEN"/>\
</g></svg>
"""

# An ICO holding one PNG per size (Windows Vista and later, and browsers).
function write_ico(path, pngs::Vector{Pair{Int,Vector{UInt8}}})
    open(path, "w") do io
        write(io, UInt16(0), UInt16(1), UInt16(length(pngs)))
        offset = 6 + 16 * length(pngs)
        for (px, data) in pngs
            b = UInt8(px >= 256 ? 0 : px)
            write(io, b, b, UInt8(0), UInt8(0), UInt16(1), UInt16(32),
                  UInt32(length(data)), UInt32(offset))
            offset += length(data)
        end
        foreach(p -> write(io, p.second), pngs)
    end
end

root = dirname(@__DIR__)
dirs = (joinpath(root, "docs", "src", "assets"), joinpath(root, "HammerheadGUI", "src", "qml", "icons"))
mktempdir() do tmp
    svg = joinpath(tmp, "logo.svg")
    write(svg, SVG)
    pngs = map((16, 24, 32, 48, 64, 128, 256)) do px
        out = joinpath(tmp, "$px.png")
        run(`$(rsvg_convert()) -w $px -h $px -o $out $svg`)
        px => read(out)
    end
    for d in dirs
        mkpath(d)
        write(joinpath(d, d == dirs[1] ? "logo.svg" : "hammerhead.svg"), SVG)
        write_ico(joinpath(d, d == dirs[1] ? "favicon.ico" : "hammerhead.ico"), collect(pngs))
    end
end
println("wrote the icons to ", join(dirs, " and "))
