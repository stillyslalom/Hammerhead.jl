# Unitful support for PhysicalScale: construct from quantities, e.g.
# `PhysicalScale(20.0u"µm", 0.5u"ms")`. The value is stripped in the
# quantity's own units and the unit name becomes the display label, so no
# canonical unit is imposed — µm and ms in, µm/ms out. The core stays plain
# numbers; this constructor is the extension's entire surface.
module HammerheadUnitfulExt

using Hammerhead
using Unitful: Unitful, ustrip, unit

Hammerhead.PhysicalScale(pixel_size::Unitful.Length, dt::Unitful.Time) =
    PhysicalScale(ustrip(pixel_size), ustrip(dt),
                  string(unit(pixel_size)), string(unit(dt)))

end # module
