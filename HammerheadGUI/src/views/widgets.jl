# Shared widget<->controller observable wiring. Observables notify on
# same-value writes, so every two-way binding needs equality guards or the
# pair loops forever.

function _sync_menu!(menu, obs::Observable)
    values = last.(menu.options[])
    on(menu.selection) do v
        v === nothing || v == obs[] || (obs[] = v)
    end
    on(obs) do v
        i = findfirst(==(v), values)
        (i === nothing || i == menu.i_selected[]) || (menu.i_selected[] = i)
    end
    menu.i_selected[] = something(findfirst(==(obs[]), values), 1)
    return menu
end

function _sync_toggle!(toggle, obs::Observable{Bool})
    on(a -> a == obs[] || (obs[] = a), toggle.active)
    on(v -> v == toggle.active[] || (toggle.active[] = v), obs)
    return toggle
end
