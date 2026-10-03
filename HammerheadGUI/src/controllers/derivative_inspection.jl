# Current-frame derivative analysis, independent of the rendering framework.
const DERIVATIVE_SUPPORT_FIELDS=(:derivative_eligibility,:derivative_x_stencil,
    :derivative_y_stencil,:derivative_finite_count)
const _DERIVATIVE_MAP_NAMES=("eligible centers","x stencil","y stencil","finite gradient components")
_derivative_grid(r)=r isa PIVResult && length(r.x)>=2 && length(r.y)>=2
_derivative_policy_name(policy)=policy===:centered ? "both immediate neighbors required" : "available immediate neighbors"
function _clear_derivatives!(ex)
    empty!(ex.derived_cache)
    ex.derivative_digest[]=nothing
end
function _derivative_input_digest(r)
    # Hash current displayed analysis inputs, not raw/history provenance. No copies.
    Hammerhead._history_digest(Dict{String,Any}("x"=>r.x,"y"=>r.y,"u"=>r.u,"v"=>r.v,
        "mask"=>r.mask,"outliers"=>r.outliers,"length_unit"=>_length_unit(r),"field_unit"=>_field_unit(r)))
end
function _checked_derivative_inputs(ex)
    r=current_result(ex)
    if ex.derivative_digest[]!==nothing && _derivative_input_digest(r)!=ex.derivative_digest[]
        message="displayed derivative inputs changed; select the derivative support tool again to rebuild the analysis"
        ex.status[]=message
        throw(ArgumentError(message))
    end
    r
end
function _explorer_derivatives(ex)
    rich=ex.tool[]===:derivative_support
    rich && _checked_derivative_inputs(ex)
    d=get(ex.derived_cache,ex.frame[],nothing)
    if d===nothing || (rich && !hasproperty(d,:support))
        r=current_result(ex)
        d=flow_derivatives(r;stencil=ex.derivative_stencil[],return_support=rich)
        ex.derived_cache[ex.frame[]]=d
        ex.derivative_digest[]=rich ? _derivative_input_digest(r) : nothing
    end
    d
end
function _prepare_derivative_tool!(ex;refresh=false)
    r=current_result(ex)
    _derivative_grid(r) || throw(ArgumentError("derivative support needs a planar grid with at least two points per axis"))
    d=get(ex.derived_cache,ex.frame[],nothing)
    if refresh || d===nothing || !hasproperty(d,:support)
        d=flow_derivatives(r;stencil=ex.derivative_stencil[],return_support=true)
        digest=_derivative_input_digest(r)
        _clear_derivatives!(ex)
        ex.derived_cache[ex.frame[]]=d
        ex.derivative_digest[]=digest
    else
        _checked_derivative_inputs(ex)
    end
    nothing
end
function _strip_derivative_support!(ex)
    for (i,d) in ex.derived_cache
        hasproperty(d,:support) && (ex.derived_cache[i]=(;dudx=d.dudx,dudy=d.dudy,dvdx=d.dvdx,dvdy=d.dvdy,valid=d.valid))
    end
    ex.derivative_digest[]=nothing
end
function _attach_derivative_state!(ex)
    previous_tool=Ref(ex.tool[])
    on(ex.tool;priority=typemax(Int)) do tool
        try
            tool in EXPLORER_TOOLS || throw(ArgumentError("unsupported explorer tool"))
            if tool===:derivative_support
                _prepare_derivative_tool!(ex)
            else
                _strip_derivative_support!(ex)
            end
        catch err
            ex.tool.val=previous_tool[]
            ex.status[]=first(split(sprint(showerror,err),'\n'))
            rethrow()
        end
        previous_tool[]=tool
        ex.field[] in available_fields(ex) || (ex.field[]=first(available_fields(ex)))
    end
    previous_policy=Ref(:available)
    on(ex.derivative_stencil;priority=typemax(Int)) do policy
        policy===previous_policy[] && return
        candidate=nothing;region=nothing;digest=nothing
        try
            policy in (:available,:centered) || throw(ArgumentError("stencil must be :available or :centered"))
            r=current_result(ex)
            if _derivative_grid(r)
                rich=ex.tool[]===:derivative_support
                candidate=flow_derivatives(r;stencil=policy,return_support=rich)
                digest=rich ? _derivative_input_digest(r) : nothing
                result=ex.circulation_result[]
                region=result===nothing ? nothing : _explorer_circulation(r,result.contour,policy)
            end
        catch err
            ex.derivative_stencil.val=previous_policy[]
            ex.status[]=first(split(sprint(showerror,err),'\n'))
            rethrow()
        end
        _clear_derivatives!(ex)
        candidate===nothing || (ex.derived_cache[ex.frame[]]=candidate)
        ex.derivative_digest[]=digest
        previous_policy[]=policy
        region===nothing || (ex.circulation_result[]=region)
        ex.field[]=ex.field[] # refresh plots/labels, retaining the analysis policy
    end
end

"""
    set_derivative_stencil!(ex::ResultExplorer, policy::Symbol)

Choose `:available` (default, one-sided fallback) or `:centered` (both immediate
eligible neighbors required). The policy persists across tools and frames and
applies to every explorer derivative-derived scalar and area-circulation result.
Existing closed area contours are recomputed before publication. Profiles and
line circulation sample u/v and remain independent of this policy. Invalid
policies/geometry throw before changing the previous policy/cache.
"""
function set_derivative_stencil!(ex::ResultExplorer,policy::Symbol)
    ex.derivative_stencil[]=policy
    ex
end

"""
    available_fields(ex::ResultExplorer)

Current result fields, plus eligibility, x/y stencil and finite-gradient-count
maps while the planar `:derivative_support` tool is active. `available_fields`
on a result retains its ordinary field list. Support maps have fixed categorical
legends and show excluded nodes; scalar color-range overrides remain stored.
"""
function available_fields(ex::ResultExplorer)
    fields=available_fields(current_result(ex))
    ex.tool[]===:derivative_support && _derivative_grid(current_result(ex)) && append!(fields,DERIVATIVE_SUPPORT_FIELDS)
    fields
end
function _derivative_map(ex,field)
    ex.tool[]===:derivative_support || throw(ArgumentError("support maps require the derivative support tool"))
    d=_explorer_derivatives(ex)
    field===:derivative_eligibility && return d.support.center_eligible
    field===:derivative_x_stencil && return d.support.x.kind
    field===:derivative_y_stencil && return d.support.y.kind
    field===:derivative_finite_count && return reduce((a,b)->a.+b,
        (Int.(getproperty(d.support.finite,key)) for key in (:dudx,:dudy,:dvdx,:dvdy)))
    throw(ArgumentError("unknown derivative support map"))
end
_derivative_map_limits(field)=field===:derivative_eligibility ? (-.5,1.5) :
    field===:derivative_finite_count ? (-.5,4.5) : (-.5,3.5)
function _derivative_map_legend(field)
    field===:derivative_eligibility && return ([0,1],["excluded","eligible"])
    field===:derivative_finite_count && return (collect(0:4),["$i / 4 finite" for i in 0:4])
    ([0,1,2,3],["unavailable","neighbor secant","forward","backward"])
end
function _current_field_label(ex)
    field=ex.field[]
    if field in DERIVATIVE_SUPPORT_FIELDS
        return _DERIVATIVE_MAP_NAMES[findfirst(==(field),DERIVATIVE_SUPPORT_FIELDS)]
    end
    label=field_label(current_result(ex),field)
    field in DERIVED_FIELDS && ex.derivative_stencil[]===:centered ? label*"; both neighbors required" : label
end
function _explorer_circulation(r,contour,stencil)
    report=circulation(r;region=contour,coverage=:report,stencil)
    (;line=circulation(r,contour),area=report.value,contour,
        report.valid_area,report.requested_area,report.coverage_fraction,report.complete,stencil)
end

"""
    derivative_support_summary(ex::ResultExplorer) -> NamedTuple

Detached scalar current-frame counts: policy, analysis state, node count,
eligible centers, structurally supported x/y stencils and finite component
quotients. Only the active planar support tool allocates rich metadata; other
kinds/singleton axes report an explicit unsupported state. Finite components
do not certify finite combined scalars, resolution, measurement origin or UQ.
Inspection hashes displayed inputs in O(nodes), detects mutation on access and
requires reselecting the tool to rebuild; this is not continuous monitoring.
"""
function derivative_support_summary(ex::ResultExplorer)
    _derivative_summary(ex,current_result(ex))
end
function _derivative_summary(ex,r,d=nothing)
    state=!(r isa PIVResult) ? :unsupported_kind : !_derivative_grid(r) ? :singleton_grid :
        ex.tool[]===:derivative_support ? :available : :off
    if state!==:available
        return (;state,policy=ex.derivative_stencil[],nodes=0,eligible=0,x_supported=0,y_supported=0,
            finite_dudx=0,finite_dudy=0,finite_dvdx=0,finite_dvdy=0)
    end
    d===nothing && (d=_explorer_derivatives(ex))
    s=d.support;f=s.finite
    (;state,policy=ex.derivative_stencil[],nodes=length(d.valid),eligible=count(s.center_eligible),
        x_supported=count(s.x.structural_supported),y_supported=count(s.y.structural_supported),
        finite_dudx=count(f.dudx),finite_dudy=count(f.dudy),finite_dvdx=count(f.dvdx),finite_dvdy=count(f.dvdy))
end
function _derivative_exclusion(r,node)
    reasons=String[]
    r.mask[node] && push!(reasons,"masked")
    r.outliers[node] && push!(reasons,"current outlier flag")
    !isfinite(r.u[node]) && push!(reasons,"nonfinite u")
    !isfinite(r.v[node]) && push!(reasons,"nonfinite v")
    isempty(reasons) ? "eligible" : join(reasons,", ")
end

"""
    describe_derivative_selection(ex::ResultExplorer) -> String

Explain the selected planar node's eligible center, actual Cartesian
contributors, displayed coordinates, signed spans/weights and finite derivative
components. Missing neighbors are boundaries or excluded immediate neighbors;
no gaps are crossed. Centered secants omit the center's algebraic value while
still requiring its eligibility. Unrepresentable metadata weights do not imply
an unavailable native quotient. Coordinates/gradients use the already converted
display units. No measurement-origin or uncertainty applicability is inferred.
"""
function describe_derivative_selection(ex::ResultExplorer)
    ex.tool[]===:derivative_support || return "Derivative support inspection is off."
    d=_explorer_derivatives(ex)
    _describe_derivative_node(ex,current_result(ex),d)
end
function _describe_derivative_node(ex,r,d)
    node=_valid_selection(r,ex.selection[])
    node isa CartesianIndex || return "Select a grid node to inspect its immediate contributors."
    s=d.support
    lines=["Node $(Tuple(node)); center: $(_derivative_exclusion(r,node))",
        "Policy: $(_derivative_policy_name(s.policy))."]
    for (name,axis,coordinates) in (("x",s.x,r.x),("y",s.y,r.y))
        dim=axis.dimension;k=node[dim]
        kind=axis.kind[node]
        push!(lines,"$name stencil: $(("unavailable","neighbor secant","forward","backward")[Int(kind)+1])")
        for offset in (-1,1)
            q=k+offset
            other=dim==2 ? CartesianIndex(node[1],q) : CartesianIndex(q,node[2])
            reason=1<=q<=length(coordinates) ? _derivative_exclusion(r,other) : "outside grid"
            push!(lines,"  $(offset==-1 ? "previous" : "next") immediate node: $reason")
        end
        kind==0 && continue
        for (which,key) in (("first",:first_index),("second",:second_index))
            q=getproperty(axis,key)[node]
            other=dim==2 ? CartesianIndex(node[1],q) : CartesianIndex(q,node[2])
            push!(lines,"  $which $(Tuple(other)): $name=$(_fmt(coordinates[q])) $(_length_unit(r)); u/v=$(_fmt(r.u[other]))/$(_fmt(r.v[other])) $(_field_unit(r))")
        end
        push!(lines,"  signed span: $(axis.span_available[node] ? _fmt(axis.signed_span[node])*" "*_length_unit(r) : "descriptor unavailable")")
        push!(lines,axis.weights_available[node] ? "  weights: $(_fmt(axis.first_weight[node])), $(_fmt(axis.second_weight[node])) (1/$(_length_unit(r)))" : "  weights unrepresentable; direct quotient can remain finite")
    end
    for key in (:dudx,:dudy,:dvdx,:dvdy)
        push!(lines,"$key: $(_fmt(getproperty(d,key)[node])) (1/$(_time_unit(r))); finite=$(getproperty(s.finite,key)[node])")
    end
    if ex.field[] in DERIVED_FIELDS
        value=_derived_field(d,ex.field[])[node]
        push!(lines,"Displayed $(field_name(ex.field[])): $(_fmt(value)); finite=$(isfinite(value))")
    end
    push!(lines,"Support describes stored displayed values, not measurement origin, resolution or uncertainty applicability.")
    join(lines,"\n")
end
function _derivative_text(ex)
    r=current_result(ex)
    d=ex.tool[]===:derivative_support ? _explorer_derivatives(ex) : nothing
    summary=_derivative_summary(ex,r,d)
    text=summary.state===:available ? join(["Derivative support; policy: $(_derivative_policy_name(summary.policy))",
        "$(summary.eligible) / $(summary.nodes) eligible centers",
        "x/y geometry: $(summary.x_supported)/$(summary.y_supported) / $(summary.nodes) nodes",
        "Finite dudx/dudy/dvdx/dvdy: $(summary.finite_dudx)/$(summary.finite_dudy)/$(summary.finite_dvdx)/$(summary.finite_dvdy)",
        "Finite components do not certify finite combined quantities.",
        "Neighbor secants need eligible centers, even when the center has zero algebraic weight.",
        "Irregular-grid secants do not imply second-order accuracy.",
        "No origin, resolution or uncertainty applicability is inferred."],"\n") : "Derivative support: $(replace(string(summary.state),'_'=>' '))."
    text,d===nothing ? "Derivative support inspection is off." : _describe_derivative_node(ex,r,d)
end
