using Test, Random

# Independent reference for the pre-support public numerical contract. It
# selects Cartesian contributors directly and never calls new geometry,
# stencil, quotient or support helpers.
function legacy_derivative_reference(field,coordinates,eligible,dimension)
    reference=fill(NaN,size(field))
    step=dimension==1 ? CartesianIndex(1,0) : CartesianIndex(0,1)
    for center in CartesianIndices(field)
        eligible[center] || continue
        k=center[dimension]
        minus,plus=center-step,center+step
        has_minus=k>1 && eligible[minus]
        has_plus=k<length(coordinates) && eligible[plus]
        first,last=has_minus && has_plus ? (minus,plus) :
            has_plus ? (center,plus) : has_minus ? (minus,center) : (center,center)
        first==last && continue
        reference[center]=(field[last]-field[first])/(coordinates[last[dimension]]-coordinates[first[dimension]])
    end
    reference
end

@testset "Derivative independent legacy-formula parity" begin
    # No seeded golden values or statistical thresholds: both methods see
    # identical generated inputs on each Julia version, including signed axes.
    rng=MersenneTwister(42103)
    for T in (Float32,Float64),spacing in (:regular,:irregular),xd in (false,true),yd in (false,true),replicate in 1:3
        x=spacing==:regular ? T.(collect(0:6).* .75) : cumsum(T(.125).+rand(rng,T,7))
        y=spacing==:regular ? T.(collect(0:5).* 1.25) : cumsum(T(.125).+rand(rng,T,6))
        xd && reverse!(x);yd && reverse!(y)
        u,v=randn(rng,T,6,7),randn(rng,T,6,7)
        valid=replicate==1 ? trues(6,7) : BitMatrix(rand(rng,6,7).>.3)
        expected=(dudx=legacy_derivative_reference(u,x,valid,2),
            dudy=legacy_derivative_reference(u,y,valid,1),
            dvdx=legacy_derivative_reference(v,x,valid,2),
            dvdy=legacy_derivative_reference(v,y,valid,1))
        ordinary=flow_derivatives(x,y,u,v;valid)
        described=flow_derivatives(x,y,u,v;valid,return_support=true)
        @test all(k->isequal(getproperty(ordinary,k),getproperty(expected,k)),keys(expected))
        @test all(k->isequal(getproperty(described,k),getproperty(expected,k)),keys(expected))
        @test ordinary.valid==described.valid==valid
    end
end

@testset "Derivative support on dense fields and descending coordinates" begin
    for T in (Float32,Float64,Int)
        x=T[0,2,4,6];y=T[0,3,6]
        u=[2xx+3yy+4 for yy in y,xx in x]
        v=[5xx-2yy+7 for yy in y,xx in x]
        ordinary=flow_derivatives(x,y,u,v)
        described=flow_derivatives(x,y,u,v;return_support=true)
        @test propertynames(ordinary)==(:dudx,:dudy,:dvdx,:dvdy,:valid)
        @test propertynames(described)==(:dudx,:dudy,:dvdx,:dvdy,:valid,:support)
        for key in (:dudx,:dudy,:dvdx,:dvdy,:valid)
            @test isequal(getproperty(ordinary,key),getproperty(described,key))
        end
        @test all(==(2),ordinary.dudx)
        @test all(==(3),ordinary.dudy)
        @test all(==(5),ordinary.dvdx)
        @test all(==(-2),ordinary.dvdy)
        @test all(==(2),vorticity(described))
        @test all(iszero,divergence(described))
        @test all(described.support.x.structural_supported)
        @test all(described.support.y.structural_supported)
        @test described.support.center_eligible===described.valid
        @test described.support.x.dimension==2
        @test described.support.y.dimension==1
        @test eltype(described.support.x.signed_span)==Float64
        @test described.support.x.kind[2,:]==[described.support.x.legend.forward,
            described.support.x.legend.centered_secant,described.support.x.legend.centered_secant,
            described.support.x.legend.backward]
        @test described.support.x.first_index[2,:]==[1,1,2,3]
        @test described.support.x.second_index[2,:]==[2,3,4,4]
        @test described.support.x.signed_span[2,:]==[2.,4.,4.,2.]
        @test described.support.x.first_weight[2,:]==[-.5,-.25,-.25,-.5]
        @test described.support.x.second_weight[2,:]==[.5,.25,.25,.5]
        @test all(described.support.finite.dudx)
        centered=flow_derivatives(x,y,u,v;stencil=:centered,return_support=true)
        @test centered.support.policy==:centered
        @test count(centered.support.x.structural_supported)==6
        @test count(centered.support.y.structural_supported)==4
        @test count(isfinite,vorticity(centered))==2
        @test all(isnan,centered.dudx[:,[1,4]])
        @test all(==(2),centered.dudx[:,2:3])
    end
    x=[3.,1.,0.];y=[4.,2.,1.]
    u=[2xx+3yy for yy in y,xx in x];v=copy(u)
    d=flow_derivatives(x,y,u,v;return_support=true)
    @test all(==(2),d.dudx)
    @test all(==(3),d.dudy)
    @test d.support.x.signed_span[2,:]==[-2.,-3.,-1.]
    @test d.support.x.first_weight[2,1]==.5
    @test d.support.x.second_weight[2,1]==-.5
    @test d.support.x.first_index[2,2]==1 && d.support.x.second_index[2,2]==3
    @test d.support.x.first_weight[2,2]==1/3
    @test d.support.x.second_weight[2,2]==-1/3
    # On irregular spacing this is exactly the neighbor secant, not the
    # general quadratic three-point derivative at x=1 (which would be 2).
    x=[0.,1.,3.];y=[0.,2.,5.]
    u=[xx^2 for yy in y,xx in x]
    d=flow_derivatives(x,y,u,zeros(3,3);return_support=true)
    @test d.dudx[2,:]==[1.,3.,4.]
    @test d.support.x.signed_span[2,2]==3
    @test d.support.x.first_weight[2,2]==-1/3
    @test d.support.x.second_weight[2,2]==1/3
    @test d.support.x.first_index[2,2]==1 && d.support.x.second_index[2,2]==3
    @test d.support.center_eligible[2,2]
end

@testset "Derivative support eligibility, gaps and arithmetic contributors" begin
    x=collect(0.:4.);y=collect(0.:2.)
    u=[2xx+3yy for yy in y,xx in x];v=[5xx-2yy for yy in y,xx in x]
    eligible=trues(3,5);eligible[2,3]=false
    d=flow_derivatives(x,y,u,v;valid=eligible,return_support=true)
    @test count(d.valid)==14
    @test d.valid==eligible && d.valid!==eligible
    @test count(d.support.x.structural_supported)==14
    @test count(d.support.y.structural_supported)==12
    @test count(isfinite,vorticity(d))==12
    @test d.support.x.kind[2,2]==d.support.x.legend.backward
    @test d.support.x.first_index[2,2]==1 && d.support.x.second_index[2,2]==2
    @test d.support.x.kind[2,4]==d.support.x.legend.forward
    @test d.support.x.first_index[2,4]==4 && d.support.x.second_index[2,4]==5
    @test !d.support.x.structural_supported[2,3]
    @test d.support.x.first_index[2,3]==d.support.x.second_index[2,3]==0
    @test isnan(d.support.x.signed_span[2,3])
    @test isnan(d.support.x.first_weight[2,3]) && isnan(d.support.x.second_weight[2,3])
    @test !d.support.x.span_available[2,3] && !d.support.x.weights_available[2,3]
    @test !d.support.y.structural_supported[1,3] && !d.support.y.structural_supported[3,3]
    @test d.valid[1,3] && !d.support.finite.dudy[1,3]
    centered=flow_derivatives(x,y,u,v;valid=eligible,stencil=:centered,return_support=true)
    @test count(centered.valid)==14
    @test count(centered.support.x.structural_supported)==6
    @test count(centered.support.y.structural_supported)==4
    @test count(isfinite,vorticity(centered))==0
    @test eligible[2,3]===false
    # Explicit array eligibility is authoritative. A centered secant does not
    # use the center value even though it requires center eligibility.
    u[2,3]=NaN
    full=flow_derivatives(x,y,u,v;valid=trues(3,5),return_support=true)
    @test full.dudx[2,3]==2
    @test full.dudy[2,3]==3
    @test full.support.x.first_index[2,3]==2 && full.support.x.second_index[2,3]==4
    @test full.support.finite.dudx[2,3] && full.support.finite.dudy[2,3]
    @test full.support.x.structural_supported[2,2] && !full.support.finite.dudx[2,2]
    implicit=flow_derivatives(x,y,u,v;return_support=true)
    @test !implicit.valid[2,3]
    @test !implicit.support.x.structural_supported[2,3]
    # A support snapshot owns its masks/indices rather than modifying inputs.
    d.support.center_eligible[1,1]=false
    d.support.x.first_index[1,1]=99
    @test eligible[1,1]
    @test x==collect(0.:4.) && y==collect(0.:2.)
end

@testset "Derivative geometry refusal and component arithmetic availability" begin
    u=ones(3,3);v=zeros(3,3);y=[0.,1.,2.]
    before=copy(u)
    for x in ([0.,NaN,2.],[0.,Inf,2.],[0.,0.,2.],[0.,2.,1.],
              [-floatmax(Float64),0.,floatmax(Float64)],
              [-floatmax(Float32),0f0,floatmax(Float32)],
              [typemin(Int),0,typemax(Int)])
        for policy in (:available,:centered),describe in (false,true)
            @test_throws ArgumentError flow_derivatives(x,y,u,v;valid=falses(3,3),stencil=policy,return_support=describe)
        end
    end
    @test u==before
    @test_throws ArgumentError flow_derivatives([0.],y,ones(3,1),zeros(3,1))
    @test_throws ArgumentError flow_derivatives(y,[0.],ones(1,3),zeros(1,3))
    @test_throws ArgumentError flow_derivatives(y,y,u,v;stencil=:unknown)
    @test_throws DimensionMismatch flow_derivatives([0.,1.],y,u,v)
    @test_throws DimensionMismatch flow_derivatives(y,y,u,v;valid=trues(2,2))
    # Adjacent integer spans can each fit even when a centered span overflows.
    @test_throws ArgumentError flow_derivatives([-typemax(Int),0,typemax(Int)],y,u,v)
    tiny=nextfloat(0.0);x=[0.,tiny];y2=[0.,1.]
    u=[0. tiny;0. tiny];v=[0. 0.;1. 1.]
    ordinary=flow_derivatives(x,y2,u,v)
    d=flow_derivatives(x,y2,u,v;return_support=true)
    @test all(==(1),ordinary.dudx)
    @test all(==(1),d.dudx)
    @test all(d.support.x.structural_supported)
    @test all(d.support.x.span_available)
    @test all(==(tiny),d.support.x.signed_span)
    @test !any(d.support.x.weights_available)
    @test all(isnan,d.support.x.first_weight) && all(isnan,d.support.x.second_weight)
    @test all(d.support.finite.dudx) && all(d.support.finite.dvdx)
    @test all(d.support.y.weights_available)
    @test all(iszero,d.dvdx)
    # Native Float32 component subtraction is preserved rather than promoted
    # into a fabricated finite quotient; the geometry remains usable.
    x=Float32[0,1,2];y=Float32[0,1,2]
    u=repeat(reshape(Float32[-floatmax(Float32),0,floatmax(Float32)],1,3),3,1)
    v=zeros(Float32,3,3)
    d=flow_derivatives(x,y,u,v;return_support=true)
    @test d.dudx[2,2]==Inf
    @test d.support.x.structural_supported[2,2] && d.support.x.weights_available[2,2]
    @test !d.support.finite.dudx[2,2]
    @test d.support.finite.dvdx[2,2]
    @test d.dudx[2,1]==Float64(floatmax(Float32))
    # Integer component overflow becomes unavailable instead of wrapping into
    # a finite false derivative, independently of checked coordinate spans.
    u=repeat(reshape([typemin(Int),0,typemax(Int)],1,3),3,1)
    d=flow_derivatives([0,1,2],[0,1,2],u,zeros(Int,3,3);return_support=true)
    @test isnan(d.dudx[2,2])
    @test d.support.x.structural_supported[2,2] && !d.support.finite.dudx[2,2]
    @test d.support.finite.dvdx[2,2]
    # Promoted descriptors retain BigFloat spans without changing the native
    # quotient or the existing Float64 derivative output convention.
    setprecision(BigFloat,128) do
        x=BigFloat[0,1,3];y=BigFloat[0,2,5]
        u=[2xx+3yy for yy in y,xx in x]
        d=flow_derivatives(x,y,u,zero.(u);return_support=true)
        @test eltype(d.dudx)==Float64
        @test eltype(d.support.x.signed_span)==BigFloat
        @test d.support.x.signed_span[2,2]==3
        @test d.support.x.first_weight[2,2]==-BigFloat(1)/3
        @test all(==(2),d.dudx) && all(==(3),d.dudy)
    end
    # Huge exact coordinate origins must not be rounded into repeated Float64
    # coordinates before native subtraction. BigInt components are unbounded.
    origin=big(2)^1000
    x=origin.+BigInt[0,1,3];y=BigInt[0,2,5]
    u=[2(xx-origin)+3yy for yy in y,xx in x]
    d=flow_derivatives(x,y,u,zero.(u);return_support=true)
    @test all(==(2),d.dudx) && all(==(3),d.dudy)
    @test eltype(d.support.x.signed_span)==BigFloat
    @test d.support.x.signed_span[2,2]==3
    @test d.support.x.first_index[2,2]==1 && d.support.x.second_index[2,2]==3
end

function derivative_support_result(;scale=nothing)
    x=collect(0.:4.);y=collect(0.:3.)
    u=[2xx+3yy for yy in y,xx in x];v=[5xx-2yy for yy in y,xx in x]
    PIVResult(x,y,u,v,ones(4,5),ones(4,5),ones(4,5),ones(4,5),falses(4,5),falses(4,5),PIVParameters(),nothing,scale)
end
@testset "Derivative result flags, physical units and derived wrappers" begin
    r=derivative_support_result()
    r.outliers[2,3]=true;r.mask[3,4]=true;r.u[2,5]=Inf
    d=flow_derivatives(r;return_support=true)
    @test !d.valid[2,3] && !d.valid[3,4] && !d.valid[2,5]
    admitted=flow_derivatives(r;include_invalid=true,return_support=true)
    @test admitted.valid[2,3] && !admitted.valid[3,4] && !admitted.valid[2,5]
    @test r.outliers[2,3] && r.mask[3,4] && r.u[2,5]==Inf
    r=derivative_support_result(;scale=PhysicalScale(;pixel_size=.02,dt=.1,length_unit="mm",time_unit="s"))
    raw=flow_derivatives(r;return_support=true)
    scaled=flow_derivatives(physical(r);return_support=true)
    again=flow_derivatives(physical(physical(r));return_support=true)
    @test raw.dudx==fill(2.,4,5)
    @test scaled.dudx≈fill(20.,4,5)
    @test scaled.dudy≈fill(30.,4,5)
    @test scaled.support.x.signed_span≈.02 .* raw.support.x.signed_span
    @test scaled.support.y.signed_span≈.02 .* raw.support.y.signed_span
    @test scaled.support.x.first_index==raw.support.x.first_index
    @test isequal(scaled.dudx,again.dudx)
    @test isequal(scaled.support.x.signed_span,again.support.x.signed_span)
    centered=flow_derivatives(r;stencil=:centered,return_support=true)
    @test isequal(vorticity(r;stencil=:centered),vorticity(centered))
    @test isequal(divergence(r;stencil=:centered,return_support=true),divergence(centered))
    @test isequal(strain_rate(r;stencil=:centered).magnitude,strain_rate(centered).magnitude)
    @test isequal(swirling_strength(r;stencil=:centered),swirling_strength(centered))
    @test isequal(q_criterion(r;stencil=:centered),q_criterion(centered))
end
