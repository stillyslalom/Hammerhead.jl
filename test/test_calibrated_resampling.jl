using Test,Hammerhead

# Independent scalar affine algebra; these oracles do not call the production
# transform/inverse helpers, interpolator or calibration fitter.
function resampling_oracle_inverse(matrix,offset,X,Y)
    a,b,c,d=Float64(matrix[1,1]),Float64(matrix[1,2]),Float64(matrix[2,1]),Float64(matrix[2,2])
    determinant=a*d-b*c
    dx,dy=Float64(X)-Float64(offset[1]),Float64(Y)-Float64(offset[2])
    ((d*dx-b*dy)/determinant,(-c*dx+a*dy)/determinant)
end
resampling_oracle_velocity(X,Y)=(3+.2X-.3Y,-2+.4X+.1Y)
resampling_oracle_intensity(X,Y)=7+2X-3Y
function resampling_affine_field(matrix,offset,x,y,::Type{T}=Float64;interval=1.) where {T}
    xx,yy=T.(x),T.(y);u=zeros(T,length(y),length(x));v=similar(u)
    for i in eachindex(yy),j in eachindex(xx)
        X=matrix[1,1]*xx[j]+matrix[1,2]*yy[i]+offset[1]
        Y=matrix[2,1]*xx[j]+matrix[2,2]*yy[i]+offset[2]
        U,V=resampling_oracle_velocity(X,Y)
        # Vector inverse has no positional offset. Pixel displacements are
        # the independently prescribed physical velocity times interval.
        u[i,j],v[i,j]=T.(resampling_oracle_inverse(matrix,zeros(2),interval*U,interval*V))
    end
    resampling_raw_field(xx,yy,u,v)
end
function resampling_raw_field(x,y,u,v;mask=falses(size(u)),outliers=falses(size(u)))
    T=eltype(u);z=zeros(T,size(u))
    PIVResult(x,y,u,v,ones(T,size(u)),z,fill(T(NaN),size(u)),fill(T(NaN),size(u)),
        outliers,mask,PIVParameters())
end
function resampling_source_snapshot(raw)
    [(name,deepcopy(getfield(raw,name))) for name in fieldnames(typeof(raw))]
end
function resampling_source_unchanged(raw,snapshot)
    all(isequal(getfield(raw,name),value) for (name,value) in snapshot)
end

@testset "Calibrated sampling independent affine basis" begin
    matrices=([1. 0;0 1], [0. -1;1 0], [-1. 0;0 1],
        [.25 0;0 2], [1. .5;0 1], [.8 -.6;1.2 1.6])
    for T in (Float32,Float64),matrix in matrices,interval in (nothing,T(.2))
        transform=PlanarTransform(T.(matrix),T[4,-7])
        A=Matrix(transform.matrix);b=Vector(transform.offset)
        raw=resampling_affine_field(A,b,[10,20,35],[21,30,50],T;
            interval=isnothing(interval) ? 1 : interval)
        snapshot=resampling_source_snapshot(raw)
        center=A*T[20,30]+b
        x=T.(center[1].+[-.25,0,.25]);y=T.(center[2].+[-.25,0,.25])
        sampled=resample_planar(raw,x,y;transform,length_unit="mm",dt=interval,
            time_unit=isnothing(interval) ? nothing : "s")
        @test all(sampled.available)
        @test eltype(sampled.u)===T
        @test eltype(sampled.x)===T
        @test size(sampled.u)==(3,3)
        atol=T===Float32 ? 8e-5 : 2e-12
        expected_u=[first(resampling_oracle_velocity(X,Y)) for Y in y,X in x]
        expected_v=[last(resampling_oracle_velocity(X,Y)) for Y in y,X in x]
        @test sampled.u≈expected_u atol=atol rtol=atol
        @test sampled.v≈expected_v atol=atol rtol=atol
        @test sampled.metadata.quantity== (isnothing(interval) ? :displacement : :velocity)
        @test sampled.metadata.component_unit== (isnothing(interval) ? "mm/frame" : "mm/s")
        @test sampled.metadata.geometry_type===Float64
        @test sampled.metadata.output_type===T
        @test sampled.metadata.applied_transform.matrix==Float64.(transform.matrix)
        @test sampled.metadata.method===:bilinear_point_sampling
        @test !(sampled isa PIVResult)
        @test !hasproperty(sampled,:uncertainty_u)
        @test resampling_source_unchanged(raw,snapshot)
        @test sampled.x !== x && sampled.y !== y
        @test sampled.metadata.source_axes.x !== raw.x
        intensity=T[resampling_oracle_intensity(A[1,1]*X+A[1,2]*Y+b[1],
            A[2,1]*X+A[2,2]*Y+b[2]) for Y in raw.y,X in raw.x]
        scalar=resample_image(intensity,x,y;transform,length_unit="mm",
            source_x=raw.x,source_y=raw.y)
        @test all(scalar.available) && eltype(scalar.values)===T
        @test scalar.values≈[resampling_oracle_intensity(X,Y) for Y in y,X in x] atol=atol rtol=atol
        descending=resampling_raw_field(reverse(raw.x),reverse(raw.y),
            reverse(raw.u;dims=(1,2)),reverse(raw.v;dims=(1,2)))
        back=resample_planar(descending,reverse(x),reverse(y);transform,length_unit="mm",dt=interval,
            time_unit=isnothing(interval) ? nothing : "s")
        @test back.u≈reverse(sampled.u;dims=(1,2)) atol=atol rtol=atol
        @test back.v≈reverse(sampled.v;dims=(1,2)) atol=atol rtol=atol
    end
end

@testset "Unequal PIV and scalar resolutions share physical coordinates" begin
    ptransform=PlanarTransform([.5 0;0 -.25],[-5.5,11.25])
    itransform=PlanarTransform([.125 0;0 -.0625],[-.125,6.0625])
    raw=resampling_affine_field(Matrix(ptransform.matrix),Vector(ptransform.offset),
        [11.,17,25,39],[21.,27,35,45])
    image=[resampling_oracle_intensity(.125*j-.125,-.0625*i+6.0625) for i in 1:97,j in 1:113]
    x=[0.,1,4,9,14];y=[0.,2,5,6]
    vectors=resample_planar(raw,x,y;transform=ptransform,length_unit="mm")
    scalar=resample_image(image,x,y;transform=itransform,length_unit="mm",value_unit="a.u.")
    @test all(vectors.available) && all(scalar.available)
    @test vectors.x==scalar.x && vectors.y==scalar.y
    @test vectors.u≈[first(resampling_oracle_velocity(X,Y)) for Y in y,X in x]
    @test scalar.values≈[resampling_oracle_intensity(X,Y) for Y in y,X in x]
    @test scalar.metadata.source_size==(97,113)
    @test scalar.metadata.value_unit=="a.u."
    @test scalar.metadata.quantity===:scalar_sample
    original=copy(image)
    crop=view(image,11:21,7:17)
    cx=[.75,1.,2.];cy=[4.75,5.,5.375]
    cropped=resample_image(crop,cx,cy;transform=itransform,length_unit="mm",
        source_x=7:17,source_y=11:21)
    @test all(cropped.available)
    @test cropped.values≈[resampling_oracle_intensity(X,Y) for Y in cy,X in cx]
    @test cropped.metadata.source_axes.x==Float64.(7:17)
    cropped.values .= -99
    cropped.metadata.source_axes.x .= -99
    @test image==original

    # This bilinear polynomial independently checks all four corner weights.
    polynomial=[1+2X+3Y+4X*Y for Y in (0.,1.),X in (0.,1.)]
    q=resample_image(polynomial,[.2,.7],[.3,.8];transform=PlanarTransform([1. 0;0 1],[0.,0.]),
        length_unit="px",source_x=[0.,1],source_y=[0.,1])
    @test q.values≈[1+2X+3Y+4X*Y for Y in (.3,.8),X in (.2,.7)]
    @test all(q.contributor_count.==4)
    # Point samples do not claim pixel-area conservation or anti-aliasing.
    checkerboard=[Float64(isodd(i+j)) for i in 1:8,j in 1:8]
    points=resample_image(checkerboard,1:2:7,1:2:7;
        transform=PlanarTransform([1. 0;0 1],[0.,0.]),length_unit="px")
    @test all(points.available) && all(iszero,points.values)
    @test sum(checkerboard)/length(checkerboard)==.5
end

@testset "Positive contributors and invalid-support populations" begin
    identity=PlanarTransform([1. 0;0 1],[0.,0.])
    x=y=[0.,.5,1.]
    expected=[X>0 && Y<1 for Y in y,X in x]
    counts=UInt8[(X in (0.,1.) ? 1 : 2)*(Y in (0.,1.) ? 1 : 2) for Y in y,X in x]
    for kind in (:mask,:outlier,:u,:v,:overlap)
        u=ones(2,2);v=fill(2.,2,2);mask=falses(2,2);outliers=falses(2,2)
        kind in (:mask,:overlap) && (mask[1,2]=true)
        kind in (:outlier,:overlap) && (outliers[1,2]=true)
        kind in (:u,:overlap) && (u[1,2]=NaN)
        kind===:v && (v[1,2]=Inf)
        raw=resampling_raw_field([0.,1],[0.,1],u,v;mask,outliers)
        snapshot=resampling_source_snapshot(raw)
        result=resample_planar(raw,x,y;transform=identity,length_unit="px")
        @test result.available== .!expected
        @test result.contributor_count==counts
        @test result.masked_support==(kind in (:mask,:overlap) ? expected : falses(3,3))
        @test result.outlier_support==(kind in (:outlier,:overlap) ? expected : falses(3,3))
        @test result.nonfinite_support==(kind in (:u,:v,:overlap) ? expected : falses(3,3))
        @test all(isnan,result.u[expected]) && all(isnan,result.v[expected])
        @test all(result.u[.!expected].==1) && all(result.v[.!expected].==2)
        admitted=resample_planar(raw,x,y;transform=identity,length_unit="px",include_invalid=true)
        @test admitted.available==(kind===:outlier ? trues(3,3) : .!expected)
        @test admitted.outlier_support==result.outlier_support
        @test resampling_source_unchanged(raw,snapshot)
    end
    raw=resampling_raw_field([0.,1],[0.,1],ones(2,2),ones(2,2))
    outer=[-.1,0.,.5,1.,1.1]
    result=resample_planar(raw,outer,outer;transform=identity,length_unit="px")
    outside=[X<0 || X>1 || Y<0 || Y>1 for Y in outer,X in outer]
    @test result.outside==outside && result.available== .!outside
    @test all(iszero,result.contributor_count[outside])
    @test !any(result.masked_support) && !any(result.nonfinite_support)
    image=[1. NaN;3. 4.];mask=Bool[0 1;0 0]
    scalar=resample_image(image,x,y;transform=identity,length_unit="px",mask,
        source_x=[0.,1],source_y=[0.,1])
    @test scalar.available== .!expected
    @test scalar.masked_support==expected && scalar.nonfinite_support==expected
    @test scalar.contributor_count==counts
end

@testset "Singleton exact coverage and arithmetic availability" begin
    identity=PlanarTransform([1. 0;0 1],[0.,0.])
    raw=resampling_raw_field([4.],[3.,8.],reshape([2.,7.],2,1),reshape([4.,9.],2,1))
    result=resample_planar(raw,[prevfloat(4.),4.,nextfloat(4.)],[3.,5.,8.];
        transform=identity,length_unit="px")
    @test result.available==Bool[0 1 0;0 1 0;0 1 0]
    @test result.contributor_count==UInt8[0 1 0;0 2 0;0 1 0]
    @test result.u[:,2]≈[2.,4.,7.]
    image=resample_image(reshape([2.,7.],2,1),[4.],[3.,5.,8.];
        transform=identity,length_unit="px",source_x=[4.],source_y=[3.,8.])
    @test image.values[:,1]≈[2.,4.,7.]
    point=resample_image(reshape([7.],1,1),[1.,2.],[1.,2.];transform=identity,length_unit="px")
    @test point.available==Bool[1 0;0 0]
    @test point.contributor_count==UInt8[1 0;0 0]

    raw32=resampling_raw_field(Float32[0,1],Float32[0,1],fill(floatmax(Float32),2,2),ones(Float32,2,2))
    scaled=resample_planar(raw32,Float32[0,2],Float32[0,2];
        transform=PlanarTransform(Float32[2 0;0 2],Float32[0,0]),length_unit="mm")
    @test !any(scaled.available) && all(scaled.arithmetic_failure)
    @test all(isnan,scaled.u) && all(isnan,scaled.v)
    @test !any(scaled.nonfinite_support) && !any(scaled.outside)
    raw=resampling_raw_field([0.,1],[0.,1],ones(2,2),ones(2,2))
    divided=resample_planar(raw,[0.],[0.];transform=identity,length_unit="mm",
        dt=nextfloat(0.),time_unit="s")
    @test divided.arithmetic_failure[1] && !divided.available[1]
    tiny=nextfloat(0.)
    @test tiny*tiny==0 && Rational{BigInt}(tiny)^2>0
    raw.u[2,2]=NaN;raw.mask[2,2]=true
    underflow=resample_planar(raw,[tiny],[tiny];transform=identity,length_unit="px")
    @test underflow.contributor_count[1]==4
    @test underflow.arithmetic_failure[1] && !underflow.available[1]
    @test underflow.nonfinite_support[1] && underflow.masked_support[1]
    @test !underflow.outside[1]
    scalar_underflow=resample_image(raw.u,[tiny],[tiny];transform=identity,length_unit="px",
        source_x=[0.,1],source_y=[0.,1],mask=raw.mask)
    @test scalar_underflow.contributor_count[1]==4 && scalar_underflow.arithmetic_failure[1]
    @test scalar_underflow.masked_support[1] && scalar_underflow.nonfinite_support[1]
end

@testset "Malformed calibration, geometry, units and double scale refusal" begin
    identity=PlanarTransform([1. 0;0 1],[0.,0.])
    raw=resampling_raw_field([0.,1],[0.,1],ones(2,2),ones(2,2))
    sample(r=raw;x=[0.,1],y=[0.,1],transform=identity,kwargs...)=
        resample_planar(r,x,y;transform,length_unit="mm",kwargs...)
    for axis in (Float64[],[0.,NaN],[0.,Inf],[0.,0.],[0.,2.,1.])
        @test_throws ArgumentError sample(;x=axis)
        @test_throws ArgumentError sample(;y=axis)
        malformed=resampling_raw_field(axis,[0.,1],ones(2,length(axis)),ones(2,length(axis)))
        @test_throws ArgumentError sample(malformed)
        @test_throws ArgumentError resample_image(ones(2,length(axis)),[0.],[0.];
            transform=identity,length_unit="mm",source_x=axis,source_y=[0.,1])
    end
    for transform in (PlanarTransform([1. 2;2 4],[0.,0.]),
                      PlanarTransform([NaN 0;0 1],[0.,0.]),
                      PlanarTransform([1. 0;0 1],[Inf,0.]),
                      PlanarTransform(Float16[1 0;0 1],Float16[0,0]),
                      PlanarTransform(BigFloat[1 0;0 1],BigFloat[0,0]))
        @test_throws ArgumentError sample(;transform)
        @test_throws ArgumentError resample_image(ones(2,2),[0.],[0.];transform,length_unit="mm")
    end
    for delay in (0.,-1.,Inf,NaN,true)
        @test_throws ArgumentError sample(;dt=delay,time_unit="s")
    end
    @test_throws ArgumentError sample(;dt=.2)
    @test_throws ArgumentError sample(;time_unit="s")
    @test_throws ArgumentError sample(;dt=.2,time_unit="")
    @test_throws ArgumentError sample(;coordinate_frame=" ")
    @test_throws ArgumentError resample_planar(raw,[0.],[0.];transform=identity,length_unit="")
    @test_throws ArgumentError resample_image(ones(2,2),[0.],[0.];transform=identity,
        length_unit="mm",value_unit="")
    @test_throws DimensionMismatch resample_image(ones(2,2),[0.],[0.];transform=identity,
        length_unit="mm",source_x=[0.])
    @test_throws DimensionMismatch resample_image(ones(2,2),[0.],[0.];transform=identity,
        length_unit="mm",mask=zeros(2,2))
    @test_throws DimensionMismatch resample_image(ones(2,2),[0.],[0.];transform=identity,
        length_unit="mm",mask=falses(1,2))
    malformed=resampling_raw_field([0.,1],[0.,1],ones(1,2),ones(1,2))
    @test_throws DimensionMismatch sample(malformed)
    scaled=with_scale(raw,PhysicalScale(;pixel_size=.2,dt=.1,length_unit="mm",time_unit="s"))
    @test_throws ArgumentError sample(scaled)
    @test_throws ArgumentError sample(physical(scaled))
    unchanged=resampling_source_snapshot(raw)
    x=[0.,.5,1.];y=copy(x)
    sampled=sample(;x,y,coordinate_frame="shared plane")
    @test sampled.metadata.coordinate_frame=="shared plane"
    sampled.x .= 100;sampled.y .= 100;sampled.u .= 100
    sampled.metadata.source_axes.x .= 100
    @test x==[0.,.5,1.] && y==[0.,.5,1.]
    @test resampling_source_unchanged(raw,unchanged)
    integer_image=resample_image([1 2;3 4],[1.,1.5,2.],[1.,1.5,2.];
        transform=identity,length_unit="px")
    @test integer_image.values≈[1. 1.5 2;2 2.5 3;3 3.5 4]
    @test all(integer_image.available)
    # Target points outside the inverse arithmetic range are arithmetic failures,
    # not a claim of geometric exclusion or contributing source pixels.
    tiny_map=PlanarTransform([floatmin(Float64) 0;0 1.],[0.,0.])
    overflow=resample_image(ones(2,2),[floatmax(Float64)],[1.];
        transform=tiny_map,length_unit="mm")
    @test overflow.arithmetic_failure[1] && !overflow.outside[1]
    @test overflow.contributor_count[1]==0 && !overflow.available[1]
end

@testset "Represented query geometry and finite tiny affine maps" begin
    identity=PlanarTransform([1. 0;0 1],[0.,0.])
    image=[1. 2.;3. 4.]
    # Geometry is explicitly Float64. The returned BigFloat labels must record
    # that represented query rather than the more precise original request.
    requested=BigFloat(1)+BigFloat(2)^(-60)
    result=resample_image(image,BigFloat[requested],BigFloat[1];transform=identity,length_unit="px")
    @test result.x==BigFloat[Float64(requested)]
    @test result.x[1]!=requested
    @test result.available[1] && result.values[1]==1
    large=BigInt(2)^53+1
    # A singleton original axis and requested label are both reduced to the same
    # Float64 coordinate; metadata and returned coordinates describe that fact.
    represented=resample_image(reshape([7.],1,1),[large],[1];transform=identity,
        length_unit="px",source_x=[large],source_y=[1])
    @test represented.x==[Float64(large)]
    @test represented.metadata.source_axes.x==[Float64(large)]
    @test represented.values[1]==7 && represented.available[1]
    tiny=BigFloat(2)^(-1100)
    @test tiny>0 && Float64(tiny)==0
    @test_throws ArgumentError resample_image(reshape([7.],1,1),[tiny],[1];
        transform=identity,length_unit="px",source_x=[tiny],source_y=[1])
    @test_throws ArgumentError resample_image(image,[tiny],[1];transform=identity,length_unit="px")
    @test_throws ArgumentError resample_image(image,[BigInt(2)^53,BigInt(2)^53+1],[1];
        transform=identity,length_unit="px")
    # An underflowing naive determinant is not a singular affine map. Its
    # inverse is representable, and exact-node samples still have one corner.
    small=1e-180
    @test small*small==0
    transform=PlanarTransform([small 0;0 small],[0.,0.])
    raw=resampling_raw_field([0.,1.],[0.,1.],ones(2,2),fill(2.,2,2))
    result=resample_planar(raw,[0.],[0.];transform,length_unit="m")
    @test result.available[1] && result.contributor_count[1]==1
    @test result.u[1]≈small && result.v[1]≈2small
    scalar=resample_image(image,[0.],[0.];transform,length_unit="m",source_x=[0.,1],source_y=[0.,1])
    @test scalar.available[1] && scalar.values[1]==1
    inverse_overflow=PlanarTransform([nextfloat(0.) 0;0 1.],[0.,0.])
    @test_throws ArgumentError resample_planar(raw,[0.],[0.];transform=inverse_overflow,length_unit="m")
    @test_throws ArgumentError resample_image(image,[0.],[0.];transform=inverse_overflow,length_unit="m")
    # A discarded masked value must not create a spurious arithmetic failure.
    mask=BitMatrix([1 0;0 0].!=0)
    huge=resampling_raw_field([0.,1],[0.,1],fill(floatmax(Float64),2,2),ones(2,2);mask)
    excluded=resample_planar(huge,[.5],[.5];transform=PlanarTransform([2. 0;0 2],[0.,0.]),length_unit="m")
    @test excluded.masked_support[1] && !excluded.available[1]
    @test !excluded.arithmetic_failure[1]
    @test excluded.contributor_count[1]==4
end
