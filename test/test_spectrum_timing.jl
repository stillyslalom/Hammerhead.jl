using Hammerhead, Test

function spectrum_timing_field(value;scale=nothing,x=[0.,1.],y=[0.,1.])
    dims=(length(y),length(x))
    u=fill(Float64(value),dims);v=fill(-Float64(value),dims)
    PIVResult(x,y,u,v,ones(dims),zeros(dims),fill(NaN,dims),fill(NaN,dims),
        falses(dims),falses(dims),PIVParameters(),nothing,scale)
end
spectrum_exact(data,key;derived=true)=Hammerhead._timing_decode(data[key];derived)

@testset "Spectrum legacy outputs and explicit interval provenance" begin
    n=100;dt=0.01;signal=2sin.(2pi*10 .* ((0:n-1).*dt))
    for window in (:none,:hann)
        legacy=power_spectrum(signal;dt,window)
        annotated=power_spectrum(signal;dt,window,return_timing=true,time_unit="s")
        @test keys(legacy)==(:frequencies,:psd)
        @test keys(annotated)==(:frequencies,:psd,:timing)
        @test legacy.frequencies==annotated.frequencies && legacy.psd==annotated.psd
        f,psd=legacy
        @test f[argmax(psd)] ≈ 10
        @test annotated.timing["source"]=="provided_interval"
        @test annotated.timing["uniformity"]=="assumed_from_interval"
        @test annotated.timing["time_unit"]=="s"
        @test annotated.timing["first_time"]===annotated.timing["last_time"]===nothing
    end
    @test sum(power_spectrum(signal;dt,window=:none).psd)/ (n*dt) ≈ 2
    default=power_spectrum(signal)
    @test default==power_spectrum(signal;dt=1.0)
    report=power_spectrum(signal;return_timing=true).timing
    @test report["source"]=="default_interval" && report["time_unit"]===nothing
    @test report["time_unit_provenance"]=="unknown" && report["sample_count"]==n
    @test power_spectrum(signal;dt=Float32(0.125)).psd==power_spectrum(signal;dt=Float32(0.125),return_timing=true).psd
    owned_interval=big(2)
    interval_report=power_spectrum(signal;dt=owned_interval,return_timing=true).timing
    @test interval_report["applied_period"]==owned_interval && interval_report["applied_period"]!==owned_interval
    @test power_spectrum(signal;dt=big"0.125").psd isa Vector{BigFloat}
    @test_throws ArgumentError power_spectrum(signal;dt=big"0.125",return_timing=true)
    @test all(isnan,power_spectrum([1.,NaN,2.]).psd) # Existing nonfinite-signal behavior.
    for options in ((;dt=0),(;dt=Inf),(;window=:hamming),(;timing_atol=1),
                    (;timing_rtol=-1),(;timing_atol=true),(;time_unit=""),(;time_unit=1))
        @test_throws ArgumentError power_spectrum(signal;options...)
    end
end

@testset "Exact spectrum timelines and epoch/unit invariance" begin
    n=64;signal=sin.(2pi*8 .* ((0:n-1)./n))
    baseline=power_spectrum(signal;dt=0.25,window=:none)
    epochs=(0,big(typemax(Int64))+100,big(10)^100)
    for epoch in epochs
        times=[epoch+k//4 for k in 0:n-1]
        spectrum=power_spectrum(signal;sample_times=times,window=:none,return_timing=true,time_unit="s")
        @test spectrum.frequencies==baseline.frequencies && spectrum.psd==baseline.psd
        @test spectrum.timing["uniformity"]=="exact"
        @test spectrum_exact(spectrum.timing,"period")==1//4
        @test spectrum_exact(spectrum.timing,"first_time";derived=false)==epoch
        @test spectrum_exact(spectrum.timing,"last_time";derived=false)==epoch+(n-1)//4
        @test spectrum_exact(spectrum.timing,"max_interval_residual")==0
        @test spectrum_exact(spectrum.timing,"max_grid_residual")==0
    end
    for T in (Int64,Float32,Float64)
        times=T.(0:n-1)
        actual=power_spectrum(signal;sample_times=times)
        @test actual==power_spectrum(signal;dt=1.)
    end
    milliseconds=power_spectrum(signal;sample_times=250 .* collect(0:n-1),window=:none,return_timing=true,time_unit="ms")
    @test milliseconds.frequencies ≈ baseline.frequencies ./ 1000
    @test milliseconds.psd ≈ baseline.psd .* 1000
    @test milliseconds.timing["time_unit"]=="ms"
    times=collect(0.:0.25:(n-1)*0.25)
    report=power_spectrum(signal;sample_times=times,return_timing=true).timing
    saved=deepcopy(report);times[1]=10.
    @test report==saved && report["source"]=="provided_sample_times"
    @test report["time_unit"]===nothing
    @test all(v->!(v isa AbstractArray),values(report)) # Bounded scalar context, no timeline retained.
end

@testset "Uniformity bounds, cumulative drift and numerical range" begin
    signal=[0.,1.,0.,-1.]
    rounded=[0.,0.1,0.2,0.3]
    @test_throws ArgumentError power_spectrum(signal;sample_times=rounded)
    accepted=power_spectrum(signal;sample_times=rounded,timing_atol=1e-15,return_timing=true)
    @test accepted.timing["uniformity"]=="within_explicit_tolerance"
    @test spectrum_exact(accepted.timing,"max_interval_residual")>0
    @test accepted.frequencies==power_spectrum(signal;dt=accepted.timing["applied_period"]).frequencies
    @test_throws ArgumentError power_spectrum(zeros(1001);sample_times=collect(0.:0.1:100.))
    @test power_spectrum(zeros(1001);sample_times=collect(0.:0.1:100.),timing_atol=1e-13).psd==zeros(501)
    irregular=[0//1,1//1,2001//1000]
    boundary=power_spectrum(zeros(3);sample_times=irregular,timing_atol=1//2000,return_timing=true)
    @test spectrum_exact(boundary.timing,"max_interval_residual")==1//2000
    @test spectrum_exact(boundary.timing,"max_grid_residual")==1//2000
    @test_throws ArgumentError power_spectrum(zeros(3);sample_times=irregular,timing_atol=999//2000000)
    @test power_spectrum(zeros(3);sample_times=irregular,timing_rtol=1//2001).psd==zeros(2)
    @test_throws ArgumentError power_spectrum(zeros(3);sample_times=irregular,timing_rtol=1//2002)
    # Each interval lies within 0.002 of the endpoint-average period, but the
    # first half drifts 0.01 away from the global FFT grid.
    drifting=[0//1;cumsum([fill(1001//1000,10);fill(999//1000,10)])]
    @test maximum(abs.(diff(drifting).-1))<=1//500
    error=try power_spectrum(zeros(21);sample_times=drifting,timing_atol=1//500);nothing catch err;err end
    @test error isa ArgumentError && occursin("cumulative grid",sprint(showerror,error))
    for times in ([0,1,3,4],[0,1,1,2],[3,2,1,0],Any[0,1,true,3],Any[0,1,NaN,3],
                  Any[0,1,Inf,3],BigFloat[0,1,2,3],[0,1,2],nothing)
        times===nothing && continue
        @test_throws ArgumentError power_spectrum(signal;sample_times=times)
    end
    @test_throws ArgumentError power_spectrum(signal;sample_times=(0,1,2,3))
    @test_throws ArgumentError power_spectrum(signal;sample_times=0:3,dt=1)
    @test_throws ArgumentError power_spectrum([0.];sample_times=[0])
    for interval in (big(10)^400,1//big(10)^400,1//big(10)^310)
        @test_throws ArgumentError power_spectrum([0.,1.];sample_times=[0,interval])
    end
    @test_throws ArgumentError power_spectrum([0.,1.];sample_times=[0.,floatmax(Float64)])
    # Same exact tolerance applies after an arbitrary epoch translation.
    epoch=big(10)^100
    @test power_spectrum(zeros(3);sample_times=epoch .+ irregular,timing_atol=1//2000).frequencies==boundary.frequencies
    @test_throws ArgumentError power_spectrum(zeros(3);sample_times=epoch .+ irregular,timing_atol=999//2000000)
end

@testset "Result-spectrum compatibility and original fill populations" begin
    sequence=[spectrum_timing_field(sin(2pi*k/10)) for k in 0:99]
    times=collect(0:99).//100
    for component in (:u,:v),window in (:none,:hann)
        old=result_spectrum(sequence,1,1;dt=0.01,component,window)
        new=result_spectrum(sequence,1,1;sample_times=times,component,window,return_timing=true,time_unit="s")
        @test old.frequencies==new.frequencies && old.psd==new.psd
        @test new.timing["invalid_count"]==0 && new.timing["fill_policy"]=="error"
        @test new.timing["component"]==String(component) && new.timing["index"]==[1,1]
        @test new.timing["attached_scale"]===nothing
        @test keys(old)==(:frequencies,:psd)
    end
    @test power_spectrum(sequence,1,1;sample_times=times)==result_spectrum(sequence,1,1;sample_times=times)
    @test power_spectrum(sequence;index=(1,1),sample_times=times)==result_spectrum(sequence,1,1;sample_times=times)
    for options in ((;),(;dt=0.01,sample_times=times),(;component=:w),(;invalid=:drop))
        @test_throws ArgumentError result_spectrum(sequence,1,1;options...)
    end
    @test_throws ArgumentError result_spectrum(sequence,0,1;dt=.01)
    for mutate in (r->r.x[1]=NaN,r->r.x[2]=2.,r->resize!(r.x,1))
        changed=deepcopy(sequence);mutate(changed[end])
        @test_throws ArgumentError result_spectrum(changed,1,1;dt=.01)
    end
    broken=spectrum_timing_field(1.)
    badshape=PIVResult(broken.x,broken.y,ones(1,2),broken.v,broken.peak_ratio,broken.correlation_moment,
        broken.uncertainty_u,broken.uncertainty_v,broken.outliers,broken.mask,broken.parameters)
    @test_throws ArgumentError result_spectrum([sequence[1],badshape],1,1;dt=.01)
    badflag=PIVResult(broken.x,broken.y,broken.u,broken.v,broken.peak_ratio,broken.correlation_moment,
        broken.uncertainty_u,broken.uncertainty_v,broken.outliers,falses(1,2),broken.parameters)
    @test_throws ArgumentError result_spectrum([sequence[1],badflag],1,1;dt=.01)
    nonfinite_grid=[spectrum_timing_field(1.;x=[0.,Inf]) for _ in 1:2]
    @test_throws ArgumentError result_spectrum(nonfinite_grid,1,1;dt=.01)
    @test_throws ArgumentError result_spectrum(PIVResult[],1,1;dt=.01)
    scales=(PhysicalScale(pixel_size=.02,dt=.001,length_unit="mm",time_unit="s"),
        PhysicalScale(pixel_size=.02,dt=.002,length_unit="mm",time_unit="s"))
    scaled=[with_scale(r,scales[isodd(k) ? 1 : 2]) for (k,r) in enumerate(sequence)]
    @test_throws ArgumentError result_spectrum(scaled,1,1;sample_times=times)
    converted=physical.(scaled)
    spectrum=result_spectrum(converted,1,1;sample_times=times,return_timing=true,time_unit="ms")
    @test spectrum.timing["time_unit"]=="ms" && spectrum.timing["attached_scale"]["time_unit"]=="s"
    @test spectrum.timing["value_basis"]=="stored_component_no_conversion"
    @test spectrum.frequencies==power_spectrum([r.u[1,1] for r in converted];sample_times=times).frequencies
    @test spectrum.psd==power_spectrum([r.u[1,1] for r in converted];sample_times=times).psd
    for scale in (nothing,PhysicalScale(pixel_size=.03,dt=.001,length_unit="mm",time_unit="s"),
        PhysicalScale(pixel_size=.02,dt=.001,length_unit="cm",time_unit="s"),
        PhysicalScale(pixel_size=.02,dt=.001,length_unit="mm",time_unit="ms"))
        mixed=[with_scale(sequence[1],scales[1]),with_scale(sequence[2],scale)]
        @test_throws ArgumentError result_spectrum(mixed,1,1;dt=.01)
    end
    equal_scale=[with_scale(r,scales[1]) for r in sequence]
    @test result_spectrum(equal_scale,1,1;sample_times=times).frequencies[argmax(result_spectrum(equal_scale,1,1;sample_times=times).psd)] ≈ 10
    @test result_spectrum(equal_scale,1,1;dt=.01).psd==result_spectrum(sequence,1,1;dt=.01).psd
    info=result_spectrum(equal_scale,1,1;dt=.01,return_timing=true).timing
    @test info["source"]=="provided_interval" && info["invalid_count"]==0 && info["time_unit"]===nothing
    @test info["attached_scale"]["dt"]==.001 && info["applied_period"]==.01
    info["attached_scale"]["length_unit"]="changed"
    @test equal_scale[1].scale.length_unit=="mm"
    # Component-specific validity is preserved: invalid v does not remove u.
    component_specific=deepcopy(sequence);component_specific[1].v[1,1]=NaN
    @test result_spectrum(component_specific,1,1;dt=.01)==result_spectrum(sequence,1,1;dt=.01)
    @test_throws ArgumentError result_spectrum(component_specific,1,1;dt=.01,component=:v)
    affected=[spectrum_timing_field(k) for k in 1:5]
    affected[1].mask[1,1]=true;affected[3].outliers[1,1]=true;affected[5].u[1,1]=NaN
    before=deepcopy(affected)
    @test_throws ArgumentError result_spectrum(affected,1,1;sample_times=0:4)
    for (invalid,filled) in ((:mean,[3.,2.,3.,4.,3.]),(:interpolate,[2.,2.,3.,4.,4.]))
        got=result_spectrum(affected,1,1;sample_times=0:4,invalid,return_timing=true)
        @test got.psd==power_spectrum(filled).psd
        @test got.timing["sample_count"]==5 && got.timing["invalid_count"]==3
        @test got.timing["fill_policy"]==String(invalid)
        @test spectrum_exact(got.timing,"period")==1
        @test keys(result_spectrum(affected,1,1;dt=1,invalid))==(:frequencies,:psd)
    end
    @test all(isequal(a.u,b.u) && a.mask==b.mask && a.outliers==b.outliers for (a,b) in zip(affected,before))
    @test_throws ArgumentError result_spectrum(affected,1,1;sample_times=[0,1,2,4,5],invalid=:mean)
    all_bad=deepcopy(affected);foreach(r->r.mask[1,1]=true,all_bad)
    @test_throws ArgumentError result_spectrum(all_bad,1,1;dt=1,invalid=:mean)
end
