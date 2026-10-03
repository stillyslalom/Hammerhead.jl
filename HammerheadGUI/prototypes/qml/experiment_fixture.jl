# Real saved file-based recipe for tests/smoke; never an automatic user input.
function experiment_fixture(directory)
    mkpath(directory)
    image=[mod(37i+13j+7i*j,251)/250 for i in 1:64,j in 1:64]
    files=[joinpath(directory,"frame-$i.png") for i in 1:4]
    for (i,path) in enumerate(files)
        Hammerhead.FileIO.save(path,Hammerhead.Gray.(circshift(image,(i-1,2(i-1)))))
    end
    mask=falses(64,64);mask[1:14,1:14].=true
    passes=multipass_parameters([32,16];max_iterations=1,uod_enable=false)
    recipe=PIVRecipe(passes;threaded=false,roi=ROI(5:60,7:62),mask,
        preprocessing=[PreprocessStep(:highpass_filter;sigma=2),PreprocessStep(:intensity_cap)],
        scale=PhysicalScale(pixel_size=.02,dt=.001,length_unit="mm",time_unit="s"))
    record=ExperimentRecord([(files[i],files[i+1]) for i in 1:3],recipe)
    path=save_experiment(joinpath(directory,"experiment.jld2"),record)
    (record=record,path=path,output=joinpath(directory,"vectors.jld2"),history=joinpath(directory,"history.jld2"))
end
