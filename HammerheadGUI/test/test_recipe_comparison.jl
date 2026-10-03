using Test, HammerheadGUI
using HammerheadGUI.Hammerhead, HammerheadGUI.GLMakie
using FileIO: save
using ImageCore: Gray, N0f8

@testset "Saved recipe representative-pair GUI comparison" begin
    C=HammerheadGUI.Controllers
    @test RecipeComparisonController().state[]===:empty
    @test_throws ArgumentError RecipeComparisonController(;pair_indices=(true,1))
    @test_throws ArgumentError RecipeComparisonController(;basis=:interpolated)
    @test_throws ArgumentError compare!(RecipeComparisonController();async=false)
    @test_throws ArgumentError save_comparison_report!(RecipeComparisonController(),"unused.toml")
    mktempdir() do dir
        image=[mod(37i+13j+7i*j,251)/250 for i in 1:48,j in 1:48]
        files=[joinpath(dir,"frame-$i.png") for i in 1:3]
        for (file,data) in zip(files,(image,circshift(image,(1,2)),circshift(image,(2,3))))
            save(file,Gray{N0f8}.(data))
        end
        scale=PhysicalScale(pixel_size=.25,dt=.5,length_unit="mm",time_unit="s")
        p=PIVParameters(window_size=16,overlap=8,uod_enable=false,max_iterations=1,uncertainty=true)
        q=PIVParameters(window_size=16,overlap=8,uod_enable=false,uod_threshold=3,max_iterations=1,uncertainty=true)
        a=ExperimentRecord([(files[1],files[2]),(files[2],files[3])],
            PIVRecipe(p;mask=falses(48,48),scale,threaded=false))
        b=ExperimentRecord([(files[2],files[3]),(files[1],files[2])],
            PIVRecipe(q;mask=falses(48,48),scale,threaded=false))
        a_path=joinpath(dir,"before.jld2");b_path=joinpath(dir,"after.jld2")
        save_experiment(a_path,a);save_experiment(b_path,b)
        extra=joinpath(dir,"production-output.jld2");write(extra,"preserve output")
        cc=RecipeComparisonController(a_path,b_path;pair_indices=(1,2),protected_paths=[extra])
        @test cc.before[]!==a && cc.before[].recipe.mask!==a.recipe.mask
        @test cc.pair_indices[]==(1,2) && cc.basis[]===:pixels && !cc.allow_environment_change[]
        @test cc.before[].recipe.passes==a.recipe.passes && cc.after[].recipe.scale.dt==.5
        @test occursin(files[1],comparison_summary(cc;section=:request))
        before=cc.before[];choices=cc.pair_indices[]
        invalid=joinpath(dir,"invalid.jld2");write(invalid,"invalid")
        @test_throws Exception open_comparison_record!(cc,:before,invalid)
        @test cc.before[]===before && cc.pair_indices[]==choices
        @test_throws ArgumentError open_comparison_record!(cc,:unknown,a)
        @test_throws ArgumentError set_comparison_pairs!(cc,1,99)
        @test cc.pair_indices[]==choices
        cc.running[]=true
        @test_throws ArgumentError open_comparison_record!(cc,:after,b)
        @test_throws ArgumentError set_comparison_pairs!(cc,1,2)
        @test_throws ArgumentError compare!(cc)
        @test_throws ArgumentError open_comparison_report!(cc,"unused.toml")
        @test_throws ArgumentError save_comparison_report!(cc,"unused.toml")
        cc.running[]=false

        @testset "Frozen request, exact ordered pair and last-report provenance" begin
            # Both observers attempt reentry after the busy guard is established.
            refused=Symbol[]
            error_listener=C.on(cc.error) do _
                try compare!(cc;async=false) catch err
                    err isa ArgumentError && push!(refused,:error)
                end
            end
            running_listener=C.on(cc.running) do busy
                busy || return
                try open_comparison_record!(cc,:before,a) catch err
                    err isa ArgumentError && push!(refused,:running)
                end
                # Direct writes describe the next attempt, never the active one.
                cc.basis[]=:physical
                cc.pair_indices[]=(2,1)
                cc.allow_environment_change[]=true
                cc.before[]=deepcopy(b)
            end
            compare!(cc;async=false)
            C.off(error_listener);C.off(running_listener)
            @test cc.state[]===:completed && cc.error[]===nothing && !cc.running[] && cc.task[]===nothing
            @test :error in refused && :running in refused
            first_report=cc.report[]
            data=pair_comparison_data(first_report)
            @test data["basis"]["mode"]=="pixels" && !data["provenance"]["allow_environment_change"]
            @test data["provenance"]["before"]["recipe_id"]==a.recipe.recipe_id
            @test data["provenance"]["before"]["pair_index"]==1 && data["provenance"]["after"]["pair_index"]==2
            @test isempty(cc.before[].runs) && isempty(cc.after[].runs) && isempty(load_experiment(a_path).runs)
            @test occursin("Before report: pair 1",comparison_summary(cc))
            @test occursin("uod_threshold",comparison_summary(cc;section=:settings))
            @test occursin("Native-grid populations",comparison_summary(cc;section=:populations))
            @test occursin("core source sha256",comparison_summary(cc;section=:provenance))

            open_comparison_record!(cc,:before,a)
            open_comparison_record!(cc,:after,b)
            cc.basis[]=:pixels;cc.allow_environment_change[]=false
            set_comparison_pairs!(cc,1,1) # different selected content despite valid indices
            compare!(cc;async=false)
            @test cc.state[]===:failed && cc.error[] isa ArgumentError && cc.report[]===first_report
            @test occursin("Previous report",cc.status[]) && occursin("Before report: pair 1",comparison_summary(cc))
            @test occursin("After report: pair 2",comparison_summary(cc))
            before_bytes=read(files[1]);extra_bytes=read(extra);record_bytes=read(a_path)
            @test_throws ArgumentError save_comparison_report!(cc,files[1])
            @test_throws ArgumentError save_comparison_report!(cc,extra)
            @test_throws ArgumentError save_comparison_report!(cc,a_path)
            @test read(files[1])==before_bytes && read(extra)==extra_bytes && read(a_path)==record_bytes
            saved=joinpath(dir,"comparison.toml")
            @test save_comparison_report!(cc,saved)==saved
            loaded=RecipeComparisonController()
            open_comparison_report!(loaded,saved)
            @test loaded.before[]===nothing && loaded.after[]===nothing
            @test pair_comparison_data(loaded.report[])==data
            @test occursin("Original inputs have not been reverified",comparison_summary(loaded))
            old=loaded.report[];origin=loaded.report_origin[]
            invalid_toml=joinpath(dir,"bad-report.toml");write(invalid_toml,"bad = true")
            @test_throws ArgumentError open_comparison_report!(loaded,invalid_toml)
            @test loaded.report[]===old && loaded.report_origin[]==origin
            sentinel=joinpath(dir,"preserve-report.toml");write(sentinel,"preserve")
            loaded.report[]._data["basis"]["unit"]="edited"
            @test_throws ArgumentError save_comparison_report!(loaded,sentinel)
            @test read(sentinel,String)=="preserve"
            open_comparison_report!(loaded,saved)

            script=joinpath(dir,"do-not-execute.jl");write(script,"error(\"not executed\")")
            scripted=ExperimentRecord([(files[1],files[2])],PIVRecipe(p;external_preprocess=ScriptReference(script;entrypoint="prepare(image)"),threaded=false))
            open_comparison_record!(loaded,:before,scripted)
            script_bytes=read(script)
            @test_throws ArgumentError save_comparison_report!(loaded,script)
            @test read(script)==script_bytes

            set_comparison_pairs!(cc,1,2)
            cc.basis[]=:physical
            compare!(cc;async=true)
            task=cc.task[]
            task===nothing || wait(task)
            @test cc.state[]===:completed && !cc.running[] && cc.task[]===nothing
            @test pair_comparison_data(cc.report[])["basis"]["unit"]=="mm/s"

            # A successful controlled rerun can have no common grid nodes.
            shifted_grid=ExperimentRecord([(files[1],files[2])],PIVRecipe(
                PIVParameters(window_size=24,overlap=16,max_iterations=1,uod_enable=false);
                scale,threaded=false))
            disjoint=RecipeComparisonController(a,shifted_grid)
            compare!(disjoint;async=false)
            @test disjoint.state[]===:completed && disjoint.error[]===nothing
            absent=pair_comparison_data(disjoint.report[])["common"]["velocity_difference"]
            @test !absent["available"] && absent["reason_code"]=="no_common_nodes"
            @test occursin("unavailable",comparison_summary(disjoint))

            @testset "Offscreen readable pages and invalid visible Run request" begin
                fig=recipe_comparison(cc)
                @test size(colorbuffer(fig;px_per_unit=1))==(800,1100)
                boxes=filter(block->block isa Textbox,fig.content)
                @test length(boxes)==2
                run_button=only(filter(block->block isa Button && block.label[]=="compare selected pair",fig.content))
                retained=cc.report[]
                boxes[1].stored_string[]="not a pair"
                run_button.clicks[]+=1
                @test cc.report[]===retained && !cc.running[] && occursin("must be integers",cc.status[])
                boxes[1].stored_string[]="99"
                run_button.clicks[]+=1
                @test cc.report[]===retained && !cc.running[] && occursin("outside",cc.status[])
                boxes[1].stored_string[]="1"
                report_menu=last(filter(block->block isa Menu,fig.content))
                report_menu.i_selected[]=3 # recorded settings section
                @test any(block->block isa Label && occursin("uod_threshold",block.text[]),fig.content)
                if haskey(ENV,"HAMMERHEAD_COMPARISON_SCREENSHOT")
                    GLMakie.save(replace(ENV["HAMMERHEAD_COMPARISON_SCREENSHOT"],".png"=>"-settings.png"),fig)
                end
                report_menu.i_selected[]=4 # populations span multiple pages
                first_page=copy(colorbuffer(fig;px_per_unit=1))
                next_button=only(filter(block->block isa Button && block.label[]=="next",fig.content))
                next_button.clicks[]+=1
                @test colorbuffer(fig;px_per_unit=1)!=first_page
                report_menu.i_selected[]=2
                if haskey(ENV,"HAMMERHEAD_COMPARISON_SCREENSHOT")
                    GLMakie.save(ENV["HAMMERHEAD_COMPARISON_SCREENSHOT"],fig)
                end
            end
        end
    end
end
