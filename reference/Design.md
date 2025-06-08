### General Project Scope and Goals

1. **Primary Objectives:**  
   - What are the primary research or application goals for this PIV suite? 
   Providing capabilities
   on par with the Prana library, but with a more modern codebase and better performance. I'd like to
   leverage threading and GPUs once the core functionality is implemented.
   - Are there specific experimental setups or flow conditions that the suite must support?
   My use case is shock tube experiments of turbulent mixing regions. The post-shock flow is subsonic, but has significant shear and compressibility effects. Refractive index gradients
   may cause distortion in the images.


2. **Target Users:**  
   - Who are the intended end users (e.g., researchers, industry practitioners, students)? 
   Mostly myself, but if it turns out well, I'd like to share it. Most of my colleagues are
   happy Matlab users, so the technical advantage of this project would have to be significant to
   get them to switch.
   - What level of technical expertise can be assumed for the users (which may influence GUI complexity and documentation depth)?
   I'm a competent user of PIV, but I've learned from experience that undocumented code is a
   footgun for future me. I'd like to make the codebase as clean and well-documented as possible,
   paying particular attention to considerations for parameter selection (e.g. particle diameter for phase correlation, choice of interpolation method, etc.)

3. **Timeline and Milestones:**  
   - What is your expected timeline for the initial MVP and subsequent releases? 
   This is a side project, but I'd like to have something useable within a month. Subsequent releases
   will be driven by my needs, but I'd like to have a solid library within 6 months.
   - Are there any critical deadlines or milestones that should influence the phased approach?
   My postdoc will conclude in about half a year, so I'd like to have a solid library by then.

### Technical and Algorithmic Considerations

4. **Preprocessing Requirements:**  
   - What are the most common types of images or data formats you plan to use (e.g., high-resolution experimental images, synthetic datasets)? 
   ~29 megapixel images from a dual-frame camera. My data sets are on the order of 200-500 images.
   - Are there any specific preprocessing techniques (beyond those listed) that you know will be essential for your datasets?
   The experiments are simultaneous PLIF/PIV, so I'll need to register the vector fields to a calibration grid and upsample/downsample to match the PLIF resolution.

5. **Core PIV Engine Details:**  
   - Do you have preferences between traditional cross-correlation versus phase correlation as the primary method?  
   I've had better luck with phase correlation, but I'd like to implement both to compare (and for easse of debugging and GPU portability).
   - How important is adaptive window sizing and deformation to your experimental conditions?
   I don't think I'll need adaptive window sizing beyond uniform 128-64-32ish adaptation, but deformation may be important to handle shear and rotation. 

6. **Subpixel Interpolation:**  
   - What level of subpixel accuracy do you require?
   I'd like to be able to resolve the particle diameter, so 1/4 pixel accuracy would be nice.
   - Do you have a preferred interpolation method (e.g., Gaussian, parabolic) or would you like to explore several options?
   I've had good results with 3-point Gaussian fits, but I'd like to implement a few options to compare.

7. **Outlier Detection and Uncertainty Quantification:**  
   - What specific criteria or statistical measures are most important for identifying outliers in your application? 
   The challenging imaging conditions cause many spurious vectors, so multiple outlier detection
   methods will be necessary, including UOD and peak ratio.
   - Which uncertainty quantification approaches (e.g., Monte Carlo simulation, peak ratio metrics) are you most interested in integrating?
   I don't have large synthetic datasets, so MC is out. Peak ratio and moment of correlation are most relevant.

### Performance and Scalability

8. **Parallel and GPU Computing:**  
   - How critical is real-time or near-real-time processing for your experiments? 
   The experiments take about 120 seconds to conduct using a largely-automated system. The main PIV failure mode on a shot-to-shot basis is poor seeding density, so it'd be good to at least have a rough field within 30 seconds.
   - Do you have existing hardware (such as GPUs) that you plan to use, or are there performance benchmarks you need to meet?
   Several large workstations with GPUs are available.

9. **Scalability Concerns:**  
   - What is the anticipated size of your datasets (e.g., number of images, resolution)? 
   200-500 images, 29 megapixels each.
   - Are there concerns about scaling the analysis for large datasets or high frame rates?
   Experiments in a nearby facility have frame rates of 3 kHz and can run for minutes at a time. I'd like to be able to handle this data, but it's not a priority.

### User Interface and Integration

10. **User Interface Expectations:**  
    - What level of interactivity do you expect from the GUI (e.g., basic parameter adjustments versus fully interactive visualization)? 
   I'd like to be able to adjust parameters and see the results in real time, but I don't need to be able to interact with the vector field directly
    - Do you have a preference for certain GUI frameworks (Dash.jl, Gtk.jl, or others)?
   I've used GLMakie for basic visualization, and I'd like to use it for the vector field visualization. I'm open to suggestions for the rest of the GUI.

11. **Scripting and Automation:**  
    - How do you envision users interacting with the suite: predominantly through a GUI, via scripting, or both? 
   I'd like to be able to run the suite from the command line, but I'd like to have a GUI for parameter selection and visualization.
    - Are there specific workflows or batch processing tasks that need to be automated?
   Nothing unusual - I'd like to give a list of image pair paths and an output directory and have the suite process them.

12. **Data Export and Reporting:**  
    - What output formats are most useful for you and your collaborators (CSV, HDF5, MATLAB, etc.)?
   Everybody loves HDF5.  
    - Would you like automated reporting features, and if so, what key metrics and visualizations should be included?
   Velocity magnitude, vector quality (first/second peak chosen, peak ratio test failed, etc.)

### Extensibility and Maintenance

13. **Plugin and Modularity Requirements:**  
    - How important is a plugin architecture to you? Are there specific extensions or custom analyses you already have in mind? 
   Julia is inherently modular, so I'm not particularly worried about this.
    - What level of documentation and community support do you expect to provide for extending the suite?
   Relatively minimal - community support is a low priority unless the project gains traction.

14. **Testing and Validation:**  
    - What types of validation datasets or benchmarks do you have available to test accuracy and reliability?
   I can download synthetic datasets, but my priority is applicability to my experimental data.
   I have several hundred image pairs from various experiments that I can use for validation.
    - Do you have preferences for testing frameworks or continuous integration setups within the Julia ecosystem?
   No strong preference, but I'd like to be able to run individual tests from the Julia REPL or VSCode.

15. **Long-Term Development and Support:**  
    - Are you planning for the suite to be used in a production environment or as a research tool that evolves over time? 
    As a research tool that evolves over time.
    - How do you foresee handling versioning, user feedback, and iterative improvements after the initial launch?
    I'll use git for versioning, and I'll be the primary user, so I'll handle user feedback and iterative improvements myself.


### Additional Clarification Questions

1. **Image Registration and Distortion Correction:**  
   - For handling refractive index gradients and image distortion, do you envision a dedicated calibration module (e.g., using a known calibration grid) that applies a specific transformation model?
   The index gradients are turbulence-induced, so they're not easily corrected. I'd like to be able to register the PIV vectors to the PLIF images using images of an existing dot grid. 
   - Would you like to support both manual and automated registration methods for the PLIF/PIV alignment?
   Manual registration will suffice, although an optimization-based refinement on the manual registration would be nice.

2. **Phase vs. Cross-Correlation Implementation:**  
   - When running both phase and cross-correlation methods, do you prefer a sequential comparison (one after the other) or parallel processing to directly compare the results?
   I don't plan to run both methods in most cases - just in the beginning to compare. It's not important to me to run them in parallel.
   - How would you like to handle discrepancies between the two methods during debugging or validation?
   I can handle this manually during the initial comparison.

3. **Window Sizing and Deformation Strategy:**  
   - For the planned 128-64-32 window adaptation, would you like the option to adjust these sizes interactively via the GUI or set them via configuration files?
   The same window sizes should be used for all images in a dataset, but I'd like to be able to adjust them interactively during the parameter-picking phase. I don't plan to use config files -
   the runs should be launched as Julia function calls with the parameters as arguments.
   - Regarding window deformation for shear and rotation, do you have a specific algorithm in mind (e.g., iterative deformation with affine transforms), or should this be an area for exploration?
   I'd like to implement a simple affine transform for shear and rotation, but I'm open to suggestions.

4. **Subpixel Interpolation Options:**  
   - You mentioned a 3-point Gaussian fit worked well previously. Would you like the suite to automatically choose the best interpolation method based on local data quality, or should the user manually select the method for each run?
   The user should select the method for each batch.
   - Do you require any additional metrics to assess the accuracy of the subpixel interpolation?
   Not at this time.

5. **Outlier Detection and Uncertainty Metrics:**  
   - For implementing multiple outlier detection methods (UOD and peak ratio), should the system flag and allow manual correction of outliers through the GUI, or would an automatic replacement be preferable?
   Automatic replacement is preferrable given the large datasets I work with.
   - Would you like the uncertainty visualization to be integrated into the real-time feedback or reserved for postprocessing analysis?
   Reserved for postprocessing analysis.

6. **Parallel Processing and GPU Acceleration:**  
   - Given your available GPU hardware, would you prefer to target a specific GPU framework (e.g., CUDA.jl) right from the start for core routines, or should GPU acceleration be introduced gradually after the CPU-based implementation is stable?
   GPU acceleration should be introduced gradually after the CPU-based implementation is stable.
   All relevant workstations have NVIDIA GPUs, so CUDA.jl is the natural choice.
   - Are there specific parts of the computation (e.g., FFTs, phase correlation) that you’d like to prioritize for GPU acceleration?
   Phase correlation and interpolation during window deformation are the most computationally intensive parts of the PIV workflow, so I'd like to prioritize those.

7. **User Interface and Real-Time Visualization:**  
   - For real-time parameter adjustments and result visualization, do you prefer a single-window interface that combines the GUI controls and vector field display, or a multi-window setup?
   A two-window setup would be ideal, with one window for parameter selection and one for vector field visualization.
   - Since you’re comfortable with GLMakie for visualization, would you like to standardize the visualization across all modules using Makie.jl, or are you open to integrating alternative Julia GUI frameworks for different tasks?
   I'd prefer a non-Makie GUI for parameter selection, but I'd like to use Makie for the vector field visualization. I'm open to suggestions for the parameter selection GUI.

8. **Workflow Automation and Batch Processing:**  
   - When processing a list of image pair paths from the command line, should there be an option to pre-define a set of parameters via a configuration file, or do you prefer interactive parameter selection before processing starts? 
   See above - I'd like to launch the runs as Julia function calls with the parameters as arguments.
   - Would you like the suite to generate a summary report (including key metrics and visualizations) automatically after batch processing is complete?
   Yes, a summary report is a priority.

9. **Documentation and Code Maintenance:**  
   - Do you have a preferred documentation style or tool (e.g., Documenter.jl) that you’d like to see implemented from the early stages? 
   I've been happy with Documenter.jl in the past.
   - What level of inline code documentation and example usage would you expect for each module, particularly for advanced features like uncertainty quantification and window deformation?
   I'd like to see a docstring for each function and a usage example for each module. I'd like to see a usage example for each advanced feature, but I don't need to see every possible use case.

10. **Testing and Validation Framework:**  
    - How would you like to structure the testing: a set of unit tests for each module and integration tests for the complete workflow, or focus primarily on end-to-end validation using your experimental data? 
   I'd like to see a set of unit tests for each module and integration tests for the complete workflow. I'd like to see end-to-end validation using my experimental data as well.
    - Are there specific performance benchmarks (e.g., processing time for a single image pair) that you’d like to target for the MVP and subsequent releases?
   I'd like to see a 29 megapixel image pair processed in under 30 seconds on a modern workstation using FFT correlation, and under 10 minutes using phase correlation.
