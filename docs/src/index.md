```@meta
CurrentModule = Hammerhead
```

# Hammerhead.jl

A modern, high-performance Julia package for Particle Image Velocimetry (PIV) analysis, designed to provide capabilities on par with established tools while leveraging Julia's performance and ecosystem.

## Overview

Hammerhead.jl provides a 2D PIV analysis pipeline for experimental fluid dynamics. The package processes image pairs to extract velocity vector fields, supporting both simple single-stage analysis and sophisticated multi-stage processing (in-progress) for improved accuracy in challenging flow conditions.

## Key Features

- **High Performance**: Type-stable implementations with optimized FFT-based correlation and parallel processing
- **Parallel Processing**: Thread-safe multi-core processing using ChunkSplitters.jl for optimal work distribution
- **Memory Efficient**: Per-thread caches with pre-computed window functions and pre-allocated buffers
- **Flexible Configuration**: Multi-stage processing with configurable window sizes, overlap ratios, and windowing functions
- **Quality Assessment**: Built-in peak ratio and correlation moment metrics for vector validation
- **Modern Architecture**: Clean APIs with property forwarding and symbol-based configuration
- **Comprehensive Windowing**: Support for all DSP.jl windowing functions (Hanning, Hamming, Blackman, Kaiser, etc.)
- **Robust Processing**: Boundary handling with symmetric padding and graceful error recovery
- **Detailed Timing**: Comprehensive performance instrumentation with TimerOutputs.jl integration

## Quick Start

### Basic PIV Analysis

```julia
using Hammerhead

# Load your image pair
img1 = load("image1.tif")
img2 = load("image2.tif")

# Perform single-stage PIV analysis
result = run_piv(img1, img2, window_size=(64, 64), overlap=(0.5, 0.5))

# Access results with clean API
displacement_x = result.u  # X-direction displacements
displacement_y = result.v  # Y-direction displacements
positions_x = result.x     # Grid x-coordinates
positions_y = result.y     # Grid y-coordinates
quality = result.peak_ratio # Quality metrics
```

### Multi-Stage Processing

```julia
# Create multi-stage configuration for improved accuracy
stages = PIVStages(3, 32,  # 3 stages ending at 32×32 windows
                   overlap=0.5,
                   window_function=:hanning)

# Perform multi-stage analysis
results = run_piv(img1, img2, stages)

# Access final stage results
final_result = results[end]
```

### Advanced Configuration

```julia
# Custom stage configuration with different parameters
stage1 = PIVStage((128, 128), overlap=(0.75, 0.75), window_function=:rectangular)
stage2 = PIVStage((64, 64), overlap=(0.5, 0.5), window_function=:hanning)
stage3 = PIVStage((32, 32), overlap=(0.25, 0.25), window_function=(:kaiser, 5.0))

results = run_piv(img1, img2, [stage1, stage2, stage3])
```

### Performance Timing

Hammerhead.jl automatically instruments all PIV operations with detailed timing data:

```julia
# Run PIV analysis (timing is automatic)
result = run_piv(img1, img2, window_size=(64, 64))

# Access timing information
using TimerOutputs  # Need to import for print_timer
timer = get_timer(result)
print_timer(timer)  # Prints detailed timing breakdown

# For multi-stage analysis, timing is in the first result
results = run_piv(img1, img2, stages)
timer = get_timer(results[1])
```

Timing data includes hierarchical breakdown of:
- FFT operations (forward, inverse, setup)
- Cross-correlation computation  
- Peak analysis and subpixel refinement
- Parallel window processing with thread-level detail
- Grid generation and cache allocation
- Result assembly and merging

## Core Data Structures

### PIVVector
Individual vector measurement containing position, displacement, and quality metrics.

### PIVResult
Container for complete analysis results with property forwarding for ergonomic access:
- `result.x`, `result.y` - Grid positions
- `result.u`, `result.v` - Displacement components  
- `result.peak_ratio` - Primary/secondary peak ratio
- `result.correlation_moment` - Correlation peak sharpness

### PIVStage
Configuration for individual processing stages with type-safe parameters:
- Window size and overlap ratios
- Windowing functions (rectangular, Hanning, Hamming, Blackman, Kaiser, etc.)
- Interpolation methods
- Deformation iterations

## Windowing Functions

Hammerhead.jl leverages DSP.jl for mathematically correct windowing functions:

```julia
# Simple windows
PIVStage((64, 64), window_function=:hanning)
PIVStage((64, 64), window_function=:blackman)

# Parametric windows  
PIVStage((64, 64), window_function=(:kaiser, 5.0))
PIVStage((64, 64), window_function=(:tukey, 0.3))
```

Supported functions include: `:rectangular`, `:hanning`, `:hamming`, `:blackman`, `:bartlett`, `:cosine`, `:lanczos`, `:triang`, `:kaiser`, `:tukey`, `:gaussian`.

## Performance

- **Parallel Processing**: Automatic multi-threaded execution using ChunkSplitters.jl for optimal CPU utilization
- **Memory Efficiency**: Per-thread caches eliminate repeated allocations and window function computations
- **Optimized Algorithms**: Fast FFT-based correlation with pre-allocated buffers and type-stable implementations
- **Scalable Design**: Efficient processing of large datasets with minimal memory overhead
- **Performance Monitoring**: Built-in timing instrumentation for performance analysis and optimization

Hammerhead.jl delivers excellent performance with near-linear scaling across available CPU cores, making it suitable for both interactive analysis and high-throughput batch processing.

## Installation

```julia
using Pkg
Pkg.add("https://github.com/stillyslalom/Hammerhead.jl")
```
