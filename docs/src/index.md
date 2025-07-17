```@meta
CurrentModule = Hammerhead
```

# Hammerhead.jl

A modern, high-performance Julia package for Particle Image Velocimetry (PIV) analysis, designed to provide capabilities on par with established tools while leveraging Julia's performance and ecosystem.

## Overview

Hammerhead.jl provides a complete PIV analysis pipeline for experimental fluid dynamics applications. The package processes image pairs to extract velocity vector fields, supporting both simple single-stage analysis and sophisticated multi-stage processing for improved accuracy in challenging flow conditions.

## Key Features

- **High Performance**: Type-stable implementations with optimized FFT-based correlation
- **Flexible Configuration**: Multi-stage processing with configurable window sizes, overlap ratios, and windowing functions
- **Quality Assessment**: Built-in peak ratio and correlation moment metrics for vector validation
- **Modern Architecture**: Clean APIs with property forwarding and symbol-based configuration
- **Comprehensive Windowing**: Support for all DSP.jl windowing functions (Hanning, Hamming, Blackman, Kaiser, etc.)
- **Robust Processing**: Boundary handling with symmetric padding and graceful error recovery

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

- Optimized for large image processing with fast correlation algorithms
- Memory efficient processing of extensive datasets 
- Type-stable implementations throughout for optimal performance

## Installation

```julia
using Pkg
Pkg.add("Hammerhead")
```

## API Reference

```@index
```

```@autodocs
Modules = [Hammerhead]
```
