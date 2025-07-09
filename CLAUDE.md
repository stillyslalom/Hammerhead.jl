# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hammerhead.jl is a Julia package for Particle Image Velocimetry (PIV) analysis, designed to provide modern, high-performance capabilities on par with the Prana library. The project targets shock tube experiments with turbulent mixing regions involving subsonic post-shock flow with significant shear and compressibility effects.

## Architecture

The codebase is currently in early development with a minimal structure:

- **Main Module** (`src/Hammerhead.jl`): Contains the core CrossCorrelator implementation and basic correlation functions
- **Reference Implementation** (`src/prana.jl`): Reference code from the Prana library
- **Tests** (`test/runtests.jl`): Basic test setup with Gaussian particle correlation tests
- **Documentation** (`docs/`): Documenter.jl setup for future documentation

### Key Components

- **CrossCorrelator**: FFT-based cross-correlation with pre-computed FFTW plans for performance
- **correlate**: Main correlation function that performs FFT-based cross-correlation with subpixel refinement
- **subpixel_gauss3**: 3-point Gaussian subpixel refinement algorithm

## Common Development Tasks

### Running Tests
```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

### Building Documentation
```bash
julia --project=docs docs/make.jl
```

### Basic Usage
```julia
using Hammerhead

# Create a correlator for images of size (64, 64)
correlator = CrossCorrelator((64, 64))

# Correlate two image patches
displacement = correlate(correlator, img1, img2)
```

## Performance Considerations

- **Target Performance**: Process 29 megapixel image pair in <30 seconds (FFT correlation) or <10 minutes (phase correlation)
- **Memory Management**: CrossCorrelator pre-allocates FFT arrays for reuse across multiple correlations
- **Future GPU Support**: Planned CUDA.jl integration for phase correlation and interpolation

## Data Formats

- **Input**: High-resolution experimental images (~29 megapixels)
- **Output**: Currently returns displacement tuples, HDF5 support planned
- **Typical Dataset**: 200-500 image pairs per experiment

## Key Algorithms

### Current Implementation
- **Cross-correlation**: FFT-based approach with FFTW plans for optimal performance
- **Subpixel refinement**: 3-point Gaussian fit (gauss3) for 1/4 pixel accuracy

### Planned Features
- **Phase correlation**: Normalized cross-power spectrum for robust matching
- **Deformable correlation**: Iterative refinement using affine transforms
- **Quality metrics**: Peak ratio and correlation moment assessment
- **Outlier detection**: Universal Outlier Detection (UOD) implementation

## Dependencies

Core dependencies from Project.toml:
- **FFTW**: Fast Fourier Transform library for correlation
- **ImageFiltering**: Image processing utilities
- **LinearAlgebra**: Matrix operations (mul!, ldiv!, inv)
- **Statistics**: Statistical functions (in compatibility bounds)

## Development Notes

- The project is in early stages with placeholder functions for preprocessing and postprocessing
- Phase correlation is currently a placeholder and needs implementation
- Future development will focus on modular architecture with separate preprocessing, postprocessing, and visualization modules
- GPU acceleration with CUDA.jl is planned for performance-critical operations

## Testing Strategy

Current test includes synthetic Gaussian particle correlation validation. Future tests should include:
- Unit tests for correlation algorithms
- Integration tests with real experimental data
- Performance benchmarks for large image processing

## Documentation

The project uses Documenter.jl for documentation generation. Key areas needing documentation:
- API reference for all exported functions
- Usage examples for common PIV workflows
- Parameter selection guidance for experimental setups