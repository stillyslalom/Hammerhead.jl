# Hammerhead.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://stillyslalom.github.io/Hammerhead.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://stillyslalom.github.io/Hammerhead.jl/dev/)
[![Build Status](https://github.com/stillyslalom/Hammerhead.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/stillyslalom/Hammerhead.jl/actions/workflows/CI.yml?query=branch%3Amain)

A modern, high-performance Julia package for Particle Image Velocimetry (PIV) analysis. Hammerhead.jl provides a complete pipeline for extracting velocity vector fields from experimental image pairs, designed to deliver capabilities comparable to established PIV tools while leveraging Julia's performance and ecosystem.

## Features

Type-stable implementations with FFT-based correlation and parallel processing using ChunkSplitters.jl. Per-thread caches with pre-computed window functions and pre-allocated buffers for memory efficiency. Multi-stage processing with configurable window sizes, overlap ratios, and windowing functions. Built-in peak ratio and correlation moment metrics for vector validation. Clean APIs with property forwarding and symbol-based configuration. Physics-based test suite with realistic particle fields and edge case coverage.

## Quick Start

```julia
using Hammerhead

# Load your image pair
img1 = load("image1.tif")
img2 = load("image2.tif")

# Perform single-stage PIV analysis
result = run_piv(img1, img2, window_size=(64, 64), overlap=(0.5, 0.5))

# Access results
displacement_x = result.u
displacement_y = result.v
positions_x = result.x
positions_y = result.y
quality = result.peak_ratio
```

## Installation

```julia
using Pkg
Pkg.add("Hammerhead")
```

For detailed documentation, examples, and API reference, see the [documentation](https://stillyslalom.github.io/Hammerhead.jl/dev/).
