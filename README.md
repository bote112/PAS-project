# Real-Time Fluid Simulation Using Weakly Compressible SPH

This project implements a real-time fluid simulation using the **Weakly Compressible Smoothed Particle Hydrodynamics (WCSPH)** method. Developed with **Taichi Lang**, this simulation system supports interactive control and realistic fluid behavior, including rigid body interactions.

## Features

- Particle-based fluid simulation using WCSPH
- Real-time parameter control (e.g., viscosity, surface tension)
- Multiple fluid block support
- Rigid body interaction
- Simulation data export for external rendering tools like Houdini
- Educational, lightweight, and modular codebase

## Demo

![Demo](DemoSPH_2.gif)

### Prerequisites

- Python 3.7 or higher
- [Taichi Lang](https://taichi-lang.org/)
- CUDA-compatible NVIDIA GPU (recommended)
- Vulkan version < 1.4 (due to backend limitations)

## Inspiration

This project was inspired by [SPH_Taichi](https://github.com/erizmr/SPH_Taichi).
