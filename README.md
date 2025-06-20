# Monte Carlo Option Pricing with C++ & CUDA

This project implements a high-performance Monte Carlo simulation framework in C++ and CUDA for pricing financial derivatives. It supports European, Asian, American, and Basket options, and enables benchmarking between CPU and GPU implementations.

## Features

- Pricing for:
  - European Options (Call/Put)
  - Asian Options (Arithmetic Average)
  - American Options (Early Exercise - CPU only)
  - Basket Options (Multiple Assets)
- CPU and GPU (CUDA) implementations
- Template-based payoff abstraction (e.g., `CallPayoff`, `PutPayoff`)
- CLI support using `cxxopts`
- RAII-based Timer utility for benchmarking
- Reproducible results via fixed-seed RNG
- Modular directory structure with clean host/device separation

## Getting Started

### Prerequisites

- C++17 compiler
- NVIDIA GPU with CUDA Toolkit installed
- CMake or Make (based on your build system)
- `cxxopts` library (included or downloadable)
