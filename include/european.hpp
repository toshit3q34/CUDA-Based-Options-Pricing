#pragma once

// CPU-based European option pricing
double europeanOptionCPU(double S0,    // Initial stock price
                         double K,     // Strike price
                         double r,     // Risk-free rate
                         double sigma, // Volatility
                         double T,     // Time to maturity
                         int paths     // Number of Monte Carlo simulations
);

// GPU-based European option pricing (host function)
double europeanOptionGPU(double S0, double K, double r, double sigma, double T,
                         int paths);

// CUDA kernel declaration (device kernel)
__global__ void europeanOptionGPUKernel(double S0, double K, double r,
                                        double sigma, double T, int paths,
                                        double *results);