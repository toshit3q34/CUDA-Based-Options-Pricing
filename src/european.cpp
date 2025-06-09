#include "european.hpp"
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>
#include <vector>

double europeanOptionCPU(double S0, double K, double r, double sigma, double T,
                         int paths) {

  std::mt19937_64 rng(42);
  std::normal_distribution<double> norm(0.0, 1.0);

  double payoff_sum = 0.0;

  for (int i = 0; i < paths; ++i) {
    double Z = norm(rng);
    double ST =
        S0 * std::exp((r - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * Z);
    double payoff = std::max(ST - K, 0.0);
    payoff_sum += payoff;
  }

  return std::exp(-r * T) * (payoff_sum / paths);
}

// CUDA kernel: one thread = one simulation path
__global__ void europeanOptionGPUKernel(double S0, double K, double r,
                                        double sigma, double T, int paths,
                                        double *results) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= paths)
    return;

  // Initialize random number generator for this thread
  curandState state;
  curand_init(42ULL, idx, 0, &state);

  double Z = curand_normal_double(&state); // standard normal
  double ST = S0 * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z);
  results[idx] = fmax(ST - K, 0.0); // call option payoff
}

// Host wrapper function: allocates memory, launches kernel, computes final
// result
double europeanOptionGPU(double S0, double K, double r, double sigma, double T,
                         int paths) {
  // Allocate device memory
  double *d_results = nullptr;
  cudaMalloc(&d_results, paths * sizeof(double));

  // Configure kernel launch parameters
  int blockSize = 256;
  int gridSize = (paths + blockSize - 1) / blockSize;

  // Launch kernel
  europeanOptionGPUKernel<<<gridSize, blockSize>>>(S0, K, r, sigma, T, paths,
                                                   d_results);
  cudaDeviceSynchronize();

  // Copy results back to host
  std::vector<double> h_results(paths);
  cudaMemcpy(h_results.data(), d_results, paths * sizeof(double),
             cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_results);

  // Compute mean payoff and discount it
  double sum = 0.0;
  for (double payoff : h_results) {
    sum += payoff;
  }

  return exp(-r * T) * (sum / static_cast<double>(paths));
}
