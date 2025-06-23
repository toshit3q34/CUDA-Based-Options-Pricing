#include "european.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>
#include <vector>

template <typename Payoff>
double EuropeanOption<Payoff>::europeanOptionCPU(int paths) {

  std::mt19937_64 rng(42);
  std::normal_distribution<double> norm(0.0, 1.0);

  double payoff_sum = 0.0;
  Payoff payoff;

  for (int i = 0; i < paths; ++i) {
    double Z = norm(rng);
    double ST =
        S0 * std::exp((r - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * Z);
    double curr_payoff = payoff(ST, K);
    payoff_sum += curr_payoff;
  }

  return std::exp(-r * T) * (payoff_sum / paths);
}

template <typename Payoff>
__global__ void europeanOptionGPUKernel(double S0, double K, double r,
                                        double sigma, double T, int paths,
                                        double *results, Payoff payoff) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= paths)
    return;

  curandState state;
  curand_init(42ULL, idx, 0, &state);

  double Z = curand_normal_double(&state);
  double ST = S0 * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z);
  results[idx] = payoff(ST, K);
}

template <typename Payoff>
double EuropeanOption<Payoff>::europeanOptionGPU(int paths) {
  double *d_results = nullptr;
  cudaMalloc(&d_results, paths * sizeof(double));

  int blockSize = 256;
  int gridSize = (paths + blockSize - 1) / blockSize;
  Payoff payoff;

  europeanOptionGPUKernel<<<gridSize, blockSize>>>(S0, K, r, sigma, T, paths,
                                                   d_results, payoff);
  cudaDeviceSynchronize();

  std::vector<double> h_results(paths);
  cudaMemcpy(h_results.data(), d_results, paths * sizeof(double),
             cudaMemcpyDeviceToHost);

  cudaFree(d_results);

  double sum = 0.0;
  for (double payoff_results : h_results) {
    sum += payoff_results;
  }

  return exp(-r * T) * (sum / static_cast<double>(paths));
}

template class EuropeanOption<CallPayoff>;
template class EuropeanOption<PutPayoff>;