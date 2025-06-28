#include "asian.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>
#include <vector>

template <typename Payoff>
double AsianOption<Payoff>::asianOptionCPU(int paths) {

  std::mt19937_64 rng(42);
  std::normal_distribution<double> norm(0.0, 1.0);

  double dT = T / tradingDays;
  double payoff_sum = 0.0;
  Payoff payoff;

  for (int i = 0; i < paths; ++i) {
    double sum_S = 0.0;
    double copy_S0 = S0;
    for (int j = 0; j < tradingDays; j++) {
      double Z = norm(rng);
      double ST = copy_S0 * std::exp((r - 0.5 * sigma * sigma) * dT +
                                sigma * std::sqrt(dT) * Z);
      if (tradingDays - j <= fixings) {
        sum_S += ST;
      }
      copy_S0 = ST;
    }
    double avg_S = sum_S / fixings;
    double curr_payoff = payoff(avg_S, K);
    payoff_sum += curr_payoff;
  }

  return std::exp(-r * T) * (payoff_sum / paths);
}

template <typename Payoff>
__global__ void asianOptionGPUKernel(double S0, double K, double r,
                                     double sigma, double T, int tradingDays,
                                     int paths, int fixings, double *results,
                                     Payoff payoff) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= paths)
    return;

  curandState state;
  curand_init(42ULL, idx, 0, &state);

  double dt = T / tradingDays;
  double sum_S = 0.0;
  double ST = S0;

  for (int j = 0; j < tradingDays; ++j) {
    double Z = curand_normal_double(&state);
    ST = ST * exp((r - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * Z);
    if (tradingDays - j <= fixings) {
      sum_S += ST;
    }
  }

  double avg_S = sum_S / fixings;
  results[idx] = payoff(avg_S, K);
}

template <typename Payoff>
double AsianOption<Payoff>::asianOptionGPU(int paths) {
  double *d_results = nullptr;
  cudaMalloc(&d_results, paths * sizeof(double));

  int blockSize = 256;
  int gridSize = (paths + blockSize - 1) / blockSize;
  Payoff payoff;

  asianOptionGPUKernel<<<gridSize, blockSize>>>(
      S0, K, r, sigma, T, tradingDays, paths, fixings, d_results, payoff);
  cudaDeviceSynchronize();

  std::vector<double> h_results(paths);
  cudaMemcpy(h_results.data(), d_results, paths * sizeof(double),
             cudaMemcpyDeviceToHost);

  cudaFree(d_results);

  double sum = 0.0;
  for (double payoff_result : h_results) {
    sum += payoff_result;
  }

  return exp(-r * T) * (sum / static_cast<double>(paths));
}

template class AsianOption<CallPayoff>;
template class AsianOption<PutPayoff>;