#include "bucket.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>
#include <vector>

template <typename Payoff>
double BucketOption<Payoff>::bucketOptionCPU(int paths) {
  std::mt19937_64 rng(42);
  std::normal_distribution<double> norm(0.0, 1.0);

  double dt = T / tradingDays;
  double total_payoff = 0.0;
  Payoff payoff;

  for (int i = 0; i < paths; ++i) {
    double St = S0;
    double sum = 0.0;

    for (int j = 0; j < tradingDays; ++j) {
      double Z = norm(rng);
      St = St * exp((r - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * Z);

      if (tradingDays - j <= fixings)
        sum += St;
    }

    double avg = sum / fixings;
    total_payoff += payoff(avg);
  }

  return exp(-r * T) * (total_payoff / paths);
}

template <typename Payoff>
__global__ void bucketOptionGPUKernel(double S0, double r, double sigma,
                                      double T, int tradingDays, int fixings,
                                      int paths, double *results,
                                      Payoff payoff) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= paths)
    return;

  curandState state;
  curand_init(42ULL, idx, 0, &state);

  double dt = T / tradingDays;
  double St = S0;
  double sum = 0.0;

  for (int j = 0; j < tradingDays; ++j) {
    double Z = curand_normal_double(&state);
    St = St * exp((r - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * Z);

    if (tradingDays - j <= fixings)
      sum += St;
  }

  double avg = sum / fixings;
  results[idx] = payoff(avg);
}

template <typename Payoff>
double BucketOption<Payoff>::bucketOptionGPU(int paths) {
  double *d_results = nullptr;
  cudaMalloc(&d_results, paths * sizeof(double));

  int blockSize = 256;
  int gridSize = (paths + blockSize - 1) / blockSize;
  Payoff payoff;

  bucketOptionGPUKernel<<<gridSize, blockSize>>>(
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
