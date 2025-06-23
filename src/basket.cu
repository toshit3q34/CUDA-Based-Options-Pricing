#include "basket.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>
#include <vector>

template <typename Payoff>
BasketOption<Payoff>::BasketOption(std::vector<double> &S0_,
                                   std::vector<double> &sigma_,
                                   std::vector<double> &w_,
                                   std::vector<double> &L_, double r_,
                                   double T_, double K_)
    : S0(S0_), sigma(sigma_), w(w_), L(L_), r(r_), T(T_), K(K_),
      N(static_cast<int>(S0_.size())) {
  double sum_w = 0.0;
  for (double weight : w_) {
    sum_w += weight;
  }

  for (double &weight : w_) {
    weight /= sum_w;
  }
}

template <typename Payoff>
double BasketOption<Payoff>::basketOptionCPU(int paths) {
  std::mt19937_64 rng(42);
  std::normal_distribution<double> norm(0.0, 1.0);

  double payoffSum = 0.0;
  Payoff payoff;

  for (int p = 0; p < paths; ++p) {
    std::vector<double> Z(N), Y(N, 0.0);
    for (double &z : Z)
      z = norm(rng);

    for (int i = 0; i < N; ++i)
      for (int k = 0; k <= i; ++k)
        Y[i] += L[i * N + k] * Z[k];

    double basket = 0.0;
    for (int i = 0; i < N; ++i) {
      double ST = S0[i] * std::exp((r - 0.5 * sigma[i] * sigma[i]) * T +
                                   sigma[i] * std::sqrt(T) * Y[i]);
      basket += w[i] * ST;
    }
    payoffSum += payoff(basket, K);
  }

  return std::exp(-r * T) * (payoffSum / paths);
}

template <typename Payoff>
void BasketOption<Payoff>::copyFromHostVector(double *&d_vect,
                                              std::vector<double> &vect) {
  cudaMalloc(&d_vect, vect.size() * sizeof(double));
  cudaMemcpy(d_vect, vect.data(), vect.size() * sizeof(double),
             cudaMemcpyHostToDevice);
}

template <typename Payoff>
__global__ void basketOptionGPUKernel(const double *d_S0, const double *d_sigma,
                                      const double *d_L, const double *d_w,
                                      int N, double r, double T, double K,
                                      int paths, double *d_results,
                                      Payoff payoff) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= paths)
    return;

  curandState state;
  curand_init(42ULL, idx, 0, &state);
  double Z[MAXN];
  for (int i = 0; i < N; ++i)
    Z[i] = curand_normal_double(&state);

  double Y[MAXN] = {0.0};
  for (int i = 0; i < N; ++i)
    for (int k = 0; k <= i; ++k)
      Y[i] += d_L[i * N + k] * Z[k];

  double basket = 0.0;
  for (int i = 0; i < N; ++i) {
    double ST = d_S0[i] * exp((r - 0.5 * d_sigma[i] * d_sigma[i]) * T +
                              d_sigma[i] * sqrt(T) * Y[i]);
    basket += d_w[i] * ST;
  }

  d_results[idx] = payoff(basket, K);
}

template <typename Payoff>
double BasketOption<Payoff>::basketOptionGPU(int paths) {
  double *d_results = nullptr;
  double *d_S0 = nullptr;
  double *d_sigma = nullptr;
  double *d_L = nullptr;
  double *d_w = nullptr;

  cudaMalloc(&d_results, paths * sizeof(double));
  copyFromHostVector(d_S0, S0);
  copyFromHostVector(d_sigma, sigma);
  copyFromHostVector(d_L, L);
  copyFromHostVector(d_w, w);

  int blockSize = 256;
  int gridSize = (paths + blockSize - 1) / blockSize;
  Payoff payoff;

  basketOptionGPUKernel<<<gridSize, blockSize>>>(
      d_S0, d_sigma, d_L, d_w, N, r, T, K, paths, d_results, payoff);
  cudaDeviceSynchronize();

  std::vector<double> h_results(paths);
  cudaMemcpy(h_results.data(), d_results, paths * sizeof(double),
             cudaMemcpyDeviceToHost);

  cudaFree(d_results);
  cudaFree(d_S0);
  cudaFree(d_sigma);
  cudaFree(d_L);
  cudaFree(d_w);

  double sum = 0.0;
  for (double payoff_result : h_results) {
    sum += payoff_result;
  }

  return exp(-r * T) * (sum / static_cast<double>(paths));
}

template class BasketOption<CallPayoff>;
template class BasketOption<PutPayoff>;