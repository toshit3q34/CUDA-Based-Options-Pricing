#include "basket.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>
#include <vector>

template <typename Payoff>
BasketOption<Payoff>::BasketOption(const std::vector<double> &S0_,
                                   const std::vector<double> &sigma_,
                                   const std::vector<double> &w_,
                                   const std::vector<double> &L_, double r_,
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