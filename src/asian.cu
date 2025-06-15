#include "asian.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>
#include <vector>

template <typename Payoff>
double AsianOption<Payoff>::asianOptionCPU(int paths, int fixings) {

  std::mt19937_64 rng(42);
  std::normal_distribution<double> norm(0.0, 1.0);

  double dT = T / tradingDays;
  double payoff_sum = 0.0;
  Payoff payoff;

  for (int i = 0; i < paths; ++i) {
    double sum_S = 0.0;
    for (int j = 0; j < 252; j++) {
      double Z = norm(rng);
      double ST = S0 * std::exp((r - 0.5 * sigma * sigma) * dT +
                                sigma * std::sqrt(dT) * Z);
      if (252 - j <= fixings) {
        sum_S += ST;
      }
      S0 = ST;
    }
    double avg_S = sum_S / fixings;
    double curr_payoff = payoff(avg_S, K);
    payoff_sum += curr_payoff;
  }

  return std::exp(-r * T) * (payoff_sum / paths);
}