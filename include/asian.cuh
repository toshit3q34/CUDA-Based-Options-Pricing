#pragma once
#include <cmath>

struct CallPayoff {
  __device__ __host__ double operator()(double S, double K) const {
    return std::fmax(S - K, 0.0);
  }
};

struct PutPayoff {
  __device__ __host__ double operator()(double S, double K) const {
    return std::fmax(K - S, 0.0);
  }
};

template <typename Payoff> class AsianOption {
private:
  double S0, K, r, sigma, T, tradingDays;

public:
  AsianOption(double _S0, double _K, double _r, double _sigma, double _T, int _tradingDays)
      : S0(_S0), K(_K), r(_r), sigma(_sigma), T(_T), tradingDays(_tradingDays) {}

  double asianOptionCPU(int paths, int fixings);
  double asianOptionGPU(int paths, int fixings);
};
