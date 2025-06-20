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