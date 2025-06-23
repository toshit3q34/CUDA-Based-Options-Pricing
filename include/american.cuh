#pragma once
#include "payoffs.cuh"

template <typename Payoff> class AmericanOption {
private:
  double S0, K, r, sigma, T;
  int steps;

public:
  AmericanOption(double _S0, double _K, double _r, double _sigma, double _T,
                 int _steps)
      : S0(_S0), K(_K), r(_r), sigma(_sigma), T(_T), steps(_steps) {}

  // Using only for Put Options (Call == European Options)
  double americanOption_LSM_CPU(int paths);
};

__host__ __device__ void quadraticRegression(double *X, double *Y, int n,
                                             double &a0, double &a1,
                                             double &a2);