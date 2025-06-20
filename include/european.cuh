#pragma once
#include "payoffs.cuh"

template <typename Payoff> class EuropeanOption {
private:
  double S0, K, r, sigma, T;

public:
  EuropeanOption(double _S0, double _K, double _r, double _sigma, double _T)
      : S0(_S0), K(_K), r(_r), sigma(_sigma), T(_T) {}

  double europeanOptionCPU(int paths);
  double europeanOptionGPU(int paths);
};

template <typename Payoff>
__global__ void europeanOptionGPUKernel(double S0, double K, double r,
                                        double sigma, double T, int paths,
                                        double *results, Payoff payoff);