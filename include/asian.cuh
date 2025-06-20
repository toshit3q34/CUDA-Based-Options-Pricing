#pragma once
#include "payoffs.cuh"

template <typename Payoff> class AsianOption {
private:
  double S0, K, r, sigma, T, tradingDays;

public:
  AsianOption(double _S0, double _K, double _r, double _sigma, double _T, int _tradingDays)
      : S0(_S0), K(_K), r(_r), sigma(_sigma), T(_T), tradingDays(_tradingDays) {}

  double asianOptionCPU(int paths, int fixings);
  double asianOptionGPU(int paths, int fixings);
};

template <typename Payoff>
__global__ void asianOptionGPUKernel(double S0, double K, double r,
                                     double sigma, double T, int tradingDays,
                                     int paths, int fixings, double *results,
                                     Payoff payoff);
