#pragma once
#include "payoffs.cuh"

template <typename Payoff> class AsianOption {
private:
  double S0, K, r, sigma, T;
  int tradingDays, fixings;

public:
  AsianOption(double _S0, double _K, double _r, double _sigma, double _T,
              int _tradingDays, int _fixings)
      : S0(_S0), K(_K), r(_r), sigma(_sigma), T(_T), tradingDays(_tradingDays),
        fixings(_fixings) {}

  double asianOptionCPU(int paths);
  double asianOptionGPU(int paths);
};

template <typename Payoff>
__global__ void asianOptionGPUKernel(double S0, double K, double r,
                                     double sigma, double T, int tradingDays,
                                     int paths, int fixings, double *results,
                                     Payoff payoff);
