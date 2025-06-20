#pragma once
#include "payoffs.cuh"
#include <vector>

template <typename Payoff> class BasketOption {
private:
  std::vector<double> S0, sigma, w, L;
  double r, T, K;
  int N;

public:
  BasketOption(const std::vector<double> &S0_,
               const std::vector<double> &sigma_, const std::vector<double> &w_,
               const std::vector<double> &L_, double r_, double T_, double K_);

  double basketOptionCPU(int paths);
  double basketOptionGPU(int paths);
};

template <typename Payoff>
__global__ void basketOptionGPUKernel(const double *d_S0, const double *d_sigma,
                                      const double *d_L, const double *d_w,
                                      int N, double r, double T, double K,
                                      int paths, double *d_results,
                                      Payoff payoff);
