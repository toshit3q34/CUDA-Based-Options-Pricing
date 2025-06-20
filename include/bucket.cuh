#pragma once

struct BucketPayoff {
  double K1, K2;

  __host__ __device__
  BucketPayoff(double _K1, double _K2) : K1(_K1), K2(_K2) {}

  __host__ __device__
  double operator()(double avg_S) const {
    if (avg_S >= K1 && avg_S <= K2)
      return avg_S - K1;
    return 0.0;
  }
};

template <typename Payoff> class BucketOption {
private:
  double S0, K1, K2, r, sigma, T;
  int tradingDays, fixings;

public:
  BucketOption(double _S0, double _K1, double _k2, double _r, double _sigma, double _T, int _tradingDays, int _fixings)
      : S0(_S0), K1(_K1), K2(_K2), r(_r), sigma(_sigma), T(_T), tradingDays(_tradingDays), fixings(_fixings) {
        if (_K1 >= _K2) {
            swap(K1, K2);
        }
    }

  double bucketOptionCPU(int paths);
  double bucketOptionGPU(int paths);
};

template <typename Payoff>
__global__ void bucketOptionGPUKernel(double S0, double K1, double K2, double r,
                                     double sigma, double T, int tradingDays,
                                     int paths, int fixings, double *results,
                                     Payoff payoff);