#include "american.cuh"
#include "benchmark.hpp"
#include "payoffs.cuh"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main() {
  double S0 = 100.0;
  double K = 100.0;
  double r = 0.05;
  double sigma = 0.2;
  double T = 1.0;
  int paths = 1000000;
  int steps = 100;

  AmericanOption<PutPayoff> opt(S0, K, r, sigma, T, steps);

  Timer timer("American Put CPU");
  double dummy = opt.americanOption_LSM_CPU(paths);

  return 0;
}
