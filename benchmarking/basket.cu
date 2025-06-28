#include "basket.cuh"
#include "benchmark.hpp"
#include "payoffs.cuh"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main() {
  double totalTime = 0.0;
  int totalIterations = 30;

  std::vector<double> S0 = {100, 95, 102, 98, 105, 110, 97, 103, 99, 101};
  std::vector<double> sigmas = {0.2,  0.25, 0.22, 0.18, 0.3,
                                0.28, 0.24, 0.26, 0.21, 0.23};
  std::vector<double> weights = {0.1, 0.1, 0.1, 0.1, 0.1,
                                 0.1, 0.1, 0.1, 0.1, 0.1};
  std::vector<double> Lflat = {
      1.0000, 0.5000, 0.8660, 0.5000, 0.2887, 0.8165, 0.5000, 0.2887, 0.2041,
      0.7906, 0.5000, 0.2887, 0.2041, 0.1750, 0.7746, 0.5000, 0.2887, 0.2041,
      0.1750, 0.1543, 0.7638, 0.5000, 0.2887, 0.2041, 0.1750, 0.1543, 0.1400,
      0.7559, 0.5000, 0.2887, 0.2041, 0.1750, 0.1543, 0.1400, 0.1302, 0.7493,
      0.5000, 0.2887, 0.2041, 0.1750, 0.1543, 0.1400, 0.1302, 0.1231, 0.7436,
  };
  double K = 100.0;
  double r = 0.05;
  double T = 1.0;
  int paths = 1000000;

  BasketOption<CallPayoff> opt(S0, sigmas, weights, Lflat, r, T, K);
  for (int i = 0; i <= totalIterations; i++) {
    std::string num = std::to_string(i);
    Timer timer(num + ". Basket Call GPU");
    double dummy = opt.basketOptionGPU(paths);
    if (i)
      totalTime += timer.getDuration();
  }

  Timer timer("Basket Call CPU");
  double dummy = opt.basketOptionCPU(paths);
  double cpuTime = timer.getDuration();
  std::cout << std::endl
            << "Average time for Basket Call GPU: "
            << totalTime / totalIterations << " seconds." << std::endl;
  std::cout << "Average time for Basket Call CPU: " << cpuTime << " seconds."
            << std::endl;

  return 0;
}