#include "benchmark.hpp"
#include "european.cuh"
#include "payoffs.cuh"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main() {
  double totalTime = 0.0;
  int totalIterations = 30;

  double S0 = 100.0;
  double K = 100.0;
  double r = 0.05;
  double sigma = 0.2;
  double T = 1.0;
  int paths = 10000000;
  EuropeanOption<CallPayoff> opt(S0, K, r, sigma, T);
  for (int i = 0; i <= totalIterations; i++) {
    std::string num = std::to_string(i);
    Timer timer(num + ". European Call GPU");
    double dummy = opt.europeanOptionGPU(paths);
    if (i)
      totalTime += timer.getDuration();
  }

  Timer timer("European Call CPU");
  double dummy = opt.europeanOptionCPU(paths);
  double cpuTime = timer.getDuration();
  std::cout << std::endl
            << "Average time for European Call GPU: "
            << totalTime / totalIterations << " seconds." << std::endl;
  std::cout << "Average time for European Call CPU: " << cpuTime << " seconds."
            << std::endl;

  return 0;
}
