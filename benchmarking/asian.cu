#include "asian.cuh"
#include "benchmark.hpp"
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
  int paths = 1000000;
  int tradingDays = 252;
  int fixings = 30;

  AsianOption<CallPayoff> opt(S0, K, r, sigma, T, tradingDays, fixings);
  for (int i = 0; i <= totalIterations; i++) {
    std::string num = std::to_string(i);
    Timer timer(num + ". Asian Call GPU");
    double dummy = opt.asianOptionGPU(paths);
    if (i)
      totalTime += timer.getDuration();
  }

  Timer timer("Asian Call CPU");
  double dummy = opt.asianOptionCPU(paths);
  double cpuTime = timer.getDuration();
  std::cout << std::endl
            << "Average time for Asian Call GPU: "
            << totalTime / totalIterations << " seconds." << std::endl;
  std::cout << "Average time for Asian Call CPU: " << cpuTime << " seconds."
            << std::endl;

  return 0;
}