#include "american.cuh"
#include "asian.cuh"
#include "basket.cuh"
#include "benchmark.hpp"
#include "cxxopts.hpp"
#include "european.cuh"
#include "payoffs.cuh"
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

std::vector<double> parseList(const std::string &csv) {
  std::vector<double> out;
  std::stringstream ss(csv);
  std::string number;
  while (std::getline(ss, number, ',')) {
    if (!number.empty())
      out.push_back(std::stod(number));
  }
  return out;
}

int main(int argc, char *argv[]) {
  cxxopts::Options cli("option_pricer", "CLI Monte-Carlo Option Pricer");

  cli.add_options()("t,type", "Option type (european/asian/basket/american)",
                    cxxopts::value<std::string>())(
      "p,payoff", "Payoff type (call/put)",
      cxxopts::value<std::string>()->default_value("call"))(
      "m,method", "Method (cpu/gpu) — ignored for American",
      cxxopts::value<std::string>()->default_value("cpu"))(
      "paths", "Monte Carlo paths",
      cxxopts::value<int>()->default_value("1000000"))(
      "fixings", "Fixings (Asian only)",
      cxxopts::value<int>()->default_value("30"))(
      "steps", "Steps (American only)",
      cxxopts::value<int>()->default_value("100"))(
      "S0", "Spot price S0", cxxopts::value<double>()->default_value("100"))(
      "K", "Strike price K", cxxopts::value<double>()->default_value("100"))(
      "sigma", "Volatility", cxxopts::value<double>()->default_value("0.2"))(
      "T", "Maturity (years)", cxxopts::value<double>()->default_value("1.0"))(
      "r", "Risk-free rate", cxxopts::value<double>()->default_value("0.05"))

      ("S0s", "Basket S0 list (comma-separated)",
       cxxopts::value<std::string>()->default_value("100,105,110"))(
          "sigmas", "Basket sigma list",
          cxxopts::value<std::string>()->default_value("0.2,0.25,0.15"))(
          "weights", "Basket weights",
          cxxopts::value<std::string>()->default_value("0.3,0.5,0.2"))(
          "L", "Cholesky matrix L (flattened, row-major)",
          cxxopts::value<std::string>()->default_value(
              "1,0,0,0.5,0.866,0,0.3,0.4,0.854"))

          ("h,help", "Print help");

  auto args = cli.parse(argc, argv);

  if (args.count("help") || !args.count("type")) {
    std::cout << cli.help() << '\n';
    return 0;
  }

  const std::string optionType = args["type"].as<std::string>();
  const std::string payoffType = args["payoff"].as<std::string>();
  const std::string method = args["method"].as<std::string>();
  const int paths = args["paths"].as<int>();
  const int fixings = args["fixings"].as<int>();
  const int steps = args["steps"].as<int>();

  const double S0 = args["S0"].as<double>();
  const double K = args["K"].as<double>();
  const double sigma = args["sigma"].as<double>();
  const double T = args["T"].as<double>();
  const double r = args["r"].as<double>();

  if (optionType == "european") {
    if (payoffType == "call") {
      EuropeanOption<CallPayoff> opt(S0, K, r, sigma, T);
      Timer timer("European Call " + method);
      double price = (method == "gpu") ? opt.europeanOptionGPU(paths)
                                       : opt.europeanOptionCPU(paths);
      std::cout << "Price: " << price << '\n';
    } else {
      EuropeanOption<PutPayoff> opt(S0, K, r, sigma, T);
      Timer timer("European Put " + method);
      double price = (method == "gpu") ? opt.europeanOptionGPU(paths)
                                       : opt.europeanOptionCPU(paths);
      std::cout << "Price: " << price << '\n';
    }

  } else if (optionType == "asian") {
    const int tradingDays = 252;
    if (payoffType == "call") {
      AsianOption<CallPayoff> opt(S0, K, r, sigma, T, tradingDays, fixings);
      Timer timer("Asian Call " + method);
      double price = (method == "gpu") ? opt.asianOptionGPU(paths)
                                       : opt.asianOptionCPU(paths);
      std::cout << "Price: " << price << '\n';
    } else {
      AsianOption<PutPayoff> opt(S0, K, r, sigma, T, tradingDays, fixings);
      Timer timer("Asian Put " + method);
      double price = (method == "gpu") ? opt.asianOptionGPU(paths)
                                       : opt.asianOptionCPU(paths);
      std::cout << "Price: " << price << '\n';
    }

  } else if (optionType == "basket") {
    auto S0v = parseList(args["S0s"].as<std::string>());
    auto sigmas = parseList(args["sigmas"].as<std::string>());
    auto weights = parseList(args["weights"].as<std::string>());
    auto Lflat = parseList(args["L"].as<std::string>());
    int N = S0v.size();

    if (sigmas.size() != N || weights.size() != N || Lflat.size() != N * N) {
      std::cerr << "Basket input vector size mismatch!\n";
      return 1;
    }

    if (payoffType == "call") {
      BasketOption<CallPayoff> opt(S0v, sigmas, weights, Lflat, r, T, K);
      Timer timer("Basket Call " + method);
      double price = (method == "gpu") ? opt.basketOptionGPU(paths)
                                       : opt.basketOptionCPU(paths);
      std::cout << "Price: " << price << '\n';
    } else {
      BasketOption<PutPayoff> opt(S0v, sigmas, weights, Lflat, r, T, K);
      Timer timer("Basket Put " + method);
      double price = (method == "gpu") ? opt.basketOptionGPU(paths)
                                       : opt.basketOptionCPU(paths);
      std::cout << "Price: " << price << '\n';
    }

  } else if (optionType == "american") {
    if (payoffType == "call") {
      std::cout << "[Note] American Call ≡ European Call (no dividends)\n";
      EuropeanOption<CallPayoff> opt(S0, K, r, sigma, T);
      Timer timer("European Call " + method);
      double price = (method == "gpu") ? opt.europeanOptionGPU(paths)
                                       : opt.europeanOptionCPU(paths);
      std::cout << "Price: " << price << '\n';
    } else {
      if (method == "gpu") {
        std::cerr << "[Error] American Put is not supported with GPU at the "
                     "moment!\n";
        return 1;
      }
      AmericanOption<PutPayoff> opt(S0, K, r, sigma, T, steps);
      Timer timer("American Put CPU");
      double price = opt.americanOption_LSM_CPU(paths);
      std::cout << "Price: " << price << '\n';
    }

  } else {
    std::cerr << "[Error] Unknown option type!\n";
    std::cout << cli.help() << '\n';
    return 1;
  }

  return 0;
}
