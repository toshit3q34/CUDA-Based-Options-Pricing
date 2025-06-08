#include <random>
#include <cmath>
#include "european.hpp"

double monte_carlo_european_call(
    double S0, double K, double r, double sigma, double T, int paths) {
    
    std::mt19937_64 rng(42);
    std::normal_distribution<double> norm(0.0, 1.0);

    double payoff_sum = 0.0;

    for (int i = 0; i < paths; ++i) {
        double Z = norm(rng);
        double ST = S0 * std::exp((r - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * Z);
        double payoff = std::max(ST - K, 0.0);
        payoff_sum += payoff;
    }

    return std::exp(-r * T) * (payoff_sum / paths);
}