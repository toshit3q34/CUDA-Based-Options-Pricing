// Monte Carlo pricing for European Call option
double monte_carlo_european_call(
    double S0,    // initial stock price
    double K,     // strike price
    double r,     // risk-free rate
    double sigma, // volatility
    double T,     // time to maturity
    int paths     // number of simulated paths
);