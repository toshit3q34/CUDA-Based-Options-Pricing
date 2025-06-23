#include "american.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <random>
#include <vector>

__host__ __device__ void quadraticRegression(double *X, double *Y, int n,
                                             double &a0, double &a1,
                                             double &a2) {
  double Sx = 0, Sx2 = 0, Sx3 = 0, Sx4 = 0;
  double Sy = 0, Sxy = 0, Sx2y = 0;

  for (int i = 0; i < n; ++i) {
    double x = X[i];
    double x2 = x * x;
    double y = Y[i];

    Sx += x;
    Sx2 += x2;
    Sx3 += x2 * x;
    Sx4 += x2 * x2;
    Sy += y;
    Sxy += x * y;
    Sx2y += x2 * y;
  }

  double D = n * (Sx2 * Sx4 - Sx3 * Sx3) - Sx * (Sx * Sx4 - Sx2 * Sx3) +
             Sx2 * (Sx * Sx3 - Sx2 * Sx2);

  if (D < 0) {
    D *= -1;
  }

  if (D < 1e-12) {
    a0 = 0;
    a1 = 0;
    a2 = 0;
    return;
  }

  double D0 = Sy * (Sx2 * Sx4 - Sx3 * Sx3) - Sx * (Sxy * Sx4 - Sx3 * Sx2y) +
              Sx2 * (Sxy * Sx3 - Sx2 * Sx2y);

  double D1 = n * (Sxy * Sx4 - Sx3 * Sx2y) - Sy * (Sx * Sx4 - Sx2 * Sx3) +
              Sx2 * (Sx * Sx2y - Sxy * Sx2);

  double D2 = n * (Sx2 * Sx2y - Sxy * Sx3) - Sx * (Sx * Sx2y - Sxy * Sx2) +
              Sy * (Sx * Sx3 - Sx2 * Sx2);

  a0 = D0 / D;
  a1 = D1 / D;
  a2 = D2 / D;
}

template <typename Payoff>
double AmericanOption<Payoff>::americanOption_LSM_CPU(int paths) {
  std::mt19937_64 rng(42);
  std::normal_distribution<double> norm(0.0, 1.0);

  double dt = T / steps;
  double disc = std::exp(-r * dt);

  std::vector<std::vector<double>> S(paths, std::vector<double>(steps + 1));
  for (int i = 0; i < paths; ++i) {
    S[i][0] = S0;
    for (int t = 1; t <= steps; ++t) {
      double Z = norm(rng);
      S[i][t] = S[i][t - 1] * std::exp((r - 0.5 * sigma * sigma) * dt +
                                       sigma * std::sqrt(dt) * Z);
    }
  }

  std::vector<double> CF(paths);
  for (int i = 0; i < paths; ++i) {
    CF[i] = std::max(K - S[i][steps], 0.0);
  }

  for (int t = steps - 1; t >= 1; --t) {
    std::vector<int> itm_idx;
    for (int i = 0; i < paths; ++i) {
      if (K > S[i][t])
        itm_idx.push_back(i);
    }

    int n = itm_idx.size();
    if (n == 0)
      continue;

    std::vector<double> X(n), Y(n);
    for (int j = 0; j < n; ++j) {
      int i = itm_idx[j];
      X[j] = S[i][t];
      Y[j] = CF[i] * disc;
    }

    double a0, a1, a2;
    quadraticRegression(X.data(), Y.data(), n, a0, a1, a2);

    for (int j = 0; j < n; ++j) {
      int i = itm_idx[j];
      double St = S[i][t];
      double continuation = a0 + a1 * St + a2 * St * St;
      double exercise = std::max(K - St, 0.0);
      CF[i] = (exercise > continuation) ? exercise : CF[i] * disc;
    }
  }

  double price = 0.0;
  for (int i = 0; i < paths; ++i) {
    price += CF[i] * disc;
  }
  return price / paths;
}
