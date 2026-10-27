// ModelStack Lab CPU kernel baseline. C++17; no third-party dependencies.
// Correctness and timing on the current host, not a claim of GPU optimization.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Tensor = std::vector<float>;

struct Params {
  unsigned size = 48;
  unsigned iterations = 20;
  bool check = false;
};

Params parse(int argc, char **argv) {
  Params p;
  for (int i = 1; i < argc; ++i) {
    std::string argument(argv[i]);
    if (argument == "--check") { p.check = true; continue; }
    if ((argument == "--size" || argument == "--iterations") && i + 1 < argc) {
      const std::string value(argv[++i]);
      if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument("positive integer option expected");
      const auto n = std::stoul(value);
      if (n > std::numeric_limits<unsigned>::max())
        throw std::out_of_range("number exceeds unsigned range");
      if (argument == "--size") p.size = static_cast<unsigned>(n);
      else p.iterations = static_cast<unsigned>(n);
      continue;
    }
    throw std::invalid_argument("unknown or incomplete argument: " + argument);
  }
  if (p.size < 1 || p.size > 256 || p.iterations < 2 || p.iterations > 2000)
    throw std::invalid_argument("size must be 1..256, iterations 2..2000");
  return p;
}

Tensor matmul(const Tensor &a, const Tensor &b, unsigned n) {
  Tensor c(static_cast<size_t>(n) * n, 0.0f);
  for (unsigned i = 0; i < n; ++i)
    for (unsigned k = 0; k < n; ++k)
      for (unsigned j = 0; j < n; ++j)
        c[static_cast<size_t>(i) * n + j] += a[static_cast<size_t>(i) * n + k] * b[static_cast<size_t>(k) * n + j];
  return c;
}

Tensor eager(const Tensor &a, const Tensor &b, const Tensor &bias, unsigned n) {
  Tensor c = matmul(a, b, n);
  for (unsigned i = 0; i < n; ++i)
    for (unsigned j = 0; j < n; ++j) c[static_cast<size_t>(i) * n + j] += bias[j];
  for (float &x : c) x = std::max(0.0f, x);
  return c;
}

Tensor fused(const Tensor &a, const Tensor &b, const Tensor &bias, unsigned n) {
  Tensor c = matmul(a, b, n);
  for (unsigned i = 0; i < n; ++i)
    for (unsigned j = 0; j < n; ++j) {
      float &value = c[static_cast<size_t>(i) * n + j];
      value = std::max(0.0f, value + bias[j]);
    }
  return c;
}

// Time is measured only after warmup. Hash-like sink prevents dead-code elimination.
volatile double checksum_sink = 0;
double median_ms(const Tensor &a, const Tensor &b, const Tensor &bias, unsigned n,
                 unsigned iterations, bool fused_kernel) {
  auto call = [&]() -> Tensor { return fused_kernel ? fused(a, b, bias, n) : eager(a, b, bias, n); };
  auto warmup = call();
  checksum_sink = checksum_sink + warmup.front();
  std::vector<double> duration;
  duration.reserve(iterations);
  for (unsigned i = 0; i < iterations; ++i) {
    const auto start = std::chrono::steady_clock::now();
    auto result = call();
    const auto finish = std::chrono::steady_clock::now();
    checksum_sink = checksum_sink + result.front();
    duration.push_back(std::chrono::duration<double, std::milli>(finish - start).count());
  }
  std::sort(duration.begin(), duration.end());
  const size_t mid = duration.size() / 2;
  return duration.size() % 2 ? duration[mid] : (duration[mid - 1] + duration[mid]) / 2.0;
}

int main(int argc, char **argv) {
  try {
    const Params p = parse(argc, argv);
    const size_t count = static_cast<size_t>(p.size) * p.size;
    std::mt19937 engine(42);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    Tensor a(count), b(count), bias(p.size);
    for (float &x : a) x = dist(engine);
    for (float &x : b) x = dist(engine);
    for (float &x : bias) x = dist(engine);
    auto reference = eager(a, b, bias, p.size);
    auto candidate = fused(a, b, bias, p.size);
    float max_error = 0;
    for (size_t i = 0; i < count; ++i) {
      if (!std::isfinite(candidate[i])) throw std::runtime_error("non-finite output");
      max_error = std::max(max_error, std::abs(reference[i] - candidate[i]));
    }
    if (max_error > 1.0e-5f) throw std::runtime_error("correctness mismatch");
    const double baseline = median_ms(a, b, bias, p.size, p.iterations, false);
    const double optimized = median_ms(a, b, bias, p.size, p.iterations, true);
    std::cout << std::fixed << std::setprecision(7)
              << "{\"backend\":\"cpp17-cpu\",\"matrix_size\":" << p.size
              << ",\"iterations\":" << p.iterations
              << ",\"max_abs_error\":" << max_error
              << ",\"eager_median_ms\":" << baseline
              << ",\"fused_median_ms\":" << optimized
              << ",\"note\":\"Synthetic single-host timings; no claimed speedup\"}" << std::endl;
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "tensorbench: " << e.what() << "\n";
    return 2;
  }
}
