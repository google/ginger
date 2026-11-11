// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ginger/turboquant/codebooks/piecewise_cubic.h"

#include <cmath>
#include <limits>
#include <numbers>

#include "ginger/internal/gunit_support.h"
#include "ginger/turboquant/codebooks/codebooks.h"

namespace ginger::turboquant_codebooks {
namespace {

double Phi(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }
double Pdf(double x) {
  if (std::isinf(x)) return 0.0;
  return std::exp(-0.5 * x * x) / std::sqrt(2 * std::numbers::pi);
}

// E[(x - c)^2; a <= x < b] for x ~ N(0, 1).
double CellMse(double a, double b, double c) {
  auto antiderivative = [c](double x) {
    const double x_pdf = std::isinf(x) ? 0.0 : x * Pdf(x);
    return Phi(x) - x_pdf + 2 * c * Pdf(x) + c * c * Phi(x);
  };
  return antiderivative(b) - antiderivative(a);
}

// Quantization MSE of |x| for x ~ N(0, 1) with the 4-bit outer buckets and
// midpoint boundaries between the inner centroids `centroid(bucket, sub)`.
template <int k, typename F>
double QuantizationMse(F centroid) {
  double mse = 0;
  for (int bucket = 0; bucket < 8; ++bucket) {
    double lo = bucket == 0 ? 0.0 : kBoundaries[bucket - 1];
    const double bucket_hi = bucket == 7
                                 ? std::numeric_limits<double>::infinity()
                                 : kBoundaries[bucket];
    for (int sub = 0; sub < (1 << k); ++sub) {
      const double c = centroid(bucket, sub);
      const double hi =
          sub + 1 < (1 << k) ? (c + centroid(bucket, sub + 1)) / 2 : bucket_hi;
      mse += CellMse(lo, hi, c);
      lo = hi;
    }
  }
  return 2 * mse;
}

template <int k>
void ExpectCloseToLloyd(double max_mse_ratio) {
  const double lloyd = QuantizationMse<k>(
      [](int b, int s) { return LloydCentroidPos<k>(b, s); });
  const double fitted = QuantizationMse<k>(
      [](int b, int s) { return LookupCentroidPos<k>(b, s); });
  EXPECT_LT(fitted / lloyd, max_mse_ratio) << "k=" << k;
}

TEST(PiecewiseCubicTest, MseCloseToLloyd) {
  ExpectCloseToLloyd<6>(1.005);
  ExpectCloseToLloyd<8>(1.01);
}

TEST(PiecewiseCubicTest, LibraryCodebookIsTheFit) {
  for (int bucket = 0; bucket < 8; ++bucket) {
    for (int sub = 0; sub < 64; ++sub) {
      EXPECT_EQ(LookupCentroidPos<6>(bucket, sub),
                EvalPiecewiseCubic<6>(bucket, sub));
    }
    for (int sub = 0; sub < 256; ++sub) {
      EXPECT_EQ(LookupCentroidPos<8>(bucket, sub),
                EvalPiecewiseCubic<8>(bucket, sub));
    }
  }
}

}  // namespace
}  // namespace ginger::turboquant_codebooks
