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

#include "ginger/projection/pca.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "Eigen/Core"
#include "ginger/dataset.h"
#include "ginger/projection/config.pb.h"
#include "ginger/projection/pca_float.h"
#include "ginger/projection/pca_training.h"
#include "ginger/projection/pca_turboquant.h"

namespace ginger {

absl::StatusOr<std::unique_ptr<PcaProjection>> PcaProjection::Create(
    const PcaProjectionConfig& config, const SpanDataset& dataset) {
  ABSL_ASSIGN_OR_RETURN(Eigen::MatrixXf projection_matrix,
                        TrainPca(config, dataset));

  switch (config.matrix_representation()) {
    case MATRIX_REPRESENTATION_TURBOQUANT: {
      return TurboQuantPcaProjection::Create(std::move(projection_matrix));
    }
    case MATRIX_REPRESENTATION_FLOAT:
      return FloatPcaProjection::Create(std::move(projection_matrix));
    case MATRIX_REPRESENTATION_UNSPECIFIED:
      return absl::InvalidArgumentError(
          "Matrix representation must be specified in PcaProjectionConfig.");
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported matrix representation: ",
                       config.matrix_representation()));
  }
}

absl::StatusOr<std::vector<float>> PcaProjection::Project(
    absl::Span<const float> query) const {
  std::vector<float> result(output_dim());
  ABSL_RETURN_IF_ERROR(Project(query, absl::MakeSpan(result)));
  return result;
}

}  // namespace ginger
