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

#ifndef GINGER_IO_SERIALIZATION_H_
#define GINGER_IO_SERIALIZATION_H_

#include <cstddef>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "ginger/turboquant/datapoint/datapoint.h"

namespace ginger {

using SerializedPoint = std::string;
using SerializedPointView = absl::string_view;

size_t SerializedPointSize(const TurboQuantBasePointView& point);
void SerializePoint(const TurboQuantBasePointView& point, absl::Span<char> out);
SerializedPoint SerializePoint(const TurboQuantBasePointView& point);

// provide bits_per_dim if extra precision is enabled else set them
// to 0. `has_3bit_prefix` must match the dataset the point was serialized from.
TurboQuantBasePointView ParseSerializedTurboQuantPoint(
    SerializedPointView point, int bits_per_dim, bool has_3bit_prefix);

}  // namespace ginger

#endif  // GINGER_IO_SERIALIZATION_H_
