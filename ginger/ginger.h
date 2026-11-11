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

#ifndef GINGER_GINGER_H_
#define GINGER_GINGER_H_

// Includes Ginger's main API: datasets, the recommended configuration, building
// and searching a `GingerSearcher`, and saving and loading it. Specialized APIs
// such as custom distance functions, kNN graph construction and TurboQuant have
// their own headers.

#include "ginger/config_recommender.h"  // IWYU pragma: export
#include "ginger/dataset.h"             // IWYU pragma: export
#include "ginger/io/io.h"               // IWYU pragma: export
#include "ginger/searcher.h"            // IWYU pragma: export
#include "ginger/searcher.pb.h"      // IWYU pragma: export
#include "ginger/searcher_builder.h"    // IWYU pragma: export
#include "ginger/types.h"               // IWYU pragma: export

#endif  // GINGER_GINGER_H_
