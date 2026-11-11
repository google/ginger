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

#ifndef GINGER_INTERNAL_SCHEDULER_TEST_STUB_H_
#define GINGER_INTERNAL_SCHEDULER_TEST_STUB_H_

#include "ginger/internal/gunit_support.h"

namespace ginger::internal {

// Parlay manages its own worker threads in the open-source build, so the
// fixture needs no scheduler.
class ParallelSchedulerReferenceTest : public ::testing::Test {};

}  // namespace ginger::internal

#endif  // GINGER_INTERNAL_SCHEDULER_TEST_STUB_H_
