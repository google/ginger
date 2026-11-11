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

#include "ginger/candidate_filtering/candidate_filter.h"

#include <vector>

#include "ginger/internal/gunit_support.h"
#include "absl/types/span.h"
#include "ginger/types.h"

namespace ginger {
namespace {

// A simple mock distance computer that assigns distance =
// static_cast<float>(cand.id * 5)
struct DummyDistanceComputer {
  void operator()(absl::Span<DistanceAndId> candidates) const {
    for (auto& cand : candidates) {
      cand.distance = static_cast<float>(cand.id * 5);
    }
  }
};

// A mock policy to test CandidateFilter independently of
// BeamMeanThresholdPolicy.
struct MockThresholdPolicy {
  bool should_filter = false;
  float thresh = 100.0f;
  int update_before_calls = 0;
  int extra_computations = 0;

  template <typename ComputeDistancesFunc>
  int UpdateBeforeFiltering(absl::Span<const BeamEntry> /*beam*/,
                            ComputeDistancesFunc& /*compute_distances*/) {
    ++update_before_calls;
    return extra_computations;
  }

  bool ShouldFilter() const { return should_filter; }

  float threshold() const { return thresh; }
};

TEST(BeamMeanThresholdPolicyTest, BelowBeamSize_NoThreshold) {
  BeamMeanThresholdPolicy policy(/*min_beam_size=*/3, /*width=*/3);
  DummyDistanceComputer compute;
  std::vector<BeamEntry> beam = {{5.0f, 1}, {10.0f, 2}};  // size 2 < 3

  EXPECT_EQ(policy.UpdateBeforeFiltering(absl::MakeSpan(beam), compute), 0);
  EXPECT_FALSE(policy.ShouldFilter());
  EXPECT_EQ(policy.threshold(), 0.0f);
}

TEST(BeamMeanThresholdPolicyTest, ReachesBeamSize_ComputesThreshold) {
  BeamMeanThresholdPolicy policy(/*min_beam_size=*/2, /*width=*/2);
  DummyDistanceComputer compute;
  // Node IDs 1 and 2 -> DummyDistanceComputer will assign distances 5.0
  // and 10.0
  std::vector<BeamEntry> beam = {{0.0f, 1}, {0.0f, 2}};

  int extra = policy.UpdateBeforeFiltering(absl::MakeSpan(beam), compute);
  EXPECT_EQ(extra, 2);
  EXPECT_TRUE(policy.ShouldFilter());
  // Mean of 5.0 and 10.0 is 7.5
  EXPECT_EQ(policy.threshold(), 7.5f);
}

TEST(BeamMeanThresholdPolicyTest, WithWidthParam_ConsidersTailOnly) {
  BeamMeanThresholdPolicy policy(/*min_beam_size=*/2, /*width=*/2);
  DummyDistanceComputer compute;
  // Beam size 4, width 2 -> should only look at last 2 elements (IDs 3 and 4)
  // Distances for ID 3, 4 -> 15.0, 20.0. Mean -> 17.5.
  std::vector<BeamEntry> beam = {{0.0f, 1}, {0.0f, 2}, {0.0f, 3}, {0.0f, 4}};

  int extra = policy.UpdateBeforeFiltering(absl::MakeSpan(beam), compute);
  EXPECT_EQ(extra, 2);
  EXPECT_TRUE(policy.ShouldFilter());
  EXPECT_EQ(policy.threshold(), 17.5f);
}

TEST(BeamMeanThresholdPolicyTest, CachedThreshold_WhenBeamBackUnchanged) {
  BeamMeanThresholdPolicy policy(/*min_beam_size=*/2, /*width=*/2);
  DummyDistanceComputer compute;
  std::vector<BeamEntry> beam = {{0.0f, 1}, {0.0f, 2}};

  // First call: computes for 1 and 2 -> mean 7.5
  EXPECT_EQ(policy.UpdateBeforeFiltering(absl::MakeSpan(beam), compute), 2);
  EXPECT_EQ(policy.threshold(), 7.5f);

  // Second call: beam.back() is still ID 2. Uses cached tail mean (7.5).
  // Running average threshold: (7.5 + 7.5) / 2 = 7.5. Extra computations = 0.
  EXPECT_EQ(policy.UpdateBeforeFiltering(absl::MakeSpan(beam), compute), 0);
  EXPECT_EQ(policy.threshold(), 7.5f);
}

TEST(BeamMeanThresholdPolicyTest, UpdatesThreshold_WhenBeamBackChanges) {
  BeamMeanThresholdPolicy policy(/*min_beam_size=*/2, /*width=*/2);
  DummyDistanceComputer compute;
  std::vector<BeamEntry> beam1 = {{0.0f, 1}, {0.0f, 2}};

  // First call: computes for 1 and 2 -> mean 7.5 (count=1, sum=7.5)
  EXPECT_EQ(policy.UpdateBeforeFiltering(absl::MakeSpan(beam1), compute), 2);
  EXPECT_EQ(policy.threshold(), 7.5f);

  // Second call: beam changes, back is now ID 4.
  // Computes for 3 and 4 -> distances 15.0, 20.0 -> tail mean 17.5.
  // New threshold sum = 7.5 + 17.5 = 25.0. Count = 2. Threshold = 12.5.
  std::vector<BeamEntry> beam2 = {{0.0f, 3}, {0.0f, 4}};
  EXPECT_EQ(policy.UpdateBeforeFiltering(absl::MakeSpan(beam2), compute), 2);
  EXPECT_EQ(policy.threshold(), 12.5f);
}

TEST(BeamMeanThresholdPolicyTest, TracksMeanCorrectlyAcrossUpdates) {
  BeamMeanThresholdPolicy policy(/*min_beam_size=*/2, /*width=*/2);
  auto compute = [&](absl::Span<DistanceAndId> cands) {
    for (auto& c : cands) {
      c.distance = static_cast<float>(c.id * 10);
    }
  };

  // Step 1: Beam with IDs {1, 2}. Distances {10.0, 20.0}. Tail mean = 15.0.
  // Threshold count = 1, sum = 15.0 -> threshold = 15.0.
  std::vector<BeamEntry> beam1 = {{0.0f, 1}, {0.0f, 2}};
  policy.UpdateBeforeFiltering(absl::MakeSpan(beam1), compute);
  EXPECT_FLOAT_EQ(policy.threshold(), 15.0f);

  // Step 2: Beam with IDs {3, 4}. Distances {30.0, 40.0}. Tail mean = 35.0.
  // Threshold count = 2, sum = 15.0 + 35.0 = 50.0 -> threshold = 25.0.
  std::vector<BeamEntry> beam2 = {{0.0f, 3}, {0.0f, 4}};
  policy.UpdateBeforeFiltering(absl::MakeSpan(beam2), compute);
  EXPECT_FLOAT_EQ(policy.threshold(), 25.0f);

  // Step 3: Beam with IDs {5, 6}. Distances {50.0, 60.0}. Tail mean = 55.0.
  // Threshold count = 3, sum = 50.0 + 55.0 = 105.0 -> threshold = 35.0.
  std::vector<BeamEntry> beam3 = {{0.0f, 5}, {0.0f, 6}};
  policy.UpdateBeforeFiltering(absl::MakeSpan(beam3), compute);
  EXPECT_FLOAT_EQ(policy.threshold(), 35.0f);
}

TEST(BeamMeanThresholdPolicyTest, SimilarBeamsProduceAnticipatedSmallDelta) {
  BeamMeanThresholdPolicy policy(/*min_beam_size=*/2, /*width=*/2);
  auto compute = [&](absl::Span<DistanceAndId> cands) {
    for (auto& c : cands) {
      if (c.id == 1)
        c.distance = 10.0f;
      else if (c.id == 2)
        c.distance = 20.0f;
      else if (c.id == 3)
        c.distance = 20.4f;
    }
  };

  // First beam: {1, 2} -> distances {10.0, 20.0}. Tail mean = 15.0.
  // Threshold = 15.0.
  std::vector<BeamEntry> beam1 = {{0.0f, 1}, {0.0f, 2}};
  policy.UpdateBeforeFiltering(absl::MakeSpan(beam1), compute);
  float initial_threshold = policy.threshold();
  EXPECT_FLOAT_EQ(initial_threshold, 15.0f);

  // Second beam: {1, 3} -> distances {10.0, 20.4}. Tail mean = 15.2.
  // Running average threshold = (15.0 + 15.2) / 2 = 15.1.
  // The anticipated delta in threshold is exactly +0.1f (from 15.0 to 15.1).
  std::vector<BeamEntry> beam2 = {{0.0f, 1}, {0.0f, 3}};
  policy.UpdateBeforeFiltering(absl::MakeSpan(beam2), compute);
  float new_threshold = policy.threshold();
  EXPECT_FLOAT_EQ(new_threshold, 15.1f);
  EXPECT_NEAR(new_threshold - initial_threshold, 0.1f, 1e-4f);
}

TEST(BeamMeanThresholdPolicyTest, ActualWidthZero_HandledCorrectly) {
  BeamMeanThresholdPolicy policy(/*min_beam_size=*/1, /*width=*/0);
  DummyDistanceComputer compute;
  std::vector<BeamEntry> beam = {{0.0f, 1}};

  EXPECT_EQ(policy.UpdateBeforeFiltering(absl::MakeSpan(beam), compute), 0);
  EXPECT_FALSE(policy.ShouldFilter());
  EXPECT_EQ(policy.threshold(), 0.0f);
}

TEST(BeamMeanThresholdPolicyTest, EmptyBeamWithZeroMinBeamSize_ReturnsZero) {
  BeamMeanThresholdPolicy policy(/*min_beam_size=*/0, /*width=*/2);
  DummyDistanceComputer compute;
  std::vector<BeamEntry> empty_beam;

  EXPECT_EQ(policy.UpdateBeforeFiltering(absl::MakeSpan(empty_beam), compute),
            0);
  EXPECT_FALSE(policy.ShouldFilter());
  EXPECT_EQ(policy.threshold(), 0.0f);
}

TEST(CandidateFilterTest, EmptyCandidates_ReturnsEmpty) {
  CandidateFilter filter(DummyDistanceComputer{}, MockThresholdPolicy{});
  std::vector<DistanceAndId> candidates;
  auto survivors = filter.FilterCandidates(absl::MakeSpan(candidates));
  EXPECT_TRUE(survivors.empty());
}

TEST(CandidateFilterTest, PolicyShouldFilterFalse_AllSurvive) {
  MockThresholdPolicy mock_policy;
  mock_policy.should_filter = false;
  mock_policy.extra_computations = 5;

  CandidateFilter filter(DummyDistanceComputer{}, mock_policy);
  std::vector<DistanceAndId> candidates = {{-1.0f, 1}, {-1.0f, 2}};

  auto survivors = filter.FilterCandidates(absl::MakeSpan(candidates));
  EXPECT_EQ(survivors.size(), 2);
  EXPECT_EQ(filter.policy().update_before_calls, 1);
}

TEST(CandidateFilterTest, PolicyShouldFilterTrue_FiltersAboveThreshold) {
  MockThresholdPolicy mock_policy;
  mock_policy.should_filter = true;
  mock_policy.thresh =
      12.0f;  // ID 1 (5.0), ID 2 (10.0) pass; ID 3 (15.0) filtered

  CandidateFilter filter(DummyDistanceComputer{}, mock_policy);
  std::vector<DistanceAndId> candidates = {{-1.0f, 1}, {-1.0f, 2}, {-1.0f, 3}};

  auto survivors = filter.FilterCandidates(absl::MakeSpan(candidates));
  EXPECT_EQ(survivors.size(), 2);
  EXPECT_EQ(survivors[0].id, 1);
  EXPECT_EQ(survivors[1].id, 2);
}

TEST(CandidateFilterTest, IntegrationWithBeamMeanThresholdPolicy) {
  BeamMeanThresholdPolicy policy(/*min_beam_size=*/2, /*width=*/2);
  CandidateFilter filter(DummyDistanceComputer{}, policy);

  std::vector<BeamEntry> beam = {{0.0f, 1}, {0.0f, 2}};  // Tail mean -> 7.5
  // Candidate IDs: 1 (5.0), 2 (10.0). Threshold is 7.5, so ID 2 should be
  // filtered out.
  std::vector<DistanceAndId> candidates = {{-1.0f, 1}, {-1.0f, 2}};

  auto survivors =
      filter.FilterCandidates(absl::MakeSpan(candidates), absl::MakeSpan(beam));
  EXPECT_EQ(survivors.size(), 1);
  EXPECT_EQ(survivors[0].id, 1);
}

TEST(NoFilterTest, ReturnsAllCandidates) {
  NoFilter no_filter;
  std::vector<DistanceAndId> candidates = {{-1.0f, 1}, {-1.0f, 2}};
  auto survivors = no_filter.FilterCandidates(absl::MakeSpan(candidates));
  EXPECT_EQ(survivors.size(), 2);
  EXPECT_EQ(survivors[0].id, 1);
  EXPECT_EQ(survivors[1].id, 2);
}

}  // namespace
}  // namespace ginger
