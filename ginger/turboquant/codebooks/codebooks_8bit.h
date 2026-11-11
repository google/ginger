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

#ifndef GINGER_TURBOQUANT_CODEBOOKS_CODEBOOKS_8BIT_H_
#define GINGER_TURBOQUANT_CODEBOOKS_CODEBOOKS_8BIT_H_

namespace ginger::turboquant_codebooks {

// Codebook for 4+4 bit encoding (16 sub-centroids per bucket).
inline constexpr float kCentroids4Plus4BitPos[8][16] = {
    {0.0079835271f, 0.023957899f, 0.040005919f, 0.055957295f, 0.071987487f,
     0.088015921f, 0.10405879f, 0.12015736f, 0.13628691f, 0.15242375f,
     0.1686416f, 0.18485625f, 0.2010753f, 0.21734004f, 0.23371305f,
     0.25001228f},
    {0.26642567f, 0.28280586f, 0.2992613f, 0.31578213f, 0.33234626f,
     0.34892866f, 0.36562559f, 0.38235432f, 0.39920196f, 0.41614047f,
     0.43308404f, 0.45005074f, 0.46709678f, 0.48425892f, 0.50141525f,
     0.51857394f},
    {0.53552204f, 0.55246949f, 0.56959373f, 0.58679867f, 0.60401624f,
     0.62116605f, 0.63836175f, 0.65572804f, 0.67319518f, 0.69080961f,
     0.70854485f, 0.72633743f, 0.74416381f, 0.76197928f, 0.77990681f,
     0.79788303f},
    {0.81555337f, 0.83296239f, 0.85047567f, 0.86809951f, 0.88591188f,
     0.90379012f, 0.92169791f, 0.93976194f, 0.95791042f, 0.97616863f,
     0.99452227f, 1.012947f, 1.031638f, 1.0503143f, 1.0690327f, 1.087837f},
    {1.1069733f, 1.1264477f, 1.1460426f, 1.1658645f, 1.1858199f, 1.2059655f,
     1.2265025f, 1.2471868f, 1.268012f, 1.2890568f, 1.3103312f, 1.3318365f,
     1.3535031f, 1.3753054f, 1.3972695f, 1.4196237f},
    {1.4424658f, 1.4659216f, 1.4895688f, 1.5134053f, 1.5374557f, 1.5616155f,
     1.5860952f, 1.6110678f, 1.6363355f, 1.6619906f, 1.6880746f, 1.7145966f,
     1.7416232f, 1.7690892f, 1.7970501f, 1.8253548f},
    {1.8542643f, 1.8838488f, 1.9140476f, 1.9448632f, 1.9760621f, 2.0079529f,
     2.0405166f, 2.0740585f, 2.1083257f, 2.1434581f, 2.1795561f, 2.2169397f,
     2.2548223f, 2.2941701f, 2.3349731f, 2.3772812f},
    {2.4266469f, 2.484513f, 2.5459106f, 2.6109061f, 2.6796186f, 2.7521441f,
     2.8329024f, 2.9193969f, 3.0137048f, 3.1177742f, 3.2340646f, 3.3691323f,
     3.5280013f, 3.7296627f, 4.0000739f, 4.4262824f},
};

}  // namespace ginger::turboquant_codebooks

#endif  // GINGER_TURBOQUANT_CODEBOOKS_CODEBOOKS_8BIT_H_
