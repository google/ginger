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

#ifndef GINGER_GRAPH_VARINT_H_
#define GINGER_GRAPH_VARINT_H_

#include <cstdint>

namespace ginger {

// Prefix-varint encoding for 32-bit unsigned integers.
// Encodes the byte length in the high bits of the first byte:
//   1 byte:  "0"    + 7 value bits
//   2 bytes: "10"   + 6 value bits + 1 byte
//   3 bytes: "110"  + 5 value bits + 2 bytes
//   4 bytes: "1110" + 4 value bits + 3 bytes
//   5 bytes: "1111" + 4 value bytes
class VarintEngine {
 public:
  static constexpr int Length32(uint32_t val) {
    if (val < kMin2Bytes) return 1;
    if (val < kMin3Bytes) return 2;
    if (val < kMin4Bytes) return 3;
    if (val < kMin5Bytes) return 4;
    return 5;
  }

  static char* Encode32(char* p, uint32_t val) {
    uint8_t* const ptr = reinterpret_cast<uint8_t*>(p);
    if (val < kMin2Bytes) {
      ptr[0] = static_cast<uint8_t>(val);
      return p + 1;
    } else if (val < kMin3Bytes) {
      val <<= 2;
      uint8_t low = static_cast<uint8_t>(val);
      ptr[0] = static_cast<uint8_t>((low >> 2) | 128);
      ptr[1] = static_cast<uint8_t>(val >> 8);
      return p + 2;
    } else if (val < kMin4Bytes) {
      val <<= 3;
      uint8_t low = static_cast<uint8_t>(val);
      ptr[0] = static_cast<uint8_t>((low >> 3) | 192);
      ptr[1] = static_cast<uint8_t>(val >> 8);
      ptr[2] = static_cast<uint8_t>(val >> 16);
      return p + 3;
    } else if (val < kMin5Bytes) {
      val <<= 4;
      uint8_t low = static_cast<uint8_t>(val);
      ptr[0] = static_cast<uint8_t>((low >> 4) | 224);
      ptr[1] = static_cast<uint8_t>(val >> 8);
      ptr[2] = static_cast<uint8_t>(val >> 16);
      ptr[3] = static_cast<uint8_t>(val >> 24);
      return p + 4;
    } else {
      ptr[0] = 0xff;
      ptr[1] = static_cast<uint8_t>(val);
      ptr[2] = static_cast<uint8_t>(val >> 8);
      ptr[3] = static_cast<uint8_t>(val >> 16);
      ptr[4] = static_cast<uint8_t>(val >> 24);
      return p + 5;
    }
  }

  static const char* Parse32(const char* p, uint32_t* val) {
    const uint8_t* const ptr = reinterpret_cast<const uint8_t*>(p);
    const uint32_t code = ptr[0];
    if (code < 128) {
      *val = code;
      return p + 1;
    } else if (code < 192) {
      const uint32_t v = ptr[1];
      *val = (code & 0x3f) | (v << 6);
      return p + 2;
    } else if (code < 224) {
      uint32_t v = ptr[2];
      v = (v << 8) | ptr[1];
      *val = (code & 0x1f) | (v << 5);
      return p + 3;
    } else if (code < 240) {
      uint32_t v = ptr[3];
      v = (v << 8) | ptr[2];
      v = (v << 8) | ptr[1];
      *val = (code & 0x0f) | (v << 4);
      return p + 4;
    } else {
      uint32_t v = ptr[4];
      v = (v << 8) | ptr[3];
      v = (v << 8) | ptr[2];
      v = (v << 8) | ptr[1];
      *val = v;
      return p + 5;
    }
  }

 private:
  static constexpr uint32_t kMin2Bytes = (1u << 7);
  static constexpr uint32_t kMin3Bytes = (1u << 14);
  static constexpr uint32_t kMin4Bytes = (1u << 21);
  static constexpr uint32_t kMin5Bytes = (1u << 28);
};

}  // namespace ginger

#endif  // GINGER_GRAPH_VARINT_H_
