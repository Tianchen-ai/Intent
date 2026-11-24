#pragma once
#include <cstdint>
namespace CI_UNIT_NAMESPACE {
struct RecordFields {
  bool grouped;
  int64_t vectorRegisters = 32;
  int64_t maxLmulEighths = 8;
  bool zvfbfmin = false;
  int64_t privateStackBytes = 65536;
};
} // namespace CI_UNIT_NAMESPACE
