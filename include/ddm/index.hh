#pragma once
#include <cstdint>

namespace ddm {
#ifdef DDM_INDEX_64
using Index = std::int64_t;
#else
using Index = std::int32_t;
#endif
} // namespace ddm
