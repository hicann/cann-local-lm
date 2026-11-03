#include "ndk_create_itf.h"

namespace hiai {
std::shared_ptr<IBuiltModel> CreateBuiltModel() { return CreateBuiltModelFromNDK(); }
} // namespace hiai
