#pragma once

#include <memory>
#include <type_traits>

namespace td {

template<typename T, typename... Args>
static inline std::shared_ptr<T> MakeShared(Args &&...args) {
  using T_nc = typename std::remove_const<T>::type;
  std::shared_ptr<T> ret = nullptr;
  try {
    ret = std::make_shared<T_nc>(std::forward<Args>(args)...);
  } catch (const std::bad_alloc &) {
    ret = nullptr;
    TDLOGE("Make shared failed");
  }
  return ret;
}
template<typename T, typename... Args>
static inline std::shared_ptr<T> MakeSharedAndThrow(Args &&...args) {
  using T_nc = typename std::remove_const<T>::type;
  return std::make_shared<T_nc>(std::forward<Args>(args)...);
}
template<typename T>
struct MakeUniq {
  using unique_object = std::unique_ptr<T>;
};

template <typename T>
struct MakeUniq<T[]> {
  using unique_array = std::unique_ptr<T[]>;
};

template <typename T, size_t B>
struct MakeUniq<T[B]> {
  struct invalid_type { };
};

template <typename T, typename... Args>
static inline typename MakeUniq<T>::unique_object MakeUnique(Args &&... args) {
  using T_nc = typename std::remove_const<T>::type;
  return std::unique_ptr<T>(new (std::nothrow) T_nc(std::forward<Args>(args)...));
}

template <typename T>
static inline typename MakeUniq<T>::unique_array MakeUnique(const size_t num) {
  return std::unique_ptr<T>(new (std::nothrow) typename std::remove_extent<T>::type[num]());
}

template <typename T, typename... Args>
static inline typename MakeUniq<T>::invalid_type MakeUnique(Args &&...) = delete;
}  // namespace td