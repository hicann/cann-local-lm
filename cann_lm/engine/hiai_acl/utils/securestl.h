/**
 * Copyright (c) 2025. All rights reserved.
 *
 * @file hiai_acl/utils/securestl.h
 * @brief nothrow-allocator smart pointer helpers.
 *
 * Ported from cann-recipes-harmony-infer-master/harmony_acl/src/utils/securestl.h
 */
#ifndef UTILS_SECURESTL_H
#define UTILS_SECURESTL_H

#include <memory>

namespace hiai {

template <typename T, typename... Ts>
std::shared_ptr<T> make_shared_nothrow(Ts&&... params)
{
    return std::shared_ptr<T>(new (std::nothrow) T(std::forward<Ts>(params)...));
}

template <typename T, typename... Ts>
std::unique_ptr<T> make_unique_nothrow(Ts&&... params)
{
    return std::unique_ptr<T>(new (std::nothrow) T(std::forward<Ts>(params)...));
}

} // namespace hiai

#endif // UTILS_SECURESTL_H
