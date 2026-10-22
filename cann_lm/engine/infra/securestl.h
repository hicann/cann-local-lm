/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
 */

#ifndef INFRA_SECURESTL_H
#define INFRA_SECURESTL_H
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
#endif // INFRA_SECURESTL_H
