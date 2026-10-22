/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
 */
#ifndef FRAMEWORK_TENSOR_BUFFER_H
#define FRAMEWORK_TENSOR_BUFFER_H
#include <cstddef>
#include <memory>

namespace hiai {
class IBuffer {
public:
    IBuffer() = default;
    virtual ~IBuffer() = default;

    virtual void* GetData() = 0;
    virtual size_t GetSize() const = 0;
};

// create empty buffer with [size] bytes long, buffer will release when IBuffer deconstruct
std::shared_ptr<IBuffer> CreateLocalBuffer(size_t size);

// create buffer from exist data, data must delete by user
std::shared_ptr<IBuffer> CreateLocalBuffer(void* data, size_t size, bool shouldOwn = false);
} // namespace hiai
#endif // FRAMEWORK_BUFFER_BUFFER_H
