/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
 */
#ifndef HIAI_FRAMEWORK_UTIL_BUFFER_H
#define HIAI_FRAMEWORK_UTIL_BUFFER_H

#include <cstdint>
#include <cstdlib>
#include <functional>

#include "tensor/buffer.h"

namespace hiai {
class BaseBuffer : public IBuffer {
public:
    BaseBuffer();

    //  attention: if other has ownership, this constructor will copy data
    BaseBuffer(const BaseBuffer& other);

    BaseBuffer(uint8_t* data, size_t size, bool shouldOwn = false);
    // 未使用：全仓零调用（上游 framework 自定义释放能力预留，勿删）
    BaseBuffer(uint8_t* data, size_t size, std::function<void(uint8_t*)> freeFunc);

    ~BaseBuffer() override;

    bool Init(size_t size, uint8_t defaultVal = 0);

    bool CopyFrom(const uint8_t* data, size_t size);

    void SetData(uint8_t* data, size_t size, bool shouldOwn = false);

    void* GetData() override;

    // 未使用：全仓零调用（外部一律经 MutableData 或 IBuffer::GetData 访问），与 void* 版构成 const 重载
    const uint8_t* GetData() const;

    uint8_t* MutableData();

    size_t GetSize() const override;

    void Clear();

private:
    // forbiden copy assignment, use CopyFrom
    BaseBuffer& operator=(const BaseBuffer& other) = delete;
    BaseBuffer& operator=(BaseBuffer&& other) = delete;

private:
    bool isOwner_{false};
    uint8_t* data_{nullptr};
    size_t size_{0};
    std::function<void(uint8_t*)> freeFunc_{nullptr};
};
} // namespace hiai
#endif // GE_BUFFER_H
